// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// Свой fp16 GEMM для префилла на sm80 (CUTLASS 2.x API из вложенного fa2_src/cutlass): y[M,N] = x[M,K] * w[N,K]^T,
// накопление fp32, выход fp16. ИНВАРИАНТЕН К ПАЧКЕ: нет slice-K (плитка варпа по K = плитке блока), один mma.m16n8k16,
// k по возрастанию -- порядок накопления каждого элемента не зависит от формы плитки по M/N, от M и от места строки
// в куске. Проверено (bench/sa/cg_test.cu, 24.10): все конфигурации и куски 512/2048/1024 вперемешку -- побитово;
// cuBLAS «целиком против кусков 1024» расходится в 27 % элементов. Поэтому конфигурацию можно подбирать по времени --
// числа от выбора не меняются. Скорость на формах префилла -- наравне с cuBLAS, на кусках 1024 gate_up +8 %.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace cgs80 {

template <int TM, int TN, int TK, int WM, int WN, int ST>
struct Cfg {
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor,      // A = x [M,K]
      cutlass::half_t, cutlass::layout::ColumnMajor,   // B = w^T (K x N по столбцам) == w [N,K] по строкам
      cutlass::half_t, cutlass::layout::RowMajor,      // D = y [M,N]
      float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<TM, TN, TK>, cutlass::gemm::GemmShape<WM, WN, TK>, cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, ST>;
  static int run(const void* x, const void* w, void* y, int M, int N, int K, int ldy, cudaStream_t s) {
    typename Gemm::Arguments args({M, N, K}, {(cutlass::half_t*)x, K}, {(cutlass::half_t*)w, K},
                                  {(cutlass::half_t*)y, ldy}, {(cutlass::half_t*)y, ldy}, {1.f, 0.f});
    Gemm g;
    if (g.can_implement(args) != cutlass::Status::kSuccess) return 1;
    return g(args, nullptr, s) == cutlass::Status::kSuccess ? 0 : 2;
  }
};

// 0: 128x256x32 s3  1: 256x128x32 s3  2: 128x128x32 s4  3: 128x128x64 s4  4: 128x256x32 s4  5: 256x128x32 s4
static int dispatch(int cfg, const void* x, const void* w, void* y, int M, int N, int K, int ldy, cudaStream_t s) {
  switch (cfg) {
    case 0: return Cfg<128, 256, 32, 64, 64, 3>::run(x, w, y, M, N, K, ldy, s);
    case 1: return Cfg<256, 128, 32, 64, 64, 3>::run(x, w, y, M, N, K, ldy, s);
    case 2: return Cfg<128, 128, 32, 64, 64, 4>::run(x, w, y, M, N, K, ldy, s);
    case 3: return Cfg<128, 128, 64, 64, 64, 4>::run(x, w, y, M, N, K, ldy, s);
    case 4: return Cfg<128, 256, 32, 64, 64, 4>::run(x, w, y, M, N, K, ldy, s);
    case 5: return Cfg<256, 128, 32, 64, 64, 4>::run(x, w, y, M, N, K, ldy, s);
  }
  return 3;
}

}  // namespace cgs80

// y = x @ w^T; out (если дан) -- [M,N] fp16 с шагом строки out.stride(0) (можно писать в срез столбцов/строк).
torch::Tensor cg_mm(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> out, int64_t cfg) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.dtype() == torch::kFloat16 && w.dtype() == torch::kFloat16, "cg_mm: fp16");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "cg_mm: x [M,K], w [N,K]");
  TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 && x.stride(0) == x.size(1) && w.stride(0) == w.size(1),
              "cg_mm: x и w плотные по строкам");
  const int M = (int)x.size(0), K = (int)x.size(1), N = (int)w.size(0);
  torch::Tensor y = out.has_value() ? *out : torch::empty({M, N}, x.options());
  TORCH_CHECK(y.size(0) == M && y.size(1) == N && y.stride(1) == 1, "cg_mm: out [M,N] с единичным шагом столбца");
  if (M == 0) return y;
  const int r = cgs80::dispatch((int)cfg, x.data_ptr(), w.data_ptr(), y.data_ptr(), M, N, K, (int)y.stride(0),
                                at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(r == 0, "cg_mm: конфигурация ", cfg, " не подходит (код ", r, ") M=", M, " N=", N, " K=", K);
  return y;
}


// ---- SiLU*mul декода (25.10): как vllm::act_and_mul_kernel<silu, act_first>, но сетка (d/2048, M) вместо M блоков.
// Побитно: s = half(x / (1 + expf(-x))), out = half(float(s) * float(y)); файл собирается без --use_fast_math.
__global__ void silu_mul_kernel(const __half* __restrict__ gu, __half* __restrict__ out, int d) {
  const int row = blockIdx.y;
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (i >= d) return;
  const __half* x = gu + (long)row * 2 * d;
  const __half* y = x + d;
  __half* o = out + (long)row * d;
  if (i + 8 <= d) {
    const uint4 xv = *reinterpret_cast<const uint4*>(x + i), yv = *reinterpret_cast<const uint4*>(y + i);
    const __half* xh = reinterpret_cast<const __half*>(&xv);
    const __half* yh = reinterpret_cast<const __half*>(&yv);
    uint4 ov;
    __half* oh = reinterpret_cast<__half*>(&ov);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float xf = __half2float(xh[j]);
      const __half s = __float2half(xf / (1.0f + expf(-xf)));
      oh[j] = __float2half(__half2float(s) * __half2float(yh[j]));
    }
    *reinterpret_cast<uint4*>(o + i) = ov;
  } else {
    for (int k = i; k < d; ++k) {
      const float xf = __half2float(x[k]);
      const __half s = __float2half(xf / (1.0f + expf(-xf)));
      o[k] = __float2half(__half2float(s) * __half2float(y[k]));
    }
  }
}

void silu_mul_f16(torch::Tensor gu, torch::Tensor out) {
  TORCH_CHECK(gu.is_cuda() && gu.dtype() == torch::kFloat16 && gu.is_contiguous() && gu.dim() == 2);
  const int M = (int)gu.size(0), d = (int)(gu.size(1) / 2);
  TORCH_CHECK(out.is_contiguous() && out.size(0) == M && out.size(1) == d && d % 8 == 0);
  if (M == 0) return;
  const dim3 grid((d + 2047) / 2048, M);
  silu_mul_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(gu.data_ptr<at::Half>()), reinterpret_cast<__half*>(out.data_ptr<at::Half>()), d);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("silu_mul_f16", &silu_mul_f16, "SiLU*mul декода широкой сеткой (побитно как vLLM)");
  m.def("cg_mm", &cg_mm, "fp16 GEMM y = x w^T, инвариантный к пачке (CUTLASS sm80, без slice-K)",
        py::arg("x"), py::arg("w"), py::arg("out") = py::none(), py::arg("cfg") = 0);
  m.attr("NCFG") = 6;
}
