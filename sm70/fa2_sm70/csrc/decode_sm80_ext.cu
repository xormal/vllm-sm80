// Отдельная единица трансляции ДЛЯ ОПЫТОВ с decode_sm80.cuh: собирается за минуту, а не за пять (split_decode.cu
// тянет всё декодное хозяйство). Боевой вход -- flash_decode_hmma_paged в split_decode.cu, ядро то же.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "decode_sm80.cuh"

__global__ void svod80_kernel(const float* __restrict__ A, const float* __restrict__ B, half* __restrict__ Out,
                              int H, int D, int ns, int ldo) {
  const int r = blockIdx.x, h = blockIdx.y;
  const long bb = ((long)r * H + h) * ns;
  float M = -CUDART_INF_F;
  for (int i = 0; i < ns; ++i) M = fmaxf(M, B[(bb + i) * 2]);
  float L = 0.f;
  for (int i = 0; i < ns; ++i) { const float m = B[(bb + i) * 2]; if (m != -CUDART_INF_F) L += __expf(m - M) * B[(bb + i) * 2 + 1]; }
  const float inv = (L > 0.f) ? 1.f / L : 0.f;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float o = 0.f;
    for (int i = 0; i < ns; ++i) { const float m = B[(bb + i) * 2]; if (m != -CUDART_INF_F) o += __expf(m - M) * A[(bb + i) * (long)D + d]; }
    Out[(long)r * ldo + h * D + d] = __float2half(o * inv);
  }
}

void decode_sm80_paged(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor block_table, int64_t block_size,
                       double scale, int64_t ns, torch::Tensor seqlens, torch::Tensor A, torch::Tensor B, torch::Tensor Out,
                       int64_t ldo, torch::Tensor sK, torch::Tensor sV, int64_t qGrp, int64_t qCausal) {
  const c10::cuda::CUDAGuard guard(Q.device());
  const int rows_total = Q.size(0), H = Q.size(1), D = Q.size(3);
  const int Hkv = K.size(K.dim() - 2), bs = (int)block_size, mb = block_table.size(1);
  int bs_log = 0; while ((1 << bs_log) < bs) ++bs_log;
  const int B_ = rows_total / (int)qGrp, GF = H / Hkv;
  const size_t sh = sizeof(fa2sm80_dec::Smem);
  static bool opt = false;
  if (!opt) { C10_CUDA_CHECK(cudaFuncSetAttribute(fa2sm80_dec::decode_sm80_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh)); opt = true; }
  TORCH_CHECK(ns % 2 == 0);
  auto st = at::cuda::getCurrentCUDAStream();
  fa2sm80_dec::decode_sm80_kernel<<<dim3(B_, Hkv, ns / 2), fa2sm80_dec::NT, sh, st>>>(
      reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()), K.data_ptr<int8_t>(), V.data_ptr<int8_t>(),
      sK.data_ptr<float>(), sV.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), H, Hkv, (float)scale, (int)ns,
      block_table.data_ptr<int>(), bs_log, mb, seqlens.data_ptr<int>(), (int)qGrp, GF, (int)qCausal, K.stride(0) / D,
      nullptr, nullptr, 0, 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  static const bool old_svod = getenv("FA2SM80_OLD_SVOD") != nullptr;
  if (old_svod)
    svod80_kernel<<<dim3(rows_total, H), 128, 0, st>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                                       reinterpret_cast<half*>(Out.data_ptr<at::Half>()), H, D, (int)ns, (int)ldo);
  else
    fa2sm80_dec::svod_sm80_kernel3<256><<<dim3(rows_total, H, (D + 31) / 32), 256, 0, st>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                                     reinterpret_cast<half*>(Out.data_ptr<at::Half>()), H, D, (int)ns, (int)ldo);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("decode_sm80_paged", &decode_sm80_paged); }
