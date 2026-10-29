// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// W8A8 ДЛЯ AMPERE (sm_80): int8-активации x int8-веса GPTQ (g=128, асимметричные) на тензорных ядрах int8
// (mma.m16n8k32, вдвое выше пик fp16), без декванта весов. Отдельный параметр запуска (FA2SM80_W8A8), раздельно
// для строк Clef и строк LLM; точность исследуется отдельно (октябрьская симуляция: группы 128 -- Clef 33/35).
//
// ВЕСА -- ТЕ ЖЕ ПЛИТКИ, ЧТО У w8a16_sm80 (второй копии весов в VRAM нет): плитка n8 x k64 = 512 Б, нить (g,t) держит
// uint4 своего столбца n = nb*8+g: .x .y -- k [8t, 8t+8) первого 32-куска, .z .w -- второго. Для mma.m16n8k32 это
// ГОТОВЫЕ B-фрагменты (b0 = .x, b1 = .y для куска 0), если логический слот k внутри 32-куска переставить согласованно
// для A и B: слот 4t+i <-> физический k 8t+i, слот 16+4t+i <-> 8t+4+i. Скалярное произведение от согласованной
// перестановки не меняется. Веса беззнаковые (q in [0,255]) -- mma .s8.u8 без перевода знака.
//
// МАТЕМАТИКА ГРУППЫ g (128 подряд k). Вес w = s(q - z), активация x ~ sx*xq (xq int8, sx = amax/127 по строке И
// группе). Вклад группы в y[m,n]: sx[m,g] * s[g,n] * (sum xq*q - z[g,n] * sum xq) -- скобка ТОЧНО в int32
// (|dot| <= 128*127*255 ~ 4.1e6), затем одно умножение на масштаб и накопление в fp32 по группам.
//
// ГЕОМЕТРИЯ (как у w8a16_sm80_gemm): блок 128 x 128, 8 варпов 2x4 (тайл варпа 64x32), стадия = одна группа k128,
// cp.async-конвейер GS стадий. A в разделяемой: строка 128 Б = 16 кусков по 8 Б, кусок p лежит на месте
// p ^ (4*(строка&3)) -- 16 нитей полуварпа (g=0..3, t=0..3) читают LDS.64 без конфликтов банков; 16-Б кусок
// глобальной памяти (куски 2q, 2q+1) ложится сплошным 16-Б куском q ^ (2*(строка&3)).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace w8a8 {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16_cg(void* s, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32(s)), "l"(g));
}
__device__ __forceinline__ void cp_async16_ca_z(void* s, const void* g, bool live) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(smem_u32(s)), "l"(g), "r"(live ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void mma_s8u8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------------------------------------------------------
// Квантование активаций по строке и группе 128: один варп на (строка, группа), 4 элемента на нить.
// xq int8 [M, K], sx fp32 [M, K/128] (amax/127; нулевая группа -> sx = 0, xq = 0), xsum int32 [M, K/128].
// ---------------------------------------------------------------------------------------------------------------
__global__ void quant_g128_kernel(const __half* __restrict__ x, int8_t* __restrict__ xq, float* __restrict__ sx,
                                  int* __restrict__ xsum, int M, int K) {
  const int G = K >> 7;
  const long w = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (w >= (long)M * G) return;
  const int m = (int)(w / G), g = (int)(w - (long)m * G);
  const long off = (long)m * K + g * 128 + lane * 4;
  const uint2 raw = *reinterpret_cast<const uint2*>(x + off);
  const __half2 h01 = *reinterpret_cast<const __half2*>(&raw.x), h23 = *reinterpret_cast<const __half2*>(&raw.y);
  const float v[4] = {__low2float(h01), __high2float(h01), __low2float(h23), __high2float(h23)};
  float a = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  const float s = a / 127.f;
  const float inv = a > 0.f ? 127.f / a : 0.f;
  int q[4], sum = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    q[i] = max(-127, min(127, __float2int_rn(v[i] * inv)));
    sum += q[i];
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  const uint32_t pk = (uint32_t)(q[0] & 0xFF) | ((uint32_t)(q[1] & 0xFF) << 8) | ((uint32_t)(q[2] & 0xFF) << 16) |
                      ((uint32_t)(q[3] & 0xFF) << 24);
  *reinterpret_cast<uint32_t*>(xq + off) = pk;
  if (lane == 0) {
    sx[(long)m * G + g] = s;
    xsum[(long)m * G + g] = sum;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// GEMM. meta [G, N]: (s fp16 | (1024+z) fp16 << 16) -- та же таблица, что у w8a16_sm80 (z = (1024+z) - 1024).
// ---------------------------------------------------------------------------------------------------------------
constexpr int BM = 128, BN = 128, BK = 128, GS = 3;
constexpr int kMagicI = 0x4B400000;          // биты fp32 1.5*2^23
constexpr float kMagicF = 12582912.f;
struct Smem {
  int8_t a[GS][BM][BK];                    // 16 КиБ на стадию, свизл 8-Б кусков
  uint4 b[GS][2][BN / 8][32];              // 2 плитки k64 x 16 n8 x 32 нити x 16 Б = 16 КиБ на стадию
  uint32_t meta[GS][BN];                   // (s | 1024+z) столбца на группу стадии
  float sx[GS][BM];
  int xsum[GS][BM];
};

template <int NW>
__global__ __launch_bounds__(256, 1) void w8a8_sm80_gemm_kernel(
    const int8_t* __restrict__ xq, const float* __restrict__ sx, const int* __restrict__ xsum,
    const uint4* __restrict__ wf, const uint32_t* __restrict__ meta, __half* __restrict__ y, int M, int N, int K,
    long ldy) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;               // 2 x 4
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int G = K >> 7, KT = K >> 6;
  auto issue = [&](int gi, int st) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {                        // A: 128 строк x 8 кусков по 16 Б = 1024 / 256 нитей
      const int c = tid + i * 256, r = c >> 3, ch = c & 7;
      const bool live = (m0 + r) < M;
      const int8_t* src = xq + (long)(live ? m0 + r : 0) * K + gi * 128 + ch * 16;
      cp_async16_ca_z(&sm.a[st][r][(ch ^ (2 * (r & 3))) * 16], src, live);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {                        // B: 2 плитки x 16 n8 x 32 = 1024 / 256 нитей
      const int c = tid + i * 256, half = c >> 9, j = (c >> 5) & 15, ln = c & 31;
      const int nb = (n0 >> 3) + j, kt = gi * 2 + half;
      cp_async16_cg(&sm.b[st][half][j][ln], wf + ((((long)(nb / NW)) * KT + kt) * NW + (nb % NW)) * 32 + ln);
    }
    if (tid < 32) {
      cp_async16_ca_z(&sm.meta[st][tid * 4], meta + (long)gi * N + n0 + tid * 4, true);
    } else if (tid < 64) {                                // sx: 128 строк x 4 Б = 32 куска по 16 Б
      const int r = (tid - 32) * 4;
      // строки за пределами M: нули (sx = 0 гасит их вклад)
      const bool live = (m0 + r) < M;
      if (live && m0 + r + 3 < M) {
        float tmp[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) tmp[e] = sx[(long)(m0 + r + e) * G + gi];
        *reinterpret_cast<float4*>(&sm.sx[st][r]) = *reinterpret_cast<const float4*>(tmp);
      } else {
#pragma unroll
        for (int e = 0; e < 4; ++e) sm.sx[st][r + e] = (m0 + r + e) < M ? sx[(long)(m0 + r + e) * G + gi] : 0.f;
      }
    } else if (tid < 96) {
      const int r = (tid - 64) * 4;
#pragma unroll
      for (int e = 0; e < 4; ++e) sm.xsum[st][r + e] = (m0 + r + e) < M ? xsum[(long)(m0 + r + e) * G + gi] : 0;
    }
  };
  float acc[4][4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;
  for (int s = 0; s < GS - 1; ++s) { if (s < G) issue(s, s); cp_commit(); }
  for (int it = 0; it < G; ++it) {
    const int st = it % GS;
    cp_wait<GS - 2>();
    __syncthreads();                                     // стадия it видна всем (и обычные записи sx/xsum)
    if (it + GS - 1 < G) issue(it + GS - 1, (it + GS - 1) % GS);
    cp_commit();
    // накопитель группы стартует с MAGIC - z*xsum: после mma в нём биты fp32 числа 1.5*2^23 + d, d = sum xq*(q-z),
    // |d| <= 128*127*256 < 2^22 -- перевод в float одним FADD (без I2F четвертьтемпа), поправка нуля -- до mma
    int ia[4][4][4];
    {
      int zcol[4][2];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e)
          zcol[j][e] = (int)__half2float(__ushort_as_half(
                           (unsigned short)(sm.meta[st][wn * 32 + j * 8 + 2 * t + e] >> 16))) - 1024;
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int xs = sm.xsum[st][wm * 64 + i * 16 + g + 8 * hh];
#pragma unroll
          for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 2; ++e) ia[i][j][2 * hh + e] = kMagicI - zcol[j][e] * xs;
        }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {                        // 4 куска k32 группы
      uint32_t bq[4][2];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint4 v = sm.b[st][c >> 1][wn * 4 + j][lane];
        bq[j][0] = (c & 1) ? v.z : v.x;
        bq[j][1] = (c & 1) ? v.w : v.y;
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r0 = wm * 64 + i * 16 + g, r1 = r0 + 8;
        const int p = 4 * c + t;                         // 8-Б кусок k [32c + 8t, +8)
        const uint2 v0 = *reinterpret_cast<const uint2*>(&sm.a[st][r0][(p ^ (4 * (r0 & 3))) * 8]);
        const uint2 v1 = *reinterpret_cast<const uint2*>(&sm.a[st][r1][(p ^ (4 * (r1 & 3))) * 8]);
        const uint32_t a[4] = {v0.x, v1.x, v0.y, v1.y};
#pragma unroll
        for (int j = 0; j < 4; ++j) mma_s8u8(ia[i][j], a, bq[j][0], bq[j][1]);
      }
    }
    // эпилог группы: скобка точно (int32 -> fp32 без потерь), затем масштаб sx*s в fp32
    float scol[4][2];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 2; ++e)
        scol[j][e] = __half2float(__ushort_as_half(
                         (unsigned short)(sm.meta[st][wn * 32 + j * 8 + 2 * t + e] & 0xFFFFu)));
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const float sr = sm.sx[st][wm * 64 + i * 16 + g + 8 * hh];
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float d = __int_as_float(ia[i][j][2 * hh + e]) - kMagicF;
            acc[i][j][2 * hh + e] = fmaf(d, sr * scol[j][e], acc[i][j][2 * hh + e]);
          }
      }
  }
  cp_wait<0>();
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = n0 + wn * 32 + j * 8 + 2 * t;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int row = m0 + wm * 64 + i * 16 + g + 8 * hh;
        if (row < M)
          *reinterpret_cast<__half2*>(y + (long)row * ldy + col) =
              __floats2half2_rn(acc[i][j][2 * hh], acc[i][j][2 * hh + 1]);
      }
    }
}

}  // namespace w8a8

std::vector<torch::Tensor> quant_g128(torch::Tensor x) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 && x.is_contiguous(), "quant_g128: x fp16 contiguous");
  const int K = (int)x.size(-1), M = (int)(x.numel() / K);
  TORCH_CHECK(K % 128 == 0, "quant_g128: K кратно 128");
  auto xq = torch::empty({M, K}, x.options().dtype(torch::kInt8));
  auto sx = torch::empty({M, K / 128}, x.options().dtype(torch::kFloat32));
  auto xs = torch::empty({M, K / 128}, x.options().dtype(torch::kInt32));
  const long warps = (long)M * (K / 128);
  if (warps > 0) {
    const int thr = 256, wpb = thr / 32;
    w8a8::quant_g128_kernel<<<(unsigned)((warps + wpb - 1) / wpb), thr, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), xq.data_ptr<int8_t>(), sx.data_ptr<float>(),
        xs.data_ptr<int>(), M, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {xq, sx, xs};
}

// out (необязательно): [M, N] fp16, столбцы подряд, шаг строк любой >= N -- срез строк большего тензора (смешанный
// шаг: строки Clef и LLM разными путями пишут в один выход без копий).
torch::Tensor w8a8_sm80_gemm(torch::Tensor xq, torch::Tensor sx, torch::Tensor xsum, torch::Tensor wf,
                             torch::Tensor meta, int64_t N, int64_t nw, c10::optional<torch::Tensor> out) {
  const c10::cuda::CUDAGuard guard(xq.device());
  TORCH_CHECK(xq.scalar_type() == torch::kInt8 && xq.is_contiguous(), "w8a8: xq int8");
  const int K = (int)xq.size(-1), M = (int)(xq.numel() / K);
  TORCH_CHECK(K % 128 == 0 && N % 128 == 0, "w8a8: K и N кратны 128");
  TORCH_CHECK(wf.numel() == (long)N * K / 4 && meta.numel() == (long)(K / 128) * N, "w8a8: wf/meta");
  TORCH_CHECK(sx.numel() == (long)M * (K / 128) && xsum.numel() == (long)M * (K / 128), "w8a8: sx/xsum");
  torch::Tensor y;
  if (out.has_value()) {
    y = *out;
    TORCH_CHECK(y.scalar_type() == torch::kFloat16 && y.dim() == 2 && y.size(0) == M && y.size(1) == N &&
                y.stride(1) == 1 && y.stride(0) >= N && y.stride(0) % 2 == 0 && y.device() == xq.device(),
                "w8a8: out [M, N] fp16, столбцы подряд");
  } else {
    y = torch::empty({M, N}, xq.options().dtype(torch::kFloat16));
  }
  if (M == 0) return y;
  dim3 grid((unsigned)(N / 128), (unsigned)((M + 127) / 128));
  const size_t smem = sizeof(w8a8::Smem);
  auto st = at::cuda::getCurrentCUDAStream();
#define W8A8_K(NW_) do { auto k = w8a8::w8a8_sm80_gemm_kernel<NW_>; static bool a = false; \
    if (!a) { C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); a = true; } \
    k<<<grid, 256, smem, st>>>(xq.data_ptr<int8_t>(), sx.data_ptr<float>(), xsum.data_ptr<int>(), \
      reinterpret_cast<const uint4*>(wf.data_ptr<int>()), reinterpret_cast<const uint32_t*>(meta.data_ptr<int>()), \
      reinterpret_cast<__half*>(y.data_ptr<at::Half>()), M, (int)N, K, (long)y.stride(0)); } while (0)
  if (nw == 2) W8A8_K(2); else if (nw == 4) W8A8_K(4); else if (nw == 8) W8A8_K(8);
  else TORCH_CHECK(false, "w8a8: nw 2/4/8");
#undef W8A8_K
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_g128", &quant_g128, "x fp16 -> (xq int8, sx fp32 по группам 128, xsum int32)");
  m.def("w8a8_sm80_gemm", &w8a8_sm80_gemm, "W8A8 sm_80: int8 x int8 GPTQ g128, mma.m16n8k32, блок 128x128",
        py::arg("xq"), py::arg("sx"), py::arg("xsum"), py::arg("wf"), py::arg("meta"), py::arg("N"), py::arg("nw"),
        py::arg("out") = py::none());
}
