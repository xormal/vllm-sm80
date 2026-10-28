// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// W8A16 (GPTQ int8, g=128, асимметричный) ДЛЯ AMPERE (sm_80): ядро декода и малых M.
//
// ЧТО ПЕРЕНЕСЕНО С VOLTA И ЧТО ЗАМЕНЕНО РОДНЫМ МЕХАНИЗМОМ (замеры вольтовского пути -- w8a16_gemv.cu,
// tm8_gemm.cu, записка 19):
//   * байт веса -> fp16 через магическую экспоненту: 0x6400|q == 1024+q, ОДНА prmt на два веса (как на
//     Volta). Нуль вычитается HSUB2 ТОЧНО: (1024+q)-(1024+z) = q-z, целое |q-z| <= 255 в fp16 без ошибки;
//     масштаб группы вносится в fp32 ПОСЛЕ умножения (на группу 8 шагов k16) -- точнее и вольтовской
//     свёртки "1024 в ранг-1" (там вычитание двух величин ~1e5 в fp32), и дескванта в fp16 у turbomind
//     ({s,-z*s}: ошибка округления -z*s ~0.1 шага кванта);
//   * подача веса: на Volta регистровый двойной буфер (cp.async нет) и 168 регистров на краю обрыва
//     занятости; здесь cp.async.cg 16 Б на нить в ЧАСТНЫЙ буфер варпа, конвейер STAGES плиток. Нить
//     читает обратно ТОЛЬКО то, что положила сама, поэтому барьеров нет вовсе: cp.async.wait_group
//     отвечает за завершение собственных копий нити. Глубина конвейера больше не стоит регистров;
//   * перестановка k внутри MMA (см. sm80_warp_mma.h): нить владеет k = [8t, 8t+8) в каждом 32-куске,
//     значит её B-фрагмент на шаг k16 -- это РОВНО одно 32-битное слово GPTQ (4 подряд идущих k одного
//     столбца), а A (x) берётся одной LDS.128 на строку на два шага;
//   * вес перепакован ("фрагментный порядок", w8a16_sm80_pack): плитка n8 x k64 = 512 Б, у нити 16 Б
//     подряд -- одна полностью коалесцированная cp.async на нить на плитку;
//   * split-K ДЕТЕРМИНИРОВАН (как свод по слотам у w8a16_gemv, но без отдельного пуска свода): блоки
//     пишут частичные суммы в свой слот, ПОСЛЕДНИЙ пришедший к плитке N блок складывает слоты в
//     фиксированном порядке и сбрасывает счётчик -- ядро самоочищается, захват в CUDA-граф законен.
//
// ГЕОМЕТРИЯ. Блок = WARPS варпов на ОДНИХ столбцах (NW n8-блоков = 8*NW столбцов); K блока делится
// между варпами, свод варпов -- через разделяемую. Сетка (N/(8*NW), splits). M <= 16*MT (MT плиток m16).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <unordered_map>

namespace w8s80 {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16_cg(void* s, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_u32(s)), "l"(g));
}
// src_size = 0 -> 16 нулевых байт (строки x за пределами M): ни ветки, ни чтения памяти.
__device__ __forceinline__ void cp_async16_ca_z(void* s, const void* g, bool live) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(smem_u32(s)), "l"(g), "r"(live ? 16 : 0));
}
__device__ __forceinline__ void cp_async4_ca(void* s, const void* g) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem_u32(s)), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Четыре байта q (k, k+1, k+2, k+3 одного столбца) -> две пары fp16 (q-z), точно.
__device__ __forceinline__ void deq4(uint32_t w, uint32_t zz2, uint32_t& lo, uint32_t& hi) {
  uint32_t a = __byte_perm(w, 0x64646464u, 0x5140u);   // [q0,0x64,q1,0x64] = (1024+q0, 1024+q1)
  uint32_t b = __byte_perm(w, 0x64646464u, 0x5342u);
  __half2 ha = __hsub2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<const __half2*>(&zz2));
  __half2 hb = __hsub2(*reinterpret_cast<__half2*>(&b), *reinterpret_cast<const __half2*>(&zz2));
  lo = *reinterpret_cast<uint32_t*>(&ha);
  hi = *reinterpret_cast<uint32_t*>(&hb);
}

constexpr int LDX = 96;   // шаг строки плитки x в half: 192 Б = 48 слов, 48 mod 32 = 16 -> LDS.128 без конфликтов

// XR -- строк x, реально кладущихся в разделяемую: 4 (декод с MTP k=3), 8, 16, 32, 64. Строки
// плитки m16 за пределами XR -- константный ноль во фрагменте A, без чтения разделяемой.
template <int NW, int XR, int STAGES>
struct WarpSmem {
  uint4 w[STAGES][NW][32];                 // вес: плитка n8 x k64 на n8-блок, 16 Б на нить
  __half x[STAGES][XR][LDX];               // x: XR строк x 64 k (+дополнение)
  uint32_t meta[STAGES][NW][8];            // (s | (1024+z) << 16) на столбец, группа плитки
};

// meta[g][n]: младшее полуслово -- масштаб s (fp16), старшее -- 1024+z (fp16), z уже с условностью v1/v2.
__device__ __forceinline__ void atomic_max_f(float* a, float v) {
  int* ai = reinterpret_cast<int*>(a);
  int old = *ai;
  while (v > __int_as_float(old)) {
    const int prev = atomicCAS(ai, old, __float_as_int(v));
    if (prev == old) break;
    old = prev;
  }
}

// Отбор B лучших n-блоков строки по грубым максимумам: РАДИКС-ОТБОР порога (4 прохода по 8 бит,
// гистограмма в разделяемой), затем сбор всех ключей выше порога и добор равных порогу. Ключ --
// упорядоченные биты float (больше значение -> больше ключ). Набор детерминирован (кроме ничьей на
// самом пороге, если равных больше свободных мест -- для максимумов блоков это пренебрежимо).
__global__ __launch_bounds__(1024) void topblk_kernel(const float* __restrict__ bmax, int NB, int B,
                                                      int* __restrict__ out) {
  __shared__ unsigned hist[256];
  __shared__ unsigned s_pref, s_mask, s_k, s_cnt;
  const int m = blockIdx.x, tid = threadIdx.x;
  constexpr int PER = 4;                                   // NB <= 4096
  unsigned key[PER];
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int i = tid + 1024 * j;
    unsigned u = 0u;
    if (i < NB) {
      u = __float_as_uint(bmax[(long)m * NB + i]);
      u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    }
    key[j] = u;                                            // 0 -- «нет элемента», меньше любого настоящего
  }
  if (tid == 0) { s_pref = 0u; s_mask = 0u; s_k = (unsigned)B; s_cnt = 0u; }
  for (int pass = 0; pass < 4; ++pass) {
    const int sh = 24 - 8 * pass;
    if (tid < 256) hist[tid] = 0u;
    __syncthreads();
    const unsigned pref = s_pref, mask = s_mask;
#pragma unroll
    for (int j = 0; j < PER; ++j)
      if ((key[j] & mask) == pref && tid + 1024 * j < NB) atomicAdd(&hist[(key[j] >> sh) & 255u], 1u);
    __syncthreads();
    if (tid < 32) {                                        // от старшего разряда: где набирается k
      unsigned loc[8], sum = 0;
#pragma unroll
      for (int b = 0; b < 8; ++b) { loc[b] = hist[255 - (tid * 8 + b)]; sum += loc[b]; }
      unsigned incl = sum;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) { const unsigned v = __shfl_up_sync(0xffffffffu, incl, o); if (tid >= o) incl += v; }
      const unsigned excl = incl - sum, k = s_k;
      if (excl < k && incl >= k) {                          // порог в этой восьмёрке корзин
        unsigned acc = excl;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
          if (acc + loc[b] >= k) {
            const unsigned digit = 255u - (unsigned)(tid * 8 + b);
            s_pref = pref | (digit << sh); s_mask = mask | (255u << sh); s_k = k - acc;
            break;
          }
          acc += loc[b];
        }
      }
    }
    __syncthreads();
  }
  const unsigned T = s_pref;                                // ключ B-го по величине
  __syncthreads();
#pragma unroll
  for (int j = 0; j < PER; ++j)
    if (key[j] > T && tid + 1024 * j < NB) out[m * B + atomicAdd(&s_cnt, 1u)] = tid + 1024 * j;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < PER; ++j)
    if (key[j] == T && tid + 1024 * j < NB) {
      const unsigned pos = atomicAdd(&s_cnt, 1u);
      if (pos < (unsigned)B) out[m * B + pos] = tid + 1024 * j;
    }
}

// Малая плотная fp16-проекция (in_proj_ba у GDN: K=5120, N=48 на ранг) при M<=8: варп на столбец, fp32-накопление.
// Вместо пары cuBLAS (GEMM 64x64 splitK + reduce, 8.7 мкс и 2 узла графа) -- один узел ~3 мкс.
// MM -- строк x в шаблоне (1/2/4/8, реальные M <= MM); MAXIT -- итераций по K на нить (K <= MAXIT*2048).
// Цикл развёрнут целиком: все чтения W и x выдаются сразу, задержка памяти платится один раз.
template <int MM, int MAXIT>
__global__ __launch_bounds__(256) void lin_small_kernel(const __half* __restrict__ x, const __half* __restrict__ w,
                                                        __half* __restrict__ y, int M, int N, int K, long ldx) {
  // блок на столбец n, 8 варпов делят K (нить -- 16 Б за итерацию), свод варпов через разделяемую
  const int n = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const __half* wr = w + (long)n * K;
  uint4 wv[MAXIT];
  uint4 xv[MAXIT][MM];
#pragma unroll
  for (int it = 0; it < MAXIT; ++it) {
    const int k = tid * 8 + it * 2048;
    if (k < K) {
      wv[it] = __ldg(reinterpret_cast<const uint4*>(wr + k));
#pragma unroll
      for (int m = 0; m < MM; ++m)
        if (m < M) xv[it][m] = __ldg(reinterpret_cast<const uint4*>(x + (long)m * ldx + k));
    }
  }
  float acc[MM];
#pragma unroll
  for (int m = 0; m < MM; ++m) acc[m] = 0.f;
#pragma unroll
  for (int it = 0; it < MAXIT; ++it) {
    const int k = tid * 8 + it * 2048;
    if (k < K) {
      const __half2* w2 = reinterpret_cast<const __half2*>(&wv[it]);
#pragma unroll
      for (int m = 0; m < MM; ++m) {
        if (m < M) {
          const __half2* x2 = reinterpret_cast<const __half2*>(&xv[it][m]);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float2 a = __half22float2(x2[e]), b = __half22float2(w2[e]);
            acc[m] = fmaf(a.x, b.x, fmaf(a.y, b.y, acc[m]));
          }
        }
      }
    }
  }
  __shared__ float part[8][MM];
#pragma unroll
  for (int m = 0; m < MM; ++m) {
    float v = acc[m];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) part[warp][m] = v;
  }
  __syncthreads();
  if (tid < M) {
    float v = 0.f;
#pragma unroll
    for (int wi = 0; wi < 8; ++wi) v += part[wi][tid];
    y[(long)tid * N + n] = __float2half(v);
  }
}

// Финал: по точным значениям кандидатов (выход ядра W8 в режиме blk) -- пара (значение, номер словаря).
__global__ __launch_bounds__(256) void candmax_kernel(const __half* __restrict__ ys, const int* __restrict__ blk,
                                                      int B, int BN, int nvalid, int vstart, int width,
                                                      float* __restrict__ pair) {
  const int m = blockIdx.x, tid = threadIdx.x;
  float bv = -INFINITY; int bc = 0x7FFFFFFF;
  for (int j = tid; j < B * BN; j += 256) {
    const int col = blk[m * B + j / BN] * BN + j % BN;
    if (col >= nvalid) continue;
    const float v = __half2float(ys[(long)m * width + (long)m * B * BN + j]);
    if (v > bv || (v == bv && col < bc)) { bv = v; bc = col; }
  }
  __shared__ float sv[256]; __shared__ int sc[256];
  sv[tid] = bv; sc[tid] = bc;
  __syncthreads();
  for (int o = 128; o > 0; o >>= 1) {
    if (tid < o) {
      const float v2 = sv[tid + o]; const int c2 = sc[tid + o];
      if (v2 > sv[tid] || (v2 == sv[tid] && c2 < sc[tid])) { sv[tid] = v2; sc[tid] = c2; }
    }
    __syncthreads();
  }
  if (tid == 0) { pair[m * 2] = sv[0]; pair[m * 2 + 1] = (float)(sc[0] + vstart); }
}

template <int NW, int XR, int STAGES, int WARPS>
__global__ __launch_bounds__(WARPS * 32) void w8a16_sm80_kernel(
    const __half* __restrict__ x, const uint4* __restrict__ wf, const uint32_t* __restrict__ meta,
    __half* __restrict__ y, float* __restrict__ ws, int* __restrict__ cnt,
    int M, int N, int K, int tiles_per_split, int ktu, const int* __restrict__ blk,
    float* __restrict__ bmax = nullptr, int nvalid = 0) {
  // bmax != nullptr: выход не пишется, вместо него bmax[row*gridDim.x + blockIdx.x] = максимум строки по
  // столбцам блока (< nvalid -- без дополнения словаря). Грубая ступень аргмакса: 3880 чисел вместо 124160.
  // ktu -- сколько плиток k64 считать (первые ktu: грубая ступень аргмакса словаря по срезу k);
  // blk -- список n-блоков (8*NW столбцов) для точной досчётки кандидатов; выход тогда плотный
  // [M, gridDim.x*8*NW]. Без них (ktu=KT, blk=nullptr) -- обычное умножение.
  extern __shared__ __align__(16) unsigned char smem_raw[];
  constexpr int MT = (XR + 15) / 16;                       // плиток m16
  using WS = WarpSmem<NW, XR, STAGES>;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  WS& sm = reinterpret_cast<WS*>(smem_raw)[warp];
  const int g = lane >> 2, t = lane & 3;
  const int KT = K >> 6;                                   // плиток k64 всего (адресация веса)
  const int bx = blk ? blk[blockIdx.x] : (int)blockIdx.x;  // n-блок веса
  const int nb0 = bx * NW;                                 // первый n8-блок
  const int NO = blk ? (int)gridDim.x * NW * 8 : N;        // ширина выхода
  const int o0 = blk ? (int)blockIdx.x * NW * 8 : nb0 * 8; // столбец выхода блока
  // K блока [kt0, kt1) делится между варпами плитками подряд
  const int kt0 = blockIdx.y * tiles_per_split, kt1 = min(ktu, kt0 + tiles_per_split);
  const int per_w = (kt1 - kt0 + WARPS - 1) / WARPS;
  const int wk0 = min(kt1, kt0 + warp * per_w), wk1 = min(kt1, wk0 + per_w);
  const int ntile = wk1 - wk0;
  const int rows = MT * 16;                                // строк аккумулятора

  // ---- выдача плитки kt в стадию st: вес (NW cp.async), x (rows*8 кусков по 16 Б), meta (по группе)
  auto issue = [&](int kt, int st) {
#pragma unroll
    for (int j = 0; j < NW; ++j)
      cp_async16_cg(&sm.w[st][j][lane], wf + (((long)bx * KT + kt) * NW + j) * 32 + lane);
#pragma unroll
    for (int c = lane; c < XR * 8; c += 32) {
      const int r = c >> 3, q = c & 7;
      const bool live = r < M;
      cp_async16_ca_z(&sm.x[st][r][q * 8], x + (long)(live ? r : 0) * K + kt * 64 + q * 8, live);
    }
    const int gi = (kt * 64) >> 7;                         // группа 128
#pragma unroll
    for (int c = lane; c < NW * 8; c += 32)
      cp_async4_ca(&sm.meta[st][c >> 3][c & 7], meta + (long)gi * N + (nb0 + (c >> 3)) * 8 + (c & 7));
  };

  float acc[MT][NW][4], accg[MT][NW][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int j = 0; j < NW; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) { acc[m][j][r] = 0.f; accg[m][j][r] = 0.f; }

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < ntile) issue(wk0 + s, s);
    cp_commit();
  }
  for (int i = 0; i < ntile; ++i) {
    const int st = i % STAGES;
    const int nx = i + STAGES - 1;
    if (nx < ntile) issue(wk0 + nx, nx % STAGES);
    cp_commit();
    cp_wait<STAGES - 1>();                                 // плитка i приехала (свои копии нити)
    __syncwarp();                                          // x и meta клали ДРУГИЕ полосы варпа
    const int kt = wk0 + i;
    // нули группы для B-столбца g и масштабы для столбцов аккумулятора 2t, 2t+1
    uint32_t zz[NW];
#pragma unroll
    for (int j = 0; j < NW; ++j) zz[j] = (sm.meta[st][j][g] & 0xFFFF0000u) | (sm.meta[st][j][g] >> 16);
#pragma unroll
    for (int c = 0; c < 2; ++c) {                          // два 32-куска k
      uint32_t a0[MT][4], a1[MT][4];
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        uint4 r0 = make_uint4(0, 0, 0, 0), r1 = make_uint4(0, 0, 0, 0);
        if (XR >= 16 || g < XR) r0 = *reinterpret_cast<const uint4*>(&sm.x[st][m * 16 + (XR >= 16 ? g : (g < XR ? g : 0))][c * 32 + 8 * t]);
        if (XR >= 16) r1 = *reinterpret_cast<const uint4*>(&sm.x[st][m * 16 + g + 8][c * 32 + 8 * t]);
        a0[m][0] = r0.x; a0[m][1] = r1.x; a0[m][2] = r0.y; a0[m][3] = r1.y;
        a1[m][0] = r0.z; a1[m][1] = r1.z; a1[m][2] = r0.w; a1[m][3] = r1.w;
      }
#pragma unroll
      for (int j = 0; j < NW; ++j) {
        const uint4 wv = sm.w[st][j][lane];
        const uint32_t w0 = c ? wv.z : wv.x, w1 = c ? wv.w : wv.y;
        uint32_t b0, b1, b2, b3;
        deq4(w0, zz[j], b0, b1);
        deq4(w1, zz[j], b2, b3);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          mma16816(accg[m][j], a0[m], b0, b1);
          mma16816(accg[m][j], a1[m], b2, b3);
        }
      }
    }
    if ((kt & 1) || i == ntile - 1) {                      // конец группы 128 (или конец полосы варпа)
#pragma unroll
      for (int j = 0; j < NW; ++j) {
        const float s0 = __half2float(__ushort_as_half((unsigned short)(sm.meta[st][j][2 * t] & 0xFFFFu)));
        const float s1 = __half2float(__ushort_as_half((unsigned short)(sm.meta[st][j][2 * t + 1] & 0xFFFFu)));
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          acc[m][j][0] += s0 * accg[m][j][0]; acc[m][j][1] += s1 * accg[m][j][1];
          acc[m][j][2] += s0 * accg[m][j][2]; acc[m][j][3] += s1 * accg[m][j][3];
          accg[m][j][0] = accg[m][j][1] = accg[m][j][2] = accg[m][j][3] = 0.f;
        }
      }
    }
    __syncwarp();                                          // стадию st перезапишет выдача i+1
  }
  cp_wait<0>();

  // ---- свод варпов блока через разделяемую (фиксированный порядок)
  constexpr int BN = NW * 8;
  __shared__ float s_rmax[64];
  if (bmax) for (int r = threadIdx.x; r < 64; r += WARPS * 32) s_rmax[r] = -INFINITY;
  __syncthreads();
  float* red = reinterpret_cast<float*>(smem_raw);         // [WARPS][rows][BN]
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int j = 0; j < NW; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int row = m * 16 + g + 8 * (r >> 1), col = j * 8 + 2 * t + (r & 1);
        red[((long)warp * rows + row) * BN + col] = acc[m][j][r];
      }
  __syncthreads();
  const int mrows = min(M, rows);
  const bool split = gridDim.y > 1;
  for (int e = threadIdx.x; e < mrows * BN; e += WARPS * 32) {
    const int row = e / BN, col = e - row * BN;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) v += red[((long)w * rows + row) * BN + col];
    if (!split) {
      if (bmax) { if (nb0 * 8 + col < nvalid) atomic_max_f(&s_rmax[row], v); }
      else y[(long)row * NO + o0 + col] = __float2half(v);
    } else ws[((long)blockIdx.y * M + row) * NO + o0 + col] = v;
  }
  if (bmax && !split) {
    __syncthreads();
    for (int r = threadIdx.x; r < mrows; r += WARPS * 32) bmax[(long)r * gridDim.x + blockIdx.x] = s_rmax[r];
  }
  if (!split) return;
  // ---- детерминированный свод split-K: последний блок плитки N складывает слоты по порядку
  __threadfence();
  __syncthreads();
  __shared__ int last;
  if (threadIdx.x == 0) {
    const int prev = atomicAdd(&cnt[blockIdx.x], 1);
    last = (prev == (int)gridDim.y - 1);
  }
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int e = threadIdx.x; e < mrows * BN; e += WARPS * 32) {
    const int row = e / BN, col = e - row * BN;
    float v = 0.f;
    for (int s = 0; s < (int)gridDim.y; ++s) v += __ldcg(&ws[((long)s * M + row) * NO + o0 + col]);
    if (bmax) { if (nb0 * 8 + col < nvalid) atomic_max_f(&s_rmax[row], v); }
    else y[(long)row * NO + o0 + col] = __float2half(v);
  }
  if (bmax) {
    __syncthreads();
    for (int r = threadIdx.x; r < mrows; r += WARPS * 32) bmax[(long)r * gridDim.x + blockIdx.x] = s_rmax[r];
  }
  if (threadIdx.x == 0) cnt[blockIdx.x] = 0;               // самоочистка: следующий вызов (и повтор графа)
}

// ------------------------------------------------------------------ перепаковка
// qweight [K/4, N] int32 (4 подряд k одного столбца в слове), qzeros [G, N/4] int32, scales [G, N] fp16.
// wf[nb][kt][lane] (uint4): нить (g,t) плитки (nb,kt) держит слова qweight[kt*16 + 8c + 2t + s][nb*8+g],
// порядок (c,s) = (0,0),(0,1),(1,0),(1,1) -> .x .y .z .w.
// Порядок плиток -- wf[nbg][kt][j][lane], nbg = nb / NW, j = nb % NW: веса варпа на плитку k64 -- СПЛОШНЫЕ
// NW*512 Б, а полоса варпа по k -- один непрерывный поток (меньше открытий строк DRAM).
__global__ void pack_kernel(const int* __restrict__ qw, uint4* __restrict__ wf, int N, int K, int NW) {
  const long total = (long)(N / 8) * (K / 64) * 32;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int lane = (int)(i & 31);
    const long tile = i >> 5;
    const int KT = K / 64;
    const int j = (int)(tile % NW);
    const long rest = tile / NW;
    const int kt = (int)(rest % KT), nbg = (int)(rest / KT);
    const int nb = nbg * NW + j;
    const int g = lane >> 2, t = lane & 3, n = nb * 8 + g;
    const int r = kt * 16 + 2 * t;
    uint4 v;
    v.x = (uint32_t)qw[(long)(r + 0) * N + n];
    v.y = (uint32_t)qw[(long)(r + 1) * N + n];
    v.z = (uint32_t)qw[(long)(r + 8) * N + n];
    v.w = (uint32_t)qw[(long)(r + 9) * N + n];
    wf[i] = v;
  }
}
__global__ void meta_kernel(const int* __restrict__ qz, const __half* __restrict__ sc, uint32_t* __restrict__ meta,
                            int G, int N, float zbias) {
  const long total = (long)G * N;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int gi = (int)(i / N), n = (int)(i - (long)gi * N);
    const int z = (qz[(long)gi * (N / 4) + (n >> 2)] >> ((n & 3) * 8)) & 0xFF;
    const __half zz = __float2half(1024.f + (float)z + zbias);   // точно: целое <= 1280
    meta[i] = (uint32_t)__half_as_ushort(sc[i]) | ((uint32_t)__half_as_ushort(zz) << 16);
  }
}

// Обратная перепаковка: фрагментный порядок -> W [N][K] fp16 (большие M: деквант + cuBLAS). Одна нить --
// один uint4 плитки (16 весов одного столбца): 4 слова по 4 подряд идущих k, запись по 8 Б.
__global__ void dequant_kernel(const uint4* __restrict__ wf, const uint32_t* __restrict__ meta,
                               __half* __restrict__ W, int N, int K, int NW) {
  const long total = (long)(N / 8) * (K / 64) * 32;
  const int KT = K / 64;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int lane = (int)(i & 31);
    const long tile = i >> 5;
    const int j = (int)(tile % NW);
    const long rest = tile / NW;
    const int kt = (int)(rest % KT), nbg = (int)(rest / KT);
    const int n = (nbg * NW + j) * 8 + (lane >> 2), t = lane & 3;
    const uint4 v = wf[i];
    const uint32_t m = meta[(long)((kt * 64) >> 7) * N + n];
    const __half2 s2 = __half2half2(__ushort_as_half((unsigned short)(m & 0xFFFFu)));
    const uint32_t zz2 = (m & 0xFFFF0000u) | (m >> 16);
    const uint32_t w4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int k = kt * 64 + 32 * (q >> 1) + 8 * t + 4 * (q & 1);
      uint32_t lo, hi;
      deq4(w4[q], zz2, lo, hi);
      __half2 a = __hmul2(*reinterpret_cast<__half2*>(&lo), s2), b = __hmul2(*reinterpret_cast<__half2*>(&hi), s2);
      uint2 o;
      o.x = *reinterpret_cast<uint32_t*>(&a); o.y = *reinterpret_cast<uint32_t*>(&b);
      *reinterpret_cast<uint2*>(W + (long)n * K + k) = o;
    }
  }
}


// =====================================================================================================
// W12A16 на sm_80 (12-битные qkv внимания, симметричные, группа 128). Формат перепаковки -- ПЛОСКОСТИ:
// младший байт + старший полубайт (вольтовская записка про int12: "8-битная плоскость + плоскость
// полубайтов, без конверсий"). Плитка n8 x k128 (= одна группа масштаба), у нити 48 Б подряд:
//   uint4 lo0 -- младшие байты 32-кусков c=0,1 (k = 32c + 8t .. +7: .x/.y -- кусок 0, .z/.w -- кусок 1),
//   uint4 lo1 -- куски 2,3;  uint4 hi -- слово c: восемь полубайтов куска c в порядке, где пара
//   (v_{2i}, v_{2i+1}) берётся ОДНИМ (w >> 4i) & 0x000F000F.
// Пара значений -> half2 точно: H = 1024+hi, L = 1024+lo (магия 0x6400), v = (H-1024)*256 + (L-3072).
// =====================================================================================================
constexpr int LDX12 = 160;   // 128 k + 32: шаг 320 Б = 80 слов, 80 mod 32 = 16 -> LDS.128 без конфликтов

template <int NW, int XR, int STAGES>
struct WarpSmem12 {
  uint4 w[STAGES][NW][32][3];
  __half x[STAGES][XR][LDX12];
  uint32_t sc[STAGES][NW][8];
};

__device__ __forceinline__ void deq_pair12(uint32_t lo4, uint32_t sel, uint32_t hiw, int sh, uint32_t& out) {
  uint32_t L = __byte_perm(lo4, 0x64646464u, sel);                 // (1024+lo_a, 1024+lo_b)
  uint32_t H = ((hiw >> sh) & 0x000F000Fu) | 0x64006400u;          // (1024+hi_a, 1024+hi_b)
  const uint32_t c1024 = 0x64006400u, c3072 = 0x6A006A00u, c256 = 0x5C005C00u;
  __half2 h = __hsub2(*reinterpret_cast<__half2*>(&H), *reinterpret_cast<const __half2*>(&c1024));
  __half2 l = __hsub2(*reinterpret_cast<__half2*>(&L), *reinterpret_cast<const __half2*>(&c3072));
  __half2 v = __hfma2(h, *reinterpret_cast<const __half2*>(&c256), l);
  out = *reinterpret_cast<uint32_t*>(&v);
}

template <int NW, int XR, int STAGES, int WARPS>
__global__ __launch_bounds__(WARPS * 32) void w12a16_sm80_kernel(
    const __half* __restrict__ x, const uint4* __restrict__ wf, const __half* __restrict__ scales,
    __half* __restrict__ y, float* __restrict__ ws, int* __restrict__ cnt,
    int M, int N, int K, int tiles_per_split) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  constexpr int MT = (XR + 15) / 16;
  using WS = WarpSmem12<NW, XR, STAGES>;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  WS& sm = reinterpret_cast<WS*>(smem_raw)[warp];
  const int g = lane >> 2, t = lane & 3;
  const int KT = K >> 7;                                   // плиток k128
  const int nb0 = blockIdx.x * NW;
  const int kt0 = blockIdx.y * tiles_per_split, kt1 = min(KT, kt0 + tiles_per_split);
  const int per_w = (kt1 - kt0 + WARPS - 1) / WARPS;
  const int wk0 = min(kt1, kt0 + warp * per_w), wk1 = min(kt1, wk0 + per_w);
  const int ntile = wk1 - wk0;
  const int rows = MT * 16;
  auto issue = [&](int kt, int st) {
#pragma unroll
    for (int j = 0; j < NW; ++j) {
      const uint4* src = wf + ((((long)blockIdx.x * KT + kt) * NW + j) * 32 + lane) * 3;
#pragma unroll
      for (int q = 0; q < 3; ++q) cp_async16_cg(&sm.w[st][j][lane][q], src + q);
    }
#pragma unroll
    for (int c = lane; c < XR * 16; c += 32) {
      const int r = c >> 4, q = c & 15;
      const bool live = r < M;
      cp_async16_ca_z(&sm.x[st][r][q * 8], x + (long)(live ? r : 0) * K + kt * 128 + q * 8, live);
    }
#pragma unroll
    for (int c = lane; c < NW * 8; c += 32) {
      // масштаб fp16 столбца -> младшее полуслово слова (cp.async 4 Б берёт пару столбцов; читаем своё)
      cp_async4_ca(&sm.sc[st][c >> 3][c & 7], scales + (long)kt * N + (nb0 + (c >> 3)) * 8 + (c & 7) - ((c & 7) & 1));
    }
  };
  float acc[MT][NW][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int j = 0; j < NW; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[m][j][r] = 0.f;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) { if (s < ntile) issue(wk0 + s, s); cp_commit(); }
  for (int i = 0; i < ntile; ++i) {
    const int st = i % STAGES;
    const int nx = i + STAGES - 1;
    if (nx < ntile) issue(wk0 + nx, nx % STAGES);
    cp_commit();
    cp_wait<STAGES - 1>();
    __syncwarp();
    float accg[MT][NW][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int j = 0; j < NW; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) accg[m][j][r] = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {                          // четыре 32-куска k
      uint32_t a0[MT][4], a1[MT][4];
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        uint4 r0 = make_uint4(0, 0, 0, 0), r1 = make_uint4(0, 0, 0, 0);
        if (XR >= 16 || g < XR) r0 = *reinterpret_cast<const uint4*>(&sm.x[st][m * 16 + (XR >= 16 ? g : (g < XR ? g : 0))][c * 32 + 8 * t]);
        if (XR >= 16) r1 = *reinterpret_cast<const uint4*>(&sm.x[st][m * 16 + g + 8][c * 32 + 8 * t]);
        a0[m][0] = r0.x; a0[m][1] = r1.x; a0[m][2] = r0.y; a0[m][3] = r1.y;
        a1[m][0] = r0.z; a1[m][1] = r1.z; a1[m][2] = r0.w; a1[m][3] = r1.w;
      }
#pragma unroll
      for (int j = 0; j < NW; ++j) {
        const uint4 lo = sm.w[st][j][lane][c >> 1];
        const uint4 hv = sm.w[st][j][lane][2];
        const uint32_t l0 = (c & 1) ? lo.z : lo.x, l1 = (c & 1) ? lo.w : lo.y;
        const uint32_t hw = (c == 0) ? hv.x : (c == 1) ? hv.y : (c == 2) ? hv.z : hv.w;
        uint32_t b0, b1, b2, b3;
        deq_pair12(l0, 0x5140u, hw, 0, b0);  deq_pair12(l0, 0x5342u, hw, 4, b1);
        deq_pair12(l1, 0x5140u, hw, 8, b2);  deq_pair12(l1, 0x5342u, hw, 12, b3);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          mma16816(accg[m][j], a0[m], b0, b1);
          mma16816(accg[m][j], a1[m], b2, b3);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < NW; ++j) {
      const uint32_t w01 = sm.sc[st][j][2 * t];            // пара (столбец 2t, 2t+1) одним словом
      const float s0 = __half2float(__ushort_as_half((unsigned short)(w01 & 0xFFFFu)));
      const float s1 = __half2float(__ushort_as_half((unsigned short)(w01 >> 16)));
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        acc[m][j][0] += s0 * accg[m][j][0]; acc[m][j][1] += s1 * accg[m][j][1];
        acc[m][j][2] += s0 * accg[m][j][2]; acc[m][j][3] += s1 * accg[m][j][3];
      }
    }
    __syncwarp();
  }
  cp_wait<0>();
  constexpr int BN = NW * 8;
  __syncthreads();
  float* red = reinterpret_cast<float*>(smem_raw);
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int j = 0; j < NW; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int row = m * 16 + g + 8 * (r >> 1), col = j * 8 + 2 * t + (r & 1);
        red[((long)warp * rows + row) * BN + col] = acc[m][j][r];
      }
  __syncthreads();
  const int mrows = min(M, rows);
  const int n0 = nb0 * 8;
  const bool split = gridDim.y > 1;
  for (int e = threadIdx.x; e < mrows * BN; e += WARPS * 32) {
    const int row = e / BN, col = e - row * BN;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) v += red[((long)w * rows + row) * BN + col];
    if (!split) y[(long)row * N + n0 + col] = __float2half(v);
    else ws[((long)blockIdx.y * M + row) * N + n0 + col] = v;
  }
  if (!split) return;
  __threadfence();
  __syncthreads();
  __shared__ int last;
  if (threadIdx.x == 0) last = (atomicAdd(&cnt[blockIdx.x], 1) == (int)gridDim.y - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int e = threadIdx.x; e < mrows * BN; e += WARPS * 32) {
    const int row = e / BN, col = e - row * BN;
    float v = 0.f;
    for (int s2 = 0; s2 < (int)gridDim.y; ++s2) v += __ldcg(&ws[((long)s2 * M + row) * N + n0 + col]);
    y[(long)row * N + n0 + col] = __float2half(v);
  }
  if (threadIdx.x == 0) cnt[blockIdx.x] = 0;
}

// STREAM-K для W12 на декоде (M<=32). Прежнее ядро упиралось не в счёт, а в ХВОСТ ВОЛНЫ: на форме qkv
// [7168,5120] 448 блоков при 3 на SM = 1.56 волны, ~71 мкс при любом M (потолок чтения 50.7 мкс).
// Здесь сетка = ровно столько блоков, сколько помещается разом (SM x cps), варпы НЕЗАВИСИМЫ (никаких
// __syncthreads): единица работы -- (n-блок, плитка k128), U = NB*KT единиц в порядке n-major, и варп gw
// берёт РАВНЫЙ непрерывный отрезок [gw*U/W, (gw+1)*U/W). В плоском формате вес единицы u лежит по адресу
// u*NW*1.5 КБ, то есть отрезок варпа -- один сплошной поток памяти. Отрезок пересекает границы n-блоков:
// n-блок, покрытый варпом целиком, пишется сразу в fp16; на стыке варп кладёт fp32-часть в ws, а
// ПОСЛЕДНИЙ пришедший (счётчик cnt[nb]) складывает части В ПОРЯДКЕ НОМЕРОВ ВАРПОВ -- результат не зависит
// от того, кто пришёл последним (детерминизм от запуска к запуску). Счётчик обнуляет он же.
template <int NW, int XR, int STAGES, int WARPS>
__global__ __launch_bounds__(WARPS * 32) void w12a16_sm80_sk_kernel(
    const __half* __restrict__ x, const uint4* __restrict__ wf, const __half* __restrict__ scales,
    __half* __restrict__ y, float* __restrict__ ws, int* __restrict__ cnt,
    int M, int N, int K, int U, int W, int pmax) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  using WS = WarpSmem12<NW, XR, STAGES>;
  constexpr int BN = NW * 8, MT = (XR + 15) / 16, ROWS = MT * 16;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  WS& sm = reinterpret_cast<WS*>(smem_raw)[warp];
  const int g = lane >> 2, t = lane & 3;
  const int KT = K >> 7;
  const int gw = blockIdx.x * WARPS + warp;
  const int u0 = (int)((long)gw * U / W), u1 = (int)((long)(gw + 1) * U / W);
  const int n = u1 - u0;
  if (n <= 0) return;
  auto wof = [&](long u) { return (int)(((u + 1) * W + U - 1) / U - 1); };   // варп, чей отрезок содержит u
  auto issue = [&](int u, int st) {
    const int nb = u / KT, kt = u - nb * KT;
    // ЛИНЕЙНОЕ копирование: вес единицы -- сплошные NW*1.5 КБ, раскладка в smem та же ([j][lane][3]),
    // но каждая инструкция берёт 512 Б подряд (16 целых секторов). Чтение "своих 48 Б" нитью
    // (src + lane*3 + q) трогало бы 32 сектора по половине и удваивало поток запросов в L2.
    const uint4* src = wf + (long)u * NW * 96;
    uint4* dstw = &sm.w[st][0][0][0];
#pragma unroll
    for (int q = 0; q < NW * 3; ++q) cp_async16_cg(dstw + q * 32 + lane, src + q * 32 + lane);
#pragma unroll
    for (int c = lane; c < XR * 16; c += 32) {
      const int r = c >> 4, q = c & 15;
      const bool live = r < M;
      cp_async16_ca_z(&sm.x[st][r][q * 8], x + (long)(live ? r : 0) * K + kt * 128 + q * 8, live);
    }
#pragma unroll
    for (int c = lane; c < NW * 8; c += 32)
      cp_async4_ca(&sm.sc[st][c >> 3][c & 7], scales + (long)kt * N + (nb * NW + (c >> 3)) * 8 + (c & 7) - ((c & 7) & 1));
  };
  float acc[MT][NW][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int j = 0; j < NW; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[m][j][r] = 0.f;
  auto flush = [&](int nb, bool full) {
    const int n0 = nb * BN;
    if (full) {
#pragma unroll
      for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int j = 0; j < NW; ++j) {
          const int col = n0 + j * 8 + 2 * t, r0 = m * 16 + g;
          if (r0 < M) *reinterpret_cast<__half2*>(&y[(long)r0 * N + col]) = __floats2half2_rn(acc[m][j][0], acc[m][j][1]);
          if (XR > 8 && r0 + 8 < M)
            *reinterpret_cast<__half2*>(&y[(long)(r0 + 8) * N + col]) = __floats2half2_rn(acc[m][j][2], acc[m][j][3]);
        }
      return;
    }
    const int fw = wof((long)nb * KT), P = wof((long)nb * KT + KT - 1) - fw + 1;
    float* base = ws + (long)nb * pmax * ROWS * BN;
    float* dst = base + (long)(gw - fw) * ROWS * BN;
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int j = 0; j < NW; ++j) {
        const int col = j * 8 + 2 * t, r0 = m * 16 + g;
        if (r0 < M) *reinterpret_cast<float2*>(&dst[r0 * BN + col]) = make_float2(acc[m][j][0], acc[m][j][1]);
        if (XR > 8 && r0 + 8 < M)
          *reinterpret_cast<float2*>(&dst[(r0 + 8) * BN + col]) = make_float2(acc[m][j][2], acc[m][j][3]);
      }
    __threadfence();
    __syncwarp();
    int old = 0;
    if (lane == 0) old = atomicAdd(&cnt[nb], 1);
    old = __shfl_sync(0xffffffffu, old, 0);
    if (old != P - 1) return;
    __threadfence();
#pragma unroll
    for (int j = 0; j < NW; ++j) {
      const int col = j * 8 + 2 * t;
#pragma unroll
      for (int h = 0; h < (XR > 8 ? 2 * MT : 1); ++h) {
        const int row = g + 8 * h;
        if (row >= M) continue;
        float2 s = make_float2(0.f, 0.f);
        for (int p = 0; p < P; ++p) {
          const float2 v = __ldcg(reinterpret_cast<const float2*>(&base[((long)p * ROWS + row) * BN + col]));
          s.x += v.x; s.y += v.y;
        }
        *reinterpret_cast<__half2*>(&y[(long)row * N + n0 + col]) = __floats2half2_rn(s.x, s.y);
      }
    }
    if (lane == 0) cnt[nb] = 0;
  };
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) { if (s < n) issue(u0 + s, s); cp_commit(); }
  int nb = u0 / KT, kt = u0 - nb * KT, kseg = kt;
  for (int i = 0; i < n; ++i) {
    const int st = i % STAGES;
    const int nx = i + STAGES - 1;
    if (nx < n) issue(u0 + nx, nx % STAGES);
    cp_commit();
    cp_wait<STAGES - 1>();
    __syncwarp();
    float accg[MT][NW][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int j = 0; j < NW; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) accg[m][j][r] = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      uint32_t a0[MT][4], a1[MT][4];
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        uint4 r0 = make_uint4(0, 0, 0, 0), r1 = make_uint4(0, 0, 0, 0);
        if (XR >= 16 || g < XR) r0 = *reinterpret_cast<const uint4*>(&sm.x[st][m * 16 + (XR >= 16 ? g : (g < XR ? g : 0))][c * 32 + 8 * t]);
        if (XR >= 16) r1 = *reinterpret_cast<const uint4*>(&sm.x[st][(m * 16 + g + 8) % XR][c * 32 + 8 * t]);
        a0[m][0] = r0.x; a0[m][1] = r1.x; a0[m][2] = r0.y; a0[m][3] = r1.y;
        a1[m][0] = r0.z; a1[m][1] = r1.z; a1[m][2] = r0.w; a1[m][3] = r1.w;
      }
#pragma unroll
      for (int j = 0; j < NW; ++j) {
        const uint4 lo = sm.w[st][j][lane][c >> 1];
        const uint4 hv = sm.w[st][j][lane][2];
        const uint32_t l0 = (c & 1) ? lo.z : lo.x, l1 = (c & 1) ? lo.w : lo.y;
        const uint32_t hw = (c == 0) ? hv.x : (c == 1) ? hv.y : (c == 2) ? hv.z : hv.w;
        uint32_t b0, b1, b2, b3;
        deq_pair12(l0, 0x5140u, hw, 0, b0);  deq_pair12(l0, 0x5342u, hw, 4, b1);
        deq_pair12(l1, 0x5140u, hw, 8, b2);  deq_pair12(l1, 0x5342u, hw, 12, b3);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          mma16816(accg[m][j], a0[m], b0, b1);
          mma16816(accg[m][j], a1[m], b2, b3);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < NW; ++j) {
      const uint32_t w01 = sm.sc[st][j][2 * t];
      const float s0 = __half2float(__ushort_as_half((unsigned short)(w01 & 0xFFFFu)));
      const float s1 = __half2float(__ushort_as_half((unsigned short)(w01 >> 16)));
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        acc[m][j][0] += s0 * accg[m][j][0]; acc[m][j][1] += s1 * accg[m][j][1];
        acc[m][j][2] += s0 * accg[m][j][2]; acc[m][j][3] += s1 * accg[m][j][3];
      }
    }
    __syncwarp();
    if (kt == KT - 1 || i == n - 1) {
      flush(nb, kseg == 0 && kt == KT - 1);
#pragma unroll
      for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int j = 0; j < NW; ++j)
#pragma unroll
          for (int r = 0; r < 4; ++r) acc[m][j][r] = 0.f;
    }
    if (++kt == KT) { kt = 0; kseg = 0; ++nb; }
  }
  cp_wait<0>();
}

// qw [K/8, 3, N] int32 (8 подряд k столбца = 96 разрядов) -> плоскости плитки n8 x k128 (см. шапку).
__global__ void pack12_kernel(const int* __restrict__ qw, uint4* __restrict__ wf, int N, int K, int NW) {
  const long total = (long)(N / 8) * (K / 128) * 32;
  const int KT = K / 128;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int lane = (int)(i & 31);
    const long tile = i >> 5;
    const int j = (int)(tile % NW);
    const long rest = tile / NW;
    const int kt = (int)(rest % KT), nbg = (int)(rest / KT);
    const int n = (nbg * NW + j) * 8 + (lane >> 2), t = lane & 3;
    uint32_t lo[8], hi[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const long b = (long)((kt * 128 + 32 * c + 8 * t) >> 3) * 3;
      const unsigned w0 = (unsigned)qw[(b + 0) * N + n], w1 = (unsigned)qw[(b + 1) * N + n], w2 = (unsigned)qw[(b + 2) * N + n];
      unsigned u[8];
      u[0] = w0 & 0xFFFu; u[1] = (w0 >> 12) & 0xFFFu; u[2] = ((w0 >> 24) & 0xFFu) | ((w1 & 0xFu) << 8);
      u[3] = (w1 >> 4) & 0xFFFu; u[4] = (w1 >> 16) & 0xFFFu; u[5] = ((w1 >> 28) & 0xFu) | ((w2 & 0xFFu) << 4);
      u[6] = (w2 >> 8) & 0xFFFu; u[7] = (w2 >> 20) & 0xFFFu;
      lo[2 * c] = (u[0] & 0xFF) | ((u[1] & 0xFF) << 8) | ((u[2] & 0xFF) << 16) | ((u[3] & 0xFF) << 24);
      lo[2 * c + 1] = (u[4] & 0xFF) | ((u[5] & 0xFF) << 8) | ((u[6] & 0xFF) << 16) | ((u[7] & 0xFF) << 24);
      uint32_t h = 0;
#pragma unroll
      for (int p = 0; p < 4; ++p) {                      // пара (v_{2p}, v_{2p+1}): разряды 4p и 16+4p
        h |= (u[2 * p] >> 8) << (4 * p);
        h |= (u[2 * p + 1] >> 8) << (16 + 4 * p);
      }
      hi[c] = h;
    }
    wf[i * 3 + 0] = make_uint4(lo[0], lo[1], lo[2], lo[3]);
    wf[i * 3 + 1] = make_uint4(lo[4], lo[5], lo[6], lo[7]);
    wf[i * 3 + 2] = make_uint4(hi[0], hi[1], hi[2], hi[3]);
  }
}

// Обратно: плоскости -> W [N, K] fp16 (для больших M: один буфер + cuBLAS). Значение = (hi*256 + lo) - 2048,
// умноженное на масштаб группы в fp32 и округлённое в fp16 -- побитово как w12_dequant_kernel (w12a16.cu).
// Нить = (плитка, полоса): 48 Б на входе, 4 куска по 8 значений = 4 записи по 16 Б.
__global__ void dequant12_kernel(const uint4* __restrict__ wf, const __half* __restrict__ sc,
                                 __half* __restrict__ out, int N, int K, int NW) {
  const long total = (long)(N / 8) * (K / 128) * 32;
  const int KT = K / 128;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int lane = (int)(i & 31);
    const long tile = i >> 5;
    const int j = (int)(tile % NW);
    const long rest = tile / NW;
    const int kt = (int)(rest % KT), nbg = (int)(rest / KT);
    const int n = (nbg * NW + j) * 8 + (lane >> 2), t = lane & 3;
    const uint4 l0 = wf[i * 3 + 0], l1 = wf[i * 3 + 1], hv = wf[i * 3 + 2];
    const uint32_t lo[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
    const uint32_t hi[4] = {hv.x, hv.y, hv.z, hv.w};
    const float s = __half2float(sc[(long)kt * N + n]);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      __align__(16) __half o[8];
#pragma unroll
      for (int v = 0; v < 8; ++v) {
        const uint32_t l = (lo[2 * c + (v >> 2)] >> (8 * (v & 3))) & 0xFFu;
        const uint32_t h = (hi[c] >> (4 * (v >> 1) + 16 * (v & 1))) & 0xFu;
        o[v] = __float2half((float)((int)(h * 256u + l) - 2048) * s);
      }
      *reinterpret_cast<uint4*>(out + (long)n * K + kt * 128 + 32 * c + 8 * t) = *reinterpret_cast<const uint4*>(o);
    }
  }
}

// =====================================================================================================
// БОЛЬШИЕ M (префилл): ПРЯМОЙ GEMM на Ampere вместо "деквант в fp16 + cuBLAS". Плитка 128x128x64, 8 варпов
// сеткой 2x4 (тайл варпа 64x32), cp.async-конвейер GS стадий. A (x) -- в разделяемой со свизлом под
// переставленную карту k (нить берёт 16 Б подряд: k = 32c + 8t .. +7 строки g и g+8): физический 16-Б кусок =
// кусок ^ ((строка & 1) << 2) -- строки g и g+1 четверти варпа ложатся на разные половины банков. B -- ТЕ ЖЕ
// фрагментные плитки, что у декодного ядра (n8 x k64, 16 Б на нить): у нити сразу готовы слова фрагмента,
// без ldmatrix. Деквант в регистрах перед MMA: (1024+q)-(1024+z) точно, затем * s в fp16 -- та же точность,
// что у прежнего пути (вес fp16 после декванта + cuBLAS).
// =====================================================================================================
// MI -- плиток m16 на варп по M: блок = 32*MI строк (MI=4: 128, MI=2: 64, MI=1: 32). Малые MI -- для
// 16 < M <= 64 (пачка запросов с MTP): x читается ОДИН раз на блок из 128 столбцов, а не на каждые
// 8*NW столбцов, как у ядра малых M (там при M=64 повторное чтение x из L2 вчетверо больше веса).
constexpr int GBN = 128, GBK = 64, GS = 3;
template <int MI>
struct GemmSmem {
  __half a[GS][32 * MI][GBK];              // 16 КиБ на стадию при MI=4, свизл кусков
  uint4 b[GS][GBN / 8][32];                // 8 КиБ на стадию: 16 n8-плиток по 512 Б
  uint32_t meta[GS][GBN];                  // (s | 1024+z) на столбец, группа плитки
};

template <int NW, int MI>
__global__ __launch_bounds__(256, 2) void w8a16_sm80_gemm(
    const __half* __restrict__ x, const uint4* __restrict__ wf, const uint32_t* __restrict__ meta,
    __half* __restrict__ y, float* __restrict__ ws, int* __restrict__ cnt, int M, int N, int K, int tps) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  constexpr int GBM = 32 * MI;
  GemmSmem<MI>& sm = *reinterpret_cast<GemmSmem<MI>*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;               // 2 x 4
  const int m0 = blockIdx.y * GBM, n0 = blockIdx.x * GBN;
  const int KT = K >> 6;
  auto issue = [&](int kt, int st) {
#pragma unroll
    for (int i = 0; i < MI; ++i) {                       // A: 32*MI строк x 8 кусков / 256 нитей
      const int c = tid + i * 256, r = c >> 3, ch = c & 7;
      const bool live = (m0 + r) < M;
      const __half* src = x + (long)(live ? m0 + r : 0) * K + kt * 64 + ch * 8;
      cp_async16_ca_z(&sm.a[st][r][(ch ^ ((r & 1) << 2)) * 8], src, live);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {                        // B: 16 плиток x 32 куска = 512 / 256 нитей
      const int c = tid + i * 256, j = c >> 5, ln = c & 31;
      const int nb = (n0 >> 3) + j;
      cp_async16_cg(&sm.b[st][j][ln], wf + ((((long)(nb / NW)) * KT + kt) * NW + (nb % NW)) * 32 + ln);
    }
    if (tid < 32) {                                      // meta: 128 столбцов по 4 Б = 32 куска по 16 Б
      cp_async16_ca_z(&sm.meta[st][tid * 4], meta + (long)((kt * 64) >> 7) * N + n0 + tid * 4, true);
    }
  };
  float acc[MI][4][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;
#pragma unroll
  // split-K: блок z берёт плитки k [k0, k0+nt). При малых M плиток N мало (down: 40 на 96 SM), и без
  // расщепления карта простаивает -- время было плоским по M (упор в длину цикла одного блока).
  const int k0 = blockIdx.z * tps, nt = min(KT, k0 + tps) - k0;
  for (int s = 0; s < GS - 1; ++s) { if (s < nt) issue(k0 + s, s); cp_commit(); }
  for (int it = 0; it < nt; ++it) {
    const int st = it % GS;
    cp_wait<GS - 2>();
    __syncthreads();                                     // стадия it видна всем; стадия it-1 всеми дочитана
    if (it + GS - 1 < nt) issue(k0 + it + GS - 1, (it + GS - 1) % GS);
    cp_commit();
    // веса варпа: 4 n8-плитки, у нити 16 Б каждой = 4 слова (c,s) = (0,0),(0,1),(1,0),(1,1)
    uint4 wv[4];
    uint32_t zz[4];
    __half2 s2[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int jj = wn * 4 + j;
      wv[j] = sm.b[st][jj][lane];
      const uint32_t mm = sm.meta[st][jj * 8 + g];
      zz[j] = (mm & 0xFFFF0000u) | (mm >> 16);
      s2[j] = __half2half2(__ushort_as_half((unsigned short)(mm & 0xFFFFu)));
    }
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      uint32_t bq[4][4];                                 // [n8][b0 b1 (шаг0) b2 b3 (шаг1)]
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t w0 = c ? wv[j].z : wv[j].x, w1 = c ? wv[j].w : wv[j].y;
        uint32_t d[4];
        deq4(w0, zz[j], d[0], d[1]);
        deq4(w1, zz[j], d[2], d[3]);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          __half2 h = __hmul2(*reinterpret_cast<__half2*>(&d[e]), s2[j]);
          bq[j][e] = *reinterpret_cast<uint32_t*>(&h);
        }
      }
#pragma unroll
      for (int i = 0; i < MI; ++i) {
        const int r0 = wm * (16 * MI) + i * 16 + g, r1 = r0 + 8;
        const int ch = 4 * c + t;
        const uint4 v0 = *reinterpret_cast<const uint4*>(&sm.a[st][r0][(ch ^ ((r0 & 1) << 2)) * 8]);
        const uint4 v1 = *reinterpret_cast<const uint4*>(&sm.a[st][r1][(ch ^ ((r1 & 1) << 2)) * 8]);
        const uint32_t a0[4] = {v0.x, v1.x, v0.y, v1.y};
        const uint32_t a1[4] = {v0.z, v1.z, v0.w, v1.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          mma16816(acc[i][j], a0, bq[j][0], bq[j][1]);
          mma16816(acc[i][j], a1, bq[j][2], bq[j][3]);
        }
      }
    }
  }
  cp_wait<0>();
  const bool split = gridDim.z > 1;
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = n0 + wn * 32 + j * 8 + 2 * t;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int row = m0 + wm * (16 * MI) + i * 16 + g + 8 * hh;
        if (row >= M) continue;
        if (!split)
          *reinterpret_cast<__half2*>(y + (long)row * N + col) = __floats2half2_rn(acc[i][j][2 * hh], acc[i][j][2 * hh + 1]);
        else
          *reinterpret_cast<float2*>(ws + ((long)blockIdx.z * M + row) * N + col) = make_float2(acc[i][j][2 * hh], acc[i][j][2 * hh + 1]);
      }
    }
  if (!split) return;
  // детерминированный свод: последний пришедший к плитке (bx, by) блок складывает слоты по порядку z
  __threadfence();
  __syncthreads();
  __shared__ int last;
  const int tile = blockIdx.y * gridDim.x + blockIdx.x;
  if (tid == 0) last = (atomicAdd(&cnt[tile], 1) == (int)gridDim.z - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  const int rows = min(GBM, M - m0);
  for (int e = tid; e < rows * (GBN / 2); e += 256) {
    const int r = e / (GBN / 2), c2 = (e - r * (GBN / 2)) * 2;
    float2 v = make_float2(0.f, 0.f);
    for (int z = 0; z < (int)gridDim.z; ++z) {
      const float2 p = __ldcg(reinterpret_cast<const float2*>(ws + ((long)z * M + m0 + r) * N + n0 + c2));
      v.x += p.x; v.y += p.y;
    }
    *reinterpret_cast<__half2*>(y + (long)(m0 + r) * N + n0 + c2) = __floats2half2_rn(v.x, v.y);
  }
  if (tid == 0) cnt[tile] = 0;                           // самоочистка (повтор графа)
}

// ---------------------------------------------------------------------------------------------------------------
// W8A16 v2 (24.10): блок 128 x (32*NJ), варп 64 x 8*NJ, ST стадий cp.async, один блок на SM (до 255 регистров).
// Конвейер фрагментов внутри k-плитки: слова B следующего шага читаются ДО MMA текущего, распаковка -- после них в
// тексте (INT/FP16-конвейер работает, пока тензорный занят). A читается раз на 32-кусок k и служит NJ плиткам n8.
// ---------------------------------------------------------------------------------------------------------------
// ЗАКРЕПЛЁННЫЙ ПОРЯДОК (урок V100: ptxas выносит все LDS после BAR пачкой, затем все HMMA). При PIN загрузки из
// разделяемой и mma.sync -- asm volatile: их взаимный порядок ptxas не меняет, распаковка зажата между ними
// зависимостями. Без PIN -- обычные загрузки и mma (решает планировщик ptxas).
__device__ __forceinline__ void mma16816_pin(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 lds128_pin(const void* p) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(smem_u32(p)));
  return v;
}
__device__ __forceinline__ uint32_t lds32_pin(const void* p) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(smem_u32(p)));
  return v;
}

template <int NJ, int ST>
struct Gemm2Smem {
  __half a[ST][128][GBK];                  // 128 строк x 64 k, свизл кусков по чётности строки
  uint4 b[ST][4 * NJ][32];                 // 4*NJ плиток n8 по 32 полосы x 16 Б
  uint32_t meta[ST][32 * NJ];              // (s | 1024+z) на столбец блока
};

template <int NJ, int ST, bool PIN>
__global__ __launch_bounds__(256, 1) void w8a16_sm80_gemm2(
    const __half* __restrict__ x, const uint4* __restrict__ wf, const uint32_t* __restrict__ meta,
    __half* __restrict__ y, float* __restrict__ ws, int* __restrict__ cnt, int M, int N, int K, int tps, int nwl) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  constexpr int BN = 32 * NJ, NT8 = 4 * NJ;
  Gemm2Smem<NJ, ST>& sm = *reinterpret_cast<Gemm2Smem<NJ, ST>*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int wm = warp >> 2, wn = warp & 3;
  const int m0 = blockIdx.y * 128, n0 = blockIdx.x * BN;
  const int KT = K >> 6;
  auto issue = [&](int kt, int st) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {                        // A: 128 строк x 8 кусков = 1024 / 256 нитей
      const int c = tid + i * 256, r = c >> 3, ch = c & 7;
      const bool live = (m0 + r) < M;
      cp_async16_ca_z(&sm.a[st][r][(ch ^ ((r & 1) << 2)) * 8], x + (long)(live ? m0 + r : 0) * K + kt * 64 + ch * 8, live);
    }
#pragma unroll
    for (int i = 0; i < NT8 * 32 / 256; ++i) {           // B: NT8 плиток x 32 куска
      const int c = tid + i * 256, j = c >> 5, ln = c & 31;
      const int nb = (n0 >> 3) + j;
      cp_async16_cg(&sm.b[st][j][ln], wf + ((((long)(nb >> nwl)) * KT + kt) * (1 << nwl) + (nb & ((1 << nwl) - 1))) * 32 + ln);
    }
    if (tid < BN / 4)                                    // meta: BN столбцов по 4 Б
      cp_async16_ca_z(&sm.meta[st][tid * 4], meta + (long)((kt * 64) >> 7) * N + n0 + tid * 4, true);
  };
  float acc[4][NJ][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;
  const int k0 = blockIdx.z * tps, nt = min(KT, k0 + tps) - k0;
#pragma unroll
  for (int s0 = 0; s0 < ST - 1; ++s0) { if (s0 < nt) issue(k0 + s0, s0); cp_commit(); }
  for (int it = 0; it < nt; ++it) {
    const int st = it % ST;
    cp_wait<ST - 2>();
    __syncthreads();
    if (it + ST - 1 < nt) issue(k0 + it + ST - 1, (it + ST - 1) % ST);
    cp_commit();
    uint32_t zz[NJ];
    __half2 s2[NJ];
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const uint32_t mm = sm.meta[st][(wn * NJ + j) * 8 + g];
      zz[j] = (mm & 0xFFFF0000u) | (mm >> 16);
      s2[j] = __half2half2(__ushort_as_half((unsigned short)(mm & 0xFFFFu)));
    }
    const uint32_t* bw = reinterpret_cast<const uint32_t*>(&sm.b[st][wn * NJ][lane]);   // шаг плитки n8: 32 uint4
    uint32_t raw[NJ];
#pragma unroll
    for (int j = 0; j < NJ; ++j) raw[j] = PIN ? lds32_pin(&bw[j * 32 * 4 + 0]) : bw[j * 32 * 4 + 0];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      uint4 v0[4], v1[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r0 = wm * 64 + i * 16 + g, r1 = r0 + 8, ch = 4 * c + t;
        const __half* p0 = &sm.a[st][r0][(ch ^ ((r0 & 1) << 2)) * 8];
        const __half* p1 = &sm.a[st][r1][(ch ^ ((r1 & 1) << 2)) * 8];
        v0[i] = PIN ? lds128_pin(p0) : *reinterpret_cast<const uint4*>(p0);
        v1[i] = PIN ? lds128_pin(p1) : *reinterpret_cast<const uint4*>(p1);
      }
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        const int step = 2 * c + s;
        uint32_t bq[NJ][2];
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
          uint32_t d0, d1;
          deq4(raw[j], zz[j], d0, d1);
          __half2 h0 = __hmul2(*reinterpret_cast<__half2*>(&d0), s2[j]);
          __half2 h1 = __hmul2(*reinterpret_cast<__half2*>(&d1), s2[j]);
          bq[j][0] = *reinterpret_cast<uint32_t*>(&h0);
          bq[j][1] = *reinterpret_cast<uint32_t*>(&h1);
        }
        if (step < 3) {                                  // слова B следующего шага -- до MMA текущего
#pragma unroll
          for (int j = 0; j < NJ; ++j) raw[j] = PIN ? lds32_pin(&bw[j * 32 * 4 + step + 1]) : bw[j * 32 * 4 + step + 1];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint32_t a[4] = {s ? v0[i].z : v0[i].x, s ? v1[i].z : v1[i].x,
                                 s ? v0[i].w : v0[i].y, s ? v1[i].w : v1[i].y};
#pragma unroll
          for (int j = 0; j < NJ; ++j) {
            if (PIN) mma16816_pin(acc[i][j], a, bq[j][0], bq[j][1]);
            else mma16816(acc[i][j], a, bq[j][0], bq[j][1]);
          }
        }
      }
    }
  }
  cp_wait<0>();
  const bool split = gridDim.z > 1;
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
      const int col = n0 + (wn * NJ + j) * 8 + 2 * t;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int row = m0 + wm * 64 + i * 16 + g + 8 * hh;
        if (row >= M) continue;
        if (!split)
          *reinterpret_cast<__half2*>(y + (long)row * N + col) = __floats2half2_rn(acc[i][j][2 * hh], acc[i][j][2 * hh + 1]);
        else
          *reinterpret_cast<float2*>(ws + ((long)blockIdx.z * M + row) * N + col) = make_float2(acc[i][j][2 * hh], acc[i][j][2 * hh + 1]);
      }
    }
  if (!split) return;
  __threadfence();
  __syncthreads();
  __shared__ int last;
  const int tile = blockIdx.y * gridDim.x + blockIdx.x;
  if (tid == 0) last = (atomicAdd(&cnt[tile], 1) == (int)gridDim.z - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  const int rows = min(128, M - m0);
  for (int e = tid; e < rows * (BN / 2); e += 256) {
    const int r = e / (BN / 2), c2 = (e - r * (BN / 2)) * 2;
    float2 v = make_float2(0.f, 0.f);
    for (int z = 0; z < (int)gridDim.z; ++z) {
      const float2 p = __ldcg(reinterpret_cast<const float2*>(ws + ((long)z * M + m0 + r) * N + n0 + c2));
      v.x += p.x; v.y += p.y;
    }
    *reinterpret_cast<__half2*>(y + (long)(m0 + r) * N + n0 + c2) = __floats2half2_rn(v.x, v.y);
  }
  if (tid == 0) cnt[tile] = 0;
}
}  // namespace w8s80

torch::Tensor w8a16_sm80_dequant(torch::Tensor wf, torch::Tensor meta, int64_t N, int64_t K, int64_t nw,
                                 c10::optional<torch::Tensor> out) {
  const c10::cuda::CUDAGuard guard(wf.device());
  torch::Tensor W;
  if (out.has_value() && out->numel() >= N * K) W = out->view(-1).narrow(0, 0, N * K).view({N, K});
  else W = torch::empty({N, K}, wf.options().dtype(torch::kFloat16));
  auto st = at::cuda::getCurrentCUDAStream();
  w8s80::dequant_kernel<<<1024, 256, 0, st>>>(reinterpret_cast<const uint4*>(wf.data_ptr<int>()),
                                              reinterpret_cast<const uint32_t*>(meta.data_ptr<int>()),
                                              reinterpret_cast<__half*>(W.data_ptr<at::Half>()), (int)N, (int)K, (int)nw);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return W;
}

std::vector<torch::Tensor> w8a16_sm80_pack(torch::Tensor qweight, torch::Tensor qzeros, torch::Tensor scales,
                                           int64_t group_size, bool v2_zero, int64_t nw) {
  TORCH_CHECK(qweight.is_cuda() && qweight.dtype() == torch::kInt32 && qweight.dim() == 2);
  TORCH_CHECK(group_size == 128, "w8a16_sm80: группа 128");
  const c10::cuda::CUDAGuard guard(qweight.device());
  auto qw = qweight.contiguous(), qz = qzeros.contiguous(), sc = scales.contiguous();
  const int N = (int)qw.size(1), K = (int)qw.size(0) * 4, G = K / 128;
  TORCH_CHECK(K % 64 == 0 && N % (8 * nw) == 0, "w8a16_sm80: K кратно 64, N кратно 8*NW");
  TORCH_CHECK(sc.size(0) == G && sc.size(1) == N && sc.dtype() == torch::kFloat16);
  auto wf = torch::empty({(long)(N / 8) * (K / 64) * 32 * 4}, qw.options());   // int32, = N*K байт
  auto meta = torch::empty({G, N}, qw.options());
  auto st = at::cuda::getCurrentCUDAStream();
  w8s80::pack_kernel<<<1024, 256, 0, st>>>(qw.data_ptr<int>(), reinterpret_cast<uint4*>(wf.data_ptr<int>()), N, K, (int)nw);
  w8s80::meta_kernel<<<512, 256, 0, st>>>(qz.data_ptr<int>(), reinterpret_cast<const __half*>(sc.data_ptr<at::Half>()),
                                          reinterpret_cast<uint32_t*>(meta.data_ptr<int>()), G, N, v2_zero ? 0.f : 1.f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {wf, meta};
}

torch::Tensor w12a16_sm80_dequant(torch::Tensor wf, torch::Tensor scales, int64_t N, int64_t K, int64_t nw,
                                  c10::optional<torch::Tensor> out) {
  const c10::cuda::CUDAGuard guard(wf.device());
  TORCH_CHECK(wf.numel() == N * K * 3 / 8 && scales.numel() == (K / 128) * N, "w12a16_sm80_dequant: формы");
  torch::Tensor W;
  if (out.has_value() && out->numel() >= N * K) W = out->view(-1).narrow(0, 0, N * K).view({N, K});
  else W = torch::empty({N, K}, wf.options().dtype(torch::kFloat16));
  w8s80::dequant12_kernel<<<1024, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(wf.data_ptr<int>()), reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(W.data_ptr<at::Half>()), (int)N, (int)K, (int)nw);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return W;
}

namespace {
struct Scratch { torch::Tensor ws, cnt; std::vector<torch::Tensor> old; };
std::unordered_map<int, Scratch>& scratch() { static std::unordered_map<int, Scratch> m; return m; }

template <int NW, int XR, int STAGES, int WARPS>
void launch(const __half* x, const uint4* wf, const uint32_t* meta, __half* y, int M, int N, int K, int splits,
            float* ws, int* cnt, cudaStream_t st, int ktu, const int* blk, int nblk,
            float* bmax = nullptr, int nvalid = 0) {
  using WS = w8s80::WarpSmem<NW, XR, STAGES>;
  constexpr int MT = (XR + 15) / 16;
  const size_t sm_pipe = sizeof(WS) * WARPS;
  const size_t sm_red = (size_t)WARPS * MT * 16 * NW * 8 * sizeof(float);
  const size_t smem = sm_pipe > sm_red ? sm_pipe : sm_red;
  auto kern = w8s80::w8a16_sm80_kernel<NW, XR, STAGES, WARPS>;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    attr = true;
  }
  const int tps = (ktu + splits - 1) / splits;
  const int sp = (ktu + tps - 1) / tps;
  dim3 grid(blk ? nblk : N / (8 * NW), sp);
  kern<<<grid, WARPS * 32, smem, st>>>(x, wf, meta, y, ws, cnt, M, N, K, tps, ktu, blk, bmax, nvalid);
}
}  // namespace

// cfg: NW (n8-блоков на варп), WARPS, STAGES, splits (0 = авто). M <= 64.
torch::Tensor w8a16_sm80_mm_impl(torch::Tensor x, torch::Tensor wf, torch::Tensor meta, int64_t N, int64_t splits,
                                 int64_t nw, int64_t warps, int64_t stages, int64_t ktu,
                                 c10::optional<torch::Tensor> blk, c10::optional<torch::Tensor> bmax_out,
                                 int64_t nvalid) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat16);
  const c10::cuda::CUDAGuard guard(x.device());
  auto xc = x.contiguous();
  const int K = (int)xc.size(-1), M = (int)(xc.numel() / K);
  TORCH_CHECK(M >= 1 && M <= 64, "w8a16_sm80_mm: M <= 64 (декодная форма)");
  TORCH_CHECK(K % 64 == 0 && N % 8 == 0);
  TORCH_CHECK(wf.numel() == (long)N * K / 4 && meta.numel() == (long)(K / 128) * N);
  if (nw <= 0) nw = 4;
  if (warps <= 0) warps = 4;
  if (stages <= 0) stages = 4;
  TORCH_CHECK(N % (8 * nw) == 0, "N кратно 8*NW");
  const int KT = K / 64;
  if (ktu <= 0 || ktu > KT) ktu = KT;
  const int* bp = nullptr;
  int nblk = 0;
  if (blk.has_value() && blk->numel() > 0) {
    TORCH_CHECK(blk->is_cuda() && blk->dtype() == torch::kInt32 && blk->is_contiguous(), "blk: int32 на карте");
    bp = blk->data_ptr<int>();
    nblk = (int)blk->numel();
  }
  const long NOUT = bp ? (long)nblk * 8 * nw : N;
  float* bmp = bmax_out.has_value() ? bmax_out->data_ptr<float>() : nullptr;
  auto y = bmp ? torch::empty({1}, xc.options()) : torch::empty({M, NOUT}, xc.options());
  if (splits <= 0) {
    // цель -- ~4 блока на SM (96 SM), не дробить плитки варпа короче 4
    const int nblk = (int)(N / (8 * nw));
    const int sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    splits = std::max<int64_t>(1, std::min<int64_t>((4L * sm + nblk - 1) / nblk, ktu / (4 * warps)));
  }
  const int dev = xc.device().index();
  auto& S = scratch()[dev];
  const long need = splits > 1 ? splits * (long)M * NOUT : 1;
  const long nct = N / 8 + 1;
  if (!S.ws.defined() || S.ws.numel() < need) {
    if (S.ws.defined()) S.old.push_back(S.ws);              // адрес мог попасть в граф -- не отпускаем
    S.ws = torch::empty({std::max<long>(need, 1L << 20)}, xc.options().dtype(torch::kFloat32));
  }
  if (!S.cnt.defined() || S.cnt.numel() < nct) {
    if (S.cnt.defined()) S.old.push_back(S.cnt);
    S.cnt = torch::zeros({std::max<long>(nct, 1L << 16)}, xc.options().dtype(torch::kInt32));
  }
  const int XRn = M <= 4 ? 4 : M <= 8 ? 8 : M <= 16 ? 16 : M <= 32 ? 32 : 64;
  auto st = at::cuda::getCurrentCUDAStream();
  auto xp = reinterpret_cast<const __half*>(xc.data_ptr<at::Half>());
  auto wp = reinterpret_cast<const uint4*>(wf.data_ptr<int>());
  auto mp = reinterpret_cast<const uint32_t*>(meta.data_ptr<int>());
  auto yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
  float* wsp = S.ws.data_ptr<float>();
  int* cp = S.cnt.data_ptr<int>();
  const int Ni = (int)N, sp = (int)splits;
#define W8S80_L(NW_, XR_, ST_, WA_) launch<NW_, XR_, ST_, WA_>(xp, wp, mp, yp, M, Ni, K, sp, wsp, cp, st, (int)ktu, bp, nblk, bmp, (int)nvalid)
#define W8S80_MT(NW_, ST_, WA_)                                         \
  do {                                                                 \
    if (XRn == 4) W8S80_L(NW_, 4, ST_, WA_);                           \
    else if (XRn == 8) W8S80_L(NW_, 8, ST_, WA_);                      \
    else if (XRn == 16) W8S80_L(NW_, 16, ST_, WA_);                    \
    else if (XRn == 32) W8S80_L(NW_, 32, ST_, WA_);                    \
    else W8S80_L(NW_, 64, ST_, WA_);                                   \
  } while (0)
  if (nw == 4 && warps == 4 && stages == 4) W8S80_MT(4, 4, 4);
  else if (nw == 4 && warps == 4 && stages == 6) W8S80_MT(4, 6, 4);
  else if (nw == 4 && warps == 8 && stages == 4) W8S80_MT(4, 4, 8);
  else if (nw == 2 && warps == 4 && stages == 4) W8S80_MT(2, 4, 4);
  else if (nw == 2 && warps == 8 && stages == 4) W8S80_MT(2, 4, 8);
  else if (nw == 8 && warps == 4 && stages == 4) W8S80_MT(8, 4, 4);
  else if (nw == 4 && warps == 2 && stages == 8) W8S80_MT(4, 8, 2);
  else if (nw == 2 && warps == 8 && stages == 6) W8S80_MT(2, 6, 8);
  else if (nw == 4 && warps == 8 && stages == 3) W8S80_MT(4, 3, 8);
  // широкий M (XR=32/64): x на 64 строки занимает почти всю стадию -- мало варпов и стадий
  else if (nw == 4 && warps == 2 && stages == 3) W8S80_MT(4, 3, 2);
  else if (nw == 4 && warps == 2 && stages == 4) W8S80_MT(4, 4, 2);
  else if (nw == 2 && warps == 2 && stages == 4) W8S80_MT(2, 4, 2);
  else if (nw == 2 && warps == 4 && stages == 3) W8S80_MT(2, 3, 4);
  else TORCH_CHECK(false, "w8a16_sm80_mm: нет инстанса nw=", nw, " warps=", warps, " stages=", stages);
#undef W8S80_MT
#undef W8S80_L
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}


#include "w8a16_sw_sm80.cuh"

// Перестановка A/B (w8a16_sw_sm80.cuh): M <= 8, раздел K между warps варпами как у w8a16_sm80_mm при splits=1.
torch::Tensor w8a16_sm80_sw(torch::Tensor x, torch::Tensor wf, torch::Tensor meta, int64_t N, int64_t nw,
                            int64_t warps, int64_t jw, int64_t d) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat16);
  const c10::cuda::CUDAGuard guard(x.device());
  auto xc = x.contiguous();
  const int K = (int)xc.size(-1), M = (int)(xc.numel() / K);
  TORCH_CHECK(M >= 1 && M <= 8, "w8a16_sm80_sw: M <= 8");
  TORCH_CHECK(K % 64 == 0 && N % (8 * nw) == 0);
  TORCH_CHECK(wf.numel() == (long)N * K / 4 && meta.numel() == (long)(K / 128) * N);
  auto y = torch::empty({M, N}, xc.options());
  auto st = at::cuda::getCurrentCUDAStream();
  auto xp = reinterpret_cast<const __half*>(xc.data_ptr<at::Half>());
  auto wp = reinterpret_cast<const uint4*>(wf.data_ptr<int>());
  auto mp = reinterpret_cast<const uint32_t*>(meta.data_ptr<int>());
  auto yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
  const int Ni = (int)N;
  const dim3 grid(Ni / (8 * (int)nw));
#define SW_GO(NW_, WA_, JW_, D_) \
  w8s80::w8a16_sw_kernel<NW_, D_, WA_, JW_><<<grid, WA_ * (NW_ / JW_) * 32, 0, st>>>(xp, wp, mp, yp, M, Ni, K)
  if (nw == 4 && warps == 4 && jw == 4 && d == 4) SW_GO(4, 4, 4, 4);
  else if (nw == 4 && warps == 4 && jw == 2 && d == 2) SW_GO(4, 4, 2, 2);
  else if (nw == 4 && warps == 4 && jw == 2 && d == 4) SW_GO(4, 4, 2, 4);
  else if (nw == 4 && warps == 8 && jw == 4 && d == 2) SW_GO(4, 8, 4, 2);
  else if (nw == 4 && warps == 8 && jw == 4 && d == 3) SW_GO(4, 8, 4, 3);
  else if (nw == 4 && warps == 8 && jw == 2 && d == 2) SW_GO(4, 8, 2, 2);
  else if (nw == 2 && warps == 4 && jw == 2 && d == 4) SW_GO(2, 4, 2, 4);
  else if (nw == 2 && warps == 4 && jw == 2 && d == 3) SW_GO(2, 4, 2, 3);
  else TORCH_CHECK(false, "w8a16_sm80_sw: нет инстанса nw=", nw, " warps=", warps, " jw=", jw, " d=", d);
#undef SW_GO
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor w8a16_sm80_mm(torch::Tensor x, torch::Tensor wf, torch::Tensor meta, int64_t N, int64_t splits,
                            int64_t nw, int64_t warps, int64_t stages, int64_t ktu,
                            c10::optional<torch::Tensor> blk) {
  return w8a16_sm80_mm_impl(x, wf, meta, N, splits, nw, warps, stages, ktu, blk, c10::nullopt, N);
}

// Грубая ступень аргмакса: максимумы строк по n-блокам [M, N/(8*nw)] по первым ktu плиткам k.
torch::Tensor w8a16_sm80_bmax(torch::Tensor x, torch::Tensor wf, torch::Tensor meta, int64_t N, int64_t ktu,
                              int64_t nvalid, int64_t splits, int64_t nw, int64_t warps, int64_t stages) {
  const int M = (int)(x.numel() / x.size(-1));
  auto bm = torch::empty({M, (long)(N / (8 * nw))}, x.options().dtype(torch::kFloat32));
  w8a16_sm80_mm_impl(x, wf, meta, N, splits, nw, warps, stages, ktu, c10::nullopt, bm, nvalid);
  return bm;
}

torch::Tensor w8a16_sm80_topblk(torch::Tensor bmax, int64_t B) {
  TORCH_CHECK(bmax.is_cuda() && bmax.dtype() == torch::kFloat32 && bmax.is_contiguous() && bmax.dim() == 2);
  const int M = (int)bmax.size(0), NB = (int)bmax.size(1);
  TORCH_CHECK(NB <= 4096 && B <= NB && B <= 1024, "topblk: NB <= 4096, B <= NB");
  auto out = torch::empty({(long)M * B}, bmax.options().dtype(torch::kInt32));
  w8s80::topblk_kernel<<<M, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(bmax.data_ptr<float>(), NB, (int)B,
                                                                          out.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor w8a16_sm80_candmax(torch::Tensor ys, torch::Tensor blk, int64_t B, int64_t BN, int64_t nvalid,
                                 int64_t vstart) {
  const int M = (int)ys.size(0);
  auto pair = torch::empty({M, 2}, ys.options().dtype(torch::kFloat32));
  w8s80::candmax_kernel<<<M, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(ys.data_ptr<at::Half>()), blk.data_ptr<int>(), (int)B, (int)BN, (int)nvalid,
      (int)vstart, (int)ys.size(1), pair.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return pair;
}

torch::Tensor lin_small_f16(torch::Tensor x, torch::Tensor w) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int K = (int)x.size(-1), M = (int)(x.numel() / K), N = (int)w.size(0);
  TORCH_CHECK(x.dtype() == torch::kFloat16 && w.dtype() == torch::kFloat16 && w.is_contiguous() && w.size(1) == K);
  TORCH_CHECK(M <= 8 && K % 8 == 0 && K <= 8192 && x.stride(-1) == 1, "lin_small_f16: M<=8, K%8==0, K<=8192");
  auto x2 = x.reshape({M, K});
  TORCH_CHECK((x2.stride(0) % 8) == 0 && (reinterpret_cast<uintptr_t>(x2.data_ptr()) & 15) == 0, "lin_small_f16: выравнивание x");
  auto y = torch::empty({M, N}, x.options());
  if (M > 0)
  {
    auto st = at::cuda::getCurrentCUDAStream();
    const __half* xp = reinterpret_cast<const __half*>(x2.data_ptr<at::Half>());
    const __half* wp = reinterpret_cast<const __half*>(w.data_ptr<at::Half>());
    __half* yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
    const long ld = x2.stride(0);
#define LS_GO(MM_, IT_) w8s80::lin_small_kernel<MM_, IT_><<<N, 256, 0, st>>>(xp, wp, yp, M, N, K, ld)
#define LS_IT(MM_) do { if (K <= 2048) LS_GO(MM_, 1); else if (K <= 4096) LS_GO(MM_, 2); \
                        else if (K <= 6144) LS_GO(MM_, 3); else LS_GO(MM_, 4); } while (0)
    if (M <= 1) LS_IT(1); else if (M <= 2) LS_IT(2); else if (M <= 4) LS_IT(4); else LS_IT(8);
#undef LS_IT
#undef LS_GO
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor w12a16_sm80_pack(torch::Tensor qw, int64_t nw) {
  TORCH_CHECK(qw.is_cuda() && qw.dtype() == torch::kInt32 && qw.dim() == 3 && qw.size(1) == 3, "w12: qw [K/8,3,N]");
  const c10::cuda::CUDAGuard guard(qw.device());
  auto q = qw.contiguous();
  const int N = (int)q.size(2), K = (int)q.size(0) * 8;
  TORCH_CHECK(K % 128 == 0 && N % (8 * nw) == 0, "w12a16_sm80: K кратно 128, N кратно 8*NW");
  auto wf = torch::empty({(long)(N / 8) * (K / 128) * 32 * 12}, q.options());   // 48 Б на нить-плитку
  w8s80::pack12_kernel<<<1024, 256, 0, at::cuda::getCurrentCUDAStream()>>>(q.data_ptr<int>(),
      reinterpret_cast<uint4*>(wf.data_ptr<int>()), N, K, (int)nw);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return wf;
}

namespace {
template <int NW, int XR, int STAGES, int WARPS>
void launch12(const __half* x, const uint4* wf, const __half* sc, __half* y, int M, int N, int K, int splits,
              float* ws, int* cnt, cudaStream_t st) {
  using WS = w8s80::WarpSmem12<NW, XR, STAGES>;
  constexpr int MT = (XR + 15) / 16;
  const size_t sm_pipe = sizeof(WS) * WARPS;
  const size_t sm_red = (size_t)WARPS * MT * 16 * NW * 8 * sizeof(float);
  const size_t smem = sm_pipe > sm_red ? sm_pipe : sm_red;
  auto kern = w8s80::w12a16_sm80_kernel<NW, XR, STAGES, WARPS>;
  static bool attr = false;
  if (!attr) { C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); attr = true; }
  const int KT = K / 128, tps = (KT + splits - 1) / splits, sp = (KT + tps - 1) / tps;
  kern<<<dim3(N / (8 * NW), sp), WARPS * 32, smem, st>>>(x, wf, sc, y, ws, cnt, M, N, K, tps);
}
}  // namespace

torch::Tensor w12a16_sm80_mm(torch::Tensor x, torch::Tensor wf, torch::Tensor scales, int64_t N, int64_t splits,
                             int64_t nw, int64_t warps, int64_t stages) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto xc = x.contiguous();
  const int K = (int)xc.size(-1), M = (int)(xc.numel() / K);
  TORCH_CHECK(M >= 1 && M <= 16, "w12a16_sm80_mm: M <= 16");
  TORCH_CHECK(K % 128 == 0 && wf.numel() == (long)N * K * 3 / 8 && scales.dtype() == torch::kFloat16);
  auto y = torch::empty({M, N}, xc.options());
  const int KT = K / 128;
  if (splits <= 0) {
    const int nblk = (int)(N / (8 * nw));
    const int sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    splits = std::max<int64_t>(1, std::min<int64_t>((4L * sm + nblk - 1) / nblk, KT / (2 * warps)));
  }
  auto& S = scratch()[xc.device().index()];
  const long need = splits > 1 ? splits * (long)M * N : 1;
  if (!S.ws.defined() || S.ws.numel() < need) {
    if (S.ws.defined()) S.old.push_back(S.ws);
    S.ws = torch::empty({std::max<long>(need, 1L << 20)}, xc.options().dtype(torch::kFloat32));
  }
  if (!S.cnt.defined() || S.cnt.numel() < N / 8 + 1) {
    if (S.cnt.defined()) S.old.push_back(S.cnt);
    S.cnt = torch::zeros({std::max<long>(N / 8 + 1, 1L << 16)}, xc.options().dtype(torch::kInt32));
  }
  const int XRn = M <= 4 ? 4 : M <= 8 ? 8 : 16;
  auto st = at::cuda::getCurrentCUDAStream();
  auto xp = reinterpret_cast<const __half*>(xc.data_ptr<at::Half>());
  auto wp = reinterpret_cast<const uint4*>(wf.data_ptr<int>());
  auto sp = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  auto yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
  float* wsp = S.ws.data_ptr<float>();
  int* cp = S.cnt.data_ptr<int>();
  const int Ni = (int)N, spl = (int)splits;
#define W12_L(NW_, XR_, ST_, WA_) launch12<NW_, XR_, ST_, WA_>(xp, wp, sp, yp, M, Ni, K, spl, wsp, cp, st)
#define W12_X(NW_, ST_, WA_) do { if (XRn == 4) W12_L(NW_, 4, ST_, WA_); else if (XRn == 8) W12_L(NW_, 8, ST_, WA_); else W12_L(NW_, 16, ST_, WA_); } while (0)
  if (nw == 2 && warps == 4 && stages == 3) W12_X(2, 3, 4);
  else if (nw == 4 && warps == 4 && stages == 3) W12_X(4, 3, 4);
  else if (nw == 2 && warps == 2 && stages == 4) W12_X(2, 4, 2);
  else if (nw == 4 && warps == 2 && stages == 4) W12_X(4, 4, 2);
  else TORCH_CHECK(false, "w12a16_sm80_mm: нет инстанса nw=", nw, " warps=", warps, " stages=", stages);
#undef W12_X
#undef W12_L
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

namespace {
template <int NW, int XR, int STAGES, int WARPS>
void launch12sk(const __half* x, const uint4* wf, const __half* sc, __half* y, int M, int N, int K, int cps,
                Scratch& S, const at::TensorOptions& opt, cudaStream_t st) {
  using WS = w8s80::WarpSmem12<NW, XR, STAGES>;
  const size_t smem = sizeof(WS) * WARPS;
  auto kern = w8s80::w12a16_sm80_sk_kernel<NW, XR, STAGES, WARPS>;
  static int maxb = -1;
  if (maxb < 0) {
    TORCH_CHECK(smem <= at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin,
                "w12a16_sm80_sk: инстанс не помещается (smem ", smem, ")");
    if (smem > 48 * 1024)
      C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, kern, WARPS * 32, smem));
  }
  TORCH_CHECK(maxb > 0, "w12a16_sm80_sk: инстанс не помещается на SM (smem ", smem, ")");
  const int sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int KT = K / 128, NB = N / (8 * NW), U = NB * KT;
  int blocks = sm * std::min(cps, maxb);                    // ровно одна волна: все варпы живут разом
  if (blocks * WARPS > U) blocks = std::max(1, U / WARPS);
  const int W = blocks * WARPS, minlen = U / W, pmax = (KT + minlen - 1) / minlen + 1;
  constexpr int ROWS = ((XR + 15) / 16) * 16;
  const long need = (long)NB * pmax * ROWS * NW * 8;
  if (!S.ws.defined() || S.ws.numel() < need) {
    if (S.ws.defined()) S.old.push_back(S.ws);
    S.ws = torch::empty({std::max<long>(need, 1L << 20)}, opt.dtype(torch::kFloat32));
  }
  if (!S.cnt.defined() || S.cnt.numel() < NB + 1) {
    if (S.cnt.defined()) S.old.push_back(S.cnt);
    S.cnt = torch::zeros({std::max<long>(NB + 1, 1L << 16)}, opt.dtype(torch::kInt32));
  }
  kern<<<blocks, WARPS * 32, smem, st>>>(x, wf, sc, y, S.ws.data_ptr<float>(), S.cnt.data_ptr<int>(),
                                         M, N, K, U, W, pmax);
}
}  // namespace

// Stream-K вариант (см. w12a16_sm80_sk_kernel): cps -- блоков на SM (обрезается занятостью), warps -- варпов
// в блоке (независимых), stages -- глубина cp.async-конвейера варпа.
torch::Tensor w12a16_sm80_sk(torch::Tensor x, torch::Tensor wf, torch::Tensor scales, int64_t N, int64_t nw,
                             int64_t warps, int64_t cps, int64_t stages) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto xc = x.contiguous();
  const int K = (int)xc.size(-1), M = (int)(xc.numel() / K);
  TORCH_CHECK(xc.dtype() == torch::kFloat16 && M >= 1 && M <= 32, "w12a16_sm80_sk: fp16, M <= 32");
  TORCH_CHECK(K % 128 == 0 && N % (8 * nw) == 0 && wf.numel() == (long)N * K * 3 / 8 &&
              scales.dtype() == torch::kFloat16, "w12a16_sm80_sk: формы");
  auto y = torch::empty({M, N}, xc.options());
  auto& S = scratch()[xc.device().index()];
  const int XRn = M <= 4 ? 4 : M <= 8 ? 8 : M <= 16 ? 16 : 32;
  auto st = at::cuda::getCurrentCUDAStream();
  auto xp = reinterpret_cast<const __half*>(xc.data_ptr<at::Half>());
  auto wp = reinterpret_cast<const uint4*>(wf.data_ptr<int>());
  auto sp = reinterpret_cast<const __half*>(scales.data_ptr<at::Half>());
  auto yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
  const int Ni = (int)N, c = (int)cps;
#define SK_L(NW_, XR_, ST_, WA_) launch12sk<NW_, XR_, ST_, WA_>(xp, wp, sp, yp, M, Ni, K, c, S, xc.options(), st)
#define SK_X(NW_, ST_, WA_) do { if (XRn == 4) SK_L(NW_, 4, ST_, WA_); else if (XRn == 8) SK_L(NW_, 8, ST_, WA_); \
    else if (XRn == 16) SK_L(NW_, 16, ST_, WA_); else SK_L(NW_, 32, ST_, WA_); } while (0)
  if (nw == 2 && warps == 4 && stages == 3) SK_X(2, 3, 4);
  else if (nw == 2 && warps == 8 && stages == 3) SK_X(2, 3, 8);
  else if (nw == 2 && warps == 4 && stages == 4) SK_X(2, 4, 4);
  else if (nw == 2 && warps == 2 && stages == 3) SK_X(2, 3, 2);
  else if (nw == 2 && warps == 2 && stages == 2) SK_X(2, 2, 2);
  else TORCH_CHECK(false, "w12a16_sm80_sk: нет инстанса nw=", nw, " warps=", warps, " stages=", stages);
#undef SK_X
#undef SK_L
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}


torch::Tensor w8a16_sm80_gemm_large(torch::Tensor x, torch::Tensor wf, torch::Tensor meta, int64_t N, int64_t nw,
                                    int64_t mi, int64_t splits) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto xc = x.contiguous();
  const int K = (int)xc.size(-1), M = (int)(xc.numel() / K);
  TORCH_CHECK(K % 64 == 0 && N % 128 == 0, "w8a16_sm80_gemm: K кратно 64, N кратно 128");
  TORCH_CHECK(wf.numel() == (long)N * K / 4 && meta.numel() == (long)(K / 128) * N);
  auto y = torch::empty({M, N}, xc.options());
  if (mi <= 0) mi = M <= 32 ? 1 : M <= 64 ? 2 : 4;
  TORCH_CHECK(mi == 1 || mi == 2 || mi == 4, "w8a16_sm80_gemm: mi 1/2/4");
  auto st = at::cuda::getCurrentCUDAStream();
  const int bm = 32 * (int)mi;
  const int KT = K / 64;
  if (splits < 1) splits = 1;
  const int tps = (KT + (int)splits - 1) / (int)splits, sp = (KT + tps - 1) / tps;
  dim3 grid(N / 128, (M + bm - 1) / bm, sp);
  float* wsp = nullptr; int* cp = nullptr;
  if (sp > 1) {
    auto& S = scratch()[xc.device().index()];
    const long need = (long)sp * M * N;
    if (!S.ws.defined() || S.ws.numel() < need) {
      if (S.ws.defined()) S.old.push_back(S.ws);
      S.ws = torch::empty({std::max<long>(need, 1L << 20)}, xc.options().dtype(torch::kFloat32));
    }
    const long tiles = (long)grid.x * grid.y;
    if (!S.cnt.defined() || S.cnt.numel() < tiles) {
      if (S.cnt.defined()) S.old.push_back(S.cnt);
      S.cnt = torch::zeros({std::max<long>(tiles, 1L << 16)}, xc.options().dtype(torch::kInt32));
    }
    wsp = S.ws.data_ptr<float>(); cp = S.cnt.data_ptr<int>();
  }
#define GEMM_K(NW_, MI_) do { const size_t smem = sizeof(w8s80::GemmSmem<MI_>); \
    auto k = w8s80::w8a16_sm80_gemm<NW_, MI_>; static bool a = false; \
    if (!a) { C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); a = true; } \
    k<<<grid, 256, smem, st>>>(reinterpret_cast<const __half*>(xc.data_ptr<at::Half>()), \
      reinterpret_cast<const uint4*>(wf.data_ptr<int>()), reinterpret_cast<const uint32_t*>(meta.data_ptr<int>()), \
      reinterpret_cast<__half*>(y.data_ptr<at::Half>()), wsp, cp, M, (int)N, K, tps); } while (0)
#define GEMM_NW(NW_) do { if (mi == 1) GEMM_K(NW_, 1); else if (mi == 2) GEMM_K(NW_, 2); else GEMM_K(NW_, 4); } while (0)
  if (nw == 2) GEMM_NW(2); else if (nw == 4) GEMM_NW(4); else if (nw == 8) GEMM_NW(8);
  else TORCH_CHECK(false, "w8a16_sm80_gemm: nw 2/4/8");
#undef GEMM_NW
#undef GEMM_K
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}


// W8A16 v2: nj -- плиток n8 на варп (8 -> блок 128x256, 4 -> 128x128), st -- стадий, splits -- расщепление k.
torch::Tensor w8a16_sm80_gemm2(torch::Tensor x, torch::Tensor wf, torch::Tensor meta, int64_t N, int64_t nw,
                               int64_t nj, int64_t stages, int64_t splits, bool pin) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto xc = x.contiguous();
  TORCH_CHECK(xc.dtype() == torch::kFloat16, "w8a16_sm80_gemm2: fp16");
  const int K = (int)xc.size(-1), M = (int)(xc.numel() / K);
  const int BN = 32 * (int)nj;
  TORCH_CHECK(K % 128 == 0 && N % BN == 0 && N % (8 * nw) == 0, "w8a16_sm80_gemm2: K кратно 128, N кратно ", BN);
  auto y = torch::empty({M, N}, xc.options());
  const int KT = K / 64;
  if (splits < 1) splits = 1;
  const int tps = (KT + (int)splits - 1) / (int)splits, sp = (KT + tps - 1) / tps;
  dim3 grid(N / BN, (M + 127) / 128, sp);
  float* wsp = nullptr; int* cp = nullptr;
  if (sp > 1) {
    auto& S = scratch()[xc.device().index()];
    const long need = (long)sp * M * N;
    if (!S.ws.defined() || S.ws.numel() < need) {
      if (S.ws.defined()) S.old.push_back(S.ws);
      S.ws = torch::empty({std::max<long>(need, 1L << 20)}, xc.options().dtype(torch::kFloat32));
    }
    const long tiles = (long)grid.x * grid.y;
    if (!S.cnt.defined() || S.cnt.numel() < tiles) {
      if (S.cnt.defined()) S.old.push_back(S.cnt);
      S.cnt = torch::zeros({std::max<long>(tiles, 1L << 16)}, xc.options().dtype(torch::kInt32));
    }
    wsp = S.ws.data_ptr<float>(); cp = S.cnt.data_ptr<int>();
  }
  auto st = at::cuda::getCurrentCUDAStream();
#define G2_K(NJ_, ST_, PIN_) do { const size_t smem = sizeof(w8s80::Gemm2Smem<NJ_, ST_>); \
    auto k = w8s80::w8a16_sm80_gemm2<NJ_, ST_, PIN_>; static bool a = false; \
    if (!a) { C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); a = true; } \
    k<<<grid, 256, smem, st>>>(reinterpret_cast<const __half*>(xc.data_ptr<at::Half>()), \
      reinterpret_cast<const uint4*>(wf.data_ptr<int>()), reinterpret_cast<const uint32_t*>(meta.data_ptr<int>()), \
      reinterpret_cast<__half*>(y.data_ptr<at::Half>()), wsp, cp, M, (int)N, K, tps, nwl); } while (0)
#define G2_P(NJ_, ST_) do { if (pin) G2_K(NJ_, ST_, true); else G2_K(NJ_, ST_, false); } while (0)
#define G2_ST(NJ_) do { if (stages == 3) G2_P(NJ_, 3); else G2_P(NJ_, 4); } while (0)
  TORCH_CHECK(nw == 2 || nw == 4 || nw == 8, "w8a16_sm80_gemm2: nw 2/4/8");
  const int nwl = nw == 2 ? 1 : nw == 4 ? 2 : 3;
  if (nj == 8) G2_ST(8); else G2_ST(4);
#undef G2_ST
#undef G2_P
#undef G2_K
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w8a16_sm80_pack", &w8a16_sm80_pack, "GPTQ int8 g128 -> фрагментный порядок sm_80 (wf, meta)",
        py::arg("qweight"), py::arg("qzeros"), py::arg("scales"), py::arg("group_size"), py::arg("v2_zero"),
        py::arg("nw") = 4);
  m.def("w8a16_sm80_dequant", &w8a16_sm80_dequant, "фрагментный порядок -> W [N,K] fp16",
        py::arg("wf"), py::arg("meta"), py::arg("N"), py::arg("K"), py::arg("nw") = 4, py::arg("out") = py::none());
  m.def("w12a16_sm80_dequant", &w12a16_sm80_dequant, "плоскости W12 -> W [N,K] fp16",
        py::arg("wf"), py::arg("scales"), py::arg("N"), py::arg("K"), py::arg("nw") = 2, py::arg("out") = py::none());
  m.def("w12a16_sm80_pack", &w12a16_sm80_pack, "W12 [K/8,3,N] -> плоскости sm_80", py::arg("qw"), py::arg("nw") = 2);
  m.def("w12a16_sm80_mm", &w12a16_sm80_mm, "W12A16 sm_80, M<=16", py::arg("x"), py::arg("wf"), py::arg("scales"),
        py::arg("N"), py::arg("splits") = 0, py::arg("nw") = 2, py::arg("warps") = 4, py::arg("stages") = 3);
  m.def("w12a16_sm80_sk", &w12a16_sm80_sk, "W12A16 sm_80, M<=32, stream-K по варпам (одна волна)",
        py::arg("x"), py::arg("wf"), py::arg("scales"), py::arg("N"), py::arg("nw") = 2, py::arg("warps") = 4,
        py::arg("cps") = 3, py::arg("stages") = 3);
  m.def("w8a16_sm80_gemm2", &w8a16_sm80_gemm2, "W8A16 sm_80 v2: блок 128x(32*nj), варп 64x(8*nj), конвейер фрагментов",
        py::arg("x"), py::arg("wf"), py::arg("meta"), py::arg("N"), py::arg("nw") = 4, py::arg("nj") = 8,
        py::arg("stages") = 3, py::arg("splits") = 1, py::arg("pin") = false);
  m.def("w8a16_sm80_gemm_large", &w8a16_sm80_gemm_large, "W8A16 sm_80 большие M: прямой GEMM 128x128x64",
        py::arg("x"), py::arg("wf"), py::arg("meta"), py::arg("N"), py::arg("nw") = 4, py::arg("mi") = 0, py::arg("splits") = 1);
  m.def("w8a16_sm80_bmax", &w8a16_sm80_bmax, "аргмакс словаря, грубая ступень: максимумы по n-блокам",
        py::arg("x"), py::arg("wf"), py::arg("meta"), py::arg("N"), py::arg("ktu"), py::arg("nvalid"),
        py::arg("splits") = 1, py::arg("nw") = 4, py::arg("warps") = 8, py::arg("stages") = 3);
  m.def("w8a16_sm80_topblk", &w8a16_sm80_topblk, "B лучших n-блоков по грубым максимумам", py::arg("bmax"), py::arg("B"));
  m.def("w8a16_sm80_candmax", &w8a16_sm80_candmax, "пара (значение, номер) по точным значениям кандидатов",
        py::arg("ys"), py::arg("blk"), py::arg("B"), py::arg("BN"), py::arg("nvalid"), py::arg("vstart"));
  m.def("lin_small_f16", &lin_small_f16, "малая плотная fp16-проекция при M<=8 (варп на столбец)");
  m.def("w8a16_sm80_sw", &w8a16_sm80_sw, "W8A16 sm_80 декод M<=8: перестановка A/B, вес прямо в регистры",
        py::arg("x"), py::arg("wf"), py::arg("meta"), py::arg("N"), py::arg("nw"), py::arg("warps"),
        py::arg("jw"), py::arg("d"));
  m.def("w8a16_sm80_mm", &w8a16_sm80_mm, "W8A16 sm_80: cp.async-конвейер варпа + m16n8k16, M<=64",
        py::arg("x"), py::arg("wf"), py::arg("meta"), py::arg("N"), py::arg("splits") = 0,
        py::arg("nw") = 4, py::arg("warps") = 4, py::arg("stages") = 4, py::arg("ktu") = 0,
        py::arg("blk") = py::none());
}
