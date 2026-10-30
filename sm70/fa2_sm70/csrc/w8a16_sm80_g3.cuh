// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// [GEMM 30.10] W8A16 v3 для средних и больших M: A В ПОРЯДКЕ ФРАГМЕНТОВ.
//
// ПОЧЕМУ. SASS главного цикла v2 (w8a16_sm80_gemm2<8,3>): 846 инструкций на 128 HMMA, из них 299 MOV/IMAD.MOV --
// фрагмент A собирается из двух LDS.128 строк g и g+8 ({v0.x, v1.x, v0.y, v1.y}), а HMMA берёт A только из четырёх
// ПОДРЯД идущих регистров; плюс 255 регистров и спиллы. У прямого ядра (w8a16_sm80_gemm) то же самое. Оба упирались
// в ~100 TF при потолке ~2 инструкций выдачи на HMMA.
//
// ЧТО ЗДЕСЬ. Отдельный проход (pack_a_kernel) раскладывает x в порядке фрагментов: для m16-плитки mt, k64-плитки kt,
// шага q = 2c+s и нити (g,t) -- uint4 {x[r0][k..k+1], x[r1][k..k+1], x[r0][k+2..k+3], x[r1][k+2..k+3]}, r0 = 16mt+g,
// r1 = r0+8, k = 64kt+32c+8t+4s. Это ровно регистры A инструкции mma.m16n8k16 при ТОЙ ЖЕ перестановке k, что у веса
// (нить владеет k = [8t, 8t+8) в 32-куске, см. шапку w8a16_sm80.cu). В ядре одна LDS.128 = готовый фрагмент, без
// MOV; копия глобальная -> разделяемая -- сплошные 2 КБ на плитку, банковых конфликтов нет (полосы подряд по 16 Б).
// Раскладка xa[kt][mt][q][lane]: строки кратные 16 -- срез без перекладки (куски конвейера блока).
//
// Плитка варпа 64x64 (4 m16 x 8 n8): на HMMA ~1.5 инструкции распаковки веса (2 PRMT + 2 HSUB2 + 2 HMUL2 на слово)
// и 1/8 LDS.128 A. Блок WM x WN варпов (BM = 64*WM, BN = 64*WN), ST стадий cp.async, плитки M идут первыми.
//
// АРИФМЕТИКА ПОБИТНО ТА ЖЕ, ЧТО У ПРЯМОГО ЯДРА: те же слова веса, (1024+q)-(1024+z) точно, затем * s в fp16 (HMUL2), шаги
// k16 в том же порядке на каждый аккумулятор, то же расщепление k (tps) и тот же детерминированный свод слотов. Размер
// плитки блока на элемент выхода не влияет -- при splits=1 выход не зависит ни от конфигурации, ни от M (инвариантен
// к пачке).
#pragma once

namespace w8s80 {

__device__ __forceinline__ void cp_async16_cg_z(void* s, const void* g, bool live) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(smem_u32(s)), "l"(g), "r"(live ? 16 : 0));
}

// x [M, K] (шаг строк ldx, кратен 8) -> xa[kt][mt][q][lane]; MTA -- m16-плиток в xa (строки за M -- нули).
__global__ void pack_a_kernel(const __half* __restrict__ x, long ldx, int M, int K, uint4* __restrict__ xa, int MTA) {
  const int KT = K >> 6;
  const long total = (long)KT * MTA * 64;                 // на плитку (kt, mt): 2 куска c x 32 нити
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int lane = (int)(i & 31), c = (int)((i >> 5) & 1);
    const long tile = i >> 6;                              // = kt * MTA + mt
    const int mt = (int)(tile % MTA), kt = (int)(tile / MTA);
    const int g = lane >> 2, t = lane & 3;
    const int r0 = mt * 16 + g, r1 = r0 + 8, k = kt * 64 + c * 32 + t * 8;
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    const uint4 v0 = r0 < M ? __ldg(reinterpret_cast<const uint4*>(x + (long)r0 * ldx + k)) : z;
    const uint4 v1 = r1 < M ? __ldg(reinterpret_cast<const uint4*>(x + (long)r1 * ldx + k)) : z;
    uint4* d = xa + (tile * 4 + 2 * c) * 32 + lane;
    d[0] = make_uint4(v0.x, v1.x, v0.y, v1.y);             // шаг s = 0
    d[32] = make_uint4(v0.z, v1.z, v0.w, v1.w);            // шаг s = 1
  }
}

template <int WM, int WN, int ST>
struct G3Smem {
  uint4 a[ST][WM * 4][4][32];              // m16-плитки блока: 4 шага x 32 нити по 16 Б = 2 КБ
  uint4 b[ST][WN * 8][32];                 // n8-плитки веса: 512 Б
  uint32_t meta[ST][WN * 64];              // (s | 1024+z) на столбец, группа плитки
};

// xa -- уже со сдвигом на первую m16-плитку вызова; lda -- шаг k64-плиток в xa (uint4) = MTA*128.
template <int WM, int WN, int ST, int MINB>
__global__ __launch_bounds__(WM * WN * 32, MINB) void w8a16_sm80_gemm3(
    const uint4* __restrict__ xa, long lda, const uint4* __restrict__ wf, const uint32_t* __restrict__ meta,
    __half* __restrict__ y, long ldy, float* __restrict__ ws, int* __restrict__ cnt,
    int M, int N, int K, int tps, int nwl) {
  constexpr int T = WM * WN * 32, MTB = WM * 4, NTB = WN * 8, BM = 64 * WM, BN = 64 * WN;
  static_assert(T % 128 == 0, "w8a16_sm80_gemm3: не меньше 4 варпов");
  constexpr int AC = MTB * 128 / T;        // копий A на нить за плитку k
  constexpr int BC = NTB * 32 / T;         // копий B на нить
  extern __shared__ __align__(16) unsigned char smem_raw[];
  G3Smem<WM, WN, ST>& sm = *reinterpret_cast<G3Smem<WM, WN, ST>*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int wm = warp / WN, wn = warp - wm * WN;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
  const int KT = K >> 6, NW = 1 << nwl;
  const int mtv = (M + 15) >> 4;
  // A: копия i -- m16-плитка блока (tid>>7) + i*(T>>7), полоса tid&127 (сдвиг в xa -- константа)
  const int at0 = tid >> 7;
  const uint4* pa = xa + (long)((m0 >> 4) + at0) * 128 + (tid & 127);
  bool alive[AC];
#pragma unroll
  for (int i = 0; i < AC; ++i) alive[i] = (m0 >> 4) + at0 + i * (T >> 7) < mtv;
  // B: копия i -- n8-плитка блока warp + i*(T>>5), полоса lane
  const uint4* pb[BC];
#pragma unroll
  for (int i = 0; i < BC; ++i) {
    const int nb = (n0 >> 3) + warp + i * (T >> 5);
    pb[i] = wf + ((long)(nb >> nwl) * KT * NW + (nb & (NW - 1))) * 32 + lane;
  }
  auto issue = [&](int kt, int st) {
    const uint4* as = pa + (long)kt * lda;
#pragma unroll
    for (int i = 0; i < AC; ++i)
      cp_async16_cg_z(&sm.a[st][at0 + i * (T >> 7)][0][0] + (tid & 127), as + i * (T >> 7) * 128, alive[i]);
    const long ko = (long)kt * NW * 32;
#pragma unroll
    for (int i = 0; i < BC; ++i) cp_async16_cg(&sm.b[st][warp + i * (T >> 5)][lane], pb[i] + ko);
    if (tid < BN / 4) cp_async16_cg(&sm.meta[st][tid * 4], meta + (long)(kt >> 1) * N + n0 + tid * 4);
  };
  float acc[4][8][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;
  const int k0 = blockIdx.z * tps, nt = min(KT, k0 + tps) - k0;
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) { if (s < nt) issue(k0 + s, s); cp_commit(); }
  for (int it = 0; it < nt; ++it) {
    const int st = it % ST;
    cp_wait<ST - 2>();
    __syncthreads();                                     // стадия it видна всем; стадия it-1 всеми дочитана
    if (it + ST - 1 < nt) issue(k0 + it + ST - 1, (it + ST - 1) % ST);
    cp_commit();
    uint32_t zz[8], s2[8];
    uint4 wv[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint32_t mm = sm.meta[st][wn * 64 + j * 8 + g];
      zz[j] = __byte_perm(mm, 0u, 0x3232u);              // (1024+z, 1024+z)
      s2[j] = __byte_perm(mm, 0u, 0x1010u);              // (s, s)
      wv[j] = sm.b[st][wn * 8 + j][lane];
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {                        // шаги k16: (c,s) = (0,0),(0,1),(1,0),(1,1)
      uint4 av[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) av[i] = sm.a[st][wm * 4 + i][q][lane];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const uint32_t w = q == 0 ? wv[j].x : q == 1 ? wv[j].y : q == 2 ? wv[j].z : wv[j].w;
        uint32_t d0, d1;
        deq4(w, zz[j], d0, d1);
        __half2 h0 = __hmul2(*reinterpret_cast<__half2*>(&d0), *reinterpret_cast<const __half2*>(&s2[j]));
        __half2 h1 = __hmul2(*reinterpret_cast<__half2*>(&d1), *reinterpret_cast<const __half2*>(&s2[j]));
        const uint32_t b0 = *reinterpret_cast<uint32_t*>(&h0), b1 = *reinterpret_cast<uint32_t*>(&h1);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint32_t a[4] = {av[i].x, av[i].y, av[i].z, av[i].w};
          mma16816(acc[i][j], a, b0, b1);
        }
      }
    }
  }
  cp_wait<0>();
  const bool split = gridDim.z > 1;
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int col = n0 + wn * 64 + j * 8 + 2 * t;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int row = m0 + wm * 64 + i * 16 + g + 8 * hh;
        if (row >= M) continue;
        if (!split)
          *reinterpret_cast<__half2*>(y + (long)row * ldy + col) = __floats2half2_rn(acc[i][j][2 * hh], acc[i][j][2 * hh + 1]);
        else
          *reinterpret_cast<float2*>(ws + ((long)blockIdx.z * M + row) * N + col) = make_float2(acc[i][j][2 * hh], acc[i][j][2 * hh + 1]);
      }
    }
  if (!split) return;
  // детерминированный свод (как у прямого ядра): последний пришедший к плитке блок складывает слоты по порядку z
  __threadfence();
  __syncthreads();
  __shared__ int last;
  const int tile = blockIdx.y * gridDim.x + blockIdx.x;
  if (tid == 0) last = (atomicAdd(&cnt[tile], 1) == (int)gridDim.z - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  const int rows = min(BM, M - m0);
  for (int e = tid; e < rows * (BN / 2); e += T) {
    const int r = e / (BN / 2), c2 = (e - r * (BN / 2)) * 2;
    float2 v = make_float2(0.f, 0.f);
    for (int z = 0; z < (int)gridDim.z; ++z) {
      const float2 p = __ldcg(reinterpret_cast<const float2*>(ws + ((long)z * M + m0 + r) * N + n0 + c2));
      v.x += p.x; v.y += p.y;
    }
    *reinterpret_cast<__half2*>(y + (long)(m0 + r) * ldy + n0 + c2) = __floats2half2_rn(v.x, v.y);
  }
  if (tid == 0) cnt[tile] = 0;                           // самоочистка (повтор графа)
}

}  // namespace w8s80
