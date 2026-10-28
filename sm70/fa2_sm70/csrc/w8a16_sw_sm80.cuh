// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// W8A16 ДЕКОД, ПЕРЕСТАНОВКА A/B (25.10): вставляется в w8a16_sm80.cu (пространство w8s80, его deq4/mma16816).
//
// Нынешнее w8a16_sm80_kernel кладёт x в A (плитка m16 при M<=8 -- половина строк нули), вес в B (n8) и тянет вес
// глобальная -> cp.async -> разделяемая -> LDS -> регистры. Профиль ncu (gate_up, M=5): ~65 инструкций на 512 Б
// веса (IMAD-перекладки 17.7, BRA.CONV/ожидания, MIO), DRAM 82 % при ЛЮБОЙ занятости (8..24 варпа на SM), и
// тёплый L2 его не ускоряет -- ядро упирается в собственный поток инструкций.
//
// Здесь вес -- операнд A (16 признаков = пара n8-полос упаковки x k16), x -- операнд B (8 токенов): MMA вдвое
// меньше, нулевых строк нет, вес идёт ПРЯМО В РЕГИСТРЫ (LDG.128 мимо L1, D плиток k64 в полёте); разделяемая --
// только для свода варпов в конце. Упаковка та же ([nb][kt][j][lane], 16 Б на нить): у нити (g,t) полосы j --
// признак g, k 8t..8t+7 каждой k32-половины плитки. Слоты k у MMA {2t,2t+1,2t+8,2t+9} <- k {8t..8t+3} (первое MMA
// половины) и {8t+4..8t+7} (второе) -- ТЕ ЖЕ наборы k на каждое MMA, что у нынешнего ядра, тот же порядок групп,
// масштаба и свода варпов: при том же разделе K между варпами (WA) выход ПОБИТНО тот же (стенд bench/sa/w8sw_test:
// 0 различий на всех формах ранга при M=1/5/8).
//
// JW полос на варп (делит NW упаковки): блок = WA (раздел K) x NW/JW (раздел полос упаковки) варпов. M <= 8.
#pragma once

namespace w8s80 {

__device__ __forceinline__ uint4 sw_ldg_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

template <int NW, int D, int WA, int JW>
__global__ __launch_bounds__(WA * (NW / JW) * 32) void w8a16_sw_kernel(
    const __half* __restrict__ x, const uint4* __restrict__ wf, const uint32_t* __restrict__ meta,
    __half* __restrict__ y, int M, int N, int K) {
  static_assert(JW % 2 == 0 && NW % JW == 0, "JW чётное и делит NW: пара n8-полос = 16 строк A");
  constexpr int NP = JW / 2, NS = NW / JW;
  const int lane = threadIdx.x & 31, warp = (threadIdx.x >> 5) / NS, ws = (threadIdx.x >> 5) % NS, jo = ws * JW;
  const int g = lane >> 2, t = lane & 3;
  const int KT = K >> 6;
  const int bx = blockIdx.x, nb0 = bx * NW;
  // раздел K между варпами -- как у w8a16_sm80_kernel при splits=1 (иначе другой порядок суммы)
  const int per_w = (KT + WA - 1) / WA;
  const int wk0 = min(KT, warp * per_w), wk1 = min(KT, wk0 + per_w);
  const int ntile = wk1 - wk0;
  const uint4* wp = wf + (((long)bx * KT + wk0) * NW + jo) * 32 + lane;
  const bool xl = g < M;
  const uint4* xp = reinterpret_cast<const uint4*>(x + (long)(xl ? g : 0) * K + wk0 * 64 + 8 * t);
  const uint32_t* mp = meta + (long)((nb0 + jo) * 8) + g;   // + gi*N + j*8

  uint4 wb[D][JW], xb[D][2];
  uint32_t mb[D][JW];
  auto load = [&](int u, int i) {
#pragma unroll
    for (int j = 0; j < JW; ++j) wb[u][j] = sw_ldg_stream(wp + ((long)i * NW + j) * 32);
    if (xl) { xb[u][0] = __ldg(xp + i * 8); xb[u][1] = __ldg(xp + i * 8 + 4); }
    else { xb[u][0] = make_uint4(0, 0, 0, 0); xb[u][1] = make_uint4(0, 0, 0, 0); }
    const int gi = (wk0 + i) >> 1;
#pragma unroll
    for (int j = 0; j < JW; ++j) mb[u][j] = __ldg(mp + (long)gi * N + j * 8);
  };
  float acc[NP][4], accg[NP][4];
#pragma unroll
  for (int p = 0; p < NP; ++p)
#pragma unroll
    for (int r = 0; r < 4; ++r) { acc[p][r] = 0.f; accg[p][r] = 0.f; }

#pragma unroll
  for (int u = 0; u < D; ++u) if (u < ntile) load(u, u);
  for (int i0 = 0; i0 < ntile; i0 += D) {
#pragma unroll
    for (int u = 0; u < D; ++u) {
      const int i = i0 + u;
      if (i < ntile) {
        const int kt = wk0 + i;
        uint32_t zz[JW];
#pragma unroll
        for (int j = 0; j < JW; ++j) zz[j] = (mb[u][j] & 0xFFFF0000u) | (mb[u][j] >> 16);
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          const uint4 xv = xb[u][c];
#pragma unroll
          for (int p = 0; p < NP; ++p) {
            const uint4 v0 = wb[u][2 * p], v1 = wb[u][2 * p + 1];
            uint32_t a[4];
            deq4(c ? v0.z : v0.x, zz[2 * p], a[0], a[2]);
            deq4(c ? v1.z : v1.x, zz[2 * p + 1], a[1], a[3]);
            mma16816(accg[p], a, xv.x, xv.y);
            deq4(c ? v0.w : v0.y, zz[2 * p], a[0], a[2]);
            deq4(c ? v1.w : v1.y, zz[2 * p + 1], a[1], a[3]);
            mma16816(accg[p], a, xv.z, xv.w);
          }
        }
        if ((kt & 1) || i == ntile - 1) {                 // конец группы 128 (или полосы варпа)
#pragma unroll
          for (int p = 0; p < NP; ++p) {
            const float s0 = __half2float(__ushort_as_half((unsigned short)(mb[u][2 * p] & 0xFFFFu)));
            const float s1 = __half2float(__ushort_as_half((unsigned short)(mb[u][2 * p + 1] & 0xFFFFu)));
            acc[p][0] += s0 * accg[p][0]; acc[p][1] += s0 * accg[p][1];
            acc[p][2] += s1 * accg[p][2]; acc[p][3] += s1 * accg[p][3];
            accg[p][0] = accg[p][1] = accg[p][2] = accg[p][3] = 0.f;
          }
        }
        if (i + D < ntile) load(u, i + D);
      }
    }
  }
  // свод варпов блока в фиксированном порядке (как у w8a16_sm80_kernel): red[warp][ток][столбец полосы]
  constexpr int BN = NW * 8;
  __shared__ float red[WA][8][BN];
#pragma unroll
  for (int p = 0; p < NP; ++p) {
    red[warp][2 * t][(jo + 2 * p) * 8 + g] = acc[p][0];
    red[warp][2 * t + 1][(jo + 2 * p) * 8 + g] = acc[p][1];
    red[warp][2 * t][(jo + 2 * p + 1) * 8 + g] = acc[p][2];
    red[warp][2 * t + 1][(jo + 2 * p + 1) * 8 + g] = acc[p][3];
  }
  __syncthreads();
  const int mrows = min(M, 8);
  for (int e = threadIdx.x; e < mrows * BN; e += WA * NS * 32) {
    const int row = e / BN, col = e - row * BN;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < WA; ++w) v += red[w][row][col];
    y[(long)row * N + nb0 * 8 + col] = __float2half(v);
  }
}

}  // namespace w8s80
