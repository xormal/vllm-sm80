// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// ДЕКОДНОЕ ВНИМАНИЕ ДЛЯ AMPERE (sm_80): int8 KV с масштабом на (токен, голову), d=256, GQA + строки MTP.
// Замена "химеры" decode_hmma.cuh на A100. Формат входов/выходов -- ТОТ ЖЕ (A/B частичных срезов +
// hmma_svod_kernel), поэтому вызов flash_decode_hmma_paged остаётся прежним.
//
// ПОЧЕМУ ХИМЕРА НЕ ГОДИТСЯ НА A100 (замер на A100, Sk=262144, qGrp=4, GF=6): 980 мкс = 278 ГБ/с, 21 %
// крыши HBM. Её схема -- вольтовская: плитка вперёд в РЕГИСТРАХ (cp.async у Volta нет), три-четыре
// барьера на плитку, один блок на SM. Связывает латентность, а не полоса.
//
// СХЕМА. Блок = (запрос, KV-голова, срез ключей), 4 варпа С РОЛЯМИ (вольтовский приём volta_fwd_ws,
// но на родных механизмах Ampere):
//   * варпы 2-3 -- ВОЗЧИКИ: cp.async байтов K/V плитки в кольцо STAGES стадий (глубина конвейера больше
//     не стоит регистров), затем деквант int8 -> fp16 ТОЧНО (XOR 0x80, PRMT в мантиссу 1024+u, HSUB2 1152)
//     в двойной fp16-буфер плитки. Нить разворачивает РОВНО те байты, что сама везла, поэтому между
//     возчиками барьера нет -- достаточно cp.async.wait_group;
//   * варпы 0-1 -- СЧЁТЧИКИ (раскладка FA2): варп = 16 строк x весь D. Q в регистрах (фрагменты A один
//     раз на блок), S = Q K^T через ldmatrix (K fp16 [ключ][d] -- k-мажорный B), онлайн-софтмакс в
//     регистрах (строка -- четвёрка полос, свод шаффлами 1 и 2), P НЕ покидает регистры: аккумулятор
//     S 16x16 и фрагмент A следующего умножения -- одна и та же раскладка. P*V через ldmatrix.trans
//     (V fp16 [ключ][d] -- n-мажорный B). Масштаб K -- на столбец S, масштаб V -- в веса P (как у химеры);
//   * рандеву -- именованные барьеры: ПОЛОН[b] (возчики bar.arrive, счётчики bar.sync) и ПУСТ[b]
//     (наоборот). bar.arrive не ждёт: возчик уходит развозить следующую плитку сразу.
// Строки: qGrp*GF <= 32 (два счётчика) -- при qGrp*GF <= 16 работает один, второй простаивает.
#pragma once
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cstdint>
// ФАЛЬСИФИКАТОРЫ СНЯТИЕМ ФАЗЫ (ответ заведомо неверен, читается только время):
//   1 -- возчики не разворачивают байты (только cp.async и рандеву);  2 -- счётчики не считают;
//   3 -- оба: чистая скорость потока cp.async через кольцо.
#ifndef FA2SM80_DEC_DIAG
#define FA2SM80_DEC_DIAG 0
#endif

namespace fa2sm80_dec {

constexpr int D = 256;
constexpr int KT = 32;                 // ключей в плитке
#ifndef FA2SM80_DEC_STAGES
#define FA2SM80_DEC_STAGES 4
#endif
constexpr int STAGES = FA2SM80_DEC_STAGES;   // стадий байтового кольца
#ifndef FA2SM80_DEC_NPW
#define FA2SM80_DEC_NPW 4              // варпов-возчиков: 2 не успевали разворачивать (фальсификатор DIAG=1: -21 %)
#endif
constexpr int NPW = FA2SM80_DEC_NPW;
constexpr int NT = 128 + 32 * NPW;     // 2 пары счётчиков + возчики
constexpr int NBAR = 32 * NPW + 64;    // участников именованного барьера: возчики + ОДНА пара счётчиков
constexpr int LDH = D + 8;             // шаг строки fp16-буфера (528 Б: строки на разных банках, кратно 16 Б)
// Шаг строки байтового кольца: 256 Б сажали ВСЕ нити варпа на одну четвёрку банков (кусок c у каждого
// ключа -- тот же банк, 8-кратная очередь на LDS.128). 272 Б: восемь ключей фазы четверти варпа -- на
// восьми разных четвёрках банков, ровно 4 вайвфронта на LDS.128 варпа (минимум).
constexpr int LDB8 = D + 16;
constexpr int BAR_FULL = 1, BAR_EMPTY = 3;   // id 1,2 и 3,4 (0 занят __syncthreads)

struct Smem {
  int8_t kb[STAGES][KT][LDB8];         // байтовое кольцо (пишет cp.async)
  int8_t vb[STAGES][KT][LDB8];
  float ksb[STAGES][KT], vsb[STAGES][KT];
  __half kh[2][KT][LDH];               // fp16-буферы плитки (пишут возчики, читают счётчики)
  __half vh[2][KT][LDH];
  float ks[2][KT], vs[2][KT];          // масштабы плитки -- в том же двойном буфере, что и fp16
  __half q[32][LDH];                   // Q блока (строки за пределами -- нули)
};

__device__ __forceinline__ uint32_t s_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(void* s, const void* g, bool live) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s_u32(s)), "l"(g), "r"(live ? 16 : 0));
}
__device__ __forceinline__ void cp4(void* s, const void* g, bool live) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(s_u32(s)), "l"(g), "r"(live ? 4 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void bar_sync(int id) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "n"(NBAR) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "n"(NBAR) : "memory"); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s_u32(p)));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s_u32(p)));
}
__device__ __forceinline__ void mma(float* d, const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// 4 знаковых байта -> 4 fp16 ТОЧНО: (1024 + (v^0x80)) - 1152 = v.
__device__ __forceinline__ void deq4(uint32_t w, uint32_t& lo, uint32_t& hi) {
  const uint32_t t = w ^ 0x80808080u;
  uint32_t a = __byte_perm(t, 0x64646464u, 0x5140u), b = __byte_perm(t, 0x64646464u, 0x5342u);
  const uint32_t m = 0x64806480u;
  __half2 ha = __hsub2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<const __half2*>(&m));
  __half2 hb = __hsub2(*reinterpret_cast<__half2*>(&b), *reinterpret_cast<const __half2*>(&m));
  lo = *reinterpret_cast<uint32_t*>(&ha); hi = *reinterpret_cast<uint32_t*>(&hb);
}

__global__ __launch_bounds__(NT, 1) void decode_sm80_kernel(
    const __half* __restrict__ Q, const int8_t* __restrict__ K, const int8_t* __restrict__ V,
    const float* __restrict__ sK, const float* __restrict__ sV,
    float* __restrict__ Aout, float* __restrict__ Bout,
    int H, int Hkv, float scale, int ns,   // ns -- срезов на ВЫХОДЕ (= 2 x блоков по оси z: по срезу на пару)
    const int* __restrict__ block_table, int bs_log, int max_blocks,
    const int* __restrict__ seqlens, int qGrp, int GF, int qCausal, long blk_stride,
    const int* __restrict__ selIdx, const int* __restrict__ selCnt, int selStride, int selBlkLog) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem& sh = *reinterpret_cast<Smem*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int rows = qGrp * GF;
  const int b = blockIdx.x, hkv = blockIdx.y, si = blockIdx.z;
  const int nsb = gridDim.z;                      // срезов-БЛОКОВ; на выходе их вдвое больше (по паре)
  const int len = seqlens[b];
  // ---- список плиток среза (как у химеры: плотный срез или блоки отбора)
  const int per_slice = ((len + KT - 1) / KT + nsb - 1) / nsb * KT;
  const int k0 = si * per_slice, k1 = min(len, k0 + per_slice);
  const int* sel = nullptr; int TPB = 1, T0 = 0, T1 = (k0 < k1) ? ((k1 - k0 + KT - 1) / KT) : 0, lim = k1;
  if (selIdx != nullptr) {
    const long o = (long)(b * qGrp) * Hkv + hkv;
    sel = selIdx + o * (long)selStride;
    TPB = (1 << selBlkLog) / KT;
    const int ntiles = selCnt[o] * TPB, per = (ntiles + nsb - 1) / nsb;
    T0 = min(si * per, ntiles); T1 = min(T0 + per, ntiles); lim = len;
  }
  auto tstart = [&](int t) -> int {
    return sel ? ((sel[t / TPB] << selBlkLog) + (t - (t / TPB) * TPB) * KT) : (k0 + t * KT);
  };
  const int ntl = T1 - T0;

  if (warp >= 4) {
    // =============================== ВОЗЧИКИ ===============================
    // Раскладка копий -- ПО СТРОКАМ: инструкция варпа = 2 ключа x 256 Б подряд (полные 128-байтные линии).
    // Было "нить = ключ": 32 куска по 16 Б с шагом 512 Б -- полусекторы, двойные запросы в L2; пустой поток
    // упирался в ~890 ГБ/с независимо от глубины кольца (замер фальсификатором DIAG=3, стадии 2/3/4).
    // Нить p возит и РАЗВОРАЧИВАЕТ куски (ключ (p>>4)+4j, столбец p&15), j=0..7 -- свои же, барьера не нужно.
    const int p = tid - 128;
    const int c16 = p & 15, key0 = p >> 4;            // ключи key0 + (2*NPW)*j, j < 16/NPW
    constexpr int KSTEP = 2 * NPW, NJ = 16 / NPW;
    auto issue = [&](int t, int st) {
      const int kb = tstart(T0 + t);
      // плитка из 32 ключей лежит в ОДНОЙ странице (bs >= 32 и кратность проверены хозяином)
      const long blk = block_table[(long)b * max_blocks + (kb >> bs_log)];
      const long wb0 = kb & ((1 << bs_log) - 1);
#pragma unroll
      for (int j = 0; j < NJ; ++j) {
        const int key = key0 + KSTEP * j;
        const bool live = kb + key < lim;
        const long base = (blk * blk_stride + (wb0 + (live ? key : 0)) * Hkv + hkv) * D + c16 * 16;
        cp16(&sh.kb[st][key][c16 * 16], K + base, live);
        cp16(&sh.vb[st][key][c16 * 16], V + base, live);
      }
      if (p < 64) {
        const int key = p & 31;
        const bool live = kb + key < lim;
        const long slot = ((blk << bs_log) + wb0 + (live ? key : 0)) * Hkv + hkv;
        if (p < 32) cp4(&sh.ksb[st][key], sK + slot, live);
        else        cp4(&sh.vsb[st][key], sV + slot, live);
      }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) { if (s < ntl) issue(s, s); cp_commit(); }
    for (int t = 0; t < ntl; ++t) {
      const int st = t % STAGES, bb = t & 1;
      if (t + STAGES - 1 < ntl) issue(t + STAGES - 1, (t + STAGES - 1) % STAGES);
      cp_commit();
      cp_wait<STAGES - 1>();
      if (t >= 2) bar_sync(BAR_EMPTY + bb);       // счётчики отпустили буфер bb (плитка t-2)
      if (FA2SM80_DEC_DIAG == 1 || FA2SM80_DEC_DIAG == 3) { bar_arrive(BAR_FULL + bb); continue; }
#pragma unroll
      for (int j = 0; j < NJ; ++j) {
        const int key = key0 + KSTEP * j;
        const uint4 k4 = *reinterpret_cast<const uint4*>(&sh.kb[st][key][c16 * 16]);
        const uint4 v4 = *reinterpret_cast<const uint4*>(&sh.vb[st][key][c16 * 16]);
        __half* kd = &sh.kh[bb][key][c16 * 16];
        __half* vd = &sh.vh[bb][key][c16 * 16];
        uint4 o0, o1;
        deq4(k4.x, o0.x, o0.y); deq4(k4.y, o0.z, o0.w); deq4(k4.z, o1.x, o1.y); deq4(k4.w, o1.z, o1.w);
        *reinterpret_cast<uint4*>(kd) = o0; *reinterpret_cast<uint4*>(kd + 8) = o1;
        deq4(v4.x, o0.x, o0.y); deq4(v4.y, o0.z, o0.w); deq4(v4.z, o1.x, o1.y); deq4(v4.w, o1.z, o1.w);
        *reinterpret_cast<uint4*>(vd) = o0; *reinterpret_cast<uint4*>(vd + 8) = o1;
      }
      if (p < 32) sh.ks[bb][p] = sh.ksb[st][p];
      else if (p < 64) sh.vs[bb][p - 32] = sh.vsb[st][p - 32];
      bar_arrive(BAR_FULL + bb);
    }
    cp_wait<0>();
    // гасим висящие прибытия счётчиков на ПУСТ последних двух плиток
    for (int t = max(ntl, 2); t < ntl + 2; ++t) bar_sync(BAR_EMPTY + (t & 1));
    return;
  }

  // =============================== СЧЁТЧИКИ ===============================
  // Пара pr берёт плитки t = pr, pr+2, ... (буфер bb == pr навсегда), у каждой пары СВОЙ онлайн-софтмакс;
  // на выходе пары -- два отдельных среза (2*si + pr), их сводит hmma_svod_kernel вместе с прочими.
  const int g = lane >> 2, tq = lane & 3;
  const int pr = warp >> 1;
  const int r0 = (warp & 1) * 16;                 // строки варпа
  // Q -> разделяемая (строки >= rows -- нули), затем фрагменты A в регистры (штатная карта, ldmatrix)
  for (int i = tid; i < 32 * (D / 8); i += 128) {
    const int r = i / (D / 8), c8 = (i % (D / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r < rows) {
      const int rq = r / GF, hh = r - rq * GF;
      v = *reinterpret_cast<const uint4*>(Q + ((long)(b * qGrp + rq) * H + hkv * GF + hh) * D + c8);
    }
    *reinterpret_cast<uint4*>(&sh.q[r][c8]) = v;
  }
  asm volatile("bar.sync 5, 128;\n" ::: "memory");   // только счётчики (обе пары)
  uint32_t qa[D / 16][4];
#pragma unroll
  for (int s = 0; s < D / 16; ++s)
    ldsm4(qa[s], &sh.q[r0 + (lane & 15)][s * 16 + (lane >> 4) * 8]);
  // предел ключей для двух строк нити (g и g+8): строка группы rq видит len-(qGrp-1-rq) ключей
  int limr[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = r0 + g + 8 * h;
    const int rq = (r < rows) ? r / GF : 0;
    limr[h] = qCausal ? (len - (qGrp - 1 - rq)) : len;
    if (r >= rows) limr[h] = -1;                  // мёртвая строка: всё маскируется
  }
  const float sl2 = scale * 1.4426950408889634f;  // exp2-домен
  float o[D / 8][4];
#pragma unroll
  for (int j = 0; j < D / 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
  float m2[2] = {-CUDART_INF_F, -CUDART_INF_F}, l[2] = {0.f, 0.f};
  const bool active = (r0 < rows);                 // второй счётчик простаивает при rows <= 16

  const int bb = pr;
  for (int t = pr; t < ntl; t += 2) {
    bar_sync(BAR_FULL + bb);
    if (active && FA2SM80_DEC_DIAG != 2 && FA2SM80_DEC_DIAG != 3) {
      const int kb = tstart(T0 + t);
      // ---- S = Q K^T: 32 ключа = 4 n8-блока
      float s[4][4];
#pragma unroll
      for (int j = 0; j < 4; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
      for (int kk = 0; kk < D / 16; ++kk) {
#pragma unroll
        for (int jb = 0; jb < 2; ++jb) {          // 16 ключей за ldmatrix.x4
          uint32_t bf[4];
          const int mi = lane >> 3, rr = lane & 7;
          ldsm4(bf, &sh.kh[bb][jb * 16 + rr + (mi >> 1) * 8][kk * 16 + (mi & 1) * 8]);
          mma(s[2 * jb], qa[kk], bf[0], bf[1]);
          mma(s[2 * jb + 1], qa[kk], bf[2], bf[3]);
        }
      }
      // ---- масштаб K, маска, онлайн-софтмакс (exp2)
      float mx[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int kc = j * 8 + 2 * tq + (e & 1), h = e >> 1;
          float v = s[j][e] * sh.ks[bb][kc] * sl2;
          if (kb + kc >= limr[h] || kb + kc >= lim) v = -CUDART_INF_F;
          s[j][e] = v;
          mx[h] = fmaxf(mx[h], v);
        }
      float corr[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
        mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
        const float mn = fmaxf(m2[h], mx[h]);
        corr[h] = (m2[h] == -CUDART_INF_F) ? 0.f : exp2f(m2[h] - mn);
        if (mn == -CUDART_INF_F) corr[h] = 1.f;
        m2[h] = mn;
        l[h] *= corr[h];
      }
      // перемасштаб O только у строк, чей максимум сдвинулся (после первых плиток -- редкость)
#pragma unroll
      for (int h = 0; h < 2; ++h)
        if (corr[h] != 1.f) {
#pragma unroll
          for (int j = 0; j < D / 8; ++j) { o[j][2 * h] *= corr[h]; o[j][2 * h + 1] *= corr[h]; }
        }
      // P (с масштабом V) -> фрагменты A второго умножения, не выходя из регистров
      uint32_t pa[2][4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float pv[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int kc = j * 8 + 2 * tq + (e & 1), h = e >> 1;
          const float pe = (s[j][e] == -CUDART_INF_F) ? 0.f : exp2f(s[j][e] - m2[h]);
          l[h] += pe;
          pv[e] = pe * sh.vs[bb][kc];
        }
        const __half2 lo = __floats2half2_rn(pv[0], pv[1]), hi = __floats2half2_rn(pv[2], pv[3]);
        // блок 16 ключей jb = j>>1: a0/a1 -- первые 8 ключей (j чётный), a2/a3 -- вторые 8
        pa[j >> 1][(j & 1) * 2 + 0] = *reinterpret_cast<const uint32_t*>(&lo);
        pa[j >> 1][(j & 1) * 2 + 1] = *reinterpret_cast<const uint32_t*>(&hi);
      }
      // ---- O += P V: k = 32 ключа (2 шага k16), n = D
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
        for (int jd = 0; jd < D / 16; ++jd) {
          uint32_t bf[4];
          const int mi = lane >> 3, rr = lane & 7;
          ldsm4t(bf, &sh.vh[bb][kk * 16 + rr + (mi & 1) * 8][jd * 16 + (mi >> 1) * 8]);
          mma(o[2 * jd], pa[kk], bf[0], bf[1]);
          mma(o[2 * jd + 1], pa[kk], bf[2], bf[3]);
        }
      }
    }
    bar_arrive(BAR_EMPTY + bb);
  }
  // ---- выдача среза: O без нормировки, (m, l) в натуральных единицах
  if (!active) return;
  const int so = 2 * si + pr;                     // срез на выходе
  const bool mine = pr < ntl;                     // у пары была хоть одна плитка
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    l[h] += __shfl_xor_sync(0xffffffffu, l[h], 1);
    l[h] += __shfl_xor_sync(0xffffffffu, l[h], 2);
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = r0 + g + 8 * h;
    if (r >= rows) continue;
    const int rq = r / GF, hh = r - rq * GF;
    const long row = (long)(b * qGrp + rq) * H + hkv * GF + hh;
    const long base = (row * ns + so) * (long)D;
#pragma unroll
    for (int j = 0; j < D / 8; ++j) {
      float2 v2 = make_float2(mine ? o[j][2 * h] : 0.f, mine ? o[j][2 * h + 1] : 0.f);
      *reinterpret_cast<float2*>(&Aout[base + j * 8 + 2 * tq]) = v2;
    }
    if (tq == 0) {
      Bout[(row * ns + so) * 2] = (mine && m2[h] != -CUDART_INF_F) ? m2[h] * 0.6931471805599453f : -CUDART_INF_F;
      Bout[(row * ns + so) * 2 + 1] = mine ? l[h] : 0.f;
    }
  }
}

// СВОД СРЕЗОВ для A100. Прежний hmma_svod_kernel на sm_80 шёл 25 мкс на вызов -- дольше самого декода
// (23 мкс при 32K): каждая нить последовательно пробегала все ns срезов дважды (максимум, сумма), затем
// по срезу на столбец -- цепочка зависимых __expf и загрузок длиной ns. Здесь: блок на (строка, голова),
// D нитей; веса срезов e^{m_i-M} считаются ОДИН раз параллельно и лежат в разделяемой, а свод столбца --
// ns НЕЗАВИСИМЫХ загрузок (развёртка по 8), то есть упор в полосу, а не в латентность цепи.
template <int NSMAX = 256>
__global__ void svod_sm80_kernel(const float* __restrict__ A, const float* __restrict__ B, half* __restrict__ Out,
                                 int H, int Dh, int ns, int ldo) {
  __shared__ float w[NSMAX];
  __shared__ float red[32];
  const int r = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int nw = (blockDim.x + 31) >> 5;
  const long bb = ((long)r * H + h) * ns;
  // 1) максимум m по срезам
  float mloc = -CUDART_INF_F;
  for (int i = tid; i < ns; i += blockDim.x) mloc = fmaxf(mloc, B[(bb + i) * 2]);
#pragma unroll
  for (int o = 16; o; o >>= 1) mloc = fmaxf(mloc, __shfl_xor_sync(0xffffffffu, mloc, o));
  if (lane == 0) red[wid] = mloc;
  __syncthreads();
  float M = -CUDART_INF_F;
  for (int i = 0; i < nw; ++i) M = fmaxf(M, red[i]);
  __syncthreads();
  // 2) веса и знаменатель
  float lloc = 0.f;
  for (int i = tid; i < ns; i += blockDim.x) {
    const float m = B[(bb + i) * 2];
    const float e = (m == -CUDART_INF_F) ? 0.f : __expf(m - M);
    w[i] = e;
    lloc += e * B[(bb + i) * 2 + 1];
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) lloc += __shfl_xor_sync(0xffffffffu, lloc, o);
  if (lane == 0) red[wid] = lloc;
  __syncthreads();
  float L = 0.f;
  for (int i = 0; i < nw; ++i) L += red[i];
  const float inv = (L > 0.f) ? 1.f / L : 0.f;
  // 3) свод столбцов: ns независимых загрузок на нить
  for (int d = tid; d < Dh; d += blockDim.x) {
    const float* a = A + bb * (long)Dh + d;
    float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
    int i = 0;
    for (; i + 4 <= ns; i += 4) {
      o0 += w[i] * a[(long)i * Dh];         o1 += w[i + 1] * a[(long)(i + 1) * Dh];
      o2 += w[i + 2] * a[(long)(i + 2) * Dh]; o3 += w[i + 3] * a[(long)(i + 3) * Dh];
    }
    for (; i < ns; ++i) o0 += w[i] * a[(long)i * Dh];
    Out[(long)r * ldo + h * Dh + d] = __float2half(((o0 + o1) + (o2 + o3)) * inv);
  }
}

// Свод, вариант 2: сетка (строки, головы, D/64) по 64 нити -- вчетверо больше блоков (свод был 16.5 мкс
// при 48 блоках: латентность L2 на цепочке из ns/4 загрузок), веса срезов пересчитываются в каждом блоке
// (ns <= 256 чисел -- дёшево), 8 независимых накопителей на нить.
template <int NSMAX = 256>
__global__ __launch_bounds__(64) void svod_sm80_kernel2(const float* __restrict__ A, const float* __restrict__ B,
                                                        half* __restrict__ Out, int H, int Dh, int ns, int ldo) {
  __shared__ float w[NSMAX];
  __shared__ float red[2];
  const int r = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const long bb = ((long)r * H + h) * ns;
  float mloc = -CUDART_INF_F;
  for (int i = tid; i < ns; i += 64) mloc = fmaxf(mloc, B[(bb + i) * 2]);
#pragma unroll
  for (int o = 16; o; o >>= 1) mloc = fmaxf(mloc, __shfl_xor_sync(0xffffffffu, mloc, o));
  if (lane == 0) red[wid] = mloc;
  __syncthreads();
  const float M = fmaxf(red[0], red[1]);
  __syncthreads();
  float lloc = 0.f;
  for (int i = tid; i < ns; i += 64) {
    const float m = B[(bb + i) * 2];
    const float e = (m == -CUDART_INF_F) ? 0.f : __expf(m - M);
    w[i] = e;
    lloc += e * B[(bb + i) * 2 + 1];
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) lloc += __shfl_xor_sync(0xffffffffu, lloc, o);
  if (lane == 0) red[wid] = lloc;
  __syncthreads();
  const float L = red[0] + red[1];
  const float inv = (L > 0.f) ? 1.f / L : 0.f;
  const int d = blockIdx.z * 64 + tid;
  if (d >= Dh) return;
  const float* a = A + bb * (long)Dh + d;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  int i = 0;
  for (; i + 8 <= ns; i += 8) {
#pragma unroll
    for (int u = 0; u < 8; ++u) acc[u] += w[i + u] * a[(long)(i + u) * Dh];
  }
  for (; i < ns; ++i) acc[0] += w[i] * a[(long)i * Dh];
  const float o = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
  Out[(long)r * ldo + h * Dh + d] = __float2half(o * inv);
}

// Свод, вариант 3 (боевой). Замер nsys: вариант 2 -- 19.7 мкс на 4.7 МБ частичных из L2 (~240 ГБ/с), упор в
// число запросов в полёте (4 варпа на SM, цепочка ns/8 раундов загрузок). Здесь блок = 32 столбца x 8 варпов:
// варп w суммирует срезы i = w, w+8, ... (все загрузки независимы), затем 8 частичных складываются через
// разделяемую. Блоков (строки x головы x D/32) = 384 при боевой форме, запросов в полёте -- в 8 раз больше.
template <int NSMAX = 256>
__global__ __launch_bounds__(256) void svod_sm80_kernel3(const float* __restrict__ A, const float* __restrict__ B,
                                                         half* __restrict__ Out, int H, int Dh, int ns, int ldo) {
  __shared__ float w[NSMAX];
  __shared__ float red[8][33];
  __shared__ float stat[2];
  const int r = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const long bb = ((long)r * H + h) * ns;
  if (wid == 0) {
    float m = -CUDART_INF_F;
    for (int i = lane; i < ns; i += 32) m = fmaxf(m, B[(bb + i) * 2]);
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float l = 0.f;
    for (int i = lane; i < ns; i += 32) {
      const float mi = B[(bb + i) * 2];
      const float e = (mi == -CUDART_INF_F) ? 0.f : __expf(mi - m);
      w[i] = e;
      l += e * B[(bb + i) * 2 + 1];
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
    if (lane == 0) stat[0] = (l > 0.f) ? 1.f / l : 0.f;
  }
  __syncthreads();
  const int d = blockIdx.z * 32 + lane;
  float acc = 0.f;
  if (d < Dh) {
    const float* a = A + bb * (long)Dh + d;
    float p[4] = {0.f, 0.f, 0.f, 0.f};
    int i = wid, u = 0;
    for (; i < ns; i += 8, u = (u + 1) & 3) p[u] += w[i] * a[(long)i * Dh];
    acc = (p[0] + p[1]) + (p[2] + p[3]);
  }
  red[wid][lane] = acc;
  __syncthreads();
  if (wid == 0 && d < Dh) {
    float o = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) o += red[k][lane];
    Out[(long)r * ldo + h * Dh + d] = __float2half(o * stat[0]);
  }
}

}  // namespace fa2sm80_dec
