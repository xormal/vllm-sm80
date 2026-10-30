// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// ПРЕФИЛЛ ВНИМАНИЯ ДЛЯ AMPERE (sm_80) НА int8-ПУЛЕ: FA2-конвейер прямо в коде GEMM, d=256, GQA, причинно.
// Замена порта volta_fwd_ws (через sm80-WarpMma он дал 52 TFLOP/s при ~125 у FA2 на этой карте).
//
// РАСКЛАДКА (FlashAttention-2): блок = (128 строк запросов, голова q). 8 варпов-СЧЁТЧИКОВ по 16 строк x
// весь D: S = Q K^T через ldmatrix (Q и K fp16 в разделяемой), онлайн-софтмакс в регистрах, P НЕ покидает
// регистры (аккумулятор S 16x16 -- это фрагмент A второго умножения), O += P V через ldmatrix.trans.
// 2 варпа-ВОЗЧИКА (вольтовские роли ws, но на cp.async): кольцо STAGES стадий int8 K/V + масштабы,
// деквант int8 -> fp16: K точно (целое), V сразу с масштабом токена (|v*s| ~ 1..3, нормальный диапазон
// fp16, а не P*s_v ~ 1e-5 в поддиапазоне). Масштаб K -- на столбец S в fp32.
// ФАЗЫ РАЗВЕДЕНЫ: буферы fp16 K и V раздельные, и у каждого своя пара именованных барьеров ПОЛОН/ПУСТ.
// Возчик разворачивает K плитки t+1, пока счётчики делают софтмакс и P*V плитки t, и V плитки t+1, пока
// идёт S плитки t+1 -- двойной буфер "по фазе" без лишней разделяемой.
#pragma once
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cstdint>

namespace fa2sm80_pf {

constexpr int D = 256;
constexpr int BQ = 128;               // строк запросов на блок
constexpr int BK = 32;                // ключей в плитке
constexpr int STAGES = 3;             // кольцо int8
constexpr int NCW = 8;                // варпов-счётчиков
#ifndef FA2SM80_PF_NPW
#define FA2SM80_PF_NPW 4
#endif
constexpr int NPW = FA2SM80_PF_NPW;   // варпов-возчиков
constexpr int NT = (NCW + NPW) * 32;
constexpr int NB = NT;                // участников барьера ПОЛОН/ПУСТ: все
constexpr int LDH = D + 8;            // шаг строки fp16 (528 Б: восемь строк ldmatrix -- разные банки)
constexpr int LDB8 = D + 16;          // шаг строки int8 (272 Б: восемь ключей четверти варпа -- разные банки)
enum { BAR_KF = 1, BAR_KE = 2, BAR_VF = 3, BAR_VE = 4, BAR_C = 5 };

struct Smem {
  __half q[BQ][LDH];
  __half kh[BK][LDH];
  __half vh[BK][LDH];
  int8_t kb[STAGES][BK][LDB8];
  int8_t vb[STAGES][BK][LDB8];
  float ksb[STAGES][BK], vsb[STAGES][BK];
  float ks[BK];
};

__device__ __forceinline__ uint32_t su(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(void* s, const void* g, bool live) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(su(s)), "l"(g), "r"(live ? 16 : 0));
}
__device__ __forceinline__ void cp4(void* s, const void* g, bool live) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(su(s)), "l"(g), "r"(live ? 4 : 0));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void bsync(int id) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "n"(NB) : "memory"); }
__device__ __forceinline__ void barrive(int id) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "n"(NB) : "memory"); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(su(p)));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(su(p)));
}
__device__ __forceinline__ void mma(float* d, const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void deq4(uint32_t w, uint32_t& lo, uint32_t& hi) {     // 4 знаковых байта -> fp16 точно
  const uint32_t t = w ^ 0x80808080u;
  uint32_t a = __byte_perm(t, 0x64646464u, 0x5140u), b = __byte_perm(t, 0x64646464u, 0x5342u);
  const uint32_t m = 0x64806480u;
  __half2 ha = __hsub2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<const __half2*>(&m));
  __half2 hb = __hsub2(*reinterpret_cast<__half2*>(&b), *reinterpret_cast<const __half2*>(&m));
  lo = *reinterpret_cast<uint32_t*>(&ha); hi = *reinterpret_cast<uint32_t*>(&hb);
}

// vOff[i] -- смещение (в байтах от Kc/Vc) 16-токенной виртуальной страницы i, vScl[i] -- индекс масштаба
// её первого токена (как у вольтовского пути: таблицы строит хозяин, ядро не делит).
template <bool CAUSAL>
__global__ __launch_bounds__(NT, 1) void fwd_i8_paged_sm80(
    const __half* __restrict__ Q, __half* __restrict__ O, float* __restrict__ LSE,
    const int8_t* __restrict__ Kc, const int8_t* __restrict__ Vc,
    const float* __restrict__ Ks, const float* __restrict__ Vs,
    const long* __restrict__ vOff, const long* __restrict__ vScl,
    int Sq, int Sk, int H, int Hkv, float sl2, int lse_dim,
    const int* __restrict__ selIdx, const int* __restrict__ selCnt, int selStride, int selGroupQ, int selSubLog,
    const int* __restrict__ vl = nullptr) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem& sh = *reinterpret_cast<Smem*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int q0 = blockIdx.x * BQ;
  const int h = blockIdx.y, hk = h / (H / Hkv);
  // [ОДИН ГРАФ 30.10] VARLEN: плитка blockIdx.x берёт строку из метаданных на устройстве (paged_varlen_meta):
  // vl[6*i] = {первый токен строки в Q/O, q0 плитки в строке, Sq строки, Sk строки, база таблиц страниц строки, 0}.
  // Sq <= 0 -- пустая плитка запаса (сетка фиксирована под граф). Отбор (selIdx) в этом режиме не используется.
  // Арифметика плитки та же, что у вызова на одну строку: выход побитно тот же.
  if (vl != nullptr) {
    const int* m = vl + 6 * (long)blockIdx.x;
    if (m[2] <= 0) return;                                 // весь блок разом -- до любых барьеров
    Q += (long)m[0] * H * D; O += (long)m[0] * H * D;
    if (LSE != nullptr) LSE += m[0];
    q0 = m[1]; Sq = m[2]; Sk = m[3]; vOff += m[4]; vScl += m[4];
  }
  const int off = Sk - Sq;                                 // позиция запроса i = off + i
  // ---- список плиток ключей
  const int kEnd = CAUSAL ? min(Sk, off + min(Sq, q0 + BQ)) : Sk;
  const int* sel = nullptr; int nsel = 0;
  if (selIdx != nullptr) {
    const int st = q0 / (BQ * selGroupQ);
    sel = selIdx + ((long)st * Hkv + hk) * selStride;
    nsel = selCnt[(long)st * Hkv + hk];
  }
  const int ntl = sel ? (nsel << selSubLog) : (kEnd + BK - 1) / BK;
  auto kb_of = [&](int t) -> int {
    return sel ? (((sel[t >> selSubLog] << selSubLog) | (t & ((1 << selSubLog) - 1))) * BK) : t * BK;
  };

  if (warp >= NCW) {
    // =========================== ВОЗЧИКИ ===========================
    // Копии ПО СТРОКАМ (инструкция варпа = 2 ключа x 256 Б подряд), см. decode_sm80.cuh: "нить = ключ" давало
    // полусекторы с шагом строки пула. Нить возит и разворачивает куски (ключ key0 + KSTEP*j, столбец c16).
    const int p = tid - NCW * 32;
    const int c16 = p & 15, key0 = p >> 4;
    constexpr int KSTEP = 2 * NPW, NJ = 16 / NPW;
    auto issue = [&](int t, int st) {
      const int kb = kb_of(t);
#pragma unroll
      for (int j = 0; j < NJ; ++j) {
        const int key = key0 + KSTEP * j;
        const int gk = kb + key;
        const bool live = gk < Sk;
        const int gkc = live ? gk : 0;
        const long base = vOff[gkc >> 4] + (long)(gkc & 15) * Hkv * D + (long)hk * D + c16 * 16;
        cp16(&sh.kb[st][key][c16 * 16], Kc + base, live);
        cp16(&sh.vb[st][key][c16 * 16], Vc + base, live);
      }
      if (p < 64) {
        const int key = p & 31, gk = kb + key;
        const bool live = gk < Sk;
        const int gkc = live ? gk : 0;
        const long si = vScl[gkc >> 4] + (long)(gkc & 15) * Hkv + hk;
        if (p < 32) cp4(&sh.ksb[st][key], Ks + si, live);
        else        cp4(&sh.vsb[st][key], Vs + si, live);
      }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) { if (s < ntl) issue(s, s); commit(); }
    for (int t = 0; t < ntl; ++t) {
      const int st = t % STAGES;
      if (t + STAGES - 1 < ntl) issue(t + STAGES - 1, (t + STAGES - 1) % STAGES);
      commit();
      wait<STAGES - 1>();
      // масштабы V ключей этой нити положили cp.async ДРУГИЕ нити (p in [32,64)): их копии завершены
      // только для них самих -- нужен барьер возчиков (без него гонка: relL2 2e-3 при 2 возчиках, мусор при 4)
      asm volatile("bar.sync 6, %0;\n" :: "r"(NPW * 32) : "memory");
      if (t > 0) bsync(BAR_KE);
#pragma unroll
      for (int j = 0; j < NJ; ++j) {
        const int key = key0 + KSTEP * j;
        const uint4 k4 = *reinterpret_cast<const uint4*>(&sh.kb[st][key][c16 * 16]);
        __half* kd = &sh.kh[key][c16 * 16];
        uint4 o0, o1;
        deq4(k4.x, o0.x, o0.y); deq4(k4.y, o0.z, o0.w); deq4(k4.z, o1.x, o1.y); deq4(k4.w, o1.z, o1.w);
        *reinterpret_cast<uint4*>(kd) = o0; *reinterpret_cast<uint4*>(kd + 8) = o1;
      }
      if (p < 32) sh.ks[p] = sh.ksb[st][p];
      barrive(BAR_KF);
      if (t > 0) bsync(BAR_VE);
#pragma unroll
      for (int j = 0; j < NJ; ++j) {
        const int key = key0 + KSTEP * j;
        const uint4 v4 = *reinterpret_cast<const uint4*>(&sh.vb[st][key][c16 * 16]);
        __half* vd = &sh.vh[key][c16 * 16];
        const __half2 s2 = __float2half2_rn(sh.vsb[st][key]);
        uint4 o0, o1;
        deq4(v4.x, o0.x, o0.y); deq4(v4.y, o0.z, o0.w); deq4(v4.z, o1.x, o1.y); deq4(v4.w, o1.z, o1.w);
        uint32_t* w0 = &o0.x; uint32_t* w1 = &o1.x;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          __half2 a = __hmul2(*reinterpret_cast<__half2*>(&w0[e]), s2); w0[e] = *reinterpret_cast<uint32_t*>(&a);
          __half2 b = __hmul2(*reinterpret_cast<__half2*>(&w1[e]), s2); w1[e] = *reinterpret_cast<uint32_t*>(&b);
        }
        *reinterpret_cast<uint4*>(vd) = o0; *reinterpret_cast<uint4*>(vd + 8) = o1;
      }
      barrive(BAR_VF);
    }
    wait<0>();
    if (ntl > 0) { bsync(BAR_KE); bsync(BAR_VE); }        // гасим последние прибытия счётчиков
    return;
  }

  // =========================== СЧЁТЧИКИ ===========================
  const int g = lane >> 2, tq = lane & 3;
  const int r0 = warp * 16;
  for (int i = tid; i < BQ * (D / 8); i += NCW * 32) {
    const int r = i / (D / 8), c8 = (i % (D / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (q0 + r < Sq) v = *reinterpret_cast<const uint4*>(Q + ((long)(q0 + r) * H + h) * D + c8);
    *reinterpret_cast<uint4*>(&sh.q[r][c8]) = v;
  }
  asm volatile("bar.sync %0, %1;\n" :: "r"((int)BAR_C), "r"(NCW * 32) : "memory");   // только счётчики
  int qpos[2];
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) qpos[hh] = (q0 + r0 + g + 8 * hh < Sq) ? off + q0 + r0 + g + 8 * hh : -1;
  float o[D / 8][4];
#pragma unroll
  for (int j = 0; j < D / 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
  float m2[2] = {-CUDART_INF_F, -CUDART_INF_F}, l[2] = {0.f, 0.f};
  const int mi = lane >> 3, rr = lane & 7;
  for (int t = 0; t < ntl; ++t) {
    const int kb = kb_of(t);
    bsync(BAR_KF);
    float s[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
      uint32_t a[4];
      ldsm4(a, &sh.q[r0 + (lane & 15)][kk * 16 + (lane >> 4) * 8]);
#pragma unroll
      for (int jb = 0; jb < 2; ++jb) {
        uint32_t bf[4];
        ldsm4(bf, &sh.kh[jb * 16 + rr + (mi >> 1) * 8][kk * 16 + (mi & 1) * 8]);
        mma(s[2 * jb], a, bf[0], bf[1]);
        mma(s[2 * jb + 1], a, bf[2], bf[3]);
      }
    }
    float ksv[4][2];
#pragma unroll
    for (int j = 0; j < 4; ++j) { ksv[j][0] = sh.ks[j * 8 + 2 * tq]; ksv[j][1] = sh.ks[j * 8 + 2 * tq + 1]; }
    barrive(BAR_KE);                                     // K больше не нужен: возчик разворачивает следующий
    float mx[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int kk = kb + j * 8 + 2 * tq + (e & 1), hh = e >> 1;
        float v = s[j][e] * ksv[j][e & 1] * sl2;
        if (kk >= Sk || qpos[hh] < 0 || (CAUSAL && kk > qpos[hh])) v = -CUDART_INF_F;
        s[j][e] = v;
        mx[hh] = fmaxf(mx[hh], v);
      }
    float corr[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffffu, mx[hh], 1));
      mx[hh] = fmaxf(mx[hh], __shfl_xor_sync(0xffffffffu, mx[hh], 2));
      const float mn = fmaxf(m2[hh], mx[hh]);
      corr[hh] = (mn == -CUDART_INF_F) ? 1.f : ((m2[hh] == -CUDART_INF_F) ? 0.f : exp2f(m2[hh] - mn));
      m2[hh] = mn;
      l[hh] *= corr[hh];
    }
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
      if (corr[hh] != 1.f) {
#pragma unroll
        for (int j = 0; j < D / 8; ++j) { o[j][2 * hh] *= corr[hh]; o[j][2 * hh + 1] *= corr[hh]; }
      }
    uint32_t pa[2][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float pv[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int hh = e >> 1;
        const float pe = (s[j][e] == -CUDART_INF_F) ? 0.f : exp2f(s[j][e] - m2[hh]);
        pv[e] = pe;
      }
      const __half2 lo = __floats2half2_rn(pv[0], pv[1]), hi = __floats2half2_rn(pv[2], pv[3]);
      // знаменатель -- по ОКРУГЛЁННЫМ весам, теми же, что видит P*V (вольтовский урок ws: иначе рассогласование)
      l[0] += __low2float(lo) + __high2float(lo);
      l[1] += __low2float(hi) + __high2float(hi);
      pa[j >> 1][(j & 1) * 2 + 0] = *reinterpret_cast<const uint32_t*>(&lo);
      pa[j >> 1][(j & 1) * 2 + 1] = *reinterpret_cast<const uint32_t*>(&hi);
    }
    bsync(BAR_VF);
#pragma unroll
    for (int kk = 0; kk < 2; ++kk)
#pragma unroll
      for (int jd = 0; jd < D / 16; ++jd) {
        uint32_t bf[4];
        ldsm4t(bf, &sh.vh[kk * 16 + rr + (mi & 1) * 8][jd * 16 + (mi >> 1) * 8]);
        mma(o[2 * jd], pa[kk], bf[0], bf[1]);
        mma(o[2 * jd + 1], pa[kk], bf[2], bf[3]);
      }
    barrive(BAR_VE);
  }
  // ---- эпилог: O / l, LSE (натуральный логарифм)
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    l[hh] += __shfl_xor_sync(0xffffffffu, l[hh], 1);
    l[hh] += __shfl_xor_sync(0xffffffffu, l[hh], 2);
  }
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    const int qr = q0 + r0 + g + 8 * hh;
    if (qr >= Sq) continue;
    const float inv = (l[hh] > 0.f) ? 1.f / l[hh] : 0.f;
    __half* orow = O + ((long)qr * H + h) * D;
#pragma unroll
    for (int j = 0; j < D / 8; ++j)
      *reinterpret_cast<__half2*>(orow + j * 8 + 2 * tq) = __floats2half2_rn(o[j][2 * hh] * inv, o[j][2 * hh + 1] * inv);
    if (tq == 0 && LSE != nullptr)
      LSE[(long)h * lse_dim + qr] = (l[hh] > 0.f) ? (m2[hh] * 0.6931471805599453f + __logf(l[hh])) : -CUDART_INF_F;
  }
}

}  // namespace fa2sm80_pf
