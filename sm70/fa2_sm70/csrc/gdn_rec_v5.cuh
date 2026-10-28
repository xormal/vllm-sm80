// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// [A100 25.10] РЕКУРРЕНТ GDN ДЕКОДА v5 -- ПЕРЕУПАКОВКА (FA2SM80_GDN_V5=1). Контракт и арифметика -- volta_gdn_rec<...,
// GATE=1, short> (int16-состояние с масштабами), выход/состояние/масштабы ПОБИТОВО те же (стенд bench/sa/gdnrec_test:
// N=1..8, MTP T=5). Отличия -- только в раскладке работы:
//   * свёртки по K -- «сведением с разбросом» (9 shfl на 8 строк вместо 40; дерево и пары сложений те же);
//   * всё, что не зависит от состояния (нормы q/k, гейтинг, слоты записи), -- в общем прологе блока;
//   * группа масштаба делится между 2 варпами (8 варпов на блок), amax группы -- именованный барьер этих двух варпов;
//   * int16 без F2I (FMUL + сложение с 1.5*2^23 -- то же округление к чётному), T=5 развёрнут.
// Стенд, мкс на вызов в графе из 48 (старое лучшее -> v5): N=1 14.9 -> 10.3, N=2 20.5 -> 13.7, N=4 30.3 -> 20.7,
// N=8 54.1 -> 40.0. Только Tn <= FA2_REC_TMAX (8) токенов на последовательность -- иначе запускающий берёт старое.
#pragma once
#include "volta_gdn_rec.cuh"

namespace fa2_sm70 {
namespace gdn {

// Сведение R частичных сумм полосы (R = 8 или 4 строк) по 32 полосам тем же деревом, что и пять xor-shfl.
// Полная сумма строки j остаётся у полос, чьи биты (4, 3, 2) [R=8] или (4, 3) [R=4] кодируют j.
template <int R>
__device__ __forceinline__ float rs_sum(const float (&x)[R], int lane) {
  constexpr unsigned F = 0xffffffffu;
  const bool b4 = lane & 16, b3 = lane & 8;
  if constexpr (R == 8) {
    const bool b2 = lane & 4;
    float y[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      const float snd = b4 ? x[m] : x[m + 4];
      const float kp = b4 ? x[m + 4] : x[m];
      y[m] = kp + __shfl_xor_sync(F, snd, 16);
    }
    float z[2];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const float snd = b3 ? y[m] : y[m + 2];
      const float kp = b3 ? y[m + 2] : y[m];
      z[m] = kp + __shfl_xor_sync(F, snd, 8);
    }
    float w = (b2 ? z[1] : z[0]) + __shfl_xor_sync(F, b2 ? z[0] : z[1], 4);
    w += __shfl_xor_sync(F, w, 2);
    w += __shfl_xor_sync(F, w, 1);
    return w;
  } else if constexpr (R == 2) {
    float w = (b4 ? x[1] : x[0]) + __shfl_xor_sync(F, b4 ? x[0] : x[1], 16);
    w += __shfl_xor_sync(F, w, 8);
    w += __shfl_xor_sync(F, w, 4);
    w += __shfl_xor_sync(F, w, 2);
    w += __shfl_xor_sync(F, w, 1);
    return w;
  } else {
    static_assert(R == 4, "R = 8, 4 или 2");
    float y[2];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const float snd = b4 ? x[m] : x[m + 2];
      const float kp = b4 ? x[m + 2] : x[m];
      y[m] = kp + __shfl_xor_sync(F, snd, 16);
    }
    float w = (b3 ? y[1] : y[0]) + __shfl_xor_sync(F, b3 ? y[0] : y[1], 8);
    w += __shfl_xor_sync(F, w, 4);
    w += __shfl_xor_sync(F, w, 2);
    w += __shfl_xor_sync(F, w, 1);
    return w;
  }
}
// строка, чья сумма у полосы lane, и полоса-источник строки j
template <int R> __device__ __forceinline__ int rs_row(int lane) {
  return R == 8 ? ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1)
       : R == 4 ? ((lane >> 4) & 1) * 2 + ((lane >> 3) & 1) : ((lane >> 4) & 1);
}
template <int R> __host__ __device__ constexpr int rs_src(int j) {
  return R == 8 ? ((j >> 2) & 1) * 16 + ((j >> 1) & 1) * 8 + (j & 1) * 4
       : R == 4 ? ((j >> 1) & 1) * 16 + (j & 1) * 8 : (j & 1) * 16;
}

__device__ __forceinline__ unsigned redux_max_u32(unsigned x) {
  unsigned r;
  asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;\n" : "=r"(r) : "r"(x));
  return r;
}

// int16 с округлением до ближайшего чётного БЕЗ F2I (четвертьскоростной конвейер): t = x*inv_s (тот же FMUL, что у
// rec_i16_st), затем t + 1.5*2^23 -- сложение округляет к целому (ulp = 1 в этой бинаде, к чётному), и младшие 16 бит
// числа = дополнительный код round(t) при |t| < 2^22. __fmul_rn/__fadd_rn не дают слить в FFMA (иное округление).
__device__ __forceinline__ unsigned pack_i16x2_rn(float x0, float x1, float inv_s) {
  const float y0 = __fadd_rn(__fmul_rn(x0, inv_s), 12582912.f);
  const float y1 = __fadd_rn(__fmul_rn(x1, inv_s), 12582912.f);
  return __byte_perm(__float_as_uint(y0), __float_as_uint(y1), 0x5410);
}

// v5: группа масштаба (8 строк с шагом 4) делится между SPLIT варпами (SPLIT=2 -> по 4 строки, 4 -> по 2): при одной
// последовательности сетка = 96 блоков, и у v4 на планировщик приходится ОДНА цепочка (CPI ~5). Здесь варпов на SM
// 8 или 16. amax группы -- через разделяемую и ИМЕНОВАННЫЙ барьер только варпов этой группы (bar.sync 1+grp), он стоит
// на записи состояния, не на критическом пути токена. Пролог общий: варп w считает нормы токена w, полоса it варпа 0 --
// гейтинг и слот токена it; один барьер блока на весь пролог. Арифметика каждого значения та же -> побитово.
template <int K, int V, int BV, bool L2N, bool SPEC, typename TI, typename IdxT, int SPLIT>
__global__ __launch_bounds__(128 * SPLIT, 1)
void gdn_rec_v5(
    const TI* __restrict__ q, const TI* __restrict__ k, const TI* __restrict__ v,
    const TI* __restrict__ beta, TI* __restrict__ o, short* __restrict__ hpool,
    const IdxT* __restrict__ ssm_idx, const int* __restrict__ n_acc, const IdxT* __restrict__ cu_seqlens,
    long stride_idx, long stride_slot, float scale, int T, int H, int HV,
    const TI* __restrict__ a_in, const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    const IdxT* __restrict__ ssm_idx_out, float* __restrict__ scl, int idx_w,
    long sq, long sk, long sv, long sa, long sb) {
  static_assert(K == 128 && BV == 32 && (SPLIT == 2 || SPLIT == 4), "K=128, BV=32, SPLIT 2|4");
  constexpr unsigned F = 0xffffffffu;
  constexpr int RW = 8 / SPLIT, TM = FA2_REC_TMAX, NW = 4 * SPLIT;
  if (sq == 0) sq = (long)H * K;
  if (sk == 0) sk = (long)H * K;
  if (sv == 0) sv = (long)HV * V;
  if (sa == 0) sa = HV;
  if (sb == 0) sb = HV;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int grp = warp & 3, part = warp >> 2;          // строки варпа: grp + 4*(part*RW + j)
  const int i_v = blockIdx.x, i_hv = blockIdx.y, i_n = blockIdx.z;
  const int i_h = i_hv / (HV / H);
  long bos = (long)i_n * T;
  int Tn = T;
  if (cu_seqlens != nullptr) {
    bos = (long)cu_seqlens[i_n];
    Tn = (int)((long)cu_seqlens[i_n + 1] - bos);
  }
  if (Tn <= 0) return;

  __shared__ __align__(16) uint2 s_qk[TM][2][32];
  __shared__ __align__(16) TI s_v[TM][BV];
  __shared__ float s_tok[TM][4];                         // rq, rk, eg, bb
  __shared__ long s_slot[TM];
  __shared__ float s_am[2][NW];
  {
    const TI* pq0 = q + bos * sq + (long)i_h * K + 4 * lane;
    const TI* pk0 = k + bos * sk + (long)i_h * K + 4 * lane;
    const TI* pv0 = v + bos * sv + (long)i_hv * V + i_v * BV;
    for (int it = warp; it < Tn; it += NW) {
      asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::
                   "r"((unsigned)__cvta_generic_to_shared(&s_qk[it][0][lane])), "l"(pq0 + (long)it * sq));
      asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::
                   "r"((unsigned)__cvta_generic_to_shared(&s_qk[it][1][lane])), "l"(pk0 + (long)it * sk));
      if (lane < BV * (int)sizeof(TI) / 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                     "r"((unsigned)__cvta_generic_to_shared(&s_v[it][lane * (16 / sizeof(TI))])),
                     "l"(pv0 + (long)it * sv + lane * (16 / sizeof(TI))));
    }
    asm volatile("cp.async.commit_group;\n" ::);
  }
  long slot_rd = bos;
  if (ssm_idx != nullptr) {
    const int it0 = SPEC ? (n_acc[i_n] - 1) : 0;
    slot_rd = ssm_idx[(long)i_n * stride_idx + it0];
    if (slot_rd < 0) return;                              // у всего блока один слот -> выходят все разом
  }
  const long hoff = ((long)i_hv * V + i_v * BV + grp + 4 * part * RW) * K + 4 * lane;
  float hreg[RW][4];
  {
    const short* p_h0 = hpool + slot_rd * stride_slot + hoff;
    const float s_rd = scl != nullptr ? scl[((slot_rd * (long)HV + i_hv) * (V / BV) + i_v) * 4 + grp] : 1.f;
#pragma unroll
    for (int j = 0; j < RW; ++j) {
      const short4 qq = *reinterpret_cast<const short4*>(p_h0 + j * 4 * K);
      hreg[j][0] = rec_i16_ld(qq.x, s_rd); hreg[j][1] = rec_i16_ld(qq.y, s_rd);
      hreg[j][2] = rec_i16_ld(qq.z, s_rd); hreg[j][3] = rec_i16_ld(qq.w, s_rd);
    }
  }
  if (warp == 0 && lane < Tn) {
    const float dtb = dt_bias[i_hv], eAl = __expf(A_log[i_hv]);
    const float a_t = rec_ldf<TI>(a_in + (bos + lane) * sa + i_hv);
    const float b_t = rec_ldf<TI>(beta + (bos + lane) * sb + i_hv);
    const float x = a_t + dtb;
    const float sp = (x <= 20.f) ? log1pf(__expf(x)) : x;
    s_tok[lane][2] = __expf(-eAl * sp);
    s_tok[lane][3] = 1.f / (1.f + __expf(-b_t));
    long sl;
    if (ssm_idx_out != nullptr) sl = ssm_idx_out[(long)i_n];
    else if (ssm_idx != nullptr) sl = (idx_w > 0 && lane >= idx_w) ? -1 : (long)ssm_idx[(long)i_n * stride_idx + lane];
    else sl = bos + lane;
    s_slot[lane] = sl;
  }
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
  if constexpr (L2N) {
    for (int it = warp; it < Tn; it += NW) {
      const uint2 rq = s_qk[it][0][lane], rk = s_qk[it][1][lane];
      const TI* hq = reinterpret_cast<const TI*>(&rq);
      const TI* hk = reinterpret_cast<const TI*>(&rk);
      float qf[4], kf[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) { qf[m] = rec_ldf<TI>(&hq[m]); kf[m] = rec_ldf<TI>(&hk[m]); }
      const float sqv = qf[0] * qf[0] + qf[1] * qf[1] + qf[2] * qf[2] + qf[3] * qf[3];
      const float skv = kf[0] * kf[0] + kf[1] * kf[1] + kf[2] * kf[2] + kf[3] * kf[3];
      const bool b4 = lane & 16;
      float w = (b4 ? skv : sqv) + __shfl_xor_sync(F, b4 ? sqv : skv, 16);
      w += __shfl_xor_sync(F, w, 8);
      w += __shfl_xor_sync(F, w, 4);
      w += __shfl_xor_sync(F, w, 2);
      w += __shfl_xor_sync(F, w, 1);
      const float other = __shfl_xor_sync(F, w, 16);
      if (lane == 0) {
        s_tok[it][0] = rsqrtf((b4 ? other : w) + 1e-6f);
        s_tok[it][1] = rsqrtf((b4 ? w : other) + 1e-6f);
      }
    }
    __syncthreads();
  }
  const int my_row = rs_row<RW>(lane);
  const int my_vrow = grp + 4 * (part * RW + my_row);
  TI* p_o = o + (bos * HV + i_hv) * V + i_v * BV + my_vrow;
  float* p_scl = scl != nullptr ? scl + (long)i_hv * (V / BV) * 4 + i_v * 4 + grp : nullptr;

  auto шаг = [&](const int it) {
    float qf[4], kf[4];
    {
      const uint2 rq = s_qk[it][0][lane], rk = s_qk[it][1][lane];
      const TI* hq = reinterpret_cast<const TI*>(&rq);
      const TI* hk = reinterpret_cast<const TI*>(&rk);
#pragma unroll
      for (int m = 0; m < 4; ++m) { qf[m] = rec_ldf<TI>(&hq[m]); kf[m] = rec_ldf<TI>(&hk[m]); }
    }
    const float4 tk = *reinterpret_cast<const float4*>(&s_tok[it][0]);
    if constexpr (L2N) {
#pragma unroll
      for (int c = 0; c < 4; ++c) { qf[c] *= tk.x; kf[c] *= tk.y; }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) qf[c] *= scale;
    const float eg = tk.z, bb = tk.w;
    float ps[RW];
#pragma unroll
    for (int j = 0; j < RW; ++j) {
      float s = 0.f;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        hreg[j][c] *= eg;
        s += hreg[j][c] * kf[c];
      }
      ps[j] = s;
    }
    const float s_my = rs_sum<RW>(ps, lane);
    const float bv_my = (rec_ldf<TI>(&s_v[it][my_vrow]) - s_my) * bb;
    float po_p[RW];
#pragma unroll
    for (int j = 0; j < RW; ++j) {
      const float bv = __shfl_sync(F, bv_my, rs_src<RW>(j));
      float po = 0.f;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        hreg[j][c] += bv * kf[c];
        po += hreg[j][c] * qf[c];
      }
      po_p[j] = po;
    }
    const float po_my = rs_sum<RW>(po_p, lane);
    if (RW == 2 ? (lane & 7) == 0 : (lane & 3) == 0) *p_o = rec_stf<TI>(po_my);
    p_o += (long)HV * V;

    const long slot = s_slot[it];
    if (slot >= 0) {
      float a = 0.f;
#pragma unroll
      for (int j = 0; j < RW; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) a = fmaxf(a, fabsf(hreg[j][c]));
      a = __uint_as_float(redux_max_u32(__float_as_uint(a)));
      if (lane == 0) s_am[it & 1][warp] = a;
      asm volatile("bar.sync %0, %1;\n" :: "r"(1 + grp), "r"(32 * SPLIT) : "memory");
      a = s_am[it & 1][grp];
#pragma unroll
      for (int p2 = 1; p2 < SPLIT; ++p2) a = fmaxf(a, s_am[it & 1][grp + 4 * p2]);
      const float s_new = (a > 0.f) ? (a / 32766.f) : 1.f;
      const float inv_s = (a > 0.f) ? (32766.f / a) : 0.f;
      if (lane == 0 && part == 0 && p_scl != nullptr) p_scl[slot * (long)HV * (V / BV) * 4] = s_new;
      short* p_ht = hpool + slot * stride_slot + hoff;
#pragma unroll
      for (int j = 0; j < RW; ++j) {
        uint2 w2;
        w2.x = pack_i16x2_rn(hreg[j][0], hreg[j][1], inv_s);
        w2.y = pack_i16x2_rn(hreg[j][2], hreg[j][3], inv_s);
        *reinterpret_cast<uint2*>(p_ht + j * 4 * K) = w2;
      }
    }
  };
  if (Tn == 5) {
#pragma unroll
    for (int it = 0; it < 5; ++it) шаг(it);
  } else {
    for (int it = 0; it < Tn; ++it) шаг(it);
  }
}

}  // namespace gdn
}  // namespace fa2_sm70
