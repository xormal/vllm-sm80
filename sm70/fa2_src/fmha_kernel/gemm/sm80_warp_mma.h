// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// WarpMma ДЛЯ AMPERE (sm_80): тот же интерфейс, что у вольтовского оператора, атом -- mma.m16n8k16.
//
// ПОЧЕМУ ЭТО НАДО БЫЛО ПИСАТЬ, А НЕ ПЕРЕСОБИРАТЬ. В volta_warp_mma.h выдача HMMA.884 стоит под
// `__CUDA_ARCH__ < 800`, иначе тело пустое: собранное под sm_80 ядро на этом операторе НЕ эмулирует
// умножение, оно его не делает вовсе -- аккумулятор остаётся нулём, ответ молча неверен. Это хуже
// пути cutlass Sm70 (тот хотя бы падает в HFMA2 и считает верно, записка 17).
//
// ЧТО СОХРАНЕНО. Плитка варпа 16*MB x 16*NB, аккумулятор acc[MB][NB][8] на плитку 16x16 (на sm_80 это
// две инструкции m16n8k16 по 4 float -- ровно те же 8 регистров), все accumulate*-варианты, visit/
// store_half. Меняется КАРТА аккумулятора (wmma_acc_row/col, они тоже переключаются по архитектуре)
// и маски межполосной редукции строки: на sm_80 строка = lane>>2, столбцы = lane&3, поэтому строку
// сводят шаффлы по 1 и 2 (на Volta -- по 2 и 4). Все ядра, где карта зашита руками, обязаны брать её
// из wmma_acc_row/col и wmma_row_shfl, а не из литералов.
//
// ГЛАВНЫЙ ХОД -- ПЕРЕСТАНОВКА k (наследник вольтовского "смещение в ранг-1 бесплатно").
// C = sum_k A[m,k] B[k,n] не зависит от ПОРЯДКА k, если A и B переставлены ОДИНАКОВО. Штатная карта
// m16n8k16 требует от нити k = {2t,2t+1} и {2t+8,2t+9} -- два куска по 4 байта через 16 байт, то есть
// четыре LDS.32 на фрагмент A или ldmatrix с её требованием выравнивания строк на 16 Б (паддинги
// вольтовских ядер вида LD = D+4 его ломают). Переставив k так, что нить t владеет ПОДРЯД идущими
// [8t, 8t+8) в каждом 32-куске (шаг 0 -- [8t,8t+4), шаг 1 -- [8t+4,8t+8)), получаем ОДНУ LDS.128 на
// строку A и столбец B на ДВА шага k16, при любом шаге строки, кратном 8 half. Байтовый операнд
// (int8/e4m3/e5m2 KV) -- одна LDS.64 на два шага, и разворот expand_* даёт пары ровно в нужном
// порядке (f01 = k, k+1; f23 = k+2, k+3). Шаг строки, кратный только 4 half, -- запасной путь
// k16 (LDS.64 на шаг).
//
// n-МАЖОРНЫЙ B ([K][N]) и m-МАЖОРНЫЙ A ([K][M]) так не взять: соседние k лежат в разных строках.
// Для них штатная карта и ldmatrix.trans (одна инструкция на плитку 16x16), если строки выровнены на
// 16 Б, иначе поэлементные LDS.U16 (верно, медленно). Байтовый n-мажорный B (V в P*V на байтовом
// KV) -- четыре LDS.U8 на столбец и шаг: верно, но медленно; горячее ядро должно класть V в
// разделяемую транспонированной (тогда это k-мажорный байтовый B, LDS.64 на два шага).
//
// Проверка -- tests/sm80/test_sm80_warp_mma.cu: ПОКРЫТИЕ (каждая ячейка записана ровно один раз) и
// значения против fp32 на всех раскладках и всех accumulate*-вариантах.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace fa2_sm70 {
namespace sm80 {

__device__ __forceinline__ void mma16816(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  // НЕ volatile: это арифметика, переупорядочивание с загрузками законно и нужно (как у HMMA.884).
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ uint32_t pack_h2(__half lo, __half hi) {
  return (uint32_t)__half_as_ushort(lo) | ((uint32_t)__half_as_ushort(hi) << 16);
}
__device__ __forceinline__ uint32_t pack_b4(uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3) {
  return (b0 & 0xFFu) | ((b1 & 0xFFu) << 8) | ((b2 & 0xFFu) << 16) | ((b3 & 0xFFu) << 24);
}
__device__ __forceinline__ bool al16(const void* p, int ld_halfs) {
  return ((reinterpret_cast<uintptr_t>(p) & 15u) == 0) && ((ld_halfs & 7) == 0);
}
// Восемь подряд идущих half: одной LDS.128, если строка выровнена на 16 Б, иначе двумя LDS.64.
// Карта перестановки k от этого НЕ зависит (её решает только K), меняется лишь ширина загрузки --
// вольтовские ядра держат шаг строки D+4 (закон банков Volta), и строки там выровнены лишь на 8 Б.
__device__ __forceinline__ uint4 ld8h(const __half* p, bool a16) {
  if (a16) return *reinterpret_cast<const uint4*>(p);
  const uint2 lo = *reinterpret_cast<const uint2*>(p), hi = *reinterpret_cast<const uint2*>(p + 4);
  return make_uint4(lo.x, lo.y, hi.x, hi.y);
}
// Восемь подряд идущих байт: LDS.64 при выравнивании на 8 Б, иначе две LDS.32.
__device__ __forceinline__ uint2 ld8b(const int8_t* p, bool a8) {
  if (a8) return *reinterpret_cast<const uint2*>(p);
  return make_uint2(*reinterpret_cast<const uint32_t*>(p), *reinterpret_cast<const uint32_t*>(p + 4));
}

}  // namespace sm80

template <int MB, int NB, BLayout BL, ALayout AL = ALayout::kKMajor>
struct WarpMma {
  static constexpr int kM = MB * 16, kN = NB * 16;
  float acc[MB][NB][8];

  // Режим перестановки k: 32-куски при K % 32 == 0, иначе 16-куски. Решение зависит ТОЛЬКО от K,
  // поэтому preload_a<K> и accumulate*_<K> с одним K всегда согласны.
  template <int K> static constexpr bool kP32 = (K % 32 == 0);

  __device__ __forceinline__ void clear() {
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int r = 0; r < 8; ++r) acc[i][j][r] = 0.f;
  }

  // ---------------------------------------------------------------- атом: плитка 16x16x16
  // a[4] -- фрагмент A (16 строк), b[4] -- два n8-фрагмента B: {b0,b1} столбцы 0..7, {b2,b3} 8..15.
  __device__ __forceinline__ static void mma16(float (&d)[8], const uint32_t (&a)[4], const uint32_t (&b)[4]) {
    sm80::mma16816(d + 0, a[0], a[1], a[2], a[3], b[0], b[1]);
    sm80::mma16816(d + 4, a[0], a[1], a[2], a[3], b[2], b[3]);
  }
  __device__ __forceinline__ void mma_f(const uint32_t (&fa)[MB][4], const uint32_t (&fb)[NB][4]) {
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j) mma16(acc[i][j], fa[i], fb[j]);
  }
  __device__ __forceinline__ static void mma_f_into(float (&dst)[MB][NB][8], const uint32_t (&fa)[MB][4],
                                                    const uint32_t (&fb)[NB][4]) {
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j) mma16(dst[i][j], fa[i], fb[j]);
  }

  // ================================================================ ПЕРЕСТАВЛЕННАЯ карта (k-major x k-major)
  // 32-кусок: нить t берёт [kc+8t, kc+8t+8) у строки g и g+8 (A) / столбца g и g+8 (B), шаг s -- половина s.
  // `swz` -- преобразование ЛОГИЧЕСКОГО смещения k (в half) в физическое (свизл вызывающего), по умолчанию 0.
  __device__ __forceinline__ static void a_p32(uint32_t (&f0)[MB][4], uint32_t (&f1)[MB][4], const __half* pa,
                                               int lda, int koff, bool a16 = true) {
    // pa -- база строки g (уже с row0+g), koff -- физическое смещение куска нити (в half)
#pragma unroll
    for (int i = 0; i < MB; ++i) {
      const uint4 r0 = sm80::ld8h(pa + (long)(i * 16) * lda + koff, a16);
      const uint4 r1 = sm80::ld8h(pa + (long)(i * 16 + 8) * lda + koff, a16);
      f0[i][0] = r0.x; f0[i][1] = r1.x; f0[i][2] = r0.y; f0[i][3] = r1.y;
      f1[i][0] = r0.z; f1[i][1] = r1.z; f1[i][2] = r0.w; f1[i][3] = r1.w;
    }
  }
  __device__ __forceinline__ static void a_p16(uint32_t (&f)[MB][4], const __half* pa, int lda, int koff) {
#pragma unroll
    for (int i = 0; i < MB; ++i) {
      const uint2 r0 = *reinterpret_cast<const uint2*>(pa + (long)(i * 16) * lda + koff);
      const uint2 r1 = *reinterpret_cast<const uint2*>(pa + (long)(i * 16 + 8) * lda + koff);
      f[i][0] = r0.x; f[i][1] = r1.x; f[i][2] = r0.y; f[i][3] = r1.y;
    }
  }
  __device__ __forceinline__ static void b_p32(uint32_t (&f0)[NB][4], uint32_t (&f1)[NB][4], const __half* pb,
                                               int ldb, int koff, bool a16 = true) {
    // pb -- база столбца g ([N][K], k подряд)
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      const uint4 c0 = sm80::ld8h(pb + (long)(j * 16) * ldb + koff, a16);
      const uint4 c1 = sm80::ld8h(pb + (long)(j * 16 + 8) * ldb + koff, a16);
      f0[j][0] = c0.x; f0[j][1] = c0.y; f0[j][2] = c1.x; f0[j][3] = c1.y;
      f1[j][0] = c0.z; f1[j][1] = c0.w; f1[j][2] = c1.z; f1[j][3] = c1.w;
    }
  }
  __device__ __forceinline__ static void b_p16(uint32_t (&f)[NB][4], const __half* pb, int ldb, int koff) {
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      const uint2 c0 = *reinterpret_cast<const uint2*>(pb + (long)(j * 16) * ldb + koff);
      const uint2 c1 = *reinterpret_cast<const uint2*>(pb + (long)(j * 16 + 8) * ldb + koff);
      f[j][0] = c0.x; f[j][1] = c0.y; f[j][2] = c1.x; f[j][3] = c1.y;
    }
  }

  // ================================================================ ШТАТНАЯ карта (n-major B / m-major A)
  // A, k-мажорный, штатно: a0=(g,2t) a1=(g+8,2t) a2=(g,2t+8) a3=(g+8,2t+8).
  __device__ __forceinline__ static void a_std(uint32_t (&f)[MB][4], const __half* sA, int lda, int row0,
                                               int lane, int k0) {
    const int g = lane >> 2, t = lane & 3;
    if constexpr (AL == ALayout::kKMajor) {
#pragma unroll
      for (int i = 0; i < MB; ++i) {
        const __half* p = sA + (long)(row0 + i * 16 + g) * lda + k0 + 2 * t;
        f[i][0] = *reinterpret_cast<const uint32_t*>(p);
        f[i][1] = *reinterpret_cast<const uint32_t*>(p + 8L * lda);
        f[i][2] = *reinterpret_cast<const uint32_t*>(p + 8);
        f[i][3] = *reinterpret_cast<const uint32_t*>(p + 8L * lda + 8);
      }
    } else {
      // [K][M]: штатно a0 = (A[m=g][k=2t], A[g][2t+1]) = (S[2t][g], S[2t+1][g]).
      if (sm80::al16(sA, lda) && ((row0 & 7) == 0) && ((k0 & 7) == 0)) {
        const int mi = lane >> 3, rr = lane & 7;
#pragma unroll
        for (int i = 0; i < MB; ++i)
          sm80::ldsm_x4_t(f[i], sA + (long)(k0 + rr + (mi >> 1) * 8) * lda + row0 + i * 16 + (mi & 1) * 8);
      } else {
#pragma unroll
        for (int i = 0; i < MB; ++i) {
          const __half* p = sA + (long)(k0 + 2 * t) * lda + row0 + i * 16 + g;
          f[i][0] = sm80::pack_h2(p[0], p[lda]);
          f[i][1] = sm80::pack_h2(p[8], p[lda + 8]);
          f[i][2] = sm80::pack_h2(p[8L * lda], p[9L * lda]);
          f[i][3] = sm80::pack_h2(p[8L * lda + 8], p[9L * lda + 8]);
        }
      }
    }
  }
  // B, штатно: b0=(k 2t..2t+1, n g), b1=(k 2t+8.., n g); f[j] = {b0,b1 | n 0..7, b0,b1 | n 8..15}.
  __device__ __forceinline__ static void b_std(uint32_t (&f)[NB][4], const __half* sB, int ldb, int col0,
                                               int lane, int k0) {
    const int g = lane >> 2, t = lane & 3;
    if constexpr (BL == BLayout::kKMajor) {
#pragma unroll
      for (int j = 0; j < NB; ++j) {
        const __half* p = sB + (long)(col0 + j * 16 + g) * ldb + k0 + 2 * t;
        f[j][0] = *reinterpret_cast<const uint32_t*>(p);
        f[j][1] = *reinterpret_cast<const uint32_t*>(p + 8);
        f[j][2] = *reinterpret_cast<const uint32_t*>(p + 8L * ldb);
        f[j][3] = *reinterpret_cast<const uint32_t*>(p + 8L * ldb + 8);
      }
    } else {
      if (sm80::al16(sB, ldb) && ((col0 & 7) == 0) && ((k0 & 7) == 0)) {
        const int mi = lane >> 3, rr = lane & 7;
#pragma unroll
        for (int j = 0; j < NB; ++j)
          sm80::ldsm_x4_t(f[j], sB + (long)(k0 + rr + (mi & 1) * 8) * ldb + col0 + j * 16 + (mi >> 1) * 8);
      } else {
#pragma unroll
        for (int j = 0; j < NB; ++j) {
          const __half* p = sB + (long)(k0 + 2 * t) * ldb + col0 + j * 16 + g;
          f[j][0] = sm80::pack_h2(p[0], p[ldb]);
          f[j][1] = sm80::pack_h2(p[8L * ldb], p[9L * ldb]);
          f[j][2] = sm80::pack_h2(p[8], p[ldb + 8]);
          f[j][3] = sm80::pack_h2(p[8L * ldb + 8], p[9L * ldb + 8]);
        }
      }
    }
  }

  static constexpr bool kPerm = (AL == ALayout::kKMajor) && (BL == BLayout::kKMajor);

  // ---------------------------------------------------------------- базовый accumulate
  template <int K>
  __device__ __forceinline__ void accumulate(const __half* sA, int lda, int row0,
                                             const __half* sB, int ldb, int col0, int lane) {
    static_assert(K % 16 == 0, "sm_80: K кратно 16 (шаг m16n8k16)");
    const int g = lane >> 2, t = lane & 3;
    if constexpr (kPerm) {
      const __half* pa = sA + (long)(row0 + g) * lda;
      const __half* pb = sB + (long)(col0 + g) * ldb;
      if constexpr (kP32<K>) {
        const bool aA = sm80::al16(sA, lda), aB = sm80::al16(sB, ldb);
#pragma unroll
        for (int kc = 0; kc < K; kc += 32) {
          uint32_t a0[MB][4], a1[MB][4], b0[NB][4], b1[NB][4];
          a_p32(a0, a1, pa, lda, kc + 8 * t, aA);
          b_p32(b0, b1, pb, ldb, kc + 8 * t, aB);
          mma_f(a0, b0);
          mma_f(a1, b1);
        }
      } else {
#pragma unroll
        for (int kc = 0; kc < K; kc += 16) {
          uint32_t a[MB][4], b[NB][4];
          a_p16(a, pa, lda, kc + 4 * t);
          b_p16(b, pb, ldb, kc + 4 * t);
          mma_f(a, b);
        }
      }
    } else {
#pragma unroll
      for (int k0 = 0; k0 < K; k0 += 16) {
        uint32_t a[MB][4], b[NB][4];
        a_std(a, sA, lda, row0, lane, k0);
        b_std(b, sB, ldb, col0, lane, k0);
        mma_f(a, b);
      }
    }
  }

  template <int K, typename Hook>
  __device__ __forceinline__ void accumulate_hooked(const __half* sA, int lda, int row0,
                                                    const __half* sB, int ldb, int col0, int lane,
                                                    Hook&& hook) {
    static_assert(K % 16 == 0, "sm_80: K кратно 16");
#pragma unroll
    for (int k0 = 0; k0 < K; k0 += 16) {
#pragma unroll
      for (int q = 0; q < 4; ++q) hook((k0 >> 2) + q);     // те же номера шагов k4, что на Volta
      uint32_t a[MB][4], b[NB][4];
      if constexpr (kPerm) {
        const int g = lane >> 2, t = lane & 3;
        a_p16(a, sA + (long)(row0 + g) * lda, lda, k0 + 4 * t);
        b_p16(b, sB + (long)(col0 + g) * ldb, ldb, k0 + 4 * t);
      } else {
        a_std(a, sA, lda, row0, lane, k0);
        b_std(b, sB, ldb, col0, lane, k0);
      }
      mma_f(a, b);
    }
  }

  // ---------------------------------------------------------------- свизл A (k-major, 16-Б куски ^ (строка & 15))
  static __device__ __forceinline__ int swz(int c, int xr) { return (((c >> 3) ^ xr) << 3) | (c & 7); }

  template <int K>
  __device__ __forceinline__ void accumulate_swzA_dc(const __half* sA, int lda, int row0, int kbase,
                                                     const __half* sB, int ldb, int col0, int lane) {
    static_assert(AL == ALayout::kKMajor, "свизл A реализован для k-мажорной раскладки");
    static_assert(K % 16 == 0, "sm_80: K кратно 16");
    const int g = lane >> 2, t = lane & 3;
    const __half* pb = (BL == BLayout::kKMajor) ? sB + (long)(col0 + g) * ldb : sB;
    // XOR по строке: у строк g и g+8 разный (r & 15), у фрагментов i -- тот же (16 & 15 == 0).
    const int r0 = row0 + g, r1 = row0 + g + 8;
    const __half* pa0 = sA + (long)r0 * lda;
    const __half* pa1 = sA + (long)r1 * lda;
    const int x0 = r0 & 15, x1 = r1 & 15;
    if constexpr (BL == BLayout::kKMajor && kP32<K>) {
#pragma unroll
      for (int kc = 0; kc < K; kc += 32) {
        uint32_t a0[MB][4], a1[MB][4], b0[NB][4], b1[NB][4];
        const int c = kbase + kc + 8 * t;                     // логический k начала куска нити
#pragma unroll
        for (int i = 0; i < MB; ++i) {
          const uint4 v0 = *reinterpret_cast<const uint4*>(pa0 + (long)(i * 16) * lda + swz(c, x0));
          const uint4 v1 = *reinterpret_cast<const uint4*>(pa1 + (long)(i * 16) * lda + swz(c, x1));
          a0[i][0] = v0.x; a0[i][1] = v1.x; a0[i][2] = v0.y; a0[i][3] = v1.y;
          a1[i][0] = v0.z; a1[i][1] = v1.z; a1[i][2] = v0.w; a1[i][3] = v1.w;
        }
        b_p32(b0, b1, pb, ldb, kc + 8 * t, sm80::al16(sB, ldb));
        mma_f(a0, b0);
        mma_f(a1, b1);
      }
    } else if constexpr (BL == BLayout::kKMajor) {
#pragma unroll
      for (int kc = 0; kc < K; kc += 16) {
        uint32_t a[MB][4], b[NB][4];
        const int c = kbase + kc + 4 * t;
#pragma unroll
        for (int i = 0; i < MB; ++i) {
          const uint2 v0 = *reinterpret_cast<const uint2*>(pa0 + (long)(i * 16) * lda + swz(c, x0));
          const uint2 v1 = *reinterpret_cast<const uint2*>(pa1 + (long)(i * 16) * lda + swz(c, x1));
          a[i][0] = v0.x; a[i][1] = v1.x; a[i][2] = v0.y; a[i][3] = v1.y;
        }
        b_p16(b, pb, ldb, kc + 4 * t);
        mma_f(a, b);
      }
    } else {
      // n-мажорный B: штатная карта; A из свизла по 32-битным словам.
#pragma unroll
      for (int k0 = 0; k0 < K; k0 += 16) {
        uint32_t a[MB][4], b[NB][4];
        const int c0 = kbase + k0 + 2 * t, c1 = c0 + 8;
#pragma unroll
        for (int i = 0; i < MB; ++i) {
          a[i][0] = *reinterpret_cast<const uint32_t*>(pa0 + (long)(i * 16) * lda + swz(c0, x0));
          a[i][1] = *reinterpret_cast<const uint32_t*>(pa1 + (long)(i * 16) * lda + swz(c0, x1));
          a[i][2] = *reinterpret_cast<const uint32_t*>(pa0 + (long)(i * 16) * lda + swz(c1, x0));
          a[i][3] = *reinterpret_cast<const uint32_t*>(pa1 + (long)(i * 16) * lda + swz(c1, x1));
        }
        b_std(b, sB, ldb, col0, lane, k0);
        mma_f(a, b);
      }
    }
  }
  template <int K>
  __device__ __forceinline__ void accumulate_swzA(const __half* sA, int lda, int row0,
                                                  const __half* sB, int ldb, int col0, int lane) {
    accumulate_swzA_dc<K>(sA, lda, row0, 0, sB, ldb, col0, lane);
  }
  // A m-мажорный со свизлом (фаза dQ): строки -- k, XOR зависит от k. Штатная карта, поэлементно.
  template <int K>
  __device__ __forceinline__ void accumulate_swzA_mmajor(const __half* sA, int lda, int row0,
                                                         const __half* sB, int ldb, int col0, int lane) {
    static_assert(AL == ALayout::kMMajor, "эта ветка для m-мажорной раскладки A");
    static_assert(K % 16 == 0, "sm_80: K кратно 16");
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int k0 = 0; k0 < K; k0 += 16) {
      uint32_t a[MB][4], b[NB][4];
      auto S = [&](int k, int m) { return sA[(long)k * lda + swz(m, k & 15)]; };
#pragma unroll
      for (int i = 0; i < MB; ++i) {
        const int m = row0 + i * 16 + g, k = k0 + 2 * t;
        a[i][0] = sm80::pack_h2(S(k, m), S(k + 1, m));
        a[i][1] = sm80::pack_h2(S(k, m + 8), S(k + 1, m + 8));
        a[i][2] = sm80::pack_h2(S(k + 8, m), S(k + 9, m));
        a[i][3] = sm80::pack_h2(S(k + 8, m + 8), S(k + 9, m + 8));
      }
      b_std(b, sB, ldb, col0, lane, k0);
      mma_f(a, b);
    }
  }

  // ---------------------------------------------------------------- A в регистрах
  // Хранилище -- вольтовская форма fa[K/4][MB][2]; на sm_80 шагу k16 номер S нужно 4 регистра на MB:
  // fa[4S + (q>>1)][i][q&1]. Вторая половина хранилища не используется (компилятор её выбросит):
  // фрагмент A на sm_80 вдвое меньше вольтовского -- там четыре пары четвёрок дублировали строки.
  template <int K>
  __device__ __forceinline__ static void put_a(uint32_t (&fa)[K / 4][MB][2], int S, const uint32_t (&f)[MB][4]) {
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int q = 0; q < 4; ++q) fa[4 * S + (q >> 1)][i][q & 1] = f[i][q];
  }
  template <int KS>
  __device__ __forceinline__ static void get_a(const uint32_t (&fa)[KS][MB][2], int S, uint32_t (&f)[MB][4]) {
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int q = 0; q < 4; ++q) f[i][q] = fa[4 * S + (q >> 1)][i][q & 1];
  }
  template <int K>
  __device__ __forceinline__ void preload_a(uint32_t (&fa)[K / 4][MB][2], const __half* sA, int lda,
                                            int row0, int lane) const {
    static_assert(K % 16 == 0, "sm_80: K кратно 16");
    if constexpr (kPerm) {
      const int g = lane >> 2, t = lane & 3;
      const __half* pa = sA + (long)(row0 + g) * lda;
      if constexpr (kP32<K>) {
        const bool aA = sm80::al16(sA, lda);
#pragma unroll
        for (int kc = 0; kc < K; kc += 32) {
          uint32_t a0[MB][4], a1[MB][4];
          a_p32(a0, a1, pa, lda, kc + 8 * t, aA);
          put_a<K>(fa, kc / 16, a0);
          put_a<K>(fa, kc / 16 + 1, a1);
        }
      } else {
#pragma unroll
        for (int kc = 0; kc < K; kc += 16) {
          uint32_t a[MB][4];
          a_p16(a, pa, lda, kc + 4 * t);
          put_a<K>(fa, kc / 16, a);
        }
      }
    } else {
#pragma unroll
      for (int k0 = 0; k0 < K; k0 += 16) {
        uint32_t a[MB][4];
        a_std(a, sA, lda, row0, lane, k0);
        put_a<K>(fa, k0 / 16, a);
      }
    }
  }
  template <int K>
  __device__ __forceinline__ void accumulate_pre(const uint32_t (&fa)[K / 4][MB][2],
                                                 const __half* sB, int ldb, int col0, int lane) {
    static_assert(K % 16 == 0, "sm_80: K кратно 16");
    if constexpr (kPerm) {
      const int g = lane >> 2, t = lane & 3;
      const __half* pb = sB + (long)(col0 + g) * ldb;
      if constexpr (kP32<K>) {
#pragma unroll
        for (int kc = 0; kc < K; kc += 32) {
          uint32_t a0[MB][4], a1[MB][4], b0[NB][4], b1[NB][4];
          get_a(fa, kc / 16, a0); get_a(fa, kc / 16 + 1, a1);
          b_p32(b0, b1, pb, ldb, kc + 8 * t, sm80::al16(sB, ldb));
          mma_f(a0, b0); mma_f(a1, b1);
        }
      } else {
#pragma unroll
        for (int kc = 0; kc < K; kc += 16) {
          uint32_t a[MB][4], b[NB][4];
          get_a(fa, kc / 16, a);
          b_p16(b, pb, ldb, kc + 4 * t);
          mma_f(a, b);
        }
      }
    } else {
#pragma unroll
      for (int k0 = 0; k0 < K; k0 += 16) {
        uint32_t a[MB][4], b[NB][4];
        get_a(fa, k0 / 16, a);
        b_std(b, sB, ldb, col0, lane, k0);
        mma_f(a, b);
      }
    }
  }
  // Частичная предзагрузка: NPRE -- в ВОЛЬТОВСКИХ шагах k4 (как у вызывающих). Карта совпадает с
  // accumulate_part<K,..> того же K только если граница NPRE*4 лежит на границе куска перестановки.
  template <int NPRE>
  __device__ __forceinline__ void preload_n(uint32_t (&fa)[NPRE ? NPRE : 1][MB][2], const __half* sA,
                                            int lda, int row0, int lane) const {
    static_assert(NPRE % 8 == 0, "sm_80: предзагрузка целыми 32-кусками k (NPRE кратно 8 шагам k4)");
    if constexpr (NPRE > 0) {
      const int g = lane >> 2, t = lane & 3;
      const __half* pa = sA + (long)(row0 + g) * lda;
#pragma unroll
      for (int kc = 0; kc < NPRE * 4; kc += 32) {
        uint32_t a0[MB][4], a1[MB][4];
        a_p32(a0, a1, pa, lda, kc + 8 * t, sm80::al16(sA, lda));
#pragma unroll
        for (int i = 0; i < MB; ++i)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            fa[4 * (kc / 16) + (q >> 1)][i][q & 1] = a0[i][q];
            fa[4 * (kc / 16 + 1) + (q >> 1)][i][q & 1] = a1[i][q];
          }
      }
    }
  }
  template <int K, int NPRE>
  __device__ __forceinline__ void accumulate_part(const uint32_t (&fa)[NPRE ? NPRE : 1][MB][2],
                                                  const __half* sA, int lda, int row0,
                                                  const __half* sB, int ldb, int col0, int lane) {
    static_assert(kPerm && kP32<K> && NPRE % 8 == 0, "accumulate_part на sm_80: k-major x k-major, K%32==0");
    const int g = lane >> 2, t = lane & 3;
    const __half* pa = sA + (long)(row0 + g) * lda;
    const __half* pb = sB + (long)(col0 + g) * ldb;
#pragma unroll
    for (int kc = 0; kc < K; kc += 32) {
      uint32_t a0[MB][4], a1[MB][4], b0[NB][4], b1[NB][4];
      if (kc < NPRE * 4) { get_a(fa, kc / 16, a0); get_a(fa, kc / 16 + 1, a1); }
      else               a_p32(a0, a1, pa, lda, kc + 8 * t, sm80::al16(sA, lda));
      b_p32(b0, b1, pb, ldb, kc + 8 * t, sm80::al16(sB, ldb));
      mma_f(a0, b0); mma_f(a1, b1);
    }
  }

  // ================================================================ ОДНОБАЙТОВЫЙ операнд B
  template <int FMT = 0>
  __device__ __forceinline__ static void expand_i8(uint32_t w, uint32_t& f0, uint32_t& f1) {
    if constexpr (FMT == 2) { w ^= fa2_sm70::prims::kFakeXorMask; fa2_sm70::prims::expand_e5m2x4(w, f0, f1); return; }
    if constexpr (FMT == 1 || FMT == 3 || FMT == 5) { fa2_sm70::prims::expand_e5m2x4(w, f0, f1); return; }
    if constexpr (FMT == 11) { fa2_sm70::prims::expand_e4m3x4(w, f0, f1); return; }
    if constexpr (FMT == 12) { fa2_sm70::prims::expand_e5m2x4(w, f0, f1); return; }
    if constexpr (FMT == 13) { fa2_sm70::prims::expand_e4m3x4_naive(w, f0, f1); return; }
    if constexpr (FMT >= 6 && FMT <= 10) { fa2_sm70::prims::expand_u8x4_biased(w, f0, f1); return; }
#if defined(FA2_BIASED_U8) && FA2_BIASED_U8
    fa2_sm70::prims::expand_u8x4_biased(w, f0, f1);
#else
    fa2_sm70::prims::expand_i8x4(w, f0, f1);
#endif
  }
  // k-мажорный байтовый B: столбец g, 8 байт [kc+8t, +8) = два шага; свизл SWZ (сдвиг куска 0/64 Б по
  // биту 3 строки) применяется к адресу ровно как на Volta: физ. = лог. ^ swj.
  template <int FMT>
  __device__ __forceinline__ static void bi8_p32(uint32_t (&f0)[NB][4], uint32_t (&f1)[NB][4],
                                                 const int8_t* const (&pbj)[NB][2], const int (&swj)[NB][2], int koff,
                                                 bool a8 = true) {
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      const uint2 c0 = sm80::ld8b(pbj[j][0] + (koff ^ swj[j][0]), a8);
      const uint2 c1 = sm80::ld8b(pbj[j][1] + (koff ^ swj[j][1]), a8);
      expand_i8<FMT>(c0.x, f0[j][0], f0[j][1]);
      expand_i8<FMT>(c1.x, f0[j][2], f0[j][3]);
      expand_i8<FMT>(c0.y, f1[j][0], f1[j][1]);
      expand_i8<FMT>(c1.y, f1[j][2], f1[j][3]);
    }
  }
  template <int FMT>
  __device__ __forceinline__ static void bi8_p16(uint32_t (&f)[NB][4], const int8_t* const (&pbj)[NB][2],
                                                 const int (&swj)[NB][2], int koff) {
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      expand_i8<FMT>(*reinterpret_cast<const uint32_t*>(pbj[j][0] + (koff ^ swj[j][0])), f[j][0], f[j][1]);
      expand_i8<FMT>(*reinterpret_cast<const uint32_t*>(pbj[j][1] + (koff ^ swj[j][1])), f[j][2], f[j][3]);
    }
  }
  template <int SWZ>
  __device__ __forceinline__ static void bi8_bases(const int8_t* sB, int ldb, int col0, int lane,
                                                   const int8_t* (&pbj)[NB][2], int (&swj)[NB][2]) {
    const int g = lane >> 2;
#pragma unroll
    for (int j = 0; j < NB; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int c = col0 + j * 16 + h * 8 + g;
        pbj[j][h] = sB + (long)c * ldb;
        swj[j][h] = SWZ ? (((c >> 3) & 1) * 64) : 0;
      }
  }
  // n-мажорный байтовый B ([K][N]): в переставленной карте нить берёт строки kbase+4t..+3 (шаг k16) у
  // столбца g и g+8 -- четыре LDS.U8 на столбец. Верно и медленно (см. шапку).
  // То же в ШТАТНОЙ карте k (2t,2t+1 | 2t+8,2t+9): ровно та, что у fp16 n-мажорного пути (ldmatrix.trans),
  // поэтому байтовый и fp16-маршруты P*V суммируют внутри MMA в ОДНОМ порядке и совпадают побитово
  // (как на Volta: test_vllm_i8_gather::test_e5m2_route_is_bit_exact).
  template <int FMT>
  __device__ __forceinline__ static void bi8_n_std(uint32_t (&f)[NB][4], const int8_t* sB, int ldb, int col0,
                                                   int lane, int k0) {
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int j = 0; j < NB; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(sB) + (long)(k0 + 2 * t) * ldb + col0 + j * 16 + h * 8 + g;
        const uint32_t w = sm80::pack_b4(p[0], p[ldb], p[8L * ldb], p[9L * ldb]);
        expand_i8<FMT>(w, f[j][2 * h], f[j][2 * h + 1]);
      }
  }
  template <int FMT>
  __device__ __forceinline__ static void bi8_n_p16(uint32_t (&f)[NB][4], const int8_t* sB, int ldb, int col0,
                                                   int lane, int krow) {
    const int g = lane >> 2;
#pragma unroll
    for (int j = 0; j < NB; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(sB) + (long)krow * ldb + col0 + j * 16 + h * 8 + g;
        const uint32_t w = sm80::pack_b4(p[0], p[ldb], p[2L * ldb], p[3L * ldb]);
        expand_i8<FMT>(w, f[j][2 * h], f[j][2 * h + 1]);
      }
  }

  template <int K, int FMT = 0, int SWZ = 0, int KSP = 1>
  __device__ __forceinline__ void accumulate_bi8(const __half* sA, int lda, int row0,
                                                 const int8_t* sB, int ldb, int col0, int lane) {
    static_assert(AL == ALayout::kKMajor, "байтовый B на sm_80: A k-мажорный");
    static_assert(K % 16 == 0, "sm_80: K кратно 16");
    const int g = lane >> 2, t = lane & 3;
    const __half* pa = sA + (long)(row0 + g) * lda;
    if constexpr (BL == BLayout::kKMajor) {
      const int8_t* pbj[NB][2]; int swj[NB][2];
      bi8_bases<SWZ>(sB, ldb, col0, lane, pbj, swj);
      float acc2[MB][NB][8];
      if constexpr (KSP == 2) {
#pragma unroll
        for (int i = 0; i < MB; ++i)
#pragma unroll
          for (int j = 0; j < NB; ++j)
#pragma unroll
            for (int r = 0; r < 8; ++r) acc2[i][j][r] = 0.f;
      }
      if constexpr (kP32<K>) {
#pragma unroll
        for (int kc = 0; kc < K; kc += 32) {
          uint32_t a0[MB][4], a1[MB][4], b0[NB][4], b1[NB][4];
          a_p32(a0, a1, pa, lda, kc + 8 * t, sm80::al16(sA, lda));
          bi8_p32<FMT>(b0, b1, pbj, swj, kc + 8 * t,
                       ((reinterpret_cast<uintptr_t>(sB) & 7u) == 0) && ((ldb & 7) == 0));
          mma_f(a0, b0);
          if constexpr (KSP == 2) mma_f_into(acc2, a1, b1); else mma_f(a1, b1);
        }
      } else {
#pragma unroll
        for (int kc = 0; kc < K; kc += 16) {
          uint32_t a[MB][4], b[NB][4];
          a_p16(a, pa, lda, kc + 4 * t);
          bi8_p16<FMT>(b, pbj, swj, kc + 4 * t);
          mma_f(a, b);
        }
      }
      if constexpr (KSP == 2) {
#pragma unroll
        for (int i = 0; i < MB; ++i)
#pragma unroll
          for (int j = 0; j < NB; ++j)
#pragma unroll
            for (int r = 0; r < 8; ++r) acc[i][j][r] += acc2[i][j][r];
      }
    } else {
#pragma unroll
      for (int kc = 0; kc < K; kc += 16) {
        uint32_t a[MB][4], b[NB][4];
        a_std(a, sA, lda, row0, lane, kc);
        bi8_n_std<FMT>(b, sB, ldb, col0, lane, kc);
        mma_f(a, b);
      }
    }
  }
  template <int K, int FMT = 0>
  __device__ __forceinline__ void accumulate_bi8_roll(const __half* sA, int lda, int row0,
                                                      const int8_t* sB, int ldb, int col0, int lane) {
    accumulate_bi8<K, FMT, 0, 1>(sA, lda, row0, sB, ldb, col0, lane);
  }
  template <int K, int FMT = 0, int KSP = 1>
  __device__ __forceinline__ void accumulate_bi8_pre(const uint32_t (&fa)[K / 4][MB][2],
                                                     const int8_t* sB, int ldb, int col0, int lane) {
    static_assert(BL == BLayout::kKMajor && AL == ALayout::kKMajor, "accumulate_bi8_pre: только k-мажорный B");
    static_assert(K % 16 == 0, "sm_80: K кратно 16");
    const int t = lane & 3;
    const int8_t* pbj[NB][2]; int swj[NB][2];
    bi8_bases<0>(sB, ldb, col0, lane, pbj, swj);
    float acc2[MB][NB][8];
    if constexpr (KSP == 2) {
#pragma unroll
      for (int i = 0; i < MB; ++i)
#pragma unroll
        for (int j = 0; j < NB; ++j)
#pragma unroll
          for (int r = 0; r < 8; ++r) acc2[i][j][r] = 0.f;
    }
    if constexpr (kP32<K>) {
#pragma unroll
      for (int kc = 0; kc < K; kc += 32) {
        uint32_t a0[MB][4], a1[MB][4], b0[NB][4], b1[NB][4];
        get_a(fa, kc / 16, a0); get_a(fa, kc / 16 + 1, a1);
        bi8_p32<FMT>(b0, b1, pbj, swj, kc + 8 * t,
                     ((reinterpret_cast<uintptr_t>(sB) & 7u) == 0) && ((ldb & 7) == 0));
        mma_f(a0, b0);
        if constexpr (KSP == 2) mma_f_into(acc2, a1, b1); else mma_f(a1, b1);
      }
    } else {
#pragma unroll
      for (int kc = 0; kc < K; kc += 16) {
        uint32_t a[MB][4], b[NB][4];
        get_a(fa, kc / 16, a);
        bi8_p16<FMT>(b, pbj, swj, kc + 4 * t);
        mma_f(a, b);
      }
    }
    if constexpr (KSP == 2) {
#pragma unroll
      for (int i = 0; i < MB; ++i)
#pragma unroll
        for (int j = 0; j < NB; ++j)
#pragma unroll
          for (int r = 0; r < 8; ++r) acc[i][j][r] += acc2[i][j][r];
    }
  }
  template <int K, int FMT = 0>
  __device__ __forceinline__ void accumulate_bi8_into(float (&dst)[MB][NB][8], const __half* sA, int lda, int row0,
                                                      const int8_t* sB, int ldb, int col0, int lane) {
    static_assert(BL == BLayout::kNMajor, "accumulate_bi8_into: только n-мажорный B (P*V)");
    static_assert(K % 16 == 0, "sm_80: K кратно 16");
    const int g = lane >> 2, t = lane & 3;
    const __half* pa = sA + (long)(row0 + g) * lda;
#pragma unroll
    for (int kc = 0; kc < K; kc += 16) {
      uint32_t a[MB][4], b[NB][4];
      a_std(a, sA, lda, row0, lane, kc);
      bi8_n_std<FMT>(b, sB, ldb, col0, lane, kc);
      mma_f_into(dst, a, b);
    }
  }

  // ================================================================ обход аккумулятора
  template <typename F>
  __device__ __forceinline__ void visit(int lane, F&& f) {
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int r = 0; r < 8; ++r)
          f(acc[i][j][r], i * 16 + wmma_acc_row(lane, r), j * 16 + wmma_acc_col(lane, r));
  }
  template <typename F>
  __device__ __forceinline__ void visit(int lane, F&& f) const {
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int r = 0; r < 8; ++r)
          f(acc[i][j][r], i * 16 + wmma_acc_row(lane, r), j * 16 + wmma_acc_col(lane, r));
  }
  __device__ __forceinline__ void store_half(__half* dst, int ld, int row0, int col0, int lane) const {
    // пары (c, c+1) соседние -- одна 32-битная запись на пару
#pragma unroll
    for (int i = 0; i < MB; ++i)
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int r = 0; r < 8; r += 2) {
          const int rr = row0 + i * 16 + wmma_acc_row(lane, r), cc = col0 + j * 16 + wmma_acc_col(lane, r);
          *reinterpret_cast<__half2*>(dst + (long)rr * ld + cc) = __floats2half2_rn(acc[i][j][r], acc[i][j][r + 1]);
        }
  }
  __device__ __forceinline__ void store_half_swz(__half* dst, int ld, int row0, int col0, int lane) const {
    visit(lane, [&](float v, int r, int c) {
      const int rr = row0 + r;
      dst[(long)rr * ld + swz(col0 + c, rr & 15)] = __float2half(v);
    });
  }
};

}  // namespace fa2_sm70
