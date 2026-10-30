// [окно 28] Деквант из раскладки Marlin (kU8 + нули, группа 128) -> W [N][K] fp16. ПОБИТНО как наш
// w8a16_sm80_dequant: (1024+q) - (1024+z) точно (HSUB), затем * s в fp16 (HMUL) -- по лучу пары те же операции.
// Раскладка (gptq_marlin_repack, 8 бит): вес [K/16, 4N] int32, плитка 16(k) x 64(n) = 256 слов подряд (плитки -- k
// снаружи, n внутри: плитка = kt * (N/64) + nt). Слово с номером 8*th + o (th 0..31, o 0..7) -- столбец 8*o + th/4
// плитки, строки 2*(th%4) + {0, 8, 1, 9} по байтам 0..3. Масштабы [K/128, N] fp16 -- перестановка внутри 64 столбцов:
// столбец 8*o + c лежит на месте 8*c + o; нули [K/128, N/4] -- те же места, 4 на слово с чередованием байтов {0,2,1,3}.
// Без torch: включается в marlin8_sm80.cu (рабочий путь) и в стенд вариантов.
#pragma once
#include <cuda_fp16.h>
#include <stdint.h>

namespace m8dq {

// v2 (окно 26, рабочее до окна 28): нить -- столбец плитки, 4 слова по 4 Б, запись 32 Б в свою строку W
// (на варп -- 32 строки W с шагом 2K Б: запись не объединяется в варпе).
__global__ void v2(const uint32_t* __restrict__ mq, const __half* __restrict__ ms, const uint32_t* __restrict__ mz,
                   __half* __restrict__ W, int N, int K) {
  const int n_tiles = N >> 6;
  const long total = (long)(K >> 4) * n_tiles * 64;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < total; i += (long)gridDim.x * blockDim.x) {
    const int cl = (int)(i & 63);
    const long tile = i >> 6;
    const int nt = (int)(tile % n_tiles), kt = (int)(tile / n_tiles);
    const int w = cl >> 4, h = (cl >> 3) & 1, c = cl & 7;
    const int n = nt * 64 + cl;
    const int g = (kt * 16) >> 7;
    const int v = n & 63, b = n >> 6;
    const int pidx = b * 64 + (v & 7) * 8 + (v >> 3);
    const __half sc = ms[(long)g * N + pidx];
    const int t4 = pidx & 3, r = t4 == 1 ? 2 : (t4 == 2 ? 1 : t4);
    const int z = (int)((mz[(long)g * (N >> 2) + (pidx >> 2)] >> (8 * r)) & 0xFFu);
    const __half zz = __float2half(1024.f + (float)z);
    const uint32_t* base = mq + tile * 256 + w * 2 + h;
    alignas(16) __half o[16];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t word = __ldg(base + (4 * c + j) * 8);
      __half q[4];
#pragma unroll
      for (int bb = 0; bb < 4; ++bb) {
        const __half hq = __ushort_as_half((unsigned short)(0x6400u | ((word >> (8 * bb)) & 0xFFu)));
        q[bb] = __hmul(__hsub(hq, zz), sc);
      }
      o[2 * j] = q[0];
      o[2 * j + 1] = q[2];
      o[2 * j + 8] = q[1];
      o[2 * j + 9] = q[3];
    }
    uint4* dst = reinterpret_cast<uint4*>(W + (long)n * K + kt * 16);
    dst[0] = *reinterpret_cast<const uint4*>(&o[0]);
    dst[1] = *reinterpret_cast<const uint4*>(&o[8]);
  }
}

__device__ __forceinline__ uint4 ld_stream_v4(const uint4* p) {
  // вес читается один раз: мимо L1, с подкачкой 256 Б в L2 (соседние плитки того же блока)
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ uint32_t hsub_hmul2(uint32_t q2, uint32_t zz2, uint32_t sc2) {
  // (q - zz) * s по лучу: HSUB2 (как HADD2 с минусом) и HMUL2, округление к ближайшему -- как скалярные __hsub/__hmul
  __half2 a = __hsub2(*reinterpret_cast<__half2*>(&q2), *reinterpret_cast<__half2*>(&zz2));
  a = __hmul2(a, *reinterpret_cast<__half2*>(&sc2));
  return *reinterpret_cast<uint32_t*>(&a);
}

// v4<KG>: блок -- 64 столбца (плитка n) x 128*KG k (KG групп масштаба по 8 плиток), 256 нитей.
//  1) нить -- в каждой группе 2 x 16 Б весов (плитки jt и jt+4, одно место q в плитке): варп читает 512 Б подряд;
//     4 слова = 4 столбца (o0..o0+3) одного c и одних строк j. Масштабы этих столбцов лежат подряд (8 Б), нули -- в
//     одном слове. Пары fp16 -- PRMT из байтов (0x6400 | b -- ровно 1024 + b), затем HSUB2 и HMUL2.
//  2) общая память S[столбец][k/2] (пары fp16), строка 64*KG + 4 слова; номер слова XOR (столбец >> 5) << 4 --
//     запись варпа (32 разных банка) и чтение 16-байтными кусками без конфликтов;
//  3) запись W: строка столбца -- 128*KG fp16 подряд (KG=1: варп пишет две строки по 256 Б, KG=2: одну 512 Б).
//  Смещения -- 32-битные без знака (N*K < 2^31 -- проверка на хосте): адреса одним IMAD.WIDE.U32.
template <int KG>
__global__ void __launch_bounds__(256) v4(const uint4* __restrict__ mq, const __half* __restrict__ ms,
                                          const uint32_t* __restrict__ mz, __half* __restrict__ W, int N, int K) {
  __shared__ alignas(16) uint32_t S[64][64 * KG + 4];
  const unsigned n_tiles = (unsigned)N >> 6, Nu = (unsigned)N, Ku = (unsigned)K;
  const unsigned nt = blockIdx.x, g0 = blockIdx.y * KG, t = threadIdx.x;
  const unsigned jt = t >> 6, q = t & 63;
  const unsigned th = q >> 1, o0 = (q & 1) << 2;
  const unsigned c = th >> 2, j = th & 3;
  uint4 wv[KG][2];
  uint2 sv[KG];
  uint32_t zw[KG];
#pragma unroll
  for (int g = 0; g < KG; ++g) {                                   // все загрузки -- до счёта
    const uint4* src = mq + (((g0 + g) * 8 + jt) * n_tiles + nt) * 64 + q;
    wv[g][0] = ld_stream_v4(src);
    wv[g][1] = ld_stream_v4(src + 4 * n_tiles * 64);              // плитка jt + 4
    sv[g] = __ldg(reinterpret_cast<const uint2*>(ms + (g0 + g) * Nu + nt * 64 + 8 * c + o0));
    zw[g] = __ldg(mz + (g0 + g) * (Nu >> 2) + nt * 16 + 2 * c + (o0 >> 2));
  }
  const unsigned sw = (o0 >> 2) << 4;                              // столбец 8*o + c >= 32  <=>  o0 == 4
#pragma unroll
  for (int g = 0; g < KG; ++g) {
    // по столбцу i (o = o0 + i): пары (zz, zz) и (s, s); байт нуля столбца i -- {0, 2, 1, 3}[i]
    const uint32_t zz[4] = {__byte_perm(zw[g], 0x64646464u, 0x4040), __byte_perm(zw[g], 0x64646464u, 0x4242),
                            __byte_perm(zw[g], 0x64646464u, 0x4141), __byte_perm(zw[g], 0x64646464u, 0x4343)};
    const uint32_t sc[4] = {__byte_perm(sv[g].x, 0, 0x1010), __byte_perm(sv[g].x, 0, 0x3232),
                            __byte_perm(sv[g].y, 0, 0x1010), __byte_perm(sv[g].y, 0, 0x3232)};
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const uint32_t ww[4] = {wv[g][p].x, wv[g][p].y, wv[g][p].z, wv[g][p].w};
      const unsigned k2 = g * 64 + (((jt + 4 * p) * 8 + j) ^ sw);   // слово пары строк (2j, 2j+1) плитки
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        uint32_t* row = S[8 * (o0 + i) + c];
        row[k2] = hsub_hmul2(__byte_perm(ww[i], 0x64646464u, 0x4240), zz[i], sc[i]);      // строки 2j, 2j+1
        row[k2 + 4] = hsub_hmul2(__byte_perm(ww[i], 0x64646464u, 0x4341), zz[i], sc[i]);  // строки 2j+8, 2j+9
      }
    }
  }
  __syncthreads();
  constexpr int CH = 16 * KG;                                      // 16-байтных кусков в строке
  constexpr int RP = 256 / CH;                                     // строк за проход
  const unsigned ch = t % CH;
  __half* dst0 = W + nt * 64 * Ku + g0 * 128 + ch * 8;
#pragma unroll
  for (int pp = 0; pp < 64 / RP; ++pp) {
    const unsigned r = t / CH + RP * pp;
    const uint4 val = *reinterpret_cast<const uint4*>(&S[r][(ch * 4) ^ ((r >> 5) << 4)]);
    *reinterpret_cast<uint4*>(dst0 + r * Ku) = val;
  }
}

}  // namespace m8dq
