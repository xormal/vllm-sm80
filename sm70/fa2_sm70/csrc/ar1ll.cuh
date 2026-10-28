// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// AR1LL (25.10): сведение двух карт через закреплённую память хоста по протоколу «данные с номером в одной записи».
// AR1H (exch_i12.cu): данные -> fence.sys -> флаг куска -> ждать флаг соседа -> читать данные: две полные задержки PCIe
// сверх переноса. Здесь ячейка 16 Б = {3 пары half (12 Б), номер e}: запись одной st.volatile.v4 без fence, читатель
// опрашивает САМИ данные, пока номер ячейки != e. Объём x4/3. Стенд bench/sa/arll_test (M=5 x 5120, 2 карты):
// AR1H ~22 мкс, LL16 ~13 мкс. Держится на целостности 16 Б записи в памяти хоста (Intel: запись PCIe внутри строки
// кэша 64 Б атомарна для читателя) -- проверено стендом с побитной сверкой суммы.
//
// Сегмент LL -- ОТДЕЛЬНЫЙ от AR1H (иначе в ячейках могли бы лежать чужие данные, случайно равные номеру). Номер e --
// счётчик БЛОКА b (seq[b]), одинаковый у обоих рангов по построению (формы вызовов одни); чётность e&1 -- двойной буфер:
// ранг доходит до обмена e+2 только получив все ячейки соседа за e+1, а сосед пишет их, лишь закончив обмен e.
// Опрос: нить 0 блока ждёт ПОСЛЕДНЮЮ ячейку блока (одна строка PCIe на опрос, без затопления чтениями при перекосе
// рангов), затем каждая нить дочитывает свою ячейку (обычно сразу). Потолок ожидания по тактам -> `срыв`.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace ar1ll {

constexpr int THR = 256;

__device__ __forceinline__ void st_vol4(uint4* p, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ uint4 ld_vol4(const uint4* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t add2(uint32_t x, uint32_t y) {
  __half2 r = __hadd2(*reinterpret_cast<__half2*>(&x), *reinterpret_cast<__half2*>(&y));
  return *reinterpret_cast<uint32_t*>(&r);
}

// y, out -- n2 пар half (uint32); ячеек = ceil(n2/3); блок b -- ячейки [b*THR, (b+1)*THR).
__global__ __launch_bounds__(THR) void ar1ll_kernel(const uint32_t* __restrict__ y, uint32_t* __restrict__ out,
                                                    uint4* my_cells, const uint4* peer_cells, int* seq, int* срыв,
                                                    long n2, long slot_cells) {
  const int b = blockIdx.x;
  __shared__ int e_s;
  if (threadIdx.x == 0) e_s = seq[b] + 1;
  __syncthreads();
  const uint32_t e = (uint32_t)e_s;
  const long cells = (n2 + 2) / 3;
  const long q = (long)b * THR + threadIdx.x;
  const long par = (long)(e & 1) * slot_cells;
  const bool live = q < cells;
  uint32_t d0 = 0, d1 = 0, d2 = 0;
  if (live) {
    d0 = y[3 * q];
    if (3 * q + 1 < n2) d1 = y[3 * q + 1];
    if (3 * q + 2 < n2) d2 = y[3 * q + 2];
    st_vol4(my_cells + par + q, make_uint4(d0, d1, d2, e));
  }
  // привратник: последняя живая ячейка блока
  if (threadIdx.x == 0) {
    long last = (long)b * THR + THR - 1;
    if (last >= cells) last = cells - 1;
    const long t0 = clock64();
    while (ld_vol4(peer_cells + par + last).w != e) {
      if (clock64() - t0 > 2000000000L) { atomicExch(срыв, 3); break; }
    }
  }
  __syncthreads();
  if (live) {
    uint4 r = ld_vol4(peer_cells + par + q);
    const long t0 = clock64();
    while (r.w != e) {
      if (clock64() - t0 > 2000000000L) { atomicExch(срыв, 3); break; }
      r = ld_vol4(peer_cells + par + q);
    }
    out[3 * q] = add2(d0, r.x);
    if (3 * q + 1 < n2) out[3 * q + 1] = add2(d1, r.y);
    if (3 * q + 2 < n2) out[3 * q + 2] = add2(d2, r.z);
  }
  if (threadIdx.x == 0) seq[b] = (int)e;
}

}  // namespace ar1ll
