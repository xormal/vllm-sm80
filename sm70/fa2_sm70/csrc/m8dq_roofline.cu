// [окно 29] Потолок декванта Marlin v4: идеальные потоковые ядра той же схемы трафика (чтение 1 Б, запись 2 Б на элемент)
// против v4 на тех же размерах. Без torch: nvcc -arch=sm_80 -O3 -I. m8dq_roofline.cu -o m8dq_roofline; ./m8dq_roofline
//   copy16  -- fp16 -> fp16 (1:1), uint4 на нить;
//   i8f16   -- int8 -> fp16 (1:2): 16 Б чтения -> 32 Б записи на нить, без перестановок (PRMT + HSUB2 как в v4);
//   fill16  -- только запись fp16;
//   v4<1>, v4<2> -- marlin8_dequant.cuh на случайных весах gate_up ранга TP2 (5120 x 17408).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "marlin8_dequant.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void copy16(const uint4* __restrict__ a, uint4* __restrict__ b, long n16) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n16; i += (long)gridDim.x * blockDim.x) b[i] = a[i];
}
__global__ void i8f16(const uint4* __restrict__ a, uint4* __restrict__ b, long n16) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n16; i += (long)gridDim.x * blockDim.x) {
    const uint4 w = m8dq::ld_stream_v4(a + i);
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
    uint32_t o[8];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      o[2 * k] = m8dq::hsub_hmul2(__byte_perm(ww[k], 0x64646464u, 0x4140), 0x64806480u, 0x3c003c00u);
      o[2 * k + 1] = m8dq::hsub_hmul2(__byte_perm(ww[k], 0x64646464u, 0x4342), 0x64806480u, 0x3c003c00u);
    }
    b[2 * i] = make_uint4(o[0], o[1], o[2], o[3]);
    b[2 * i + 1] = make_uint4(o[4], o[5], o[6], o[7]);
  }
}
__global__ void fill16(uint4* __restrict__ b, long n16) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n16; i += (long)gridDim.x * blockDim.x)
    b[i] = make_uint4(i, i, i, i);
}

template <class F> float timeit(F f, int reps = 20) {
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (int i = 0; i < 3; ++i) f();
  CK(cudaDeviceSynchronize());
  float best = 1e9f;
  for (int r = 0; r < 5; ++r) {
    CK(cudaEventRecord(e0));
    for (int i = 0; i < reps; ++i) f();
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    best = std::min(best, ms / reps);
  }
  return best * 1e3f;   // мкс
}

int main() {
  const int K = 5120, N = 17408;
  const long ne = (long)N * K;
  int sms; CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  uint8_t* q; __half* w; __half* src16; __half* ms; uint32_t* mz;
  CK(cudaMalloc(&q, ne)); CK(cudaMalloc(&w, ne * 2)); CK(cudaMalloc(&src16, ne * 2));
  CK(cudaMalloc(&ms, (ne / 128) * 2)); CK(cudaMalloc(&mz, (ne / 128)));
  CK(cudaMemset(q, 0x5a, ne)); CK(cudaMemset(src16, 0, ne * 2)); CK(cudaMemset(ms, 0x10, (ne / 128) * 2));
  CK(cudaMemset(mz, 0x80, ne / 128));
  printf("SM %d, gate_up %d x %d (%.0f МБ int8)\n", sms, K, N, ne / 1e6);
  for (int bpsm : {4, 8, 16, 32}) {
    const int grid = sms * bpsm;
    const float tc = timeit([&] { copy16<<<grid, 256>>>((const uint4*)src16, (uint4*)w, ne * 2 / 16); });
    const float ti = timeit([&] { i8f16<<<grid, 256>>>((const uint4*)q, (uint4*)w, ne / 16); });
    const float tf = timeit([&] { fill16<<<grid, 256>>>((uint4*)w, ne * 2 / 16); });
    printf("блоков/SM %2d: copy16 %.1f мкс %.0f ГБ/с | i8f16 %.1f мкс %.0f ГБ/с | fill16 %.1f мкс %.0f ГБ/с\n", bpsm,
           tc, ne * 4 / tc / 1e3, ti, ne * 3 / ti / 1e3, tf, ne * 2 / tf / 1e3);
  }
  const float t1 = timeit([&] { m8dq::v4<1><<<dim3(N / 64, K / 128), 256>>>((const uint4*)q, ms, mz, w, N, K); });
  const float t2 = timeit([&] { m8dq::v4<2><<<dim3(N / 64, K / 256), 256>>>((const uint4*)q, ms, mz, w, N, K); });
  printf("v4<1> %.1f мкс %.0f ГБ/с | v4<2> %.1f мкс %.0f ГБ/с\n", t1, ne * 3 / t1 / 1e3, t2, ne * 3 / t2 / 1e3);
  CK(cudaGetLastError());
  return 0;
}
