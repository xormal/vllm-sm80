// Тест WarpMma на sm_80 (sm80_warp_mma.h): ПОКРЫТИЕ (каждая ячейка плитки записана ровно один раз)
// и ЗНАЧЕНИЯ против fp32-эталона, для всех раскладок операндов и всех accumulate*-вариантов.
//   nvcc -O3 -std=c++17 -arch=sm_80 -I../../fa2_src/fmha_kernel -I../../fa2_src/fmha_kernel/gemm \
//        test_sm80_warp_mma.cu -o /tmp/t && /tmp/t
// Значение без покрытия ничего не доказывает (вольтовский урок: оператор, считавший четверть плитки,
// проходил сверку значений), поэтому счётчик ячеек проверяется первым.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <cuda_fp16.h>
#include "gemm/volta_warp_mma.h"

using namespace fa2_sm70;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

constexpr int MB = 2, NB = 2, TM = MB * 16, TN = NB * 16;

enum Mode { M_ACC = 0, M_SWZA = 1, M_PRE = 2, M_PART = 3, M_BI8 = 4, M_BI8PRE = 5, M_BI8INTO = 6, M_HOOK = 7, M_STORE = 8 };

// A: kKMajor -> [TM][lda] (k подряд); kMMajor -> [K][lda] (m подряд).
// B: kKMajor -> [TN][ldb] (k подряд); kNMajor -> [K][ldb] (n подряд). Байтовый B -- те же формы в байтах.
template <int K, BLayout BL, ALayout AL, int MODE, int FMT = 0, int SWZ = 0, int KSP = 1>
__global__ void kern(const __half* gA, const void* gB, int lda, int ldb, float* C, int* cnt, __half* H) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int lane = threadIdx.x;
  const int arows = (AL == ALayout::kKMajor) ? TM : K;
  const int brows = (BL == BLayout::kKMajor) ? TN : K;
  __half* sA = reinterpret_cast<__half*>(smem);
  unsigned char* sBraw = smem + ((arows * lda * 2 + 127) / 128) * 128;
  const bool bytesB = (MODE == M_BI8 || MODE == M_BI8PRE || MODE == M_BI8INTO);
  for (int i = lane; i < arows * lda; i += 32) sA[i] = gA[i];
  const int bbytes = brows * ldb * (bytesB ? 1 : 2);
  for (int i = lane; i < bbytes; i += 32) sBraw[i] = reinterpret_cast<const unsigned char*>(gB)[i];
  __syncwarp();
  const __half* sB = reinterpret_cast<const __half*>(sBraw);
  const int8_t* sB8 = reinterpret_cast<const int8_t*>(sBraw);
  WarpMma<MB, NB, BL, AL> w; w.clear();
  if constexpr (MODE == M_ACC) w.template accumulate<K>(sA, lda, 0, sB, ldb, 0, lane);
  if constexpr (MODE == M_SWZA) w.template accumulate_swzA<K>(sA, lda, 0, sB, ldb, 0, lane);
  if constexpr (MODE == M_PRE) {
    uint32_t fa[K / 4][MB][2];
    w.template preload_a<K>(fa, sA, lda, 0, lane);
    w.template accumulate_pre<K>(fa, sB, ldb, 0, lane);
  }
  if constexpr (MODE == M_PART) {
    constexpr int NPRE = 8;
    uint32_t fa[NPRE][MB][2];
    w.template preload_n<NPRE>(fa, sA, lda, 0, lane);
    w.template accumulate_part<K, NPRE>(fa, sA, lda, 0, sB, ldb, 0, lane);
  }
  if constexpr (MODE == M_BI8) w.template accumulate_bi8<K, FMT, SWZ, KSP>(sA, lda, 0, sB8, ldb, 0, lane);
  if constexpr (MODE == M_BI8PRE) {
    uint32_t fa[K / 4][MB][2];
    w.template preload_a<K>(fa, sA, lda, 0, lane);
    w.template accumulate_bi8_pre<K, FMT, KSP>(fa, sB8, ldb, 0, lane);
  }
  if constexpr (MODE == M_BI8INTO) w.template accumulate_bi8_into<K, FMT>(w.acc, sA, lda, 0, sB8, ldb, 0, lane);
  if constexpr (MODE == M_HOOK) {
    int calls = 0;
    w.template accumulate_hooked<K>(sA, lda, 0, sB, ldb, 0, lane, [&](int s) { calls += s + 1; });
    if (calls != (K / 4) * (K / 4 + 1) / 2) w.acc[0][0][0] = 1e30f;   // каждый шаг k4 позван ровно раз
  }
  if constexpr (MODE == M_STORE) {
    w.template accumulate<K>(sA, lda, 0, sB, ldb, 0, lane);
    w.store_half(H, TN, 0, 0, lane);
  }
  w.visit(lane, [&](float v, int r, int c) {
    C[r * TN + c] = v;
    atomicAdd(&cnt[r * TN + c], 1);
  });
}

static float h2f(__half h) { return __half2float(h); }
static uint16_t f2hbits(float f) { __half h = __float2half(f); uint16_t u; memcpy(&u, &h, 2); return u; }
static float hbits2f(uint16_t u) { __half h; memcpy(&h, &u, 2); return __half2float(h); }

// значение байта после разворота (то, что видит MMA)
static float decode_byte(uint8_t b, int fmt) {
  if (fmt == 1) return hbits2f((uint16_t)b << 8);                         // e5m2 = старший байт fp16
  if (fmt >= 6 && fmt <= 10) return 1024.f + (float)b;                    // смещённый байт
  if (fmt == 11) {                                                         // e4m3 /256 (expand_e4m3x4)
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v = (e == 0) ? (m / 8.f) * powf(2.f, -6.f) : (1.f + m / 8.f) * powf(2.f, (float)e - 7.f);
    return (s ? -v : v) / 256.f;
  }
  return 1024.f + (float)(uint8_t)(b ^ 0x80);                             // FMT 0: знаковый, 1152+v
}

static int g_fail = 0;

template <int K, BLayout BL, ALayout AL, int MODE, int FMT = 0, int SWZ = 0, int KSP = 1>
void run(const char* name, int lda, int ldb) {
  const bool bytesB = (MODE == M_BI8 || MODE == M_BI8PRE || MODE == M_BI8INTO);
  const int arows = (AL == ALayout::kKMajor) ? TM : K;
  const int brows = (BL == BLayout::kKMajor) ? TN : K;
  std::vector<__half> A(arows * lda);
  std::vector<uint16_t> Bh(brows * ldb);
  std::vector<uint8_t> B8(brows * ldb);
  srand(1234 + K + MODE * 7 + FMT * 13 + SWZ);
  for (auto& x : A) x = __float2half(((rand() % 2001) - 1000) / 1000.f);
  for (auto& x : Bh) x = f2hbits(((rand() % 2001) - 1000) / 1000.f);
  for (auto& x : B8) { x = (uint8_t)(rand() & 0xFF); if (FMT == 11 && ((x & 0x78) == 0x78)) x &= 0xF7; if (FMT == 1 && ((x & 0x7C) == 0x7C)) x &= 0xFB; }
  // логический доступ
  auto Aval = [&](int m, int k) { return h2f(AL == ALayout::kKMajor ? A[m * lda + k] : A[k * lda + m]); };
  auto Aswz = [&](int m, int k) {   // свизл A: физ. 8-half кусок = (k>>3) ^ (m & 15)
    const int pc = (((k >> 3) ^ (m & 15)) << 3) | (k & 7);
    return h2f(A[m * lda + pc]);
  };
  auto Bval = [&](int k, int n) -> float {
    if (bytesB) {
      if (BL == BLayout::kKMajor) {
        int off = k;
        if (SWZ) off ^= (((n >> 3) & 1) * 64);
        return decode_byte(B8[n * ldb + off], FMT);
      }
      return decode_byte(B8[k * ldb + n], FMT);
    }
    return hbits2f(BL == BLayout::kKMajor ? Bh[n * ldb + k] : Bh[k * ldb + n]);
  };
  std::vector<float> ref(TM * TN, 0.f);
  for (int m = 0; m < TM; ++m)
    for (int n = 0; n < TN; ++n) {
      double s = 0;
      for (int k = 0; k < K; ++k) s += (double)(MODE == M_SWZA ? Aswz(m, k) : Aval(m, k)) * Bval(k, n);
      ref[m * TN + n] = (float)s;
    }
  __half* dA; void* dB; float* dC; int* dc; __half* dH;
  CK(cudaMalloc(&dA, A.size() * 2)); CK(cudaMalloc(&dB, brows * ldb * 2)); CK(cudaMalloc(&dC, TM * TN * 4));
  CK(cudaMalloc(&dc, TM * TN * 4)); CK(cudaMalloc(&dH, TM * TN * 2));
  CK(cudaMemcpy(dA, A.data(), A.size() * 2, cudaMemcpyHostToDevice));
  if (bytesB) CK(cudaMemcpy(dB, B8.data(), B8.size(), cudaMemcpyHostToDevice));
  else CK(cudaMemcpy(dB, Bh.data(), Bh.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemset(dc, 0, TM * TN * 4)); CK(cudaMemset(dC, 0, TM * TN * 4));
  const int smem = ((arows * lda * 2 + 127) / 128) * 128 + brows * ldb * 2 + 128;
  auto kp = kern<K, BL, AL, MODE, FMT, SWZ, KSP>;
  CK(cudaFuncSetAttribute(kp, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  kp<<<1, 32, smem>>>(dA, dB, lda, ldb, dC, dc, dH);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<float> C(TM * TN); std::vector<int> cnt(TM * TN); std::vector<__half> H(TM * TN);
  CK(cudaMemcpy(C.data(), dC, TM * TN * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(cnt.data(), dc, TM * TN * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(H.data(), dH, TM * TN * 2, cudaMemcpyDeviceToHost));
  int badcov = 0; double maxrel = 0, num = 0, den = 0;
  for (int i = 0; i < TM * TN; ++i) {
    if (cnt[i] != 1) ++badcov;
    const double d = C[i] - ref[i];
    num += d * d; den += (double)ref[i] * ref[i];
    maxrel = fmax(maxrel, fabs(d) / (fabs(ref[i]) + 1e-2 * (1 + fabs(ref[i]))));
  }
  double rel = sqrt(num / (den + 1e-30));
  double hrel = 0;
  if (MODE == M_STORE) for (int i = 0; i < TM * TN; ++i) hrel = fmax(hrel, fabs(h2f(H[i]) - C[i]) / (fabs(C[i]) + 1e-3));
  const bool ok = badcov == 0 && rel < 2e-5 && hrel < 1e-3;
  printf("%-44s K=%3d lda=%3d ldb=%3d  cover_bad=%d relL2=%.2e %s\n", name, K, lda, ldb, badcov, rel, ok ? "OK" : "FAIL");
  if (!ok) ++g_fail;
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dc); cudaFree(dH);
}

int main() {
  using BK = BLayout; using AK = ALayout;
  // fp16 x fp16, все четыре ориентации, 32-кусок (K%32==0) и 16-кусок (K=48, K=16)
  run<64, BK::kKMajor, AK::kKMajor, M_ACC>("acc  A:k B:k  (perm32)", 64 + 8, 64 + 8);
  run<64, BK::kKMajor, AK::kKMajor, M_ACC>("acc  A:k B:k  (perm32, pad 32)", 64 + 32, 64 + 32);
  run<64, BK::kKMajor, AK::kKMajor, M_ACC>("acc  A:k B:k  (perm32, ld=D+4, 8B rows)", 64 + 4, 64 + 4);
  run<256, BK::kKMajor, AK::kKMajor, M_PRE>("preload+pre (perm32, lda=D+4, ldb=D+8)", 256 + 4, 256 + 8);
  run<64, BK::kKMajor, AK::kKMajor, M_PART>("preload_n+part (ld=D+4)", 64 + 4, 64 + 4);
  run<64, BK::kKMajor, AK::kKMajor, M_BI8, 8>("bi8 k FMT8 (lda=D+4, ldb%8=4)", 64 + 4, 64 + 4);
  run<48, BK::kKMajor, AK::kKMajor, M_ACC>("acc  A:k B:k  (perm16, K=48, ld%8=4)", 48 + 4, 48 + 4);
  run<16, BK::kKMajor, AK::kKMajor, M_ACC>("acc  A:k B:k  (K=16)", 16 + 8, 16 + 8);
  run<64, BK::kNMajor, AK::kKMajor, M_ACC>("acc  A:k B:n  (ldmatrix.trans)", 64 + 8, TN + 8);
  run<64, BK::kNMajor, AK::kKMajor, M_ACC>("acc  A:k B:n  (scalar, ldb%8=4)", 64 + 8, TN + 4);
  run<64, BK::kKMajor, AK::kMMajor, M_ACC>("acc  A:m B:k  (ldmatrix.trans A)", TM + 8, 64 + 8);
  run<64, BK::kKMajor, AK::kMMajor, M_ACC>("acc  A:m B:k  (scalar A)", TM + 4, 64 + 4);
  run<64, BK::kNMajor, AK::kMMajor, M_ACC>("acc  A:m B:n  (both trans)", TM + 8, TN + 8);
  run<32, BK::kNMajor, AK::kMMajor, M_ACC>("acc  A:m B:n  (both scalar)", TM + 4, TN + 4);
  // свизл A (lda -- степень двойки), с k- и n-мажорным B
  run<64, BK::kKMajor, AK::kKMajor, M_SWZA>("swzA A:k B:k", 128, 64 + 8);
  run<64, BK::kNMajor, AK::kKMajor, M_SWZA>("swzA A:k B:n", 128, TN + 8);
  // A в регистрах
  run<64, BK::kKMajor, AK::kKMajor, M_PRE>("preload_a + accumulate_pre (k,k)", 64 + 8, 64 + 8);
  run<64, BK::kNMajor, AK::kKMajor, M_PRE>("preload_a + accumulate_pre (k,n)", 64 + 8, TN + 8);
  run<48, BK::kKMajor, AK::kKMajor, M_PRE>("preload_a + accumulate_pre (perm16)", 48 + 4, 48 + 4);
  run<64, BK::kKMajor, AK::kKMajor, M_PART>("preload_n<8> + accumulate_part", 64 + 8, 64 + 8);
  run<64, BK::kKMajor, AK::kKMajor, M_HOOK>("accumulate_hooked", 64 + 8, 64 + 8);
  run<64, BK::kKMajor, AK::kKMajor, M_STORE>("store_half", 64 + 8, 64 + 8);
  // байтовый B
  run<64, BK::kKMajor, AK::kKMajor, M_BI8, 0>("bi8 k  FMT0 signed", 64 + 8, 64 + 16);
  run<64, BK::kKMajor, AK::kKMajor, M_BI8, 1>("bi8 k  FMT1 e5m2", 64 + 8, 64 + 16);
  run<64, BK::kKMajor, AK::kKMajor, M_BI8, 8>("bi8 k  FMT8 biased", 64 + 8, 64 + 16);
  run<64, BK::kKMajor, AK::kKMajor, M_BI8, 11>("bi8 k  FMT11 e4m3", 64 + 8, 64 + 16);
  run<128, BK::kKMajor, AK::kKMajor, M_BI8, 8, 1>("bi8 k  FMT8 SWZ=1 (K=128)", 128 + 8, 128 + 16);
  run<64, BK::kKMajor, AK::kKMajor, M_BI8, 8, 0, 2>("bi8 k  FMT8 KSP=2", 64 + 8, 64 + 16);
  run<48, BK::kKMajor, AK::kKMajor, M_BI8, 8>("bi8 k  FMT8 perm16 (K=48)", 48 + 4, 48 + 4);
  run<64, BK::kNMajor, AK::kKMajor, M_BI8, 8>("bi8 n  FMT8 (P*V)", 64 + 8, TN + 4);
  run<64, BK::kNMajor, AK::kKMajor, M_BI8, 11>("bi8 n  FMT11 e4m3 (P*V)", 64 + 8, TN + 4);
  run<64, BK::kKMajor, AK::kKMajor, M_BI8PRE, 8>("bi8_pre FMT8", 64 + 8, 64 + 16);
  run<64, BK::kKMajor, AK::kKMajor, M_BI8PRE, 0, 0, 2>("bi8_pre FMT0 KSP=2", 64 + 8, 64 + 16);
  run<64, BK::kNMajor, AK::kKMajor, M_BI8INTO, 8>("bi8_into FMT8", 64 + 8, TN + 4);
  printf(g_fail ? "\n%d FAILED\n" : "\nALL OK\n", g_fail);
  return g_fail ? 1 : 0;
}
