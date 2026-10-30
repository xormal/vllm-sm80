// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// [ФРИЗЫ 30.10] MARLIN В НАШЕМ СТЕКЕ: W8A16 GPTQ int8 АСИММЕТРИЧНЫЙ (kU8 + нули), fp16, группа 128.
//
// ПОЧЕМУ. Окно 23 (стенд scripts/marlin_w8_bench.py + marlin_w8_decode_bench.py, ncu, A100 PG199, ранг TP2):
//   * M 4-16 (декод без MTP-пачки): наш w8a16_sm80_mm = Marlin (~1.0-1.1 ТБ/с, упор в память);
//   * M 17-64 (8 чатов x MTP k=4 = 40 строк): наш маршрут (прямое ядро) 21.9 мс на проход модели, Marlin 13.3 мс --
//     у прямого 136 блоков = 0.71 волны и в 2.6 раза больше инструкций; Marlin -- сетка = числу SM, полосатая
//     раздача плиток с последовательным сведением по замкам, глубокий конвейер (167 КБ разделяемой), DRAM 75 %;
//   * M 128-1024 (куски Clef): Marlin 1.1-1.5x быстрее любого нашего маршрута (440: 77 против 96-102 мс на проход);
//   * M >= 2048: наш деквант + cuBLAS быстрее (2088: 305 против 327 мс; 4096: 501 против 627) -- остаётся нашим.
// Штатный Marlin vLLM для 8 бит умеет только симметричный u8b128, а модель -- асимметричный GPTQ (нули 21..231).
// Шаблон Marlin нули для kU8 поддерживает (has_zp), не хватает только экземпляров ядер -- они здесь (15 штук: те же
// конфигурации, что generate_kernels.py даёт для u8b128). Хост-часть -- marlin.cu vLLM (Apache-2.0, Neural Magic /
// Elias Frantar, IST-DASLab), перенесена как есть, кроме: act_order не нужен (desc_act=false), выбор ядра -- из
// своего списка, буферы (выход, fp32-сведение, замки) -- ЗАРАНЕЕ и на поток, без torch::empty на вызов: окно 23 --
// смешанный шаг делает ~3600 выделений через распределитель, этих на горячем пути быть не должно. Замки самоочищаются
// (как у штатного Marlin), захват в CUDA-граф законен.
//
// Формат: вес -- gptq_marlin_repack(qweight, 8 бит), масштабы -- marlin_permute_scales, нули -- marlin_zero_points
// (8 бит: перестановка + чередование {0,2,1,3} + упаковка по столбцам) -- всё из vllm (fa2sm80_w8.py).
#define MARLIN_NAMESPACE_NAME fa2m8
#include "kernel.h"
#include "marlin_template.h"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <unordered_map>

namespace fa2m8 {

template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);
template __global__ void Marlin<vllm::kFloat16.id(), vllm::kU8.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 4, 8, false>(MARLIN_KERNEL_PARAMS);

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS) {};
using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128}};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}};

typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
                          int prob_n, int prob_k, int num_bits, int group_size,
                          bool has_act_order, bool is_k_full, int stages) {
  bool cache_scales_chunk = has_act_order && !is_k_full;

  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = div_ceil(tb_k, 32);  // Worst case is 32 group size
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups =
        tb_groups * stages * 2;          // Chunk size is 2x pipeline over dim K
    load_groups = max(load_groups, 32);  // We load at least 32 scale groups
    return load_groups * tb_n * 2;
  } else {
    int tb_scales = tb_groups * tb_n * 2;

    return tb_scales * stages;
  }
}

int get_kernel_cache_size(thread_config_t const& th_config, int thread_m_blocks,
                          int prob_m, int prob_n, int prob_k, int num_bits,
                          int group_size, bool has_act_order, bool is_k_full,
                          int has_zp, bool is_zp_float, bool is_a_8bit,
                          int stages) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;
  int sh_a_size = stages * (tb_m * tb_k) * (is_a_8bit ? 1 : 2);
  int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size =
      (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);

  int sh_s_size =
      get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits,
                            group_size, has_act_order, is_k_full, stages);
  int sh_g_idx_size = has_act_order && !is_k_full ? stages * tb_k / 4 : 0;
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float)
      sh_zp_size = sh_s_size;
    else if (num_bits == 4)
      sh_zp_size = sh_s_size / 4;
    else if (num_bits == 8)
      sh_zp_size = sh_s_size / 2;
  }

  int total_size =
      tmp_size + sh_a_size + sh_s_size + sh_zp_size + sh_g_idx_size;

  return total_size;
}

bool is_valid_config(thread_config_t const& th_config, int thread_m_blocks,
                     int prob_m, int prob_n, int prob_k, int num_bits,
                     int group_size, bool has_act_order, bool is_k_full,
                     int has_zp, bool is_zp_float, bool is_a_8bit, int stages,
                     int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  // Check that pipeline fits into cache
  int cache_size = get_kernel_cache_size(
      th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits, group_size,
      has_act_order, is_k_full, has_zp, is_zp_float, is_a_8bit, stages);
  return cache_size <= max_shared_mem;
}

MarlinFuncPtr get_marlin_kernel(
    const vllm::ScalarType a_type, const vllm::ScalarType b_type,
    const vllm::ScalarType c_type, const vllm::ScalarType s_type,
    int thread_m_blocks, int thread_n_blocks, int thread_k_blocks,
    bool m_block_size_8, bool has_act_order, bool has_zp, int group_blocks,
    int threads, bool is_zp_float, int stages) {
  int num_bits = b_type.size_bits();
  auto kernel = MarlinDefault;

  #include "marlin8_selector.inc"

  return kernel;
}

exec_config_t determine_exec_config(
    const vllm::ScalarType& a_type, const vllm::ScalarType& b_type,
    const vllm::ScalarType& c_type, const vllm::ScalarType& s_type, int prob_m,
    int prob_n, int prob_k, int thread_m_blocks, bool m_block_size_8,
    int num_bits, int group_size, bool has_act_order, bool is_k_full,
    bool has_zp, bool is_zp_float, int is_a_8bit, int stages,
    int max_shared_mem, int sms) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1
                                        ? large_batch_thread_configs
                                        : small_batch_thread_configs;
  int thread_configs_size =
      thread_m_blocks > 1
          ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
          : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(th_config, thread_m_blocks, prob_m, prob_n, prob_k,
                         num_bits, group_size, has_act_order, is_k_full, has_zp,
                         is_zp_float, is_a_8bit, stages,
                         max_shared_mem - 512)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(th_config, thread_m_blocks, prob_m,
                                           prob_n, prob_k, num_bits, group_size,
                                           has_act_order, is_k_full, has_zp,
                                           is_zp_float, is_a_8bit, stages);

    int group_blocks = 0;
    if (!has_act_order) {
      group_blocks = group_size == -1 ? -1 : group_size / 16;
    }

    auto kernel =
        get_marlin_kernel(a_type, b_type, c_type, s_type, thread_m_blocks,
                          th_config.thread_n / 16, th_config.thread_k / 16,
                          m_block_size_8, has_act_order, has_zp, group_blocks,
                          th_config.num_threads, is_zp_float, stages);

    if (kernel == MarlinDefault) continue;

    return {1, th_config};
  }

  return exec_cfg;
}

void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* a_s, void* b_s, void* g_s, void* zp, void* g_idx,
               void* perm, void* a_tmp, int prob_m, int prob_n, int prob_k,
               int lda, void* workspace, vllm::ScalarType const& a_type,
               vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
               vllm::ScalarType const& s_type, bool has_bias,
               bool has_act_order, bool is_k_full, bool has_zp, int num_groups,
               int group_size, int dev, cudaStream_t stream, int thread_k_init,
               int thread_n_init, int sms, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float) {
  bool is_a_8bit = a_type.size_bits() == 8;
  TORCH_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m,
              ", ", prob_n, ", ", prob_k, "]");

  int group_blocks = 0;
  if (has_act_order) {
    if (is_k_full) {
      TORCH_CHECK(group_size != -1);
      group_blocks = group_size / 16;
      TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                  " is not divisible by group_blocks = ", group_blocks);
    } else {
      TORCH_CHECK(group_size == 0);
      group_blocks = 0;
    }
  } else {
    if (group_size == -1) {
      group_blocks = -1;
    } else {
      group_blocks = group_size / 16;
      TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                  " is not divisible by group_blocks = ", group_blocks);
    }
  }

  int num_bits = b_type.size_bits();
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;

  const int4* bias_ptr = (const int4*)b_bias;
  const float* a_s_ptr = (const float*)a_s;
  const int4* b_s_ptr = (const int4*)b_s;
  const uint16_t* g_s_ptr = (const uint16_t*)g_s;

  const int4* zp_ptr = (const int4*)zp;
  const int* g_idx_ptr = (const int*)g_idx;
  const int* perm_ptr = (const int*)perm;
  int4* a_tmp_ptr = (int4*)a_tmp;
  int* locks = (int*)workspace;

  if (has_act_order) {
    // Permute A columns
    int block_rows = div_ceil(prob_m, sms);
    // avoid ">>>" being formatted to "> > >"
    // clang-format off
    TORCH_CHECK(false, "fa2m8: act_order не поддерживается");
    // clang-format on
    A_ptr = a_tmp_ptr;
    lda = prob_k;

    // If we have a full K, then we can run the non-act-order version of Marlin
    // (since the weight rows are reordered by increasing group ids, and by
    // having a full K, we have full original groups)
    if (is_k_full) has_act_order = false;
  }

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  TORCH_CHECK(max_shared_mem > 0);

  int major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                         dev);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                         dev);
  TORCH_CHECK(major_capability * 10 + minor_capability >= 75,
              "marlin kernel only support Turing or newer GPUs.");
  int stages = 4;
  if (major_capability == 7 && minor_capability == 5) {
    stages = 2;
    TORCH_CHECK(a_type == vllm::kFloat16 || a_type == vllm::kS8,
                "Turing only support FP16 or INT8 activation.");
  }
  if (a_type == vllm::kFE4M3fn) {
    TORCH_CHECK(
        major_capability * 10 + minor_capability == 89 ||
            major_capability * 10 + minor_capability == 120,
        "Marlin W4A8-FP8 only support SM89 or SM120 device (It is slower than "
        "Marlin W4A16 on other devices).");
  }

  int max_par = 16;
  if (prob_n <= 4096) max_par = 16 * 8;
  int max_shared_mem_new = max_shared_mem;
  int rest_m = prob_m;
  int max_thread_m_blocks = 4;
  while (rest_m) {
    int par_count = rest_m / (max_thread_m_blocks * 16);
    if (par_count > max_par) par_count = max_par;
    int prob_m_split =
        par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;

    int thread_k = thread_k_init;
    int thread_n = thread_n_init;

    int thread_m_blocks = min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
    int m_block_size_8 = prob_m_split <= 8 && a_type.size_bits() == 16;

    // Set thread config
    exec_config_t exec_cfg;
    thread_config_t thread_tfg;
    if (thread_k != -1 && thread_n != -1) {
      thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
      exec_cfg = exec_config_t{1, thread_tfg};
      TORCH_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
                  " is not divisible by thread_n = ", thread_n);
      TORCH_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
                  " is not divisible by thread_k = ", thread_k);
    } else {
      // Auto config
      exec_cfg = determine_exec_config(
          a_type, b_type, c_type, s_type, prob_m_split, prob_n, prob_k,
          thread_m_blocks, m_block_size_8, num_bits, group_size, has_act_order,
          is_k_full, has_zp, is_zp_float, is_a_8bit, stages, max_shared_mem,
          sms);
      thread_tfg = exec_cfg.tb_cfg;
      if (thread_tfg.thread_n != -1) {
        if (prob_n / thread_tfg.thread_n *
                div_ceil(prob_m_split, thread_m_blocks * 16) * 4 <=
            sms) {
          if (is_valid_config({128, 64, 128}, thread_m_blocks, prob_m_split,
                              prob_n, prob_k, num_bits, group_size,
                              has_act_order, is_k_full, has_zp, is_zp_float,
                              is_a_8bit, stages, max_shared_mem_new)) {
            thread_tfg = {128, 64, 128};
            exec_cfg = {1, thread_tfg};
          }
        }
      }

      if (thread_tfg.thread_k == -1 && max_thread_m_blocks > 1) {
        max_thread_m_blocks--;
        continue;
      }
    }

    int num_threads = thread_tfg.num_threads;
    thread_k = thread_tfg.thread_k;
    thread_n = thread_tfg.thread_n;
    int blocks = sms * exec_cfg.blocks_per_sm;
    if (exec_cfg.blocks_per_sm > 1)
      max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;

    TORCH_CHECK(
        is_valid_config(thread_tfg, thread_m_blocks, prob_m_split, prob_n,
                        prob_k, num_bits, group_size, has_act_order, is_k_full,
                        has_zp, is_zp_float, is_a_8bit, stages,
                        max_shared_mem_new),
        "Invalid thread config: thread_m_blocks = ", thread_m_blocks,
        ", thread_k = ", thread_tfg.thread_k,
        ", thread_n = ", thread_tfg.thread_n,
        ", num_threads = ", thread_tfg.num_threads, " for MKN = [", prob_m,
        ", ", prob_k, ", ", prob_n, "] and num_bits = ", num_bits,
        ", prob_m_split = ", prob_m_split, ", group_size = ", group_size,
        ", has_act_order = ", has_act_order, ", is_k_full = ", is_k_full,
        ", has_zp = ", has_zp, ", is_zp_float = ", is_zp_float,
        ", stages = ", stages, ", max_shared_mem_new = ", max_shared_mem_new);

    auto kernel = get_marlin_kernel(
        a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
        thread_k_blocks, m_block_size_8, has_act_order, has_zp, group_blocks,
        num_threads, is_zp_float, stages);

    if (kernel == MarlinDefault) {
      TORCH_CHECK(false, "Unsupported shapes: MNK = [", prob_m, ", ", prob_n,
                  ", ", prob_k, "]", ", has_act_order = ", has_act_order,
                  ", num_groups = ", num_groups, ", group_size = ", group_size,
                  ", prob_m_split = ", prob_m_split,
                  ", thread_m_blocks = ", thread_m_blocks,
                  ", thread_n_blocks = ", thread_n_blocks,
                  ", thread_k_blocks = ", thread_k_blocks,
                  ", num_threads = ", num_threads, ", num_bits = ", num_bits);
    }

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_new);

    bool part_use_atomic_add =
        use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048;

    // avoid ">>>" being formatted to "> > >"
    // clang-format off
    kernel<<<blocks, num_threads, max_shared_mem_new, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr,
        g_idx_ptr, num_groups,
        prob_m_split, prob_n, prob_k, lda, locks, has_bias, part_use_atomic_add,
        use_fp32_reduce, max_shared_mem_new);
    // clang-format on

    bool is_a_8bit = a_type.size_bits() == 8;
    A_ptr += prob_m_split * (lda / (is_a_8bit ? 16 : 8));
    a_s_ptr += prob_m_split;
    C_ptr += prob_m_split * (prob_n / 8);
    rest_m -= prob_m_split;
  }
}


}  // namespace fa2m8

// Обратная раскладка Marlin -> W [N][K] fp16 (большие M: деквант + cuBLAS, конвейер блока) -- marlin8_dequant.cuh
// (там же раскладка и варианты). Вариант: аргумент variant >= 0 или FA2SM80_M8DQ (2 -- прежний v2, 41 -- v4 по 128 k,
// 42 -- v4 по 256 k); по умолчанию 42 при K % 256 == 0, иначе 41.
#include "marlin8_dequant.cuh"
#include <cstdlib>

namespace {
// Замки и fp32-буфер сведения -- ОДНИ на карту (не на поток): захват графа идёт на своём потоке, и буфер «на поток»
// там не находится (окно 23: отказ захвата). Умножения marlin8 идут только на основном потоке прохода -- двух
// одновременных вызовов на одной карте нет.
struct Scratch {
  torch::Tensor locks;                              // замки полос (sms), самоочищаются
  torch::Tensor ctmp;                               // fp32-сведение: sms * 64 * max_thread_n
  std::vector<torch::Tensor> old;                   // прежние буферы (могли попасть в захваченный граф)
};
std::unordered_map<int, Scratch>& scratch() { static std::unordered_map<int, Scratch> s; return s; }
int sms_of(int dev) { int s = 0; cudaDeviceGetAttribute(&s, cudaDevAttrMultiProcessorCount, dev); return s; }
bool capturing(cudaStream_t st) {
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  C10_CUDA_CHECK(cudaStreamIsCapturing(st, &cs));
  return cs != cudaStreamCaptureStatusNone;
}
}  // namespace

// Заранее (до захвата графов): замки и fp32-буфер сведения карты.
void marlin8_reserve(torch::Tensor like) {
  const c10::cuda::CUDAGuard guard(like.device());
  const int dev = like.get_device(), sms = sms_of(dev);
  auto& S = scratch()[dev];
  if (!S.locks.defined()) S.locks = torch::zeros({sms}, like.options().dtype(torch::kInt32));
  const long need = (long)sms * 64 * fa2m8::max_thread_n;
  if (!S.ctmp.defined() || S.ctmp.numel() < need) {
    if (S.ctmp.defined()) S.old.push_back(S.ctmp);
    S.ctmp = torch::empty({need}, like.options().dtype(torch::kFloat32));
  }
}

// y[M, N] = x[M, K] @ W^T; W -- Marlin-формат kU8 с нулями. out -- готовый [M, N] fp16 (столбцы подряд) или None.
torch::Tensor marlin8_gemm(torch::Tensor a, torch::Tensor b_q_weight, torch::Tensor b_scales, torch::Tensor b_zeros,
                           int64_t size_n, bool use_fp32_reduce, c10::optional<torch::Tensor> out) {
  const c10::cuda::CUDAGuard guard(a.device());
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::ScalarType::Half && a.dim() == 2 && a.stride(1) == 1 &&
              a.stride(0) % 8 == 0 && (reinterpret_cast<uintptr_t>(a.data_ptr()) % 16) == 0,
              "marlin8: a [M,K] fp16, столбцы подряд, строки по 16 Б");
  const int M = (int)a.size(0), K = (int)a.size(1), N = (int)size_n;
  TORCH_CHECK(K % fa2m8::tile_size == 0 && b_q_weight.size(0) == K / fa2m8::tile_size &&
              (b_q_weight.size(1) / fa2m8::tile_size) * 4 == N && b_q_weight.is_contiguous(), "marlin8: вес");
  TORCH_CHECK(b_scales.dim() == 2 && b_scales.size(1) == N && b_scales.size(0) * 128 == K && b_scales.is_contiguous() &&
              b_scales.scalar_type() == at::ScalarType::Half, "marlin8: масштабы [K/128, N] fp16");
  TORCH_CHECK(b_zeros.dim() == 2 && b_zeros.size(0) == K / 128 && b_zeros.size(1) == N / 4 && b_zeros.is_contiguous() &&
              b_zeros.scalar_type() == at::ScalarType::Int, "marlin8: нули [K/128, N/4] int32");
  torch::Tensor c;
  if (out.has_value()) {
    c = *out;
    TORCH_CHECK(c.scalar_type() == at::ScalarType::Half && c.size(0) == M && c.size(1) == N && c.is_contiguous(),
                "marlin8: out [M, N] fp16 подряд");
  } else {
    c = torch::empty({M, N}, a.options());
  }
  if (M == 0) return c;
  const int dev = a.get_device(), sms = sms_of(dev);
  auto st = at::cuda::getCurrentCUDAStream(dev);
  auto& S = scratch()[dev];
  if (!S.locks.defined()) {
    TORCH_CHECK(!capturing(st), "marlin8: замки не заведены до захвата (marlin8_reserve)");
    S.locks = torch::zeros({sms}, a.options().dtype(torch::kInt32));
  }
  torch::Tensor ct;
  if (use_fp32_reduce) {
    const long need = (long)sms * std::min((M + 15) / 16 * 16, 64) * fa2m8::max_thread_n;
    if (!S.ctmp.defined() || S.ctmp.numel() < need) {
      TORCH_CHECK(!capturing(st), "marlin8: буфер сведения не заведён до захвата (marlin8_reserve)");
      if (S.ctmp.defined()) S.old.push_back(S.ctmp);
      S.ctmp = torch::empty({(long)sms * 64 * fa2m8::max_thread_n}, a.options().dtype(torch::kFloat32));
    }
    ct = S.ctmp;
  }
  fa2m8::marlin_mm(a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(), use_fp32_reduce ? ct.data_ptr() : nullptr,
                   nullptr, nullptr, b_scales.data_ptr(), nullptr, b_zeros.data_ptr(), nullptr, nullptr, nullptr,
                   M, N, K, (int)a.stride(0), S.locks.data_ptr(), vllm::kFloat16, vllm::kU8, vllm::kFloat16,
                   vllm::kFloat16, false, false, true, true, K / 128, 128, dev, st, -1, -1, sms, false,
                   use_fp32_reduce, false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

// W [N, K] fp16 из раскладки Marlin (out -- готовый буфер не меньше N*K, иначе новый).
torch::Tensor marlin8_dequant(torch::Tensor b_q_weight, torch::Tensor b_scales, torch::Tensor b_zeros, int64_t N, int64_t K,
                              c10::optional<torch::Tensor> out, int64_t variant) {
  const c10::cuda::CUDAGuard guard(b_q_weight.device());
  TORCH_CHECK(N % 64 == 0 && K % 128 == 0 && b_q_weight.numel() == N * K / 4 && b_scales.numel() == (K / 128) * N &&
              b_zeros.numel() == (K / 128) * (N / 4), "marlin8_dequant: формы");
  torch::Tensor W;
  if (out.has_value()) {
    TORCH_CHECK(out->scalar_type() == at::ScalarType::Half && out->is_contiguous() && out->numel() >= N * K,
                "marlin8_dequant: out fp16 подряд, не меньше N*K");
    W = out->view(-1).narrow(0, 0, N * K).view({N, K});
  } else {
    W = torch::empty({N, K}, b_scales.options());
  }
  TORCH_CHECK(N * K < (1LL << 31), "marlin8_dequant: N*K >= 2^31 (смещения 32-битные)");
  static const int env_var = [] { const char* e = std::getenv("FA2SM80_M8DQ"); return e ? std::atoi(e) : -1; }();
  int var = variant >= 0 ? (int)variant : env_var;
  if (var < 0) var = K % 256 == 0 ? 42 : 41;
  if (var == 42 && K % 256 != 0) var = 41;
  auto st = at::cuda::getCurrentCUDAStream();
  const auto* mq = reinterpret_cast<const uint32_t*>(b_q_weight.data_ptr<int>());
  const auto* ms = reinterpret_cast<const __half*>(b_scales.data_ptr<at::Half>());
  const auto* mz = reinterpret_cast<const uint32_t*>(b_zeros.data_ptr<int>());
  auto* w = reinterpret_cast<__half*>(W.data_ptr<at::Half>());
  TORCH_CHECK((reinterpret_cast<uintptr_t>(w) % 16) == 0, "marlin8_dequant: W не выровнен на 16 Б");
  // v4 читает вес по 16 Б и масштабы по 8 Б; при другом выравнивании -- v2 (тот же ответ), а не отказ в бою
  if ((reinterpret_cast<uintptr_t>(mq) % 16) != 0 || (reinterpret_cast<uintptr_t>(ms) % 8) != 0) var = 2;
  if (var == 2) {
    const long total = (N / 64) * (K / 16) * 64;
    m8dq::v2<<<(int)std::min<long>((total + 255) / 256, 8192), 256, 0, st>>>(mq, ms, mz, w, (int)N, (int)K);
  } else if (var == 42) {
    m8dq::v4<2><<<dim3((unsigned)(N / 64), (unsigned)(K / 256)), 256, 0, st>>>(
        reinterpret_cast<const uint4*>(mq), ms, mz, w, (int)N, (int)K);
  } else {
    m8dq::v4<1><<<dim3((unsigned)(N / 64), (unsigned)(K / 128)), 256, 0, st>>>(
        reinterpret_cast<const uint4*>(mq), ms, mz, w, (int)N, (int)K);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return W;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("marlin8_dequant", &marlin8_dequant, "раскладка Marlin -> W [N,K] fp16 (побитно как w8a16_sm80_dequant)",
        py::arg("b_q_weight"), py::arg("b_scales"), py::arg("b_zeros"), py::arg("N"), py::arg("K"),
        py::arg("out") = py::none(), py::arg("variant") = -1);
  m.def("marlin8_gemm", &marlin8_gemm, "Marlin W8A16 kU8 + нули (fp16, g128): y = x @ W^T",
        py::arg("a"), py::arg("b_q_weight"), py::arg("b_scales"), py::arg("b_zeros"), py::arg("size_n"),
        py::arg("use_fp32_reduce") = true, py::arg("out") = py::none());
  m.def("marlin8_reserve", &marlin8_reserve, "Marlin8: буферы до захвата графов");
}
