// Copyright (c) 2026 xormal (Alexander Romanoff)
// SPDX-License-Identifier: MIT
//
// ================================================================================================
// ОБМЕН TP В 12 РАЗРЯДАХ: УПАКОВКА И РАЗБОР. Лестница разрядности, приложенная к УЗКОМУ КАНАЛУ.
// ================================================================================================
//
// ЗАЧЕМ ИМЕННО СЮДА, А НЕ В ЯДРА. Замерено 14.08.2026 снятием фазы: обмен TP -- 58% префилла на паре
// PCIe, и внутри него перекос рангов 2-3%, остальное ПЕРЕВОЗКА. Дальше замерена сама перевозка,
// один и тот же all_reduce при меняющемся размере:
//
//     t = 0.164 мс + 0.1770 мс/МБ     (6 точек, 10..321 МБ)
//     на боевых 160.6 МБ: постоянная часть 0.6 %, БАЙТЫ 99.4 %
//
// То есть канал платит РОВНО за байты, и сужение формата переводится в скорость один в один. И чем
// у́же канал, тем дороже разряд: на мосту PCIe идёт 5.6 ГБ/с против 25-46 у NVLink -- здесь та же
// правка стоит в разы больше. Это ровно то место, где лестница ([[fa2-sm70-bit-ladder]]) работает:
// «выигрыш ТАМ, ГДЕ СВЯЗЫВАЕТ ПАМЯТЬ». Внутри тензорных ядер она бы проиграла (+16.5 % на разбор в
// фазу, которая и связывает), а на проводе разбирать нечего -- складываем один раз в fp16.
//
// ПОЧЕМУ 12, А НЕ 8. Свёртка складывает ДВЕ частичные суммы, и квантуется только ЧУЖАЯ (своя уже
// лежит точной). Дрейф за 64 слоя, считанный по замеренному закону 2^-b (шаг в 2 разряда = x4.02):
//     fp16 сейчас 0.24 %   int14 0.12 % (ЛУЧШЕ нынешнего)   int12 0.46 %   int8 7.5 % -- НЕЛЬЗЯ.
// int12 снимает треть обмена при дрейфе вдвое от нынешнего; int8 снял бы половину, но ценой,
// которой платить нельзя.
//
// ПОЧЕМУ ГРУППА 128 И ПОЧЕМУ ВАРП НА ГРУППУ. Байтовое выравнивание даёт группировка по ЧЕТЫРЕ:
// 4 значения по 12 разрядов = 6 байт = ТРИ uint16, ни одного невыровненного обращения. Варп из 32
// полос кроет ровно 128 значений по 4 на полосу, а максимум по группе берётся сдвигами внутри
// варпа -- без разделяемой памяти и без барьеров.
//
// РАСКЛАДКА УПАКОВАННОГО -- ТРИ ПЛОСКОСТИ, А НЕ ЧЕРЕДОВАНИЕ. Наивно полоса писала бы свои 6 байт
// подряд (шаг 6 -- ни коалесцирования, ни выравнивания). Здесь варп пишет сперва все w0, потом все
// w1, потом w2: три подряд идущих отрезка по 32 uint16. Обращения слитные, адрес кратен 2, а
// обратное преобразование тривиально -- те же три отрезка.
// ================================================================================================
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

namespace {

constexpr int G = 128;          // значений в группе масштаба (= один варп x 4)
constexpr int VPL = 4;          // значений на полосу
constexpr int WPB = 8;          // варпов в блоке

__device__ __forceinline__ float amax4(float a, float b, float c, float d) {
  return fmaxf(fmaxf(fabsf(a), fabsf(b)), fmaxf(fabsf(c), fabsf(d)));
}

// x [ngrp * G] fp16 -> packed [ngrp * 96] uint16, scales [ngrp] fp16
// bits -- ЧИСЛО ЗНАЧАЩИХ РАЗРЯДОВ (12 боевое). Контейнер и байты НЕ МЕНЯЮТСЯ, грубеет только
// точность: уровней (2^(bits-1)-1), а хранимое домножается на 2^(12-bits), чтобы разбор шёл тем же
// масштабом. Нужен как ПОЛОЖИТЕЛЬНЫЙ КОНТРОЛЬ приёмки: без пути, который ОБЯЗАН разойтись,
// совпадение выводов не отличить от слепоты прибора.
__global__ void pack_i12_kernel(const __half* __restrict__ x, unsigned short* __restrict__ pk,
                                __half* __restrict__ sc, long ngrp, int bits) {
  const int lane = threadIdx.x & 31;
  const long g = (long)blockIdx.x * WPB + (threadIdx.x >> 5);
  if (g >= ngrp) return;
  const __half* px = x + g * G + lane * VPL;
  // 4 подряд идущих полуслова = 8 байт, адрес кратен 8 (группа кратна 128) -- одна широкая загрузка
  const uint2 raw = *reinterpret_cast<const uint2*>(px);
  const __half* h = reinterpret_cast<const __half*>(&raw);
  float v[VPL];
#pragma unroll
  for (int i = 0; i < VPL; ++i) v[i] = __half2float(h[i]);

  float m = amax4(v[0], v[1], v[2], v[3]);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
  // ПУСТАЯ ГРУППА -- ОТДЕЛЬНЫЙ СЛУЧАЙ, А НЕ "почти ноль": деление на ноль дало бы NaN там, где
  // ответ ровно нулевой, и он бы поехал по проводу.
  const int lvl = (1 << (bits - 1)) - 1, sh = 12 - bits;
  const float inv = (m > 0.f) ? ((float)lvl / m) : 0.f;
  if (lane == 0) sc[g] = __float2half((m > 0.f) ? (m / 2047.0f) : 0.f);

  unsigned int u[VPL];
#pragma unroll
  for (int i = 0; i < VPL; ++i) {
    int q = __float2int_rn(v[i] * inv);
    q = max(-lvl, min(lvl, q)) << sh;
    u[i] = (unsigned int)(q + 2048);          // 12 разрядов без знака, 0..4095
  }
  const unsigned short w0 = (unsigned short)( u[0]        | ((u[1] & 0xFu) << 12));
  const unsigned short w1 = (unsigned short)(((u[1] >> 4) & 0xFFu) | ((u[2] & 0xFFu) << 8));
  const unsigned short w2 = (unsigned short)(((u[2] >> 8) & 0xFu)  |  (u[3] << 4));
  unsigned short* p = pk + g * 96;
  p[0 * 32 + lane] = w0;
  p[1 * 32 + lane] = w1;
  p[2 * 32 + lane] = w2;
}

// out = base + разбор(packed)*scale
__global__ void unpack_add_i12_kernel(const __half* __restrict__ base,
                                      const unsigned short* __restrict__ pk,
                                      const __half* __restrict__ sc,
                                      __half* __restrict__ out, long ngrp) {
  const int lane = threadIdx.x & 31;
  const long g = (long)blockIdx.x * WPB + (threadIdx.x >> 5);
  if (g >= ngrp) return;
  const unsigned short* p = pk + g * 96;
  const unsigned int w0 = p[0 * 32 + lane], w1 = p[1 * 32 + lane], w2 = p[2 * 32 + lane];
  const float s = __half2float(sc[g]);
  unsigned int u[VPL];
  u[0] =  w0 & 0xFFFu;
  u[1] = ((w0 >> 12) & 0xFu) | ((w1 & 0xFFu) << 4);
  u[2] = ((w1 >> 8) & 0xFFu) | ((w2 & 0xFu) << 8);
  u[3] =  (w2 >> 4) & 0xFFFu;

  const __half* pb = base + g * G + lane * VPL;
  const uint2 raw = *reinterpret_cast<const uint2*>(pb);
  const __half* hb = reinterpret_cast<const __half*>(&raw);
  __half res[VPL];
#pragma unroll
  for (int i = 0; i < VPL; ++i)
    res[i] = __float2half(__half2float(hb[i]) + ((float)((int)u[i] - 2048)) * s);
  *reinterpret_cast<uint2*>(out + g * G + lane * VPL) = *reinterpret_cast<const uint2*>(res);
}

#define CHK(t, dt, name)                                                              \
  TORCH_CHECK((t).is_cuda() && (t).is_contiguous(), name " must be contiguous CUDA");  \
  TORCH_CHECK((t).scalar_type() == (dt), name " has wrong dtype")

}  // namespace

// ЗАПИСЬ В ГОТОВЫЕ БУФЕРЫ. Нужна не ради красоты: упакованное и масштабы кладутся в ОДИН кусок
// памяти, и тогда по проводу идёт ОДНА коллективная операция вместо двух. Замерено в сервере:
// байты снялись ровно на x1.320, а времени пришло вдвое меньше расчётного -- разницу съели вторая
// операция (свой запуск и своя синхронизация, 128 раз на кусок) и выделения буферов на каждый вызов.
void pack_i12_into(torch::Tensor x, torch::Tensor pk, torch::Tensor sc, int64_t bits) {
  const c10::cuda::CUDAGuard guard(x.device());
  CHK(x, torch::kHalf, "x"); CHK(pk, torch::kInt16, "packed"); CHK(sc, torch::kHalf, "scales");
  const long n = x.numel();
  TORCH_CHECK(n % G == 0, "pack_i12_into: число элементов должно делиться на 128");
  const long ngrp = n / G;
  TORCH_CHECK(pk.numel() >= ngrp * 96 && sc.numel() >= ngrp, "pack_i12_into: буфер мал");
  const int thr = WPB * 32;
  const long blocks = (ngrp + WPB - 1) / WPB;
  pack_i12_kernel<<<(int)blocks, thr, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const __half*)x.data_ptr(), (unsigned short*)pk.data_ptr(), (__half*)sc.data_ptr(), ngrp,
      (int)bits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<torch::Tensor, torch::Tensor> pack_i12(torch::Tensor x) {
  const c10::cuda::CUDAGuard guard(x.device());
  CHK(x, torch::kHalf, "x");
  const long n = x.numel();
  TORCH_CHECK(n % G == 0, "pack_i12: число элементов должно делиться на 128");
  const long ngrp = n / G;
  auto pk = torch::empty({ngrp * 96}, torch::dtype(torch::kInt16).device(x.device()));
  auto sc = torch::empty({ngrp}, torch::dtype(torch::kHalf).device(x.device()));
  const int thr = WPB * 32;
  const long blocks = (ngrp + WPB - 1) / WPB;
  pack_i12_kernel<<<(int)blocks, thr, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const __half*)x.data_ptr(), (unsigned short*)pk.data_ptr(), (__half*)sc.data_ptr(), ngrp, 12);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {pk, sc};
}

torch::Tensor unpack_add_i12(torch::Tensor base, torch::Tensor pk, torch::Tensor sc) {
  const c10::cuda::CUDAGuard guard(base.device());
  CHK(base, torch::kHalf, "base"); CHK(pk, torch::kInt16, "packed"); CHK(sc, torch::kHalf, "scales");
  const long n = base.numel();
  TORCH_CHECK(n % G == 0, "unpack_add_i12: число элементов должно делиться на 128");
  const long ngrp = n / G;
  TORCH_CHECK(pk.numel() == ngrp * 96 && sc.numel() == ngrp, "unpack_add_i12: несогласованные размеры");
  auto out = torch::empty_like(base);
  const int thr = WPB * 32;
  const long blocks = (ngrp + WPB - 1) / WPB;
  unpack_add_i12_kernel<<<(int)blocks, thr, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const __half*)base.data_ptr(), (const unsigned short*)pk.data_ptr(),
      (const __half*)sc.data_ptr(), (__half*)out.data_ptr(), ngrp);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// =================================================================================================
// ФУНДАМЕНТ ПЕРЕКРЫТИЯ: ПРЯМАЯ ЗАПИСЬ В ПАМЯТЬ СОСЕДА (задача 201, нативный ход).
//
// ЗАЧЕМ. Обмен -- 26.7 с из 103 с префилла (после P2P). Уменьшить ОБЪЁМ переносов нельзя
// (теорема cut-rank, research/fable/third_way_answer.md), но работа провода МЕНЬШЕ работы счёта,
// то есть вся помещается в тень. Спрятать её можно только СЛИЯНИЕМ обмена с телом, которое его
// порождает: эпилог GEMM, досчитав плитку, кладёт её сразу в буфер соседа.
//
// ПОЧЕМУ ЧЕРЕЗ IPC, А НЕ ЧЕРЕЗ NCCL. Воркеры vLLM -- ОТДЕЛЬНЫЕ ПРОЦЕССЫ, поэтому указатель
// соседа получается только через cudaIpcMemHandle. P2P между всеми парами есть (замерено 21.08).
//
// ГРАНИЦА, ОПЛАЧЕННАЯ ОТКАЗОМ: буфер обязан быть выделен ВНЕ ГРАФОВОГО ПУЛА. Проверено падением
// штатного custom_all_reduce: `cudaIpcGetMemHandle` на буферах графа даёт 'invalid argument'
// (custom_all_reduce.cuh:455). Поэтому здесь буфер -- обычный `torch.empty`, созданный ДО
// захвата, и он НЕ ПЕРЕЕЗЖАЕТ: захваченный граф держал бы протухший указатель.
#include <cuda_runtime.h>
#include <cuda.h>
#include <dlfcn.h>
#include <string>
#include "ar1ll.cuh"

// Ручка на СВОЙ буфер -- её отдают соседу один раз при поднятии (через обычный обмен объектами).
// ТРЕБОВАНИЕ К ВХОДУ: тензор обязан быть ЦЕЛЫМ ВЫДЕЛЕНИЕМ, а не видом внутри чужого блока.
// Иначе ручка укажет на начало ЧУЖОГО выделения, сосед прибавит нулевое смещение и запишет мимо
// -- тихая порча вместо отказа. Проверяем это `storage_offset() == 0` и совпадением размера
// хранилища с размером тензора; драйверный `cuMemGetAddressRange` не берём НАМЕРЕННО -- он
// требует линковки с libcuda, а собранное расширение падает на `undefined symbol` (проверено).
// ЗАМЕРЕНО ОТКАЗОМ 21.08: проверок ниже по ТЕНЗОРУ НЕ ХВАТАЕТ. У торча своё хранилище и своя
// нарезка: маленький тензор (флаги, 256 байт) живёт ВНУТРИ общего сегмента cudaMalloc на 2 МиБ,
// при этом `storage_offset()==0` и размер хранилища сходится -- обе проверки проходят. А ручка
// драйвера указывает на НАЧАЛО СЕГМЕНТА, и сосед пишет мимо: тихая порча без единого отказа.
// Поэтому базу спрашиваем У ДРАЙВЕРА и требуем совпадения. Символ берём через dlsym, а не
// линковкой: `-lcuda` роняет сборку расширения на undefined symbol (проверено).
static CUresult (*_адресный_диапазон)(CUdeviceptr*, size_t*, CUdeviceptr) = nullptr;
static void _взять_символ() {
  if (_адресный_диапазон) return;
  void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
  if (!h) h = dlopen("libcuda.so.1", RTLD_LAZY);
  TORCH_CHECK(h, "не открывается libcuda.so.1 -- база выделения непроверяема");
  *(void**)&_адресный_диапазон = dlsym(h, "cuMemGetAddressRange_v2");
  if (!_адресный_диапазон) *(void**)&_адресный_диапазон = dlsym(h, "cuMemGetAddressRange");
  TORCH_CHECK(_адресный_диапазон, "в libcuda нет cuMemGetAddressRange");
}

static void _проверить_целое(at::Tensor t, const char* кто) {
  TORCH_CHECK(t.is_cuda(), кто, ": тензор обязан быть на карте");
  TORCH_CHECK(t.is_contiguous(), кто, ": тензор обязан быть непрерывным");
  TORCH_CHECK(t.storage_offset() == 0, кто, ": тензор обязан начинаться с начала хранилища");
  TORCH_CHECK((int64_t)t.storage().nbytes() == t.numel() * t.element_size(), кто,
              ": тензор обязан занимать ВСЁ хранилище (это ЦЕЛОЕ выделение, а не вид)");
  _взять_символ();
  CUdeviceptr база = 0; size_t размер = 0;
  CUresult r = _адресный_диапазон(&база, &размер, (CUdeviceptr)t.data_ptr());
  TORCH_CHECK(r == CUDA_SUCCESS, кто, ": драйвер не дал границы выделения (код ", (int)r, ")");
  TORCH_CHECK((void*)база == t.data_ptr(), кто,
              ": буфер НЕ в начале выделения драйвера (смещение ",
              (int64_t)((char*)t.data_ptr() - (char*)база),
              " байт). Ручка укажет на чужое начало, и сосед запишет мимо. "
              "Берите буфер через ipc_alloc, а не через обычное выделение торча.");
}

// СВОЁ выделение под обмен: прямой cudaMalloc, мимо кэширующего распределителя торча. Так база
// совпадает с началом буфера по построению, и буфер не переезжает вместе с графовым пулом
// (штатный custom_all_reduce падает именно на этом -- ручка на графовом буфере невыдаваема).
static at::Tensor ipc_alloc(int64_t numel, at::ScalarType dt) {
  const int64_t nb = numel * (int64_t)c10::elementSize(dt);
  void* p = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&p, (size_t)nb));
  C10_CUDA_CHECK(cudaMemset(p, 0, (size_t)nb));
  auto opts = at::TensorOptions().dtype(dt).device(at::kCUDA, at::cuda::current_device());
  return at::from_blob(p, {numel}, [](void* q) { cudaFree(q); }, opts);
}

// ВИД НА ЧУЖОЙ УКАЗАТЕЛЬ, БЕЗ ВЛАДЕНИЯ. Нужен ровно для одного: у vLLM буфер своего all-reduce
// (`buffer_ptrs[rank]`) УЖЕ зарегистрирован через IPC при инициализации и живёт ВНЕ графового
// пула, но в питон он приходит целым числом. Графобезопасный путь обмена (`registered=True`)
// требует ВХОД в этом буфере -- значит нужно всего лишь представить его тензором.
// Деаллокатора НЕТ намеренно: владелец буфера -- vLLM, мы только смотрим.
static at::Tensor tensor_from_ptr(int64_t ptr, int64_t numel, at::ScalarType dt) {
  TORCH_CHECK(ptr != 0, "tensor_from_ptr: нулевой указатель");
  auto opts = at::TensorOptions().dtype(dt).device(at::kCUDA, at::cuda::current_device());
  return at::from_blob(reinterpret_cast<void*>(ptr), {numel}, opts);
}

// ВОЗВРАТ -- pybind11::bytes, А НЕ std::string: ручка это ДВОИЧНЫЕ данные, и питон пытался
// разобрать их как UTF-8 ('invalid start byte'). Строка здесь не текст, а 64 байта драйвера.
static pybind11::bytes ipc_handle(at::Tensor t) {
  _проверить_целое(t, "ipc_handle");
  cudaIpcMemHandle_t h;
  C10_CUDA_CHECK(cudaIpcGetMemHandle(&h, t.data_ptr()));
  return pybind11::bytes((const char*)&h, sizeof(h));
}

// Смещение оставлено в интерфейсе ради будущих видов, но при целом выделении оно всегда 0.
static int64_t ipc_offset(at::Tensor t) {
  _проверить_целое(t, "ipc_offset");
  return 0;
}

// Открыть ручку соседа и вернуть АДРЕС как число. Открывается ОДИН раз за подъём: повторное
// открытие той же ручки даёт ошибку 'already mapped', а закрытие в горячем пути стоило бы дороже
// самого обмена.
static int64_t ipc_open(const pybind11::bytes& handle, int64_t offset) {
  std::string hs = handle;
  TORCH_CHECK(hs.size() == sizeof(cudaIpcMemHandle_t), "ipc_open: ручка не того размера");
  cudaIpcMemHandle_t h;
  memcpy(&h, hs.data(), sizeof(h));
  void* p = nullptr;
  C10_CUDA_CHECK(cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess));
  return (int64_t)((char*)p + offset);
}

// Проба механизма: записать содержимое своего тензора ПРЯМО в память соседа. Это не боевое тело,
// а ПОЛОЖИТЕЛЬНЫЙ КОНТРОЛЬ -- без него нельзя отличить «перекрытие не помогает» от «запись не
// доехала». Копия идёт ядром, а не cudaMemcpy: в эпилоге GEMM будет ровно такая запись.
__global__ void poke_peer_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, long n4) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) dst[i] = src[i];
}

static void poke_peer(at::Tensor src, int64_t peer_ptr) {
  TORCH_CHECK(src.is_cuda() && src.is_contiguous(), "poke_peer: нужен непрерывный тензор на карте");
  const long nb = (long)src.numel() * (long)src.element_size();
  TORCH_CHECK(nb % 16 == 0, "poke_peer: длина обязана быть кратна 16 байтам");
  const long n4 = nb / 16;
  const int thr = 256;
  poke_peer_kernel<<<(int)((n4 + thr - 1) / thr), thr, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint4*)src.data_ptr(), (uint4*)peer_ptr, n4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------------------------
// КУСОЧНЫЙ ALL-REDUCE НА ДВЕ КАРТЫ (задача 201). Смысл не в скорости самой передачи, а в том, что
// она НЕ ЖДЁТ конца умножения: кусок выхода уехал соседу, как только досчитан. Барьер на весь
// тензор заменён флагами ПО КУСКАМ -- это и есть переход от «фазы после фазы» к перекрытию.
// Все четыре тела -- примитивы БЕЗ питона в пути: дирижировать ими будет одна операция.
// ---------------------------------------------------------------------------------------------

// Передача куска движком копирования, а НЕ ядром. Ядро (poke_peer) отнимает SM у умножения --
// ровно у той работы, под которую мы прячемся; DMA идёт мимо SM. Разница замеряется, обе есть.
static void ar2_copy(at::Tensor src, int64_t peer_ptr) {
  TORCH_CHECK(src.is_cuda() && src.is_contiguous(), "ar2_copy: нужен непрерывный кусок на карте");
  const size_t nb = (size_t)src.numel() * (size_t)src.element_size();
  C10_CUDA_CHECK(cudaMemcpyAsync((void*)peer_ptr, src.data_ptr(), nb, cudaMemcpyDeviceToDevice,
                                 at::cuda::getCurrentCUDAStream()));
}

// Флаг ставится ОТДЕЛЬНЫМ телом ПОСЛЕ копии в ТОМ ЖЕ потоке: поток сериализует операции, значит
// к моменту запуска этого тела байты куска уже в памяти соседа. __threadfence_system нужен
// потому, что читатель -- ДРУГОЕ устройство: без него запись может остаться в кэше нашей карты.
__global__ void ar2_signal_kernel(int* flag, int val) {
  __threadfence_system();
  *(volatile int*)flag = val;
}

static void ar2_signal(int64_t peer_flag_ptr, int64_t idx, int64_t val) {
  ar2_signal_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      (int*)peer_flag_ptr + idx, (int)val);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Ожидание с ПОТОЛКОМ ПО ТАКТАМ. Вечный spin на карте -- это не «зависнет опыт», а зависшая карта
// и перезагрузка машины; на боевом сервере такая цена недопустима. Потолок ~2e9 тактов (около
// 1.4 с) заведомо больше любого честного куска и заведомо меньше сторожевого таймера драйвера.
__global__ void ar2_wait_kernel(const int* flag, int val, int* срыв) {
  const long потолок = 2000000000L;
  const long t0 = clock64();
  while (*(volatile const int*)flag < val) {
    if (clock64() - t0 > потолок) { *(volatile int*)срыв = 1; return; }
  }
}

static void ar2_wait(at::Tensor flags, int64_t idx, int64_t val, at::Tensor срыв) {
  TORCH_CHECK(flags.scalar_type() == at::kInt && срыв.scalar_type() == at::kInt,
              "ar2_wait: флаги обязаны быть int32");
  ar2_wait_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const int*)flags.data_ptr() + idx, (int)val, (int*)срыв.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Сведение: y += recv, парами через half2 (тот же размен «широкое и мало», что в эпилоге GEMM).
__global__ void ar2_add_kernel(__half2* __restrict__ y, const __half2* __restrict__ r, long n2) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n2) y[i] = __hadd2(y[i], r[i]);
}

static void ar2_add(at::Tensor y, at::Tensor r) {
  TORCH_CHECK(y.is_cuda() && y.is_contiguous() && r.is_contiguous(), "ar2_add: непрерывные на карте");
  TORCH_CHECK(y.scalar_type() == at::kHalf && r.scalar_type() == at::kHalf, "ar2_add: только fp16");
  TORCH_CHECK(y.numel() == r.numel() && y.numel() % 2 == 0, "ar2_add: равные длины, кратные 2");
  const long n2 = y.numel() / 2;
  const int thr = 256;
  ar2_add_kernel<<<(int)((n2 + thr - 1) / thr), thr, 0, at::cuda::getCurrentCUDAStream()>>>(
      (__half2*)y.data_ptr(), (const __half2*)r.data_ptr(), n2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// -------------------------------------------------------------------------------------------
// ДИРИЖЁР: ЦЕЛОЕ СВЕДЕНИЕ НА ДВЕ КАРТЫ ОДНИМ ВЫЗОВОМ, ГРАФОБЕЗОПАСНОЕ (задача 201, шаг 2).
// -------------------------------------------------------------------------------------------
// ЗАЧЕМ ОТДЕЛЬНОЕ ТЕЛО, А НЕ ЧЕТЫРЕ ПРИМИТИВА ИЗ ПИТОНА. Номер последовательности при
// ЗАХВАТЕ ГРАФА ЗАПЕКАЕТСЯ: на воспроизведении питон не выполняется, и ждущий будет сверять
// флаг со СТАРЫМ числом -- он окажется уже выставлен с прошлого прохода, ожидание пройдёт
// мимо, и сосед прочитает НЕ ТЕ байты. Порча тихая: ответ остаётся правдоподобным.
// Поэтому счётчик живёт В ПАМЯТИ УСТРОЙСТВА и растёт на КАЖДОМ воспроизведении -- у обоих
// рангов одинаково, потому что код у них один и тот же и число вызовов за проход равное.
//
// ПОРЯДОК ВНУТРИ ПОТОКА (он же и есть гарантия): счётчик -> копия соседу -> флаг соседу ->
// ожидание чужого флага -> сложение. Поток сериализует, значит к моменту выставления флага
// байты уже у соседа; `__threadfence_system` нужен потому, что читатель -- ДРУГОЕ устройство.
//
// ОЖИДАНИЕ С ПОТОЛКОМ. Вечный spin на карте лечится только перезагрузкой машины, а это боевой
// сервер. Потолок в тактах заведомо больше честного куска и заведомо меньше сторожа драйвера.
__global__ void ar2_bump_kernel(int* seq) {
  if (threadIdx.x == 0) atomicAdd(seq, 1);
}

__global__ void ar2_signal_seq_kernel(int* peer_flag, const int* seq) {
  if (threadIdx.x == 0) {
    __threadfence_system();
    *(volatile int*)peer_flag = *(volatile const int*)seq;
  }
}

__global__ void ar2_wait_seq_kernel(const int* flag, const int* seq, int* срыв) {
  const long потолок = 2000000000L;
  const long t0 = clock64();
  const int цель = *(volatile const int*)seq;
  while (*(volatile const int*)flag < цель) {
    if (clock64() - t0 > потолок) { *(volatile int*)срыв = 1; return; }
  }
}

// y  -- вход-выход (fp16, непрерывный); recv -- МОЙ приёмник (ipc_alloc, зарегистрирован у соседа);
// peer_recv_ptr -- адрес ПРИЁМНИКА СОСЕДА; my_flag -- мой флаг (ipc_alloc); peer_flag_ptr -- флаг соседа;
// seq -- счётчик на устройстве (int32, один на место вызова); срыв -- признак истёкшего ожидания.
// ВСТРЕЧНЫЙ ФЛАГ «ДОЧИТАНО». Без него есть ГОНКА: я перезаписываю приёмник соседа под
// следующий проход, пока он ещё складывает предыдущий. Поток сериализует только МОИ
// операции, порядок ЧУЖИХ он не задаёт. Гонка тихая и зависит от нагрузки: под гейтом
// (лишний коллектив работает барьером) значения совпадали, без него выход становился
// словесной кашей -- поймано 27.08.
// Ждём, пока сосед объявит, что дочитал ПРЕДЫДУЩИЙ обмен по этому месту. В установившемся
// режиме он давно дочитал, поэтому ожидание почти бесплатно.
__global__ void ar2_wait_free_kernel(const int* дочитано, const int* seq, int* срыв) {
  const long потолок = 2000000000L;
  const long t0 = clock64();
  const int нужно = *(volatile const int*)seq;   // до инкремента: номер ПРЕДЫДУЩЕГО обмена
  while (*(volatile const int*)дочитано < нужно) {
    if (clock64() - t0 > потолок) { *(volatile int*)срыв = 2; return; }
  }
}

__global__ void ar2_ack_kernel(int* peer_ack, const int* seq) {
  if (threadIdx.x == 0) {
    __threadfence_system();
    *(volatile int*)peer_ack = *(volatile const int*)seq;
  }
}

// ОТПРАВКА ОДНИМ ТЕЛОМ: данные И флаг пишет ОДНО ядро. Прежде данные ехали ДВИЖКОМ
// КОПИРОВАНИЯ, а флаг ставило ядро -- два РАЗНЫХ исполнителя пишут в память соседа, и
// порядок между ними мой поток не задаёт. Под нагрузкой это и портило выход: бисекция
// 27.08 показала, что до 16 мест вызова за проход всё верно, а с 64 -- каша, при том что
// значения на каждом сведении совпадали с NCCL (гейт со своим положительным контролем).
// Здесь порядок гарантирован ПРОГРАММНО: копия -> `__threadfence_system` -> флаг, и всё
// внутри одного ядра. Последний доработавший блок объявляет флаг (выбор через atomicAdd).
// [СЧЁТЧИК, ПЕРЕЖИВАЮЩИЙ ГРАФ, 07.09] Со стороны питона вызовы обмена посчитать
// НЕЛЬЗЯ: при полном графе питон в шаге не исполняется, и все ядра приходят из
// повтора. Счётчик обязан жить В ЯДРЕ -- тогда он считает и повторы графа.
// Цена: один atomicAdd на запуск (нить 0 блока 0), то есть ничто.
__device__ unsigned long long ar2_schetchik = 0ULL;
__device__ unsigned long long ar2_elementov = 0ULL;

__global__ void ar2_send_kernel(__half2* __restrict__ dst, const __half2* __restrict__ src,
                                long n2, int* peer_flag, const int* seq, int* готовых,
                                int блоков) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    atomicAdd(&ar2_schetchik, 1ULL);
    atomicAdd(&ar2_elementov, (unsigned long long)n2);
  }
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long шаг = (long)gridDim.x * blockDim.x;
  for (; i < n2; i += шаг) dst[i] = src[i];
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();                       // мои записи видны ДРУГОМУ устройству
    const int было = atomicAdd(готовых, 1);
    if (было == блоков - 1) {                     // последний блок -- объявляет
      *(volatile int*)готовых = 0;                // счётчик к следующему разу
      __threadfence_system();
      *(volatile int*)peer_flag = *(volatile const int*)seq;
    }
  }
}

// Сложение В ОТДЕЛЬНЫЙ ВЫХОД: out = y + recv. Нужно, чтобы обойтись БЕЗ клонирования входа.
// Семантика «не на месте» обязательна (см. §65-ж: сведение на месте затирало остаток слоя),
// но клон стоит целой копии; здесь копии нет -- выход всё равно надо куда-то писать.
__global__ void ar2_add_out_kernel(__half2* __restrict__ out, const __half2* __restrict__ y,
                                   const __half2* __restrict__ r, long n2) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n2) out[i] = __hadd2(y[i], r[i]);
}

static void ar2_all_reduce_out(at::Tensor y, at::Tensor out, at::Tensor recv,
                               int64_t peer_recv_ptr, at::Tensor my_flag, int64_t peer_flag_ptr,
                               at::Tensor seq, at::Tensor срыв,
                               at::Tensor my_ack, int64_t peer_ack_ptr) {
  TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kHalf,
              "ar2_all_reduce: y -- непрерывный fp16 на карте");
  TORCH_CHECK(recv.numel() >= y.numel(), "ar2_all_reduce: приёмник меньше входа");
  TORCH_CHECK(y.numel() % 2 == 0, "ar2_all_reduce: длина обязана быть чётной");
  TORCH_CHECK(seq.scalar_type() == at::kInt && my_flag.scalar_type() == at::kInt
              && срыв.scalar_type() == at::kInt, "ar2_all_reduce: счётчик и флаги int32");
  auto поток = at::cuda::getCurrentCUDAStream();
  // 0) убедиться, что сосед дочитал ПРЕДЫДУЩИЙ обмен по этому месту (счётчик ещё не рос)
  ar2_wait_free_kernel<<<1, 1, 0, поток>>>((const int*)my_ack.data_ptr(),
                                           (const int*)seq.data_ptr(), (int*)срыв.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  ar2_bump_kernel<<<1, 1, 0, поток>>>((int*)seq.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const long n2s = y.numel() / 2;
  const int thr_s = 256;
  int бл = (int)((n2s + thr_s - 1) / thr_s);
  if (бл > 64) бл = 64;                            // счётчик блоков живёт в `срыв`+1
  ar2_send_kernel<<<бл, thr_s, 0, поток>>>(
      (__half2*)peer_recv_ptr, (const __half2*)y.data_ptr(), n2s,
      (int*)peer_flag_ptr, (const int*)seq.data_ptr(),
      (int*)срыв.data_ptr() + 1, бл);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  ar2_wait_seq_kernel<<<1, 1, 0, поток>>>((const int*)my_flag.data_ptr(),
                                          (const int*)seq.data_ptr(), (int*)срыв.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const long n2 = y.numel() / 2;
  const int thr = 256;
  ar2_add_out_kernel<<<(int)((n2 + thr - 1) / thr), thr, 0, поток>>>(
      (__half2*)out.data_ptr(), (const __half2*)y.data_ptr(),
      (const __half2*)recv.data_ptr(), n2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // 5) объявить соседу, что его байты дочитаны и место свободно
  ar2_ack_kernel<<<1, 1, 0, поток>>>((int*)peer_ack_ptr, (const int*)seq.data_ptr());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------------------------
// AR1: ВСЁ СВЕДЕНИЕ ОДНИМ ЯДРОМ (путь A100). У AR2 шесть запусков на сведение (ожидание «дочитано»,
// счётчик, отправка, ожидание флага, сложение, «дочитано» соседу) -- в графе это шесть узлов по
// ~2 мкс плюс две полные задержки PCIe между ними. Здесь тот же протокол внутри одного ядра:
//   * блок b владеет НЕПОДВИЖНЫМ куском [b*C, (b+1)*C) (C = 2048 half = 16 Б на нить), и у куска
//     СВОИ флаг, «дочитано» и счётчик. Блоки не ждут друг друга -- межблочной синхронизации нет;
//   * число блоков зависит от размера, но счётчик куска b растёт ровно на тех сведениях, где кусок
//     b был, -- одинаково у обоих рангов (формы у них одни), значит номера сходятся по построению;
//   * порядок: ждать «дочитано» прошлого обмена -> свои байты соседу -> fence.sys -> флаг соседу
//     (st.release.sys) -> ждать свой флаг (ld.acquire.sys, локальная память) -> чтение приёмника
//     мимо L1 -> сумма -> «дочитано» соседу (release: чтения приёмника упорядочены до него).
// Ожидание с потолком по тактам, как у AR2: срыв пишется в `срыв` и проверяется на каждом проходе.
// ---------------------------------------------------------------------------------------------
constexpr int AR1_THR = 256;
constexpr long AR1_CHUNK = AR1_THR * 8;

__device__ __forceinline__ int ar1_ld_acq(const int* p) {
  int v;
  asm volatile("ld.acquire.sys.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void ar1_st_rel(int* p, int v) {
  asm volatile("st.release.sys.global.b32 [%0], %1;\n" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void ar1_wait(const int* p, int цель, int* срыв, int код) {
  const long t0 = clock64();
  while (ar1_ld_acq(p) < цель) {
    if (clock64() - t0 > 2000000000L) { atomicExch(срыв, код); return; }
  }
}

// VEC: все указатели кратны 16 Б -- нить везёт uint4; иначе (вид со смещением) -- поэлементно.
// Протокол от VEC не зависит, поэтому ранги с разным выравниванием не расходятся.
template <bool VEC>
__global__ __launch_bounds__(AR1_THR) void ar1_kernel(
    const __half* __restrict__ y, __half* __restrict__ out, const __half* recv, __half* peer_recv,
    const int* flag, int* peer_flag, const int* ack, int* peer_ack, int* seq, int* срыв, long n) {
  const int b = blockIdx.x;
  const long i0 = (long)b * AR1_CHUNK + (long)threadIdx.x * 8;
  __shared__ int e_s;
  if (threadIdx.x == 0) {
    const int e = seq[b] + 1;
    ar1_wait(ack + b, e - 1, срыв, 2);            // сосед дочитал прошлый обмен по этому куску
    e_s = e;
  }
  __syncthreads();
  const int e = e_s;
  uint4 v = make_uint4(0, 0, 0, 0);
  __half* vh = reinterpret_cast<__half*>(&v);
  if (VEC && i0 + 8 <= n) {
    v = *reinterpret_cast<const uint4*>(y + i0);
    *reinterpret_cast<uint4*>(peer_recv + i0) = v;
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (i0 + j < n) { vh[j] = y[i0 + j]; peer_recv[i0 + j] = vh[j]; }
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    ar1_st_rel(peer_flag + b, e);
    ar1_wait(flag + b, e, срыв, 1);
  }
  __syncthreads();
  if (VEC && i0 + 8 <= n) {
    uint4 r = __ldcg(reinterpret_cast<const uint4*>(recv + i0));
    __half2* v2 = reinterpret_cast<__half2*>(&v);
    const __half2* r2 = reinterpret_cast<const __half2*>(&r);
#pragma unroll
    for (int j = 0; j < 4; ++j) v2[j] = __hadd2(v2[j], r2[j]);
    *reinterpret_cast<uint4*>(out + i0) = v;
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (i0 + j < n) out[i0 + j] = __hadd(vh[j], __ldcg(recv + i0 + j));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    ar1_st_rel(peer_ack + b, e);
    seq[b] = e;
  }
}

// Аргументы как у ar2_all_reduce_out, но flag/ack/seq -- ОКНА по куску на место (длина >= блоков).
static void ar1_all_reduce_out(at::Tensor y, at::Tensor out, at::Tensor recv,
                               int64_t peer_recv_ptr, at::Tensor my_flag, int64_t peer_flag_ptr,
                               at::Tensor seq, at::Tensor срыв,
                               at::Tensor my_ack, int64_t peer_ack_ptr) {
  TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kHalf,
              "ar1_all_reduce: y -- непрерывный fp16 на карте");
  TORCH_CHECK(out.is_contiguous() && out.numel() == y.numel(), "ar1_all_reduce: выход");
  TORCH_CHECK(recv.numel() >= y.numel(), "ar1_all_reduce: приёмник меньше входа");
  const long n = y.numel();
  const int бл = (int)((n + AR1_CHUNK - 1) / AR1_CHUNK);
  TORCH_CHECK(my_flag.numel() >= бл && my_ack.numel() >= бл && seq.numel() >= бл,
              "ar1_all_reduce: окно флагов меньше числа кусков");
  if (n == 0) return;
  auto поток = at::cuda::getCurrentCUDAStream();
  const bool vec = ((reinterpret_cast<uintptr_t>(y.data_ptr()) | reinterpret_cast<uintptr_t>(out.data_ptr())
                     | reinterpret_cast<uintptr_t>(recv.data_ptr()) | (uintptr_t)peer_recv_ptr) & 15) == 0;
  auto k = vec ? ar1_kernel<true> : ar1_kernel<false>;
  k<<<бл, AR1_THR, 0, поток>>>((const __half*)y.data_ptr(), (__half*)out.data_ptr(),
                               (const __half*)recv.data_ptr(), (__half*)peer_recv_ptr,
                               (const int*)my_flag.data_ptr(), (int*)peer_flag_ptr,
                               (const int*)my_ack.data_ptr(), (int*)peer_ack_ptr,
                               (int*)seq.data_ptr(), (int*)срыв.data_ptr(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------------------------
// AR1H: ТО ЖЕ ОДНО ЯДРО, НО ЧЕРЕЗ ПАМЯТЬ ХОСТА. Замер 23.10 на паре GPU0-1 (A100, корневой комплекс
// Whitley, топология NODE): P2P-запись ядром в IPC-память соседа МОЛЧА НЕ ДОХОДИТ (проба: сосед читает
// нули, отказа нет) -- отсюда и зависание штатного custom all-reduce. NCCL на этой паре ходит через SHM.
// Здесь так же, но без прокси и шести запусков: общий сегмент shm закреплён (cudaHostRegister) в ОБОИХ
// процессах, ранг пишет свой кусок в СВОЙ участок хоста и флаг, сосед читает его оттуда мимо кэшей.
// Двойной буфер по чётности номера обмена вместо «дочитано»: ранг доходит до обмена e+2 только увидев
// флаг соседа за e+1, а сосед ставит его, лишь закончив обмен e целиком (порядок потока) -- значит
// участок e&1 к этому времени дочитан. Флаги монотонны, ожидание с потолком (срыв -> `срыв`).
// ---------------------------------------------------------------------------------------------
template <bool VEC>
__global__ __launch_bounds__(AR1_THR) void ar1h_kernel(
    const __half* __restrict__ y, __half* __restrict__ out, __half* my_slot, const __half* peer_slot,
    int* my_flag, const int* peer_flag, int* seq, int* срыв, long n, long slot_elems) {
  const int b = blockIdx.x;
  const long i0 = (long)b * AR1_CHUNK + (long)threadIdx.x * 8;
  __shared__ int e_s;
  if (threadIdx.x == 0) e_s = seq[b] + 1;
  __syncthreads();
  const int e = e_s;
  const long par = (long)(e & 1) * slot_elems;
  uint4 v = make_uint4(0, 0, 0, 0);
  __half* vh = reinterpret_cast<__half*>(&v);
  if (VEC && i0 + 8 <= n) {
    v = *reinterpret_cast<const uint4*>(y + i0);
    *reinterpret_cast<uint4*>(my_slot + par + i0) = v;
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (i0 + j < n) { vh[j] = y[i0 + j]; my_slot[par + i0 + j] = vh[j]; }
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    ar1_st_rel(my_flag + b, e);
    ar1_wait(peer_flag + b, e, срыв, 1);
  }
  __syncthreads();
  if (VEC && i0 + 8 <= n) {
    const uint4 r = __ldcv(reinterpret_cast<const uint4*>(peer_slot + par + i0));
    __half2* v2 = reinterpret_cast<__half2*>(&v);
    const __half2* r2 = reinterpret_cast<const __half2*>(&r);
#pragma unroll
    for (int j = 0; j < 4; ++j) v2[j] = __hadd2(v2[j], r2[j]);
    *reinterpret_cast<uint4*>(out + i0) = v;
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (i0 + j < n) out[i0 + j] = __hadd(vh[j], __ldcv(peer_slot + par + i0 + j));
  }
  if (threadIdx.x == 0) seq[b] = e;
}

// my_slot/peer_slot -- адреса устройства участков места [2 чётности][slot_elems] в общем сегменте хоста;
// my_flag/peer_flag -- окна флагов места (по куску); seq -- окно счётчиков места на устройстве.
static void ar1h_all_reduce_out(at::Tensor y, at::Tensor out, int64_t my_slot, int64_t peer_slot,
                                int64_t my_flag, int64_t peer_flag, at::Tensor seq, at::Tensor срыв,
                                int64_t slot_elems) {
  TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kHalf,
              "ar1h_all_reduce: y -- непрерывный fp16 на карте");
  TORCH_CHECK(out.is_contiguous() && out.numel() == y.numel(), "ar1h_all_reduce: выход");
  const long n = y.numel();
  TORCH_CHECK(n <= slot_elems, "ar1h_all_reduce: участок меньше входа");
  const int бл = (int)((n + AR1_CHUNK - 1) / AR1_CHUNK);
  TORCH_CHECK(seq.numel() >= бл, "ar1h_all_reduce: окно счётчиков меньше числа кусков");
  if (n == 0) return;
  const bool vec = ((reinterpret_cast<uintptr_t>(y.data_ptr()) | reinterpret_cast<uintptr_t>(out.data_ptr())
                     | (uintptr_t)my_slot | (uintptr_t)peer_slot) & 15) == 0 && slot_elems % 8 == 0;
  auto k = vec ? ar1h_kernel<true> : ar1h_kernel<false>;
  k<<<бл, AR1_THR, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const __half*)y.data_ptr(), (__half*)out.data_ptr(), (__half*)my_slot, (const __half*)peer_slot,
      (int*)my_flag, (const int*)peer_flag, (int*)seq.data_ptr(), (int*)срыв.data_ptr(), n, slot_elems);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// AR1LL (ar1ll.cuh): my_cells/peer_cells -- адреса устройства ячеек места [2 чётности][slot_cells] в ОТДЕЛЬНОМ
// сегменте хоста; seq -- окно счётчиков блоков места (своё, не AR1H).
static void ar1ll_all_reduce_out(at::Tensor y, at::Tensor out, int64_t my_cells, int64_t peer_cells,
                                 at::Tensor seq, at::Tensor срыв, int64_t slot_cells) {
  TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kHalf, "ar1ll: y -- непрерывный fp16");
  TORCH_CHECK(out.is_contiguous() && out.numel() == y.numel(), "ar1ll: выход");
  const long n = y.numel();
  TORCH_CHECK(n % 2 == 0 && ((reinterpret_cast<uintptr_t>(y.data_ptr()) | reinterpret_cast<uintptr_t>(out.data_ptr())) & 3) == 0,
              "ar1ll: чётное число и выравнивание 4 Б");
  const long n2 = n / 2, cells = (n2 + 2) / 3;
  TORCH_CHECK(cells <= slot_cells, "ar1ll: ячеек места меньше нужного");
  const int bl = (int)((cells + ar1ll::THR - 1) / ar1ll::THR);
  TORCH_CHECK(seq.numel() >= bl, "ar1ll: окно счётчиков меньше числа блоков");
  if (n == 0) return;
  ar1ll::ar1ll_kernel<<<bl, ar1ll::THR, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const uint32_t*)y.data_ptr(), (uint32_t*)out.data_ptr(), (uint4*)my_cells, (const uint4*)peer_cells,
      seq.data_ptr<int>(), срыв.data_ptr<int>(), n2, slot_cells);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Закрепить чужое (shm) отображение хоста за CUDA и вернуть адрес устройства.
static int64_t host_register(int64_t ptr, int64_t nbytes) {
  C10_CUDA_CHECK(cudaHostRegister((void*)ptr, (size_t)nbytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
  void* d = nullptr;
  C10_CUDA_CHECK(cudaHostGetDevicePointer(&d, (void*)ptr, 0));
  return (int64_t)d;
}

// Чтение и сброс счётчика обмена (см. комментарий у ar2_schetchik).
std::vector<int64_t> ar2_counter_read(bool reset) {
  unsigned long long h_c = 0, h_e = 0;
  cudaMemcpyFromSymbol(&h_c, ar2_schetchik, sizeof(h_c));
  cudaMemcpyFromSymbol(&h_e, ar2_elementov, sizeof(h_e));
  if (reset) {
    unsigned long long z = 0;
    cudaMemcpyToSymbol(ar2_schetchik, &z, sizeof(z));
    cudaMemcpyToSymbol(ar2_elementov, &z, sizeof(z));
  }
  return {(int64_t)h_c, (int64_t)h_e};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_i12_into", &pack_i12_into, "то же, но в готовые буферы (одна коллективная операция)",
        pybind11::arg("x"), pybind11::arg("pk"), pybind11::arg("sc"), pybind11::arg("bits") = 12);
  m.def("pack_i12", &pack_i12, "упаковка fp16 -> знаковые 12 разрядов (группа 128, масштаб fp16)");
  m.def("unpack_add_i12", &unpack_add_i12, "base + разбор(12 разрядов)");
  m.def("ipc_alloc", &ipc_alloc, "СВОЙ буфер под обмен (прямой cudaMalloc, мимо торча)",
        pybind11::arg("numel"), pybind11::arg("dtype"));
  m.def("ipc_handle", &ipc_handle, "ручка на СВОЙ буфер для соседа (начало выделения)");
  m.def("tensor_from_ptr", &tensor_from_ptr, "вид на чужой указатель без владения",
        pybind11::arg("ptr"), pybind11::arg("numel"), pybind11::arg("dtype"));
  m.def("ipc_offset", &ipc_offset, "смещение вида внутри своего выделения");
  m.def("ipc_open", &ipc_open, "открыть ручку соседа -> адрес как число",
        pybind11::arg("handle"), pybind11::arg("offset") = 0);
  m.def("poke_peer", &poke_peer, "ЯДРОМ записать свой тензор в память соседа (контроль)");
  m.def("ar1_all_reduce_out", &ar1_all_reduce_out, "сведение двух карт ОДНИМ ядром (окна флагов по кускам)");
  m.def("ar1h_all_reduce_out", &ar1h_all_reduce_out, "сведение двух карт одним ядром ЧЕРЕЗ ПАМЯТЬ ХОСТА");
  m.def("ar1ll_all_reduce_out", &ar1ll_all_reduce_out, "сведение двух карт: данные с номером в 16 Б записи (хост)");
  m.def("host_register", &host_register, "cudaHostRegister(mapped|portable) -> адрес устройства");
  m.def("ar2_counter_read", &ar2_counter_read,
        "счётчик запусков обмена, переживающий граф; reset=true обнуляет",
        pybind11::arg("reset") = false);
  m.def("ar2_copy", &ar2_copy, "движком копирования: кусок -> память соседа");
  m.def("ar2_signal", &ar2_signal, "поставить флаг готовности куска У СОСЕДА",
        pybind11::arg("peer_flag_ptr"), pybind11::arg("idx"), pybind11::arg("val"));
  m.def("ar2_wait", &ar2_wait, "ждать флаг куска (с потолком по тактам)",
        pybind11::arg("flags"), pybind11::arg("idx"), pybind11::arg("val"), pybind11::arg("срыв"));
  m.def("ar2_add", &ar2_add, "y += recv парами half2");
  m.def("ar2_all_reduce_out", &ar2_all_reduce_out,
        "целое сведение на две карты одним вызовом, выход ОТДЕЛЬНЫЙ (без клона входа)");
}
