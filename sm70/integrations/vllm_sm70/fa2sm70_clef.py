# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ==================================================================================================
# CLEF В БОЕВОЙ СЕТИ (FA2SM70_CLEF=1): классификатор Cloudflare/clef поверх той же сети, без второй
# модели и без потери обычных эндпоинтов.
# ==================================================================================================
#
# ЧТО ДЕЛАЕТ. Эндпоинт /v1/systemone (формат Jev/SystemOne, как у Clef) кодирует запрос по шаблону
# Clef и отправляет его в движок как обычный префилл с max_tokens=1. Для строк ЭТОГО запроса:
#   * в проекциях слоёв 2-63 к выходу добавляется поправка (обратная аблитерация + LoRA Clef);
#     базовые строки того же шага не трогаются ни одним байтом;
#   * после прохода скрытые после финальной нормы уходят в отвод головы (ранг 0), в конце запроса
#     голова считает логиты вариантов; результат кладётся в /dev/shm и читается эндпоинтом.
# Весь счёт -- расширение fa2sm70_clef (cuBLAS + свои ядра), питона на пути шага нет: питон здесь --
# только приём запроса, сегменты строк на шаг и выдача результата.
#
# ИЗОЛЯЦИЯ КЭША. Запрос Clef идёт с cache_salt (его KV и состояние GDN не достанутся базовым) и с
# skip_reading_prefix_cache (голове нужны скрытые ВСЕХ позиций, попадание в кэш их бы съело).
# Шаг со строками Clef идёт без графа (force_eager): тело поправки при захвате было бы пустым.
#
# РЫЧАГИ
#   FA2SM70_CLEF=1                 включить (умолчание 0: ни одного изменения пути)
#   FA2SM70_CLEF_DIR               каталог адаптеров (clef_lora_r256_fp16, deabl_*, joint_head_fp16);
#                                  по умолчанию <каталог модели>/clef (раскладка релиза на HF)
#   FA2SM70_CLEF_DEABL             файл обратной аблитерации (deabl_abl_fp16.safetensors); "" = нет
#   FA2SM70_CLEF_NODELTA=1         без поправки (эталонный прогон на весах самого Clef)
#   FA2SM70_CLEF_NSLOT             ячеек кольца подкачки (3)
#   FA2SM70_CLEF_STREAM_FROM       слой, с которого адаптер подкачивается с хоста (40; 1000 -- всё на карте)
#   FA2SM70_CLEF_R                 одновременных запросов Clef (2)
#   FA2SM70_CLEF_OUT               куда класть результаты (/dev/shm/fa2sm70_clef)
#   FA2SM70_CLEF_NMAX              предел длины записи в токенах (16384; голова сверена до 32768)
#   FA2SM70_CLEF_MEDIA=1           картинки в записи ("images"), как у Clef; умолчание 0 = отказ 400
#   FA2SM70_CLEF_IMGMAX            картинок на запись (16)
#
# КАРТИНКИ (13.10). Визуальный энкодер Clef ПОБАЙТОВО равен энкодеру боевой сети (333/333 тензора),
# поэтому новых весов нет: энкодер считает vLLM (до _model_forward, поправка его не касается), а
# поправка Clef ложится на все строки записи, включая строки картинок. Токены картинок стоят сразу
# после "STATE:\n", как в encode_record Clef. Движку уходят СВЁРНУТЫЕ заполнители (по одному
# <|image_pad|> на картинку) -- vLLM разворачивает их сам; спаны считаются по РАЗВЁРНУТОЙ записи
# (число токенов картинки -- тем же процессором изображений). Воркер сверяет длину и контрольную
# сумму записи в движке с тем, что посчитал эндпоинт, и при расхождении отказывает, а не считает
# по сдвинутым спанам.
import json
import zlib
from array import array
import math
import os
import re
import sys
import time
import traceback

ВКЛ = os.environ.get("FA2SM70_CLEF", "0") == "1"
_DIR = os.environ.get("FA2SM70_CLEF_DIR", "")
_DEABL = os.environ.get("FA2SM70_CLEF_DEABL", "deabl_abl_fp16.safetensors")
_БЕЗ_ПОПРАВКИ = os.environ.get("FA2SM70_CLEF_NODELTA", "0") == "1"
_NSLOT = int(os.environ.get("FA2SM70_CLEF_NSLOT", "3"))
# [A100 24.10] С какого слоя адаптер ПОДКАЧИВАЕТСЯ из закреплённой памяти хоста (кольцо _NSLOT ячеек) вместо того,
# чтобы жить на карте. 40 -- как на V100 (32 ГБ: слои LoRA 40-63, ~895 МБ на карту, шли по PCIe на КАЖДОМ шаге
# Clef). На A100-40GB память есть: 1000 = всё резидентно, подкачки нет, числа побитово те же.
_STREAM_FROM = int(os.environ.get("FA2SM70_CLEF_STREAM_FROM", "40"))
_R = int(os.environ.get("FA2SM70_CLEF_R", "2"))
_OUT = os.environ.get("FA2SM70_CLEF_OUT", "/dev/shm/fa2sm70_clef")
_NMAX = int(os.environ.get("FA2SM70_CLEF_NMAX", "16384"))
# [26.10] СЧЁТЧИК ВХОДЯЩИХ ТОКЕНОВ Clef (раздельно с моделью): записи и токены, отданные движку, нарастающим итогом с
# пуска; строка в лог не чаще раза в FA2SM70_CLEF_TOKLOG секунд (по приходу записи). Токены модели (чаты и прочее) =
# vllm:prompt_tokens_total - эти. Стоимость -- два сложения на запрос.
_ТОК = {"записей": 0, "вход": 0, "t": 0.0}
_ТОК_ПЕРИОД = int(os.environ.get("FA2SM70_CLEF_TOKLOG", "600"))
_MEDIA = os.environ.get("FA2SM70_CLEF_MEDIA", "0") == "1"
# [18.10, фаззинг 14.10 clefvb A1] Вытеснение посреди префилла: кусок не продолжает сложенное ->
# ячейку отпустить и принять запрос заново (без этого спаны складывались дважды). Умолчание 0.
_A1 = os.environ.get("FA2SM70_CLEF_A1", "0") == "1"
if _A1:
    print(f"[fa2_sm70 18.10] CLEF_A1=1 ({__file__})", flush=True)
_IMGMAX = int(os.environ.get("FA2SM70_CLEF_IMGMAX", "16"))
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
# диагностика (12.10): in_proj_ba остаётся штатным (своя операция не ставится, поправка ba не действует)
_БЕЗ_BA = os.environ.get("FA2SM70_CLEF_NOBA", "0") == "1"
# диагностика (12.10, поиск сдвига базового декода): nosetup | nohead | notrim (через запятую)
_ДИАГ = set(x for x in os.environ.get("FA2SM70_CLEF_DIAG", "").split(",") if x)
_QOMAX, _QNMAX = 448, 64
_SALT = "fa2sm70-clef"

_АКТИВЕН = [False]
_СЕГМЕНТЫ = [None]    # [(row0, n, slot, pos0, fin)] прохода с Clef -- строки для FA2SM80_W8A8=clef|llm
# [ПЛАВНОСТЬ 29.10] ГРАФОБЕЗОПАСНАЯ ПОПРАВКА (FA2SM70_CLEF_GRAPH=1): шаги с Clef до FA2SM70_CLEF_GRAPH_MAX токенов идут
# кусочными графами семейства has_clef (cudagraph_dispatcher) вместо force_eager; поправка -- clef.cu delta_graph
# (строки Clef через постоянные буферы на GPU). FA2SM70_CLEF_GRAPH_FILE=<путь>: "0" в файле -- прежний путь
# (force_eager), "1" -- графовый; перечитывается раз в секунду (A/B в одном экземпляре). Нужны размеры графов
# до FA2SM70_CLEF_GRAPH_MAX в FA2SM80_CG_SIZES.
_ГРАФ = os.environ.get("FA2SM70_CLEF_GRAPH", "0") == "1"
_ГРАФ_МАКС = int(os.environ.get("FA2SM70_CLEF_GRAPH_MAX", "512"))
_ГРАФ_ФАЙЛ = os.environ.get("FA2SM70_CLEF_GRAPH_FILE", "")
_ГРАФ_ГОТОВ = [False]
# [ПЛАВНОСТЬ 29.10] ГОЛОВА ВНЕ КРИТИЧЕСКОГО ПУТИ (FA2SM70_CLEF_HEAD_ASYNC=1): tap и голова (head_run: 9 ступеней весов
# H2D ~234 МиБ + внимание по записи) идут на побочном потоке rank 0 после прохода; основной поток сразу берёт
# следующий шаг, а TP-сосед не ждёт голову на ближайшем сведении (профиль окна 2: H2D головы 19 мс на шаг с концом
# записи). Результат забирается без ожидания по событию в следующих шагах; ячейка головы освобождается только после
# чтения результата (иначе новая запись затёрла бы накопители, пока их читает голова). Не совместима с графами has_clef.
_ГОЛОВА_АСИНХ = os.environ.get("FA2SM70_CLEF_HEAD_ASYNC", "0") == "1"
_ГОЛОВА_ФАЙЛ = os.environ.get("FA2SM70_CLEF_HEAD_ASYNC_FILE", "")   # "0"/"1" -- A/B в одном экземпляре
_ГОЛОВА_ПЕРЕКЛ = {"t": 0.0, "вкл": True}
# [ОДИН ГРАФ 30.10] FA2SM70_CLEF_HEAD_ASYNC_GRAPH=1: асинхронная голова и при графах has_clef (tap -- на основном потоке).
_ГОЛОВА_С_ГРАФАМИ = os.environ.get("FA2SM70_CLEF_HEAD_ASYNC_GRAPH", "0") == "1"
_ЖДАТЬ_РЕЗУЛЬТАТ_С = float(os.environ.get("FA2SM70_CLEF_RESULT_WAIT_S",
                                         "60" if (_ГОЛОВА_АСИНХ or _ГОЛОВА_ФАЙЛ) else "2"))


def _голова_асинх():
    if not _ГОЛОВА_АСИНХ:
        return False
    if _ГОЛОВА_ФАЙЛ:
        сейчас = time.monotonic()
        if сейчас - _ГОЛОВА_ПЕРЕКЛ["t"] >= 1.0:
            _ГОЛОВА_ПЕРЕКЛ["t"] = сейчас
            try:
                with open(_ГОЛОВА_ФАЙЛ) as f:
                    _ГОЛОВА_ПЕРЕКЛ["вкл"] = f.read(1) != "0"
            except OSError:
                _ГОЛОВА_ПЕРЕКЛ["вкл"] = True
        return _ГОЛОВА_ПЕРЕКЛ["вкл"]
    return True
# [БЕЗ ХОСТА 30.10] Приём записи без cudaStreamSynchronize в req_begin (clef.cu, sync=false: закреплённый буфер ячейки +
# событие): FA2SM70_NOSYNC_ACC2=1 или уровень >= 1 в FA2SM70_NOSYNC_FILE (A/B в одном экземпляре).
_БХ_ФАЙЛ = os.environ.get("FA2SM70_NOSYNC_FILE", "")
_БХ = {"t": 0.0, "ур": 1 if os.environ.get("FA2SM70_NOSYNC_ACC2", "0") == "1" else 0}


def _без_хоста_уровень():
    if _БХ_ФАЙЛ:
        сейчас = time.monotonic()
        if сейчас - _БХ["t"] >= 1.0:
            _БХ["t"] = сейчас
            try:
                with open(_БХ_ФАЙЛ) as f:
                    v = f.read(1)
                _БХ["ур"] = int(v) if v.isdigit() else 0
            except OSError:
                pass
    return _БХ["ур"]


def _граф_включён():
    if not _ГРАФ_ГОТОВ[0]:
        return False
    if _ГРАФ_ФАЙЛ:
        # [ФРИЗЫ 30.10] значение, согласованное между рангами TP на шаге планировщика (fa2sm70_fullmix): граф и eager
        # ходят разными протоколами обмена -- расхождение рангов вело к ожиданию до срыва
        import fa2sm70_fullmix as _FMx
        return _FMx.значение_файла(_ГРАФ_ФАЙЛ) != "0"
    return True


def шаг_с_поправкой():
    """[ОДИН ГРАФ 30.10] Для конвейеров (blockpipe/rowpipe): поправка Clef в этом проходе нужна -- eager-шаг со строками
    Clef или шаг семейства has_clef (в т.ч. ЗАХВАТ графа, где строк ещё нет, но вызовы поправки обязаны попасть в граф)."""
    return bool(_АКТИВЕН[0]) or (_ГРАФ_ГОТОВ[0] and _шаг_графовый())


def _шаг_графовый():
    """Текущий проход -- шаг семейства has_clef (кусочный граф или eager с графовой поправкой)."""
    if not _ГРАФ_ГОТОВ[0]:
        return False
    from vllm.forward_context import get_forward_context, is_forward_context_available
    if not is_forward_context_available():
        return False
    bd = get_forward_context().batch_descriptor
    return bd is not None and getattr(bd, "has_clef", False)


def _лог(*a):
    print("[fa2_sm70 CLEF]", *a, file=sys.stderr, flush=True)


# ================================================================================================
# ВОРКЕР
# ================================================================================================
class _Состояние:
    def __init__(self):
        self.ext = None
        self.rank = 0
        self.golova = False
        self.slots = {}          # req_id -> (ячейка, ключ, N)
        self.free = []
        self.lm_head = None
        self.ok = False
        self.otkaz = set()       # req_id запросов, которым отказано (до их конца)
        # ВЫТЕСНЕНИЕ (14.10): req_id -> позиция, с которой обязан начаться следующий кусок. Вытесненный
        # запрос остаётся в runner.requests (ячейка цела), а после возврата его num_computed = 0
        # (skip_reading_prefix_cache): без сверки отвод сложил бы пройденные позиции в спаны второй раз.
        self.next = {}
        # асинхронная голова: [(ячейка, ключ, n, событие, хост-копия результата, [записан, попыток])] -- до публикации
        self.ждут = []
        # [ФРИЗЫ 30.10] последнее событие работы карты по ячейке (tap, голова, копия результата на хост -- побочный
        # поток): новый req_begin (основной поток) ставится ПОСЛЕ него ожиданием на карте -- без ожидания хоста.
        # Иначе он обнулил бы накопители и переписал данные запроса под ещё идущим tap прежней записи.
        self.tap_ev = {}
        self.поток_головы = None
        self.сборщик = None


_С = _Состояние()


def _сумма(ids):
    """контрольная сумма записи: эндпоинт и воркер считают её одинаково"""
    return zlib.crc32(array("i", ids).tobytes())

_РАЗБОР = re.compile(r"(?:^|\.)layers\.(\d+)\.(self_attn\.qkv_proj|self_attn\.o_proj|linear_attn\.in_proj_qkvz|"
                     r"linear_attn\.in_proj_ba|linear_attn\.out_proj|mlp\.gate_up_proj|mlp\.down_proj)$")
_MOD = {"self_attn.qkv_proj": 0, "linear_attn.in_proj_qkvz": 0, "linear_attn.in_proj_ba": 1,
        "self_attn.o_proj": 2, "linear_attn.out_proj": 2, "mlp.gate_up_proj": 3, "mlp.down_proj": 4}


def _хук(x, y, ключи):
    if _ГРАФ_ГОТОВ[0] and _шаг_графовый():
        from fa2sm70_clef_hook import row_window_offset
        offset = row_window_offset()
        if offset is not None and hasattr(_С.ext, "delta_keyed_graph_range"):
            # [ОДИН ГРАФ 30.10] окно строк в графе: строки по маске шага, не по списку
            _С.ext.delta_keyed_graph_range(x.reshape(-1, x.shape[-1]), y.view(-1, y.shape[-1]), ключи, offset)
            return
        _С.ext.delta_keyed_graph(x.reshape(-1, x.shape[-1]), y.view(-1, y.shape[-1]), ключи)
        return
    if not _АКТИВЕН[0]:
        return
    from fa2sm70_clef_hook import row_window_offset
    offset = row_window_offset()
    if offset is not None:
        _С.ext.delta_keyed_range(x.reshape(-1, x.shape[-1]), y.view(-1, y.shape[-1]), ключи, offset)
        return
    _С.ext.delta_keyed(x.reshape(-1, x.shape[-1]), y.view(-1, y.shape[-1]), ключи)


def _хук_диап(x, y, ключи, r0):
    from fa2sm70_clef_hook import row_window_offset
    offset = row_window_offset()
    if _ГРАФ_ГОТОВ[0] and _шаг_графовый() and hasattr(_С.ext, "delta_keyed_graph_range"):
        # [ОДИН ГРАФ 30.10] кусок конвейера в шаге has_clef (захват и повтор): маска строк шага на карте
        _С.ext.delta_keyed_graph_range(x.reshape(-1, x.shape[-1]), y.view(-1, y.shape[-1]), ключи,
                                       int(r0) + (offset if offset is not None else 0))
        return
    if not _АКТИВЕН[0]:
        return
    _С.ext.delta_keyed_range(x.reshape(-1, x.shape[-1]), y.view(-1, y.shape[-1]), ключи,
                            int(r0) + (offset if offset is not None else 0))


def _регистрировать_лин():
    """операция для неквантованного in_proj_ba: то же базовое тело, что у vLLM в компилируемой
    области (_maybe_sm70_dense_forward -> sm70_f16_gemm, иначе F.linear), плюс поправка"""
    import torch
    import torch.nn.functional as F
    from vllm.utils.torch_utils import direct_register_custom_op
    from fa2sm70_clef_hook import с_поправкой

    def _clef_lin(x: torch.Tensor, w: torch.Tensor, prepared: bool) -> torch.Tensor:
        if prepared:
            return torch.ops._C.sm70_f16_gemm(x, w)
        return F.linear(x, w)

    def _clef_lin_fake(x: torch.Tensor, w: torch.Tensor, prepared: bool) -> torch.Tensor:
        return x.new_empty((x.shape[0], w.shape[0]))

    direct_register_custom_op("fa2sm70_clef_lin", с_поправкой(_clef_lin, (1,)), mutates_args=[],
                              fake_impl=_clef_lin_fake)


def _обернуть_лин(inner, prepared):
    """quant_method слоя in_proj_ba: ПОДКЛАСС исходного метода (isinstance в vLLM не расходится),
    apply -- через fa2sm70_clef_lin, всё остальное -- состояние исходного объекта"""
    import torch

    def apply(self, layer, x, bias=None):
        x2 = x.reshape(-1, x.shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        y = torch.ops.vllm.fa2sm70_clef_lin(x2, layer.weight, self._clef_prepared)
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], y.shape[-1])

    cls = type("Clef" + type(inner).__name__, (type(inner),), {"apply": apply})
    obj = cls.__new__(cls)
    obj.__dict__.update(inner.__dict__)
    obj._clef_prepared = bool(prepared)
    return obj


def _настроить(runner):
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "clef"))
    import clef_golova
    import clef_sborka
    from fa2_sm70._ext import clef_ext

    rank, tp = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    _С.rank = rank
    cfg = runner.model_config.hf_text_config
    lt = list(getattr(cfg, "layer_types"))
    kd = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    vd = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    ext = clef_ext()
    t0 = time.time()
    занято0 = torch.cuda.memory_allocated()
    каталог = _DIR or os.path.join(runner.model_config.model, "clef")
    if _БЕЗ_ПОПРАВКИ or "nodelta" in _ДИАГ:
        sob = []
    else:
        deabl = os.path.join(каталог, _DEABL) if _DEABL else None
        sob = clef_sborka.sobrat(os.path.join(каталог, "clef_lora_r256_fp16.safetensors"), deabl, rank, tp,
                                 stream_from=_STREAM_FROM, layer_types=lt, key_dim=kd, value_dim=vd)
    max_rows = int(runner.scheduler_config.max_num_batched_tokens)
    clef_sborka.ustanovit(ext, sob, nslot=_NSLOT, max_rows=min(max_rows, 4096))
    ожидаемые = {}
    for L, st, host, ops in sob:
        for mod in set(int(m) for m in ops[:, 0].tolist()):
            ожидаемые[(L, mod)] = int((ops[:, 3] + ops[:, 5])[(ops[:, 0] == mod) & (ops[:, 1] > 0)].max())
    if _БЕЗ_BA:
        ожидаемые = {k: v for k, v in ожидаемые.items() if k[1] != 1}
    # ключи модулей: тензоры, которые тело умножения получает аргументами
    _регистрировать_лин()
    найдено, ширина_плохо = set(), []
    _гр_размеры = []
    for name, m in runner.model.named_modules():
        if "mtp" in name:
            continue
        mm = _РАЗБОР.search(name)
        if not mm:
            continue
        L, mod = int(mm.group(1)), _MOD[mm.group(2)]
        if mod == 1 and _БЕЗ_BA:
            continue
        if (L, mod) not in ожидаемые:
            continue
        шир = getattr(m, "output_size_per_partition", None)
        if mod in (2, 4):
            шир = getattr(m, "output_size", шир)
        _вх = getattr(m, "input_size_per_partition", None) or getattr(m, "input_size", 0)
        _гр_размеры.append((int(_вх or 0), int(шир or 0)))
        if шир is not None and ожидаемые[(L, mod)] > шир:
            ширина_плохо.append((name, ожидаемые[(L, mod)], шир))
        ключи = []
        for a in ("scales", "qweight", "w12_qw", "w12_sc"):
            t = getattr(m, a, None)
            if isinstance(t, torch.Tensor) and t.numel() > 0:
                ключи.append(t)
        pack = getattr(m, "fa2sm70_tm8", None)
        if pack is not None and isinstance(pack[0], torch.Tensor) and pack[0].numel() > 0:
            ключи.append(pack[0])
        if mod == 1:
            from vllm.model_executor.layers.linear import UnquantizedLinearMethod
            if not isinstance(m.quant_method, UnquantizedLinearMethod):
                raise RuntimeError(f"{name}: ожидался неквантованный in_proj_ba, а метод {type(m.quant_method)}")
            prepared = (getattr(m, "_sm70_f16_prepared", False) and not getattr(m, "_sm70_f16_forbidden", False)
                        and hasattr(torch.ops._C, "sm70_f16_gemm"))
            m.quant_method = _обернуть_лин(m.quant_method, prepared)
            ключи = [m.weight]
        if not ключи:
            raise RuntimeError(f"{name}: нет тензора-ключа")
        for k in ключи:
            ext.bind(k, L, mod)
        найдено.add((L, mod))
    нет = sorted(set(ожидаемые) - найдено)
    if нет or ширина_плохо:
        raise RuntimeError(f"не найдены модули {нет[:6]} / ширина {ширина_плохо[:3]}")
    _С.golova = rank == 0 and "nohead" not in _ДИАГ
    if _С.golova:
        clef_golova.zagruzit(ext, os.path.join(каталог, "joint_head_fp16.safetensors"), R=_R,
                             Nmax=_NMAX, Qomax=_QOMAX, Qnmax=_QNMAX)
        os.makedirs(_OUT, exist_ok=True)
        _С.lm_head = _ЧтецСловаря(runner.model_config.model)
    _С.free = list(range(_R))
    _С.ext = ext
    if _ГРАФ and hasattr(ext, "init_graph") and _гр_размеры:
        kmax = max(k for k, _ in _гр_размеры)
        nmax = max(n for _, n in _гр_размеры)
        ext.init_graph(_ГРАФ_МАКС, kmax, nmax, 4)
        _ГРАФ_ГОТОВ[0] = True
        _лог(f"графобезопасная поправка: шаги с Clef до {_ГРАФ_МАКС} токенов -- графами has_clef "
             f"(буферы {_ГРАФ_МАКС} x K {kmax} / N {nmax})")
    elif _ГРАФ:
        _лог("графобезопасная поправка НЕ включена: расширение без init_graph или нет модулей")
    import fa2sm70_clef_hook
    fa2sm70_clef_hook.ХУК[0] = _хук
    if hasattr(_С.ext, "delta_keyed_range"):
        fa2sm70_clef_hook.ХУК_ДИАП[0] = _хук_диап
    _С.ok = True
    torch.cuda.synchronize()
    # СЛЕД CLEF ВЫЧИТАЕТСЯ ИЗ ПОРОГА ПОДРЕЗКИ (12.10, стенд кандидата). Пул KV задан байтами
    # (--kv-cache-memory-bytes), поэтому всё, что занято здесь, берётся из запаса под рабочую память
    # шага: на ранге 0 свободно 398 МиБ вместо 634. Порог подрезки (FA2SM70_MEM_TRIM_MB=400) подобран
    # под запас базы, и смешанная нагрузка уронила движок: не нашлось 20 МиБ при 330 МиБ держанных,
    # но дробленых -- подрезка не сработала, потому что 330 < 400. Уменьшая порог на свой след,
    # возвращаем драйверу ту же гарантию свободной памяти, что у базы без рычага.
    след = (torch.cuda.memory_allocated() - занято0) / 2**20
    подрез = ""
    gmr = sys.modules.get("vllm.v1.worker.gpu_model_runner")
    порог = getattr(gmr, "_ПОДРЕЗ_МиБ", 0) if gmr is not None else 0
    if порог and "notrim" not in _ДИАГ:
        gmr._ПОДРЕЗ_МиБ = max(64, int(порог - след))
        подрез = f", порог подрезки {порог} -> {gmr._ПОДРЕЗ_МиБ} МиБ"
    _лог(f"ранг {rank}/{tp}: поправка на {len(найдено)} модулях {len(sob)} слоёв, голова={_С.golova}, "
         f"память карты {torch.cuda.memory_allocated() / 2**30:.2f} ГиБ занято торчем, "
         f"след Clef {след:.0f} МиБ{подрез}, {time.time() - t0:.1f} с; stats {ext.stats()}")


class _ЧтецСловаря:
    """строки lm_head с диска (вес словаря не меняется ни у ABL, ни у Clef)"""

    def __init__(self, model_dir):
        from safetensors import safe_open
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        путь = os.path.join(model_dir, idx["lm_head.weight"])
        self.f = safe_open(путь, "pt")
        self.sl = self.f.get_slice("lm_head.weight")
        # [ПАМЯТЬ 26.10] ЧТЕНИЕ СТРОК ОДНИМ ИНДЕКСИРОВАНИЕМ. Прежде mean() делал по тензору на КАЖДЫЙ токен спана
        # (self.sl[t:t+1], 20 КБ) и ещё временный [n, 5120] fp32 под cat -- тысячи мелких выделений на запись Clef
        # на ранге 0 (голова только там) при ~10 тыс. записей/ч и 198 потоках: дыры в аренах glibc, RSS Worker_TP0
        # рос ~0.6 ГБ/ч до oom-kill (26.10 02:42), у TP1 роста нет. Здесь -- отображение тензора lm_head из файла
        # (np.memmap по смещению из заголовка safetensors) и одна выборка строк: 2 выделения вместо n+1. Те же строки
        # fp16 -> .float() -> .mean(0) над тем же тензором [n, 5120] -> результат ПОБИТОВО прежний.
        # FA2SM70_CLEF_LEX_MMAP=0 -- прежний путь.
        self.mm = None
        if os.environ.get("FA2SM70_CLEF_LEX_MMAP", "1") == "1":
            try:
                import struct
                import numpy as np
                with open(путь, "rb") as fh:
                    n = struct.unpack("<Q", fh.read(8))[0]
                    hdr = json.loads(fh.read(n))
                t = hdr["lm_head.weight"]
                # BF16 (так в слепке) numpy не знает: отображаем как uint16 и переинтерпретируем в torch.bfloat16
                self.mm_dt = {"F16": None, "BF16": "bf16"}.get(t["dtype"], "нет")
                if self.mm_dt != "нет" and len(t["shape"]) == 2:
                    a, b = t["data_offsets"]
                    assert b - a == t["shape"][0] * t["shape"][1] * 2
                    self.mm = np.memmap(путь, dtype=np.float16 if self.mm_dt is None else np.uint16, mode="r",
                                        offset=8 + n + a, shape=tuple(t["shape"]))
            except Exception as ex:  # noqa: BLE001
                _лог(f"строки словаря: отображение не вышло ({type(ex).__name__}: {ex}) -- прежний путь")
                self.mm = None

    def mean(self, ids):
        import torch
        if self.mm is not None:
            import numpy as np
            rows = torch.from_numpy(np.asarray(self.mm[np.asarray(ids, dtype=np.int64)]))
            if self.mm_dt == "bf16":
                rows = rows.view(torch.bfloat16)
            return rows.float().mean(0)
        rows = [self.sl[t:t + 1].float() for t in ids]
        return torch.cat(rows).mean(0)


def _приём(runner, rid, req, c):
    """новый запрос Clef: ячейка головы и её данные (ранг 0)"""
    import torch
    N = int(req.num_prompt_tokens)
    # сверка ДО выдачи ячейки и на ОБОИХ рангах: отказ симметричен, поправка не ляжет ни на одном
    # [ПАМЯТЬ 26.10] без list(): array("i", ...) берёт последовательность как есть -- без полной копии промпта
    if "n" in c and (N != int(c["n"]) or _сумма(req.prompt_token_ids) != int(c["h"])):
        raise ValueError(f"запись в движке ({N} токенов) не равна записи эндпоинта ({c['n']})")
    # [ФРИЗЫ 30.10] ячейки освобождаются сразу и на обоих рангах (повтор упорядочен на карте, см. ниже) -- ожидания
    # хоста здесь нет: декод не останавливается; списки свободных ячеек рангов TP совпадают (отказ симметричен).
    # [ревизия] проверки по данным запроса -- ТОЖЕ на обоих рангах и до выдачи ячейки: отказ только на ранге 0 (у ранга 1
    # ячейка уже выдана) развёл бы сегменты шага у рангов -> разные пути и число сведений -> зависание
    spans, counts, types = c["spans"], c["counts"], c["types"]
    Qn, Qo = len(counts), sum(counts)
    if N > _NMAX or Qn > _QNMAX or Qo > _QOMAX or len(spans) != Qn + Qo or not counts or max(counts) > 256:
        raise ValueError(f"запрос вне пределов: N={N} Qn={Qn} Qo={Qo}")
    for s, (a, b) in enumerate(spans):
        if not (0 <= a < b <= N):
            raise ValueError(f"спан {s} [{a},{b}) вне [0,{N})")
    if not _С.free:
        return None
    slot = _С.free.pop(0)
    _С.slots[rid] = (slot, c.get("key", rid), N)
    if not _С.golova:
        return slot
    # [ПАМЯТЬ 26.10] постоянный буфер вместо torch.full((N,)) на каждую запись (до 136 КБ -- у порога mmap glibc,
    # источник дыр). req_begin копирует из обычной памяти (cudaMemcpyAsync с pageable забирает данные до возврата),
    # значит буфер свободен сразу после вызова. Содержимое то же: -1 и номера спанов.
    if getattr(_С, "spanid_buf", None) is None or _С.spanid_buf.numel() < N:
        _С.spanid_buf = torch.empty(max(N, _NMAX), dtype=torch.int32)
    spanid = _С.spanid_buf[:N]
    spanid.fill_(-1)
    for s, (a, b) in enumerate(spans):
        if not (0 <= a < b <= N):
            raise ValueError(f"спан {s} [{a},{b}) вне [0,{N})")
        spanid[a:b] = s
    inv_len = torch.tensor([1.0 / (b - a) for a, b in spans], dtype=torch.float32)
    qidx = torch.tensor(sum(([i] * k for i, k in enumerate(counts)), []), dtype=torch.int32)
    qoff = torch.tensor([0] + list(torch.tensor(counts).cumsum(0).tolist()), dtype=torch.int32)
    ids = req.prompt_token_ids   # [ПАМЯТЬ 26.10] без list(): нужны только срезы спанов
    lex = torch.stack([_С.lm_head.mean(ids[a:b]) for a, b in spans[Qn:]]).half()
    ev = _С.tap_ev.get(slot)
    if ev is not None:
        # прежняя запись ячейки (tap/голова/копия результата на побочном потоке) -- до обнуления накопителей и данных:
        # ожидание на КАРТЕ, хост не ждёт
        torch.cuda.current_stream().wait_event(ev)
    if _без_хоста_уровень() >= 1 and "sync" in (getattr(_С.ext.req_begin, "__doc__", "") or ""):
        _С.ext.req_begin(slot, N, spanid, inv_len, qidx, qoff, torch.tensor(types, dtype=torch.int32), lex, False)
    else:
        _С.ext.req_begin(slot, N, spanid, inv_len, qidx, qoff, torch.tensor(types, dtype=torch.int32), lex)
    return slot


def _записать(ключ, данные):
    p = os.path.join(_OUT, ключ + ".json")
    with open(p + ".tmp", "w") as f:
        json.dump(данные, f)
    os.replace(p + ".tmp", p)


def _шаг(runner, so):
    """после _update_states: приём/уход запросов Clef и сегменты строк на этот шаг"""
    import numpy as np
    runner._clef_segs = None
    _собрать_результаты()
    for rid in list(_С.slots):
        if rid in so.finished_req_ids or rid not in runner.requests:
            slot = _С.slots.pop(rid)[0]
            _С.next.pop(rid, None)
            _освободить_ячейку(slot)
    if _С.otkaz:
        _С.otkaz = {r for r in _С.otkaz if r in runner.requests and r not in so.finished_req_ids}
    segs, row0 = [], 0
    ib = runner.input_batch
    for i, rid in enumerate(ib.req_ids):
        n = int(so.num_scheduled_tokens.get(rid, 0))
        req = runner.requests.get(rid)
        sp = getattr(req, "sampling_params", None) if req is not None else None
        c = (sp.extra_args or {}).get("fa2sm70_clef") if sp is not None and sp.extra_args else None
        if c is not None and n > 0:
            pos0 = int(ib.num_computed_tokens_cpu[i])
            if rid in _С.otkaz:
                row0 += n
                continue
            if _A1 and rid in _С.slots and pos0 != _С.next.get(rid, 0):
                # кусок не продолжает уже сложенное (вытеснение и возврат с нуля): ячейку отпустить
                # и принять запрос заново -- req_begin обнулит спаны; pos0 != 0 -- отказ ниже
                slot = _С.slots.pop(rid)[0]
                _лог(f"запрос {rid}: кусок с позиции {pos0}, а сложено до {_С.next.pop(rid, 0)} -- приём заново")
                _освободить_ячейку(slot)
            if rid not in _С.slots:
                try:
                    if pos0 != 0:
                        raise ValueError(f"первый кусок с позиции {pos0}: префикс взят из кэша")
                    if _приём(runner, rid, req, c) is None:
                        raise ValueError("нет свободной ячейки головы")
                except Exception as e:  # noqa: BLE001
                    _лог(f"отказ запросу {rid}: {e}")
                    _С.otkaz.add(rid)
                    if _С.golova:
                        _записать(c.get("key", rid), {"error": str(e)})
                    row0 += n
                    continue
            slot, key, N = _С.slots[rid]
            fin = int(pos0 + n >= N)
            segs.append((row0, n, slot, pos0, fin))
            _С.next[rid] = pos0 + n
        row0 += n
    if segs:
        runner._clef_segs = np.array(segs, dtype=np.int32)


def _освободить_ячейку(slot):
    """Ячейка головы -- сразу в свободные (на обоих рангах одинаково). Её незавершённая работа на побочном потоке
    (tap, голова, копия результата) упорядочена с новым приёмом ожиданием на карте (_приём: wait_event по tap_ev);
    результат уже копируется на хост в собственный буфер записи -- ячейка ему не нужна."""
    if _С.golova:
        _С.ext.req_end(slot)
    _С.free.append(slot)


# RLock: тот же замок держит захват графа (ленивые графы цикла MTP) -- см. _замок_захвата
_ЗАМОК_ГОЛОВЫ = __import__("threading").RLock()
_ПОПЫТОК_ЗАПИСИ = 500       # ~1 с повторов (тик сборщика 2 мс), затем ошибка в ответ
# [ФРИЗЫ 30.10] Список готовящихся результатов -- под своим коротким замком. Публикация (копия на хост, сериализация,
# запись файла) -- только в сборщике и ВНЕ замков: основной поток не ждёт ни .cpu(), ни диск, ни сборщика.
_ЗАМОК_СПИСКА = __import__("threading").Lock()


def _замок_захвата():
    """Захват графа и CUDA-вызовы сборщика взаимно исключены. Синхронизирующая копия из другого потока ломает
    глобальный захват МОЛЧА, а захват графа цикла MTP (eagle) к тому же ставит на процесс sync_debug_mode "error":
    .cpu() сборщика в этот миг бросал «called a synchronizing CUDA operation» (окно 10: HTTP 500 записи Clef)."""
    import torch
    G = torch.cuda.graphs.graph
    if getattr(G, "_fa2_clef_lock", False):
        return
    вход, выход = G.__enter__, G.__exit__

    def __enter__(self):
        _ЗАМОК_ГОЛОВЫ.acquire()
        try:
            return вход(self)
        except BaseException:
            _ЗАМОК_ГОЛОВЫ.release()
            raise

    def __exit__(self, *a):
        try:
            return выход(self, *a)
        finally:
            _ЗАМОК_ГОЛОВЫ.release()

    G.__enter__, G.__exit__ = __enter__, __exit__
    G._fa2_clef_lock = True


def _собрать_результаты():
    """Основной поток: только учёт -- записанные сборщиком результаты снимаются из списка. Ни копий, ни файлов,
    ни ожидания."""
    if not _С.ждут:
        return
    with _ЗАМОК_СПИСКА:
        _С.ждут = [item for item in _С.ждут if not item[5][0]]


def _сборщик_голов():
    """Фон rank 0: пока движок простаивает (шагов нет), результаты всё равно пишутся -- API ждёт файл."""
    import torch
    torch.cuda.set_device(_С.поток_головы.device)
    while True:
        time.sleep(0.002)
        if not _С.ждут:
            continue
        with _ЗАМОК_СПИСКА:
            работа = [it for it in _С.ждут if not it[5][0]]
        try:
            _опубликовать(работа)
        finally:
            # [ревизия] последние ссылки на закреплённую хост-копию и событие могут быть здесь: их освобождение -- тоже
            # вызовы CUDA (кэш закреплённой памяти отмечает событие, cudaEventDestroy) -- под замком захвата
            with _ЗАМОК_ГОЛОВЫ:
                работа = None


def _опубликовать(работа):
    """Сборщик: готовые результаты (событие прошло) -- в файлы. Ссылки на элементы держит вызывающий список."""
    for item in работа:
        slot, key, n, ev, хост, записан = item
        try:
            # CUDA-вызовы сборщика -- опрос события (здесь) и освобождение хост-копии/события (у вызывающего) -- под
            # замком захвата: вызов из другой нити во время глобального захвата графа срывает захват (окно 24:
            # ev.query() вне замка -> захват графа цикла MTP сорван, пул захвата испорчен, OOM, движок упал).
            # Результат уже на хосте (копия на побочном потоке сразу за головой); сериализация и диск -- вне замков.
            with _ЗАМОК_ГОЛОВЫ:
                if not ev.query():
                    continue
            _записать(key, {"logits": хост.float().tolist(), "n": n})
            записан[0] = True
        except Exception as e:  # noqa: BLE001
            # режим sync_debug "error" чужого захвата (вне самого захвата) -- повторить на следующем тике
            записан[1] += 1
            if записан[1] >= _ПОПЫТОК_ЗАПИСИ:
                _записать(key, {"error": f"голова: {e}"})
                записан[0] = True


def _после_прохода(runner, segs, out):
    import torch
    hs = out[0] if isinstance(out, (tuple, list)) else out
    if _С.golova and (not _ГРАФ_ГОТОВ[0] or _ГОЛОВА_С_ГРАФАМИ) and _голова_асинх():
        if _С.поток_головы is None:
            _С.поток_головы = torch.cuda.Stream(device=hs.device)
        if _С.сборщик is None:
            _замок_захвата()
            _С.сборщик = __import__("threading").Thread(target=_сборщик_голов, name="clef-head-collect", daemon=True)
            _С.сборщик.start()
        поб = _С.поток_головы
        if _ГРАФ_ГОТОВ[0] and hasattr(_С.ext, "run_pending_heads"):
            # [ОДИН ГРАФ 30.10] hs может быть статическим выходом графа (следующий повтор его перезапишет): tap -- на
            # основном потоке, только сбор строк Clef в накопители; [ФРИЗЫ 30.10] головы концов записей -- на побочном
            _С.ext.tap(hs, False)
            поб.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(поб):
                _С.ext.run_pending_heads()
        elif _ГРАФ_ГОТОВ[0]:
            _С.ext.tap(hs)                     # прежняя сборка clef.cu: голова внутри tap (на основном потоке)
            поб.wait_stream(torch.cuda.current_stream())
        else:
            поб.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(поб):
                _С.ext.tap(hs)
            hs.record_stream(поб)
        _С.ext.set_step(torch.empty(0, 5, dtype=torch.int32))
        плохо = _С.ext.step_bad()
        готовые = []
        for row0, n, slot, pos0, fin in segs.tolist():
            if not fin:
                continue
            key = next((k for s, k, _ in _С.slots.values() if s == slot), None)
            if key is None:
                continue
            if _С.ext.ready(slot) and not плохо:
                # результат -- на хост сразу за головой, на том же побочном потоке: ячейка (E.result) свободна для
                # следующей записи, как только поток прошёл копию; сборщик читает только хост
                r = _С.ext.result(slot)
                хост = torch.empty(r.shape, dtype=r.dtype, pin_memory=True)
                with torch.cuda.stream(поб):
                    хост.copy_(r, non_blocking=True)
                готовые.append((slot, key, pos0 + n, хост))
            else:
                _записать(key, {"error": f"голова не посчитана: {_С.ext.last_error()}"})
        ev_шага = torch.cuda.Event()           # tap, головы и копии результатов шага прошли
        ev_шага.record(поб)
        for _r0, _n, _slot, _p0, _fin in segs.tolist():
            _С.tap_ev[_slot] = ev_шага
        if готовые:
            with _ЗАМОК_СПИСКА:
                for slot, key, n_, хост in готовые:
                    _С.ждут.append((slot, key, n_, ev_шага, хост, [False, 0]))
        _собрать_результаты()
        return
    if _С.поток_головы is not None:
        # [ФРИЗЫ 30.10] tap и головы прошлых шагов на побочном потоке (в т. ч. промежуточных кусков, у которых нет
        # записи в ждут) -- до синхронного tap на основном: общий E.ws_hn и накопители. Ожидание на карте.
        torch.cuda.current_stream().wait_stream(_С.поток_головы)
    if _С.ждут:
        _собрать_результаты()
    if _С.golova:
        _С.ext.tap(hs)
    _С.ext.set_step(torch.empty(0, 5, dtype=torch.int32))
    if not _С.golova:
        return
    плохо = _С.ext.step_bad()
    for row0, n, slot, pos0, fin in segs.tolist():
        if not fin:
            continue
        key = next((k for s, k, _ in _С.slots.values() if s == slot), None)
        if key is None:
            continue
        if _С.ext.ready(slot) and not плохо:
            _записать(key, {"logits": _С.ext.result(slot).float().cpu().tolist(), "n": pos0 + n})
        else:
            _записать(key, {"error": f"голова не посчитана: {_С.ext.last_error()}"})


def установить_воркер():
    from vllm.v1.worker import gpu_model_runner as gmr
    R = gmr.GPUModelRunner
    _load, _upd = R.load_model, R._update_states
    _det, _fwd = R._determine_batch_execution_and_padding, R._model_forward

    def load_model(self, *a, **kw):
        r = _load(self, *a, **kw)
        if "nosetup" in _ДИАГ:
            for x in _ДИАГ:
                if x.startswith("dummy"):
                    import torch
                    self._clef_dummy = torch.empty(int(x[5:]) * 2**20, dtype=torch.uint8, device="cuda")
                    self._clef_dummy.fill_(0)
                    torch.cuda.synchronize()
            _лог(f"ДИАГНОСТИКА {sorted(_ДИАГ)}: настройка пропущена, подмены раннера стоят")
            return r
        try:
            _настроить(self)
        except Exception:  # noqa: BLE001
            _лог("НАСТРОЙКА НЕ УДАЛАСЬ -- Clef выключен, базовая сеть работает как есть:\n" + traceback.format_exc())
            _С.ok = False
        return r

    def _update_states(self, so):
        r = _upd(self, so)
        if _С.ok:
            _шаг(self, so)
        return r

    def _determine_batch_execution_and_padding(self, *a, **kw):
        if getattr(self, "_clef_segs", None) is not None:
            if _граф_включён():
                kw["force_has_clef"] = True      # кусочный граф has_clef; больше FA2SM70_CLEF_GRAPH_MAX -- eager
            else:
                kw["force_eager"] = True
        return _det(self, *a, **kw)

    def _model_forward(self, *a, **kw):
        segs = getattr(self, "_clef_segs", None)
        if segs is None:
            return _fwd(self, *a, **kw)
        import torch
        self._clef_segs = None
        _С.ext.set_step(torch.from_numpy(segs))
        if _ГРАФ_ГОТОВ[0] and _шаг_графовый():
            import numpy as np
            строки = np.concatenate([np.arange(int(r0), int(r0) + int(n), dtype=np.int32)
                                     for r0, n in segs[:, :2]]) if len(segs) else np.zeros(0, np.int32)
            _С.ext.graph_set_rows(torch.from_numpy(строки))
        _АКТИВЕН[0] = True
        _СЕГМЕНТЫ[0] = segs
        try:
            out = _fwd(self, *a, **kw)
        finally:
            _АКТИВЕН[0] = False
            _СЕГМЕНТЫ[0] = None
        _после_прохода(self, segs, out)
        return out

    R.load_model = load_model
    R._update_states = _update_states
    R._determine_batch_execution_and_padding = _determine_batch_execution_and_padding
    R._model_forward = _model_forward


# ================================================================================================
# СЕРВЕР: /v1/systemone
# ================================================================================================
SYSTEM_PROMPT = ("Read the complete state and schema. Decide every field jointly. Each answer "
                 "must be exactly one of that field's allowed options.")
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}


def _render(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _варианты(q):
    t = str(q["type"])
    if t == "noul":
        cr = {"true": "The proposition is true or the answer is yes.",
              "false": "The proposition is false or the answer is no."}
        cr.update(q.get("criteria") or {})
        return [(k, cr[k]) for k in ("true", "false")]
    if t == "choice":
        return sorted((str(k), v) for k, v in q["criteria"].items())
    return [(str(i), v) for i, v in enumerate(q["criteria"])]


def _токены_картинок(ip, imgs, mk):
    """число токенов каждой картинки процессором изображений сети. Загрузчик vLLM отдаёт картинку в обёртке
    MediaWithBytes (PIL + исходные байты, для хеша кэша); процессор transformers её не принимает -- снимаем
    обёртку здесь, а движку уходит она сама, как у чата."""
    m2 = int(ip.merge_size) ** 2
    return [int(ip(images=[getattr(im, "media", im)], return_tensors="pt", **mk)["image_grid_thw"].prod()) // m2
            for im in imgs]


def _медиа_ид(tok, counts):
    """токены картинок, как их даёт _encode_media Clef: (свёрнутые -- движку, развёрнутые -- для спанов)"""
    base = tok(IMAGE_PLACEHOLDER * len(counts) + "\n", add_special_tokens=False).input_ids
    pad = tok.convert_tokens_to_ids("<|image_pad|>")
    full, i = [], 0
    for t in base:
        if t == pad:
            full += [pad] * counts[i]
            i += 1
        else:
            full.append(t)
    if i != len(counts):
        raise ValueError("image placeholder tokenization")
    return base, full


_РАЗРЕЗ_СТРОКИ = re.compile(r"\n(?=\S)")


def _T_префикс(T, text, K):
    """T(text)[:K] без токенизации всего text. Префикс режется ТОЛЬКО после "\\n" перед непробельным символом --
    граница предразбиения Qwen, BPE её не пересекает: T(text[:b]) -- ровно начало T(text). Префикс удлиняется вдвое,
    пока токенов меньше K; нет границы -- целиком, как раньше. Короткий text -- целиком (прежний путь)."""
    if K <= 0:
        return []
    if len(text) <= max(8 * K, 1 << 16):
        return T(text)[:K]
    cut = 4 * K
    while cut < len(text):
        m = _РАЗРЕЗ_СТРОКИ.search(text, cut)
        if m is None:
            break
        part = T(text[:m.end()])
        if len(part) >= K:
            return part[:K]
        cut = 2 * m.end()
    return T(text)[:K]


def закодировать(tok, record, max_length=16384, media=None):
    """повтор encode_record из релиза Clef; media = _медиа_ид(...) или None.
    Возвращает (запись для спанов, вопросы, запись для движка) -- без картинок обе записи равны."""
    T = lambda s: tok(s, add_special_tokens=False).input_ids  # noqa: E731
    schema = T("\n\nSCHEMA FIELDS:\n")
    qs = []
    for qi, (qid, q) in enumerate(record["questions"].items()):
        schema += T(f"\nFIELD {qi + 1}\nID: {qid}\nTYPE: {q['type']}\nINSTRUCTION: ")
        a = len(schema)
        ins = q.get("instructions")
        if ins is None or ins == "":
            ins = str(qid)
        schema += T(_render(ins))
        qspan = (a, len(schema))
        schema += T("\nALLOWED OPTIONS:\n")
        ospans, oids = [], []
        for oi, (oid, desc) in enumerate(_варианты(q)):
            schema += T(f"OPTION {oi + 1}: ")
            s0 = len(schema)
            sem = {"option_id": oid}
            if desc is not None:
                sem["description"] = desc
            schema += T(_render(sem))
            ospans.append((s0, len(schema)))
            oids.append(oid)
            schema += T("\n")
        schema += T("END FIELD\n")
        qs.append((str(qid), QUESTION_TYPES[str(q["type"])], qspan, ospans, oids))
    prefix = T(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
    suffix = T("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
    m_eng, m_full = media if media is not None else ([], [])
    fixed = len(prefix) + len(m_full) + len(schema) + len(suffix)
    if fixed > max_length:
        raise ValueError(f"schema requires {fixed} tokens before state; maximum is {max_length}")
    # [ПАМЯТЬ 27.10] только нужный префикс state (было: T(весь state)[:K] -- десятки МБ текста токенизировались
    # целиком в цикле событий ради <= 34K токенов: всплеск ГБ и стоп API-сервера). Побитово то же, см. _T_префикс.
    state = _T_префикс(T, _render(record["state"]), max_length - fixed)
    off = len(prefix) + len(m_full) + len(state)
    ids = prefix + m_full + state + schema + suffix
    eng = prefix + m_eng + state + schema + suffix
    qs = [(qid, t, (a + off, b + off), [(s + off, e + off) for s, e in osp], oids) for qid, t, (a, b), osp, oids in qs]
    return ids, qs, eng


def _ответ(q, p):
    if q["type"] == "noul":
        return {"type": "noul", "noul": round(p["true"], 4)}
    if q["type"] == "choice":
        opts = [str(o) for o in q["criteria"]]
        ch = max(opts, key=p.__getitem__)
        return {"type": "choice", "choice": ch, "confidence": round(p[ch], 4),
                "probabilities": {o: round(p[o], 4) for o in opts}}
    lv = [str(i) for i in range(len(q["criteria"]))]
    return {"type": "score", "score": round(sum(i * p[l] for i, l in enumerate(lv)), 4),
            "confidence": round(max(p[l] for l in lv), 4), "legend": dict(zip(lv, q["criteria"])),
            "probabilities": {l: round(p[l], 4) for l in lv}}


def _url(x):
    """картинка записи: строка (http(s)/data-URL или голый base64) либо {"url"} / {"image_url": ...}"""
    if isinstance(x, dict):
        x = x.get("image_url", x.get("url"))
        if isinstance(x, dict):
            x = x.get("url")
    if not isinstance(x, str) or not x:
        raise ValueError("each image must be a URL, a data URL or base64 string")
    if x.startswith(("http://", "https://", "data:")):
        return x
    return "data:image/png;base64," + x


def установить_сервер():
    import asyncio
    import uuid
    from vllm.entrypoints.openai.generate import api_router as gen

    _исх = gen.register_generate_api_routers
    _сем = [None]
    _вход = [None]
    _ДОПУСК = int(os.environ.get("FA2SM70_CLEF_ADMIT", str(2 * _R)))
    _мед = [None, None]          # загрузчик картинок (как у чата), процессор изображений

    def _загрузчик(ec):
        if _мед[0] is None:
            from vllm import envs
            from vllm.multimodal.media import MEDIA_CONNECTOR_REGISTRY
            mc = ec.model_config
            _мед[0] = MEDIA_CONNECTOR_REGISTRY.load(
                envs.VLLM_MEDIA_CONNECTOR,
                media_io_kwargs=getattr(mc.multimodal_config, "media_io_kwargs", None),
                allowed_local_media_path=mc.allowed_local_media_path or "",
                allowed_media_domains=mc.allowed_media_domains)
        return _мед[0]

    def _токенов_картинки(ec, pil, mk):
        """число токенов каждой картинки -- тем же процессором изображений, что у движка и у Clef"""
        if _мед[1] is None:
            from transformers import AutoImageProcessor
            _мед[1] = AutoImageProcessor.from_pretrained(ec.model_config.model)
        return _токены_картинок(_мед[1], pil, mk)

    def register_generate_api_routers(app):
        _исх(app)
        from fastapi import Request
        from fastapi.responses import JSONResponse

        # [НАГРУЗКА 06.11] ПРЕДЕЛ ОДНОВРЕМЕННЫХ ЗАПРОСОВ НА КЛИЕНТА (fa2sm80_clientlimit, FA2SM80_CLIENT_LIMIT): самый
        # внутренний слой -- ждущий запрос виден прибору ниже как «в обработке», а тело его ещё не прочитано.
        _вернуть_память = None
        try:
            import fa2sm80_clientlimit

            fa2sm80_clientlimit.поставить(app, _лог)
            _вернуть_память = fa2sm80_clientlimit.вернуть_память
        except Exception as e:  # noqa: BLE001
            _лог(f"предел клиентов НЕ поставлен: {type(e).__name__}: {e}")

        # [ПАМЯТЬ 27.10] ОХРАНА ОТ ОГРОМНЫХ ТЕЛ (fa2sm80_bigguard): тела >= 16 МБ по одному, проверка памяти до
        # чтения тела, нехватка -> ожидание, потом 503. Ставится РАНЬШЕ прибора ниже: прибор -- внешний слой (ВХОД
        # пишется сразу, ВЫХОД -- с итоговым кодом, в т.ч. 503).
        try:
            import fa2sm80_bigguard

            fa2sm80_bigguard.поставить(app, _лог)
        except Exception as e:  # noqa: BLE001
            _лог(f"охрана огромных тел НЕ поставлена: {type(e).__name__}: {e}")

        # [ПАМЯТЬ 27.10] ПРИБОР ВХОДЯЩИХ ЗАПРОСОВ API-сервера (диагностика, без ограничений для клиентов). API-сервер
        # раздувается до 9-12 ГБ за минуты после старта (oom-kill по кругу); виновника не видно в журнале доступа
        # (строка пишется при ответе). Пишем: крупные запросы при поступлении/ответе; и при скачке RSS > 1 ГБ за 10 с --
        # список запросов в обработке (путь, размер, возраст).
        import threading as _thr
        _в_обработке = {}
        _БОЛЬШОЙ_ВХОД = int(os.environ.get("FA2SM80_REQLOG_KB", "512")) * 1024

        def _rss_мб():
            try:
                return int(open("/proc/self/status").read().split("RssAnon:")[1].split()[0]) // 1024
            except Exception:  # noqa: BLE001
                return -1

        @app.middleware("http")
        async def _fa2_вход(request, call_next):
            cl = int(request.headers.get("content-length") or 0)
            кл = request.client.host if request.client else "?"
            ключ = id(request)
            _в_обработке[ключ] = (request.method, request.url.path, cl, time.time(), кл)
            большой = cl >= _БОЛЬШОЙ_ВХОД
            if большой:
                _лог(f"ВХОД {request.method} {request.url.path} {cl / 2**20:.1f} МБ от {кл}, RSS API {_rss_мб()} МБ")
            t0 = time.time()
            try:
                resp = await call_next(request)
            finally:
                _в_обработке.pop(ключ, None)
            dt = time.time() - t0
            if большой or dt > 60:
                _лог(f"ВЫХОД {request.url.path} {cl / 2**20:.1f} МБ: {resp.status_code} за {dt:.1f} с, RSS API {_rss_мб()} МБ")
            if большой and _вернуть_память is not None:
                # [ПАМЯТЬ 08.11] тело 8.4 МБ (ответ 400) оставило API +740 МБ: вернуть освобождённое системе
                try:
                    _вернуть_память(_лог)
                except Exception:  # noqa: BLE001
                    pass
            return resp

        def _сторож():
            прежн = _rss_мб()
            while True:
                time.sleep(10)
                r = _rss_мб()
                if r - прежн > 1024:
                    сейчас = time.time()
                    спис = sorted(_в_обработке.values(), key=lambda v: -v[2])[:12]
                    _лог(f"СКАЧОК RSS API {прежн} -> {r} МБ за 10 с; в обработке {len(_в_обработке)}: " + "; ".join(
                        f"{m} {pth} {c / 2**20:.1f}МБ {сейчас - t:.0f}с {k}" for m, pth, c, t, k in спис))
                прежн = r

        _thr.Thread(target=_сторож, name="fa2-reqwatch", daemon=True).start()

        # [ПАМЯТЬ 26.10] ШЛЮЗ ДОПУСКА. Семафор движка (_сем, FA2SM70_CLEF_R) стоял ПОСЛЕ разбора тела, загрузки картинок
        # и токенизации: после падения клиент присылал накопленное разом, сотни запросов ждали семафор, держа списки
        # токенов и раскодированные изображения -- API-сервер 1.3 -> 4.3 ГБ за 10 мин, cgroup 14 ГБ, oom-kill воркера,
        # и так по кругу (26.10 20:54 -> 23:11, 13 перезапусков). Теперь до шлюза запрос -- лёгкая корутина с НЕпрочитанным
        # телом; разбор и подготовка -- только у FA2SM70_CLEF_ADMIT допущенных (по умолчанию 2*R: подготовка следующих
        # идёт, пока считаются текущие). Отказов нет -- поведение для клиента прежнее, меняется только память.
        @app.post("/v1/systemone")
        async def systemone(raw: Request):
            if _вход[0] is None:
                _вход[0] = asyncio.Semaphore(_ДОПУСК)
            async with _вход[0]:
                return await _systemone(raw)

        async def _systemone(raw):
            from vllm.inputs import TokensPrompt
            from vllm.sampling_params import SamplingParams
            try:
                body = await raw.json()
                qs = body.get("questions")
                if not isinstance(body.get("model"), str) or "state" not in body:
                    raise ValueError("model and state are required")
                if not isinstance(qs, dict) or not qs:
                    raise ValueError("at least one question is required")
                if body.get("videos"):
                    raise ValueError("videos are not supported by this endpoint yet")
                imgs = body.get("images") or []
                if imgs and not _MEDIA:
                    raise ValueError("images are not supported by this endpoint")
                if not isinstance(imgs, list) or len(imgs) > _IMGMAX:
                    raise ValueError(f"images must be a list of at most {_IMGMAX} items")
                mk = body.get("media_kwargs") or {}
                if not isinstance(mk, dict):
                    raise ValueError("media_kwargs must be an object")
                for qid, q in qs.items():
                    if q.get("type") not in QUESTION_TYPES:
                        raise ValueError(f"{qid}: type must be noul, choice, or score")
                    if q["type"] != "noul" and not q.get("criteria"):
                        raise ValueError(f"{qid}: criteria must not be empty")
                ec = raw.app.state.engine_client
                tok = ec.get_tokenizer()
                media, pil = None, []
                if imgs:
                    con = _загрузчик(ec)
                    pil = [await con.fetch_image_async(_url(x)) for x in imgs]
                    media = _медиа_ид(tok, await asyncio.to_thread(_токенов_картинки, ec, pil, mk))
                ids, enc, eng = закодировать(tok, body, min(int(body.get("max_length", _NMAX)), _NMAX), media)
            except Exception as e:  # noqa: BLE001
                return JSONResponse({"error": {"message": str(e), "type": "invalid_request_error"}}, status_code=400)
            key = uuid.uuid4().hex
            _ТОК["записей"] += 1
            _ТОК["вход"] += len(eng)
            if time.time() - _ТОК["t"] >= _ТОК_ПЕРИОД:
                _ТОК["t"] = time.time()
                _лог(f"итого Clef с пуска: записей {_ТОК['записей']}, входящих токенов {_ТОК['вход']}")
            spans = [list(q[2]) for q in enc] + [list(s) for q in enc for s in q[3]]
            extra = {"fa2sm70_clef": {"key": key, "spans": spans, "counts": [len(q[3]) for q in enc],
                                      "types": [q[1] for q in enc], "n": len(ids), "h": _сумма(ids)}}
            prompt = TokensPrompt(prompt_token_ids=eng, cache_salt=_SALT)
            if pil:
                prompt["multi_modal_data"] = {"image": pil}
                if mk:
                    prompt["mm_processor_kwargs"] = mk
            sp = SamplingParams(max_tokens=1, temperature=0.0, extra_args=extra)
            sp.skip_reading_prefix_cache = True
            if _сем[0] is None:
                _сем[0] = asyncio.Semaphore(_R)
            путь = os.path.join(_OUT, key + ".json")
            async with _сем[0]:
                async for _ in ec.generate(prompt, sp, request_id=f"systemone-{key}"):
                    pass
                for _ in range(max(1, int(_ЖДАТЬ_РЕЗУЛЬТАТ_С / 0.01))):
                    if os.path.exists(путь):
                        break
                    await asyncio.sleep(0.01)
            try:
                res = json.load(open(путь))
                os.remove(путь)
            except Exception as e:  # noqa: BLE001
                return JSONResponse({"error": {"message": f"no result: {e}", "type": "server_error"}}, status_code=500)
            if "error" in res:
                return JSONResponse({"error": {"message": res["error"], "type": "server_error"}}, status_code=500)
            lg, i, answers = res["logits"], 0, {}
            for qid, _t, _qs, osp, oids in enc:
                z = lg[i:i + len(oids)]
                i += len(oids)
                m = max(z)
                e = [math.exp(v - m) for v in z]
                s = sum(e)
                answers[qid] = _ответ(qs[qid], dict(zip(oids, [v / s for v in e])))
            return {"model": body["model"], "answers": answers,
                    "usage": {"input_tokens": len(ids), "output_tokens": 0}}

    gen.register_generate_api_routers = register_generate_api_routers
    try:
        from vllm.entrypoints.openai import api_server as _api
        if getattr(_api, "register_generate_api_routers", None) is _исх:
            _api.register_generate_api_routers = register_generate_api_routers
    except Exception:  # noqa: BLE001
        pass


def _это_сервер():
    try:
        return b"vllm.entrypoints.openai.api_server" in open("/proc/self/cmdline", "rb").read()
    except Exception:  # noqa: BLE001
        return False


if ВКЛ:
    try:
        if _это_сервер():
            установить_сервер()
            _лог("рычаг включён: эндпоинт /v1/systemone")
        else:
            установить_воркер()
    except Exception:  # noqa: BLE001
        _лог("установка не удалась:\n" + traceback.format_exc())
