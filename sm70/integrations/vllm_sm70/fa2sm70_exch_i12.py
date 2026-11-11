# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ==================================================================================================
# ОБМЕН TP В 12 РАЗРЯДАХ. Размен «сжатие против узкого канала» -- единственный оставшийся рычаг.
# ==================================================================================================
#
# ЦЕПОЧКА ЗАМЕРОВ, КОТОРАЯ СЮДА ПРИВЕЛА (14.08.2026, всё на паре 0-1):
#   1. снятие фазы: обмен TP = 58 % префилла (97.8 -> 41.1 с);
#   2. барьер ДО свёртки: перекос рангов 2-3 %, ПЕРЕВОЗКА 97-98 % -- ждать некого, платятся байты;
#   3. кривая размера: t = 0.164 мс + 0.1770 мс/МБ -- постоянная часть 0.6 %, БАЙТЫ 99.4 %;
#   4. канал в обход NCCL: прямой P2P 6.56 ГБ/с, через хост 1.53 -- NCCL идёт прямым и берёт 89 %;
#   5. параллельные переезды: 8 потоков дают +3 %, не x8 -- упирается МОСТ, не поток и не канал;
#   6. ширина линка под нагрузкой: PCIe 3.0 **x8** при максимуме x16 (у трёх карт из четырёх).
# Вывод: канал насыщен (6.56 из 7.88 теоретических у x8) и шире не станет -- платформа. Значит
# единственный оставшийся ход -- ВЕЗТИ МЕНЬШЕ БАЙТ.
#
# ПОЧЕМУ 12 РАЗРЯДОВ, А НЕ 8. Квантуется только ЧУЖАЯ половина суммы (своя уже лежит точной).
# Замерено этим ядром: relL2 суммы 3.98e-04 против 2.25e-04 у нынешнего fp16-пути -- дрейф x1.77
# (закон 2^-b обещал x1.93). У int8 он был бы x31 -- цена, которой платить нельзя.
#
# ЧТО ЭТО ДАЁТ, ЧЕСТНО. Байт 160.6 -> 121.7 МБ (x1.320); упаковка и разбор 1.02 мс против 5.93 мс
# сэкономленной перевозки; ИТОГО обмен **x1.251**. Не x1.33: 6 % съедает сама упаковка, и это
# записано здесь, а не спрятано.
#
# ГДЕ НЕ ПРИМЕНЯТЬ И ПОЧЕМУ. На ДЕКОДЕ свёртки мелкие ([1..8, 5120] = 10-80 КБ), там цена -- БАРЬЕР,
# а не байты ([[fa2-sm70-barier-a-ne-baity]]), и упаковка добавила бы задержку в фазу, которая от
# байт не зависит. Поэтому порог по размеру, а не «включить везде».
#
# ВКЛЮЧЕНИЕ: FA2SM70_GATES='FA2SM70_ОБМЕН_I12=1'  (bash не умеет кириллические имена -- см. ворота
# в sitecustomize.py). Подмена -- в КОММУНИКАТОРЕ, ниже регистрации операции: подмена выше рвёт
# граф torch.compile (проверено, движок падает в profile_run).
import os
import sys

# ИМЕНА ASCII -- ОСНОВНЫЕ. Кириллические оставлены совместимостью, но включать рычаг из оболочки
# ими НЕЛЬЗЯ: bash отвергает такие идентификаторы И В ПРИСВОЕНИИ, и в export, причём с кодом 0 --
# рычаг молча не включается. Поэтому боевой путь -- FA2SM70_EXCH12, а ворота FA2SM70_GATES остаются
# только для тех рычагов, у которых ASCII-имени нет.
def _рычаг(ascii_имя, кир_имя, умолч="0"):
    v = os.environ.get(ascii_имя)
    return v if v is not None else os.environ.get(кир_имя, умолч)


_ПОРОГ = int(_рычаг("FA2SM70_EXCH12_MIN", "FA2SM70_ОБМЕН_ПОРОГ", str(1 << 20)))
# ЧИСЛО ЗНАЧАЩИХ РАЗРЯДОВ. 12 -- боевое. Меньшие значения НЕ УСКОРЯЮТ (байты те же), они нужны
# ТОЛЬКО как положительный контроль приёмки: путь, который ОБЯЗАН разойтись с эталоном.
_БИТ = int(os.environ.get("FA2SM70_EXCH12_BITS", "12"))
# [A100 24.10] ПЕРЕВОЗКА КОПИРОВЩИКОМ (FA2SM80_EXCH_CE=1, пара карт с рабочим P2P): упакованная половина идёт
# cudaMemcpyAsync прямо в приёмник соседа (IPC), затем флаг у соседа (ar2_signal, __threadfence_system), ожидание
# своего флага (ar2_wait, с потолком) и распаковка. Замер 24.10 на паре 0-1 (iommu=pt): NCCL all_gather по P2P
# берёт 8.1-8.2 ГБ/с на направление при встречной полосе копировщиков 10.5 -- то есть ~78 % канала, и его ядра
# занимают SM, а копировщик -- нет (важно для перекрытия с умножением, fa2sm80_rowpipe). Двойной буфер по
# чётности номера обмена: ранг дойдёт до обмена e+2, только увидев флаг соседа за e+1, а тот ставит его после
# копии e+1, то есть после распаковки e (порядок потока) -- приёмник чётности e к этому времени дочитан.
_CE = os.environ.get("FA2SM80_EXCH_CE", "0") == "1"
# [ПАМЯТЬ 29.10] FA2SM80_EXCH_OUT_MAIN=1: выход сведения, запущенного с ПОБОЧНОГО потока blockpipe, берётся из пула
# ОСНОВНОГО потока (record_stream на побочный) и пишется готовым буфером (unpack_add_i12_into). Иначе он кэшировался
# побочным потоком: снимок аллокатора 29.10 -- сегмент 220 МиБ без живых блоков, недоступный основному; отсюда
# повторы выделения (сброс кэша с синхронизацией устройства) и перекартирование страниц на каждом префилле.
# Ядро и арифметика те же -- результат побитно тот же.
_ВЫХОД_ОСН = os.environ.get("FA2SM80_EXCH_OUT_MAIN", "0") == "1"
import threading as _threading
_ПОТОК_ВЫХОДА = _threading.local()


class выход_на:
    """Контекст: сведения внутри берут выход из пула потока `осн` (blockpipe вокруг вызова на побочном потоке)."""

    def __init__(self, осн):
        self.осн = осн

    def __enter__(self):
        self.было = getattr(_ПОТОК_ВЫХОДА, "s", None)
        _ПОТОК_ВЫХОДА.s = self.осн
        return self

    def __exit__(self, *a):
        _ПОТОК_ВЫХОДА.s = self.было
        return False


class в_выход:
    """[ФРИЗЫ 30.10] Контекст: сведение внутри пишет результат В ГОТОВЫЙ тензор `out` (тот же numel, подряд) --
    в т. ч. на месте входа (out = вход: ядро читает элемент до записи того же элемента). Конвейер блока так
    сводит куски в свои постоянные буферы и прямо в срез выхода вызова: без выделения под каждый кусок (аудит окна 23:
    распаковка сведения -- 344 выделения и 3.8 ГБ на смешанный шаг) и без копии `out[r0:r1].copy_(dr)`."""

    def __init__(self, out):
        self.out = out

    def __enter__(self):
        self.было = getattr(_ПОТОК_ВЫХОДА, "out", None)
        _ПОТОК_ВЫХОДА.out = self.out
        return self

    def __exit__(self, *a):
        _ПОТОК_ВЫХОДА.out = self.было
        return False
_CE_МБ = int(os.environ.get("FA2SM80_EXCH_CE_MB", "72"))      # ёмкость приёмника на чётность
# [ОДИН ГРАФ 30.10] FA2SM70_EXCH_CE_GRAPH=1: в ЗАХВАТЕ CUDA-графа обмен идёт тем же копировщиком (ce_graph_*: номер на
# устройстве, один приёмник и встречный флаг «дочитано»), а не NCCL-сбором. Свой приёмник FA2SM70_EXCH_CE_GRAPH_MB
# (умолчание 24: кусок шага 2112 x 5120 в 12 разрядах -- 16.4 МБ) и свои флаги; поднимается вместе с eager-путём
# (первый большой обмен -- проход профилирования, до захвата). Без флага -- прежний путь.
_CE_ГРАФ = os.environ.get("FA2SM70_EXCH_CE_GRAPH", "0") == "1"
_CE_ГРАФ_МБ = int(os.environ.get("FA2SM70_EXCH_CE_GRAPH_MB", "24"))
# [БЕЗ ХОСТА 30.10] Проверка срыва ожидания флага соседа (раз в 256 обменов) была .item() -- полная синхронизация:
# профиль окна 12 -- 18-81 мс главного потока, CPU не готовит следующий шаг, пока GPU не опустеет. При уровне >= 1
# в FA2SM70_NOSYNC_FILE (или FA2SM70_NOSYNC_ACC2=1) -- неблокирующий снимок в закреплённую память и чтение на
# СЛЕДУЮЩЕЙ проверке, если событие снимка уже прошло; [СРЫВ 04.11] проверка -- на каждом обмене, отказ тот же.
# Потолок ожидания в ядрах -- FA2SM80_EXCH_WAIT_MS по стенным часам (умолчание 300000 -- последняя страховка, больше
# дедлайна RPC шага; протокол -- в exch_i12.cu у ar2_wait_kernel); здесь --
# только для сообщений, читает ту же переменную.
_ПОТОЛОК_МС = int(os.environ.get("FA2SM80_EXCH_WAIT_MS", "300000"))
_БХ_ФАЙЛ = os.environ.get("FA2SM70_NOSYNC_FILE", "")
_БХ = {"t": 0.0, "ур": 1 if os.environ.get("FA2SM70_NOSYNC_ACC2", "0") == "1" else 0}


def _без_хоста_уровень():
    if _БХ_ФАЙЛ:
        import time as _t
        сейчас = _t.monotonic()
        if сейчас - _БХ["t"] >= 1.0:
            _БХ["t"] = сейчас
            try:
                with open(_БХ_ФАЙЛ) as f:
                    v = f.read(1)
                _БХ["ур"] = int(v) if v.isdigit() else 0
            except OSError:
                pass
    return _БХ["ур"]

if _рычаг("FA2SM70_EXCH12", "FA2SM70_ОБМЕН_I12") == "1":
    import torch
    import torch.distributed as dist
    from fa2sm70_workspace import CapturedWorkspace

    С = {"n": 0, "ушло": 0, "было_бы": 0}
    # [ЗДЕСЬ БЫЛА УТЕЧКА, УРОНИВШАЯ БОЕВОЙ 17.08 ЧЕРЕЗ 4 ЧАСА РАБОТЫ]
    # Первая редакция держала буферы В СЛОВАРЕ ПО ТОЧНОЙ ФОРМЕ (numel, карта) без чистки, с
    # посылкой "форм за проход считанные единицы". На БЕНЧЕ это верно: длины фиксированы, форм
    # горстка. В БОЮ ложно: форма = остаток последнего куска префилла (длина mod 4096) плюс
    # визуальные токены картинок -- почти каждый запрос давал НОВУЮ форму и +15..60 МБ навсегда.
    # Замерено молотком: текст (4 длины) -- полка; картинки -- +63 МиБ/мин БЕЗ насыщения; чистый
    # форк -- ноль. За 4 часа набежало +5.5 ГиБ, и лёгкий запрос добил сервер (86 МиБ не нашлось).
    # Теперь -- ОДИН РАСТУЩИЙ буфер на карту: ёмкость сходится к самому крупному куску (~96 МБ на
    # карту с соседями) и дальше не растёт; меньшие формы берут ПРЕФИКСНЫЕ виды того же куска.
    # Раздельные ёмкости: CE нужен только send, NCCL дополнительно нужен gather.
    # Старые адреса, использованные при capture, живут до завершения процесса.
    # Eager-only поколения освобождаются при росте; это не кеш по точной форме.
    _КЭШ = {}    # карта -> {роль: CapturedWorkspace}
    _ext = None
    _CE_СОСТ = {}   # карта -> состояние перевозки копировщиком (или None, если поднять не удалось)

    def _буфер(ключ, роль, nb, device, capturing):
        буферы = _КЭШ.setdefault(ключ, {})
        б = буферы.get(роль)
        if б is None:
            б = буферы[роль] = CapturedWorkspace()
        return б.get(nb, lambda size: torch.empty(size, dtype=torch.uint8, device=device),
                     capturing=capturing)

    def _ce(сам):
        """Один раз на карту: приёмник 2 x ёмкость и 2 флага вне графового пула, обмен ручками через gloo."""
        ключ = сам.device.index if hasattr(сам, "device") else torch.cuda.current_device()
        if ключ in _CE_СОСТ:
            return _CE_СОСТ[ключ]
        _CE_СОСТ[ключ] = None
        try:
            e = _расш()
            if hasattr(e, "ce_sync_wait_ceiling"):          # [СРЫВ 04.11] тот же потолок и для ядер AR1 этой карты
                e.ce_sync_wait_ceiling()
            cap = _CE_МБ << 20
            recv = e.ipc_alloc(2 * cap, torch.uint8)
            flag = e.ipc_alloc(2, torch.int32)
            мои = {"r": (bytes(e.ipc_handle(recv)), e.ipc_offset(recv)),
                   "f": (bytes(e.ipc_handle(flag)), e.ipc_offset(flag))}
            гр = _CE_ГРАФ and hasattr(e, "ce_graph_send")
            if гр:
                recv_g = e.ipc_alloc(_CE_ГРАФ_МБ << 20, torch.uint8)
                fl_g = e.ipc_alloc(2, torch.int32)                 # [0] -- флаг «данные у тебя», [1] -- «дочитано»
                мои["rg"] = (bytes(e.ipc_handle(recv_g)), e.ipc_offset(recv_g))
                мои["fg"] = (bytes(e.ipc_handle(fl_g)), e.ipc_offset(fl_g))
            мир, ранг = сам.world_size, сам.rank
            собрано = [None] * мир
            собрано[ранг] = мои
            for i in range(мир):
                буф = собрано[i:i + 1]
                dist.broadcast_object_list(буф, src=dist.get_process_group_ranks(сам.cpu_group)[i],
                                           group=сам.cpu_group)
                собрано[i] = буф[0]
            чуж = собрано[1 - ранг]
            st = {"e": e, "cap": cap, "recv": recv, "flag": flag, "seq": 0, "n": 0,
                  # [0] срыв, [1] не занят, [2] наибольшее ожидание (мс), [3] ожиданий дольше 100 мс
                  "срыв": torch.zeros(4, dtype=torch.int32, device=recv.device),
                  "p_recv": e.ipc_open(чуж["r"][0], чуж["r"][1]),
                  "p_flag": e.ipc_open(чуж["f"][0], чуж["f"][1])}
            if гр and "rg" in чуж:
                p_fg = e.ipc_open(чуж["fg"][0], чуж["fg"][1])
                st.update({"cap_g": _CE_ГРАФ_МБ << 20, "recv_g": recv_g, "fl_g": fl_g,
                           "seq_g": torch.zeros(1, dtype=torch.int32, device=recv.device),
                           "p_recv_g": e.ipc_open(чуж["rg"][0], чуж["rg"][1]),
                           "p_flag_g": p_fg, "p_ack_g": p_fg + 4})
            dist.barrier(group=сам.cpu_group)
            _CE_СОСТ[ключ] = st
            print(f"[fa2_sm70 ОБМЕН-12бит] перевозка КОПИРОВЩИКОМ по P2P: приёмник 2 x {_CE_МБ} МиБ"
                  + (f"; в графе -- копировщиком, приёмник {_CE_ГРАФ_МБ} МиБ" if "cap_g" in st else ""),
                  file=sys.stderr, flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[fa2_sm70 ОБМЕН-12бит] копировщик не поднят ({ex}) -- NCCL", file=sys.stderr, flush=True)
        return _CE_СОСТ[ключ]

    def _расш():
        global _ext
        if _ext is None:
            sys.path.insert(0, os.environ.get("FA2SM70_PATH", "."))
            from fa2_sm70._ext import exch_ext
            _ext = exch_ext()
        return _ext

    def _распаковать(e, база, pk, sc, capturing):
        """unpack_add_i12; с побочного потока blockpipe -- в выход из пула основного (см. _ВЫХОД_ОСН)."""
        цель = getattr(_ПОТОК_ВЫХОДА, "out", None)
        if (цель is not None and not capturing and цель.numel() == база.numel() and цель.is_contiguous()
                and цель.dtype == база.dtype and hasattr(e, "unpack_add_i12_into")):
            e.unpack_add_i12_into(база, pk, sc, цель)
            С["в_выход"] = С.get("в_выход", 0) + 1
            return цель
        осн = getattr(_ПОТОК_ВЫХОДА, "s", None) if _ВЫХОД_ОСН and not capturing else None
        if осн is not None and hasattr(e, "unpack_add_i12_into"):
            тек = torch.cuda.current_stream()
            if тек != осн:
                with torch.cuda.stream(осн):
                    out = torch.empty_like(база)
                out.record_stream(тек)
                e.unpack_add_i12_into(база, pk, sc, out)
                return out
        return e.unpack_add_i12(база, pk, sc)

    def _срыв_текст(h):
        return (f"обмен копировщиком: срыв ожидания флага соседа (код {int(h[0])}; потолок {_ПОТОЛОК_МС} мс; "
                f"наибольшее ожидание {int(h[2])} мс, дольше 100 мс -- {int(h[3])} раз)")

    def _срыв_асинх(st):
        """Прошлый снимок флага срыва -- если его копия уже прошла (event.query без ожидания); затем новый снимок.
        [СРЫВ 04.11] Зовётся на КАЖДОМ обмене (было -- раз в 256: срыв виден с опозданием до 256 обменов, а всё это
        время ранг складывал устаревшие байты соседа). Новый снимок ставится, только когда прошлый доехал, -- в полёте
        не больше одной копии 16 байт. Снимок несёт и статистику ожиданий (срыв[2..3], пишут ядра ожидания)."""
        ev = st.get("срыв_ev")
        if ev is not None:
            if not ev.query():
                return                                  # прошлый снимок ещё едет -- проверим на следующем обмене
            h = st["срыв_h"]
            if int(h[0]):
                raise RuntimeError(_срыв_текст(h))
            мс = int(h[2])
            if мс >= 200 and мс > st.get("ожид_лог", 0) * 1.25:
                st["ожид_лог"] = мс
                print(f"[fa2_sm70 ОБМЕН-12бит] долгое ожидание соседа: наибольшее {мс} мс, дольше 100 мс -- "
                      f"{int(h[3])} раз (потолок {_ПОТОЛОК_МС} мс)", file=sys.stderr, flush=True)
        if "срыв_h" not in st:
            st["срыв_h"] = torch.zeros(st["срыв"].numel(), dtype=torch.int32, pin_memory=True)
        st["срыв_h"].copy_(st["срыв"], non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        st["срыв_ev"] = ev

    def _обёртка(старый):
        def _f(self, input_, *a, **k):
            n = input_.numel()
            if (n < _ПОРОГ or input_.dtype != torch.float16 or n % 128 != 0):
                return старый(self, input_, *a, **k)
            мир = dist.get_world_size()
            ранг = dist.get_rank()
            if мир < 2:
                return input_
            e = _расш()
            x = input_.contiguous()
            # ОДИН КУСОК ПАМЯТИ НА УПАКОВАННОЕ И МАСШТАБЫ -> ОДНА коллективная операция вместо двух.
            # NCCL не везёт int16 ("data type is not supported: Short"), а байты везёт -- поэтому
            # общий буфер байтовый, а виды на него ставятся без единой копии.
            ngrp = n // 128
            nb_pk, nb_sc = ngrp * 96 * 2, ngrp * 2
            nb = nb_pk + nb_sc
            ключ = x.device.index
            capturing = torch.cuda.is_current_stream_capturing()
            свой = _буфер(ключ, "send", nb, x.device, capturing)
            # ПРЕФИКСНЫЕ виды непрерывны, поэтому годятся и упаковщику, и NCCL без единой копии.
            pk = свой[:nb_pk].view(torch.int16)
            sc = свой[nb_pk:nb].view(torch.float16)
            e.pack_i12_into(x, pk, sc, _БИТ)
            if _CE and мир == 2 and capturing and _CE_ГРАФ:
                # [ОДИН ГРАФ 30.10] захват: копировщик с номером на устройстве; состояние поднято ДО захвата
                st = _CE_СОСТ.get(ключ)
                if st is not None and "cap_g" in st and nb <= st["cap_g"]:
                    e.ce_graph_send(свой[:nb], st["p_recv_g"], st["fl_g"][1:2], st["p_flag_g"], st["seq_g"], st["срыв"])
                    e.ce_graph_wait(st["fl_g"][0:1], st["seq_g"], st["срыв"])
                    ч = st["recv_g"][:nb]
                    out = _распаковать(e, x, ч[:nb_pk].view(torch.int16), ч[nb_pk:nb].view(torch.float16), capturing)
                    e.ce_graph_ack(st["p_ack_g"], st["seq_g"])
                    С["граф"] = С.get("граф", 0) + 1
                    return out.view_as(input_)
            if _CE and мир == 2 and not capturing:
                st = _ce(self)
                if st is not None and nb <= st["cap"]:
                    par = st["seq"] & 1
                    st["seq"] += 1
                    val = st["seq"]
                    e.ar2_copy(свой[:nb], st["p_recv"] + par * st["cap"])     # моя половина -> приёмник соседа
                    e.ar2_signal(st["p_flag"], par, val)                       # флаг у соседа после копии
                    e.ar2_wait(st["flag"], par, val, st["срыв"])               # жду половину соседа
                    ч = st["recv"][par * st["cap"]:par * st["cap"] + nb]
                    out = _распаковать(e, x, ч[:nb_pk].view(torch.int16), ч[nb_pk:nb].view(torch.float16), capturing)
                    st["n"] += 1
                    if _без_хоста_уровень() >= 1:
                        _срыв_асинх(st)
                    elif st["n"] & 255 == 0 and int(st["срыв"][0].item()):
                        raise RuntimeError(_срыв_текст(st["срыв"].cpu()))
                    С["n"] += 1
                    С["ушло"] += nb
                    С["было_бы"] += n * 2
                    return out.view_as(input_)
            # До этой точки успешный CE не требует NCCL storage. Не резервировать
            # 2*nb заранее: при M=8192,H=5120 это лишние 121,25 МиБ на ранг.
            все = _буфер(ключ, "gather", nb * мир, x.device, capturing)
            # НАТИВНЫЙ СБОР: прямой ncclAllGather на ТЕКУЩЕМ потоке, без питоновской обвязки
            # torch.distributed. Он же ЗАХВАТЫВАЕТСЯ в CUDA-граф -- поэтому отката по
            # is_current_stream_capturing больше нет (он стоял без объяснения и запирал нас
            # в кусочном графе).
            _pn = getattr(self, "pynccl_comm", None)
            if _pn is not None and not _pn.disabled:
                _pn.all_gather(все[:nb * мир], свой[:nb])
            else:
                dist.all_gather([все[r * nb:(r + 1) * nb] for r in range(мир)], свой[:nb])
            out = x
            for r in range(мир):
                if r == ранг:
                    continue                       # СВОЯ часть -- точная, через провод не ехала
                ч = все[r * nb:(r + 1) * nb]
                out = _распаковать(e, out, ч[:nb_pk].view(torch.int16),
                                   ч[nb_pk:nb].view(torch.float16), capturing)
            С["n"] += 1
            С["ушло"] += nb_pk + nb_sc
            С["было_бы"] += n * 2
            if С["n"] & (С["n"] - 1) == 0:
                print(f"[fa2_sm70 ОБМЕН-12бит] свёрток {С['n']}: по проводу "
                      f"{С['ушло']/1e9:.2f} ГБ вместо {С['было_бы']/1e9:.2f} "
                      f"(x{С['было_бы']/max(1,С['ушло']):.3f} меньше)", file=sys.stderr, flush=True)
            return out.view_as(input_)
        return _f

    сколько = 0
    for путь, имя in (
        ("vllm.distributed.device_communicators.base_device_communicator", "DeviceCommunicatorBase"),
        ("vllm.distributed.device_communicators.cuda_communicator", "CudaCommunicator"),
    ):
        try:
            мод = __import__(путь, fromlist=[имя])
            кл = getattr(мод, имя)
            if "all_reduce" in кл.__dict__:
                кл.all_reduce = _обёртка(кл.__dict__["all_reduce"])
                сколько += 1
        except Exception:
            pass
    print(f"[fa2_sm70] ОБМЕН TP В 12 РАЗРЯДАХ включён (классов {сколько}, порог {_ПОРОГ} элементов). "
          f"разрядов {_БИТ}. Ответы ВЕРНЫ при 12; меньше -- ТОЛЬКО контроль приёмки.", file=sys.stderr, flush=True)
    if сколько == 0:
        print("[fa2_sm70] ОБМЕН-12бит: НИ ОДИН класс не подменён -- ПРИБОР СЛЕП",
              file=sys.stderr, flush=True)
