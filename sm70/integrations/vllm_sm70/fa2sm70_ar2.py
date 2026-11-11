# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ==================================================================================================
# СВОЁ СВЕДЕНИЕ TP НА ДВЕ КАРТЫ ВМЕСТО NCCL (задача 201, шаг 3 -- обвязка)
# ==================================================================================================
#
# ЗАЧЕМ. Замер 27.08: сведение мелкого тензора стоит 56.7-91.7 мкс у NCCL и 26.7 мкс у нашего
# тела -- при том что ПРЯМАЯ КОПИЯ между картами стоит 24.0 мкс. То есть NCCL кладёт поверх
# железного пола 30-65 мкс, и это НЕ лечится его ручками (перебраны NCCL_PROTO LL/LL128/Simple,
# NCCL_ALGO=Ring, NCCL_MAX_NCHANNELS=1 -- пол не двигается). На шаг декода приходится около
# 128 сведений, значит приз 3.8-8.3 мс на шаг.
#
# ПОЧЕМУ НЕ ШТАТНЫЙ custom_all_reduce vLLM. Он требует IPC-ручку на буфере CUDA-ГРАФА, а V100
# её не даёт: `custom_all_reduce.cuh:455 'invalid argument'`. Заглушить регистрацию нельзя --
# у C++-стороны копятся незарегистрированные буферы и первый же шаг встаёт (проверено шестью
# пусками: кольцо из 170, 4 и 1 участка, с нашим графом и без него -- виснет всегда).
# Наше тело берёт буферы через `ipc_alloc` -- ВНЕ графового пула, и потому законно.
#
# ЧЕМ ГАРАНТИРОВАНА ПРАВИЛЬНОСТЬ В ГРАФЕ. Номер последовательности при захвате ЗАПЕКАЕТСЯ, и
# четыре примитива из питона тут не годятся: на воспроизведении ждущий сверял бы флаг со СТАРЫМ
# числом, тот уже выставлен с прошлого прохода, ожидание прошло бы мимо и сосед прочитал бы НЕ
# ТЕ байты (порча ТИХАЯ -- ответ остаётся правдоподобным). Поэтому счётчик живёт В ПАМЯТИ
# УСТРОЙСТВА и растёт на каждом воспроизведении: `ar2_all_reduce` делает это сам.
# Проверено: 20 воспроизведений графа подряд -- расхождений НОЛЬ.
#
# МЕСТО ВЫЗОВА = СВОЙ ПРИЁМНИК, СВОЙ ФЛАГ, СВОЙ СЧЁТЧИК. Счётчик мест обнуляется на каждом
# проходе (`set_forward_context` зовётся ровно раз на проход), поэтому k-е сведение прохода на
# обоих рангах берёт k-й комплект ПО ПОСТРОЕНИЮ -- даже если до прохода ранги успели сделать
# разное число служебных сведений.
#
# ГРАНИЦА ПРИМЕНИМОСТИ -- ТОЛЬКО МЕЛКИЕ ТЕНЗОРЫ. Приз лежит в латентности, а не в полосе: на
# 40 МиБ NVLink даёт x6.3 и NCCL там уместен. Крупные сведения (префилл) уходят прежним путём.
# Решение принимается ТОЛЬКО по форме и типу тензора -- они одинаковы у обоих рангов, значит
# ранги не могут разойтись по путям и заклиниться.
#
# ВЫКЛЮЧАТЕЛЬ: FA2SM70_AR2=1. Размер участка: FA2SM70_AR2_SLOT_KB (умолчание 256),
# число мест вызова: FA2SM70_AR2_SLOTS (умолчание 160).
import os
import sys

import torch


_СЧЁТ_ВЫЗОВОВ = [0, 0] if os.environ.get("FA2SM70_AR2_COUNT", "0") == "1" else None


def установить() -> None:
    if os.environ.get("FA2SM70_AR2", "0") != "1":
        return
    try:
        import torch.distributed as dist
    except Exception as ex:  # noqa: BLE001
        print(f"[fa2_sm70 AR2] нет torch.distributed: {ex}", file=sys.stderr, flush=True)
        return

    _УЧАСТОК = int(os.environ.get("FA2SM70_AR2_SLOT_KB", "256")) * 1024
    _МЕСТ = int(os.environ.get("FA2SM70_AR2_SLOTS", "160"))
    # AR1 (путь A100): тот же протокол ОДНИМ ядром вместо шести запусков; флаги и счётчики -- у
    # каждого куска 4 КиБ отдельно, поэтому на место их _ШИР штук.
    #   FA2SM80_AR1=1 (host) -- через общий закреплённый сегмент ХОСТА (ar1h). На паре GPU0-1 этого
    #     стенда P2P-запись в IPC-память соседа молча не доходит (проба 23.10 читает нули), NCCL ходит
    #     через SHM; замер в графе: M=4 19.4 мкс против 28.3 у NCCL, M=16 43.0 против 67.1, точно.
    #   FA2SM80_AR1=p2p -- окна флагов в IPC-памяти соседа (ar1), для машин с рабочим P2P/NVLink.
    _AR1 = os.environ.get("FA2SM80_AR1", "0")
    _AR1H = _AR1 in ("1", "host")
    _AR1P = _AR1 == "p2p"
    _ШИР = max(1, _УЧАСТОК // 4096) if (_AR1H or _AR1P) else 1
    # AR1LL (25.10, ar1ll.cuh): данные с номером в одной 16 Б записи, без fence и отдельного флага -- стенд M=5: ~13
    # против ~22 мкс у AR1H, 3М сведений без единой ошибки. Отдельный сегмент хоста (ячейки LL никогда не видят чужих
    # данных), свои счётчики блоков. Крупнее участка ячеек -- AR1H.
    _LL = _AR1H and os.environ.get("FA2SM80_AR_LL", "0") == "1"
    _LLB = int(os.environ.get("FA2SM80_AR_LL_KB", "128")) * 1024
    # СЧЁТЧИК МЕСТ -- СВОЙ НА КОММУНИКАТОР. Общий на всех давал порчу: в проходе работает не
    # только группа TP (черновик и иные двухранговые группы зовут тот же метод), их вызовы
    # делили один счётчик, и комплекты «приёмник+флаг+счётчик» у рангов разъезжались.
    # Поймано гейтом 17*23: сервер поднимался, сведение шло, ответ был СЛОВЕСНОЙ КАШЕЙ.
    _счёт = {}          # id(коммуникатора) -> [индекс места в текущем проходе]
    _сост = {}          # id(коммуникатора) -> приёмники, флаги, счётчики, адреса соседа
    _ГЕЙТ = int(os.environ.get("FA2SM70_AR2_GATE", "0"))   # сверить N первых сведений с NCCL
    _сверено = [0]
    # ПОЛОЖИТЕЛЬНЫЙ КОНТРОЛЬ ГЕЙТА (FA2SM70_AR2_PORCHA=1): намеренно портим результат ПОСЛЕ
    # своего сведения. Гейт ОБЯЗАН это поймать. Пока он не ловит заведомую порчу, его
    # «совпали» не доказывает ничего -- это наш же закон о положительном контроле эталона.
    _ПОРЧА = os.environ.get("FA2SM70_AR2_PORCHA", "0") == "1"
    # ОКНО МЕСТ: своё сведение берут только места [ОТ, ДО). Нужно для бисекции по ДИАПАЗОНУ,
    # а не по числу: она отделяет «портит конкретный слой» от «портит множественность».
    _НА_МЕСТЕ = os.environ.get("FA2SM70_AR2_INPLACE", "0") == "1"
    _ОТ = int(os.environ.get("FA2SM70_AR2_FROM", "0"))
    _ДО = int(os.environ.get("FA2SM70_AR2_TO", "1000000"))
    # ЗАЩИТА ОТ ПОВТОРНОГО ВХОДА. Подменены ДВА класса, а верхний (`CudaCommunicator`) зовёт
    # нижний (`DeviceCommunicatorBase`) -- без этой защиты один и тот же тензор сводится
    # ДВАЖДЫ: y = 2*(y_A + y_B). Порча тихая И НЕВИДИМАЯ ГЕЙТУ: эталон считается тем же
    # удвоенным путём, поэтому сверка показывает совпадение, а ответ -- словесная каша.
    # Поймано так: 100000 сведений сверены с NCCL «совпали», гейт 17*23 при этом ПРОВАЛЕН.
    _внутри = [False]

    def _расширение():
        if "ext" in _сост:
            return _сост["ext"]
        # Берём СОБРАННОЕ тело тем же путём, что и обёртка обмена, а не одноимённый
        # питоновский модуль: `fa2sm70_exch_i12` -- это ОБЁРТКА, у неё нет наших тел, и
        # попытка взять её приводила к «module has no attribute ar2_all_reduce» и тихому
        # откату на прежний путь (поймано подъёмом 27.08).
        from fa2_sm70._ext import exch_ext
        е = exch_ext()
        if not hasattr(е, "ar1h_all_reduce_out" if _AR1H else "ar1_all_reduce_out" if _AR1P
                       else "ar2_all_reduce_out"):
            raise RuntimeError("в собранном расширении нет ar2_all_reduce -- нужна пересборка")
        _сост["ext"] = е
        return е

    def _подготовить_хост(сам):
        """AR1H: общий сегмент shm [ранг][место][2 чётности][участок] + [ранг][место][_ШИР флагов],
        закреплённый за CUDA в обоих процессах. Создаёт ранг 0, имя уходит соседу через gloo,
        после подключения соседа имя удаляется (сегмент живёт, пока отображён)."""
        import ctypes
        from multiprocessing import shared_memory
        e = _расширение()
        dev = сам.device
        N = _УЧАСТОК // 2
        на_ранг = _МЕСТ * (2 * _УЧАСТОК + 4 * _ШИР)
        гр = dist.get_process_group_ranks(сам.cpu_group)
        имя = [f"fa2ar1h_{os.getuid()}_{os.getpid()}_{id(сам) & 0xFFFFFF:x}" if сам.rank == 0 else None]
        if сам.rank == 0:
            shm = shared_memory.SharedMemory(name=имя[0], create=True, size=2 * на_ранг)
            shm.buf[:] = bytes(2 * на_ранг)
        dist.broadcast_object_list(имя, src=гр[0], group=сам.cpu_group)
        if сам.rank != 0:
            shm = shared_memory.SharedMemory(name=имя[0], create=False)
        dist.barrier(group=сам.cpu_group)
        if сам.rank == 0:
            shm.unlink()
        база = e.host_register(ctypes.addressof(ctypes.c_char.from_buffer(shm.buf)), 2 * на_ранг)
        мой, чуж = база + сам.rank * на_ранг, база + (1 - сам.rank) * на_ранг
        флаги = 2 * _УЧАСТОК * _МЕСТ
        seq_all = torch.zeros(_МЕСТ * _ШИР, dtype=torch.int32, device=dev)
        ll = {}
        if _LL:
            на_ранг_ll = _МЕСТ * 2 * _LLB
            имя_ll = [имя[0] + "_ll" if сам.rank == 0 else None]
            if сам.rank == 0:
                shm_ll = shared_memory.SharedMemory(name=имя_ll[0], create=True, size=2 * на_ранг_ll)
                shm_ll.buf[:] = bytes(2 * на_ранг_ll)
            dist.broadcast_object_list(имя_ll, src=гр[0], group=сам.cpu_group)
            if сам.rank != 0:
                shm_ll = shared_memory.SharedMemory(name=имя_ll[0], create=False)
            dist.barrier(group=сам.cpu_group)
            if сам.rank == 0:
                shm_ll.unlink()
            база_ll = e.host_register(ctypes.addressof(ctypes.c_char.from_buffer(shm_ll.buf)), 2 * на_ранг_ll)
            мой_ll, чуж_ll = база_ll + сам.rank * на_ранг_ll, база_ll + (1 - сам.rank) * на_ранг_ll
            seq_ll = torch.zeros(_МЕСТ * _ШИР, dtype=torch.int32, device=dev)
            ll = {"shm_ll": shm_ll, "ll_cells": _LLB // 16,
                  "ll_my": [мой_ll + k * 2 * _LLB for k in range(_МЕСТ)],
                  "ll_peer": [чуж_ll + k * 2 * _LLB for k in range(_МЕСТ)],
                  "ll_seq": [seq_ll[k * _ШИР:(k + 1) * _ШИР] for k in range(_МЕСТ)]}
            print(f"[fa2_sm70 AR2] AR1LL: сегмент ячеек {2 * на_ранг_ll >> 20} МиБ, до {_LLB // 16 * 6} элементов",
                  file=sys.stderr, flush=True)
        print(f"[fa2_sm70 AR2] своё сведение готово: мест {_МЕСТ} по {_УЧАСТОК >> 10} КиБ "
              f"(AR1H: одно ядро через память хоста, сегмент {2 * на_ранг >> 20} МиБ)",
              file=sys.stderr, flush=True)
        return {"e": e, "h": True, "shm": shm, "N": N,
                "my_slot": [мой + k * 2 * _УЧАСТОК for k in range(_МЕСТ)],
                "peer_slot": [чуж + k * 2 * _УЧАСТОК for k in range(_МЕСТ)],
                "my_flag": [мой + флаги + 4 * _ШИР * k for k in range(_МЕСТ)],
                "peer_flag": [чуж + флаги + 4 * _ШИР * k for k in range(_МЕСТ)],
                "seq": [seq_all[k * _ШИР:(k + 1) * _ШИР] for k in range(_МЕСТ)],
                "срыв": torch.zeros(2, dtype=torch.int32, device=dev), **ll}

    def _подготовить(сам):
        """Один раз на группу: буферы вне графового пула + обмен ручками через gloo."""
        # [СРЫВ 04.11] потолок ожиданий соседа в ядрах AR1/AR1H/AR1LL <- FA2SM80_EXCH_WAIT_MS (подготовка идёт вне
        # захвата; у прежней сборки расширения функции нет -- там потолок прежний)
        _синх = getattr(_расширение(), "ce_sync_wait_ceiling", None)
        if _синх is not None:
            _синх()
        if _AR1H:
            return _подготовить_хост(сам)
        e = _расширение()
        dev = сам.device
        N = _УЧАСТОК // 2                       # элементов fp16 в участке
        # ОДНО ОТОБРАЖЕНИЕ НА ВСЕ МЕСТА. Прежде на каждое место шёл свой `ipc_alloc` и свой
        # `ipc_open` -- и за границей между 16 и 64 отображениями выход становился кашей
        # (бисекция 27.08: 1/4/16 мест -- верно, 64/128/160 -- каша). Числа при этом СОВПАДАЛИ
        # с NCCL на каждом сведении, то есть портились не значения, а чужая память.
        # Теперь буфер ОДИН, места -- смещения внутри него: отображение ровно одно на соседа.
        recv_all = e.ipc_alloc(N * _МЕСТ, torch.float16)
        recv = [recv_all[k * N:(k + 1) * N] for k in range(_МЕСТ)]
        flag = e.ipc_alloc(_МЕСТ * _ШИР, torch.int32); flag.zero_()
        ack = e.ipc_alloc(_МЕСТ * _ШИР, torch.int32); ack.zero_()   # «дочитано» -- встречный флаг
        seq_all = torch.zeros(_МЕСТ * _ШИР, dtype=torch.int32, device=dev)
        seq = [seq_all[k * _ШИР:(k + 1) * _ШИР] for k in range(_МЕСТ)]
        # [0] -- признак срыва ожидания, [1] -- счётчик доработавших блоков отправки
        срыв = torch.zeros(2, dtype=torch.int32, device=dev)
        мои = {"r": (bytes(e.ipc_handle(recv_all)), e.ipc_offset(recv_all)),
               "f": (bytes(e.ipc_handle(flag)), e.ipc_offset(flag)),
               "a": (bytes(e.ipc_handle(ack)), e.ipc_offset(ack))}
        мир = сам.world_size
        собрано = [None] * мир
        собрано[сам.rank] = мои
        for i in range(мир):
            буф = собрано[i:i + 1]
            dist.broadcast_object_list(буф, src=dist.get_process_group_ranks(сам.cpu_group)[i],
                                       group=сам.cpu_group)
            собрано[i] = буф[0]
        чуж = собрано[1 - сам.rank]
        сосед_база = e.ipc_open(чуж["r"][0], чуж["r"][1])
        сосед_recv = [сосед_база + k * N * 2 for k in range(_МЕСТ)]
        сосед_flag = e.ipc_open(чуж["f"][0], чуж["f"][1])
        сосед_ack = e.ipc_open(чуж["a"][0], чуж["a"][1])
        print(f"[fa2_sm70 AR2] своё сведение готово: мест {_МЕСТ} по {_УЧАСТОК >> 10} КиБ"
              f"{' (AR1: одно ядро)' if _AR1 else ''}",
              file=sys.stderr, flush=True)
        return {"e": e, "recv_all": recv_all, "recv": recv, "flag": flag, "ack": ack,
                "seq": seq, "срыв": срыв,
                "сосед_recv": сосед_recv, "сосед_flag": сосед_flag, "сосед_ack": сосед_ack}

    # [РЕГИСТРАЦИЯ ОПЕРАЦИИ -- ЛЕЧЕНИЕ ТИХОЙ ПОРЧИ]
    # Прямой вызов C++-тела из питона идёт МИМО схемы: компилятор не знает, что тело МЕНЯЕТ
    # вход на месте, и вправе переставить вызов, вынести его или взять устаревшее значение.
    # Признак ровно такой, какой и наблюдался: числа в каждом сведении СОВПАДАЮТ с NCCL
    # (гейт со сверкой это подтвердил, а его собственный положительный контроль -- поймал
    # намеренную порчу 1.5x), а выход сети -- словесная каша, побитово одинаковая.
    # Лечение -- объявить операцию со схемой и `mutates_args`, как сделано у остальных наших тел.
    _оп = None
    _оп_h = None
    _оп_ll = [None]
    try:
        from vllm.utils.torch_utils import direct_register_custom_op
    except Exception:  # noqa: BLE001
        try:
            from vllm.utils import direct_register_custom_op
        except Exception:  # noqa: BLE001
            direct_register_custom_op = None
    if direct_register_custom_op is not None:
        # ИМЕНА ПАРАМЕТРОВ -- ТОЛЬКО ЛАТИНИЦА: разборщик схемы torch кириллицу не принимает
        # («expected a valid token but found ' '»), и регистрация ТИХО не удавалась.
        def _тело(y: torch.Tensor, out: torch.Tensor, recv: torch.Tensor, peer_recv: int,
                  flag: torch.Tensor, peer_flag: int, seq: torch.Tensor,
                  fail: torch.Tensor, ack: torch.Tensor, peer_ack: int) -> None:
            # СЧЁТЧИК В САМОМ ТЕЛЕ ОПЕРАЦИИ: обёртка класса ловит не все вызовы --
            # vLLM зовёт зарегистрированную операцию напрямую, и счётчик в обёртке
            # молчал при живых ядрах в трассе. Считать надо в самой нижней точке.
            if _СЧЁТ_ВЫЗОВОВ is not None:
                _СЧЁТ_ВЫЗОВОВ[0] += 1
                _СЧЁТ_ВЫЗОВОВ[1] += y.numel()
                if _СЧЁТ_ВЫЗОВОВ[0] % 2000 == 0:
                    print(f"[AR2 СЧЁТ] вызовов {_СЧЁТ_ВЫЗОВОВ[0]}, "
                          f"средний размер {_СЧЁТ_ВЫЗОВОВ[1]/_СЧЁТ_ВЫЗОВОВ[0]:.0f}",
                          file=sys.stderr, flush=True)
            е = _расширение()
            (е.ar1_all_reduce_out if _AR1P else е.ar2_all_reduce_out)(
                y, out, recv, peer_recv, flag, peer_flag, seq, fail, ack, peer_ack)

        def _тело_пусто(y: torch.Tensor, out: torch.Tensor, recv: torch.Tensor, peer_recv: int,
                        flag: torch.Tensor, peer_flag: int, seq: torch.Tensor,
                        fail: torch.Tensor, ack: torch.Tensor, peer_ack: int) -> None:
            return

        def _тело_h(y: torch.Tensor, out: torch.Tensor, my_slot: int, peer_slot: int,
                    my_flag: int, peer_flag: int, seq: torch.Tensor, fail: torch.Tensor,
                    slot_elems: int) -> None:
            _расширение().ar1h_all_reduce_out(y, out, my_slot, peer_slot, my_flag, peer_flag,
                                              seq, fail, slot_elems)

        def _тело_h_пусто(y: torch.Tensor, out: torch.Tensor, my_slot: int, peer_slot: int,
                          my_flag: int, peer_flag: int, seq: torch.Tensor, fail: torch.Tensor,
                          slot_elems: int) -> None:
            return

        def _тело_ll(y: torch.Tensor, out: torch.Tensor, my_cells: int, peer_cells: int,
                     seq: torch.Tensor, fail: torch.Tensor, slot_cells: int) -> None:
            _расширение().ar1ll_all_reduce_out(y, out, my_cells, peer_cells, seq, fail, slot_cells)

        def _тело_ll_пусто(y: torch.Tensor, out: torch.Tensor, my_cells: int, peer_cells: int,
                           seq: torch.Tensor, fail: torch.Tensor, slot_cells: int) -> None:
            return

        if _LL:
            try:
                direct_register_custom_op("fa2sm80_ar1ll_all_reduce", _тело_ll,
                                          mutates_args=["out", "seq", "fail"], fake_impl=_тело_ll_пусто)
                _оп_ll[0] = torch.ops.vllm.fa2sm80_ar1ll_all_reduce
            except Exception as ex:  # noqa: BLE001
                print(f"[fa2_sm70 AR2] объявить AR1LL не вышло: {ex}", file=sys.stderr, flush=True)
        try:
            direct_register_custom_op("fa2sm80_ar1h_all_reduce", _тело_h,
                                      mutates_args=["out", "seq", "fail"], fake_impl=_тело_h_пусто)
            _оп_h = torch.ops.vllm.fa2sm80_ar1h_all_reduce
        except Exception as ex:  # noqa: BLE001
            _оп_h = None
            print(f"[fa2_sm70 AR2] объявить AR1H не вышло: {ex}", file=sys.stderr, flush=True)
        try:
            direct_register_custom_op("fa2sm70_ar2_all_reduce", _тело,
                                      mutates_args=["out", "recv", "flag", "seq", "fail", "ack"],
                                      fake_impl=_тело_пусто)
            _оп = torch.ops.vllm.fa2sm70_ar2_all_reduce
            print("[fa2_sm70 AR2] операция объявлена со схемой (mutates_args)",
                  file=sys.stderr, flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[fa2_sm70 AR2] объявить операцию не вышло: {ex}", file=sys.stderr, flush=True)

    _ФАЙЛ = os.environ.get("FA2SM70_AR2_FILE", "")
    _реж = [1, 0]   # [включено, счётчик опросов]
    _СРЫВ_ФАТАЛЕН = os.environ.get("FA2SM70_AR_STALL_FATAL", "1") == "1"

    def _обёртка(прежний):
        # [СЧЁТЧИК ВЫЗОВОВ, 07.09] Трасса показала, что при спекуляции ядер обмена
        # на шаг в 2.3 раза больше. Логических обменов по слоям столько же, значит
        # надо считать ВЫЗОВЫ, а не выводить их из трассы делением на tau.
        def _f(сам, input_):
            if _СЧЁТ_ВЫЗОВОВ is not None:
                _СЧЁТ_ВЫЗОВОВ[0] += 1
                _СЧЁТ_ВЫЗОВОВ[1] += input_.numel()
                if _СЧЁТ_ВЫЗОВОВ[0] % 2000 == 0:
                    print(f"[AR2 СЧЁТ] вызовов {_СЧЁТ_ВЫЗОВОВ[0]}, "
                          f"элементов {_СЧЁТ_ВЫЗОВОВ[1]/1e6:.1f}M, "
                          f"средний размер {_СЧЁТ_ВЫЗОВОВ[1]/_СЧЁТ_ВЫЗОВОВ[0]:.0f}",
                          file=sys.stderr, flush=True)
            if _ФАЙЛ:
                # [A/B ВНУТРИ ОДНОГО ПРОЦЕССА] Сквозняк на этой машине мерить перезапусками
                # нельзя: соседний боевой даёт монотонный дрейф (записка 25 §65). Переключатель
                # файлом даёт оба плеча в ОДНОМ временном окне -- тот же приём, что дал число
                # для tau дерева (§66). Опрос редкий: раз в 512 вызовов, то есть ~раз в 3 шага.
                _реж[1] += 1
                if _реж[1] % 512 == 1:
                    try:
                        with open(_ФАЙЛ) as _fh:
                            _реж[0] = 1 if _fh.read(4).strip() == "1" else 0
                    except Exception:
                        _реж[0] = 1
                if not _реж[0]:
                    return прежний(сам, input_)
            if _внутри[0]:                       # вложенный вызов -- отдаём прежнему пути
                return прежний(сам, input_)
            if getattr(сам, "world_size", 1) != 2:
                return прежний(сам, input_)
            if (input_.dtype is not torch.float16 or not input_.is_contiguous()
                    or input_.numel() % 2 != 0
                    or input_.numel() * 2 > _УЧАСТОК):
                return прежний(сам, input_)
            ключ = id(сам)
            счёт = _счёт.setdefault(ключ, [0])
            k = счёт[0]
            if k < _ОТ or k >= _ДО:             # вне окна -- прежним путём, но место СЧИТАЕМ
                счёт[0] += 1
                return прежний(сам, input_)
            if k >= _МЕСТ:                      # мест не хватило -- честно отдаём NCCL
                счёт[0] += 1
                return прежний(сам, input_)
            с = _сост.get(ключ)
            if с is None:
                if torch.cuda.is_current_stream_capturing():
                    return прежний(сам, input_)  # готовиться внутри захвата нельзя
                try:
                    с = _подготовить(сам)
                except Exception as ex:  # noqa: BLE001
                    print(f"[fa2_sm70 AR2] подготовка не удалась, иду прежним путём: {ex}",
                          file=sys.stderr, flush=True)
                    _сост[ключ] = False
                    return прежний(сам, input_)
                _сост[ключ] = с
            if с is False:
                return прежний(сам, input_)
            счёт[0] += 1
            # ГЕЙТ СЛИЧЕНИЯ: порча обмена ТИХАЯ, поэтому первые N сведений считаем ОБОИМИ
            # путями и сверяем. Эталон -- NCCL на копии; расхождение печатается с номером
            # места, чтобы было видно, на каком именно вызове ранги разъезжаются.
            эталон = None
            if _ГЕЙТ and _сверено[0] < _ГЕЙТ and not torch.cuda.is_current_stream_capturing():
                _внутри[0] = True
                try:
                    эталон = прежний(сам, input_.clone())
                finally:
                    _внутри[0] = False
            # НЕ НА МЕСТЕ. Штатный путь для части форм возвращает НОВЫЙ тензор, НЕ трогая
            # вход (`all_reduce_symmetric_with_copy` и подобные). Если свести вход на месте,
            # всё, что ещё держит его -- прежде всего ОСТАТОК слоя, сохранённый до вызова, --
            # получит уже сведённое значение. Порча растёт с числом затронутых слоёв, что и
            # объясняет «16 мест верно, 64 -- каша» при ВЕРНЫХ значениях каждого сведения.
            # Выключатель на всякий случай: FA2SM70_AR2_INPLACE=1 вернёт прежнее поведение.
            # БЕЗ КЛОНА: вход уезжает соседу как есть, сумма пишется в СВЕЖИЙ выход.
            цель_т = input_ if _НА_МЕСТЕ else torch.empty_like(input_)
            _яч = (input_.numel() // 2 + 2) // 3
            if (с.get("h") and с.get("ll_my") is not None and _яч <= с["ll_cells"]
                    and (_яч + 255) // 256 <= _ШИР
                    and (input_.data_ptr() | цель_т.data_ptr()) & 3 == 0):
                (_оп_ll[0] or (lambda *a: с["e"].ar1ll_all_reduce_out(*a)))(
                    input_, цель_т, с["ll_my"][k], с["ll_peer"][k], с["ll_seq"][k], с["срыв"], с["ll_cells"])
            elif с.get("h"):
                (_оп_h or (lambda *a: с["e"].ar1h_all_reduce_out(*a)))(
                    input_, цель_т, с["my_slot"][k], с["peer_slot"][k], с["my_flag"][k],
                    с["peer_flag"][k], с["seq"][k], с["срыв"], с["N"])
            elif _оп is not None:
                _оп(input_, цель_т, с["recv"][k], с["сосед_recv"][k],
                    с["flag"][k * _ШИР:(k + 1) * _ШИР], с["сосед_flag"] + 4 * _ШИР * k,
                    с["seq"][k], с["срыв"],
                    с["ack"][k * _ШИР:(k + 1) * _ШИР], с["сосед_ack"] + 4 * _ШИР * k)
            else:
                (с["e"].ar1_all_reduce_out if _AR1 else с["e"].ar2_all_reduce_out)(
                    input_, цель_т, с["recv"][k], с["сосед_recv"][k],
                    с["flag"][k * _ШИР:(k + 1) * _ШИР], с["сосед_flag"] + 4 * _ШИР * k,
                    с["seq"][k], с["срыв"],
                    с["ack"][k * _ШИР:(k + 1) * _ШИР], с["сосед_ack"] + 4 * _ШИР * k)
            if _ПОРЧА:
                цель_т.mul_(1.5)
            if эталон is not None:
                н = эталон.float().abs().max().item()
                # ПРИБОР БЫЛ СЛЕП: он сверялся на ПРОГРЕВОЧНЫХ проходах, где активации НУЛИ,
                # а на нулях любая порча невидима (1.5 * 0 == 0). Положительный контроль это
                # и вскрыл: намеренное умножение на 1.5 гейт НЕ ПОЙМАЛ, объявив «совпали».
                # Поэтому пустые тензоры НЕ ЗАСЧИТЫВАЮТСЯ и не сверяются.
                if н < 1e-6:
                    return цель_т
                _сверено[0] += 1
                d = (цель_т.float() - эталон.float()).abs().max().item()
                if d / н > 1e-3:
                    print(f"[fa2_sm70 AR2 ГЕЙТ] РАСХОЖДЕНИЕ на месте {k}: {d/н:.3e} "
                          f"(форма {tuple(цель_т.shape)}, срывов {int(с['срыв'][0].item())})",
                          file=sys.stderr, flush=True)
                elif _сверено[0] == _ГЕЙТ:
                    print(f"[fa2_sm70 AR2 ГЕЙТ] {_ГЕЙТ} сведений сверены с NCCL: совпали",
                          file=sys.stderr, flush=True)
            return цель_т
        return _f

    # Счётчик мест обнуляется на КАЖДОМ проходе -- иначе ранги разъедутся по комплектам.
    try:
        import vllm.forward_context as _фк
        _прежний_контекст = _фк.set_forward_context

        def _контекст(*a, **kw):
            # СРЫВ ОЖИДАНИЯ ЧИТАЕТСЯ НА КАЖДОМ ПРОХОДЕ. Спин имеет потолок по тактам, и при
            # срыве тело берёт УСТАРЕВШИЕ байты молча -- ровно та тихая порча, ради которой
            # ставился гейт. Без этой проверки срыв неотличим от верного обмена.
            # [25.10] БЕЗ СИНХРОНИЗАЦИИ: `.item()` на каждом проходе сливал очередь карты прямо перед
            # запуском графа цели (трасса: ~1.2 мс простоя на шаг, пока хозяин запускает граф). Флаг
            # копируется в закреплённую память хозяина без ожидания и читается на СЛЕДУЮЩЕМ проходе --
            # срыв виден с опозданием на один проход, но виден.
            _захват = torch.cuda.is_current_stream_capturing()
            _срыв_был = 0
            for _к, _с in _сост.items():
                if _с is False:
                    continue
                try:
                    _хост = _с.get("срыв_хост")
                    if _хост is None:
                        _хост = _с["срыв_хост"] = torch.zeros(1, dtype=_с["срыв"].dtype, pin_memory=True)
                    _в = int(_хост[0])
                    if _в:
                        print(f"[fa2_sm70 AR2] СРЫВ ОЖИДАНИЯ ({_в}) -- байты соседа не дошли за "
                              f"потолок, значения УСТАРЕЛИ", file=sys.stderr, flush=True)
                        _срыв_был = _срыв_был or _в
                        _хост.zero_()
                        if not _захват:
                            _с["срыв"][0].zero_()
                    if not _захват:
                        _хост.copy_(_с["срыв"][0:1], non_blocking=True)
                except Exception:  # noqa: BLE001
                    continue
            # [СРЫВ 04.11] Срыв -- ОТКАЗ, а не строчка в журнале: прошлый проход сложил устаревшие байты соседа,
            # и его выход (токены, ответы Clef) уже порченый. Тот же исход, что у срыва обмена i12: исключение
            # из шага; немой ранг завершает процесс (multiproc_executor), движок падает и поднимается заново.
            # FA2SM70_AR_STALL_FATAL=0 -- прежнее «напечатать и продолжить».
            if _срыв_был and _СРЫВ_ФАТАЛЕН:
                raise RuntimeError(f"сведение AR: срыв ожидания соседа (код {_срыв_был}) -- выход прошлого "
                                   f"прохода посчитан по устаревшим байтам")
            if _СЧЁТ_ВЫЗОВОВ is not None:
                # СЧЁТЧИК ЯДРА, а не питона: он растёт и при повторе графа.
                _СЧЁТ_ВЫЗОВОВ[0] += 1
                if _СЧЁТ_ВЫЗОВОВ[0] % 200 == 0:
                    try:
                        _c, _e = _расширение().ar2_counter_read(False)
                        print(f"[AR2 ЯДРО] шагов~{_СЧЁТ_ВЫЗОВОВ[0]} запусков_ядра {_c} "
                              f"= {_c/_СЧЁТ_ВЫЗОВОВ[0]:.1f} на шаг", file=sys.stderr, flush=True)
                    except Exception as _ex:
                        print(f"[AR2 ЯДРО] чтение не удалось: {_ex}", file=sys.stderr, flush=True)
            for с in _счёт.values():
                с[0] = 0
            return _прежний_контекст(*a, **kw)

        _фк.set_forward_context = _контекст
    except Exception as ex:  # noqa: BLE001
        print(f"[fa2_sm70 AR2] сброс счётчика не поставлен: {ex}", file=sys.stderr, flush=True)
        return

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
    if сколько:
        print(f"[fa2_sm70] СВОЁ СВЕДЕНИЕ TP включено (классов {сколько}, участок "
              f"{_УЧАСТОК >> 10} КиБ, мест {_МЕСТ}); крупные сведения идут прежним путём",
              file=sys.stderr, flush=True)
    else:
        print("[fa2_sm70 AR2] НИ ОДИН класс не подменён -- рычаг СЛЕП", file=sys.stderr, flush=True)
