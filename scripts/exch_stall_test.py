#!/usr/bin/env python3
"""[СРЫВ 04.11] Стенд: опоздание соседа в обмене TP -- две карты (0, 1), только при остановленном боевом.

Бой 04.11 08:21: TP1 объявил срыв ожидания обмена i12 (потолок был 2e9 тактов ~1.4 с), сосед опоздал дольше. Теперь
ожидание не решает об отказе: потолок по стенным часам FA2SM80_EXCH_WAIT_MS (боевой 300000) -- последняя страховка. Здесь ранг 0 перед обменом №ПОЗДНИЙ спит ЗАДЕРЖКА
секунд на хосте (то же, что остановка хозяина TP0 на повторе выделения), ранг 1 идёт дальше и ждёт на карте.

Два прогона, каждый -- свои процессы (потолок читается один раз на процесс):
  * потолок 300000 мс (боевой), задержка 3 с: срыва НЕТ, выход побитно = эталону (упаковка соседа по gloo + тот же разбор),
    статистика ожидания у ранга 1 -- около 3000 мс;
  * потолок 1000 мс, задержка 3 с -- ПОЛОЖИТЕЛЬНЫЙ КОНТРОЛЬ: срыв ОБЯЗАН случиться (и у eager-пути i12, и у AR1).
Пути: eager i12 (ar2_copy -> ar2_signal -> ar2_wait, как fa2sm70_exch_i12 при FA2SM80_EXCH_CE=1) и AR1 одним ядром
(ar1_all_reduce_out, потолок из памяти устройства -- ce_sync_wait_ceiling).

    python3 scripts/exch_stall_test.py OUT.json
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sm70"))

ДЛИНА = 2048 * 5120               # элементов fp16: кусок конвейера блока
ОБМЕНОВ = 8
ПОЗДНИЙ = 4                       # перед этим обменом ранг 0 спит
ДЛИНА_AR = 40 * 2048              # AR1: 40 кусков по 2048 (флагов и номеров -- 64)
ЗАДЕРЖКА = float(os.environ.get("STALL_S", "3.0"))


def ранг_main(r, итог, порт):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(порт)
    torch.cuda.set_device(r)
    dist.init_process_group("gloo", rank=r, world_size=2)
    from fa2_sm70._ext import exch_ext
    e = exch_ext()
    e.ce_sync_wait_ceiling()
    dev = torch.device("cuda", r)
    cap = 24 << 20
    recv = e.ipc_alloc(2 * cap, torch.uint8)
    fl = e.ipc_alloc(2, torch.int32)
    ar_recv = e.ipc_alloc(ДЛИНА_AR, torch.float16)
    ar_fl = e.ipc_alloc(64, torch.int32)
    ar_ack = e.ipc_alloc(64, torch.int32)
    мои = [(bytes(e.ipc_handle(t)), e.ipc_offset(t)) for t in (recv, fl, ar_recv, ar_fl, ar_ack)]
    все = [None, None]
    dist.all_gather_object(все, мои)
    p_recv, p_fl, p_ar_recv, p_ar_fl, p_ar_ack = [e.ipc_open(*h) for h in все[1 - r]]
    срыв = torch.zeros(4, dtype=torch.int32, device=dev)        # как st["срыв"] обмена i12
    ar_срыв = torch.zeros(2, dtype=torch.int32, device=dev)
    ar_seq = torch.zeros(64, dtype=torch.int32, device=dev)
    dist.barrier()

    def упаковать(x):
        ngrp = x.numel() // 128
        nb_pk, nb = ngrp * 96 * 2, ngrp * 96 * 2 + ngrp * 2
        buf = torch.empty(nb, dtype=torch.uint8, device=dev)
        e.pack_i12_into(x, buf[:nb_pk].view(torch.int16), buf[nb_pk:nb].view(torch.float16), 12)
        return buf, nb_pk, nb

    def обмен_i12(x, номер):
        buf, nb_pk, nb = упаковать(x)
        par = номер & 1
        e.ar2_copy(buf, p_recv + par * cap)
        e.ar2_signal(p_fl, par, номер + 1)
        e.ar2_wait(fl, par, номер + 1, срыв)
        ч = recv[par * cap:par * cap + nb]
        return e.unpack_add_i12(x, ч[:nb_pk].view(torch.int16), ч[nb_pk:nb].view(torch.float16))

    def эталон_i12(x):
        buf, nb_pk, nb = упаковать(x)
        мой = buf.cpu()
        чужой = torch.empty_like(мой)
        if r == 0:
            dist.send(мой, 1); dist.recv(чужой, 1)
        else:
            dist.recv(чужой, 0); dist.send(мой, 0)
        ч = чужой.to(dev)
        return e.unpack_add_i12(x, ч[:nb_pk].view(torch.int16), ч[nb_pk:nb].view(torch.float16))

    g = torch.Generator(device=dev).manual_seed(7 + r)
    рез = {"ранг": r, "потолок_мс": int(os.environ.get("FA2SM80_EXCH_WAIT_MS", "300000")), "задержка_с": ЗАДЕРЖКА,
           "i12_плохих": 0}
    # eager i12: обмены подряд в одном потоке, хозяин ранга 0 опаздывает перед обменом ПОЗДНИЙ
    входы = [torch.randn(ДЛИНА, device=dev, generator=g).half() for _ in range(ОБМЕНОВ)]
    torch.cuda.synchronize()
    dist.barrier()
    выходы = []
    for k, x in enumerate(входы):
        if r == 0 and k == ПОЗДНИЙ:
            time.sleep(ЗАДЕРЖКА)
        выходы.append(обмен_i12(x, k))
    torch.cuda.synchronize()
    рез["i12_срыв"] = срыв.tolist()            # [срыв, -, наибольшее ожидание мс, ожиданий > 100 мс]
    dist.barrier()
    for x, y in zip(входы, выходы):
        if not torch.equal(y, эталон_i12(x)):
            рез["i12_плохих"] += 1
    # AR1 одним ядром: то же опоздание
    xa = [torch.randn(ДЛИНА_AR, device=dev, generator=g).half() for _ in range(ОБМЕНОВ)]
    torch.cuda.synchronize()
    dist.barrier()
    ya = []
    for k, x in enumerate(xa):
        if r == 0 and k == ПОЗДНИЙ:
            time.sleep(ЗАДЕРЖКА)
        out = torch.empty_like(x)
        e.ar1_all_reduce_out(x, out, ar_recv, p_ar_recv, ar_fl, p_ar_fl, ar_seq, ar_срыв, ar_ack, p_ar_ack)
        ya.append(out)
    torch.cuda.synchronize()
    рез["ar1_срыв"] = ar_срыв.tolist()
    dist.barrier()
    # эталон AR1 -- своё + чужое (по gloo): сложение двух fp16 перестановочно, сверка побитная
    плохих = 0
    for x, y in zip(xa, ya):
        чужой = torch.empty_like(x.cpu())
        if r == 0:
            dist.send(x.cpu(), 1); dist.recv(чужой, 1)
        else:
            dist.recv(чужой, 0); dist.send(x.cpu(), 0)
        if not torch.equal(y, x + чужой.to(dev)):
            плохих += 1
    рез["ar1_плохих"] = плохих
    итог[r] = рез
    print(f"ранг {r}:", json.dumps(рез, ensure_ascii=False), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def прогон(потолок_мс, порт):
    os.environ["FA2SM80_EXCH_WAIT_MS"] = str(потолок_мс)
    итог = mp.Manager().dict()
    mp.spawn(ранг_main, args=(итог, порт), nprocs=2, join=True)
    return dict(итог)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else ""
    рез = {"боевой_300с": прогон(300000, 29581), "контроль_1с": прогон(1000, 29582)}
    т, к = рез["боевой_300с"], рез["контроль_1с"]
    вердикт = {
        "300 с: срыва нет": all(т[r]["i12_срыв"][0] == 0 and т[r]["ar1_срыв"][0] == 0 for r in (0, 1)),
        "300 с: выход побитно верен": all(т[r]["i12_плохих"] == 0 and т[r]["ar1_плохих"] == 0 for r in (0, 1)),
        "300 с: ранг 1 ждал >= 0.8 задержки": т[1]["i12_срыв"][2] >= 800 * ЗАДЕРЖКА,
        "1 с: срыв пойман (i12)": к[1]["i12_срыв"][0] != 0,
        "1 с: срыв пойман (AR1)": к[1]["ar1_срыв"][0] != 0,
    }
    рез["вердикт"] = вердикт
    print(json.dumps(вердикт, ensure_ascii=False, indent=1))
    if out:
        json.dump(рез, open(out, "w"), ensure_ascii=False, indent=1)
    sys.exit(0 if all(вердикт.values()) else 1)
