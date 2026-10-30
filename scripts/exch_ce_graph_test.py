#!/usr/bin/env python3
"""[ОДИН ГРАФ 30.10] Стенд: 12-разрядный обмен копировщиком В CUDA-ГРАФЕ (exch_i12.cu ce_graph_*) -- две карты (0, 1).

Каждый ранг: граф из нескольких обменов разной длины (как куски конвейера блока), захват один раз, затем R повторов с
новыми данными во входах графа, вперемешку с eager-обменами прежнего пути (свои буферы, номер на хосте). Эталон -- та же
упаковка соседа, полученная по gloo, и тот же разбор: выход обязан совпасть ПОБИТНО. Плюс: нет срыва ожиданий (флаги),
номер на устройстве = число графовых обменов. Выход: json (argv[1]) от ранга 0.
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sm70"))

ДЛИНЫ = [512 * 5120, 1024 * 5120, 40 * 5120 * 8, 2048 * 5120, 2112 * 5120]   # элементов fp16 (кратно 128)
ПОВТОРОВ = int(os.environ.get("REPS", "300"))


def ранг_main(r, out_path):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    torch.cuda.set_device(r)
    dist.init_process_group("gloo", rank=r, world_size=2)
    from fa2_sm70._ext import exch_ext
    e = exch_ext()
    dev = torch.device("cuda", r)
    cap_g = 24 << 20
    recv_g = e.ipc_alloc(cap_g, torch.uint8)
    fl_g = e.ipc_alloc(2, torch.int32)
    recv_e = e.ipc_alloc(2 * cap_g, torch.uint8)        # eager-путь: чётности, номер на хосте
    fl_e = e.ipc_alloc(2, torch.int32)
    мои = [(bytes(e.ipc_handle(t)), e.ipc_offset(t)) for t in (recv_g, fl_g, recv_e, fl_e)]
    все = [None, None]
    dist.all_gather_object(все, мои)
    чуж = все[1 - r]
    p_recv_g = e.ipc_open(*чуж[0])
    p_fg = e.ipc_open(*чуж[1])
    p_recv_e = e.ipc_open(*чуж[2])
    p_fe = e.ipc_open(*чуж[3])
    seq_g = torch.zeros(1, dtype=torch.int32, device=dev)
    срыв = torch.zeros(2, dtype=torch.int32, device=dev)
    dist.barrier()

    def упаковать(x):
        n = x.numel()
        ngrp = n // 128
        nb_pk, nb = ngrp * 96 * 2, ngrp * 96 * 2 + ngrp * 2
        buf = torch.empty(nb, dtype=torch.uint8, device=dev)
        e.pack_i12_into(x, buf[:nb_pk].view(torch.int16), buf[nb_pk:nb].view(torch.float16), 12)
        return buf, nb_pk, nb

    def обмен_граф(x):
        buf, nb_pk, nb = упаковать(x)
        e.ce_graph_send(buf, p_recv_g, fl_g[1:2], p_fg, seq_g, срыв)
        e.ce_graph_wait(fl_g[0:1], seq_g, срыв)
        ч = recv_g[:nb]
        out = e.unpack_add_i12(x, ч[:nb_pk].view(torch.int16), ч[nb_pk:nb].view(torch.float16))
        e.ce_graph_ack(p_fg + 4, seq_g)
        return out

    сост_e = {"seq": 0}

    def обмен_eager(x):
        buf, nb_pk, nb = упаковать(x)
        par = сост_e["seq"] & 1
        сост_e["seq"] += 1
        val = сост_e["seq"]
        e.ar2_copy(buf, p_recv_e + par * cap_g)
        e.ar2_signal(p_fe, par, val)
        e.ar2_wait(fl_e, par, val, срыв)
        ч = recv_e[par * cap_g:par * cap_g + nb]
        return e.unpack_add_i12(x, ч[:nb_pk].view(torch.int16), ч[nb_pk:nb].view(torch.float16))

    def эталон(x):
        """упаковка соседа по gloo + тот же разбор"""
        buf, nb_pk, nb = упаковать(x)
        мой = buf.cpu()
        чужой = torch.empty_like(мой)
        if r == 0:
            dist.send(мой, 1); dist.recv(чужой, 1)
        else:
            dist.recv(чужой, 0); dist.send(мой, 0)
        ч = чужой.to(dev)
        return e.unpack_add_i12(x, ч[:nb_pk].view(torch.int16), ч[nb_pk:nb].view(torch.float16))

    входы = [torch.zeros(n, dtype=torch.float16, device=dev) for n in ДЛИНЫ]
    s = torch.cuda.Stream(device=dev)
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):                          # прогрев вне захвата (тот же путь, номер растёт у обоих)
        for x in входы:
            обмен_граф(x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    гр = torch.cuda.CUDAGraph()
    with torch.cuda.graph(гр):
        выходы = [обмен_граф(x) for x in входы]
    torch.cuda.synchronize()
    dist.barrier()
    рез = {"повторов": ПОВТОРОВ, "длины": ДЛИНЫ, "плохих": 0, "eager_плохих": 0}
    g = torch.Generator(device=dev).manual_seed(100 + r)
    t_гр = 0.0
    for k in range(ПОВТОРОВ):
        for x in входы:
            x.copy_(torch.randn(x.numel(), device=dev, generator=g).half())
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        гр.replay()
        torch.cuda.synchronize()
        t_гр += time.perf_counter() - t0
        if k % 10 == 0 or k < 5:
            for x, y in zip(входы, выходы):
                if not torch.equal(y, эталон(x)):
                    рез["плохих"] += 1
        if k % 3 == 0:                                   # eager-обмен между повторами (свои буферы)
            xe = torch.randn(ДЛИНЫ[k % len(ДЛИНЫ)], device=dev, generator=g).half()
            ye = обмен_eager(xe)
            if not torch.equal(ye, эталон(xe)):
                рез["eager_плохих"] += 1
    torch.cuda.synchronize()
    рез["срыв"] = срыв.tolist()
    рез["номер_на_устройстве"] = int(seq_g.item())
    рез["ожидаемый_номер"] = len(ДЛИНЫ) * (ПОВТОРОВ + 1)        # прогрев + повторы
    рез["мс_на_повтор"] = round(t_гр / ПОВТОРОВ * 1e3, 3)
    print(f"ранг {r}:", json.dumps(рез, ensure_ascii=False), flush=True)
    if r == 0 and out_path:
        json.dump(рез, open(out_path, "w"), ensure_ascii=False, indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(ранг_main, args=(sys.argv[1] if len(sys.argv) > 1 else "",), nprocs=2, join=True)
