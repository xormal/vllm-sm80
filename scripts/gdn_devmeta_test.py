#!/usr/bin/env python3
"""[ОДИН ГРАФ 30.10] Префилл GDN с метаданными чанков на устройстве (FA2SM80_GDN_DEVMETA) -- проверка на одной карте.

1) побитно: гибрид A100 с метаданными с хоста (prepare_chunk_*, _chunk_meta) против gdn_meta на устройстве -- выход o
   и финальное состояние на наборах длин (кратные 64, некратные, длина < 64, одна последовательность, 1..32 штук);
2) без хоста: путь DEVMETA под torch.cuda.set_sync_debug_mode("error") -- любая синхронизация карта->хост -- отказ;
3) граф: вызов DEVMETA захвачен в torch.cuda.graph при фиксированных (T, N); повтор с ДРУГИМ разбиением тех же T на
   N последовательностей и другими входами -- побитно против eager с метаданными с хоста;
4) время: CPU на вызов (без синхронизации) и GPU на вызов, оба пути.
Формы -- слой GDN Qwen3.8-27B на ранге TP2: H=24 голов значения, Hg=8 голов ключа, d=128, состояние fp32.
"""
import json
import os
import random
import sys
import time

import torch

H, HG, D = 24, 8, 128
dev = torch.device("cuda")


def входы(lens, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    T, N = sum(lens), len(lens)
    q = torch.randn(1, T, HG, D, device=dev, generator=g).half()
    k = torch.randn(1, T, HG, D, device=dev, generator=g).half()
    v = (torch.randn(1, T, H, D, device=dev, generator=g) * 0.5).half()
    gg = -torch.rand(1, T, H, device=dev, generator=g) * 0.3          # лог-затухание
    beta = torch.rand(1, T, H, device=dev, generator=g).half()
    h0 = torch.randn(N, H, D, D, device=dev, generator=g) * 0.05
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=dev)
    return q, k, v, gg, beta, h0, cu


def вызов(chunk_fn, q, k, v, gg, beta, h0, cu, devmeta):
    os.environ["FA2SM80_GDN_DEVMETA"] = "1" if devmeta else "0"
    return chunk_fn(q, k, v, gg, beta, scale=D ** -0.5, initial_state=h0.clone(), output_final_state=True,
                    cu_seqlens=cu, head_first=False, use_qk_l2norm_in_kernel=True)


def main():
    os.environ["FA2SM80_GDN_HYB"] = "1"
    from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule as chunk_fn
    рез = {"побитно": [], "граф": [], "время": []}
    random.seed(30)
    наборы = [[64], [100], [37], [8192], [2048], [1000, 2000, 333], [64, 64, 64], [17, 300, 5, 900, 64],
              [random.randint(1, 700) for _ in range(32)], [random.randint(200, 1500) for _ in range(6)]]
    for i, lens in enumerate(наборы):
        a = входы(lens, 100 + i)
        o0, h0 = вызов(chunk_fn, *a, devmeta=False)
        o1, h1 = вызов(chunk_fn, *a, devmeta=True)
        r = {"длины": lens if len(lens) <= 8 else f"{len(lens)} шт., T={sum(lens)}", "o": bool(torch.equal(o0, o1)),
             "ht": bool(torch.equal(h0, h1))}
        рез["побитно"].append(r)
        print("побитно", json.dumps(r, ensure_ascii=False), flush=True)
    # 2) без хоста
    a = входы([1000, 2000, 333], 7)
    вызов(chunk_fn, *a, devmeta=True)                    # прогрев: компиляция Triton вне режима
    torch.cuda.synchronize()
    try:
        torch.cuda.set_sync_debug_mode("error")
        вызов(chunk_fn, *a, devmeta=True)
        рез["без_синхронизаций"] = True
    except RuntimeError as e:
        рез["без_синхронизаций"] = False
        рез["синхронизация"] = str(e)[:300]
    finally:
        torch.cuda.set_sync_debug_mode("default")
    print("без синхронизаций:", рез["без_синхронизаций"], рез.get("синхронизация", ""), flush=True)
    # 3) граф: фиксированные T и N, статические буферы
    T, N = 3333, 5
    lensA = [1000, 2000, 133, 100, 100]
    st = list(входы(lensA, 11))
    os.environ["FA2SM80_GDN_DEVMETA"] = "1"
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            chunk_fn(*st[:5], scale=D ** -0.5, initial_state=st[5], output_final_state=True, cu_seqlens=st[6],
                     head_first=False, use_qk_l2norm_in_kernel=True)
    torch.cuda.current_stream().wait_stream(s)
    граф = torch.cuda.CUDAGraph()
    h0_стат = st[5].clone()
    with torch.cuda.graph(граф):
        o_г, ht_г = chunk_fn(*st[:5], scale=D ** -0.5, initial_state=h0_стат, output_final_state=True,
                             cu_seqlens=st[6], head_first=False, use_qk_l2norm_in_kernel=True)
    for j, lens in enumerate([[1000, 2000, 133, 100, 100], [64, 3000, 69, 100, 100], [3329, 1, 1, 1, 1],
                              [700, 700, 700, 700, 533]]):
        assert sum(lens) == T and len(lens) == N
        нов = входы(lens, 200 + j)
        for buf, src in zip(st, нов):
            buf.copy_(src)
        h0_стат.copy_(нов[5])
        граф.replay()
        o_e, ht_e = вызов(chunk_fn, *нов, devmeta=False)
        r = {"длины": lens, "o": bool(torch.equal(o_г, o_e)), "ht": bool(torch.equal(ht_г, ht_e))}
        рез["граф"].append(r)
        print("граф", json.dumps(r, ensure_ascii=False), flush=True)
    # 4) время
    for lens in ([2048], [400], [1000, 2000, 333]):
        a = входы(lens, 5)
        for dm in (False, True):
            for _ in range(3):
                вызов(chunk_fn, *a, devmeta=dm)
            torch.cuda.synchronize()
            n = 20
            t0 = time.perf_counter()
            for _ in range(n):
                вызов(chunk_fn, *a, devmeta=dm)
            t_cpu = (time.perf_counter() - t0) / n * 1e3
            torch.cuda.synchronize()
            t_all = (time.perf_counter() - t0) / n * 1e3
            r = {"длины": lens, "devmeta": dm, "мс_CPU_на_вызов": round(t_cpu, 3), "мс_до_конца_GPU": round(t_all, 3)}
            рез["время"].append(r)
            print("время", json.dumps(r, ensure_ascii=False), flush=True)
    if len(sys.argv) > 1:
        json.dump(рез, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
