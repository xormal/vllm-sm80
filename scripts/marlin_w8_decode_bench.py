#!/usr/bin/env python3
"""[ФРИЗЫ 30.10] Декод и малые M: боевой маршрут W8 (кэш подбора state/w8tune: w8a16_sm80_mm или прямой по корзине)
против Marlin u8b128 -- одна карта. Формы ранга TP2; M = 4, 8, 16, 40, 64. Выход: json (argv[1])."""
import json
import os
import sys

import torch

sys.path.insert(0, os.environ.get("FA2SM70_PATH") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sm70"))
from fa2_sm70._ext import w8a16_sm80_ext  # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # noqa: E402
    marlin_make_empty_g_idx, marlin_make_workspace_new, marlin_permute_scales)
from vllm.scalar_type import scalar_types  # noqa: E402

dev = torch.device("cuda")
e = w8a16_sm80_ext()
ТЮН = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "state", "w8tune")
ФОРМЫ = {"gate_up": (5120, 17408), "down": (8704, 5120), "qkvz": (5120, 8192), "out": (3072, 5120)}
СЛОЁВ = {"gate_up": 64, "down": 64, "qkvz": 48, "out": 64}


def время(fn, повт=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    лучшее = 1e9
    for _ in range(3):
        e0.record()
        for _ in range(повт):
            fn()
        e1.record()
        torch.cuda.synchronize()
        лучшее = min(лучшее, e0.elapsed_time(e1) / повт)
    return лучшее * 1e3


рез = {}
ws = marlin_make_workspace_new(dev, 4)
gi = marlin_make_empty_g_idx(dev)
for имя, (K, N) in ФОРМЫ.items():
    т = json.load(open(os.path.join(ТЮН, f"NVIDIA_DRIVE-PG199-PROD_K{K}_N{N}_v2.json")))
    nw = int(т["nw"])
    g = torch.Generator(device=dev).manual_seed(K + N)
    qw = torch.randint(-2**31, 2**31 - 1, (K // 4, N), dtype=torch.int32, device=dev, generator=g)
    qz = torch.randint(-2**31, 2**31 - 1, (K // 128, N // 4), dtype=torch.int32, device=dev, generator=g)
    sc = (torch.rand(K // 128, N, device=dev, generator=g) * 0.004 + 0.001).half()
    wf, meta = e.w8a16_sm80_pack(qw, qz, sc, 128, True, nw)
    mq = ops.gptq_marlin_repack(qw, gi, K, N, 8)
    ms = marlin_permute_scales(sc, K, N, 128)
    рез[имя] = []
    for M in (4, 8, 16, 40, 64):
        b = next(x for x in (4, 8, 16, 32, 64) if x >= M)
        x = (torch.randn(M, K, device=dev, generator=g) * 0.5).half()
        if т.get("route", {}).get(str(b)) == "d":
            mi, sp = т["dcfg"][str(b)]
            наш = время(lambda: e.w8a16_sm80_gemm_large(x, wf, meta, N, nw, mi, sp, 1))
            маршрут = f"прямой mi={mi} sp={sp}"
        else:
            wa, st, sp = т["cfg"][str(b)]
            наш = время(lambda: e.w8a16_sm80_mm(x, wf, meta, N, sp, nw, wa, st))
            маршрут = f"mm wa={wa} st={st} sp={sp}"
        мар = min(время(lambda fp=fp: ops.marlin_gemm(x, None, mq, None, ms, None, None, None, gi, gi, ws,
                                                     scalar_types.uint8b128, M, N, K, is_k_full=True,
                                                     use_atomic_add=False, use_fp32_reduce=fp, is_zp_float=False))
                  for fp in (True, False))
        r = {"M": M, "наш_мкс": round(наш, 1), "маршрут": маршрут, "marlin_мкс": round(мар, 1),
             "наш_ГБс": round(K * N / наш / 1e3, 0), "marlin_ГБс": round(K * N / мар / 1e3, 0)}
        рез[имя].append(r)
        print(имя, json.dumps(r, ensure_ascii=False), flush=True)
for i, M in enumerate((4, 8, 16, 40, 64)):
    tn = sum(СЛОЁВ[n] * рез[n][i]["наш_мкс"] for n in ФОРМЫ) / 1e3
    tm = sum(СЛОЁВ[n] * рез[n][i]["marlin_мкс"] for n in ФОРМЫ) / 1e3
    print(f"проход декода M={M}: боевой маршрут {tn:.1f} мс, Marlin {tm:.1f} мс", flush=True)
json.dump(рез, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
