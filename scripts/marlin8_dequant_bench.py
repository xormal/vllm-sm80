#!/usr/bin/env python3
"""[окно 28] Деквант из раскладки Marlin (marlin8_dequant) против нашего (w8a16_sm80_dequant): побитно и время.

Большие шаги E (конвейер блока, M >= 2048) медленнее base на 25+ мс; деквант Marlin стоит в конвейере вместо нашего.
Варианты marlin8_dequant (аргумент variant): 2 -- прежний v2, 41 -- v4 по 128 k, 42 -- v4 по 256 k (marlin8_dequant.cuh).
Побитно -- на масштабах 1e-3..5e-3 и на крошечных 1e-7..1e-5 (субнормальные fp16). Предел -- копирование fp16 той же
длины (чтение 2 Б + запись 2 Б на элемент; деквант: 1 Б + 2 Б). Формы ранга TP2 и число слоёв -- как в marlin8_test.py;
итог -- деквант всех весов ранга за шаг. Выход: json (argv[1]).
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.environ.get("FA2SM70_PATH") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sm70"))
from fa2_sm70._ext import marlin8_ext, w8a16_sm80_ext  # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # noqa: E402
    marlin_make_empty_g_idx, marlin_permute_scales, marlin_zero_points)

dev = torch.device("cuda")
e = w8a16_sm80_ext()
m8 = marlin8_ext()
ТЮН = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "state", "w8tune")
ФОРМЫ = {"gate_up": (5120, 17408), "down": (8704, 5120), "qkvz": (5120, 8192), "out": (3072, 5120)}
СЛОЁВ = {"gate_up": 64, "down": 64, "qkvz": 48, "out": 64}


def время(fn, повт=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    лучшее = 1e9
    for _ in range(5):
        e0.record()
        for _ in range(повт):
            fn()
        e1.record()
        torch.cuda.synchronize()
        лучшее = min(лучшее, e0.elapsed_time(e1) / повт)
    return лучшее * 1e3


ВАРИАНТЫ = (2, 41, 42)


def main(out_path):
    gi = marlin_make_empty_g_idx(dev)
    рез = {"формы": {}, "побитно": {str(v): True for v in ВАРИАНТЫ}}
    шаг = {**{str(v): 0.0 for v in ВАРИАНТЫ}, "наш": 0.0, "копия": 0.0, "i8_в_fp16": 0.0}
    for имя, (K, N) in ФОРМЫ.items():
        т = json.load(open(os.path.join(ТЮН, f"NVIDIA_DRIVE-PG199-PROD_K{K}_N{N}_v2.json")))
        nw = int(т["nw"])
        g = torch.Generator(device=dev).manual_seed(K * 3 + N)
        qw = torch.randint(-2**31, 2**31 - 1, (K // 4, N), dtype=torch.int32, device=dev, generator=g)
        qz = torch.randint(-2**31, 2**31 - 1, (K // 128, N // 4), dtype=torch.int32, device=dev, generator=g)
        z = torch.stack([(qz >> (8 * i)) & 0xFF for i in range(4)], -1).reshape(K // 128, N)
        mz = marlin_zero_points(z.int(), K // 128, N, 8).to(dev)
        mq = ops.gptq_marlin_repack(qw, gi, K, N, 8)
        r = {}
        for режим, (lo, hi) in (("обычные", (1e-3, 5e-3)), ("крошечные", (1e-7, 1e-5))):
            sc = (torch.rand(K // 128, N, device=dev, generator=g) * (hi - lo) + lo).half()
            wf, meta = e.w8a16_sm80_pack(qw, qz, sc, 128, True, nw)
            W = e.w8a16_sm80_dequant(wf, meta, N, K, nw)                  # эталон: наш деквант
            ms = marlin_permute_scales(sc, K, N, 128)
            Wm = torch.empty_like(W)
            for v in ВАРИАНТЫ:
                Wm.fill_(1.0)
                m8.marlin8_dequant(mq, ms, mz, N, K, Wm, v)
                ок = bool(torch.equal(Wm, W))
                r[f"v{v}_побитно_{режим}"] = ок
                рез["побитно"][str(v)] = рез["побитно"][str(v)] and ок
        гб = N * K * 3 / 1e9                                              # int8 читается, fp16 пишется
        for v in ВАРИАНТЫ:
            t = время(lambda v=v: m8.marlin8_dequant(mq, ms, mz, N, K, Wm, v))
            r[f"v{v}_мкс"] = round(t, 1)
            r[f"v{v}_ГБ_с"] = round(гб / t * 1e6)
            шаг[str(v)] += СЛОЁВ[имя] * t / 1e3
        tн = время(lambda: e.w8a16_sm80_dequant(wf, meta, N, K, nw, Wm))
        tк = время(lambda: Wm.copy_(W))
        i8 = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=dev, generator=g)
        tп = время(lambda: Wm.copy_(i8))                                   # предел нашей схемы: 1 Б чтения, 2 Б записи
        del i8
        r.update({"наш_мкс": round(tн, 1), "наш_ГБ_с": round(гб / tн * 1e6), "копия_мкс": round(tк, 1),
                  "копия_ГБ_с": round(N * K * 4 / 1e9 / tк * 1e6), "i8_в_fp16_мкс": round(tп, 1),
                  "i8_в_fp16_ГБ_с": round(гб / tп * 1e6)})
        шаг["i8_в_fp16"] += СЛОЁВ[имя] * tп / 1e3
        шаг["наш"] += СЛОЁВ[имя] * tн / 1e3
        шаг["копия"] += СЛОЁВ[имя] * tк / 1e3
        рез["формы"][имя] = r
        print(имя, json.dumps(r, ensure_ascii=False), flush=True)
        del W, Wm, mq, wf, meta
        torch.cuda.empty_cache()
    рез["шаг_мс"] = {k: round(v, 2) for k, v in шаг.items()}
    годные = [v for v in ВАРИАНТЫ if рез["побитно"][str(v)]]
    рез["лучший"] = min(годные, key=lambda v: шаг[str(v)]) if годные else None
    print("деквант всех весов ранга за шаг, мс:", json.dumps(рез["шаг_мс"], ensure_ascii=False), "побитно:",
          json.dumps(рез["побитно"]), "лучший:", рез["лучший"], flush=True)
    json.dump(рез, open(out_path, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
