#!/usr/bin/env python3
"""[ФРИЗЫ 30.10] Стенд marlin8 (Marlin kU8 + нули в нашей сборке, marlin8_sm80.cu) -- одна карта.

1) точность: против нашего декванта (w8a16_sm80_dequant: s * (q - z), асимметричный GPTQ, нули v2) + GEMM в fp32 --
   отн. ошибка того же порядка, что у нашего маршрута (fp16-веса, fp32-накопление, другой порядок сложения);
1б) деквант из раскладки Marlin (marlin8_dequant) ПОБИТНО против нашего w8a16_sm80_dequant (большие M идут им);
2) граф: захват marlin8_gemm со статическими входом и выходом, повтор с другими данными -- побитно против eager;
3) время: marlin8 против штатного Marlin u8b128 (то же ядро без нулей) и боевого маршрута: M <= 64 -- кэш подбора
   (w8a16_sm80_mm / прямой), M > 64 -- лучшее из прямого (mi 2/4, расщ. 1/2) и декванта + cuBLAS.
Формы ранга TP2; итог -- GEMM-часть прохода модели по M. Выход: json (argv[1]).
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.environ.get("FA2SM70_PATH") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sm70"))
from fa2_sm70._ext import marlin8_ext, w8a16_sm80_ext  # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # noqa: E402
    marlin_make_empty_g_idx, marlin_make_workspace_new, marlin_permute_scales, marlin_zero_points)
from vllm.scalar_type import scalar_types  # noqa: E402

dev = torch.device("cuda")
e = w8a16_sm80_ext()
m8 = marlin8_ext()
ТЮН = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "state", "w8tune")
ФОРМЫ = {"gate_up": (5120, 17408), "down": (8704, 5120), "qkvz": (5120, 8192), "out": (3072, 5120)}
СЛОЁВ = {"gate_up": 64, "down": 64, "qkvz": 48, "out": 64}
МЫ = (1, 4, 16, 40, 64, 128, 256, 440, 512, 1024, 2088)


def время(fn, повт=10):
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


def main(out_path):
    m8.marlin8_reserve(torch.empty(1, device=dev))
    ws = marlin_make_workspace_new(dev, 4)
    gi = marlin_make_empty_g_idx(dev)
    рез = {"формы": {}, "проход": {}, "граф_побитно": None, "деквант_побитно": {}}
    for имя, (K, N) in ФОРМЫ.items():
        т = json.load(open(os.path.join(ТЮН, f"NVIDIA_DRIVE-PG199-PROD_K{K}_N{N}_v2.json")))
        nw = int(т["nw"])
        g = torch.Generator(device=dev).manual_seed(K * 3 + N)
        qw = torch.randint(-2**31, 2**31 - 1, (K // 4, N), dtype=torch.int32, device=dev, generator=g)
        qz = torch.randint(-2**31, 2**31 - 1, (K // 128, N // 4), dtype=torch.int32, device=dev, generator=g)
        sc = (torch.rand(K // 128, N, device=dev, generator=g) * 0.004 + 0.001).half()
        wf, meta = e.w8a16_sm80_pack(qw, qz, sc, 128, True, nw)
        W = e.w8a16_sm80_dequant(wf, meta, N, K, nw)                       # [N, K] fp16: s * (q - z)
        # Marlin: вес, масштабы, нули (распакованы из GPTQ: 4 нуля на int32 по столбцам, младший байт -- первый)
        mq = ops.gptq_marlin_repack(qw, gi, K, N, 8)
        ms = marlin_permute_scales(sc, K, N, 128)
        z = torch.stack([(qz >> (8 * i)) & 0xFF for i in range(4)], -1).reshape(K // 128, N)
        mz = marlin_zero_points(z.int(), K // 128, N, 8).to(dev)
        mq0 = ops.gptq_marlin_repack(qw, gi, K, N, 8)                     # штатный u8b128 (для времени)
        Wm = m8.marlin8_dequant(mq, ms, mz, N, K)
        рез["деквант_побитно"][имя] = bool(torch.equal(Wm, W))
        print(имя, "деквант Marlin против нашего: побитно", рез["деквант_побитно"][имя], flush=True)
        # [окно 27] время декванта: большие шаги E медленнее base, деквант Marlin стоит в конвейере блока
        tdm = время(lambda: m8.marlin8_dequant(mq, ms, mz, N, K, Wm))
        tdн = время(lambda: e.w8a16_sm80_dequant(wf, meta, N, K, nw, Wm))
        рез.setdefault("деквант_мкс", {})[имя] = {"marlin8": round(tdm, 1), "наш": round(tdн, 1),
                                                   "ГБ_с_marlin8": round(N * K * 3 / tdm / 1e3, 0)}
        print(имя, "деквант, мкс: marlin8", round(tdm, 1), "наш", round(tdн, 1), flush=True)
        del Wm
        строки = []
        for M in МЫ:
            x = (torch.randn(M, K, device=dev, generator=g) * 0.5).half()
            ref = torch.mm(x.float(), W.float().t())
            y8 = m8.marlin8_gemm(x, mq, ms, mz, N)
            err = ((y8.float() - ref).norm() / ref.norm()).item()
            # наш маршрут для сравнения точности (тот же эталон)
            yq = e.w8a16_sm80_gemm_large(x, wf, meta, N, nw, 4, 1, 1)
            err_q = ((yq.float() - ref).norm() / ref.norm()).item()
            t8 = min(время(lambda fp=fp: m8.marlin8_gemm(x, mq, ms, mz, N, fp)) for fp in (True, False))
            t0 = время(lambda: ops.marlin_gemm(x, None, mq0, None, ms, None, None, None, gi, gi, ws,
                                               scalar_types.uint8b128, M, N, K, is_k_full=True, use_atomic_add=False,
                                               use_fp32_reduce=True, is_zp_float=False))
            if M <= 64:
                b = next(v for v in (4, 8, 16, 32, 64) if v >= M)
                if т.get("route", {}).get(str(b)) == "d":
                    mi, sp = т["dcfg"][str(b)]
                    tн = время(lambda: e.w8a16_sm80_gemm_large(x, wf, meta, N, nw, mi, sp, 1))
                else:
                    wa, st, sp = т["cfg"][str(b)]
                    tн = время(lambda: e.w8a16_sm80_mm(x, wf, meta, N, sp, nw, wa, st))
            else:
                td = min(время(lambda mi=mi, sp=sp: e.w8a16_sm80_gemm_large(x, wf, meta, N, nw, mi, sp, 1))
                         for mi in (2, 4) for sp in (1, 2))
                Wb = torch.empty_like(W)
                tq = время(lambda: e.w8a16_sm80_dequant(wf, meta, N, K, nw, Wb)) + время(lambda: torch.mm(x, W.t()))
                tн = min(td, tq)
            r = {"M": M, "marlin8_мкс": round(t8, 1), "marlin_u8b128_мкс": round(t0, 1), "наш_мкс": round(tн, 1),
                 "marlin8_ош": round(err, 6), "наш_ош": round(err_q, 6), "выигрыш": round(tн / t8, 2)}
            строки.append(r)
            print(имя, json.dumps(r, ensure_ascii=False), flush=True)
        if имя == "gate_up":                                             # граф: повтор с новыми данными
            xs = torch.zeros(440, K, device=dev, dtype=torch.half)
            ys = torch.empty(440, N, device=dev, dtype=torch.half)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                m8.marlin8_reserve(xs)
                m8.marlin8_gemm(xs, mq, ms, mz, N, True, ys)
            torch.cuda.current_stream().wait_stream(s)
            гр = torch.cuda.CUDAGraph()
            with torch.cuda.graph(гр):
                m8.marlin8_gemm(xs, mq, ms, mz, N, True, ys)
            плохих = 0
            for k in range(5):
                xs.copy_(torch.randn(440, K, device=dev, generator=g).half())
                гр.replay()
                torch.cuda.synchronize()
                if not torch.equal(ys, m8.marlin8_gemm(xs, mq, ms, mz, N, True)):
                    плохих += 1
            рез["граф_побитно"] = f"{5 - плохих}/5"
            print("граф: побитно", рез["граф_побитно"], flush=True)
        рез["формы"][имя] = строки
        del W, mq, mq0, wf, meta
        torch.cuda.empty_cache()
    for i, M in enumerate(МЫ):
        t8 = sum(СЛОЁВ[n] * рез["формы"][n][i]["marlin8_мкс"] for n in ФОРМЫ) / 1e3
        tн = sum(СЛОЁВ[n] * рез["формы"][n][i]["наш_мкс"] for n in ФОРМЫ) / 1e3
        рез["проход"][M] = {"marlin8_мс": round(t8, 1), "наш_мс": round(tн, 1)}
        print(f"проход M={M}: marlin8 {t8:.1f} мс, боевой маршрут {tн:.1f} мс", flush=True)
    json.dump(рез, open(out_path, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
