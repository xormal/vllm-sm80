#!/usr/bin/env python3
"""[ФРИЗЫ 30.10] Стенд: штатный Marlin W8A16 (u8b128, уже собран в vllm/_C для sm_80) против нашего маршрута W8 -- одна карта.

Окно 20/21: шаг с куском Clef ~440 токенов -- 273 мс, из них GEMM весов 150 мс (~79 TF; на 2048 токенах тот же GEMM
даёт 185 TF), обмен TP 62 мс. Это и есть «фриз» чатов: пока идёт шаг с куском Clef, декод стоит. Мелкие куски
(256-512) не спасают, пока GEMM на средних M так неэффективен. Marlin режет M на срезы по 64 строки (веса
перечитываются на каждый срез) -- выигрыш на средних M нужно измерить, а не предполагать.

Модель -- асимметричный GPTQ int8 (нули 21..231), Marlin u8b128 -- симметричный: для ВРЕМЕНИ это не важно (шаблон
Marlin умеет kU8 с нулями, нужны только экземпляры ядер), для ТОЧНОСТИ здесь -- сверка u8b128 с деквантом (q-128)*s.
Формы -- ранг TP2 Qwen3.8-27B; M от 16 до 4096. Для каждого M: Marlin (fp32-сведение да/нет), наш прямой
w8a16_sm80_gemm_large (mi 2/4, расщепление 1/2), v3 (gemm3x, лучший экземпляр), деквант + cuBLAS, и fp16 cuBLAS на
готовых fp16-весах (потолок). Итог -- оценка GEMM-части прохода модели при M = 64, 256, 440, 1024, 2088.
Выход: json (argv[1]).
"""
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
ФОРМЫ = {"gate_up": (5120, 17408), "down": (8704, 5120), "qkvz": (5120, 8192), "out": (3072, 5120)}
СЛОЁВ = {"gate_up": 64, "down": 64, "qkvz": 48, "out": 64}
МЫ = [int(x) for x in os.environ.get("MS", "16,40,64,128,256,440,512,768,1024,2088,4096").split(",")]
ИНСТ = [(2, 2, 3), (2, 2, 4), (1, 4, 3), (1, 4, 4), (2, 4, 3), (2, 4, 4), (4, 2, 3)]
# v4 (конвейер Marlin, w8a16_sm80_g4.cuh): (wm, wn, nj, stages); есть только в сборке с g4
ИНСТ4 = [(1, 4, 8, 3), (1, 4, 8, 4), (2, 2, 8, 3), (2, 4, 8, 3), (1, 4, 4, 3), (1, 4, 4, 4), (1, 8, 4, 3), (1, 8, 4, 4),
         (2, 2, 4, 3), (2, 4, 4, 3), (4, 2, 4, 3)]
РАСЩ = (1, 2, 3, 4)
ПОБИТНО_M = (130, 440, 1024)


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
    return лучшее * 1e3            # мкс


def main(out_path):
    e.w8a16_sm80_g3_reserve(torch.empty(1, device=dev), 4096 * 8704 // 2, 4 * 512 * 17408, 1 << 16)
    ws = marlin_make_workspace_new(dev, 4)
    g_idx = marlin_make_empty_g_idx(dev)
    рез = {"формы": {}, "проход": {}}
    for имя, (K, N) in ФОРМЫ.items():
        g = torch.Generator(device=dev).manual_seed(K * 7 + N)
        qw = torch.randint(-2**31, 2**31 - 1, (K // 4, N), dtype=torch.int32, device=dev, generator=g)
        qz = torch.randint(-2**31, 2**31 - 1, (K // 128, N // 4), dtype=torch.int32, device=dev, generator=g)
        sc = (torch.rand(K // 128, N, device=dev, generator=g) * 0.004 + 0.001).half()
        wf, meta = e.w8a16_sm80_pack(qw, qz, sc, 128, True, 4)
        W = e.w8a16_sm80_dequant(wf, meta, N, K, 4)                    # наш деквант (с нулями): fp16 [N, K]
        mq = ops.gptq_marlin_repack(qw, g_idx, K, N, 8)
        ms = marlin_permute_scales(sc, K, N, 128)
        # эталон для u8b128: (q - 128) * s
        qb = torch.stack([(qw >> (8 * i)) & 0xFF for i in range(4)], 1).reshape(K, N).float()
        W128 = ((qb - 128.0) * sc.float().repeat_interleave(128, 0)).half()            # [K, N]
        del qb
        строки = []
        for M in МЫ:
            x = (torch.randn(M, K, device=dev, generator=g) * 0.5).half()

            def marlin(fp32):
                return ops.marlin_gemm(x, None, mq, None, ms, None, None, None, g_idx, g_idx, ws,
                                       scalar_types.uint8b128, M, N, K, is_k_full=True, use_atomic_add=False,
                                       use_fp32_reduce=fp32, is_zp_float=False)
            ref = torch.mm(x.float(), W128.float())
            ym = marlin(True)
            err = ((ym.float() - ref).norm() / ref.norm()).item()
            t_m = {fp: время(lambda fp=fp: marlin(fp)) for fp in (True, False)}
            t_d = min(время(lambda mi=mi, sp=sp: e.w8a16_sm80_gemm_large(x, wf, meta, N, 4, mi, sp, 1))
                      for mi in (2, 4) for sp in (1, 2))
            t_g3 = t_g4 = None
            g4_конф, g4_плохих = None, 0
            if M > 64:
                ts = []
                for wm, wn, st in ИНСТ:
                    if N % (64 * wn):
                        continue
                    for sp in РАСЩ:
                        ts.append(время(lambda: e.w8a16_sm80_gemm3x(x, wf, meta, N, 4, wm, wn, st, sp)))
                t_g3 = min(ts) if ts else None
                if hasattr(e, "w8a16_sm80_gemm4x"):
                    for wm, wn, nj, st in ИНСТ4:
                        if N % (8 * nj * wn):
                            continue
                        for sp in РАСЩ:
                            if M in ПОБИТНО_M:          # побитно против прямого ядра при том же расщеплении
                                y4 = e.w8a16_sm80_gemm4x(x, wf, meta, N, 4, wm, wn, nj, st, sp)
                                if not torch.equal(y4, e.w8a16_sm80_gemm_large(x, wf, meta, N, 4, 4, sp, 1)):
                                    g4_плохих += 1
                            t = время(lambda: e.w8a16_sm80_gemm4x(x, wf, meta, N, 4, wm, wn, nj, st, sp))
                            if t_g4 is None or t < t_g4:
                                t_g4, g4_конф = t, (wm, wn, nj, st, sp)
            Wb = torch.empty_like(W)
            t_dq = время(lambda: e.w8a16_sm80_dequant(wf, meta, N, K, 4, Wb))
            t_cb = время(lambda: torch.mm(x, W.view(N, K).t()))
            тф = lambda t: round(2 * M * N * K / t / 1e6, 1)              # noqa: E731
            наш = min(t for t in (t_d, t_g3, t_dq + t_cb) if t)
            наш4 = min(t for t in (t_d, t_g3, t_g4, t_dq + t_cb) if t)
            r = {"M": M, "marlin_мкс": round(min(t_m.values()), 1), "marlin_fp32_мкс": round(t_m[True], 1),
                 "прямой_мкс": round(t_d, 1), "v3_мкс": round(t_g3, 1) if t_g3 else None,
                 "v4_мкс": round(t_g4, 1) if t_g4 else None, "v4_конф": g4_конф, "v4_побитно_плохих": g4_плохих,
                 "v4_тф": тф(t_g4) if t_g4 else None, "наш_с_v4_мкс": round(наш4, 1),
                 "деквант_мкс": round(t_dq, 1), "cublas_мкс": round(t_cb, 1), "наш_лучший_мкс": round(наш, 1),
                 "marlin_тф": тф(min(t_m.values())), "наш_тф": тф(наш), "fp16_потолок_тф": тф(t_cb),
                 "marlin_отн_ошибка": round(err, 5)}
            строки.append(r)
            print(имя, json.dumps(r, ensure_ascii=False), flush=True)
        рез["формы"][имя] = строки
        del W, W128, mq, wf, meta
        torch.cuda.empty_cache()
    for M in МЫ:
        tm = sum(СЛОЁВ[n] * next(r for r in рез["формы"][n] if r["M"] == M)["marlin_мкс"] for n in ФОРМЫ) / 1e3
        tn = sum(СЛОЁВ[n] * next(r for r in рез["формы"][n] if r["M"] == M)["наш_лучший_мкс"] for n in ФОРМЫ) / 1e3
        t4 = sum(СЛОЁВ[n] * next(r for r in рез["формы"][n] if r["M"] == M)["наш_с_v4_мкс"] for n in ФОРМЫ) / 1e3
        рез["проход"][M] = {"marlin_мс": round(tm, 1), "наш_мс": round(tn, 1), "наш_с_v4_мс": round(t4, 1)}
        print(f"проход модели M={M}: GEMM весов Marlin {tm:.1f} мс, наш лучший {tn:.1f} мс, с v4 {t4:.1f} мс", flush=True)
    json.dump(рез, open(out_path, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
