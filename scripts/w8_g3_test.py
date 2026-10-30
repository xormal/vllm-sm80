#!/usr/bin/env python3
"""[GEMM 30.10] Стенд W8A16 v3 (w8a16_sm80_gemm3: x в порядке фрагментов, варп 64x64) -- одна карта.

1) побитно: v3 против прямого ядра (w8a16_sm80_gemm_large, mi=4) при том же расщеплении k, все инстансы v3,
   формы слоёв Qwen3.8-27B на ранге TP2, M от 65 до 4096; срез строк (упаковка целиком, вызов на подотрезок);
   отклонение от декванта в fp16 + GEMM (для справки: другой порядок сложения);
2) без хоста: упаковка + умножение под torch.cuda.set_sync_debug_mode("error");
3) граф: захват gemm3x со статическими входом и выходом, повтор с другими данными -- побитно против eager;
4) время: лучший v3 (с упаковкой и без), лучший прямой, деквант + GEMM; оценка GEMM-части прохода модели
   (48 слоёв GDN + 16 внимания, qkv внимания -- W12, не входит) при M = 430 (запись Clef) и 2088 (шаг бюджета).
Выход: json (argv[1]).
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sm70"))
from fa2_sm70._ext import w8a16_sm80_ext  # noqa: E402

dev = torch.device("cuda")
e = w8a16_sm80_ext()
print("so:", e.__file__, flush=True)
ИНСТ = [(2, 2, 3), (2, 2, 4), (1, 4, 3), (1, 4, 4), (2, 4, 3), (2, 4, 4), (4, 2, 3)]
ФОРМЫ = {"gate_up": (5120, 17408), "down": (8704, 5120), "qkvz": (5120, 8192), "out": (3072, 5120)}
СЛОЁВ = {"gate_up": 64, "down": 64, "qkvz": 48, "out": 64}     # out: out_proj GDN (48) + o_proj внимания (16)


def веса(K, N, seed, nw=4):
    g = torch.Generator(device=dev).manual_seed(seed)
    qw = torch.randint(-2**31, 2**31 - 1, (K // 4, N), dtype=torch.int32, device=dev, generator=g)
    qz = torch.randint(-2**31, 2**31 - 1, (K // 128, N // 4), dtype=torch.int32, device=dev, generator=g)
    sc = (torch.rand(K // 128, N, device=dev, generator=g) * 0.004 + 0.001).half()
    wf, meta = e.w8a16_sm80_pack(qw, qz, sc, 128, True, nw)
    return wf, meta


def вход(M, K, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    return (torch.randn(M, K, device=dev, generator=g) * 0.5).half()


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


def годится(wm, wn, N):
    return N % (64 * wn) == 0


def main():
    рез = {"побитно": [], "срез": [], "время": [], "проход": {}}
    nw = 4
    пакеты = {имя: веса(K, N, 11 + i, nw) for i, (имя, (K, N)) in enumerate(ФОРМЫ.items())}
    # рабочая область split-K и буфер упаковки -- заранее (как при загрузке весов)
    e.w8a16_sm80_g3_reserve(torch.empty(1, device=dev), 4096 * 8704 // 2, 4 * 512 * 17408, 1 << 16)
    # ---- 1) побитно
    for имя, (K, N) in ФОРМЫ.items():
        wf, meta = пакеты[имя]
        for M in (65, 100, 130, 256, 400, 512, 777, 1024, 2088, 4096):
            x = вход(M, K, M)
            эт = e.w8a16_sm80_gemm_large(x, wf, meta, N, nw, 4, 1)
            for wm, wn, st in ИНСТ:
                if not годится(wm, wn, N):
                    continue
                y = e.w8a16_sm80_gemm3x(x, wf, meta, N, nw, wm, wn, st, 1)
                рав = bool(torch.equal(y, эт))
                if not рав:
                    d = (y.float() - эт.float()).abs()
                    print("НЕ ПОБИТНО", имя, M, (wm, wn, st), float(d.max()), int((d > 0).sum()), flush=True)
                рез["побитно"].append({"форма": имя, "M": M, "cfg": [wm, wn, st, 1], "побитно": рав})
            if M <= 512:
                for sp in (2, 3, 4):
                    эт2 = e.w8a16_sm80_gemm_large(x, wf, meta, N, nw, 4, sp)
                    y2 = e.w8a16_sm80_gemm3x(x, wf, meta, N, nw, 2, 2, 3, sp)
                    рез["побитно"].append({"форма": имя, "M": M, "cfg": [2, 2, 3, sp], "побитно": bool(torch.equal(y2, эт2))})
            if M in (400, 2088):
                W = e.w8a16_sm80_dequant(wf, meta, N, K, nw)
                yq = torch.mm(x, W.view(N, K).t())
                rel = float((эт.float() - yq.float()).norm() / yq.float().norm())
                рез["побитно"].append({"форма": имя, "M": M, "rel_к_декванту": rel})
        ok = sum(r.get("побитно", True) for r in рез["побитно"] if r["форма"] == имя)
        n = sum("побитно" in r for r in рез["побитно"] if r["форма"] == имя)
        print(f"побитно {имя}: {ok}/{n}", flush=True)
    # срез: упаковка всего x, вызовы на куски строк (начала кратны 16) -- как конвейер блока
    K, N = ФОРМЫ["gate_up"]
    wf, meta = пакеты["gate_up"]
    x = вход(2088, K, 5)
    xa = e.w8a16_sm80_pack_a(x)
    mta = (2088 + 15) // 16
    for r0, r1 in ((0, 512), (512, 2048), (2048, 2088), (16, 400)):
        y = e.w8a16_sm80_gemm3(xa, r1 - r0, K, mta, r0 // 16, wf, meta, N, nw, 2, 2, 3, 1)
        эт = e.w8a16_sm80_gemm_large(x[r0:r1].contiguous(), wf, meta, N, nw, 4, 1)
        рез["срез"].append({"строки": [r0, r1], "побитно": bool(torch.equal(y, эт))})
    # выход в срез большего тензора (ldy) и вход-срез (ldx)
    big = torch.zeros(600, N + 256, dtype=torch.float16, device=dev)
    xb = вход(600, K + 128, 6)[:, 64:64 + K]
    e.w8a16_sm80_gemm3x(xb, wf, meta, N, nw, 2, 2, 3, 1, big[:, 128:128 + N])
    эт = e.w8a16_sm80_gemm_large(xb.contiguous(), wf, meta, N, nw, 4, 1)
    рез["срез"].append({"строки": "ldx/ldy", "побитно": bool(torch.equal(big[:, 128:128 + N], эт)),
                        "поля_нули": bool(big[:, :128].abs().max() == 0 and big[:, 128 + N:].abs().max() == 0)})
    print("срез:", рез["срез"], flush=True)
    # ---- 2) без хоста
    x = вход(400, K, 7)
    out = torch.empty(400, N, dtype=torch.float16, device=dev)
    e.w8a16_sm80_gemm3x(x, wf, meta, N, nw, 2, 2, 3, 2, out)
    torch.cuda.synchronize()
    try:
        torch.cuda.set_sync_debug_mode("error")
        e.w8a16_sm80_gemm3x(x, wf, meta, N, nw, 2, 2, 3, 2, out)
        рез["без_синхронизаций"] = True
    except RuntimeError as ex:
        рез["без_синхронизаций"] = False
        рез["синхронизация"] = str(ex)[:300]
    finally:
        torch.cuda.set_sync_debug_mode("default")
    print("без синхронизаций:", рез["без_синхронизаций"], flush=True)
    # ---- 3) граф
    xs = torch.zeros(430, K, dtype=torch.float16, device=dev)
    ys = torch.zeros(430, N, dtype=torch.float16, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            e.w8a16_sm80_gemm3x(xs, wf, meta, N, nw, 1, 4, 3, 2, ys)
    torch.cuda.current_stream().wait_stream(s)
    гр = torch.cuda.CUDAGraph()
    with torch.cuda.graph(гр):
        e.w8a16_sm80_gemm3x(xs, wf, meta, N, nw, 1, 4, 3, 2, ys)
    рез["граф"] = []
    for seed in (1, 2, 3):
        xs.copy_(вход(430, K, 100 + seed))
        гр.replay()
        эт = e.w8a16_sm80_gemm_large(xs, wf, meta, N, nw, 4, 2)
        рез["граф"].append(bool(torch.equal(ys, эт)))
    print("граф:", рез["граф"], flush=True)
    # ---- 4) время
    import importlib
    try:
        CG = importlib.import_module("fa2sm80_cg") if os.environ.get("FA2SM80_CG") == "1" else None
    except ImportError:
        CG = None
    сумма = {}
    for имя, (K, N) in ФОРМЫ.items():
        wf, meta = пакеты[имя]
        W = e.w8a16_sm80_dequant(wf, meta, N, K, nw)
        for M in (130, 256, 430, 512, 1024, 2088, 4096):
            x = вход(M, K, 9)
            fl = 2.0 * M * N * K
            # v3: лучшая конфигурация
            лучш = None
            for wm, wn, st in ИНСТ:
                if not годится(wm, wn, N):
                    continue
                for sp in ((1, 2, 3, 4) if M <= 512 else (1,)):
                    t = время(lambda: e.w8a16_sm80_gemm3x(x, wf, meta, N, nw, wm, wn, st, sp))
                    if лучш is None or t < лучш[0]:
                        лучш = (t, [wm, wn, st, sp])
            xa = e.w8a16_sm80_pack_a(x)
            mta = (M + 15) // 16
            wm, wn, st, sp = лучш[1]
            t_mm = время(lambda: e.w8a16_sm80_gemm3(xa, M, K, mta, 0, wf, meta, N, nw, wm, wn, st, sp))
            t_pk = время(lambda: e.w8a16_sm80_pack_a(x, xa))
            # прямой: лучший (mi, splits)
            пр = None
            for mi in (2, 4):
                for sp2 in ((1, 2, 3, 4) if M <= 512 else (1,)):
                    t = время(lambda: e.w8a16_sm80_gemm_large(x, wf, meta, N, nw, mi, sp2, 1))
                    if пр is None or t < пр[0]:
                        пр = (t, [mi, sp2])
            t_dq = время(lambda: e.w8a16_sm80_dequant(wf, meta, N, K, nw, W))
            t_cb = время(lambda: torch.mm(x, W.view(N, K).t()))
            r = {"форма": имя, "M": M, "v3_мкс": round(лучш[0], 1), "v3_cfg": лучш[1], "v3_TF": round(fl / лучш[0] / 1e6, 1),
                 "v3_без_упаковки_мкс": round(t_mm, 1), "упаковка_мкс": round(t_pk, 1),
                 "прямой_мкс": round(пр[0], 1), "прямой_cfg": пр[1], "прямой_TF": round(fl / пр[0] / 1e6, 1),
                 "деквант_мкс": round(t_dq, 1), "cublas_мкс": round(t_cb, 1),
                 "деквант+cublas_TF": round(fl / (t_dq + t_cb) / 1e6, 1), "cublas_TF": round(fl / t_cb / 1e6, 1)}
            рез["время"].append(r)
            print("время", json.dumps(r, ensure_ascii=False), flush=True)
            for путь, t in (("v3", лучш[0]), ("прямой", пр[0]), ("деквант+cublas", t_dq + t_cb), ("cublas_fp16", t_cb)):
                сумма.setdefault((M, путь), 0.0)
                сумма[(M, путь)] += СЛОЁВ[имя] * t
        del W
        torch.cuda.empty_cache()
    for (M, путь), t in sorted(сумма.items()):
        рез["проход"].setdefault(str(M), {})[путь] = round(t / 1e3, 2)    # мс на проход модели
    print("GEMM-часть прохода, мс:", json.dumps(рез["проход"], ensure_ascii=False), flush=True)
    if len(sys.argv) > 1:
        json.dump(рез, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
