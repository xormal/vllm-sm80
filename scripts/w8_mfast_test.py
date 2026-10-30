#!/usr/bin/env python3
"""[ЗАГРУЗКА ВЕСОВ 30.10] Проверка прямого GEMM w8a16_sm80_gemm_large с mfast=1 (окно обслуживания, одна карта).

Два процесса с разными .so одного имени (fa2sm70_w8a16_sm80) -- каталог сборки через FA2SM70_BUILD_DIR:
  --save  <файл.pt>   (старая .so): выходы прежнего пути на фиксированных весах и входах;
  --check <файл.pt>   (новая .so): mfast=0 и mfast=1 -- побитно против сохранённого прежнего; out= срез и x срез --
                      побитно против mfast=1 и без порчи соседних ячеек; время прежнего порядка против нового и
                      против декванта в fp16 + cuBLAS (как маршрут "q"); relL2 к fp32 по дескванту (здравость).
Формы -- слои модели на ранге TP2; (mi, splits) -- из кэша подбора боевого (тот же выбор, что в сервере).
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, "/home/alex/devpack_sm70_20261022/кандидат/vllm-sm70/sm70")
from fa2_sm70._ext import w8a16_sm80_ext  # noqa: E402

ФОРМЫ = [("qkvz", 5120, 8192), ("qkv", 5120, 7168), ("out_proj", 3072, 5120), ("o_proj", 10240, 2560),
         ("gate_up", 5120, 17408), ("down", 8704, 5120)]
M_НАБОР = (65, 130, 256, 300, 430, 512, 700, 1024, 1500, 2048)
КОРЗ = (32, 64, 128, 256, 512)
dev = torch.device("cuda")


def кэш(K, N):
    кат = os.environ.get("FA2SM80_W8_TUNE_DIR", os.path.expanduser("~/.cache/vllm-sm80/w8tune"))
    пути = glob.glob(os.path.join(кат, f"*_K{K}_N{N}_v2.json"))
    return json.load(open(пути[0])) if пути else None


def выбор(д, m):
    """(mi, splits) как в сервере: корзина прямого пути до 512, выше -- (4, 1)."""
    b = next((c for c in КОРЗ if c >= m), None)
    if д and b is not None and str(b) in д.get("dcfg", {}):
        return tuple(д["dcfg"][str(b)])
    return (4, 1)


def веса(K, N, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    qw = torch.randint(0, 256, (K, N), dtype=torch.int32, device=dev, generator=g)
    qwp = (qw[0::4] | (qw[1::4] << 8) | (qw[2::4] << 16) | (qw[3::4] << 24)).contiguous()
    z = torch.randint(100, 156, (K // 128, N), dtype=torch.int32, device=dev, generator=g)
    qz = (z[:, 0::4] | (z[:, 1::4] << 8) | (z[:, 2::4] << 16) | (z[:, 3::4] << 24)).contiguous()
    sc = (torch.rand(K // 128, N, device=dev, generator=g) * 2e-3 + 5e-4).half()
    return qwp, qz, sc


def вход(M, K, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    return (torch.randn(M, K, device=dev, generator=g) * 0.5).half()


def время(fn, n=10):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    лучшее = 1e9
    for _ in range(3):
        e0.record()
        for _ in range(n):
            fn()
        e1.record()
        torch.cuda.synchronize()
        лучшее = min(лучшее, e0.elapsed_time(e1) / n)
    return лучшее * 1e3   # мкс


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    ap.add_argument("--check")
    ap.add_argument("--json")
    ap.add_argument("--so", help="готовая .so (без сборки): прежняя боевая")
    a = ap.parse_args()
    if a.so:
        import importlib.util
        spec = importlib.util.spec_from_file_location("fa2sm70_w8a16_sm80", a.so)
        e = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(e)
    else:
        e = w8a16_sm80_ext()
    print("so:", e.__file__, flush=True)
    сохр = torch.load(a.check) if a.check else {}
    выход, итог = {}, []
    for si, (имя, K, N) in enumerate(ФОРМЫ):
        д = кэш(K, N)
        nw = int(д["nw"]) if д else 4
        qwp, qz, sc = веса(K, N, 100 + si)
        wf, meta = e.w8a16_sm80_pack(qwp, qz, sc, 128, False, nw)
        # рабочая область split-K -- как в сервере при загрузке (прежний путь растит её до 2 x 512 строк)
        e.w8a16_sm80_gemm_large(torch.zeros(512, K, device=dev, dtype=torch.float16), wf, meta, N, nw, 4, 2)
        Wd = e.w8a16_sm80_dequant(wf, meta, N, K, nw, torch.empty(N * K, dtype=torch.float16, device=dev)).view(N, K)
        for M in M_НАБОР:
            mi, sp = выбор(д, M)
            x = вход(M, K, 1000 * si + M)
            ключ = f"{имя}/{M}"
            if a.save:
                выход[ключ] = e.w8a16_sm80_gemm_large(x, wf, meta, N, nw, mi, sp).cpu()
                continue
            y0 = e.w8a16_sm80_gemm_large(x, wf, meta, N, nw, mi, sp, 0)
            y1 = e.w8a16_sm80_gemm_large(x, wf, meta, N, nw, mi, sp, 1)
            рез = {"форма": имя, "K": K, "N": N, "M": M, "mi": mi, "splits": sp, "nw": nw}
            if ключ in сохр:
                ст = сохр[ключ].to(dev)
                рез["mfast0_как_старая_so"] = bool(torch.equal(y0, ст))
                рез["mfast1_как_старая_so"] = bool(torch.equal(y1, ст))
            рез["mfast1_как_mfast0"] = bool(torch.equal(y0, y1))
            # out= срез строк большего тензора и x -- срез с шагом K+64
            big = torch.full((M + 6, N + 128), 7.0, dtype=torch.float16, device=dev)
            o = big[3:3 + M, 64:64 + N]
            xb = torch.zeros(M, K + 64, dtype=torch.float16, device=dev)
            xb[:, :K] = x
            e.w8a16_sm80_gemm_large(xb[:, :K], wf, meta, N, nw, mi, sp, 1, o)
            вне = torch.ones_like(big, dtype=torch.bool)
            вне[3:3 + M, 64:64 + N] = False
            рез["out_срез_побитно"] = bool(torch.equal(o, y1))
            рез["out_соседи_целы"] = bool((big[вне] == 7.0).all())
            ref = x.float() @ Wd.float().t()
            рез["relL2_к_fp32"] = float((y1.float() - ref).norm() / ref.norm())
            t0 = время(lambda: e.w8a16_sm80_gemm_large(x, wf, meta, N, nw, mi, sp, 0))
            t1 = время(lambda: e.w8a16_sm80_gemm_large(x, wf, meta, N, nw, mi, sp, 1))
            buf = torch.empty(N * K, dtype=torch.float16, device=dev)

            def _q():
                W = e.w8a16_sm80_dequant(wf, meta, N, K, nw, buf)
                return torch.mm(x, W.view(N, K).t())

            tq = время(_q)
            fl = 2.0 * M * N * K
            рез.update({"мкс_прежний": round(t0, 1), "мкс_mfast": round(t1, 1), "мкс_деквант_cublas": round(tq, 1),
                        "TF_прежний": round(fl / t0 / 1e6, 1), "TF_mfast": round(fl / t1 / 1e6, 1),
                        "TF_деквант": round(fl / tq / 1e6, 1)})
            итог.append(рез)
            print(json.dumps(рез, ensure_ascii=False), flush=True)
    if a.save:
        torch.save(выход, a.save)
        print(f"сохранено {len(выход)} выходов в {a.save}")
    if a.json:
        json.dump(итог, open(a.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
