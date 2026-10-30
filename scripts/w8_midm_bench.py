#!/usr/bin/env python3
"""[СРЕДНИЕ M 30.10] Средние M (строки записи Clef, 65..1024): нынешний прямой GEMM (w8a16_sm80_gemm_large с (mi, splits)
из кэша подбора боевого) против v2 (w8a16_sm80_gemm2: блок 128 x 32*nj, один блок на SM, конвейер фрагментов) при
nj 4/8, стадиях 3/4, PIN, расщеплении k 1..4, и против декванта + cuBLAS. Побитное совпадение v2 с прямым ядром при
splits=1 (одинаковый порядок сложения по k) -- значит маршрут на v2 не меняет выходов. Одна карта, окно обслуживания.
Профиль окна 7: шаг с записью Clef ~400 токенов -- прямой GEMM 111 мс из ~212 мс GPU-работы ранга.
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
M_НАБОР = (130, 256, 300, 430, 512, 700, 1024)
КОРЗ = (32, 64, 128, 256, 512)
dev = torch.device("cuda")


def кэш(K, N):
    кат = os.environ.get("FA2SM80_W8_TUNE_DIR", os.path.expanduser("~/.cache/vllm-sm80/w8tune"))
    пути = glob.glob(os.path.join(кат, f"*_K{K}_N{N}_v2.json"))
    return json.load(open(пути[0])) if пути else None


def выбор(д, m):
    b = next((c for c in КОРЗ if c >= m), None)
    if д and b is not None and str(b) in д.get("dcfg", {}):
        return tuple(д["dcfg"][str(b)])
    return (4, 1)


def время(fn, n=20):
    for _ in range(3):
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
    return лучшее * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    e = w8a16_sm80_ext()
    print("so:", e.__file__, flush=True)
    итог = []
    for si, (имя, K, N) in enumerate(ФОРМЫ):
        д = кэш(K, N)
        nw = int(д["nw"]) if д else 4
        g = torch.Generator(device=dev).manual_seed(100 + si)
        qw = torch.randint(0, 256, (K, N), dtype=torch.int32, device=dev, generator=g)
        qwp = (qw[0::4] | (qw[1::4] << 8) | (qw[2::4] << 16) | (qw[3::4] << 24)).contiguous()
        z = torch.randint(100, 156, (K // 128, N), dtype=torch.int32, device=dev, generator=g)
        qz = (z[:, 0::4] | (z[:, 1::4] << 8) | (z[:, 2::4] << 16) | (z[:, 3::4] << 24)).contiguous()
        sc = (torch.rand(K // 128, N, device=dev, generator=g) * 2e-3 + 5e-4).half()
        wf, meta = e.w8a16_sm80_pack(qwp, qz, sc, 128, False, nw)
        del qw
        buf = torch.empty(N * K, dtype=torch.float16, device=dev)
        for M in M_НАБОР:
            x = (torch.randn(M, K, device=dev, generator=g) * 0.5).half()
            mi, sp = выбор(д, M) if M <= 512 else (4, 1)
            fl = 2.0 * M * N * K
            y_ref1 = e.w8a16_sm80_gemm_large(x, wf, meta, N, nw, 4, 1)          # splits=1: эталон порядка сложения
            t_cur = время(lambda: e.w8a16_sm80_gemm_large(x, wf, meta, N, nw, mi, sp))
            t_q = время(lambda: torch.mm(x, e.w8a16_sm80_dequant(wf, meta, N, K, nw, buf).view(N, K).t()))
            рез = {"форма": имя, "K": K, "N": N, "M": M, "nw": nw, "текущий": [mi, sp],
                   "TF_текущий": round(fl / t_cur / 1e6, 1), "TF_деквант": round(fl / t_q / 1e6, 1), "v2": []}
            лучший = None
            for nj in (8, 4):
                if N % (32 * nj) or N % (8 * nw):
                    continue
                for stg in (3, 4):
                    for pin in (False, True):
                        for spl in (1, 2, 3, 4):
                            try:
                                y2 = e.w8a16_sm80_gemm2(x, wf, meta, N, nw, nj, stg, spl, pin)
                            except RuntimeError as ex:
                                рез["v2"].append({"nj": nj, "st": stg, "pin": pin, "sp": spl, "ошибка": str(ex)[:120]})
                                continue
                            same = bool(torch.equal(y2, y_ref1)) if spl == 1 else None
                            t2 = время(lambda: e.w8a16_sm80_gemm2(x, wf, meta, N, nw, nj, stg, spl, pin))
                            r = {"nj": nj, "st": stg, "pin": pin, "sp": spl, "TF": round(fl / t2 / 1e6, 1)}
                            if same is not None:
                                r["побитно_как_прямой_sp1"] = same
                            else:
                                r["maxabs_к_sp1"] = float((y2.float() - y_ref1.float()).abs().max())
                            рез["v2"].append(r)
                            if лучший is None or r["TF"] > лучший["TF"]:
                                лучший = r
            рез["v2_лучший"] = лучший
            итог.append(рез)
            print(json.dumps({k: v for k, v in рез.items() if k != "v2"}, ensure_ascii=False), flush=True)
    if a.json:
        json.dump(итог, open(a.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
