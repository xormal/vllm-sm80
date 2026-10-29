#!/usr/bin/env python3
"""[W8A8 29.10] Проверка ядра w8a8_sm80 на GPU (окно обслуживания, одна карта).

1) реализация: W8A8 против эталона ТОЙ ЖЕ математики (активации sx*xq по группам 128, вес s(q-z)) в float64 --
   расхождение обязано быть на уровне накопления fp32;
2) ошибка квантования активаций: W8A8 против нынешнего W8A16 (деквант весов в fp16 + GEMM) -- relL2;
3) скорость на формах слоёв модели (TP2): квантование + GEMM W8A8 против декванта + cuBLAS.
"""
import json
import sys
import time

import torch

sys.path.insert(0, "/home/alex/devpack_sm70_20261022/кандидат/vllm-sm70/sm70")
from fa2_sm70._ext import w8a16_sm80_ext, w8a8_sm80_ext  # noqa: E402

torch.manual_seed(29)
dev = torch.device("cuda")
e16, e8 = w8a16_sm80_ext(), w8a8_sm80_ext()
NW = 4


def gptq_weights(N, K):
    qw = torch.randint(0, 256, (K, N), dtype=torch.int32, device=dev)             # k, n
    qwp = (qw[0::4] | (qw[1::4] << 8) | (qw[2::4] << 16) | (qw[3::4] << 24)).contiguous()   # [K/4, N]
    z = torch.randint(100, 156, (K // 128, N), dtype=torch.int32, device=dev)
    qz = (z[:, 0::4] | (z[:, 1::4] << 8) | (z[:, 2::4] << 16) | (z[:, 3::4] << 24)).contiguous()  # [G, N/4]
    sc = (torch.rand(K // 128, N, device=dev) * 2e-3 + 5e-4).half()
    return qw, z, qwp, qz, sc


def timeit(fn, n=20):
    fn(); torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


res = []
for (name, K, N) in [("gate_up", 5120, 17408), ("down", 8704, 5120), ("qkvz", 5120, 8192), ("out_proj", 3072, 5120),
                     ("qkv", 5120, 7168), ("o_proj", 10240, 2560)]:
    qw, z, qwp, qz, sc = gptq_weights(N, K)
    wf, meta = e16.w8a16_sm80_pack(qwp, qz, sc, 128, False, NW)
    # плотный вес для эталона: s (q - z), fp64 -- с той же поправкой нуля, что у упаковки
    zb = (meta.view(torch.int32) >> 16).to(torch.int32)          # fp16-биты (1024+z+zbias)
    zf = (zb.to(torch.int16).view(torch.float16).float() - 1024.0).view(K // 128, N)
    W = (qw.double() - zf.double().repeat_interleave(128, 0)) * sc.double().repeat_interleave(128, 0)   # [K, N]
    for M in (128, 1000, 2048, 8192):
        x = (torch.randn(M, K, device=dev) * 0.5).half()
        x[:, ::97] *= 20.0                                         # каналы-выбросы, как у реальных активаций
        xq, sx, xs = e8.quant_g128(x)
        y8 = e8.w8a8_sm80_gemm(xq, sx, xs, wf, meta, N, NW)
        xd = (xq.double().view(M, K // 128, 128) * sx.double().view(M, K // 128, 1)).view(M, K)
        yref8 = xd @ W                                            # та же математика в fp64
        yref = x.double() @ W                                     # без квантования активаций
        Wh = e16.w8a16_sm80_dequant(wf, meta, N, K, NW, torch.empty(N * K, dtype=torch.float16, device=dev))
        y16 = x @ Wh.view(N, K).t()
        big = torch.full((M + 96, N), float("nan"), dtype=torch.float16, device=dev)   # out= -- срез строк
        e8.w8a8_sm80_gemm(xq, sx, xs, wf, meta, N, NW, big[40:40 + M])
        out_ok = bool(torch.equal(big[40:40 + M], y8)) and bool(big[:40].isnan().all()) and bool(big[40 + M:].isnan().all())
        rel = lambda a, b: float((a.double() - b).norm() / b.norm())
        r = {"слой": name, "M": M, "K": K, "N": N,
             "реализация_relL2": rel(y8, yref8), "w8a8_vs_fp64": rel(y8, yref), "w8a16_vs_fp64": rel(y16, yref),
             "w8a8_vs_w8a16": rel(y8, y16.double()), "out_срез_побитно": out_ok,
             "мс_w8a8": timeit(lambda: e8.w8a8_sm80_gemm(*e8.quant_g128(x), wf, meta, N, NW)),
             "мс_w8a8_только_gemm": timeit(lambda: e8.w8a8_sm80_gemm(xq, sx, xs, wf, meta, N, NW)),
             "мс_деквант_gemm": timeit(lambda: x @ e16.w8a16_sm80_dequant(
                 wf, meta, N, K, NW, Wh).view(N, K).t())}
        r["TOPS_w8a8"] = 2 * M * N * K / r["мс_w8a8_только_gemm"] / 1e9
        res.append(r)
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}, ensure_ascii=False),
              flush=True)
        del x, xq, sx, xs, y8, xd, yref8, yref, y16, big
    del qw, z, qwp, qz, sc, wf, meta, W, Wh
    torch.cuda.empty_cache()
json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"), ensure_ascii=False, indent=1)
