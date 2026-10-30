#!/usr/bin/env python3
"""[ФРИЗЫ 30.10] Один GEMM W8A16 под ncu: Marlin u8b128 (vllm/_C), наш v3 или v4 (fa2_sm70 из FA2SM70_PATH /
FA2SM70_BUILD_DIR). 5 прогревочных вызовов + 1 (ncu: --launch-skip 5 --launch-count 1 по фильтру ядра).
Аргументы: вид (marlin|d|g3|g4; d -- прямой w8a16_sm80_gemm_large: mi,sp) форма (gate_up|down|qkvz|out) M [конфиг через запятую: g3 -- wm,wn,st,sp; g4 -- wm,wn,nj,st,sp].
"""
import os
import sys

import torch

sys.path.insert(0, os.environ.get("FA2SM70_PATH") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sm70"))
from fa2_sm70._ext import w8a16_sm80_ext  # noqa: E402

ФОРМЫ = {"gate_up": (5120, 17408), "down": (8704, 5120), "qkvz": (5120, 8192), "out": (3072, 5120)}
вид, форма, M = sys.argv[1], sys.argv[2], int(sys.argv[3])
конф = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else []
K, N = ФОРМЫ[форма]
dev = torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(1)
qw = torch.randint(-2**31, 2**31 - 1, (K // 4, N), dtype=torch.int32, device=dev, generator=g)
qz = torch.randint(-2**31, 2**31 - 1, (K // 128, N // 4), dtype=torch.int32, device=dev, generator=g)
sc = (torch.rand(K // 128, N, device=dev, generator=g) * 0.004 + 0.001).half()
x = (torch.randn(M, K, device=dev, generator=g) * 0.5).half()
if вид == "marlin":
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_empty_g_idx, marlin_make_workspace_new, marlin_permute_scales)
    from vllm.scalar_type import scalar_types
    gi = marlin_make_empty_g_idx(dev)
    ws = marlin_make_workspace_new(dev, 4)
    mq = ops.gptq_marlin_repack(qw, gi, K, N, 8)
    ms = marlin_permute_scales(sc, K, N, 128)

    def f():
        return ops.marlin_gemm(x, None, mq, None, ms, None, None, None, gi, gi, ws, scalar_types.uint8b128, M, N, K,
                               is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
else:
    e = w8a16_sm80_ext()
    wf, meta = e.w8a16_sm80_pack(qw, qz, sc, 128, True, 4)
    if вид == "d":
        mi, sp = конф

        def f():
            return e.w8a16_sm80_gemm_large(x, wf, meta, N, 4, mi, sp, 1)
    elif вид == "g3":
        wm, wn, st, sp = конф

        def f():
            return e.w8a16_sm80_gemm3x(x, wf, meta, N, 4, wm, wn, st, sp)
    else:
        wm, wn, nj, st, sp = конф

        def f():
            return e.w8a16_sm80_gemm4x(x, wf, meta, N, 4, wm, wn, nj, st, sp)
for _ in range(6):
    f()
torch.cuda.synchronize()
print("готово", вид, форма, M, конф)
