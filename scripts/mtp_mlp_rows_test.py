"""[ПАМЯТЬ 31.10] Стенд MLP головы MTP по кускам (FA2SM80_MTP_MLP_ROWS, fa2sm70_mtp_w8.py), одна карта.

1) Нарезка: куски покрывают [0, M) подряд, каждый длиннее 512 строк (маршрут gate_up тот же, что у целого M).
2) SiLU: Triton-ядро головы против того, что inductor сшивает из SiluAndMul.forward_native (так шёл прежний проход
   черновика) -- побитно, на формах декода и префилла и на разных масштабах входа.
3) Пик памяти gate_up + SiLU при M=8192: целым проходом против кусков (torch.mm вместо W8 -- те же размеры буферов).

    CUDA_VISIBLE_DEVICES=0 python scripts/mtp_mlp_rows_test.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sm70", "integrations", "vllm_sm70"))
import fa2sm70_mtp_w8 as T  # noqa: E402

I = 8704
плохо = 0

for ряды in (1024, 2048):
    for M in range(2048, 20000, 7):
        к = T._куски_mlp(M, ряды)
        ok = к[0][0] == 0 and к[-1][1] == M and all(a[1] == b[0] for a, b in zip(к, к[1:])) \
            and all(r1 - r0 > 512 for r0, r1 in к)
        if not ok:
            плохо += 1
            print(f"нарезка: M={M} ряды={ряды}: {к[:3]}..{к[-2:]}")
print(f"нарезка: {'ok' if not плохо else 'ПЛОХО'}")

ref = torch.compile(lambda x: F.silu(x[..., :I]) * x[..., I:], dynamic=True)
силу = T._силу()
torch.manual_seed(0)
for M in (1, 5, 9, 45, 1024, 2731, 8192):
    for масштаб in (0.5, 4.0, 30.0):
        g = (torch.randn(M, 2 * I, device="cuda") * масштаб).half()
        a = torch.empty(M, I, dtype=torch.float16, device="cuda")
        силу(g, a)
        r = ref(g)
        разн = int((a.view(torch.int16) != r.view(torch.int16)).sum())
        плохо += разн != 0
        if разн:
            print(f"SiLU M={M} x{масштаб}: различий {разн} из {a.numel()}")
print(f"SiLU против inductor: {'побитно' if not плохо else 'РАЗЛИЧИЯ'}")

K, N, M = 5120, 2 * I, 8192
x = torch.randn(M, K, device="cuda").half()
W = (torch.randn(N, K, device="cuda") * 0.02).half()


def пик(fn):
    torch.cuda.synchronize()
    база = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    return out, (torch.cuda.max_memory_allocated() - база) / 2 ** 20


def целиком():
    return ref(torch.mm(x, W.t()))


def кусками():
    a = torch.empty(M, I, dtype=torch.float16, device="cuda")
    for r0, r1 in T._куски_mlp(M, 1024):
        силу(torch.mm(x[r0:r1], W.t()), a[r0:r1])
    return a


целиком()
a0, п0 = пик(целиком)
a1, п1 = пик(кусками)
разн = int((a0.view(torch.int16) != a1.view(torch.int16)).sum())
print(f"пик gate_up + SiLU при M={M}: целиком {п0:.0f} МиБ, кусками {п1:.0f} МиБ (выход отличается в {разн} -- "
      f"torch.mm, не свой GEMM: справочно)")
sys.exit(1 if плохо else 0)
