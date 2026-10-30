#!/usr/bin/env python3
"""[ОДИН ГРАФ 30.10] Prefill внимания одним вызовом на слой (paged_varlen_meta + attn_fwd_i8_paged_varlen) -- одна карта.

1) побитно: varlen против attn_fwd_volta_i8_paged на каждую строку (FA2SM80_PF_MINSQ=1 -- то же SM80-ядро при любом Sq),
   с пустыми строками запаса; две геометрии пула (страница 64 и 1568 токенов, K/V чередуются в странице);
2) без хоста: построитель + ядро под torch.cuda.set_sync_debug_mode("error");
3) граф: захват при фиксированных (R, T, ntmax, pgcap), повтор с ДРУГИМИ строками и таблицами -- побитно против eager;
4) время: путь по строкам (как в бою: таблицы страниц torch-операциями + запуск на строку) против varlen, CPU и GPU.
Формы -- слой внимания Qwen3.8-27B на ранге TP2: H=12, Hkv=2, d=256.
"""
import json
import os
import random
import sys
import time

os.environ["FA2SM80_PF_MINSQ"] = "1"          # эталон -- то же SM80-ядро при любом Sq (читается один раз)
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sm70"))
from fa2_sm70._ext import prefill_ext  # noqa: E402

H, HKV, D = 12, 2, 256
dev = torch.device("cuda")
e = prefill_ext()
print("so:", e.__file__, flush=True)


def пул(nblk, bs, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    buf = torch.randint(-127, 128, (nblk, 2, bs, HKV, D), dtype=torch.int8, device=dev, generator=g)
    ks = torch.rand(nblk * bs * HKV, device=dev, generator=g) * 0.02 + 0.005
    vs = torch.rand(nblk * bs * HKV, device=dev, generator=g) * 0.02 + 0.005
    return buf[:, 0], buf[:, 1], ks, vs


def строки(lens, nblk, bs, maxblk, seed):
    """lens: [(Sq, Sk)] -> Q [T, H, D], таблица блоков [R, maxblk] (разные страницы у строк)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    T = sum(sq for sq, _ in lens)
    Q = (torch.randn(max(T, 1), H, D, device=dev, generator=g) * 0.5).half()
    perm = torch.randperm(nblk, generator=torch.Generator().manual_seed(seed))
    bt = torch.zeros(len(lens), maxblk, dtype=torch.int32)
    k = 0
    for i, (_, sk) in enumerate(lens):
        n = (sk + bs - 1) // bs
        bt[i, :n] = perm[k:k + n].to(torch.int32)
        k += n
    return Q, bt.to(dev)


def эталон(Q, bt, lens, K, V, ks, vs, scale):
    out = torch.empty_like(Q)
    t = 0
    for i, (sq, sk) in enumerate(lens):
        if sq == 0:
            continue
        o, _ = e.attn_fwd_volta_i8_paged(Q[t:t + sq].unsqueeze(0).contiguous(), K, V, ks, vs,
                                         bt[i].contiguous(), sk, scale, True)
        out[t:t + sq] = o[0]
        t += sq
    return out


def varlen(Q, bt, lens, K, V, ks, vs, scale, bs, ntmax, pgcap, R=None):
    R = R or len(lens)
    sq = [l[0] for l in lens] + [0] * (R - len(lens))
    sk = [l[1] for l in lens] + [0] * (R - len(lens))
    # как в бою: обычная память + non_blocking (копия забирает данные до возврата, синхронизации нет)
    мета = torch.tensor([[0] + list(torch.tensor(sq).cumsum(0).tolist())[:-1], sq, sk,
                         [min(i, len(lens) - 1) for i in range(R)]], dtype=torch.int32).to(dev, non_blocking=True)
    q0, sqt, skt, ridx = мета[0], мета[1], мета[2], мета[3]
    tiles, vo, vsc, err = e.paged_varlen_meta(q0, sqt, skt, ridx, bt, bs, K.stride(0), HKV, ntmax, pgcap)
    out = torch.empty_like(Q)
    e.attn_fwd_i8_paged_varlen(Q, K, V, ks, vs, tiles, vo, vsc, scale, True, out)
    return out, err, (q0, sqt, skt, ridx)


def ntmax_for(T, R):
    return (T + 127) // 128 + R


def main():
    рез = {"побитно": [], "граф": [], "время": []}
    random.seed(30)
    scale = D ** -0.5
    for bs in (64, 1568):
        nblk = max(64, (8 * 40000) // bs + 8)
        K, V, ks, vs = пул(nblk, bs, 7 + bs)
        maxblk = (40000 + bs - 1) // bs
        наборы = [[(400, 400)], [(400, 9000)], [(2048, 2048)], [(37, 3000)], [(5, 5), (400, 1200), (130, 34816)],
                  [(random.randint(1, 600), 0) for _ in range(6)], [(1000, 33000), (17, 17), (2048, 6000)]]
        for j, lens in enumerate(наборы):
            lens = [(sq, max(sk, sq)) for sq, sk in lens]
            Q, bt = строки(lens, nblk, bs, maxblk, 100 * bs + j)
            T = sum(sq for sq, _ in lens)
            R = len(lens) + 2                                        # две пустые строки запаса
            ntm = ntmax_for(T, R)
            pgc = sum((sk + 15) // 16 for _, sk in lens) + 64
            ref = эталон(Q, bt, lens, K, V, ks, vs, scale)
            out, err, _ = varlen(Q, bt, lens, K, V, ks, vs, scale, bs, ntm, pgc, R=R)
            r = {"bs": bs, "строки": lens, "побитно": bool(torch.equal(out[:T], ref[:T])), "err": int(err.item()),
                 "maxabs": float((out[:T].float() - ref[:T].float()).abs().max())}
            рез["побитно"].append(r)
            print("побитно", json.dumps(r, ensure_ascii=False), flush=True)
    # 2) без хоста
    bs = 1568
    nblk = (8 * 40000) // bs + 8
    K, V, ks, vs = пул(nblk, bs, 99)
    maxblk = (40000 + bs - 1) // bs
    lens = [(400, 9000), (5, 5), (1000, 2000)]
    Q, bt = строки(lens, nblk, bs, maxblk, 5)
    varlen(Q, bt, lens, K, V, ks, vs, scale, bs, 64, 4096, R=4)        # прогрев
    torch.cuda.synchronize()
    try:
        torch.cuda.set_sync_debug_mode("error")
        varlen(Q, bt, lens, K, V, ks, vs, scale, bs, 64, 4096, R=4)
        рез["без_синхронизаций"] = True
    except RuntimeError as ex:
        рез["без_синхронизаций"] = False
        рез["синхронизация"] = str(ex)[:300]
    finally:
        torch.cuda.set_sync_debug_mode("default")
    print("без синхронизаций:", рез["без_синхронизаций"], рез.get("синхронизация", ""), flush=True)
    # 3) граф: R=6, T=3000, ntmax и pgcap фиксированы; статические входы
    R, T, NTM, PGC = 6, 3000, ntmax_for(3000, 6), 6 * ((40000 + 15) // 16)
    q0_s = torch.zeros(R, dtype=torch.int32, device=dev)
    sq_s = torch.zeros(R, dtype=torch.int32, device=dev)
    sk_s = torch.zeros(R, dtype=torch.int32, device=dev)
    ri_s = torch.arange(R, dtype=torch.int32, device=dev)
    bt_s = torch.zeros(R, maxblk, dtype=torch.int32, device=dev)
    Q_s = torch.zeros(T, H, D, dtype=torch.float16, device=dev)
    out_s = torch.zeros_like(Q_s)

    def тело():
        tiles, vo, vsc, err = e.paged_varlen_meta(q0_s, sq_s, sk_s, ri_s, bt_s, bs, K.stride(0), HKV, NTM, PGC)
        e.attn_fwd_i8_paged_varlen(Q_s, K, V, ks, vs, tiles, vo, vsc, scale, True, out_s)

    def поставить(lens, seed):
        Q, bt = строки(lens, nblk, bs, maxblk, seed)
        sq = [l[0] for l in lens] + [0] * (R - len(lens))
        sk = [l[1] for l in lens] + [0] * (R - len(lens))
        q0_s.copy_(torch.tensor([0] + list(torch.tensor(sq).cumsum(0).tolist())[:-1], dtype=torch.int32))
        sq_s.copy_(torch.tensor(sq, dtype=torch.int32))
        sk_s.copy_(torch.tensor(sk, dtype=torch.int32))
        bt_s.zero_()
        bt_s[:len(lens)] = bt
        Q_s.zero_()
        Q_s[:sum(sq)] = Q[:sum(sq)]
        return Q, bt

    поставить([(1000, 2000), (400, 400)], 1)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            тело()
    torch.cuda.current_stream().wait_stream(s)
    граф = torch.cuda.CUDAGraph()
    with torch.cuda.graph(граф):
        тело()
    for j, lens in enumerate([[(1000, 2000), (400, 400)], [(5, 5), (400, 9000), (130, 34816), (2048, 2048)],
                              [(2990, 2990)], [(500, 600), (500, 700), (500, 800), (500, 900), (500, 1000), (5, 30000)]]):
        lens = [(sq, max(sk, sq)) for sq, sk in lens]
        assert sum(l[0] for l in lens) <= T and len(lens) <= R
        Q, bt = поставить(lens, 200 + j)
        граф.replay()
        ref = эталон(Q, bt, lens, K, V, ks, vs, scale)
        n = sum(l[0] for l in lens)
        r = {"строки": lens, "побитно": bool(torch.equal(out_s[:n], ref[:n]))}
        рез["граф"].append(r)
        print("граф", json.dumps(r, ensure_ascii=False), flush=True)
    # 4) время: 9 строк (8 спекулятивных по 5 токенов + запись 400) и 1 строка 2048
    for lens in ([(5, 3000)] * 8 + [(400, 400)], [(2048, 6000)], [(400, 400)]):
        Q, bt = строки(lens, nblk, bs, maxblk, 9)
        T = sum(l[0] for l in lens)
        for путь in ("строки", "varlen"):
            fn = (lambda: эталон(Q, bt, lens, K, V, ks, vs, scale)) if путь == "строки" else \
                 (lambda: varlen(Q, bt, lens, K, V, ks, vs, scale, bs, ntmax_for(T, len(lens)), 4096))
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            n = 30
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            t_cpu = (time.perf_counter() - t0) / n * 1e3
            torch.cuda.synchronize()
            t_all = (time.perf_counter() - t0) / n * 1e3
            r = {"строки": f"{len(lens)} шт., T={T}", "путь": путь, "мс_CPU": round(t_cpu, 3), "мс_до_конца_GPU": round(t_all, 3)}
            рез["время"].append(r)
            print("время", json.dumps(r, ensure_ascii=False), flush=True)
    if len(sys.argv) > 1:
        json.dump(рез, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
