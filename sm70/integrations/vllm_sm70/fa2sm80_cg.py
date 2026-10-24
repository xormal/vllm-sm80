# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
"""Свой fp16 GEMM префилла (cg_sm80.cu, CUTLASS sm80 без slice-K): y = x @ w^T, ИНВАРИАНТНЫЙ К ПАЧКЕ.

FA2SM80_CG=1 -- вместо torch.mm (cuBLAS) в больших M: распаковка W8 + GEMM (fa2sm80_w8), куски конвейера строк и
конвейера блока. cuBLAS выбирает разные алгоритмы для разных M (куски разной длины, разный состав пачки) и даёт разные
числа для одной и той же строки; здесь все конфигурации считают элемент в одном порядке -- строка побитово та же при
любом M и любом месте в куске. Поэтому конфигурация подбирается ПО ВРЕМЕНИ (на корзину M, один раз) -- на числа выбор
не влияет. Во время захвата CUDA-графа подбор не идёт (конфигурация 2 -- 128x128x32).
Таблица выбора -- ~/.cache/vllm-sm80/cgtune.json (bench/tune_cg.py вне сервера: формы W8 x корзины M 512..8192);
подбор в работе -- только для незнакомых (корзина, N, K), результат дописывается в тот же файл. Подбор в работе
(синхронизации на каждую новую корзину) тормозил Clef: запросы разной длины -> много корзин.
"""
import os
import sys

_ВКЛ = os.environ.get("FA2SM80_CG", "0") == "1"
_ВЫБОР = {}          # (корзина M, N, K) -> конфигурация
_ТАБЛИЦА = os.path.expanduser(os.environ.get("FA2SM80_CG_TABLE", "~/.cache/vllm-sm80/cgtune.json"))
_ЗАГРУЖЕНО = [False]
_ext = None
_С = {"вызовов": 0, "подборов": 0}


def включён():
    return _ВКЛ


def _лог(msg):
    print(f"[fa2sm80 CG] {msg}", file=sys.stderr, flush=True)


def _расш():
    global _ext
    if _ext is None:
        import fa2_sm70._ext as E
        _ext = E.cg_sm80_ext()
    return _ext


def _корзина(M):
    return min((M + 511) // 512 * 512, 8192) if M <= 8192 else (M + 4095) // 4096 * 4096


def _загрузить():
    _ЗАГРУЖЕНО[0] = True
    try:
        import json
        for k, v in json.load(open(_ТАБЛИЦА)).items():
            m, n, kk = (int(t) for t in k.split(","))
            _ВЫБОР[(m, n, kk)] = int(v)
        _лог(f"таблица конфигураций: {len(_ВЫБОР)} записей ({_ТАБЛИЦА})")
    except FileNotFoundError:
        _лог(f"таблицы {_ТАБЛИЦА} нет -- подбор в работе")
    except Exception as ex:  # noqa: BLE001
        _лог(f"таблица {_ТАБЛИЦА} не прочитана: {ex}")


def _дописать(ключ, c):
    try:
        import json
        таб = {}
        if os.path.exists(_ТАБЛИЦА):
            таб = json.load(open(_ТАБЛИЦА))
        таб[f"{ключ[0]},{ключ[1]},{ключ[2]}"] = c
        os.makedirs(os.path.dirname(_ТАБЛИЦА), exist_ok=True)
        tmp = f"{_ТАБЛИЦА}.{os.getpid()}"
        json.dump(таб, open(tmp, "w"))
        os.replace(tmp, _ТАБЛИЦА)
    except Exception:  # noqa: BLE001
        pass


def _подобрать(x, w, ключ):
    import torch
    e = _расш()
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    лучш, tbest = 2, None
    времена = []
    for c in range(e.NCFG):
        try:
            e.cg_mm(x, w, y, c)
        except RuntimeError:
            времена.append(None)
            continue
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(3):
            e.cg_mm(x, w, y, c)
        b.record()
        b.synchronize()
        t = a.elapsed_time(b) / 3
        времена.append(t)
        if tbest is None or t < tbest:
            лучш, tbest = c, t
    _ВЫБОР[ключ] = лучш
    _С["подборов"] += 1
    _дописать(ключ, лучш)
    fl = 2.0 * M * N * K
    _лог(f"подбор M={M} (корзина {ключ[0]}) N={N} K={K}: конфигурация {лучш}, {fl / tbest / 1e9:.0f} TF; "
         + " ".join("-" if t is None else f"{fl / t / 1e9:.0f}" for t in времена))
    return лучш


def mm(x, w, out=None):
    """y = x @ w^T; x [M,K], w [N,K] -- fp16, плотные по строкам; out -- [M,N] (можно срез строк большего тензора)."""
    import torch
    e = _расш()
    M, K = x.shape
    N = w.shape[0]
    if not _ЗАГРУЖЕНО[0]:
        _загрузить()
    ключ = (_корзина(M), N, K)
    c = _ВЫБОР.get(ключ)
    if c is None:
        c = 2 if torch.cuda.is_current_stream_capturing() else _подобрать(x, w, ключ)
    _С["вызовов"] += 1
    return e.cg_mm(x, w, out, c)
