# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
"""W8A8 на больших M (префилл) -- отдельный параметр запуска, раздельно для строк Clef и строк LLM.

FA2SM80_W8A8=off|clef|llm|all (умолчание off -- ни одной лишней команды, модуль даже не грузит ядро):
  clef -- int8-активации только в строках сегментов Clef шага (точность Clef исследуется отдельно: если ответы
          не страдают -- приемлемо), строки LLM -- прежним путём;
  llm  -- только в строках LLM (при падении точности LLM -- выключать);
  all  -- во всех строках большого шага.
FA2SM80_W8A8_FILE=<файл> -- A/B в одном экземпляре: тот же выбор словом из файла. Файл перечитывается по
  СЧЁТЧИКУ вызовов (каждые _ПЕРИОД), а не по времени: оба ранга TP проходят одну и ту же последовательность
  вызовов и переключаются на одном и том же слое одного шага.
FA2SM80_W8A8_MIN=<строк> -- куски строк короче идут прежним путём (плитка ядра 128 строк).
FA2SM80_W8A8_CHECK=<n> -- первые n вызовов на форму сверяются с прежним fp16-путём (relL2 в журнал).

Математика (ядро w8a8_sm80.cu): активации -- int8 по строке и группе 128 (sx = amax/127), веса -- те же
упакованные плитки w8a16_sm80, скобка группы точно в int32, масштаб sx*s -- в fp32. Только вне захвата графов
(большие M всегда без графа); поправка Clef добавляется к выходу как прежде (вход x -- fp16).
"""
import os
import sys

_РЕЖИМЫ = ("off", "clef", "llm", "all")
_РЕЖИМ0 = os.environ.get("FA2SM80_W8A8", "off").strip().lower()
_ФАЙЛ = os.environ.get("FA2SM80_W8A8_FILE", "")
_MIN = int(os.environ.get("FA2SM80_W8A8_MIN", "128"))
_СВЕРКА = int(os.environ.get("FA2SM80_W8A8_CHECK", "0"))
_ПЕРИОД = 64          # ~4 чтения файла на шаг префилла (сотни вызовов на шаг)
_С = {"режим": _РЕЖИМ0 if _РЕЖИМ0 in _РЕЖИМЫ else "off", "n": 0, "w8a8": 0, "строк": 0, "fp16": 0}
_СВЕРЕНО = {}
_расширение = [None]


def _лог(msg):
    print(f"[fa2sm80 W8A8] {msg}", file=sys.stderr, flush=True)


def включён():
    return _С["режим"] != "off" or bool(_ФАЙЛ)


def _расш():
    if _расширение[0] is None:
        import fa2_sm70._ext as E
        _расширение[0] = E.w8a8_sm80_ext()
    return _расширение[0]


def режим():
    """Текущий режим; файл -- по счётчику вызовов (симметрично на рангах TP)."""
    if _ФАЙЛ:
        _С["n"] += 1
        if _С["n"] % _ПЕРИОД == 1:
            try:
                with open(_ФАЙЛ) as f:
                    v = f.read(16).strip().lower()
            except OSError:
                v = _РЕЖИМ0
            v = v if v in _РЕЖИМЫ else "off"
            if v != _С["режим"]:
                _лог(f"режим {_С['режим']} -> {v} (вызов {_С['n']})")
                _С["режим"] = v
    return _С["режим"]


def _сегменты_clef():
    """Строки Clef текущего прохода [(row0, n)] в координатах целого шага; None -- прохода с Clef нет."""
    кл = sys.modules.get("fa2sm70_clef")
    if кл is None or not getattr(кл, "_АКТИВЕН", [False])[0]:
        return None
    segs = getattr(кл, "_СЕГМЕНТЫ", [None])[0]
    return None if segs is None else [(int(r0), int(n)) for r0, n in segs[:, :2]]


def куски(r0, m):
    """Разбиение строк [0, m) вызова на [(a, b, w8a8?)]; r0 -- смещение вызова в шаге (кусок конвейера, окно
    строк Clef). None -- весь вызов прежним путём (режим off, захват графа, нет подходящих строк)."""
    р = режим()
    if р == "off" or m < _MIN:
        return None
    import torch
    if torch.cuda.is_current_stream_capturing():
        return None
    if р == "all":
        return [(0, m, True)]
    from fa2sm70_clef_hook import row_window_offset
    off = row_window_offset()
    g0 = r0 + (off if off is not None else 0)
    segs = _сегменты_clef() or []
    # строки Clef внутри [g0, g0+m) -- в локальных координатах вызова
    клеф = []
    for s0, n in segs:
        a, b = max(s0, g0), min(s0 + n, g0 + m)
        if a < b:
            клеф.append((a - g0, b - g0))
    клеф.sort()
    out, p = [], 0
    for a, b in клеф:
        if a > p:
            out.append((p, a, р == "llm"))
        out.append((a, b, р == "clef"))
        p = b
    if p < m:
        out.append((p, m, р == "llm"))
    # короткие куски W8A8 -- прежним путём; соседние одного пути -- слить
    итог = []
    for a, b, w in out:
        w = w and (b - a) >= _MIN
        if итог and итог[-1][2] == w:
            итог[-1] = (итог[-1][0], b, w)
        else:
            итог.append((a, b, w))
    if not any(w for _, _, w in итог):
        return None
    return итог


def mm(x, wf, meta, N, nw, out=None):
    """y = W8A8(x) для всех строк x [M, K] fp16; out -- [M, N] fp16 (можно срез строк большего тензора)."""
    e = _расш()
    xc = x if x.is_contiguous() else x.contiguous()
    xq, sx, xs = e.quant_g128(xc)
    y = e.w8a8_sm80_gemm(xq, sx, xs, wf, meta, N, nw, out)
    _С["w8a8"] += 1
    _С["строк"] += x.shape[0]
    return y


def смешанный(x, wf, meta, K, N, nw, план, fp16, out=None):
    """Выход [M, N]: куски плана W8A8 -- ядром, остальные -- fp16(x_кусок, out_кусок). fp16 пишет в out."""
    import torch
    M = x.shape[0]
    if out is None:
        out = torch.empty((M, N), dtype=torch.float16, device=x.device)
    for a, b, w in план:
        if w:
            mm(x[a:b], wf, meta, N, nw, out[a:b])
        else:
            _С["fp16"] += 1
            fp16(x[a:b], out[a:b])
    if _СВЕРКА and _СВЕРЕНО.get((K, N), 0) < _СВЕРКА:
        _СВЕРЕНО[(K, N)] = _СВЕРЕНО.get((K, N), 0) + 1
        y0 = torch.empty_like(out)
        fp16(x, y0)
        d = (out.float() - y0.float()).norm() / y0.float().norm().clamp_min(1e-30)
        _лог(f"сверка K={K} N={N} M={M} план={[(a, b, int(w)) for a, b, w in план]}: relL2 к fp16 {float(d):.3e}")
    return out


def счётчики():
    return dict(_С)
