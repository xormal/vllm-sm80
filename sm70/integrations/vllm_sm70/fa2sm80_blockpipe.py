# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
"""Префилл: КОНВЕЙЕР ОБМЕНА ЧЕРЕЗ ВЕСЬ MLP-БЛОК (FA2SM80_BLOCKPIPE=1).

Профиль префилла 8K (24.10, боевая связка с конвейером строк и копировщиком): окно 2097 мс, вычислитель простаивает
710 мс -- ждёт обмен. Конвейер строк (fa2sm80_rowpipe) прячет сведение куска только за умножением ТОЙ ЖЕ операции, а
у o_proj/out_proj обмен 8192 строк идёт 6.1 мс против 1.3 мс умножения, у down -- 6.1 против 3.6. Всё, что идёт
после сведения до следующего смешивания токенов, построчно: add+RMSNorm -> gate_up -> SiluAndMul -> down. Поэтому
одна операция на весь отрезок

    out_proj/o_proj -> сведение -> add + GemmaRMSNorm -> gate_up -> SiluAndMul -> down -> сведение

режет строки на куски: сведение куска i идёт на побочном потоке высокого приоритета, пока основной поток считает
MLP куска i-1 (и умножения out_proj следующих кусков). Обмен на слой 12.2 мс против ~12.5 мс счёта отрезка --
почти целиком под счётом; открытыми остаются начало (первое сведение out_proj) и конец (последнее сведение down).

ДЕКОД И МАЛЫЕ M (M < FA2SM80_BLOCKPIPE_MIN) -- те же ядра подряд на текущем потоке: вызовы слоёв как в исходном
проходе. Ветвление по M -- внутри операции (dynamo видит одну непрозрачную операцию для всех M).

Смеситель (GDN / полное внимание) с флагом _fa2_defer_proj возвращает выход ДО выходной проекции (правка в
qwen3_5.py / qwen3_next.py), проход слоя заменяется: input_layernorm -> смеситель -> эта операция.
Остаток (residual) -- fp32, как у исходного GemmaRMSNorm для fp16 (x.float() + residual.float()).
Clef: дельта по окну строк каждого куска через ХУК_ДИАП (clef.cu delta_keyed_range), как в конвейере строк.
"""
import os
import sys

_ВКЛ = os.environ.get("FA2SM80_BLOCKPIPE", "0") == "1"
_SILU = os.environ.get("FA2SM80_SILU", "0") == "1"
_SILU_СВЕРКА = [int(os.environ.get("FA2SM80_SILU_CHECK", "0"))]
# Через границу слоя (FA2SM80_BLOCKPIPE_X=1): операция слоя L отдаёт ГОТОВЫЕ входные проекции слоя L+1 (его
# input_layernorm + in_proj по кускам, как только сведено down куска) -- хвост сведений прячется за in_proj.
_ВКЛ_X = os.environ.get("FA2SM80_BLOCKPIPE_X", "0") == "1"
_УЗКО = 256          # проекции уже этого (in_proj_ba, N=48) -- torch.mm: плитка своего GEMM почти пуста
_MIN = int(os.environ.get("FA2SM80_BLOCKPIPE_MIN", "2048"))
_ROWS = int(os.environ.get("FA2SM80_BLOCKPIPE_ROWS", "1024"))
# Opt-in until GPU parity/throughput gates: preserves scheduler and mixer batches.
_FALLBACK_ROWS = int(os.environ.get("FA2SM80_BLOCKPIPE_FALLBACK_ROWS", "0"))
if _FALLBACK_ROWS < 0:
    raise ValueError("FA2SM80_BLOCKPIPE_FALLBACK_ROWS must be nonnegative")
# План кусков (FA2SM80_BLOCKPIPE_PLAN="голова,середина,хвост1,хвост2,..."): маленький первый кусок -- раньше первое
# сведение out_proj, крупные в середине -- GEMM на 2048 строк эффективнее (на 1024 строках ~184 TF против ~205),
# убывающий хвост -- короче последнее сведение down, которое ждёт следующий слой. Пусто -- равные куски по ROWS.
_PLAN = [int(v) for v in os.environ.get("FA2SM80_BLOCKPIPE_PLAN", "").split(",") if v.strip()]


def _куски(M):
    if len(_PLAN) < 3 or M < sum(_PLAN[:1] + _PLAN[2:]) + _PLAN[1]:
        return [(r0, min(M, r0 + _ROWS)) for r0 in range(0, M, _ROWS)]
    голова, середина, хвост = _PLAN[0], _PLAN[1], _PLAN[2:]
    размеры = [голова]
    ост = M - голова - sum(хвост)
    while ост > 0:
        размеры.append(min(середина, ост))
        ост -= размеры[-1]
    размеры += хвост
    out, r0 = [], 0
    for n in размеры:
        out.append((r0, r0 + n))
        r0 += n
    return out
_РЕЕСТР = {}          # ключ (prefix слоя) -> слой декодера
_ПОБОЧНЫЙ = {}
_БУФ_B = {}           # деквант-буфер out_proj (gate_up держит общий буфер fa2sm80_w8)
_БУФ_C = {}           # деквант-буфер down (распаковывается на третьем потоке, пока идут умножения out_proj)
_ДЕКВАНТ_ПОТОК = {}
# out_proj на столько кусков вперёд. 99 (все сразу) -- лучше: чередование out/down (2) заставляло основной поток ждать
# сведения out_proj перед нормой (профиль 8K: простой 127 -> 197 мс). Оставлено переключаемым.
_ВПЕРЁД = int(os.environ.get("FA2SM80_BLOCKPIPE_AHEAD", "99"))
_ПРИОР_ДКВ = int(os.environ.get("FA2SM80_BLOCKPIPE_DQ_PRIO", "0"))      # приоритет потока распаковки (-1 -- высокий)
# [25.10] РАСПАКОВКА ВПЕРЁД (FA2SM80_BLOCKPIPE_DQ_AHEAD=1): в конце операции слоя L, после последнего down-GEMM, на
# потоке распаковки ставятся down и out_proj слоя L+1 в их же буферы (C и B) -- распаковка ложится на хвост сведений
# down слоя L и на смеситель слоя L+1 (скан GDN латентен, SM свободны), а не конкурирует с out_proj/MLP. Запас годен
# ТОЛЬКО следующему по счёту вызову и только своему слою (иначе -- обычная распаковка); веса те же -> побитово.
_DQ_ВПЕРЁД = os.environ.get("FA2SM80_BLOCKPIPE_DQ_AHEAD", "0") == "1"
_ЗАПАС = {}           # dev.index -> (номер вызова, ключ слоя, Wo, Wd, событие)
_С = {"конвейер": 0, "прямо": 0, "clef": 0}


def _лог(msg):
    print(f"[fa2sm80 BLOCKPIPE] {msg}", file=sys.stderr, flush=True)


_норма_ядро = None


def _норма():
    """Triton: y (fp16) + residual (fp16|fp32) -> residual' (fp32), h = rmsnorm(residual') * (1 + w) -> fp16.
    Как GemmaRMSNorm._forward_static_with_residual для fp16."""
    global _норма_ядро
    if _норма_ядро is not None:
        return _норма_ядро
    import triton
    import triton.language as tl

    @triton.jit
    def _k(y_ptr, r_ptr, w_ptr, h_ptr, ro_ptr, N, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        m = offs < N
        y = tl.load(y_ptr + row * N + offs, mask=m, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + row * N + offs, mask=m, other=0.0).to(tl.float32)
        x = y + r
        tl.store(ro_ptr + row * N + offs, x, mask=m)
        var = tl.sum(x * x, axis=0) / N
        xn = x * tl.rsqrt(var + eps)
        w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(h_ptr + row * N + offs, (xn * (1.0 + w)).to(tl.float16), mask=m)

    def норма(y, r, w, eps, h, ro):
        n = y.shape[-1]
        rows = y.shape[0]
        if rows:
            _k[(rows,)](y, r, w, h, ro, n, eps, BLOCK=triton.next_power_of_2(n), num_warps=8)

    _норма_ядро = норма
    return норма


def установить():
    if not _ВКЛ:
        return False
    import torch
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.utils.torch_utils import direct_register_custom_op
    from vllm.model_executor.models import qwen3_next as QN

    def _clef_в_шаге():
        кл = sys.modules.get("fa2sm70_clef")
        return кл is not None and bool(getattr(кл, "_АКТИВЕН", [False])[0])

    def _w8(layer):
        return getattr(layer, "fa2sm80_kn", None) is not None and getattr(layer, "bias", None) is None

    def _деквант(layer, buf):
        import fa2sm80_w8 as W8
        K, N = layer.fa2sm80_kn
        т = W8._ТЮН[(K, N)]
        return W8._расш().w8a16_sm80_dequant(layer.fa2sm80_wf, layer.fa2sm80_meta, N, K, т["nw"], buf)

    def _буфер_b(dev, n):
        b = _БУФ_B.get(dev)
        if b is None or b.numel() < n:
            _БУФ_B[dev] = b = torch.empty(n, dtype=torch.float16, device=dev)
        return b

    def _буфер_c(dev, n):
        b = _БУФ_C.get(dev)
        if b is None or b.numel() < n:
            _БУФ_C[dev] = b = torch.empty(n, dtype=torch.float16, device=dev)
        return b

    def _прямой_кусок(L, core, residual):
        """Малые M (декод, графы) и всё, что конвейер не берёт: исходная последовательность слоёв."""
        _С["прямо"] += 1
        proj = L.linear_attn.out_proj if L.layer_type == "linear_attention" else L.self_attn.o_proj
        y = proj(core)[0]
        M, H = y.shape
        h = torch.empty((M, H), dtype=torch.float16, device=y.device)
        ro = torch.empty((M, H), dtype=torch.float32, device=y.device)
        n = L.post_attention_layernorm
        _норма()(y, residual, n.weight.data, n.variance_epsilon, h, ro)
        # Все потребители y поставлены в тот же поток. Не держать его до down/TP.
        # residual/core и ro остаются живыми; входы не переиспользуются in-place.
        del y
        gu = L.mlp.gate_up_proj(h)[0]
        del h
        a = torch.empty((M, gu.shape[1] // 2), dtype=gu.dtype, device=gu.device)
        if _SILU and M <= 64 and gu.is_contiguous():
            # [25.10] широкая сетка вместо M блоков vLLM; побитно (сверка FA2SM80_SILU_CHECK)
            import fa2sm80_cg as CG
            CG._расш().silu_mul_f16(gu, a)
            if _SILU_СВЕРКА[0] > 0 and not torch.cuda.is_current_stream_capturing():
                _SILU_СВЕРКА[0] -= 1
                a0 = torch.empty_like(a)
                torch.ops._C.silu_and_mul(a0, gu)
                _лог(f"SILU сверка M={M} d={a.shape[1]}: различий {int((a.view(torch.int16) != a0.view(torch.int16)).sum())}")
                del a0
        else:
            torch.ops._C.silu_and_mul(a, gu)
        del gu
        d = L.mlp.down_proj(a)[0]
        return d, ro

    def _прямо(L, core, residual):
        if (_FALLBACK_ROWS and core.dim() == 2 and core.shape[0] >= _MIN
                and core.shape[0] > _FALLBACK_ROWS):
            import fa2sm70_clef_hook as Hk
            # Old extensions without range corrections must retain the original
            # path; do not silently apply global Clef segments to local rows.
            if not _clef_в_шаге() or Hk.ХУК_ДИАП[0] is not None:
                from fa2sm80_fallback import chunked_fallback
                _С["fallback_chunks"] = _С.get("fallback_chunks", 0) + 1
                return chunked_fallback(core, residual, _FALLBACK_ROWS,
                                        lambda x, r: _прямой_кусок(L, x, r))
        return _прямой_кусок(L, core, residual)

    def _проекции_след(N2):
        """Входные проекции следующего слоя: [(слой, w8?)] -- GDN: in_proj_qkvz (W8) и in_proj_ba (fp16); внимание: qkv_proj."""
        if N2.layer_type == "linear_attention":
            return [N2.linear_attn.in_proj_qkvz, N2.linear_attn.in_proj_ba]
        return [N2.self_attn.qkv_proj]

    def _след_прямо(N2, out, ro):
        """Следующий слой подряд: input_layernorm + входные проекции вызовами модулей (поправки Clef -- как обычно)."""
        M, H = out.shape
        h2 = torch.empty((M, H), dtype=torch.float16, device=out.device)
        ro2 = torch.empty((M, H), dtype=torch.float32, device=out.device)
        n2 = N2.input_layernorm
        _норма()(out, ro, n2.weight.data, n2.variance_epsilon, h2, ro2)
        return tuple(p(h2)[0] for p in _проекции_след(N2)) + (ro2,)

    def _тело(core, residual, key, next_key=None):
        L = _РЕЕСТР[key]
        N2 = _РЕЕСТР[next_key] if next_key else None
        proj = L.linear_attn.out_proj if L.layer_type == "linear_attention" else L.self_attn.o_proj
        gup, dnp = L.mlp.gate_up_proj, L.mlp.down_proj
        M = core.shape[0]
        if M < _MIN or core.dim() != 2 or not (_w8(proj) and _w8(gup) and _w8(dnp)):
            out, ro = _прямо(L, core, residual)
            return (out, ro) if N2 is None else _след_прямо(N2, out, ro)
        хук = None
        if _clef_в_шаге():
            import fa2sm70_clef_hook as Hk
            хук = Hk.ХУК_ДИАП[0]
            if хук is None:
                out, ro = _прямо(L, core, residual)
                return (out, ro) if N2 is None else _след_прямо(N2, out, ro)
            _С["clef"] += 1
        # Следующий слой по кускам -- только без Clef (поправки входных проекций fp16 идут через модули) и если вход
        # GDN -- W8, а у внимания -- плотный fp16 без смещения. Иначе -- конвейер блока, затем следующий слой подряд.
        прос = _проекции_след(N2) if N2 is not None else []
        след_куски = (N2 is not None and хук is None
                      and all((_w8(p) if getattr(p, "fa2sm80_kn", None) is not None else
                               (getattr(p, "bias", None) is None and getattr(p, "weight", None) is not None
                                and p.weight.dtype == torch.float16 and p.weight.dim() == 2)) for p in прос))
        _С["конвейер"] += 1
        if _С["конвейер"] in (1, 1000, 100000) or (хук is not None and _С["clef"] == 1):
            _лог(f"конвейер блока работает: {key}, M={M}, кусок {_ROWS} строк (вызовов {_С['конвейер']}, "
                 f"с Clef {_С['clef']}, прямо {_С['прямо']})")
        import fa2sm80_w8 as W8
        dev = core.device
        Ko, No = proj.fa2sm80_kn
        Kg, Ng = gup.fa2sm80_kn
        Kd, Nd = dnp.fa2sm80_kn
        nw = L.post_attention_layernorm
        норма = _норма()
        осн = torch.cuda.current_stream()
        поб = _ПОБОЧНЫЙ.get(dev.index)
        if поб is None:
            поб = _ПОБОЧНЫЙ[dev.index] = torch.cuda.Stream(device=dev, priority=-1)
        out = torch.empty((M, Nd), dtype=torch.float16, device=dev) if not след_куски else None
        ro = torch.empty((M, No), dtype=torch.float32, device=dev)
        куски = _куски(M)
        if след_куски:
            ro2 = torch.empty((M, No), dtype=torch.float32, device=dev)
            выходы2 = [torch.empty((M, p.fa2sm80_kn[1] if getattr(p, "fa2sm80_kn", None) is not None else p.weight.shape[0]),
                                   dtype=torch.float16, device=dev) for p in прос]
            n2 = N2.input_layernorm
        # 1) out_proj по кускам, сведение каждого -- на побочном потоке
        import fa2sm80_cg as CG
        умн = CG.mm if CG.включён() else (lambda x_, w_: torch.mm(x_, w_.t()))
        # Распаковка весов MLP -- на третьем потоке с самого начала: память (распаковка) параллельно тензорным умножениям
        # out_proj. Ждёт всё, что было на основном потоке до операции (буфер A мог читать предыдущий слой).
        дкв = _ДЕКВАНТ_ПОТОК.get(dev.index)
        if дкв is None:
            дкв = _ДЕКВАНТ_ПОТОК[dev.index] = torch.cuda.Stream(device=dev, priority=_ПРИОР_ДКВ)
        _С["вызов"] = _С.get("вызов", 0) + 1
        зп = _ЗАПАС.pop(dev.index, None)
        if зп is not None and not (зп[0] == _С["вызов"] - 1 and зп[1] == key and not след_куски):
            зп = None
        дкв.wait_stream(осн)
        with torch.cuda.stream(дкв):
            Wg = _деквант(gup, W8._буфер_для(dev, Kg * Ng))
            Wd = зп[3] if зп is not None else _деквант(dnp, _буфер_c(dev, Kd * Nd))
            ev_w = torch.cuda.Event()
            ev_w.record(дкв)
        # буфер B -- сразу под наибольшее (out_proj, затем вход W8 следующего слоя): без перевыделения под живым тензором
        кнB = [Ko * No] + [p.fa2sm80_kn[0] * p.fa2sm80_kn[1] for p in прос if след_куски and getattr(p, "fa2sm80_kn", None)]
        if зп is not None:
            осн.wait_event(зп[4])
            Wo = зп[2]
            _С["запас"] = _С.get("запас", 0) + 1
            if _С["запас"] in (1, 1000):
                _лог(f"распаковка вперёд работает: {key} (вызовов с запасом {_С['запас']})")
        else:
            Wo = _деквант(proj, _буфер_b(dev, max(кнB)))
        поб.wait_stream(осн)
        сведённые, готово = {}, {}

        def _out(i):
            r0, r1 = куски[i]
            y = умн(core[r0:r1], Wo)
            if хук is not None:
                хук(core[r0:r1], y, [proj.scales], r0)
            ev = torch.cuda.Event()
            ev.record(осн)
            with torch.cuda.stream(поб):
                поб.wait_event(ev)
                yr = tensor_model_parallel_all_reduce(y)
                evr = torch.cuda.Event()
                evr.record(поб)
            y.record_stream(поб)
            yr.record_stream(осн)
            сведённые[i], готово[i] = yr, evr

        # 1) out_proj на _ВПЕРЁД кусков вперёд; дальше -- по одному куску перед MLP очередного: сведения на побочном
        #    потоке чередуются out/down и не копятся к концу слоя.
        # при переходе через границу -- все out_proj вперёд: буфер B потом занимает вход следующего слоя
        вперёд = len(куски) if след_куски else _ВПЕРЁД
        for i in range(min(вперёд, len(куски))):
            _out(i)
        # 2) веса MLP распакованы на третьем потоке -- дождаться
        осн.wait_event(ev_w)
        if след_куски:
            # вход W8 следующего слоя -- в буфер B после умножений out_proj (порядок основного потока), на третьем потоке
            ev_o = torch.cuda.Event()
            ev_o.record(осн)
            Wn = []
            дкв.wait_event(ev_o)
            with torch.cuda.stream(дкв):
                for p in прос:
                    Wn.append(_деквант(p, _буфер_b(dev, max(кнB))) if getattr(p, "fa2sm80_kn", None) is not None
                              else p.weight)
                ev_w2 = torch.cuda.Event()
                ev_w2.record(дкв)
            ждём_w2 = [True]
            сведённые_d = {}

        def _след(j):
            # кусок j следующего слоя: дождаться сведения down куска j -> input_layernorm L+1 -> входные проекции
            r0_, r1_ = куски[j]
            dr_, evd_ = сведённые_d.pop(j)
            осн.wait_event(evd_)
            if ждём_w2[0]:
                осн.wait_event(ev_w2)
                ждём_w2[0] = False
            h2 = torch.empty((r1_ - r0_, No), dtype=torch.float16, device=dev)
            норма(dr_, ro[r0_:r1_], n2.weight.data, n2.variance_epsilon, h2, ro2[r0_:r1_])
            for W_, y_ in zip(Wn, выходы2):
                if CG.включён() and W_.shape[0] >= _УЗКО:
                    try:
                        умн(h2, W_, y_[r0_:r1_])
                        continue
                    except RuntimeError:
                        pass
                torch.mm(h2, W_.t(), out=y_[r0_:r1_])
        # 3) MLP по кускам, как только сведено out_proj куска; сведение down -- на побочном потоке
        for i, (r0, r1) in enumerate(куски):
            if i + вперёд < len(куски):
                _out(i + вперёд)
            yr, evr = сведённые.pop(i), готово.pop(i)
            осн.wait_event(evr)
            h = torch.empty((r1 - r0, No), dtype=torch.float16, device=dev)
            норма(yr, residual[r0:r1], nw.weight.data, nw.variance_epsilon, h, ro[r0:r1])
            g = умн(h, Wg)
            if хук is not None:
                хук(h, g, [gup.scales], r0)
            a = torch.empty((r1 - r0, Ng // 2), dtype=torch.float16, device=dev)
            torch.ops._C.silu_and_mul(a, g)
            d = умн(a, Wd)
            if хук is not None:
                хук(a, d, [dnp.scales], r0)
            ev = torch.cuda.Event()
            ev.record(осн)
            with torch.cuda.stream(поб):
                поб.wait_event(ev)
                dr = tensor_model_parallel_all_reduce(d)
                if след_куски:
                    evd = torch.cuda.Event()
                    evd.record(поб)
                else:
                    out[r0:r1].copy_(dr)
            d.record_stream(поб)
            if след_куски:
                dr.record_stream(осн)
                сведённые_d[i] = (dr, evd)
        N3 = _РЕЕСТР.get(getattr(L, "_fa2_bp_next", ""))
        if _DQ_ВПЕРЁД and not след_куски and N3 is not None:
            proj3 = N3.linear_attn.out_proj if N3.layer_type == "linear_attention" else N3.self_attn.o_proj
            dnp3 = N3.mlp.down_proj
            if (_w8(proj3) and _w8(dnp3) and proj3.fa2sm80_kn == proj.fa2sm80_kn
                    and dnp3.fa2sm80_kn == dnp.fa2sm80_kn):
                ev_k = torch.cuda.Event()
                ev_k.record(осн)                 # после последнего down-GEMM и всех out_proj: буферы B и C свободны
                дкв.wait_event(ev_k)
                with torch.cuda.stream(дкв):
                    Wo3 = _деквант(proj3, _буфер_b(dev, max(кнB)))
                    Wd3 = _деквант(dnp3, _буфер_c(dev, Kd * Nd))
                    ev3 = torch.cuda.Event()
                    ev3.record(дкв)
                _ЗАПАС[dev.index] = (_С["вызов"], N3._fa2_bp_key, Wo3, Wd3, ev3)
        if след_куски:
            # куски следующего слоя -- после всего MLP, по порядку: сведения down шли, пока считался MLP остальных,
            # хвост (последние сведения) -- за входными проекциями предыдущих кусков
            for j in range(len(куски)):
                _след(j)
            осн.wait_stream(поб)
            for y_ in выходы2:
                y_.record_stream(осн)
            return tuple(выходы2) + (ro2,)
        осн.wait_stream(поб)
        out.record_stream(поб)
        if N2 is not None:                  # следующий слой по кускам нельзя (Clef и т. п.) -- подряд через модули
            return _след_прямо(N2, out, ro)
        return out, ro

    def _blockpipe(core: torch.Tensor, residual: torch.Tensor, key: str) -> tuple[torch.Tensor, torch.Tensor]:
        return _тело(core, residual, key)

    def _bpx_gdn(core: torch.Tensor, residual: torch.Tensor, key: str,
                 next_key: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _тело(core, residual, key, next_key)

    def _bpx_attn(core: torch.Tensor, residual: torch.Tensor, key: str,
                  next_key: str) -> tuple[torch.Tensor, torch.Tensor]:
        return _тело(core, residual, key, next_key)

    def _blockpipe_fake(core: torch.Tensor, residual: torch.Tensor, key: str) -> tuple[torch.Tensor, torch.Tensor]:
        L = _РЕЕСТР[key]
        H = L.mlp.down_proj.output_size_per_partition
        return (core.new_empty((core.shape[0], H)),
                core.new_empty((core.shape[0], H), dtype=torch.float32))

    direct_register_custom_op("fa2sm80_blockpipe", _blockpipe, mutates_args=[], fake_impl=_blockpipe_fake)

    def _размеры_след(next_key):
        N2 = _РЕЕСТР[next_key]
        return [p.output_size_per_partition for p in _проекции_след(N2)], N2.input_layernorm.weight.shape[0]

    def _bpx_gdn_fake(core: torch.Tensor, residual: torch.Tensor, key: str,
                      next_key: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (n1, n2), H = _размеры_след(next_key)
        T = core.shape[0]
        return core.new_empty((T, n1)), core.new_empty((T, n2)), core.new_empty((T, H), dtype=torch.float32)

    def _bpx_attn_fake(core: torch.Tensor, residual: torch.Tensor, key: str,
                       next_key: str) -> tuple[torch.Tensor, torch.Tensor]:
        (n1,), H = _размеры_след(next_key)
        T = core.shape[0]
        return core.new_empty((T, n1)), core.new_empty((T, H), dtype=torch.float32)

    direct_register_custom_op("fa2sm80_bpx_gdn", _bpx_gdn, mutates_args=[], fake_impl=_bpx_gdn_fake)
    direct_register_custom_op("fa2sm80_bpx_attn", _bpx_attn, mutates_args=[], fake_impl=_bpx_attn_fake)

    from vllm.model_executor.models import qwen3_5 as Q5
    Слой = QN.Qwen3NextDecoderLayer
    _прежний_fwd = Слой.forward

    def _регистрация(self, a, kw):
        prefix = kw.get("prefix", a[2] if len(a) > 2 else "")
        self._fa2_bp_key = ""
        mlp = getattr(self, "mlp", None)
        if not prefix or "mtp" in prefix or mlp is None or not hasattr(mlp, "gate_up_proj"):
            return
        mixer = self.linear_attn if self.layer_type == "linear_attention" else self.self_attn
        if prefix in _РЕЕСТР:
            return
        _РЕЕСТР[prefix] = self
        self._fa2_bp_key = prefix
        mixer._fa2_defer_proj = True
        import re as _re
        m = _re.search(r"^(.*\.layers\.)(\d+)$", prefix)
        self._fa2_bp_prev = f"{m.group(1)}{int(m.group(2)) - 1}" if m else ""
        self._fa2_bp_next = f"{m.group(1)}{int(m.group(2)) + 1}" if m else ""

    # Qwen3_5DecoderLayer.__init__ зовёт super(Qwen3NextDecoderLayer, self).__init__() -- МИМО __init__ родителя:
    # регистрация вешается на оба класса; forward у Qwen3_5DecoderLayer унаследован от Qwen3NextDecoderLayer.
    def _обёртка_init(прежний):
        def __init__(self, *a, **kw):
            прежний(self, *a, **kw)
            if getattr(self, "_fa2_bp_key", None) is None:
                _регистрация(self, a, kw)
        return __init__

    def _принимает(слой):
        """Слой может получить готовые входные проекции: полное внимание (qkv_proj) или GDN с раздельными
        in_proj_qkvz/in_proj_ba. Одно и то же правило у отдающего и у принимающего -- цепочка согласована."""
        if слой.layer_type != "linear_attention":
            return True
        return bool(getattr(слой.linear_attn, "use_split_input_projections", False))

    def forward(self, hidden_states, residual, positions=None, **kwargs):
        key = getattr(self, "_fa2_bp_key", "")
        if not key:
            return _прежний_fwd(self, hidden_states, residual, positions=positions, **kwargs)
        # Цепочка через границу слоя: предыдущий слой отдал ГОТОВЫЕ входные проекции этого (input_layernorm уже
        # применён, residual уже сложен) -- кортеж вместо hidden_states. Решения -- по реестру при трассировке.
        получает = _ВКЛ_X and self._fa2_bp_prev in _РЕЕСТР and _принимает(self)
        отдаёт = _ВКЛ_X and self._fa2_bp_next in _РЕЕСТР and _принимает(_РЕЕСТР[self._fa2_bp_next])
        if получает:
            pre = hidden_states
            if self.layer_type == "linear_attention":
                core = self.linear_attn(hidden_states=pre[0], output=pre[0], pre_proj=pre)
            else:
                core = self.self_attn(hidden_states=pre[0], output=pre[0], positions=positions, pre_proj=pre)
        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            if self.layer_type == "linear_attention":
                core = self.linear_attn(hidden_states=hidden_states, output=hidden_states)
            else:
                core = self.self_attn(hidden_states=hidden_states, output=hidden_states, positions=positions)
        if отдаёт:
            if _РЕЕСТР[self._fa2_bp_next].layer_type == "linear_attention":
                qkvz, ba, residual = torch.ops.vllm.fa2sm80_bpx_gdn(core, residual, key, self._fa2_bp_next)
                return (qkvz, ba), residual
            qkv, residual = torch.ops.vllm.fa2sm80_bpx_attn(core, residual, key, self._fa2_bp_next)
            return (qkv,), residual
        out, residual = torch.ops.vllm.fa2sm80_blockpipe(core, residual, key)
        return out, residual

    for К in (QN.Qwen3NextDecoderLayer, Q5.Qwen3_5DecoderLayer):
        if "__init__" in К.__dict__:
            К.__init__ = _обёртка_init(К.__dict__["__init__"])
    Слой.forward = forward
    _лог(f"включён: out_proj -> сведение -> add+norm -> MLP -> сведение одной операцией, при M >= {_MIN} куски по "
         f"{_ROWS} строк, сведения на побочном потоке высокого приоритета"
         + ("; ЧЕРЕЗ ГРАНИЦУ СЛОЯ: входные проекции следующего слоя -- в той же операции" if _ВКЛ_X else ""))
    return True


def счётчики():
    return dict(_С)
