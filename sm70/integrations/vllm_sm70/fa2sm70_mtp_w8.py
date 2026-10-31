# -*- coding: utf-8 -*-
"""Тело головы MTP в int8 на НАШЕМ ядре (рычаг FA2SM70_MTP_W8).

ЗАЧЕМ. Голова MTP приезжает в слепке НЕ КВАНТОВАННОЙ (обычные `.weight`, bf16), хотя вся
остальная сеть -- GPTQ-int8. Черновик проходит эту голову ТРИЖДЫ за шаг, и карточный прибор
(записка 25 §249) намерил: тело головы 3.94 мс из 10.59 мс прохода черновика, то есть 7 %
всего шага. Замер на НАСТОЯЩИХ весах при M=1 (§251):

    плотный bf16   992 мкс/проход (405 МиБ на ранг)
    наше int8      362 мкс/проход (202 МиБ на ранг)      x2.74

Плюс освобождается 202 МиБ на ранг -- единственный найденный способ дать место int4-словарю
черновика (303 МиБ, +9 % декода), который иначе не уживается с 262 144 токенами (§247).

ПОЧЕМУ ЭТО НЕ ТРОГАЕТ КАЧЕСТВО. Черновик только ПРЕДЛАГАЕТ; каждый токен проверяется целью,
и отбраковка гарантирует тождество выхода. Ошибка квантования (7e-3 по слою) может лишь
чуть-чуть снизить ПРИЁМКУ -- это скорость, а не качество. Гейт: tau до и после.

ФОРМАТ. Упаковка -- ровно GPTQ v1 8 бит (qweight [K/4,N] int32, scales [K/G,N], qzeros
[K/G,N/4] со значением z-1), проверена гейтом: чужой `ops.gptq_gemm` принимает её с
относительной ошибкой 1.2e-3 при M=1..64. Это обязательное условие: ПЕРВЫЙ проход черновика
после префилла идёт с большим M, где наше gemv-тело проигрывает, и отвод должен работать.
"""

import os
import sys

import torch

_ВКЛ = os.environ.get("FA2SM70_MTP_W8", "0") == "1"
_G = int(os.environ.get("FA2SM70_MTP_W8_GROUP", "128"))
_MMAX = int(os.environ.get("FA2SM70_MTP_W8_MMAX", "4"))

# [ПАМЯТЬ 17.09] ДЕЛЕНИЕ ШИРОКОГО ТЕЛА ПО СТОЛБЦАМ.
# Чужой `ops.gptq_gemm` заводит буфер деквантизации РАЗМЕРОМ С ВЕСЬ ВЕС: K*N*2 байт.
# Для gate_up головы MTP (K=5120, N=17408) это 170 МиБ ОДНИМ куском, и он берётся на
# каждом чанке префилла (M=2048 > порога M=4). Замер 17.09 на стенде: при util 0.96
# свободно 109.5 МиБ, и восемь одновременных запросов с картинками дают ровно пять
# отказов «Tried to allocate 170.00 MiB» на обеих картах -- движок в бою от этого умер.
# Деление веса на S столбцовых кусков делает буфер в S раз меньше (170 -> 42.5 при S=4)
# и убирает единственную точку отказа. Цена -- S пусков ядра вместо одного; берём её
# только на широких формах (N >= SPLIT_NMIN), поэтому в декоде это 3 лишних пуска на
# проход черновика, а не на каждый слой.
_NSPLIT = int(os.environ.get("FA2SM70_MTP_W8_NSPLIT", "1"))
_SPLIT_NMIN = int(os.environ.get("FA2SM70_MTP_W8_SPLIT_NMIN", "8192"))

# [ПАМЯТЬ 31.10] MLP ГОЛОВЫ ПО КУСКАМ СТРОК (A100, FA2SM80_MTP_MLP_ROWS=N, 0 -- выкл.).
# 31.10 19:30 боевой упал OOM в черновике на шаге 8192 токенов: gate_up головы держал выход [M, 17408] (272 МиБ), а
# SiluAndMul заводил поверх [M, 8704] (136 МиБ) -- не хватило 6 МиБ (свободно 130, ещё 499 дробных в резерве). MLP
# основной сети режет на куски конвейер блока, голову MTP он не берёт. Здесь gate_up + SiluAndMul -- одна непрозрачная
# операция: при M >= FA2SM80_MTP_MLP_MIN строки идут равными кусками ~N, выход [M, 8704] заполняется по кускам, пик --
# выход + один кусок gate_up (34 МиБ при N=1024) вместо 408 МиБ. Выход побитно тот же: при M > 512 gate_up -- деквант +
# свой GEMM (FA2SM80_CG, инвариантен к пачке), куски при N >= 1024 и M >= 2048 длиннее 512 строк и идут тем же
# маршрутом; SiLU -- то же выражение, что сшитое inductor-ядро прежнего прохода: (g * sigmoid(g)) * u в fp32, одно
# округление. FA2SM80_MTP_MLP_CHECK=n -- первые n резаных вызовов сверяются с целым проходом (память целого -- на время
# сверки). Отказ памяти на куске -- кусок вдвое меньше (до 128 строк) с той же строки: обмена TP в операции нет, ранги
# не расходятся; куски <= 512 строк идут другим ядром -- черновик такого шага может разойтись в последнем бите.
_РЯДЫ = int(os.environ.get("FA2SM80_MTP_MLP_ROWS", "0"))
_РЯДЫ_MIN = int(os.environ.get("FA2SM80_MTP_MLP_MIN", "2048"))
_СВЕРКА = [int(os.environ.get("FA2SM80_MTP_MLP_CHECK", "0"))]
_С = {"слоёв": 0, "МиБ": 0.0, "делённых": 0, "mlp_кусками": 0, "mlp_отказов": 0}
_силу_ядро = None


def _лог(msg):
    print(f"[fa2_sm70 MTP-W8] {msg}", file=sys.stderr, flush=True)


def _силу():
    """Triton: a = (g * sigmoid(g)) * u, g и u -- половины строки gate_up [M, 2I]; fp32, одно округление в fp16.
    То же выражение, что inductor сшивает из SiluAndMul.forward_native -- побитно как прежний проход."""
    global _силу_ядро
    if _силу_ядро is not None:
        return _силу_ядро
    import triton
    import triton.language as tl

    @triton.jit
    def _k(g_ptr, a_ptr, n, I: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        row = i // I
        col = i % I
        g = tl.load(g_ptr + (col + 2 * I * row), m).to(tl.float32)
        u = tl.load(g_ptr + (I + col + 2 * I * row), m).to(tl.float32)
        tl.store(a_ptr + i, (g * tl.sigmoid(g)) * u, m)

    def силу(g, a):
        n = a.numel()
        if n:
            _k[(triton.cdiv(n, 1024),)](g, a, n, I=a.shape[1], BLOCK=1024, num_warps=4)

    _силу_ядро = силу
    return силу


def _куски_mlp(M, ряды):
    """Равные куски по ~ряды строк: последний не короче прочих (короткий хвост ушёл бы другим маршрутом GEMM)."""
    шаг = -(-M // -(-M // ряды))
    return [(r0, min(M, r0 + шаг)) for r0 in range(0, M, шаг)]


def _gate_silu(x: torch.Tensor, wf: torch.Tensor, meta: torch.Tensor, sc: torch.Tensor,
               K: int, N: int) -> torch.Tensor:
    M = x.shape[0]
    a = x.new_empty((M, N // 2))
    силу = _силу()
    if _РЯДЫ <= 0 or M < _РЯДЫ_MIN or torch.cuda.is_current_stream_capturing():
        силу(torch.ops.vllm.fa2sm80_w8_mm(x, wf, meta, sc, K, N), a)
        return a
    _С["mlp_кусками"] += 1
    if _С["mlp_кусками"] in (1, 1000):
        _лог(f"MLP головы по кускам работает: M={M}, кусок ~{_РЯДЫ} строк (вызовов {_С['mlp_кусками']})")
    куски = _куски_mlp(M, _РЯДЫ)
    while куски:
        r0, r1 = куски[0]
        try:
            g = torch.ops.vllm.fa2sm80_w8_mm(x[r0:r1], wf, meta, sc, K, N)
        except torch.OutOfMemoryError:
            if r1 - r0 <= 128:
                raise
            _С["mlp_отказов"] += 1
            _лог(f"отказ памяти на куске {r1 - r0} строк (M={M}) -- остаток кусками вдвое меньше "
                 f"(отказов {_С['mlp_отказов']})")
            шаг = (r1 - r0 + 1) // 2
            куски = [(a0, min(M, a0 + шаг)) for a0 in range(r0, M, шаг)]
            continue
        силу(g, a[r0:r1])
        del g
        куски.pop(0)
    if _СВЕРКА[0] > 0:
        a0 = torch.empty_like(a)
        силу(torch.ops.vllm.fa2sm80_w8_mm(x, wf, meta, sc, K, N), a0)
        разн = int((a.view(torch.int16) != a0.view(torch.int16)).sum())
        # прогоны профиля и прогрева при старте подают одинаковые строки -- такая сверка слабее и в счёт не идёт
        одинаковые = torch.equal(x[0], x[M // 2])
        if not одинаковые:
            _СВЕРКА[0] -= 1
        _лог(f"сверка MLP головы по кускам M={M} ({'строки одинаковые' if одинаковые else 'строки разные'}): "
             f"различий {разн} из {a.numel()}")
        del a0
    return a


def _мтп(prefix: str) -> bool:
    return prefix.startswith("mtp.") or ".mtp." in prefix


def _упаковать(W: torch.Tensor):
    """W [K,N] (half/bf16) -> (qweight [K/4,N] int32, qzeros [K/G,N/4] int32, scales [K/G,N])."""
    K, N = W.shape
    Wg = W.reshape(K // _G, _G, N).float()
    амп = torch.maximum(Wg.amax(1).abs(), Wg.amin(1).abs()).clamp_min(1e-8)
    sc = амп / 127.0
    q = torch.clamp(torch.round(Wg / sc.unsqueeze(1)) + 128, 0, 255).to(torch.int32)
    q = q.reshape(K, N)
    qw = torch.zeros(K // 4, N, dtype=torch.int32, device=W.device)
    for i in range(4):
        qw |= (q[i::4] & 0xFF) << (8 * i)
    del q, Wg
    # gptq v1 хранит z-1; ядро и чужое тело прибавляют 1 обратно -> нулевая точка 128.
    qz = torch.zeros(K // _G, N // 4, dtype=torch.int32, device=W.device)
    for i in range(4):
        qz |= (127 & 0xFF) << (8 * i)
    return qw.contiguous(), qz.contiguous(), sc.half().contiguous()


def _разделить(qw, qz, sc, s: int):
    """[K/4,N],[K/G,N/4],[K/G,N] -> s непрерывных кусков по столбцам (или None)."""
    n = int(qw.shape[1])
    if s < 2 or n % (4 * s):
        return None
    ш = n // s
    куски = []
    for i in range(s):
        a, b = i * ш, (i + 1) * ш
        куски.append((qw[:, a:b].contiguous(), qz[:, a // 4:b // 4].contiguous(),
                      sc[:, a:b].contiguous()))
    return куски


def установить() -> bool:
    if not _ВКЛ:
        return False
    try:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    except Exception as e:  # noqa: BLE001
        print(f"[fa2_sm70 MTP-W8] движок не отдал UnquantizedLinearMethod ({str(e)[:60]})",
              file=sys.stderr, flush=True)
        return False
    if getattr(UnquantizedLinearMethod, "_фа2_мтп_w8", False):
        return True
    _s80 = hasattr(torch.ops.vllm, "fa2sm80_w8_mm")      # A100: общая операция fa2sm80_w8 (FA2SM80_W8=1)
    if not _s80 and not hasattr(torch.ops.vllm, "fa2sm70_w8_mm"):
        print("[fa2_sm70 MTP-W8] операция fa2sm70_w8_mm не зарегистрирована "
              "(нужен FA2SM70_W8=1) -- подмены нет", file=sys.stderr, flush=True)
        return False

    if not _s80 and _NSPLIT > 1 and not hasattr(torch.ops.vllm, "fa2sm70_w8_mm_ch"):
        from typing import List

        from vllm.utils.torch_utils import direct_register_custom_op

        def _w8_mm_ch(x: torch.Tensor, qws: List[torch.Tensor], qzs: List[torch.Tensor],
                      scs: List[torch.Tensor], gi: torch.Tensor, group: int, v2: bool,
                      mmax: int, ntotal: int) -> torch.Tensor:
            # Выход заводится ОДИН раз и заполняется по кускам: пик = выход + буфер куска,
            # а не выход + буфер ЦЕЛОГО веса.
            y = x.new_empty((x.shape[0], ntotal))
            сдв = 0
            for qw, qz, sc in zip(qws, qzs, scs):
                ш = int(qw.shape[1])
                y[:, сдв:сдв + ш] = torch.ops.vllm.fa2sm70_w8_mm(x, qw, qz, sc, gi, group,
                                                                 v2, mmax)
                сдв += ш
            return y

        def _w8_mm_ch_fake(x: torch.Tensor, qws: List[torch.Tensor], qzs: List[torch.Tensor],
                           scs: List[torch.Tensor], gi: torch.Tensor, group: int, v2: bool,
                           mmax: int, ntotal: int) -> torch.Tensor:
            return x.new_empty((x.shape[0], ntotal))

        try:
            direct_register_custom_op("fa2sm70_w8_mm_ch", _w8_mm_ch, mutates_args=[],
                                      fake_impl=_w8_mm_ch_fake)
        except Exception as e:  # noqa: BLE001
            print(f"[fa2_sm70 MTP-W8] составная свёртка не зарегистрирована ({str(e)[:70]}) "
                  f"-- деления не будет", file=sys.stderr, flush=True)

    _чужой_обработать = UnquantizedLinearMethod.process_weights_after_loading

    def process_weights_after_loading(self, layer):
        _чужой_обработать(self, layer)
        try:
            prefix = getattr(layer, "prefix", "") or ""
            w = getattr(layer, "weight", None)
            if not _мтп(prefix) or w is None or w.dim() != 2 or not w.is_cuda:
                return
            N, K = w.shape                      # торч хранит [выход, вход]
            if K % (4 * _G) or N % 4:
                print(f"[fa2_sm70 MTP-W8] {prefix}: форма K={K} N={N} не годится -- оставлен "
                      f"плотным", file=sys.stderr, flush=True)
                return
            qw, qz, sc = _упаковать(w.data.t().contiguous())
            layer.фа2_qweight, layer.фа2_qzeros, layer.фа2_scales = qw, qz, sc
            layer.фа2_gidx = torch.empty((0,), dtype=torch.int, device=w.device)
            layer._мтп_w8 = True
            layer._мтп_ш = int(qw.shape[1])
            if _s80:
                # [A100] фрагментный порядок w8a16_sm80; большие M -- через ОБЩИЙ деквант-буфер fa2sm80_w8,
                # отдельного буфера на вес (170 МиБ у gate_up, отказы на Volta 17.09) не заводится
                import fa2sm80_w8
                wf, meta = fa2sm80_w8.упаковать_вне_gptq(qw, qz, sc, False, K, N)
                layer.фа2_s80 = (wf, meta, sc, K, N)
                layer.фа2_qweight = layer.фа2_qzeros = None
            elif (_NSPLIT > 1 and N >= _SPLIT_NMIN
                    and hasattr(torch.ops.vllm, "fa2sm70_w8_mm_ch")):
                куски = _разделить(qw, qz, sc, _NSPLIT)
                if куски is not None:
                    layer.фа2_qw_ч = [к[0] for к in куски]
                    layer.фа2_qz_ч = [к[1] for к in куски]
                    layer.фа2_sc_ч = [к[2] for к in куски]
                    layer.фа2_qweight = layer.фа2_qzeros = layer.фа2_scales = None
                    layer._мтп_w8_ч = True
                    _С["делённых"] += 1
                    print(f"[fa2_sm70 MTP-W8] {prefix}: N={N} поделён на {_NSPLIT} "
                          f"(буфер деквантизации {K * N * 2 / 2 ** 20:.0f} -> "
                          f"{K * N * 2 / _NSPLIT / 2 ** 20:.0f} МиБ)",
                          file=sys.stderr, flush=True)
            было = w.numel() * w.element_size()
            стало = qw.numel() * 4 + qz.numel() * 4 + sc.numel() * 2
            layer._parameters.pop("weight", None)
            layer.weight = None
            _С["слоёв"] += 1
            _С["МиБ"] += (было - стало) / 2 ** 20
        except Exception as e:  # noqa: BLE001
            # Отказ здесь обязан ДЕГРАДИРОВАТЬ, а не ронять воркер: плотный путь остаётся.
            print(f"[fa2_sm70 MTP-W8] {getattr(layer,'prefix','?')}: упаковка не вышла "
                  f"({str(e)[:80]}) -- слой оставлен плотным", file=sys.stderr, flush=True)

    _чужой_apply = UnquantizedLinearMethod.apply

    def apply(self, layer, x, bias=None):
        # Условие СТАТИЧНО для слоя (проставлено при загрузке), поэтому его разрешение на
        # трассировке законно -- см. разбор в fa2sm70_w8.py.
        if not getattr(layer, "_мтп_w8", False):
            return _чужой_apply(self, layer, x, bias)
        out_shape = x.shape[:-1] + (layer._мтп_ш,)
        xr = x.reshape(-1, x.shape[-1])
        s80 = getattr(layer, "фа2_s80", None)
        if s80 is not None:
            if xr.dtype != torch.float16:
                xr = xr.half()
            y = torch.ops.vllm.fa2sm80_w8_mm(xr, s80[0], s80[1], s80[2], s80[3], s80[4])
            if bias is not None:
                y = y + bias
            return y.reshape(out_shape)
        if getattr(layer, "_мтп_w8_ч", False):
            y = torch.ops.vllm.fa2sm70_w8_mm_ch(xr, layer.фа2_qw_ч, layer.фа2_qz_ч,
                                                layer.фа2_sc_ч, layer.фа2_gidx, _G, False,
                                                _MMAX, layer._мтп_ш)
        else:
            y = torch.ops.vllm.fa2sm70_w8_mm(xr, layer.фа2_qweight, layer.фа2_qzeros,
                                             layer.фа2_scales, layer.фа2_gidx, _G, False,
                                             _MMAX)
        if bias is not None:
            y = y + bias
        return y.reshape(out_shape)

    if _s80:
        from vllm.utils.torch_utils import direct_register_custom_op

        def _gate_silu_fake(x: torch.Tensor, wf: torch.Tensor, meta: torch.Tensor, sc: torch.Tensor,
                            K: int, N: int) -> torch.Tensor:
            return x.new_empty((x.shape[0], N // 2))

        # Операция регистрируется и при выключенном рычаге: файл кэша компиляции хэшируется по содержимому
        # трассированных файлов, а не по рычагу -- граф с ней, поднятый из кэша, должен находить её (без рычага --
        # целым проходом, побитно как прежде). Отказ регистрации не должен снять int8 головы: только без кусков.
        try:
            direct_register_custom_op("fa2sm80_mtp_gate_silu", _gate_silu, mutates_args=[],
                                      fake_impl=_gate_silu_fake)
        except Exception as e:  # noqa: BLE001
            _лог(f"операция MLP головы не зарегистрирована ({str(e)[:120]}) -- MLP головы целиком, как прежде")
    if _s80 and _РЯДЫ > 0 and hasattr(torch.ops.vllm, "fa2sm80_mtp_gate_silu"):
        from vllm.model_executor.models import qwen2_moe as Q2
        _прежний_mlp = Q2.Qwen2MoeMLP.forward

        def mlp_forward(self, x):
            # Условие статично для слоя (фа2_s80 проставлен при загрузке) -- разрешение на трассировке законно.
            s80 = getattr(self.gate_up_proj, "фа2_s80", None)
            if (s80 is None or self.expert_gate is not None or x.dim() != 2
                    or type(self.act_fn).__name__ != "SiluAndMul"):
                return _прежний_mlp(self, x)
            a = torch.ops.vllm.fa2sm80_mtp_gate_silu(x if x.dtype == torch.float16 else x.half(),
                                                     s80[0], s80[1], s80[2], s80[3], s80[4])
            out, _ = self.down_proj(a)
            return out

        Q2.Qwen2MoeMLP.forward = mlp_forward
        _лог(f"MLP головы: gate_up + SiluAndMul одной операцией, при M >= {_РЯДЫ_MIN} куски по ~{_РЯДЫ} строк"
             + (f", сверка первых {_СВЕРКА[0]}" if _СВЕРКА[0] > 0 else ""))

    UnquantizedLinearMethod.process_weights_after_loading = process_weights_after_loading
    UnquantizedLinearMethod.apply = apply
    UnquantizedLinearMethod._фа2_мтп_w8 = True
    print(f"[fa2_sm70 MTP-W8] тело головы MTP переводится в int8 на наше ядро "
          f"(группа {_G}, порог M={_MMAX})", file=sys.stderr, flush=True)
    return True


def счётчики():
    return dict(_С)
