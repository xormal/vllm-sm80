# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
"""Префилл: КОНВЕЙЕР «умножение -> сведение» в строко-параллельных слоях (o_proj, out_proj GDN, down_proj).

Включается FA2SM80_ROWPIPE=1. Обмен TP -- самая крупная доля префилла (43 % до P2P), и он стоял СТРОГО после
умножения: слой ждал, пока сведение закончится. Здесь выход слоя режется по строкам (токенам) на куски по
FA2SM80_ROWPIPE_ROWS: кусок i умножается на основном потоке, а его сведение (тот же путь: int12-упаковка ->
all_gather -> распаковка со сложением) идёт на ПОБОЧНОМ потоке ВЫСОКОГО приоритета, пока умножается кусок
i+1. Приоритет обязателен: cuBLAS занимает все SM, и без него блоки сведения ждали бы конца умножения.

ПОЧЕМУ ОПЕРАЦИЕЙ, А НЕ ПРАВКОЙ ПРОХОДА. Прошлое перекрытие (fa2sm70_overlap, две половины куска) шло ручным
проходом вне torch.compile и теряло слитые ядра -- x0.83. Здесь torch.compile видит ОДНУ непрозрачную операцию
на месте «умножение + сведение», всё вокруг компилируется как прежде. Ветвление по числу строк -- ВНУТРИ
операции (ветвь в forward поставила бы dynamo охрану по размеру и перекомпиляцию): при M < FA2SM80_ROWPIPE_MIN
(декод, захваченные графы) путь дословно прежний -- умножение и сведение на текущем потоке.

ВЕС W8 РАСПАКОВЫВАЕТСЯ ОДИН РАЗ на слой (общий буфер fa2sm80_w8), куски -- torch.mm по распакованному: наивная
нарезка звала бы распаковку на каждый кусок. Численность: cuBLAS на куске может выбрать другой алгоритм, чем на
целом M, -- строки не побитово те же (упаковка int12 -- по группам внутри строки, от нарезки не зависит).
"""
import os
import sys

_ВКЛ = os.environ.get("FA2SM80_ROWPIPE", "0") == "1"
_MIN = int(os.environ.get("FA2SM80_ROWPIPE_MIN", "2048"))     # строк, с которых режем
_ROWS = int(os.environ.get("FA2SM80_ROWPIPE_ROWS", "1024"))   # строк в куске
_РЕЕСТР = {}        # ключ (prefix слоя) -> слой
_ПОБОЧНЫЙ = {}      # карта -> поток высокого приоритета
_С = {"конвейер": 0, "прямо": 0}


def _лог(msg):
    print(f"[fa2sm80 ROWPIPE] {msg}", file=sys.stderr, flush=True)


def установить():
    if not _ВКЛ:
        return False
    import torch
    from vllm.model_executor.layers import linear as L
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.utils.torch_utils import direct_register_custom_op

    def _clef_в_шаге():
        """В шаге есть строки Clef. Дельта Clef добавляется обёрткой ВНУТРИ тела умножения
        (fa2sm70_clef_hook.с_поправкой), строки Clef заданы номерами в ЦЕЛОМ шаге: кусочный путь (распаковка +
        torch.mm мимо тела) зовёт ХУК_ДИАП со смещением куска (clef.cu delta_keyed_range); без него -- прежний путь."""
        кл = sys.modules.get("fa2sm70_clef")
        if кл is None:
            return False
        f = getattr(кл, "шаг_с_поправкой", None)     # [ОДИН ГРАФ 30.10] и захват/повтор графа has_clef
        return bool(f()) if f is not None else bool(getattr(кл, "_АКТИВЕН", [False])[0])

    def _умножитель(layer, x, bias_):
        """Функция умножения куска: для W8 большого M -- распаковка ОДИН раз + torch.mm по кускам."""
        kn = getattr(layer, "fa2sm80_kn", None)
        if kn is not None and bias_ is None:
            import fa2sm80_w8 as W8
            K, N = kn
            т = W8._ТЮН.get((K, N))
            if (т is not None or W8.марлин_ли(layer)) and K * N <= W8._БУФ_МАКС:
                W = W8.деквант_слоя(layer, W8._буфер_для(x.device, K * N))   # [ФРИЗЫ 30.10] любая раскладка
                import fa2sm80_cg as CG
                if CG.включён():
                    return (lambda xi, out: CG.mm(xi, W, out)), N, True
                Wt = W.t()
                return (lambda xi, out: torch.mm(xi, Wt, out=out)), N, True
        def _общий(xi, out):
            out.copy_(layer.quant_method.apply(layer, xi, bias_))
        return _общий, layer.output_size_per_partition, False

    def _rowpipe(x: torch.Tensor, key: str) -> torch.Tensor:
        layer = _РЕЕСТР[key]
        bias_ = None if (layer.tp_rank > 0 or layer.skip_bias_add) else layer.bias
        M = x.shape[0]
        if M < _MIN or x.dim() != 2:
            _С["прямо"] += 1
            return tensor_model_parallel_all_reduce(layer.quant_method.apply(layer, x, bias_))
        умн, N, w8 = _умножитель(layer, x, bias_)
        clef = _clef_в_шаге()
        хук_д = None
        if clef:
            import fa2sm70_clef_hook as H
            хук_д = H.ХУК_ДИАП[0]
            if хук_д is None or not w8:
                _С["прямо"] += 1                  # дельту Clef по кускам дать нечем -- прежний путь
                return tensor_model_parallel_all_reduce(layer.quant_method.apply(layer, x, bias_))
            _С["clef"] = _С.get("clef", 0) + 1
        _С["конвейер"] += 1
        if _С["конвейер"] in (1, 1000, 100000) or _С.get("clef") == 1:
            _лог(f"конвейер работает: слой {key}, M={M}, кусок {_ROWS} строк (вызовов {_С['конвейер']}, "
                 f"с Clef {_С.get('clef', 0)})")
        out = torch.empty((M, N), dtype=x.dtype if x.dtype == torch.float16 else torch.float16, device=x.device)
        осн = torch.cuda.current_stream()
        поб = _ПОБОЧНЫЙ.get(x.device.index)
        if поб is None:
            поб = _ПОБОЧНЫЙ[x.device.index] = torch.cuda.Stream(device=x.device, priority=-1)
        поб.wait_stream(осн)                      # всё, что было до операции (вход, буфер обмена), готово
        for r0 in range(0, M, _ROWS):
            r1 = min(M, r0 + _ROWS)
            y = torch.empty((r1 - r0, N), dtype=out.dtype, device=x.device)
            умн(x[r0:r1], y)
            if хук_д is not None:                 # дельта Clef по окну строк куска -- до сведения, на основном потоке
                хук_д(x[r0:r1], y, [layer.scales], r0)
            ev = torch.cuda.Event()
            ev.record(осн)
            with torch.cuda.stream(поб):
                поб.wait_event(ev)
                r = tensor_model_parallel_all_reduce(y)
                out[r0:r1].copy_(r)
            y.record_stream(поб)                  # память куска не отдавать основному потоку до конца сведения
        осн.wait_stream(поб)
        out.record_stream(поб)
        return out

    def _rowpipe_fake(x: torch.Tensor, key: str) -> torch.Tensor:
        return x.new_empty((x.shape[0], _РЕЕСТР[key].output_size_per_partition))

    direct_register_custom_op("fa2sm80_rowpipe", _rowpipe, mutates_args=[], fake_impl=_rowpipe_fake)

    _прежний = L.RowParallelLinear.forward
    _прежний_init = L.RowParallelLinear.__init__

    def __init__(self, *a, **kw):
        # РЕГИСТРАЦИЯ ПРИ ПОСТРОЕНИИ, а не в forward: изменение глобального словаря во время трассировки
        # dynamo -- лишние охраны. forward видит только атрибут модуля (для dynamo -- константу).
        _прежний_init(self, *a, **kw)
        key = getattr(self, "prefix", "") or ""
        self._rowpipe_ok = False
        if key and self.reduce_results and self.tp_size > 1 and self.input_is_parallel:
            if key in _РЕЕСТР:
                _РЕЕСТР[key]._rowpipe_ok = False      # два слоя с одним prefix -- не рискуем ни одним
            else:
                _РЕЕСТР[key] = self
                self._rowpipe_ok = True

    def forward(self, input_):
        if not getattr(self, "_rowpipe_ok", False):
            return _прежний(self, input_)
        x = input_.reshape(-1, input_.shape[-1])
        y = torch.ops.vllm.fa2sm80_rowpipe(x, self.prefix)
        output = y.reshape(input_.shape[:-1] + (y.shape[-1],))
        if not self.return_bias:
            return output
        return output, (self.bias if self.skip_bias_add else None)

    L.RowParallelLinear.__init__ = __init__
    L.RowParallelLinear.forward = forward
    _лог(f"включён: строко-параллельные слои при M >= {_MIN} режутся по {_ROWS} строк, сведение -- на "
         f"побочном потоке высокого приоритета")
    return True


def счётчики():
    return dict(_С)
