# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
"""GPTQ int8 (g=128, асимметричный) на A100: наше ядро w8a16_sm80 вместо turbomind s884 / gemv / exllama.

Включается FA2SM80_W8=1 (на sm_80+; на Volta не трогает ничего). Заменяет связку вольтовских путей
fa2sm70_w8 (FA2SM70_W8 / FA2SM70_TM8 на A100 выключать: turbomind s884 на sm_80 собирается в пустую выдачу).

ЧТО ДЕЛАЕТ.
  * при загрузке слоя: вес перепаковывается в фрагментный порядок (w8a16_sm80_pack), исходный qweight
    ОСВОБОЖДАЕТСЯ (замена, а не добавка: те же K*N байт), scales/qzeros остаются -- scales служит
    ключом модуля для Clef (fa2sm70_clef ищет тензор-ключ среди аргументов тела);
  * конфигурация ядра (NW, варпы, стадии, расщепление K) ПОДБИРАЕТСЯ ЗАМЕРОМ один раз на форму (K, N)
    и корзину M -- вне захвата графов (обработка весов идёт до профилирования и захвата);
  * тело: M <= 64 -> w8a16_sm80_mm (cp.async-конвейер + m16n8k16, связано чтением весов);
          M > 64  -> деквант в ОБЩИЙ буфер fp16 + cuBLAS (префилл: связано счётом, cuBLAS на A100 ~215 TF).
"""
import os
import sys
import torch

_ВКЛ = os.environ.get("FA2SM80_W8", "0") == "1"
_MMAX = 64
_ЛОГ = os.environ.get("FA2SM80_W8_LOG", "1") == "1"
_С = {"малое": 0, "большое": 0, "слоёв": 0, "байт": 0}
_расширение = [None]
_буфер = {}          # device -> общий fp16-буфер под деквант больших M
_ТЮН = {}            # (K, N) -> {"nw": int, "cfg": {корзина M: (warps, stages, splits)}}
_ИНСТАНСЫ = {        # что собрано в w8a16_sm80.cu (nw -> [(warps, stages)])
    2: [(4, 4), (8, 4), (8, 6), (2, 4), (4, 3)],
    4: [(4, 4), (4, 6), (2, 8), (8, 3), (2, 3), (2, 4)],
}
_КОРЗИНЫ = (4, 8, 16, 32, 64)
_ПРЯМОЙ_ММАКС = int(os.environ.get("FA2SM80_W8_DIRECT_MMAX", "512"))
# ПРЯМОЙ GEMM С split-K (w8a16_sm80_gemm_large: блок 32*MI строк x 128 столбцов, детерминированный свод по z).
# Замер 23.10 (bench_w8_midm): при M=32 обгоняет ядро малых M (down 64 против 107 мкс), при M=48-64 -- в
# 2-2.5 раза (gate_up 156 против 326), при M=128 -- прежний прямой без расщепления до 2.3x (down 158 против
# 363). Подбор (mi, splits) -- на корзину, вместе с выбором маршрута для корзин 32/64.
_ПРЯМ_КОРЗ = (32, 64, 128, 256, 512)
_ПРЯМ_MIDM = os.environ.get("FA2SM80_W8_MIDM", "1") == "1"   # 0 -- прежний маршрут (A/B)
_ПРЯМ_NMAX = 32768        # шире (словарь) -- без подбора: рабочая область split-K росла бы до сотен МБ
_БУФ_МАКС = 96 * 2**20   # элементов: деквант-буфер под большие M только для обычных слоёв (gate_up 89M)
# [24.10] ПРИБОР для п.7 плана (int8 x int8 с каналами-выбросами в fp16): FA2SM80_ACT_STATS=<файл.jsonl>. На больших M
# (префилл) каждый STATS_EVERY-й вызов формы (K,N) пишет: абсолютные максимумы по каналам (верхушку), и ошибку ВЫХОДА слоя
# relL2(y_q, y) при поточечном int8-квантовании активаций -- целиком и с T каналами-выбросами в fp16 (T=0/8/32/128).
# Стоит времени (полный эталонный matmul в fp32) -- только для отдельного прогона, не для замеров скорости.
# [24.10] СИМУЛЯЦИЯ W8A8 (п.7 плана): FA2SM80_W8A8_SIM=T -- на больших M активации квантуются в int8 ПО СТРОКЕ
# (масштаб = max|x| строки / 127), T каналов с наибольшим max|x| в этом вызове остаются fp16 (схема LLM.int8);
# умножение -- прежним путём (FA2SM80_W8A8_SIM_G -- масштаб по группам K). Ничего не ускоряет: отвечает на вопрос, держит ли такой путь точность Clef, ДО ядра.
_W8A8_SIM = int(os.environ.get("FA2SM80_W8A8_SIM", "-1"))


# FA2SM80_W8A8_SIM_G=G: масштаб на строку И группу из G каналов K (G=128 -- группы весов), иначе на строку.
_W8A8_SIM_G = int(os.environ.get("FA2SM80_W8A8_SIM_G", "0"))


def _квант8(xr, G):
    """int8 с симметричным масштабом: на строку (G=0) или на строку и группу G каналов K; возвращает x̂ в fp32."""
    M, K = xr.shape
    if G > 0 and K % G == 0:
        v = xr.view(M, K // G, G)
        s = v.abs().amax(2, keepdim=True).clamp_min(1e-12) / 127.0
        return (torch_round_clamp(v / s) * s).view(M, K)
    s = xr.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127.0
    return torch_round_clamp(xr / s) * s


def torch_round_clamp(v):
    return v.round().clamp(-127, 127)


def _w8a8_сим(x, T):
    import torch
    xf = x.float()
    out_ch = None
    if T > 0:
        out_ch = torch.topk(xf.abs().amax(0), T).indices
        xr = xf.clone()
        xr[:, out_ch] = 0.0
    else:
        xr = xf
    xq = _квант8(xr, _W8A8_SIM_G)
    if out_ch is not None:
        xq[:, out_ch] = xf[:, out_ch]
    return xq.to(x.dtype)


_АКТ_ФАЙЛ = os.environ.get("FA2SM80_ACT_STATS", "")
_АКТ_КАЖДЫЙ = int(os.environ.get("FA2SM80_ACT_STATS_EVERY", "37"))
_АКТ_СЧЁТ = {}


def _акт_статистика(x, W, K, N):
    import json
    import torch
    n = _АКТ_СЧЁТ.get((K, N), 0) + 1
    _АКТ_СЧЁТ[(K, N)] = n
    if n % _АКТ_КАЖДЫЙ != 1:
        return
    xf = x.float()
    y = xf @ W.float().t()
    ny = y.norm().item() + 1e-30
    am = xf.abs().amax(0)                                  # максимум по каналу
    order = torch.argsort(am, descending=True)
    rec = {"K": K, "N": N, "M": int(x.shape[0]), "call": n,
           "amax_top": [round(v, 3) for v in am[order[:8]].tolist()],
           "amax_med": round(am.median().item(), 4), "rel": {}}
    for T in (0, 8, 32, 128):
        out_ch = order[:T]
        xr = xf.clone()
        if T:
            xr[:, out_ch] = 0.0
        xq = _квант8(xr, 0)                                               # поточечный масштаб
        if T:
            xq[:, out_ch] = xf[:, out_ch].half().float()                  # выбросы -- в fp16
        yq = xq @ W.float().t()
        rec["rel"][str(T)] = round(((yq - y).norm().item()) / ny, 6)
    for G, T in ((128, 0), (64, 0), (32, 0), (128, 8)):                    # масштаб на строку и группу G
        xr = xf.clone()
        if T:
            xr[:, order[:T]] = 0.0
        xq = _квант8(xr, G)
        if T:
            xq[:, order[:T]] = xf[:, order[:T]].half().float()
        yq = xq @ W.float().t()
        rec["rel"][f"g{G}t{T}"] = round(((yq - y).norm().item()) / ny, 6)
    yh = torch.mm(x.half(), W.t()).float()                                # контроль: шум боевого fp16-пути (выход в fp16)
    rec["rel_fp16_ref"] = round(((yh - y).norm().item()) / ny, 7)
    with open(_АКТ_ФАЙЛ, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _лог(msg):
    if _ЛОГ:
        print(f"[fa2_sm80 W8] {msg}", file=sys.stderr, flush=True)


def _расш():
    if _расширение[0] is None:
        try:
            import fa2_sm70._ext as E
        except ImportError:
            sys.path.insert(0, os.environ.get("FA2SM70_PATH", "."))
            import fa2_sm70._ext as E
        _расширение[0] = E.w8a16_sm80_ext()
    return _расширение[0]


def _корзина(m):
    for c in _КОРЗИНЫ:
        if m <= c:
            return c
    return _КОРЗИНЫ[-1]


def _время(fn, повт=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    лучшее = 1e9
    for _ in range(3):
        e0.record()
        for _ in range(повт):
            fn()
        e1.record()
        torch.cuda.synchronize()
        лучшее = min(лучшее, e0.elapsed_time(e1) / повт)
    return лучшее * 1e3


# КЭШ ПОДБОРА НА ДИСКЕ. Подбор по замеру недетерминирован: при близких временах побеждают разные
# (warps, stages, splits) -- от пуска к пуску и даже между рангами одного пуска (сверка 23.10: два
# пуска разошлись на 6 формах из 14). Другой split-K = другой порядок суммирования, и жадный текст
# двух пусков расходится через 34-107 токенов при том, что каждое ядро верно. Поэтому первый пуск
# пишет выбор в файл по форме, все следующие и ОБА ранга берут его оттуда: числа воспроизводимы.
# FA2SM80_W8_TUNE_DIR (умолчание ~/.cache/vllm-sm80/w8tune), пустая строка -- без кэша.
_КЭШ_ПОДБОРА = os.environ.get("FA2SM80_W8_TUNE_DIR", os.path.expanduser("~/.cache/vllm-sm80/w8tune"))


def _кэш_подбора(K, N):
    if not _КЭШ_ПОДБОРА:
        return None
    имя = torch.cuda.get_device_name().replace(" ", "_").replace("/", "_")
    return os.path.join(_КЭШ_ПОДБОРА, f"{имя}_K{K}_N{N}_v2.json")


def _подобрать(qw, qz, sc, v2, K, N):
    """Подбор на ПЕРВОМ слое формы: nw упаковки + (warps, stages, splits) на корзину M."""
    ext = _расш()
    dev = qw.device
    путь = _кэш_подбора(K, N)
    if путь and os.path.exists(путь):
        try:
            import json
            д = json.load(open(путь))
            итог = {"nw": int(д["nw"]), "cfg": {int(b): tuple(c) for b, c in д["cfg"].items()},
                    "mmax": int(д["mmax"]),
                    "dcfg": {int(b): tuple(c) for b, c in д.get("dcfg", {}).items()},
                    "route": {int(b): r for b, r in д.get("route", {}).items()}}
            _лог(f"подбор K={K} N={N}: из кэша {путь}: nw={итог['nw']} {итог['cfg']}")
            wf, meta = ext.w8a16_sm80_pack(qw, qz, sc, 128, v2, итог["nw"])
            _рабочая_область(ext, wf, meta, K, N, итог["nw"])
            return итог, (wf, meta)
        except Exception as ex:  # noqa: BLE001
            _лог(f"кэш подбора {путь} не читается ({ex}) -- подбираю заново")
    итог = {"nw": None, "cfg": {}}
    лучшее_nw = (1e18, None)
    упаковки = {}
    for nw in _ИНСТАНСЫ:
        if N % (8 * nw):
            continue
        упаковки[nw] = ext.w8a16_sm80_pack(qw, qz, sc, 128, v2, nw)
    замеры = {}
    for nw, (wf, meta) in упаковки.items():
        for b in _КОРЗИНЫ:
            x = torch.randn(b, K, device=dev, dtype=torch.float16) * 0.1
            for (wa, st) in _ИНСТАНСЫ[nw]:
                for sp in (0, 1, 2, 4, 8):
                    try:
                        t = _время(lambda: ext.w8a16_sm80_mm(x, wf, meta, N, sp, nw, wa, st))
                    except RuntimeError:
                        continue          # разделяемой не хватает при этой ширине x
                    k = (nw, b)
                    if k not in замеры or t < замеры[k][0]:
                        замеры[k] = (t, (wa, st, sp))
    # nw выбирается по сумме лучших времён корзин 4 и 16 (декод с MTP и малые пачки)
    for nw in упаковки:
        s = sum(замеры[(nw, b)][0] for b in (4, 16) if (nw, b) in замеры)
        if s < лучшее_nw[0]:
            лучшее_nw = (s, nw)
    nw = лучшее_nw[1]
    итог["nw"] = nw
    рабочая = None
    for b in _КОРЗИНЫ:
        if (nw, b) in замеры:
            итог["cfg"][b] = замеры[(nw, b)][1]
            рабочая = b
    итог["mmax"] = рабочая            # шире -- кусками по этой корзине (разделяемая не вмещает x)
    итог["dcfg"], итог["route"] = {}, {}
    if N % 128 == 0 and N <= _ПРЯМ_NMAX:
        wf, meta = упаковки[nw]
        _рабочая_область(ext, wf, meta, K, N, nw)
        for b in _ПРЯМ_КОРЗ:
            x = torch.randn(b, K, device=dev, dtype=torch.float16) * 0.1
            лучш = None
            for mi in (1, 2, 4):
                if mi > 1 and 32 * mi > b:
                    continue
                for sp in (1, 2, 3, 4, 6, 8):
                    if sp * b > 1024:
                        continue
                    t = _время(lambda: ext.w8a16_sm80_gemm_large(x, wf, meta, N, nw, mi, sp))
                    if лучш is None or t < лучш[0]:
                        лучш = (t, (mi, sp))
            итог["dcfg"][b] = лучш[1]
            if b <= _MMAX:
                малое = замеры.get((nw, b), (1e18,))[0]
                итог["route"][b] = "d" if лучш[0] < малое else "s"
        _лог(f"прямой K={K} N={N}: " + " ".join(f"M<={b}:{итог['dcfg'][b]}" for b in _ПРЯМ_КОРЗ)
             + " маршрут " + str(итог["route"]))
    _лог(f"подбор K={K} N={N}: nw={nw} " + " ".join(
        f"M<={b}:{замеры[(nw, b)][1]}={замеры[(nw, b)][0]:.1f}us({K * N / замеры[(nw, b)][0] / 1e3:.0f}ГБ/с)"
        for b in _КОРЗИНЫ if (nw, b) in замеры))
    if путь:
        try:
            import json
            os.makedirs(os.path.dirname(путь), exist_ok=True)
            врем = f"{путь}.{os.getpid()}"
            json.dump({"nw": nw, "cfg": {str(b): list(c) for b, c in итог["cfg"].items()},
                       "mmax": итог["mmax"],
                       "dcfg": {str(b): list(c) for b, c in итог["dcfg"].items()},
                       "route": {str(b): r for b, r in итог["route"].items()}}, open(врем, "w"))
            if not os.path.exists(путь):
                os.replace(врем, путь)    # первый записавший ранг задаёт выбор для всех следующих пусков
            else:
                os.remove(врем)
        except Exception as ex:  # noqa: BLE001
            _лог(f"кэш подбора не записан: {ex}")
    return итог, упаковки[nw]


def _рабочая_область(ext, wf, meta, K, N, nw):
    """Рабочая область split-K растёт только вверх (старые буферы держатся ради графов) -- поэтому её
    размер берётся СРАЗУ максимальным для формы (splits*M <= 1024) одним пуском, до захвата графов."""
    if N % 128 or N > _ПРЯМ_NMAX:
        return
    x = torch.zeros(512, K, device=wf.device, dtype=torch.float16)
    try:
        ext.w8a16_sm80_gemm_large(x, wf, meta, N, nw, 4, 2)
    except RuntimeError:
        pass


def _буфер_для(dev, n):
    b = _буфер.get(dev)
    if b is None or b.numel() < n:
        _буфер[dev] = torch.empty(n, dtype=torch.float16, device=dev)
    return _буфер[dev]


def установить():
    if not _ВКЛ:
        return False
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        _лог("не sm_80+: подмены нет")
        return False
    from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod
    from vllm.utils.torch_utils import direct_register_custom_op

    _чужой_обработать = GPTQLinearMethod.process_weights_after_loading
    _чужой_apply = GPTQLinearMethod.apply

    _зонд = [None]

    def process_weights_after_loading(self, layer):
        _чужой_обработать(self, layer)
        if _зонд[0] is None:
            # gptq_shuffle движка на 8 битах обязан быть тождественным: наш упаковщик читает ИСХОДНУЮ
            # раскладку GPTQ. Проверяем исполнением (тот же зонд, что у вольтовского пути).
            from fa2sm70_w8 import _perestanovka_tozhdestvennaya
            _зонд[0] = _perestanovka_tozhdestvennaya()
            _лог(f"зонд gptq_shuffle: {'ТОЖДЕСТВЕН' if _зонд[0] else 'НЕ тождествен -- подмена ОТКЛЮЧЕНА'}")
        if not _зонд[0]:
            return
        qw = layer.qweight
        K, N = qw.shape[0] * 4, qw.shape[1]
        годен = (self.quant_config.weight_bits == 8 and layer.g_idx.numel() == 0
                 and int(self.quant_config.group_size) == 128 and K % 64 == 0 and N % 16 == 0)
        if not годен:
            _лог(f"слой {tuple(qw.shape)} не годен (бит={self.quant_config.weight_bits}, g_idx={layer.g_idx.numel()}) -- чужое тело")
            return
        v2 = bool(self.use_v2_format)
        ключ = (K, N)
        if ключ not in _ТЮН:
            итог, (wf, meta) = _подобрать(qw, layer.qzeros, layer.scales, v2, K, N)
            _ТЮН[ключ] = итог
        else:
            wf, meta = _расш().w8a16_sm80_pack(qw, layer.qzeros, layer.scales, 128, v2, _ТЮН[ключ]["nw"])
        layer.fa2sm80_wf = wf
        layer.fa2sm80_meta = meta
        layer.fa2sm80_kn = (K, N)
        # исходный вес больше не нужен: те же K*N байт уже лежат в wf
        layer.qweight = torch.nn.Parameter(torch.empty(0, N, dtype=qw.dtype, device=qw.device), requires_grad=False)
        del qw
        if K * N <= _БУФ_МАКС:
            _буфер_для(wf.device, K * N)   # общий буфер под деквант больших M -- до профилирования памяти
        _С["слоёв"] += 1
        _С["байт"] += K * N
        if _С["слоёв"] in (1, 8, 64, 128, 256):
            _лог(f"перепаковано слоёв: {_С['слоёв']}, {_С['байт'] / 2**30:.2f} ГиБ")

    GPTQLinearMethod.process_weights_after_loading = process_weights_after_loading

    # МАЛЫЕ ПЛОТНЫЕ fp16-ПРОЕКЦИИ (in_proj_ba у GDN: K=5120, N=48 на ранг) при M<=8 -- своим ядром: один узел
    # графа ~3 мкс вместо пары cuBLAS (GEMM splitK + reduce, 8.7 мкс) на каждый из 48 слоёв GDN шага.
    # Выбор по M -- ВНУТРИ операции (форма партии символьна при компиляции). FA2SM80_LIN_SMALL=0 -- выкл.
    if os.environ.get("FA2SM80_LIN_SMALL", "1") == "1":
        import torch.nn.functional as _F
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        def _лин(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            m = x.numel() // x.shape[-1]
            if 0 < m <= 8 and x.dtype == torch.float16 and x.stride(-1) == 1:
                xc = x.reshape(m, x.shape[-1])
                if xc.stride(0) % 8 == 0 and xc.data_ptr() % 16 == 0:
                    return _расш().lin_small_f16(xc, w).view(*x.shape[:-1], w.shape[0])
            return _F.linear(x, w)

        def _лин_fake(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return x.new_empty((*x.shape[:-1], w.shape[0]))

        direct_register_custom_op("fa2sm80_lin_small", _лин, mutates_args=[], fake_impl=_лин_fake)
        _чужой_лин = UnquantizedLinearMethod.apply

        def _лин_apply(self, layer, x, bias=None):
            w = getattr(layer, "weight", None)
            if (bias is None and w is not None and w.dtype == torch.float16 and w.dim() == 2
                    and w.shape[0] <= 256 and w.shape[1] % 8 == 0 and w.shape[1] <= 8192
                    and w.is_contiguous() and x.dtype == torch.float16):
                return torch.ops.vllm.fa2sm80_lin_small(x, w)
            return _чужой_лин(self, layer, x, bias)

        UnquantizedLinearMethod.apply = _лин_apply
        _лог("малые плотные fp16-проекции (N<=256, M<=8) -- своим ядром lin_small_f16")

    # fp32 ДЛЯ ПАРАМЕТРОВ ГЕЙТИНГА GDN (A_log, dt_bias). Модель создаёт их в dtype развёртки (fp16), а наш
    # рекуррент берёт fp32: `.float()` в каждом слое давал 2 узла графа декода на каждый из 48 слоёв GDN
    # (96 крошечных копий на шаг). Переводим один раз после загрузки весов; Triton-гейтинг префилла fp32 берёт.
    # FA2SM80_GDN_F32_PARAMS=0 -- выкл.
    if os.environ.get("FA2SM80_GDN_F32_PARAMS", "1") == "1":
        try:
            from vllm.model_executor.model_loader.base_loader import BaseModelLoader
            _чужой_load = BaseModelLoader.load_model

            def _load_f32(self, *a, **kw):
                model = _чужой_load(self, *a, **kw)
                n = 0
                for _, m in model.named_modules():
                    for имя in ("A_log", "dt_bias"):
                        p = getattr(m, имя, None)
                        if (isinstance(p, torch.nn.Parameter) and p.dtype in (torch.float16, torch.bfloat16)
                                and hasattr(m, "A_log") and hasattr(m, "dt_bias")):
                            p.data = p.data.float()
                            n += 1
                if n:
                    _лог(f"параметры гейтинга GDN переведены в fp32: {n}")
                return model

            BaseModelLoader.load_model = _load_f32
        except Exception as ex:  # noqa: BLE001
            _лог(f"перевод A_log/dt_bias в fp32 не поставлен: {ex}")

    def _w8_mm(x: torch.Tensor, wf: torch.Tensor, meta: torch.Tensor, sc: torch.Tensor,
               K: int, N: int) -> torch.Tensor:
        ext = _расш()
        m = x.shape[0]
        т = _ТЮН[(K, N)]
        if _ПРЯМ_MIDM and m <= _MMAX and т["route"].get(_корзина(m)) == "d":
            _С["малое"] += 1
            mi, sp = т["dcfg"][_корзина(m)]
            return ext.w8a16_sm80_gemm_large(x, wf, meta, N, т["nw"], mi, sp)
        if m <= _MMAX:
            _С["малое"] += 1
            мм = т["mmax"]
            if m > мм:
                wa, st, sp = т["cfg"][мм]
                return torch.cat([ext.w8a16_sm80_mm(x[i:i + мм], wf, meta, N, sp, т["nw"], wa, st)
                                  for i in range(0, m, мм)], 0)
            wa, st, sp = т["cfg"][_корзина(m)]
            return ext.w8a16_sm80_mm(x, wf, meta, N, sp, т["nw"], wa, st)
        _С["большое"] += 1
        if m <= _ПРЯМОЙ_ММАКС and N % 128 == 0:
            # прямой Ampere-GEMM (cp.async 3 стадии + деквант в регистрах): при M=256 быстрее деквант+cuBLAS
            # (замер bench_w8_large: 61-89 TF против 47-71), при M>=1024 пока медленнее (~100 против ~167 TF)
            dc = т["dcfg"]
            b = next((c for c in _ПРЯМ_КОРЗ if c >= m), None)
            if _ПРЯМ_MIDM and b is not None and b in dc:
                mi, sp = dc[b]
                return ext.w8a16_sm80_gemm_large(x, wf, meta, N, т["nw"], mi, sp)
            return ext.w8a16_sm80_gemm_large(x, wf, meta, N, т["nw"], 4, 1)
        if K * N > _БУФ_МАКС:
            # словарь (124160x5120): буфер fp16 вышел бы 1.27 ГБ -- идём ядром кусками
            мм = т["mmax"]
            wa, st, sp = т["cfg"][мм]
            return torch.cat([ext.w8a16_sm80_mm(x[i:i + мм], wf, meta, N, sp, т["nw"], wa, st)
                              for i in range(0, m, мм)], 0)
        W = ext.w8a16_sm80_dequant(wf, meta, N, K, т["nw"], _буфер_для(x.device, K * N))
        if _АКТ_ФАЙЛ:
            try:
                _акт_статистика(x, W.view(N, K), K, N)
            except Exception as ex:  # noqa: BLE001
                _лог(f"прибор активаций: {ex}")
        if _W8A8_SIM >= 0:
            x = _w8a8_сим(x, _W8A8_SIM)
        import fa2sm80_cg as CG
        if CG.включён() and x.is_contiguous():
            return CG.mm(x, W.view(N, K))      # свой GEMM: инвариантен к пачке
        return torch.mm(x, W.t())

    def _w8_mm_fake(x, wf, meta, sc, K: int, N: int):
        return x.new_empty((x.shape[0], N))

    from fa2sm70_clef_hook import с_поправкой
    direct_register_custom_op("fa2sm80_w8_mm", с_поправкой(_w8_mm, (3,)), mutates_args=[],
                              fake_impl=_w8_mm_fake)

    def apply(self, layer, x, bias=None):
        kn = getattr(layer, "fa2sm80_kn", None)
        if kn is None:
            return _чужой_apply(self, layer, x, bias)
        K, N = kn
        xr = x.reshape(-1, x.shape[-1])
        if xr.dtype != torch.float16:
            xr = xr.half()
        y = torch.ops.vllm.fa2sm80_w8_mm(xr, layer.fa2sm80_wf, layer.fa2sm80_meta, layer.scales, K, N)
        if bias is not None:
            y = y + bias
        return y.reshape(x.shape[:-1] + (N,))

    GPTQLinearMethod.apply = apply
    _лог("GPTQ int8 переведён на w8a16_sm80 (M<=64 -- ядро, больше -- деквант+cuBLAS)")
    return True


def упаковать_вне_gptq(qw, qz, sc, v2, K, N):
    """Для чужих модулей (словарь, MTP): подбор по форме + упаковка; вызывать вне захвата графов."""
    if (K, N) not in _ТЮН:
        итог, pack = _подобрать(qw, qz, sc, v2, K, N)
        _ТЮН[(K, N)] = итог
        return pack
    return _расш().w8a16_sm80_pack(qw, qz, sc, 128, v2, _ТЮН[(K, N)]["nw"])


def счётчики():
    return dict(_С)
