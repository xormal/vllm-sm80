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
_С = {"малое": 0, "большое": 0, "слоёв": 0, "байт": 0, "марлин": 0, "марлин_деквант": 0}
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
# [ЗАГРУЗКА ВЕСОВ 30.10] FA2SM80_W8_MFAST=1: прямой GEMM больших M (M > 64) -- плитки M первыми (веса из HBM один
# раз на вызов, а не на каждую полосу строк), MI=4 без спиллов, рабочая область split-K без роста. Подбор -- в
# (mi, splits) -- из того же кэша подбора, что у прежнего пути, поэтому выход прямого ядра побитно прежний. Корзины
# 1024/2048 сравниваются замером при загрузке с деквантом в fp16 + GEMM: маршрут "d" (прямой) или "q" (деквант) --
# свой кэш (_v2lr.json). FA2SM80_W8_MFAST_FILE=<файл> -- "0"/"1" из файла, перечитывается по СЧЁТЧИКУ вызовов
# больших M (раз в _MF_ПЕРИОД): оба ранга TP проходят одни и те же вызовы и переключаются на одном и том же.
_MFAST = os.environ.get("FA2SM80_W8_MFAST", "0") == "1"
_MFAST_ФАЙЛ = os.environ.get("FA2SM80_W8_MFAST_FILE", "")
_MF_ПЕРИОД = 256
_MF = {"n": 0, "вкл": _MFAST}
_ПРЯМ_КОРЗ_MF = (1024, 2048)


def _mfast():
    if _MFAST_ФАЙЛ:
        _MF["n"] += 1
        if _MF["n"] % _MF_ПЕРИОД == 1:
            try:
                with open(_MFAST_ФАЙЛ) as f:
                    _MF["вкл"] = f.read(1) == "1"
            except OSError:
                _MF["вкл"] = _MFAST
    return _MF["вкл"]
# [СРЕДНИЕ M 30.10] FA2SM80_W8_G2=1: корзины прямого пути 128/256/512 -- ядро v2 (w8a16_sm80_gemm2: блок 128 x 32*nj,
# один блок на SM, конвейер фрагментов), если замер при загрузке быстрее нынешнего (mi, splits) не менее чем на 5 % и
# выход ПОБИТНО тот же (то же расщепление k -- тот же порядок сложения; проверяется torch.equal при подборе). Профиль
# окна 12: шаг с записью Clef ~400 токенов -- прямой GEMM 112 мс из ~300. Выбор -- в кэш *_v2g2.json. Уровень >= 3 в
# FA2SM70_NOSYNC_FILE -- то же из файла (A/B в одном экземпляре; файл перечитывается по счётчику вызовов).
_G2 = os.environ.get("FA2SM80_W8_G2", "0") == "1"
_G2_ФАЙЛ = os.environ.get("FA2SM70_NOSYNC_FILE", "")
_G2_С = {"n": 0, "вкл": _G2}


def _g2_вкл():
    if _G2_ФАЙЛ:
        _G2_С["n"] += 1
        if _G2_С["n"] % _MF_ПЕРИОД == 1:
            try:
                with open(_G2_ФАЙЛ) as f:
                    v = f.read(1)
                _G2_С["вкл"] = v.isdigit() and int(v) >= 3
            except OSError:
                _G2_С["вкл"] = _G2
    return _G2_С["вкл"]


# [GEMM 30.10] FA2SM80_W8_G3=1: M > 64 -- ядро v3 (w8a16_sm80_gemm3: x в порядке фрагментов, плитка варпа 64x64, без
# MOV-сборки фрагмента A; стенд scripts/w8_g3_test.py). Подбор при загрузке на корзину M: (wm, wn, стадий, splits) v3
# против нынешнего маршрута корзины (прямое ядро при M <= 512, деквант + GEMM выше); кэш *_v2g3.json (первый записавший
# ранг). "win" -- корзины, где v3 быстрее (обычный маршрут), "all" -- лучшая конфигурация v3 на каждую корзину (конвейер
# блока при включённом v3 берёт её всегда: распаковка весов в fp16 и её буферы не нужны). FA2SM80_W8_G3_FILE=<файл> --
# "0"/"1" из файла, перечитывается по СЧЁТЧИКУ вызовов (оба ранга TP переключаются на одном и том же вызове).
_G3 = os.environ.get("FA2SM80_W8_G3", "0") == "1"
_G3_ФАЙЛ = os.environ.get("FA2SM80_W8_G3_FILE", "")
_G3_С = {"n": 0, "вкл": _G3}
_G3_КОРЗ = (128, 256, 512, 1024, 2048, 4096, 8192)
_G3_ИНСТ = ((2, 2, 3), (2, 2, 4), (1, 4, 3), (1, 4, 4), (2, 4, 3), (2, 4, 4), (4, 2, 3))


def g3_вкл():
    if not (_G3 or _G3_ФАЙЛ):
        return False
    if _G3_ФАЙЛ:
        _G3_С["n"] += 1
        if _G3_С["n"] % _MF_ПЕРИОД == 1:
            try:
                with open(_G3_ФАЙЛ) as f:
                    _G3_С["вкл"] = f.read(1) == "1"
            except OSError:
                _G3_С["вкл"] = _G3
    return _G3_С["вкл"]


def _g3_корзина(m):
    return next((c for c in _G3_КОРЗ if c >= m), _G3_КОРЗ[-1])


def _g3_выровнен(x):
    return x.dim() == 2 and x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0


def g3_слой(layer, x, out=None):
    """v3 для слоя W8 по конфигурации корзины ("all"); None -- для формы нет подбора v3."""
    K, N = layer.fa2sm80_kn
    т = _ТЮН.get((K, N), {})
    g3 = т.get("g3")
    if not g3:
        return None
    b = _g3_корзина(x.shape[0])
    cfg = g3["all"].get(b)
    if cfg is None:
        if not g3["all"]:
            return None
        cfg = g3["all"][min(g3["all"], key=lambda c: abs(c - b))]      # корзина без замера -- ближайшая
    if not _g3_выровнен(x):
        x = x.contiguous()
    _С["g3"] = _С.get("g3", 0) + 1
    wm, wn, st, sp = cfg
    return _расш().w8a16_sm80_gemm3x(x, layer.fa2sm80_wf, layer.fa2sm80_meta, N, т["nw"], wm, wn, st, sp, out)


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
# [25.10] Перестановка A/B для M<=8 (w8a16_sw_sm80.cuh): вес -- операнд A, прямо в регистры. Тот же раздел K между
# варпами -> побитно тот же выход (стенд bench/sa/w8sw_test). Таблица: (nw, warps) нынешней конфигурации -> (jw, d)
# лучшего побитного варианта; формы с warps=2 (qkvz, qkv внимания) новым ядром медленнее -- остаются прежним.
_W8SW = os.environ.get("FA2SM80_W8SW", "0") == "1"
_W8SW_ТАБЛ = {(4, 4): (2, 2), (4, 8): (2, 2), (2, 4): (2, 4)}   # словарь (N=124160) -- прежним: не быстрее
_W8SW_СВЕРКА = int(os.environ.get("FA2SM80_W8SW_CHECK", "0"))
_W8SW_СВЕРЕНО = {}
# [W8A8 29.10] FA2SM80_W8A8=off|clef|llm|all -- int8-активации на больших M (fa2sm80_w8a8); off -- модуль не трогается
import fa2sm80_w8a8 as _W8A8
if not _W8A8.включён():
    _W8A8 = None


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


# [ФРИЗЫ 30.10] FA2SM80_W8_MARLIN=1: вес -- в раскладке Marlin (marlin8_sm80.cu: шаблон Marlin с нулями kU8 в нашей
# сборке) ВМЕСТО нашей фрагментной (те же K*N байт). M <= FA2SM80_W8_MARLIN_MMAX -- marlin8_gemm; больше -- деквант из
# раскладки Marlin (ПОБИТНО как прежний w8a16_sm80_dequant) + наш cuBLAS-путь, конвейер блока -- тем же деквантом.
# Окно 23 (стенд marlin8_test.py против боевого маршрута): M=40 1.91x, 64 1.68x, 128-440 1.36-1.44x, 1024 1.07x,
# 2088 0.90x -- отсюда порог 1024. Точность та же (отн. ошибка 2.07e-4 против того же эталона, что у нашего пути).
_MARLIN = os.environ.get("FA2SM80_W8_MARLIN", "0") == "1"
_MARLIN_MMAX = int(os.environ.get("FA2SM80_W8_MARLIN_MMAX", "1024"))
_м8_расш = [None]
# [ФРИЗЫ 30.10] Раскладка Marlin -- признак КОНКРЕТНОГО веса (слоя), а не формы (K, N): слой MTP (FA2SM70_MTP_W8) и словарь
# пакуются НАШЕЙ раскладкой (упаковать_вне_gptq) и зовут ту же операцию с теми же (K, N), что MLP модели. Окно 25/25b:
# признак по форме в _ТЮН отправлял веса MTP в Marlin -> мусорные логиты черновика -> illegal memory access в черновике.
_МАРЛИН_ПТР = set()          # data_ptr весов в раскладке Marlin
# FA2SM80_W8_MARLIN_EXT=1 (при FA2SM80_W8_MARLIN): и чужие модули -- слой MTP (FA2SM70_MTP_W8, каждый шаг декода при
# M 8-40, где Marlin ~1.9x) и словарь (FA2SM70_LMH) -- пакуются в раскладку Marlin; оба зовут только fa2sm80_w8_mm.
_MARLIN_ВНЕ = os.environ.get("FA2SM80_W8_MARLIN_EXT", "0") == "1"


def _м8():
    if _м8_расш[0] is None:
        import fa2_sm70._ext as E
        _м8_расш[0] = E.marlin8_ext()
    return _м8_расш[0]


def _марлин_части(meta, K, N):
    """meta в режиме Marlin -- [G, N] int32 (ТА ЖЕ форма, что у нашей meta -- окно 24b: скомпилированный граф из кэша
    проверяет формы весов слоёв), внутри подряд: масштабы fp16 [G, N] (перестановка Marlin), нули [G, N/4], хвост."""
    G = K // 128
    плоск = meta.view(-1)
    return плоск[: G * N // 2].view(torch.float16).view(G, N), плоск[G * N // 2: 3 * G * N // 4].view(G, N // 4)


def _марлин_вес(wf, K):
    """Вес Marlin хранится одномерным [N*K/4] (форма нашего wf); ядру -- вид [K/16, 4N]."""
    return wf.view(K // 16, -1)


def марлин_слой(layer, x, out=None):
    """[ФРИЗЫ 30.10] y = x @ W^T слоя в раскладке Marlin прямо marlin8_gemm (конвейер блока на средних M: без декванта)."""
    K, N = layer.fa2sm80_kn
    ms, mz = _марлин_части(layer.fa2sm80_meta, K, N)
    if not (x.stride(-1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0):
        x = x.contiguous()
    _С["марлин"] += 1
    return _м8().marlin8_gemm(x, _марлин_вес(layer.fa2sm80_wf, K), ms, mz, N, True, out)


def марлин_ли(layer):
    return bool(getattr(layer, "fa2sm80_marlin", False))


def деквант_слоя(layer, buf):
    """W [N, K] fp16 слоя в общий буфер -- в любой раскладке (конвейер блока/строк)."""
    K, N = layer.fa2sm80_kn
    if марлин_ли(layer):
        ms, mz = _марлин_части(layer.fa2sm80_meta, K, N)
        return _м8().marlin8_dequant(_марлин_вес(layer.fa2sm80_wf, K), ms, mz, N, K, buf)
    return _расш().w8a16_sm80_dequant(layer.fa2sm80_wf, layer.fa2sm80_meta, N, K, _ТЮН[(K, N)]["nw"], buf)


def _в_марлин(layer, qw, v2, K, N):
    """Раскладка Marlin из GPTQ: вес (gptq_marlin_repack), масштабы (marlin_permute_scales), нули (распакованы из
    qzeros: 4 на int32 по столбцам, младший байт первый; v1 -- нуль+1, как zbias у meta_kernel; marlin_zero_points)."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales, marlin_zero_points
    gi = torch.empty(0, dtype=torch.int, device=qw.device)
    mq = ops.gptq_marlin_repack(qw.contiguous(), gi, K, N, 8)
    ms = marlin_permute_scales(layer.scales.contiguous(), K, N, 128)
    qz = layer.qzeros
    z = torch.stack([(qz >> (8 * i)) & 0xFF for i in range(4)], -1).reshape(K // 128, N).int()
    if not v2:
        z = z + 1
    if int(z.max()) > 255:
        return None
    mz = marlin_zero_points(z, K // 128, N, 8).to(qw.device)
    G = K // 128
    meta = torch.zeros(G * N, dtype=torch.int32, device=qw.device)
    meta[: G * N // 2].copy_(ms.reshape(-1).view(torch.int32))
    meta[G * N // 2: 3 * G * N // 4].copy_(mz.reshape(-1))
    return mq.reshape(-1), meta.view(G, N)


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


def _маршрут_больших(ext, wf, meta, K, N, nw, dev):
    """[ЗАГРУЗКА ВЕСОВ 30.10] Корзины 1024/2048: прямой GEMM (mfast, MI=4) против декванта в fp16 + GEMM -- замером.
    Выбор пишется в свой кэш: первый записавший ранг задаёт его для следующих пусков (как у подбора)."""
    import json
    import fa2sm80_cg as CG
    путь = _кэш_подбора(K, N)
    путь = путь[:-len("_v2.json")] + "_v2lr.json" if путь else None
    if путь and os.path.exists(путь):
        try:
            итог = {int(b): r for b, r in json.load(open(путь)).items()}
            _лог(f"маршрут больших M K={K} N={N}: из кэша {итог}")
            return итог
        except Exception as ex:  # noqa: BLE001
            _лог(f"кэш маршрута {путь} не читается ({ex}) -- замеряю")
    итог = {}
    for b in _ПРЯМ_КОРЗ_MF:
        x = torch.randn(b, K, device=dev, dtype=torch.float16) * 0.1

        def _деквант():
            W = ext.w8a16_sm80_dequant(wf, meta, N, K, nw, _буфер_для(dev, K * N))
            return CG.mm(x, W.view(N, K)) if CG.включён() else torch.mm(x, W.view(N, K).t())

        try:
            tп = _время(lambda: ext.w8a16_sm80_gemm_large(x, wf, meta, N, nw, 4, 1, 1), повт=5)
            tq = _время(_деквант, повт=5)
        except RuntimeError as ex:
            _лог(f"маршрут больших M K={K} N={N} M={b}: замер не удался ({ex}) -- деквант")
            итог[b] = "q"
            continue
        итог[b] = "d" if tп < tq else "q"
        _лог(f"маршрут больших M K={K} N={N} M<={b}: прямой {tп:.0f} мкс, деквант+GEMM {tq:.0f} мкс -> {итог[b]}")
    if путь:
        try:
            врем = f"{путь}.{os.getpid()}"
            json.dump({str(b): r for b, r in итог.items()}, open(врем, "w"))
            if not os.path.exists(путь):
                os.replace(врем, путь)
            else:
                os.remove(врем)
        except Exception as ex:  # noqa: BLE001
            _лог(f"кэш маршрута не записан: {ex}")
    return итог


def _маршрут_g2(ext, wf, meta, K, N, nw, dev, dcfg):
    """[СРЕДНИЕ M 30.10] Корзины 128/256/512 прямого пути: v2 при том же расщеплении k -- только побитно равные и
    быстрее нынешнего на 5 %+. {корзина: (nj, стадий, pin, splits)}; кэш *_v2g2.json (первый записавший ранг)."""
    import json
    путь = _кэш_подбора(K, N)
    путь = путь[:-len("_v2.json")] + "_v2g2.json" if путь else None
    if путь and os.path.exists(путь):
        try:
            итог = {int(b): tuple(r) for b, r in json.load(open(путь)).items()}
            _лог(f"v2 средних M K={K} N={N}: из кэша {итог}")
            return итог
        except Exception as ex:  # noqa: BLE001
            _лог(f"кэш v2 {путь} не читается ({ex}) -- замеряю")
    итог = {}
    if not hasattr(ext, "w8a16_sm80_gemm2") or K % 128:
        return итог
    for b in (128, 256, 512):
        if b not in dcfg:
            continue
        mi, sp = dcfg[b]
        x = torch.randn(b, K, device=dev, dtype=torch.float16) * 0.1
        try:
            эт = ext.w8a16_sm80_gemm_large(x, wf, meta, N, nw, mi, sp)
            t0 = _время(lambda: ext.w8a16_sm80_gemm_large(x, wf, meta, N, nw, mi, sp), повт=5)
        except RuntimeError as ex:
            _лог(f"v2 средних M K={K} N={N} M={b}: замер прежнего не удался ({ex})")
            continue
        лучш = None
        for nj in (8, 4):
            if N % (32 * nj) or N % (8 * nw):
                continue
            for stg in (3, 4):
                for pin in (False, True):
                    try:
                        y = ext.w8a16_sm80_gemm2(x, wf, meta, N, nw, nj, stg, sp, pin)
                        if not torch.equal(y, эт):
                            continue
                        t = _время(lambda: ext.w8a16_sm80_gemm2(x, wf, meta, N, nw, nj, stg, sp, pin), повт=5)
                    except RuntimeError:
                        continue
                    if лучш is None or t < лучш[0]:
                        лучш = (t, (nj, stg, pin, sp))
        if лучш is not None and лучш[0] < 0.95 * t0:
            итог[b] = лучш[1]
        _лог(f"v2 средних M K={K} N={N} M<={b}: прежний ({mi},{sp}) {t0:.0f} мкс, v2 "
             f"{'%.0f мкс %s' % лучш if лучш else 'нет побитно равного'} -> {'v2' if b in итог else 'прежний'}")
    if путь:
        try:
            врем = f"{путь}.{os.getpid()}"
            json.dump({str(b): list(r) for b, r in итог.items()}, open(врем, "w"))
            if not os.path.exists(путь):
                os.replace(врем, путь)
            else:
                os.remove(врем)
        except Exception as ex:  # noqa: BLE001
            _лог(f"кэш v2 не записан: {ex}")
    return итог


def _маршрут_g3(ext, wf, meta, K, N, nw, dev, dcfg):
    """[GEMM 30.10] Корзины 128..8192: лучший v3 против нынешнего маршрута (прямое ядро (mi, splits) из dcfg при M <= 512,
    иначе деквант в fp16 + GEMM). {"win": {b: cfg}, "all": {b: cfg}, "t": {b: [v3, нынешний, GEMM без распаковки]}} (мкс);
    кэш *_v2g3.json (первый записавший ранг задаёт выбор для следующих пусков и обоих рангов)."""
    import json
    import fa2sm80_cg as CG
    путь = _кэш_подбора(K, N)
    путь = путь[:-len("_v2.json")] + "_v2g3.json" if путь else None
    if путь and os.path.exists(путь):
        try:
            д = json.load(open(путь))
            итог = {к: {int(b): (tuple(v) if к != "t" else list(v)) for b, v in д[к].items()} for к in ("win", "all", "t")}
            _лог(f"v3 K={K} N={N}: из кэша, выигрывает в корзинах {sorted(итог['win'])}")
            return итог
        except Exception as ex:  # noqa: BLE001
            _лог(f"кэш v3 {путь} не читается ({ex}) -- замеряю")
    итог = {"win": {}, "all": {}, "t": {}}
    if not hasattr(ext, "w8a16_sm80_gemm3x") or K % 128 or N % 128:
        return итог
    W = None
    if K * N <= _БУФ_МАКС:
        W = torch.empty(N * K, dtype=torch.float16, device=dev)          # временный: после замера освобождается
    for b in _G3_КОРЗ:
        x = torch.randn(b, K, device=dev, dtype=torch.float16) * 0.1
        лучш = None
        for wm, wn, st in _G3_ИНСТ:
            if N % (64 * wn):
                continue
            for sp in ((1, 2, 3, 4) if b <= 512 else (1,)):
                if sp * b > 1024 and sp > 1:
                    continue
                try:
                    t = _время(lambda: ext.w8a16_sm80_gemm3x(x, wf, meta, N, nw, wm, wn, st, sp), повт=5)
                except RuntimeError:
                    continue
                if лучш is None or t < лучш[0]:
                    лучш = (t, (wm, wn, st, sp))
        if лучш is None:
            continue
        итог["all"][b] = лучш[1]
        t_cg = 0.0
        try:
            if b <= _ПРЯМОЙ_ММАКС and b in dcfg:
                mi, sp = dcfg[b]
                t_ref = _время(lambda: ext.w8a16_sm80_gemm_large(x, wf, meta, N, nw, mi, sp), повт=5)
            elif W is not None:
                Wv = ext.w8a16_sm80_dequant(wf, meta, N, K, nw, W).view(N, K)
                умн = (lambda: CG.mm(x, Wv)) if CG.включён() else (lambda: torch.mm(x, Wv.t()))
                t_cg = _время(умн, повт=5)
                t_ref = _время(lambda: (ext.w8a16_sm80_dequant(wf, meta, N, K, nw, W), умн()), повт=5)
            else:
                t_ref = 1e18
        except RuntimeError as ex:
            _лог(f"v3 K={K} N={N} M={b}: замер нынешнего не удался ({ex})")
            t_ref = 1e18
        итог["t"][b] = [round(лучш[0], 1), round(t_ref, 1), round(t_cg, 1)]
        if лучш[0] < 0.97 * t_ref:
            итог["win"][b] = лучш[1]
        _лог(f"v3 K={K} N={N} M<={b}: v3 {лучш[0]:.0f} мкс {лучш[1]}, нынешний {t_ref:.0f} мкс"
             + (f" (GEMM без распаковки {t_cg:.0f})" if t_cg else "") + f" -> {'v3' if b in итог['win'] else 'нынешний'}")
        del x
    del W
    torch.cuda.empty_cache()
    if путь:
        try:
            врем = f"{путь}.{os.getpid()}"
            json.dump({к: {str(b): list(v) for b, v in итог[к].items()} for к in ("win", "all", "t")}, open(врем, "w"))
            if not os.path.exists(путь):
                os.replace(врем, путь)
            else:
                os.remove(врем)
        except Exception as ex:  # noqa: BLE001
            _лог(f"кэш v3 не записан: {ex}")
    return итог


def _g3_резерв(ext, K, N, dev):
    """Буфер упаковки x (текущий поток) под наибольшую корзину и рабочая область split-K -- до профилирования и захвата."""
    if hasattr(ext, "w8a16_sm80_g3_reserve"):
        ext.w8a16_sm80_g3_reserve(torch.empty(1, device=dev), (K // 64) * (_G3_КОРЗ[-1] // 16) * 512, 1024 * N, 1 << 16)


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
        if _MARLIN and K % 128 == 0 and N % 64 == 0:      # форма -- одинакова на рангах TP: решение симметрично
            пара = _в_марлин(layer, qw, v2, K, N)
            if пара is None:
                # [ревизия] отказ по ЗНАЧЕНИЯМ нулей своего шарда -- у другого ранга TP его может не быть: разная
                # раскладка дала бы разные пути конвейера (разное число сведений) -- зависание. Лучше не стартовать.
                raise RuntimeError(f"FA2SM80_W8_MARLIN: слой K={K} N={N} -- нуль+1 > 255 (GPTQ v1), Marlin невозможен; "
                                   "снимите FA2SM80_W8_MARLIN")
            if пара is not None:
                mq, meta = пара
                layer.fa2sm80_wf = mq
                layer.fa2sm80_meta = meta
                layer.fa2sm80_kn = (K, N)
                layer.fa2sm80_marlin = True
                _МАРЛИН_ПТР.add(mq.data_ptr())
                layer.qweight = torch.nn.Parameter(torch.empty(0, N, dtype=qw.dtype, device=qw.device),
                                                   requires_grad=False)
                del qw
                # _ТЮН не трогаем: им пользуются веса НАШЕЙ раскладки той же формы (MTP, упаковать_вне_gptq)
                _м8().marlin8_reserve(meta)
                if K * N <= _БУФ_МАКС:
                    _буфер_для(mq.device, K * N)
                _С["слоёв"] += 1
                _С["байт"] += K * N
                if _С["слоёв"] in (1, 8, 64, 128, 256):
                    _лог(f"Marlin: перепаковано слоёв {_С['слоёв']}, {_С['байт'] / 2**30:.2f} ГиБ")
                return
        if ключ not in _ТЮН:
            итог, (wf, meta) = _подобрать(qw, layer.qzeros, layer.scales, v2, K, N)
            if ((_MFAST or _MFAST_ФАЙЛ) and K * N <= _БУФ_МАКС and N % 128 == 0 and N <= _ПРЯМ_NMAX
                    and итог.get("dcfg")):
                итог["lroute"] = _маршрут_больших(_расш(), wf, meta, K, N, итог["nw"], wf.device)
            if ((_G2 or _G2_ФАЙЛ) and _ПРЯМ_MIDM and N % 128 == 0 and N <= _ПРЯМ_NMAX and итог.get("dcfg")):
                итог["g2"] = _маршрут_g2(_расш(), wf, meta, K, N, итог["nw"], wf.device, итог["dcfg"])
            if (_G3 or _G3_ФАЙЛ) and N % 128 == 0 and N <= _ПРЯМ_NMAX and K % 128 == 0:
                итог["g3"] = _маршрут_g3(_расш(), wf, meta, K, N, итог["nw"], wf.device, итог.get("dcfg", {}))
                _g3_резерв(_расш(), K, N, wf.device)
            _ТЮН[ключ] = итог
        else:
            wf, meta = _расш().w8a16_sm80_pack(qw, layer.qzeros, layer.scales, 128, v2, _ТЮН[ключ]["nw"])
        layer.fa2sm80_wf = wf
        layer.fa2sm80_meta = meta
        layer.fa2sm80_kn = (K, N)
        # исходный вес больше не нужен: те же K*N байт уже лежат в wf
        layer.qweight = torch.nn.Parameter(torch.empty(0, N, dtype=qw.dtype, device=qw.device), requires_grad=False)
        del qw
        g3 = _ТЮН[ключ].get("g3")
        # [GEMM 30.10] v3 без переключателя-файла и выигрывает во всех корзинах выше прямого пути -- деквант не нужен
        без_декванта = (_G3 and not _G3_ФАЙЛ and g3
                        and all(b in g3["win"] for b in _G3_КОРЗ if b > _ПРЯМОЙ_ММАКС))
        if K * N <= _БУФ_МАКС and not без_декванта:
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
        if wf.data_ptr() in _МАРЛИН_ПТР:
            return _марлин_mm(x, wf, meta, K, N)
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
            if _W8SW and m <= 8 and N <= 65536:
                вар = _W8SW_ТАБЛ.get((т["nw"], wa))
                if вар is not None:
                    nbk = N // (8 * т["nw"])
                    спл = sp if sp > 0 else max(1, min((4 * 96 + nbk - 1) // nbk, (K // 64) // (4 * wa)))
                    if спл == 1:
                        y = ext.w8a16_sm80_sw(x, wf, meta, N, т["nw"], wa, вар[0], вар[1])
                        if _W8SW_СВЕРКА and _W8SW_СВЕРЕНО.get((K, N), 0) < _W8SW_СВЕРКА \
                                and not torch.cuda.is_current_stream_capturing():
                            _W8SW_СВЕРЕНО[(K, N)] = _W8SW_СВЕРЕНО.get((K, N), 0) + 1
                            y0 = ext.w8a16_sm80_mm(x, wf, meta, N, sp, т["nw"], wa, st)
                            разн = int((y.view(torch.int16) != y0.view(torch.int16)).sum())
                            _лог(f"W8SW сверка K={K} N={N} M={m}: различий {разн} из {y.numel()}")
                        return y
            return ext.w8a16_sm80_mm(x, wf, meta, N, sp, т["nw"], wa, st)
        _С["большое"] += 1
        if _W8A8 is not None and K % 128 == 0 and N % 128 == 0 and K * N <= _БУФ_МАКС:
            план = _W8A8.куски(0, m)
            if план is not None:
                return _W8A8.смешанный(x, wf, meta, K, N, т["nw"], план,
                                       lambda xi, oi: oi.copy_(_большое(ext, xi, wf, meta, т, K, N)))
        return _большое(ext, x, wf, meta, т, K, N)

    def _марлин_mm(x, wf, meta, K: int, N: int):
        ms, mz = _марлин_части(meta, K, N)
        m = x.shape[0]
        if m <= _MARLIN_MMAX or K * N > _БУФ_МАКС:
            _С["марлин"] += 1
            if not (x.stride(-1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0):
                x = x.contiguous()
            return _м8().marlin8_gemm(x, _марлин_вес(wf, K), ms, mz, N, True)
        _С["марлин_деквант"] += 1
        W = _м8().marlin8_dequant(_марлин_вес(wf, K), ms, mz, N, K, _буфер_для(x.device, K * N))
        import fa2sm80_cg as CG
        if CG.включён() and x.is_contiguous():
            return CG.mm(x, W)
        return torch.mm(x, W.t())

    def _большое(ext, x, wf, meta, т, K: int, N: int):
        m = x.shape[0]
        g3 = т.get("g3")
        if g3 and g3["win"] and g3_вкл():
            cfg = g3["win"].get(_g3_корзина(m))
            if cfg is not None:
                _С["g3"] = _С.get("g3", 0) + 1
                wm, wn, st, sp = cfg
                return ext.w8a16_sm80_gemm3x(x if _g3_выровнен(x) else x.contiguous(), wf, meta, N, т["nw"], wm, wn, st, sp)
        mf = bool(_MFAST or _MFAST_ФАЙЛ) and _mfast()
        прямой = m <= _ПРЯМОЙ_ММАКС
        if mf and not прямой:
            # [ЗАГРУЗКА ВЕСОВ 30.10] 512 < M <= 2048: маршрут корзины выбран замером при подборе
            бк = next((c for c in _ПРЯМ_КОРЗ_MF if c >= m), None)
            прямой = бк is not None and т.get("lroute", {}).get(бк) == "d"
        if прямой and N % 128 == 0:
            # прямой Ampere-GEMM (cp.async 3 стадии + деквант в регистрах): при M=256 быстрее деквант+cuBLAS
            # (замер bench_w8_large: 61-89 TF против 47-71), при M>=1024 пока медленнее (~100 против ~167 TF)
            dc = т["dcfg"]
            b = next((c for c in _ПРЯМ_КОРЗ if c >= m), None)
            if _ПРЯМ_MIDM and b is not None and b in dc:
                g2 = т.get("g2")
                if g2 and b in g2 and not mf and _g2_вкл():
                    nj, stg, pin, sp2 = g2[b]
                    return ext.w8a16_sm80_gemm2(x, wf, meta, N, т["nw"], nj, stg, sp2, pin)
                mi, sp = dc[b]
                return ext.w8a16_sm80_gemm_large(x, wf, meta, N, т["nw"], mi, sp, int(mf))
            return ext.w8a16_sm80_gemm_large(x, wf, meta, N, т["nw"], 4, 1, int(mf))
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
    if _MARLIN and _MARLIN_ВНЕ and K % 128 == 0 and N % 64 == 0:
        сл = type("_Слой", (), {})()
        сл.scales, сл.qzeros = sc, qz
        пара = _в_марлин(сл, qw, v2, K, N)
        if пара is not None:
            mq, meta = пара
            _МАРЛИН_ПТР.add(mq.data_ptr())
            _м8().marlin8_reserve(meta)
            _лог(f"чужой модуль K={K} N={N}: раскладка Marlin")
            return mq, meta
    if (K, N) not in _ТЮН:
        итог, pack = _подобрать(qw, qz, sc, v2, K, N)
        _ТЮН[(K, N)] = итог
        return pack
    return _расш().w8a16_sm80_pack(qw, qz, sc, 128, v2, _ТЮН[(K, N)]["nw"])


def счётчики():
    return dict(_С)
