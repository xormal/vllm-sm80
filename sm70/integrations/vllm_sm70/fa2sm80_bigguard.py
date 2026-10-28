# [ПАМЯТЬ 27.10] ОХРАНА API-СЕРВЕРА ОТ ОГРОМНЫХ ТЕЛ ЗАПРОСОВ.
#
# Боевой sm80 падал по oom-kill от одиночных запросов в сотни МБ: /tokenize 494-618 МБ (26-27.10), чат 394.6 МБ
# (27.10 13:20). Разбор такого тела, рендер шаблона и токенизация -- это гигабайты в API-сервере, а cgroup сервиса
# общий с воркерами (21G). Здесь (до чтения тела, клиенту поведение для обычных запросов прежнее):
#   * тела >= FA2SM80_BIG_BODY_MB (16) идут СТРОГО ПО ОДНОМУ (остальные ждут лёгкой корутиной, тело не читается);
#   * перед допуском -- проверка памяти: нужно ~FA2SM80_BIG_MEM_X (8) x тело + FA2SM80_BIG_MEM_FIXED_GB (1),
#     свободно = min по всем уровням cgroup с лимитом (max - current + файловый кэш) и по хосту (MemAvailable минус
#     FA2SM80_HOST_RESERVE_GB: memguard убивает срез при MemAvailable < 3 ГБ);
#   * не хватает -- ждём освобождения до FA2SM80_BIG_WAIT_S (300) с, потом 503 с Retry-After (ответ OpenAI-формата);
#   * после огромного запроса -- malloc_trim(0) через 5 с (glibc отдаёт освобождённое системе).
# Оценка 8x: /tokenize чата 617.8 МБ поднимал API на ~4-5 ГБ (разбор тела + рендер + номера). FA2SM80_BIG_BODY_MB=0 --
# выключить.
import asyncio
import os
import threading
import time

_ПОРОГ = int(float(os.environ.get("FA2SM80_BIG_BODY_MB", "16")) * 2**20)
_КРАТНО = float(os.environ.get("FA2SM80_BIG_MEM_X", "8"))
_ПОСТОЯННО = float(os.environ.get("FA2SM80_BIG_MEM_FIXED_GB", "1")) * 2**30
_ЖДАТЬ = float(os.environ.get("FA2SM80_BIG_WAIT_S", "300"))
_ХОСТ_РЕЗЕРВ = float(os.environ.get("FA2SM80_HOST_RESERVE_GB", "3.5")) * 2**30


def _каталог_cgroup():
    try:
        for line in open("/proc/self/cgroup"):
            if line.startswith("0::"):
                return "/sys/fs/cgroup" + line.strip()[3:]
    except OSError:
        pass
    return None


def свободно():
    """Байт, которые API-сервер может ещё занять, не упершись ни в один лимит (cgroup всех уровней, хост)."""
    вар = []
    d = _каталог_cgroup()
    while d and d.startswith("/sys/fs/cgroup/"):
        try:
            m = open(d + "/memory.max").read().strip()
            if m != "max":
                занято = int(open(d + "/memory.current").read())
                st = dict(line.split() for line in open(d + "/memory.stat"))
                # файловый кэш ядро отдаст под нагрузкой; активную часть -- считаем наполовину
                кэш = int(st.get("inactive_file", 0)) + int(st.get("active_file", 0)) // 2
                вар.append(int(m) - занято + кэш)
        except (OSError, ValueError):
            pass
        d = os.path.dirname(d)
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                вар.append(int(line.split()[1]) * 1024 - _ХОСТ_РЕЗЕРВ)
                break
    except OSError:
        pass
    return min(вар) if вар else float("inf")


def _вернуть_память():
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass


def поставить(app, лог):
    """Повесить охрану на FastAPI-приложение (до чтения тела). лог(str) -- куда писать решения."""
    if not _ПОРОГ:
        return
    from fastapi.responses import JSONResponse

    from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse

    сем = [None]

    @app.middleware("http")
    async def _fa2_охрана(request, call_next):
        cl = int(request.headers.get("content-length") or 0)
        if cl < _ПОРОГ:
            return await call_next(request)
        if сем[0] is None:
            сем[0] = asyncio.Semaphore(1)
        путь, мб = request.url.path, cl / 2**20
        нужно = _КРАТНО * cl + _ПОСТОЯННО
        t0 = time.time()
        async with сем[0]:
            t1 = time.time()
            св = свободно()
            while св < нужно:
                if time.time() - t1 > _ЖДАТЬ:
                    лог(f"ОТКАЗ 503 {путь} {мб:.1f} МБ: нужно ~{нужно / 2**30:.1f} ГиБ, свободно {св / 2**30:.1f} ГиБ "
                        f"(ждали память {time.time() - t1:.0f} с, очередь {t1 - t0:.0f} с)")
                    ош = ErrorResponse(error=ErrorInfo(
                        message=(f"The server does not have enough free memory to process a {мб:.0f} MB request body "
                                 f"right now (needs ~{нужно / 2**30:.1f} GiB, {max(св, 0) / 2**30:.1f} GiB free). "
                                 f"Please retry later or send a smaller request."),
                        type="ServiceUnavailableError", code=503))
                    return JSONResponse(content=ош.model_dump(), status_code=503, headers={"Retry-After": "120"})
                await asyncio.sleep(2)
                св = свободно()
            if time.time() - t0 > 1:
                лог(f"ДОПУСК {путь} {мб:.1f} МБ после ожидания {time.time() - t0:.0f} с (очередь {t1 - t0:.0f} с): "
                    f"нужно ~{нужно / 2**30:.1f} ГиБ, свободно {св / 2**30:.1f} ГиБ")
            try:
                return await call_next(request)
            finally:
                threading.Timer(5.0, _вернуть_память).start()

    лог(f"охрана огромных тел: >= {_ПОРОГ / 2**20:.0f} МБ по одному, память ~{_КРАТНО:g}x тела + "
        f"{_ПОСТОЯННО / 2**30:g} ГиБ, ожидание до {_ЖДАТЬ:.0f} с, иначе 503; свободно сейчас {свободно() / 2**30:.1f} ГиБ")
