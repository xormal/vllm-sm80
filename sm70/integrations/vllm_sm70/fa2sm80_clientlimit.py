"""[НАГРУЗКА 06.11] Предел одновременных запросов на клиента (по адресу) и возврат памяти API-сервера после крупных тел.

ПРЕДЕЛ. 06.11 16:23 внешний клиент (пакетная генерация синтетических задач, 18 параллельных запросов,
1.1-1.4 тыс. в час, длинные ответы) занял KV на 85-98 %: очередь движка 10-20, ожидание ~28 с у всех, у Clef поток
записей упал с 3-7 тыс./ч до сотен. Временно стоял предел соединений в iptables (сверх предела -- сброс соединения,
у клиента ошибки). Здесь то же самое без ошибок: запросы адреса сверх предела ЖДУТ своей очереди внутри API-сервера,
тело ещё не прочитано (лёгкая корутина), в движок и в KV попадает не больше N запросов этого адреса.
FA2SM80_CLIENT_LIMIT="192.0.2.10=4,192.0.2.11=2" -- адрес=предел через запятую; пусто -- ничего не ставится.
На боевом значение берётся из $STATE_DIR/client-limit (вне репозитория, scripts/serve_candidate_live.sh).
Ограничиваются POST-запросы (генерация, токенизация, systemone); GET (/health, /metrics, /v1/models) идут мимо.
Слот держится до КОНЦА ответа (чистый ASGI, а не BaseHTTPMiddleware: у потокового ответа call_next возвращается
до тела, и слот освободился бы раньше времени).

ВОЗВРАТ ПАМЯТИ. 08.11 ~08:50 тело 8.4 МБ от Clef (вход длиннее контекста, ответ 400) подняло RSS API с 1449 до 2186 МБ, и
память осталась у процесса: glibc отдаёт системе только вершину кучи. После крупного запроса (>= FA2SM80_REQLOG_KB,
512 КБ) -- malloc_trim(0) в отдельной нити, не чаще раза в FA2SM80_TRIM_EVERY_S (30) секунд, без задержки ответа.
FA2SM80_TRIM_AFTER_BIG=0 -- выключить.
"""
import asyncio
import ctypes
import os
import threading
import time


def разобрать(строка):
    """"a=4, b=2" -> {"a": 4, "b": 2}; кривые элементы пропускаются (о них -- в журнал)."""
    пределы, кривые = {}, []
    for кус in (строка or "").split(","):
        кус = кус.strip()
        if not кус:
            continue
        адр, _, n = кус.partition("=")
        try:
            n = int(n)
        except ValueError:
            n = 0
        if not адр.strip() or n < 1:
            кривые.append(кус)
            continue
        пределы[адр.strip()] = n
    return пределы, кривые


class ПределКлиентов:
    """Чистый ASGI-слой: не больше пределы[адрес] одновременных POST-запросов с адреса; остальные ждут по очереди."""

    def __init__(self, app, пределы, лог=None):
        self.app = app
        self.пределы = dict(пределы)
        self.лог = лог or (lambda s: None)
        self.сем = {}
        self.ждут = {}
        self.допущено = {}
        self._сводка_t = 0.0

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        адр = (scope.get("client") or ("?", 0))[0]
        n = self.пределы.get(адр)
        if n is None:
            return await self.app(scope, receive, send)
        сем = self.сем.get(адр)
        if сем is None:
            сем = self.сем[адр] = asyncio.Semaphore(n)
        self.ждут[адр] = self.ждут.get(адр, 0) + 1
        try:
            await сем.acquire()
        finally:
            self.ждут[адр] -= 1
        try:
            self.допущено[адр] = self.допущено.get(адр, 0) + 1
            сейчас = time.monotonic()
            if self.ждут[адр] and сейчас - self._сводка_t > 300:
                self._сводка_t = сейчас
                self.лог(f"предел клиента {адр}: в работе {n}, ждут {self.ждут[адр]}, допущено всего {self.допущено[адр]}")
            return await self.app(scope, receive, send)
        finally:
            сем.release()


_ТРИМ = {"t": 0.0, "идёт": False}
_ЗАМОК = threading.Lock()


def _rss_мб():
    try:
        with open("/proc/self/status") as f:
            return int(f.read().split("RssAnon:")[1].split()[0]) // 1024
    except Exception:  # noqa: BLE001
        return -1


def вернуть_память(лог=None, каждые_с=None):
    """malloc_trim(0) в отдельной нити после крупного запроса; не чаще раза в каждые_с секунд, не два сразу."""
    if os.environ.get("FA2SM80_TRIM_AFTER_BIG", "1") != "1":
        return False
    каждые_с = float(os.environ.get("FA2SM80_TRIM_EVERY_S", "30")) if каждые_с is None else каждые_с
    with _ЗАМОК:
        сейчас = time.monotonic()
        if _ТРИМ["идёт"] or сейчас - _ТРИМ["t"] < каждые_с:
            return False
        _ТРИМ["идёт"], _ТРИМ["t"] = True, сейчас

    def _нить():
        try:
            до, t0 = _rss_мб(), time.monotonic()
            ctypes.CDLL("libc.so.6").malloc_trim(0)
            после = _rss_мб()
            if лог is not None and до - после >= 64:
                лог(f"malloc_trim: RSS API {до} -> {после} МБ за {time.monotonic() - t0:.2f} с")
        except Exception as e:  # noqa: BLE001
            if лог is not None:
                лог(f"malloc_trim не вышел: {type(e).__name__}: {e}")
        finally:
            with _ЗАМОК:
                _ТРИМ["идёт"] = False

    threading.Thread(target=_нить, name="fa2-trim", daemon=True).start()
    return True


def поставить(app, лог):
    """Ставит предел по FA2SM80_CLIENT_LIMIT (если задан). Возвращает словарь пределов (пустой -- ничего не ставилось)."""
    пределы, кривые = разобрать(os.environ.get("FA2SM80_CLIENT_LIMIT", ""))
    if кривые:
        лог(f"FA2SM80_CLIENT_LIMIT: пропущены кривые элементы {кривые}")
    if пределы:
        app.add_middleware(ПределКлиентов, пределы=пределы, лог=лог)
        лог("предел одновременных запросов на клиента: " + ", ".join(f"{a}={n}" for a, n in пределы.items())
            + " (сверх предела запрос ждёт в API, тело не читается)")
    return пределы
