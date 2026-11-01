"""[ПАМЯТЬ 01.11] Фаззер утечек памяти сервера (владелец 01.11: «надо с утечкой памяти бороться, проверить всё
фаззингом, чтобы такого не было»). 01.11 15:47 боевой упал CUDA OOM: графы черновика MTP захватывались на ходу под
каждый новый ключ (пачка x корзина длины контекста), и приватные пулы за 14 ч выросли 92 -> 496 МиБ.

Случайная смесь, как у боевого, но с упором на разнообразие форм: число одновременных запросов меняется фазами
1..CMAX, длины промптов -- от коротких до LONG_MAX токенов (разные корзины max_seq_len у графов), стрим и обрывы
посреди стрима, чат с инструментами, записи Clef (/v1/systemone), /tokenize, запросы длиннее контекста (ждём 4xx),
n=2, разные параметры выборки. Раз в 45 с -- проба: один и тот же промпт при temperature=0 (сколько разных ответов).
После прогона -- разбор MEMTRACE из журнала сервера за время прогона: приватные пулы по картам и по пулам, выделено,
повторы аллокатора, память процессов на хосте, захваты и выселения графов MTP.

    python scripts/leak_fuzz.py URL OUT.json СЕКУНД [ЖУРНАЛ_СЕРВЕРА]
    env: SEED (21), CMAX (12), LONG_MAX (72000), POOL_GROWTH_MAX (64, МиБ), MIN_CAPS (8)

Вердикт памяти: прирост приватных пулов от четверти прогона до конца не больше POOL_GROWTH_MAX МиБ на каждой карте
при не меньше MIN_CAPS захватах графов MTP за то же время (меньше захватов -- «мало захватов», вывода нет).
"""
import collections
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request

URL = sys.argv[1]
OUT = sys.argv[2]
СЕКУНД = float(sys.argv[3])
ЖУРНАЛ = sys.argv[4] if len(sys.argv) > 4 else ""
MODEL = "Qwen3.8-27B-ABLITERATED"
SEED = int(os.environ.get("SEED", "21"))
CMAX = int(os.environ.get("CMAX", "12"))
LONG_MAX = int(os.environ.get("LONG_MAX", "72000"))
POOL_GROWTH_MAX = float(os.environ.get("POOL_GROWTH_MAX", "64"))
MIN_CAPS = int(os.environ.get("MIN_CAPS", "8"))
КОНТЕКСТ = 262144

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sm70", "integrations", "vllm_sm70",
                                "clef"))
import proverka_stenda as PS  # noqa: E402

DOCS = PS.текст_доков(400000)
ТОКЕНЫ = []
стоп = threading.Event()
замок = threading.Lock()
итоги = collections.defaultdict(collections.Counter)    # тип -> исход -> число
задержки = collections.defaultdict(list)
пробы = []
здоровье = {"проверок": 0, "отказов": 0, "подряд_макс": 0}
текущее_c = [1]
ход_c = []


def отметить(тип, исход, t0=None):
    with замок:
        итоги[тип][исход] += 1
        if t0 is not None and исход == "ok":
            задержки[тип].append(time.perf_counter() - t0)


def запрос(path, body, timeout=900):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def исход_ошибки(e):
    if isinstance(e, urllib.error.HTTPError):
        return f"{e.code // 100}xx" if e.code != 400 else "400"
    return f"сбой:{type(e).__name__}"


def промпт(n, r):
    """n токенов: 8 случайных в начале (без попаданий в кэш префиксов) + куски корпуса с случайных мест"""
    out = [r.randint(1000, 60000) for _ in range(min(8, n))]
    while len(out) < n:
        k = min(n - len(out), r.randint(500, 8000))
        off = r.randrange(0, len(ТОКЕНЫ) - k)
        out.extend(ТОКЕНЫ[off:off + k])
    return out


def длина(r):
    x = r.random()
    if x < 0.60:
        return r.randint(64, 2048)
    if x < 0.85:
        return r.randint(2048, 12000)
    return r.randint(16000, LONG_MAX)


def выборка(r, body):
    if r.random() < 0.5:
        body["temperature"] = 0.0
    else:
        body["temperature"] = round(r.uniform(0.3, 1.0), 2)
        body["top_p"] = round(r.uniform(0.8, 1.0), 2)
        body["top_k"] = r.choice([-1, 20, 50])
    return body


def читать_стрим(r, resp, обрыв):
    """строки SSE; обрыв > 0 -- закрыть соединение после стольких событий (клиент ушёл посреди ответа)"""
    n = 0
    for line in resp:
        line = line.decode(errors="replace").strip()
        if not line.startswith("data: "):
            continue
        if line == "data: [DONE]":
            return "ok"
        d = json.loads(line[6:])
        if "error" in d:
            return "ошибка_в_потоке"
        n += 1
        if обрыв and n >= обрыв:
            return "обрыв"
    return "без_DONE"


def т_gen(r):
    t0 = time.perf_counter()
    body = выборка(r, {"model": MODEL, "prompt": промпт(длина(r), r), "max_tokens": r.randint(16, 400),
                       "ignore_eos": r.random() < 0.5})
    тип = "gen"
    if r.random() < 0.03:
        body["n"] = 2
        тип = "gen_n2"
    стрим = r.random() < 0.8
    обрыв = r.randint(1, 20) if стрим and r.random() < 0.15 else 0
    body["stream"] = стрим
    try:
        with запрос("/v1/completions", body) as resp:
            if стрим:
                отметить(тип, читать_стрим(r, resp, обрыв), t0)
            else:
                d = json.loads(resp.read())
                отметить(тип, "ok" if d.get("choices") else "пусто", t0)
    except Exception as e:  # noqa: BLE001
        отметить(тип, исход_ошибки(e))


ИНСТРУМЕНТЫ = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file",
                                      "parameters": {"type": "object", "properties": {
                                          "path": {"type": "string"}, "offset": {"type": "integer"},
                                          "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "run", "description": "Run a shell command",
                                      "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
                                                     "required": ["cmd"]}}},
]
ПРОСЬБЫ = ["Прочитай файл /etc/hosts и покажи строки с localhost.",
           "Запусти ls -la /var/log и скажи, какой файл самый большой.",
           "Кратко перескажи текст ниже в трёх пунктах.",
           "Найди в тексте ниже все упоминания ядер и перечисли их.",
           "Read the file ./README.md starting at line 10, 20 lines."]


def т_chat(r):
    t0 = time.perf_counter()
    off = r.randrange(0, len(DOCS) - 40000)
    текст = DOCS[off:off + r.randint(200, 40000)]
    body = выборка(r, {"model": MODEL, "max_tokens": r.randint(64, 300),
                       "messages": [{"role": "user", "content": r.choice(ПРОСЬБЫ) + "\n\n" + текст}]})
    if r.random() < 0.6:
        body["tools"] = ИНСТРУМЕНТЫ
        body["tool_choice"] = "auto"
    стрим = r.random() < 0.6
    обрыв = r.randint(1, 30) if стрим and r.random() < 0.1 else 0
    body["stream"] = стрим
    try:
        with запрос("/v1/chat/completions", body) as resp:
            if стрим:
                отметить("chat", читать_стрим(r, resp, обрыв), t0)
            else:
                d = json.loads(resp.read())
                отметить("chat", "ok" if d.get("choices") else "пусто", t0)
    except Exception as e:  # noqa: BLE001
        отметить("chat", исход_ошибки(e))


def т_clef(r):
    t0 = time.perf_counter()
    n = r.randint(300, 4000)
    off = r.randrange(0, len(DOCS) - n * 4)
    body = {"model": "clef", "max_length": n,
            "state": f"Record fuzz-{r.random():.8f}. The switch is {'ON' if r.random() < 0.5 else 'OFF'}.\n"
                     + DOCS[off:off + n * 4],
            "questions": {"switch": {"type": "choice", "instructions": "What is the switch position?",
                                     "criteria": {"on": "The switch is ON.", "off": "The switch is OFF."}}}}
    try:
        with запрос("/v1/systemone", body) as resp:
            resp.read()
            отметить("clef", "ok", t0)
    except Exception as e:  # noqa: BLE001
        отметить("clef", исход_ошибки(e))


def т_tokenize(r):
    t0 = time.perf_counter()
    off = r.randrange(0, len(DOCS) - 1000)
    try:
        with запрос("/tokenize", {"model": MODEL, "prompt": DOCS[off:off + r.randint(1000, 200000)]}) as resp:
            resp.read()
            отметить("tokenize", "ok", t0)
    except Exception as e:  # noqa: BLE001
        отметить("tokenize", исход_ошибки(e))


def т_длиннее(r):
    """длиннее контекста -- сервер обязан отказать 4xx, не упасть"""
    body = {"model": MODEL, "prompt": промпт(КОНТЕКСТ + r.randint(1, 8000), r), "max_tokens": 8}
    try:
        with запрос("/v1/completions", body) as resp:
            resp.read()
            отметить("длиннее_контекста", "принят(!)")
    except Exception as e:  # noqa: BLE001
        отметить("длиннее_контекста", исход_ошибки(e))


ЗАДАЧИ = [(т_gen, 0.45), (т_clef, 0.30), (т_chat, 0.15), (т_tokenize, 0.06), (т_длиннее, 0.04)]


def работник(k):
    r = random.Random(f"{SEED}-раб-{k}")
    while not стоп.is_set():
        if k >= текущее_c[0]:
            time.sleep(0.2)
            continue
        f = r.choices([f for f, _ in ЗАДАЧИ], [w for _, w in ЗАДАЧИ])[0]
        f(r)


def пробник():
    r = random.Random(f"{SEED}-проба")
    body = {"model": MODEL, "prompt": промпт(600, r), "max_tokens": 48, "temperature": 0.0, "ignore_eos": True}
    while not стоп.wait(45):
        try:
            with запрос("/v1/completions", body, timeout=600) as resp:
                d = json.loads(resp.read())
            пробы.append(hashlib.sha1(d["choices"][0]["text"].encode()).hexdigest()[:12])
        except Exception as e:  # noqa: BLE001
            пробы.append(исход_ошибки(e))


def сторож():
    подряд = 0
    while not стоп.wait(5):
        здоровье["проверок"] += 1
        try:
            with urllib.request.urlopen(URL + "/health", timeout=10) as resp:
                resp.read()
            подряд = 0
        except Exception:  # noqa: BLE001
            здоровье["отказов"] += 1
            подряд += 1
            здоровье["подряд_макс"] = max(здоровье["подряд_макс"], подряд)


RX_GPU = re.compile(r"MEMTRACE\] (\d\d:\d\d:\d\d) GPU(\d): выделено (\d+) резерв (\d+) пик (\d+) \| приватные пулы "
                    r"(\d+) \(занято (\d+), сегментов (\d+)\) МиБ \| повторов (\d+), OOM (\d+)")
RX_ПУЛЫ = re.compile(r"MEMTRACE\] \d\d:\d\d:\d\d GPU(\d) по пулам \([^)]*\): (.*)$")
RX_ХОСТ = re.compile(r"\((\w+?)(?:_DP0|_TP\d)? pid=\d+\).*MEMTRACE\] .* pid=\d+: heap=(\d+) anon=(\d+)")
RX_ПРОЦ = re.compile(r"^\S*\((\w+) pid=")


def разбор_журнала(путь, с_байта):
    """MEMTRACE и графы MTP из журнала сервера начиная с байта с_байта"""
    if not путь or not os.path.exists(путь):
        return {}
    with open(путь, "rb") as f:
        f.seek(с_байта)
        строки = f.read().decode("utf-8", errors="replace").splitlines()
    строки = [re.sub(r"\x1b\[[0-9;]*m", "", s) for s in строки]
    gpu = collections.defaultdict(list)          # карта -> [(номер строки, словарь)]
    пулы, хост, графы = {}, collections.defaultdict(list), []
    for i, s in enumerate(строки):
        m = RX_GPU.search(s)
        if m:
            t, d, a, rz, pk, pl, plu, n, rt, oo = m.groups()
            gpu[d].append((i, {"t": t, "выделено": int(a), "резерв": int(rz), "пик": int(pk), "пулы": int(pl),
                               "пулы_занято": int(plu), "сегментов": int(n), "повторов": int(rt), "oom": int(oo)}))
            continue
        m = RX_ПУЛЫ.search(s)
        if m:
            пулы[m.group(1)] = m.group(2)
            continue
        m = RX_ХОСТ.search(s)
        if m:
            p = RX_ПРОЦ.search(s)
            хост[p.group(1) if p else m.group(1)].append((int(m.group(2)), int(m.group(3))))
            continue
        if "[MTP ГРАФ" in s and ("захвачен:" in s or "выселен" in s or "отказ захвата" in s):
            if "TP0" in s or "Worker" not in s:
                вид = ("цикл" if "ЦИКЛА" in s else "тело") + ("_захват" if "захвачен:" in s else
                                                               "_выселение" if "выселен" in s else "_отказ")
                графы.append((i, вид))
    рез = {"карты": {}, "по_пулам": пулы, "хост": {}, "графы": dict(collections.Counter(v for _, v in графы)),
           "сбои": {k: sum(k in s for s in строки) for k in
                    ("CUDA out of memory", "EngineDeadError", "illegal memory access", "Traceback",
                     "500 Internal Server Error")}}
    вердикт = "чисто"
    for d, ряд in sorted(gpu.items()):
        if len(ряд) < 4:
            вердикт = "мало замеров"
            continue
        i_тёпл, тёпл = ряд[len(ряд) // 4]
        перв, посл = ряд[0][1], ряд[-1][1]
        захватов_после = sum(1 for i, v in графы if i > i_тёпл and v.endswith("_захват"))
        прирост = посл["пулы"] - тёпл["пулы"]
        рез["карты"][f"GPU{d}"] = {
            "замеров": len(ряд), "пулы_первый": перв["пулы"], "пулы_четверть": тёпл["пулы"], "пулы_последний":
            посл["пулы"], "пулы_макс": max(v["пулы"] for _, v in ряд), "пулы_занято_последний": посл["пулы_занято"],
            "сегментов": посл["сегментов"], "выделено_мин": min(v["выделено"] for _, v in ряд),
            "выделено_макс": max(v["выделено"] for _, v in ряд), "пик": посл["пик"],
            "повторов_прирост": посл["повторов"] - перв["повторов"], "oom": посл["oom"],
            "прирост_пулов_после_четверти": прирост, "захватов_после_четверти": захватов_после}
        if прирост > POOL_GROWTH_MAX:
            вердикт = "РОСТ"
        elif захватов_после < MIN_CAPS and вердикт == "чисто":
            вердикт = "мало захватов"
    for имя, ряд in хост.items():
        рез["хост"][имя] = {"heap_первый": ряд[0][0], "heap_последний": ряд[-1][0], "anon_первый": ряд[0][1],
                            "anon_последний": ряд[-1][1]}
    рез["вердикт_памяти"] = вердикт if gpu else "нет MEMTRACE"
    return рез


def main():
    global ТОКЕНЫ
    with запрос("/tokenize", {"model": MODEL, "prompt": DOCS, "add_special_tokens": False}) as resp:
        ТОКЕНЫ = json.loads(resp.read())["tokens"]
    с_байта = os.path.getsize(ЖУРНАЛ) if ЖУРНАЛ and os.path.exists(ЖУРНАЛ) else 0
    rc = random.Random(f"{SEED}-c")
    потоки = [threading.Thread(target=работник, args=(k,), daemon=True) for k in range(CMAX)]
    потоки += [threading.Thread(target=пробник, daemon=True), threading.Thread(target=сторож, daemon=True)]
    for t in потоки:
        t.start()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < СЕКУНД:
        текущее_c[0] = rc.choices(range(1, CMAX + 1), [3 if c <= 9 else 1 for c in range(1, CMAX + 1)])[0]
        ход_c.append(текущее_c[0])
        стоп.wait(min(rc.uniform(15, 45), max(0.0, СЕКУНД - (time.perf_counter() - t0))))
    стоп.set()
    срок = time.perf_counter() + 180                     # хвост запросов; зависший сервер не держит разбор
    for t in потоки:
        t.join(timeout=max(0.0, срок - time.perf_counter()))
    time.sleep(25)                                       # последний замер MEMTRACE после хвоста запросов
    рез = {"seed": SEED, "секунд": СЕКУНД, "cmax": CMAX, "фазы_c": ход_c,
           "запросы": {k: dict(v) for k, v in итоги.items()},
           "задержка_медиана_с": {k: round(sorted(v)[len(v) // 2], 2) for k, v in задержки.items() if v},
           "пробы": пробы, "проб_разных_ответов": len({p for p in пробы if not p.startswith(("4xx", "5xx", "400",
                                                                                              "сбой"))}),
           "здоровье": здоровье, "журнал": разбор_журнала(ЖУРНАЛ, с_байта)}
    в5 = sum(n for v in итоги.values() for k, n in v.items() if k == "5xx")
    сбоев = sum(n for v in итоги.values() for k, n in v.items() if k.startswith("сбой"))
    принят = sum(v.get("принят(!)", 0) for v in итоги.values())
    рез["вердикт_запросов"] = "чисто" if в5 == 0 and сбоев == 0 and принят == 0 and здоровье["отказов"] == 0 \
        else f"5xx {в5}, сбоев соединения {сбоев}, принято длиннее контекста {принят}, отказов health {здоровье['отказов']}"
    json.dump(рез, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("ФАЗЗ запросы: " + "; ".join(f"{k} {dict(v)}" for k, v in sorted(итоги.items())))
    print(f"ФАЗЗ пробы: {len(пробы)}, разных ответов {рез['проб_разных_ответов']}; health: {здоровье}")
    ж = рез["журнал"]
    for d, v in ж.get("карты", {}).items():
        print(f"ФАЗЗ память {d}: пулы {v['пулы_первый']} -> {v['пулы_четверть']} (четверть) -> {v['пулы_последний']} "
              f"МиБ (занято {v['пулы_занято_последний']}, сегментов {v['сегментов']}), прирост после четверти "
              f"{v['прирост_пулов_после_четверти']} при {v['захватов_после_четверти']} захватах; выделено "
              f"{v['выделено_мин']}..{v['выделено_макс']}, пик {v['пик']}, повторов +{v['повторов_прирост']}, "
              f"OOM {v['oom']}")
    for d, s in ж.get("по_пулам", {}).items():
        print(f"ФАЗЗ пулы GPU{d}: {s}")
    print(f"ФАЗЗ графы MTP: {ж.get('графы')}; хост: {ж.get('хост')}; сбои в журнале: {ж.get('сбои')}")
    print(f"ФАЗЗ вердикт: память {ж.get('вердикт_памяти')}; запросы {рез['вердикт_запросов']}")


if __name__ == "__main__":
    main()
