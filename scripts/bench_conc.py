#!/usr/bin/env python3
"""[МЕТРИКИ 30.10] Префилл и генерация LLM и Clef при конкурентности c=1/4/8 (владелец 30.10: «сними метрики c=1,c=4,c=8,
префилл и генерация и LLM и clef»).

  * LLM генерация: c потоков /v1/completions, у каждого свой промпт 2048 токенов ЖИВОГО текста (срез токенов корпуса
    docs/ со своим смещением -- без попаданий в префикс-кэш; на случайных id принятие MTP скачет от промпта к промпту:
    окно 21 -- 4.3 против 2.9 токена на шаг при c=1), 512 токенов на выход (ignore_eos, T=0). Смещения -- от SEED,
    одинаковые для всех конфигураций. Установившийся декод -- окно [последний первый токен, первый
    последний токен], токены всех потоков в нём (метки времени чанков; чанк шага MTP несёт несколько токенов); плюс
    TTFT и скорость одного потока;
  * LLM префилл: c одновременных запросов по 8192 токена, 1 токен на выход -- токены промптов / стена; TTFT;
  * Clef: c потоков шлют записи подряд (/v1/systemone, запись в формате живой смеси: текст ~2000 токенов, max_length
    2000, один вопрос choice) ~45 с -- записей/с, входных токенов/с, задержка p50/p95;
  * смесь: генерация LLM на c потоков (как выше) при c потоках Clef одновременно -- декод чатов под нагрузкой Clef и
    паузы между чанками (фризы): p50/p99/max разрыва в окне установившегося декода;
  * сочетания (владелец 30.10: «при 4 потоках llm, 1 clef; при максимальном clef сколько декода/префилла в llm на
    1/4/8 и больше; все с разделением на префилл декод»): COMBOS="L:C,..." -- L потоков LLM при C потоках Clef, два
    замера на сочетание: декод (генерация как выше) и префилл (как выше, 8192 токена, 1 на выход) -- плюс пропускная и
    задержка Clef за то же время.
Выход: json (argv[2]) и сводка в stdout.
"""
import json
import os
import random
import statistics as st
import sys
import threading
import time
import urllib.request

URL = sys.argv[1]
OUT = sys.argv[2]
LABEL = sys.argv[3] if len(sys.argv) > 3 else ""
MODEL = "Qwen3.8-27B-ABLITERATED"
CONC = [int(x) for x in os.environ.get("CONC", "1,4,8").split(",")]
PARTS = os.environ.get("PARTS", "gen,prefill,clef,mix").split(",")
SEED = int(os.environ.get("SEED", "21"))
rnd = random.Random(SEED)


def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ids(n):
    return [rnd.randint(1000, 60000) for _ in range(n)]


КОРПУС = []


def текст(n, ключ):
    """n токенов живого текста со смещением от (SEED, ключ)"""
    r = random.Random(f"{SEED}-{ключ}")
    off = r.randrange(0, len(КОРПУС) - n)
    return КОРПУС[off:off + n]


class Оборван(RuntimeError):
    pass


def stream_ts(prompt, max_tokens):
    """метки времени каждого выходного токена (токены чанка -- одной меткой) и время отправки. СТРОГО: ошибка в
    потоке SSE, отсутствие [DONE] или не ровно max_tokens токенов (ignore_eos) -- исключение, а не «быстрый» замер
    по оборванному ответу."""
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True,
            "stream": True, "logprobs": 1}
    req = urllib.request.Request(URL + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ts = []
    готово = False
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                готово = True
                break
            d = json.loads(line[6:])
            if "error" in d:
                raise Оборван(f"ошибка в потоке: {str(d['error'])[:200]}")
            ch = d.get("choices")
            if ch and ch[0].get("logprobs") and ch[0]["logprobs"].get("tokens"):
                now = time.perf_counter()
                ts.extend([now] * len(ch[0]["logprobs"]["tokens"]))
    if not готово:
        raise Оборван(f"поток без [DONE] после {len(ts)} токенов")
    if len(ts) != max_tokens:
        raise Оборван(f"токенов {len(ts)} вместо {max_tokens}")
    return t0, ts


def параллельно(c, fn):
    """fn(k) в c потоках; исключение любого потока -- исключение вызова (после ожидания всех)."""
    out, ош = [None] * c, [None] * c

    def раб(k):
        try:
            out[k] = fn(k)
        except BaseException as e:  # noqa: BLE001
            ош[k] = e
    th = [threading.Thread(target=раб, args=(k,)) for k in range(c)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    плохие = [e for e in ош if e is not None]
    if плохие:
        raise плохие[0]
    return out


def llm_gen(c, метка="gen"):
    pr = [текст(2048, f"{метка}-{c}-{k}") for k in range(c)]
    res = параллельно(c, lambda k: stream_ts(pr[k], 512))
    a = max(ts[0] for _, ts in res)
    b = min(ts[-1] for _, ts in res)
    n = sum(sum(1 for x in ts if a < x <= b) for _, ts in res)
    один = [(len(ts) - 1) / (ts[-1] - ts[0]) for _, ts in res if ts[-1] > ts[0]]
    разрывы = []
    for _, ts in res:
        u = sorted(set(x for x in ts if a <= x <= b))
        разрывы += [y - x for x, y in zip(u, u[1:])]
    разрывы.sort()
    # [ревизия] паузы по ПОЛНЫМ ответам (от первого до последнего токена каждого потока) -- окно установившегося
    # декода выше отрезает начало и хвост потоков, а фриз может прийтись именно туда
    полные = []
    for _, ts in res:
        u = sorted(set(ts))
        полные += [y - x for x, y in zip(u, u[1:])]
    полные.sort()

    def q(f, xs=None):
        xs = разрывы if xs is None else xs
        return round(xs[min(len(xs) - 1, int(f * len(xs)))], 3) if xs else None
    return {"c": c, "декод_всего_ток_с": round(n / (b - a), 1) if b > a else None,
            "декод_на_поток_ток_с": round(st.mean(один), 1),
            "ttft_с": round(st.median(ts[0] - t0 for t0, ts in res), 3),
            "разрыв_p50_с": q(0.5), "разрыв_p99_с": q(0.99), "разрыв_max_с": round(разрывы[-1], 3) if разрывы else None,
            "окно_с": round(b - a, 1),
            "полн_разрыв_p50_с": q(0.5, полные), "полн_разрыв_p99_с": q(0.99, полные),
            "полн_разрыв_max_с": round(полные[-1], 3) if полные else None,
            "токенов_на_шаг": round(st.mean(len(ts) / len(set(ts)) for _, ts in res), 2),
            "запрос_с_медиана": round(st.median(ts[-1] - t0 for t0, ts in res), 2),
            "запрос_с_max": round(max(ts[-1] - t0 for t0, ts in res), 2)}


def под_clef(c_clef, fn, *a):
    """fn(*a) при c_clef потоках Clef одновременно; к результату -- пропускная и задержка Clef за то же время"""
    стоп = threading.Event()
    к = {}
    th = threading.Thread(target=lambda: к.update(clef(c_clef, стоп=стоп)))
    th.start()
    try:
        time.sleep(3)                                   # Clef уже в очереди
        r = fn(*a)
    finally:                                            # клиенты Clef останавливаются при любом исходе
        стоп.set()
        th.join()
    r["clef_c"] = c_clef
    r["clef_записей_с"] = к.get("записей_с")
    r["clef_вход_ток_с"] = к.get("вход_ток_с")
    r["clef_задержка_p50_с"] = к.get("задержка_p50_с")
    r["clef_задержка_p95_с"] = к.get("задержка_p95_с")
    r["clef_ошибок"] = к.get("ошибок")
    return r


def смесь(c):
    return под_clef(c, llm_gen, c, "mix")


def сочетание(lc):
    """'L:C' -- декод и префилл L потоков LLM при C потоках Clef (разные промпты у сочетаний и у замеров)"""
    L, C = (int(x) for x in lc.split(":"))
    рез = {"llm_c": L, "clef_c": C}
    for имя, fn, a in (("декод", llm_gen, (L, f"k{L}x{C}")), ("префилл", llm_prefill, (L,))):
        try:
            рез[имя] = под_clef(C, fn, *a)
        except Exception as e:  # noqa: BLE001 -- провал замера виден в отчёте
            рез[имя] = {"ошибка": f"{type(e).__name__}: {str(e)[:300]}"}
    return рез


def llm_prefill(c, L=8192):
    pr = [ids(L) for _ in range(c)]
    t0 = time.perf_counter()
    res = параллельно(c, lambda k: stream_ts(pr[k], 1))
    стена = max(ts[-1] for _, ts in res) - t0
    return {"c": c, "токенов": L, "префилл_ток_с": round(c * L / стена, 0),
            "ttft_с_медиана": round(st.median(ts[0] - t00 for t00, ts in res), 3), "стена_с": round(стена, 2)}


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sm70", "integrations", "vllm_sm70",
                                "clef"))
import proverka_stenda as PS  # noqa: E402
DOCS = PS.текст_доков(400000)
CLEF_LEN = int(os.environ.get("CLEF_LEN", "2000"))


def clef_rec(k):
    off = rnd.randrange(0, len(DOCS) - CLEF_LEN * 4)
    return {"model": "clef", "max_length": CLEF_LEN,
            "state": f"Record conc-{k}-{rnd.random():.6f}. The switch is ON.\n" + DOCS[off:off + CLEF_LEN * 4],
            "questions": {"switch": {"type": "choice", "instructions": "What is the switch position?",
                                     "criteria": {"on": "The switch is ON.", "off": "The switch is OFF."}}}}


def clef(c, секунд=float(os.environ.get("CLEF_SEC", "45")), стоп=None):
    t_end = time.perf_counter() + секунд
    lat, tok, ok, err = [], [0], [0], [0]
    lock = threading.Lock()

    def worker(k):
        i = 0
        while (time.perf_counter() < t_end) if стоп is None else not стоп.is_set():
            r = clef_rec(k * 1000 + i)
            i += 1
            t = time.perf_counter()
            try:
                d = post("/v1/systemone", r)
            except Exception:  # noqa: BLE001
                with lock:
                    err[0] += 1
                continue
            dt = time.perf_counter() - t
            with lock:
                if "answers" in d:
                    ok[0] += 1
                    lat.append(dt)
                    tok[0] += int(d.get("usage", {}).get("input_tokens", 0))
                else:
                    err[0] += 1
    t0 = time.perf_counter()
    параллельно(c, worker)
    стена = time.perf_counter() - t0
    lat.sort()
    return {"c": c, "записей_с": round(ok[0] / стена, 2), "вход_ток_с": round(tok[0] / стена, 0),
            "задержка_p50_с": round(lat[len(lat) // 2], 2) if lat else None,
            "задержка_p95_с": round(lat[int(0.95 * (len(lat) - 1))], 2) if lat else None,
            "ток_на_запись": round(tok[0] / max(ok[0], 1)), "ошибок": err[0]}


COMBOS = [x for x in os.environ.get("COMBOS", "").split(",") if x]


def main():
    рез = {"метка": LABEL, "llm_генерация": [], "llm_префилл": [], "clef": [], "смесь": [], "сочетания": []}
    post("/v1/completions", {"model": MODEL, "prompt": ids(64), "max_tokens": 4})          # прогрев
    КОРПУС.extend(post("/tokenize", {"model": MODEL, "prompt": DOCS[:150000], "add_special_tokens": False})["tokens"])
    for часть, ключ, имя, fn in (("gen", "llm_генерация", "генерация", llm_gen), ("prefill", "llm_префилл", "префилл", llm_prefill),
                                 ("clef", "clef", "clef", clef), ("mix", "смесь", "смесь", смесь)):
        for c in CONC if часть in PARTS else ():
            try:
                r = fn(c)
            except Exception as e:  # noqa: BLE001 -- провал сценария виден в отчёте, а не превращается в замер
                r = {"c": c, "ошибка": f"{type(e).__name__}: {str(e)[:300]}"}
            рез[ключ].append(r)
            print(LABEL, имя, json.dumps(r, ensure_ascii=False), flush=True)
            json.dump(рез, open(OUT, "w"), ensure_ascii=False, indent=1)     # частичный итог переживает обрыв окна
    for lc in COMBOS if "combo" in PARTS else ():
        r = сочетание(lc)
        рез["сочетания"].append(r)
        print(LABEL, "сочетание", json.dumps(r, ensure_ascii=False), flush=True)
        json.dump(рез, open(OUT, "w"), ensure_ascii=False, indent=1)
    json.dump(рез, open(OUT, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
