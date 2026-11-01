"""[02.11] Лёгкая проба боевого на фоне живого потока Clef: (1) задержка записей Clef по размерам, (2) декод одного
чата, (3) префилл 8K/32K. По одному запросу за раз -- нагрузка как у одного клиента; перед каждым ждём пустую очередь.
Чужая нагрузка за каждый замер пишется рядом (чужих запросов, входящих токенов в секунду по /metrics: в них и
попадания в кэш префиксов, так что это верхняя оценка чужого префилла). Окна совсем без запросов на боевом бывают
редко (Clef идёт непрерывно), поэтому замер -- на живом фоне, а не в тишине.

    python3 scripts/live_clef_traffic_probe.py URL OUT.json
Итог 02.11: docs/sm80/results/live-20261102-clef-traffic.json, разбор в docs/sm80/LOAD_TESTS.md.
"""
import os
import json
import random
import re
import statistics as st
import sys
import time
import urllib.error
import urllib.request

URL, OUT = sys.argv[1], sys.argv[2]
MODEL = "Qwen3.8-27B-ABLITERATED"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sm70", "integrations", "vllm_sm70",
                                "clef"))
import proverka_stenda as PS  # noqa: E402


def post(path, body, timeout=600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def метрики():
    txt = urllib.request.urlopen(URL + "/metrics", timeout=5).read().decode()
    out = {}
    for name in ("num_requests_running", "num_requests_waiting", "request_success_total", "prompt_tokens_total"):
        out[name] = sum(float(v) for v in re.findall(rf"^vllm:{name}(?:{{[^}}]*}})? ([0-9.e+-]+)$", txt, re.M))
    return out


def без_очереди():
    while метрики()["num_requests_waiting"] > 0:
        time.sleep(0.5)
    return метрики()


def фон(м0, м1, dt, свои):
    return {"чужих_запросов": м1["request_success_total"] - м0["request_success_total"] - 1,
            "чужой_префилл_ток_с": round((м1["prompt_tokens_total"] - м0["prompt_tokens_total"] - свои) / dt)}


ВОПРОСЫ = {"urgency": {"type": "noul", "instructions": "Does this message express urgency?"},
           "department": {"type": "choice", "instructions": "Which team should handle the message?",
                          "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages",
                                       "sales": "Buying or pricing", "other": "Anything else"}},
           "priority": {"type": "score", "criteria": ["Can wait", "This week", "Today"]}}


def записи_clef():
    r = random.Random(7)
    базовые = PS.записи()
    out = [("короткая", базовые[1]), ("короткая", базовые[3]), ("12 вопросов", базовые[-1])]
    for символов in (1500, 4000, 8000, 30000):
        off = r.randrange(0, 200000)
        out.append((f"текст {символов} симв", {"model": "clef", "state": PS.текст_доков(символов, off),
                                               "questions": ВОПРОСЫ}))
    return out


def фаза_clef(раундов):
    рез = []
    for k in range(раундов):
        for вид, rec in записи_clef():
            м0 = без_очереди()
            t0 = time.perf_counter()
            try:
                with post("/v1/systemone", rec) as resp:
                    d = json.loads(resp.read())
                код = 200
            except urllib.error.HTTPError as e:
                d, код = {}, e.code
            dt = time.perf_counter() - t0
            time.sleep(1.0)
            м1 = метрики()
            n = d.get("usage", {}).get("input_tokens", 0)
            x = {"вид": вид, "код": код, "вход_ток": n, "задержка_с": round(dt, 3), "в_работе_до": м0["num_requests_running"]}
            x.update(фон(м0, м1, dt + 1.0, n))
            рез.append(x)
            print("clef", json.dumps(x, ensure_ascii=False), flush=True)
            time.sleep(1.0)
    return рез


def чат_декод():
    body = {"model": MODEL, "max_tokens": 512, "temperature": 0.0, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "Подробно объясни, как устроен KV-кэш в инференсе трансформеров "
                                                     "и почему он ограничивает длину контекста. Ответ на русском."}]}
    t0 = time.perf_counter()
    метки, usage = [], None
    with post("/v1/chat/completions", body) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices"):
                дел = d["choices"][0].get("delta", {})
                if дел.get("content") or дел.get("reasoning_content") or дел.get("reasoning"):
                    метки.append(time.perf_counter())
    n = usage["completion_tokens"]
    разрывы = sorted(b - a for a, b in zip(метки, метки[1:]))
    return {"ttft_с": round(метки[0] - t0, 3), "токенов": n, "декод_ток_с": round((n - 1) / (метки[-1] - метки[0]), 1),
            "токенов_на_чанк": round(n / len(метки), 2), "разрыв_p50_мс": round(1000 * разрывы[len(разрывы) // 2], 1),
            "разрыв_p99_мс": round(1000 * разрывы[int(len(разрывы) * .99)], 1),
            "промпт": usage["prompt_tokens"]}, метки[-1] - t0


def префилл(корпус, n, r):
    off = r.randrange(0, len(корпус) - n)
    токены = [r.randint(1000, 60000) for _ in range(8)] + корпус[off:off + n - 8]
    t0 = time.perf_counter()
    with post("/v1/completions", {"model": MODEL, "prompt": токены, "max_tokens": 1, "temperature": 0.0}) as resp:
        d = json.loads(resp.read())
    dt = time.perf_counter() - t0
    return {"промпт": d["usage"]["prompt_tokens"], "время_с": round(dt, 3),
            "префилл_ток_с": round(d["usage"]["prompt_tokens"] / dt)}, dt


def main():
    итог = {"clef": фаза_clef(3), "декод": [], "префилл8k": [], "префилл32k": []}
    with post("/tokenize", {"model": MODEL, "prompt": PS.текст_доков(400000), "add_special_tokens": False}) as resp:
        корпус = json.loads(resp.read())["tokens"]
    r = random.Random()
    for вид in ["декод"] * 6 + ["префилл8k"] * 4 + ["префилл32k"] * 3:
        м0 = без_очереди()
        if вид == "декод":
            x, dt = чат_декод()
        else:
            x, dt = префилл(корпус, 8192 if вид == "префилл8k" else 32768, r)
        time.sleep(1.0)
        м1 = метрики()
        x["в_работе_до"] = м0["num_requests_running"]
        x.update(фон(м0, м1, dt + 1.0, x["промпт"]))
        итог[вид].append(x)
        print(вид, json.dumps(x, ensure_ascii=False), flush=True)
        time.sleep(2.0)
    json.dump(итог, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("--- ИТОГ")
    по_виду = {}
    for x in итог["clef"]:
        по_виду.setdefault(x["вид"], []).append(x)
    for вид, xs in по_виду.items():
        t = sorted(x["задержка_с"] for x in xs)
        print(f"clef {вид}: вход ~{st.median(x['вход_ток'] for x in xs):.0f} ток, задержка медиана {st.median(t):.2f} с, "
              f"макс {t[-1]:.2f} с ({len(t)} замеров, коды {sorted({x['код'] for x in xs})}), "
              f"чужой префилл медиана {st.median(x['чужой_префилл_ток_с'] for x in xs):.0f} ток/с")
    for вид in ("декод", "префилл8k", "префилл32k"):
        xs = итог[вид]
        ключ = "декод_ток_с" if вид == "декод" else "префилл_ток_с"
        v = sorted(x[ключ] for x in xs)
        доп = f", разрыв p50 {st.median(x['разрыв_p50_мс'] for x in xs):.1f} мс, ток/чанк {st.median(x['токенов_на_чанк'] for x in xs):.2f}" \
            if вид == "декод" else ""
        print(f"{вид}: {ключ} медиана {st.median(v)} (мин {v[0]}, макс {v[-1]}){доп}; чужой префилл за замер медиана "
              f"{st.median(x['чужой_префилл_ток_с'] for x in xs):.0f} ток/с, чужих запросов медиана "
              f"{st.median(x['чужих_запросов'] for x in xs):.0f}")


if __name__ == "__main__":
    main()
