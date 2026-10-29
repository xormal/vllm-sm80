#!/usr/bin/env python3
"""[ПЛАВНОСТЬ 29.10] Нагрузка «как в бою» для A/B бюджета prefill при декоде.

C8 чатов -- замкнутый цикл потоковых completions (prompt --prompt-tokens, вывод --tokens, ignore_eos); записи Clef --
открытый поток Пуассона (--clef-rate в секунду) с распределением длин по живому кандидату 29.10
(vllm:request_prefill_kv_computed_tokens: 57 % <=500, 8 % <=1K, 10 % <=2K, 11 % <=5K, 13 % <=10K, хвост до 34816).
Бюджеты (--budgets) чередуются фазами по --phase-seconds через файл FA2SM80_PREFILL_BUDGET_FILE сервера (без
перезапуска). Интервалы -- между SSE-чанками (MTP-чанк с несколькими токенами -- одно событие). В фазу идут только
интервалы, оба конца которых внутри фазы (с отступом --settle-seconds после переключения).
"""
import argparse
import concurrent.futures as cf
import importlib.util
import json
import math
import random
import re
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
РАСПРЕДЕЛЕНИЕ = [(0.57, 400), (0.08, 800), (0.096, 1500), (0.108, 3500), (0.13, 8000), (0.008, 15000), (0.008, 34816)]


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--url", default="http://127.0.0.1:8086")
    p.add_argument("--model", default="Qwen3.8-27B-ABLITERATED")
    p.add_argument("--budget-file", type=Path, required=True)
    p.add_argument("--budgets", default="0,2048,1024,4096")
    p.add_argument("--phase-file", type=Path, default=None,
                   help="чередовать по фазам значения в этом файле (переключатель варианта) вместо бюджета")
    p.add_argument("--phase-values", default="", help="значения для --phase-file через запятую")
    p.add_argument("--clef-concurrency", type=int, default=0,
                   help="N>0: очередь Clef всегда полна -- N потоков шлют записи подряд (вместо потока Пуассона)")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--phase-seconds", type=float, default=150)
    p.add_argument("--settle-seconds", type=float, default=10)
    p.add_argument("--chats", type=int, default=8)
    p.add_argument("--prompt-tokens", type=int, default=2048)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--clef-rate", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=29)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.phase_file is not None:
        раздел = "|" if "|" in a.phase_values else ","      # значения с запятыми (план нарезки) -- через |
        values, target = [x for x in a.phase_values.split(раздел) if x], a.phase_file
    else:
        values, target = a.budgets.split(","), a.budget_file
    plan = [b for _ in range(a.rounds) for b in values]

    def post(path, body, timeout=900):
        req = urllib.request.Request(a.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    spec = importlib.util.spec_from_file_location("fixtures", REPO / "sm70/integrations/vllm_sm70/clef/proverka_stenda.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    docs = fixture.текст_доков(400000)
    corpus = post("/tokenize", {"model": a.model, "prompt": docs[:40000], "add_special_tokens": False})["tokens"]
    rnd = random.Random(a.seed)
    stop = threading.Event()
    t0 = time.monotonic()
    phases = []          # (budget, start, end)
    phases_wall = []     # (значение, время начала по часам) -- сверка с журналом памяти сервера
    clef_sent = [0]
    chat_events, chat_reqs, clef_reqs, errors = [], [], [], []
    lock = threading.Lock()

    def chat_worker(w):
        n = 0
        while not stop.is_set():
            n += 1
            head = f"Session {w}-{n}-{rnd.random():.6f}. Continue the technical exposition in detail.\n"
            try:
                hid = post("/tokenize", {"model": a.model, "prompt": head, "add_special_tokens": False})["tokens"]
                off = (w * 7919 + n * 104729) % len(corpus)
                body = (corpus[off:] + corpus) * (a.prompt_tokens // len(corpus) + 2)
                ids = hid + body[:a.prompt_tokens - len(hid)]
                req = urllib.request.Request(a.url + "/v1/completions", json.dumps({
                    "model": a.model, "prompt": ids, "max_tokens": a.tokens, "temperature": 0, "ignore_eos": True,
                    "stream": True, "stream_options": {"include_usage": True}}).encode(),
                    {"Content-Type": "application/json"})
                sent = time.monotonic() - t0
                ev = []
                with urllib.request.urlopen(req, timeout=900) as r:
                    for line in r:
                        if not line.startswith(b"data: "):
                            continue
                        if line.strip() == b"data: [DONE]":
                            break
                        v = json.loads(line[6:])
                        if "error" in v:
                            raise RuntimeError(v["error"])
                        if v.get("choices") and v["choices"][0].get("text"):
                            ev.append(time.monotonic() - t0)
                with lock:
                    chat_reqs.append({"w": w, "sent": sent, "first": ev[0] if ev else None,
                                      "end": time.monotonic() - t0, "chunks": len(ev)})
                    chat_events.append(ev)
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append({"kind": "chat", "t": time.monotonic() - t0, "error": repr(e)[:300]})
                stop.wait(2)

    def clef_one(i, length):
        off = rnd.randrange(0, max(1, len(docs) - length * 4))
        rec = {"model": "clef", "max_length": length,
               "state": f"Record livemix-{i}-{rnd.random():.6f}. The switch is ON.\n" + docs[off:off + length * 4],
               "questions": {"switch": {"type": "choice", "instructions": "What is the switch position?",
                                        "criteria": {"on": "The switch is ON.", "off": "The switch is OFF."}}}}
        sent = time.monotonic() - t0
        try:
            ok = "answers" in post("/v1/systemone", rec)
            with lock:
                clef_reqs.append({"i": i, "len": length, "sent": sent, "end": time.monotonic() - t0, "ok": ok})
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append({"kind": "clef", "t": time.monotonic() - t0, "len": length, "error": repr(e)[:300]})

    def длина():
        x, acc = rnd.random(), 0.0
        for pr, ln in РАСПРЕДЕЛЕНИЕ:
            acc += pr
            if x <= acc:
                return ln
        return РАСПРЕДЕЛЕНИЕ[-1][1]

    def clef_arrivals(pool):
        i = 0
        while not stop.is_set():
            stop.wait(rnd.expovariate(a.clef_rate))
            if stop.is_set():
                break
            i += 1
            clef_sent[0] += 1
            pool.submit(clef_one, i, длина())

    def clef_worker(w):          # очередь всегда полна: следующая запись сразу после ответа
        i = 0
        while not stop.is_set():
            i += 1
            with lock:
                clef_sent[0] += 1
            clef_one(w * 100000 + i, длина())

    with cf.ThreadPoolExecutor(64) as pool:
        for w in range(a.chats):
            pool.submit(chat_worker, w)
        if a.clef_concurrency > 0:
            for w in range(a.clef_concurrency):
                pool.submit(clef_worker, w)
        else:
            threading.Thread(target=clef_arrivals, args=(pool,), daemon=True).start()
        for b in plan:
            target.write_text(str(b))
            start = time.monotonic() - t0
            phases_wall.append((b, time.strftime("%H:%M:%S")))
            time.sleep(a.phase_seconds)
            phases.append((b, start, time.monotonic() - t0))
            print(f"фаза бюджет={b} {start:.0f}-{phases[-1][2]:.0f} с, ошибок пока {len(errors)}", flush=True)
        stop.set()
        if a.phase_file is None:
            a.budget_file.write_text("0")
    итог = {}
    for b, s, e in phases:
        lo = s + a.settle_seconds
        gaps = [y - x for ev in chat_events for x, y in zip(ev, ev[1:]) if x >= lo and y <= e]
        chunks = sum(1 for ev in chat_events for x in ev if lo <= x <= e)
        clef_done = [c for c in clef_reqs if lo <= c["end"] <= e and c["ok"]]
        clef_tok = sum(c["len"] for c in clef_done)
        ttft = [c["first"] - c["sent"] for c in chat_reqs if c["first"] and lo <= c["sent"] <= e]
        d = итог.setdefault(str(b), {"gaps": [], "chunks": 0, "seconds": 0.0, "clef_n": 0, "clef_tok": 0,
                                       "clef_lat": [], "ttft": []})
        d["gaps"] += gaps
        d["chunks"] += chunks
        d["seconds"] += e - lo
        d["clef_n"] += len(clef_done)
        d["clef_tok"] += clef_tok
        d["clef_lat"] += [c["end"] - c["sent"] for c in clef_done]
        d["ttft"] += ttft
    сводка = {}
    for b, d in итог.items():
        g = d["gaps"]
        сводка[b] = {"seconds": round(d["seconds"]), "gaps": len(g),
                     "gap_p50": percentile(g, .5), "gap_p90": percentile(g, .9), "gap_p95": percentile(g, .95),
                     "gap_p99": percentile(g, .99), "gap_max": max(g) if g else None,
                     "gaps_over_0_5s_pct": 100 * sum(x > .5 for x in g) / max(1, len(g)),
                     "gaps_over_1s_pct": 100 * sum(x > 1 for x in g) / max(1, len(g)),
                     "chat_chunks_per_s": d["chunks"] / max(1e-9, d["seconds"]),
                     "clef_done_per_min": 60 * d["clef_n"] / max(1e-9, d["seconds"]),
                     "clef_input_tok_per_s": d["clef_tok"] / max(1e-9, d["seconds"]),
                     "clef_lat_p50": percentile(d["clef_lat"], .5), "clef_lat_p95": percentile(d["clef_lat"], .95),
                     "ttft_p50": percentile(d["ttft"], .5), "ttft_p95": percentile(d["ttft"], .95)}
    out = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, "phases": phases,
           "phases_wall": phases_wall,
           "summary": сводка, "errors": errors[:200], "error_count": len(errors),
           "clef_sent": clef_sent[0], "clef_finished": len(clef_reqs)}
    (a.output / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    (a.output / "raw.json").write_text(json.dumps({"chat_events": chat_events, "chat_reqs": chat_reqs,
                                                   "clef_reqs": clef_reqs}, ensure_ascii=False))
    for b, v in сводка.items():
        print(b, json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()}), flush=True)


if __name__ == "__main__":
    main()
