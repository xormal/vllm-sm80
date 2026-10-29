#!/usr/bin/env python3
"""[ПАМЯТЬ/ПЛАВНОСТЬ 29.10] Окно 2: перебор форм декода для графов тела MTP, скорость декода, профиль шага.

sweep   -- для каждого (число чатов, длина контекста) C потоковых completions: общий префикс (из кэша префиксов,
           чтобы не платить prefill заново) + уникальный хвост 64 токена, вывод --tokens, ignore_eos. Скорость
           декода потока = (токенов-1)/(последний чанк - первый). Набирает ключи графов тела (запросы x шаг x
           корзина 16К длины контекста).
profile -- C8 декод + записи Clef ~400 токенов каждые 0,5 с; между /start_profile и /stop_profile -- --seconds.
det     -- по одному запросу (жадно, 96 токенов, logprobs): prompt 100-600 токенов (смешанные размеры 64-512 после
           дополнения) -> номера токенов для побитовой сверки двух экземпляров (eager против кусочных графов).
"""
import argparse
import concurrent.futures as cf
import importlib.util
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("sweep", "profile", "det"))
    p.add_argument("output", type=Path)
    p.add_argument("--url", default="http://127.0.0.1:8086")
    p.add_argument("--model", default="Qwen3.8-27B-ABLITERATED")
    p.add_argument("--concurrency", default="1,2,4,8")
    p.add_argument("--contexts", default="2048,18432,34816")
    p.add_argument("--tokens", type=int, default=192)
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--tag", default="")
    p.add_argument("--fixed", action="store_true", help="одинаковые промпты от прогона к прогону (A/B скорости)")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    def post(path, body, timeout=900):
        req = urllib.request.Request(a.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        return json.loads(data) if data else {}

    spec = importlib.util.spec_from_file_location("fixtures", REPO / "sm70/integrations/vllm_sm70/clef/proverka_stenda.py")
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    docs = fx.текст_доков(400000)
    corpus = post("/tokenize", {"model": a.model, "prompt": docs[:200000], "add_special_tokens": False})["tokens"]

    def stream(ids, n):
        req = urllib.request.Request(a.url + "/v1/completions", json.dumps({
            "model": a.model, "prompt": ids, "max_tokens": n, "temperature": 0, "ignore_eos": True,
            "stream": True}).encode(), {"Content-Type": "application/json"})
        t0 = time.monotonic()
        ev = []
        with urllib.request.urlopen(req, timeout=900) as r:
            for line in r:
                if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                    v = json.loads(line[6:])
                    if v.get("choices") and v["choices"][0].get("text"):
                        ev.append(time.monotonic() - t0)
        return ev

    def prompt(ctx, tag):
        body = (corpus * (ctx // len(corpus) + 1))[:ctx - 64]
        off = sum(map(ord, tag)) * 7919 % 300000           # детерминированно (hash() у строк соли процесса)
        tail = post("/tokenize", {"model": a.model, "prompt": f" [{tag}] " + docs[off:][:600],
                                  "add_special_tokens": False})["tokens"][:64]
        return body + tail

    if a.mode == "det":
        out = []
        for n in (100, 180, 260, 333, 420, 511, 600):
            ids = (corpus[n * 37:] + corpus)[:n]
            r = post("/v1/completions", {"model": a.model, "prompt": ids, "max_tokens": 96, "temperature": 0,
                                         "ignore_eos": True, "logprobs": 1})
            ch = r["choices"][0]
            out.append({"n": n, "tokens": ch["logprobs"]["tokens"], "lp": ch["logprobs"]["token_logprobs"]})
            print(f"det n={n}: {len(ch['logprobs']['tokens'])} токенов", flush=True)
        (a.output / f"det{('-' + a.tag) if a.tag else ''}.json").write_text(json.dumps(out, ensure_ascii=False))
        return
    if a.mode == "sweep":
        res = []
        for ctx in [int(x) for x in a.contexts.split(",")]:
            post("/v1/completions", {"model": a.model, "prompt": prompt(ctx, "warm"), "max_tokens": 1})  # префикс в кэш
            for c in [int(x) for x in a.concurrency.split(",")]:
                prompts = [prompt(ctx, f"fixed-{ctx}-{c}-{i}" if a.fixed else f"{a.tag}-{ctx}-{c}-{i}")
                           for i in range(c)]
                bar = threading.Barrier(c)

                def one(i):
                    bar.wait()
                    return stream(prompts[i], a.tokens)
                with cf.ThreadPoolExecutor(c) as pool:
                    evs = list(pool.map(one, range(c)))
                # ignore_eos: всего ровно --tokens токенов, первый чанк после prefill -- один токен
                rates = [(a.tokens - 1) / (e[-1] - e[0]) for e in evs if len(e) > 2 and e[-1] > e[0]]
                chunks_s = [len(e) / max(1e-9, e[-1] - e[0]) for e in evs if len(e) > 2]
                r = {"ctx": ctx, "conc": c, "tok_s_per_stream": statistics.mean(rates) if rates else None,
                     "chunk_rate_per_stream": statistics.mean(chunks_s) if chunks_s else None,
                     "ttft_mean": statistics.mean(e[0] for e in evs if e) if evs else None,
                     "chunks": [len(e) for e in evs], "decode_chunk_s_min": min(chunks_s) if chunks_s else None,
                     "rates": rates}
                res.append(r)
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                                  if k != "rates"}), flush=True)
        (a.output / f"sweep{('-' + a.tag) if a.tag else ''}.json").write_text(json.dumps(res, indent=2))
    else:
        stop = threading.Event()

        def chat(i):
            ids = prompt(2048, f"prof-{i}")
            while not stop.is_set():
                stream(ids, 128)

        def clef():
            k = 0
            while not stop.is_set():
                k += 1
                rec = {"model": "clef", "max_length": 400,
                       "state": f"Record prof-{k}. The switch is ON.\n" + docs[k * 1777 % 300000:][:1600],
                       "questions": {"switch": {"type": "choice", "instructions": "What is the switch position?",
                                                "criteria": {"on": "The switch is ON.", "off": "The switch is OFF."}}}}
                threading.Thread(target=lambda r=rec: post("/v1/systemone", r), daemon=True).start()
                stop.wait(0.5)
        with cf.ThreadPoolExecutor(10) as pool:
            for i in range(8):
                pool.submit(chat, i)
            pool.submit(clef)
            time.sleep(25)                       # выйти на установившийся режим
            post("/start_profile", {})
            time.sleep(a.seconds)
            post("/stop_profile", {}, timeout=1800)
            stop.set()
        print("профиль снят", flush=True)


if __name__ == "__main__":
    main()
