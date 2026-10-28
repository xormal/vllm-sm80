#!/usr/bin/env python3
"""Cancellation, prefix reuse, max_tokens=1 and K4 prefill-tail state probes."""
import argparse
import concurrent.futures as cf
import json
from pathlib import Path
import re
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8086"
MODEL = "Qwen3.8-27B-ABLITERATED-candidate"


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as response: return json.loads(response.read())


def metrics():
    with urllib.request.urlopen(URL + "/metrics", timeout=5) as response: text = response.read().decode()
    return {name: sum(float(x) for x in re.findall(rf'^vllm:{name}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)', text, re.M))
            for name in ("num_requests_running", "num_requests_waiting", "prefix_cache_hits_total", "num_preemptions_total")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    text = ("Каждый запрос имеет отдельное состояние. При отмене запроса его память освобождается. "
            "The computation must preserve the previous tokens and attention mask.\n") * 300
    corpus = post("/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": False})["tokens"]
    def prompt(length):
        head = post("/tokenize", {"model": MODEL, "prompt": f"State test {length}. Continue this document.\n",
                                 "add_special_tokens": False})["tokens"]
        return head + (corpus * (length // len(corpus) + 1))[:length - len(head)]
    def call(ids, count=32, salt=None):
        return post("/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": count,
                    "temperature": 0, "seed": 17, "ignore_eos": True, "logprobs": 5,
                    **({"cache_salt": salt} if salt else {})})
    def tokens(response):
        return response["choices"][0]["logprobs"]["tokens"]
    def stream(ids, ready, cancel=False):
        body = {"model": MODEL, "prompt": ids, "max_tokens": 2048,
                "temperature": 0, "seed": 17, "ignore_eos": True, "stream": True}
        req = urllib.request.Request(URL + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        item = {"started": time.monotonic(), "chunks": 0}
        with urllib.request.urlopen(req, timeout=300) as response:
            for line in response:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]": continue
                value = json.loads(line[6:])
                if value.get("choices") and value["choices"][0].get("text"):
                    item["chunks"] += 1
                    if item["chunks"] >= 3:
                        ready.set()
                        if cancel: break
        item["finished"] = time.monotonic()
        return item
    results = {}
    ids = prompt(4096)
    before = call(ids)
    cancelled = stream(ids, threading.Event(), cancel=True)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if metrics()["num_requests_running"] == 0: break
        time.sleep(.2)
    after = call(ids)
    results["cancel"] = {"before": before, "after": after, "same_tokens": tokens(before) == tokens(after), "stream": cancelled}
    print("cancel preserves output", results["cancel"]["same_tokens"], flush=True)
    ids = prompt(24576)
    first = call(ids, 1)
    m0 = metrics()
    again = call(ids)
    m1 = metrics()
    results["prefix_max_tokens_1"] = {"first": first, "again": again, "hits": m1["prefix_cache_hits_total"]-m0["prefix_cache_hits_total"],
                                      "same_first_token": tokens(first)[0] == tokens(again)[0]}
    print("prefix reuse", results["prefix_max_tokens_1"]["hits"], "same first", results["prefix_max_tokens_1"]["same_first_token"], flush=True)
    for length in (8196, 8197, 8198, 16389):
        ids = prompt(length)
        # Different cache domains force real prefill in both trials, while
        # preserving identical token inputs. Otherwise the solo trial warms
        # the prefix and the mixed trial barely exercises the intended tail.
        solo = call(ids, salt=f"tail-{length}-solo-{time.time_ns()}")
        ready = threading.Event()
        with cf.ThreadPoolExecutor(1) as pool:
            future = pool.submit(stream, prompt(1024), ready)
            if not ready.wait(60): raise RuntimeError("Decode neighbour failed to start")
            started = time.monotonic()
            mixed = call(ids, salt=f"tail-{length}-mixed-{time.time_ns()}")
            neighbour = future.result()
        item = {"solo": solo, "mixed": mixed, "neighbour": neighbour, "mixed_started": started,
                "overlap": started < neighbour["finished"], "same_tokens": tokens(solo) == tokens(mixed)}
        results[str(length)] = item
        (args.output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print("prefill tail", length, "same", item["same_tokens"], "overlap", item["overlap"], flush=True)
    (args.output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
