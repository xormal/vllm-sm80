#!/usr/bin/env python3
"""Fixed localhost SSE workload; intervals are between chunks, not imaginary tokens."""
import argparse
import concurrent.futures as cf
import importlib.util
import json
import math
import re
from pathlib import Path
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8086"
MODEL = "Qwen3.8-27B-ABLITERATED-candidate"
REPO = Path(__file__).resolve().parent.parent


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.loads(response.read())


def get_metrics():
    with urllib.request.urlopen(URL + "/metrics", timeout=2) as response:
        return response.read().decode()


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--cases", default="pure1,pure8,mixed2k,mixed8k,mixed34k")
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = REPO / "sm70/integrations/vllm_sm70/clef/proverka_stenda.py"
    spec = importlib.util.spec_from_file_location("fixtures", source)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    docs = fixture.текст_доков(300000)
    corpus = post("/tokenize", {"model": MODEL, "prompt": docs[:20000], "add_special_tokens": False})["tokens"]
    all_results = []
    for repeat in range(args.repeat):
        for case in args.cases.split(","):
            if case not in ("pure1", "pure8", "pure40", "mixed2k", "mixed8k", "mixed34k", "mixedqueue34k"):
                raise ValueError(case)
            count = {"pure1": 1, "pure40": 40, "mixedqueue34k": 20}.get(case, 8)
            clef_count = 10 if case == "mixedqueue34k" else 4
            ready_count = min(count, 8)
            clef_length = {"mixed2k": 2048, "mixed8k": 8192, "mixed34k": 34816, "mixedqueue34k": 34816}.get(case)
            path = args.output / f"{case}-{repeat}"
            path.mkdir(exist_ok=True)
            (path / "metrics-before.txt").write_text(get_metrics())
            started = time.monotonic()
            barrier = threading.Barrier(count)
            ready = threading.Event()
            lock = threading.Lock()
            seen = set()

            def stream(index):
                item = {"index": index, "events": [], "ok": False}
                try:
                    head = post("/tokenize", {"model": MODEL, "prompt": f"Document {case}-{repeat}-{index}. Continue the exposition.\n",
                                              "add_special_tokens": False})["tokens"]
                    ids = head + corpus[:2048 - len(head)]
                    req = urllib.request.Request(URL + "/v1/completions", json.dumps({
                        "model": MODEL, "prompt": ids, "max_tokens": args.tokens,
                        "temperature": 0, "seed": 17, "ignore_eos": True, "logprobs": 1,
                        "stream": True, "stream_options": {"include_usage": True}}).encode(),
                        {"Content-Type": "application/json"})
                    barrier.wait(timeout=30)
                    item["sent"] = time.monotonic() - started
                    with urllib.request.urlopen(req, timeout=600) as response:
                        for line in response:
                            if not line.startswith(b"data: "): continue
                            if line.strip() == b"data: [DONE]": break
                            value = json.loads(line[6:])
                            if "error" in value: raise RuntimeError(value["error"])
                            if value.get("usage"): item["usage"] = value["usage"]
                            if value.get("choices"):
                                choice = value["choices"][0]
                                tokens = (choice.get("logprobs") or {}).get("tokens") or []
                                if tokens:
                                    item["events"].append({"t": time.monotonic() - started, "tokens": len(tokens)})
                                    with lock:
                                        seen.add(index)
                                        if len(seen) >= ready_count: ready.set()
                    item["ok"] = item.get("usage", {}).get("completion_tokens") == args.tokens
                    if not item["ok"]: item["error"] = "Incomplete token count"
                except Exception as exc:
                    item["error"] = repr(exc)
                finally:
                    item["finished"] = time.monotonic() - started
                    (path / f"chat-{index}.json").write_text(json.dumps(item, indent=2))
                return item

            def clef(index):
                record = {"model": "clef", "max_length": clef_length,
                          "state": f"Record {case}-{repeat}-{index}. The switch is ON.\n" + docs[:clef_length * 4],
                          "questions": {"switch": {"type": "choice", "instructions": "What is the switch position?",
                                        "criteria": {"on": "The switch is ON.", "off": "The switch is OFF."}}}}
                item = {"index": index, "sent": time.monotonic() - started, "ok": False}
                try:
                    item["response"] = post("/v1/systemone", record)
                    item["ok"] = "answers" in item["response"]
                except Exception as exc: item["error"] = repr(exc)
                item["finished"] = time.monotonic() - started
                (path / f"clef-{index}.json").write_text(json.dumps(item, ensure_ascii=False, indent=2))
                return item

            monitor_stop = threading.Event()
            metric_samples = []
            def monitor():
                while not monitor_stop.is_set():
                    sample = {"t": time.monotonic() - started}
                    try:
                        text = get_metrics()
                        for key in ("num_requests_running", "num_requests_waiting", "num_preemptions_total"):
                            sample[key] = sum(float(value) for value in re.findall(
                                rf'^vllm:{key}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)', text, re.M))
                    except Exception as exc: sample["error"] = repr(exc)
                    metric_samples.append(sample)
                    monitor_stop.wait(.5)
            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()
            with cf.ThreadPoolExecutor(count + clef_count) as pool:
                chats = [pool.submit(stream, i) for i in range(count)]
                clefs = []
                coverage = ready.wait(timeout=120)
                seen_at_gate = len(seen)
                if clef_length and coverage:
                    clefs = [pool.submit(clef, i) for i in range(clef_count)]
                chat_results = [future.result() for future in chats]
                clef_results = [future.result() for future in clefs]
            monitor_stop.set()
            monitor_thread.join(timeout=3)
            (path / "metrics-during.json").write_text(json.dumps(metric_samples, indent=2))
            gaps = [b["t"] - a["t"] for item in chat_results for a, b in zip(item["events"], item["events"][1:])]
            total = sum(item.get("usage", {}).get("completion_tokens", 0) for item in chat_results)
            seconds = time.monotonic() - started
            overlap = (not clef_length or (len(clef_results) == clef_count and
                       max(item["sent"] for item in clef_results) < min(item["finished"] for item in chat_results)))
            result = {"case": case, "repeat": repeat, "seconds": seconds,
                      "start_gate_met": coverage, "start_gate_requests": ready_count, "seen_at_gate": seen_at_gate,
                      "chat_requests": count, "clef_requests": len(clef_results),
                      "max_engine_waiting": max((x.get("num_requests_waiting", 0) for x in metric_samples), default=None), "clef_started_before_any_chat_finished": overlap,
                      "ok": coverage and overlap and all(x["ok"] for x in chat_results + clef_results),
                      "total_tokens": total, "aggregate_tokens_per_s_including_prefill": total / seconds,
                      "ttft_s": [x["events"][0]["t"] - x["sent"] if x["events"] else None for x in chat_results],
                      "chunk_gap_p50_s": percentile(gaps, .50), "chunk_gap_p95_s": percentile(gaps, .95),
                      "chunk_gap_p99_s": percentile(gaps, .99), "chunk_gap_max_s": max(gaps) if gaps else None,
                      "clef_latency_s": [x["finished"] - x["sent"] for x in clef_results]}
            all_results.append(result)
            (args.output / "results.json").write_text(json.dumps(all_results, indent=2))
            (path / "metrics-after.txt").write_text(get_metrics())
            print(json.dumps(result), flush=True)
            if not result["ok"]: raise RuntimeError("Workload or overlap gate failed; inspect per-request artifacts")


if __name__ == "__main__":
    main()
