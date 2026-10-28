#!/usr/bin/env python3
"""Controlled localhost smoke, no CUDA imports; not a quality/performance gate."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("basic", "boundaries", "concurrent", "clef", "long"), default="basic")
    args = parser.parse_args()
    url = "http://127.0.0.1:8086"
    results = []

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as response:
            raw = response.read()
            return json.loads(raw) if raw else None

    def record(name, function):
        start = time.monotonic()
        try:
            value = function()
            result = {"name": name, "ok": True, "seconds": time.monotonic() - start, "result": value}
        except Exception as exc:
            result = {"name": name, "ok": False, "seconds": time.monotonic() - start, "error": repr(exc)}
            raise
        finally:
            results.append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps(result, ensure_ascii=False), flush=True)

    models = request("/v1/models")
    model = models["data"][0]["id"]

    def completion(prompt, count=32):
        value = request("/v1/completions", {"model": model, "prompt": prompt,
                        "max_tokens": count, "temperature": 0.0, "ignore_eos": True})
        assert value["usage"]["completion_tokens"] == count, value
        assert value["choices"][0]["text"], value
        return value

    def token_ids(text):
        return request("/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})["tokens"]

    if args.stage == "basic":
        record("models", lambda: models)
        record("completion", lambda: completion("The capital of France is"))
        record("chat", lambda: request("/v1/chat/completions", {"model": model,
               "messages": [{"role": "user", "content": "Сколько будет 2 + 2? Ответь одним числом."}],
               "temperature": 0, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}}))
    elif args.stage in ("boundaries", "concurrent", "long"):
        corpus = token_ids("The system stores records in memory. Each request has its own identifier. "
                           "Результаты независимых запросов должны сохраняться раздельно.\n" * 20)
        def prompt(length, salt):
            head = token_ids(f"Validation document {salt}. Continue the text.\n")
            return head + (corpus * ((length // len(corpus)) + 1))[:length - len(head)]
        if args.stage in ("boundaries", "long"):
            lengths = (8191, 8192, 8193) if args.stage == "boundaries" else (32768, 131072, 212992)
            for length in lengths:
                record(f"prefill-{length}", lambda n=length: completion(prompt(n, str(n)), 16))
        else:
            for concurrency in (8, 16):
                prompts = [prompt(1024, f"C{concurrency}-{i}") for i in range(concurrency)]
                def run_batch():
                    with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                        return list(pool.map(lambda ids: completion(ids, 64), prompts))
                record(f"concurrent-{concurrency}", run_batch)
    else:
        record("clef-short", lambda: request("/v1/systemone", {
            "model": model, "state": "The switch is ON. The room is empty.",
            "questions": {"switch": {"type": "choice", "instructions": "What is the switch position?",
                "criteria": {"on": "The switch is ON.", "off": "The switch is OFF."}}}}))


if __name__ == "__main__":
    main()
