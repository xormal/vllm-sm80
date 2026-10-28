#!/usr/bin/env python3
"""64 MiB+ wire bodies on the private candidate; shared immutable client payload."""
import argparse
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8086"
MODEL = "Qwen3.8-27B-ABLITERATED-candidate"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--route", choices=("clef", "chat", "tokenize", "chat-padded"), required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--mib", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 40 or not 61 <= args.mib <= 80:
        raise ValueError("Outside the authorized bounded stress envelope")
    args.output.mkdir(parents=True, exist_ok=True)
    line = "The switch is ON. This is a validation record with a bounded model context.\n"
    text = (line * (args.mib * 1048576 // len(line) + 1))[:args.mib * 1048576]
    if args.route == "clef":
        route = "/v1/systemone"
        body = {"model": MODEL, "state": text, "max_length": 34816,
                "questions": {"switch": {"type": "choice", "instructions": "What is the switch position?",
                              "criteria": {"on": "The switch is ON.", "off": "The switch is OFF."}}}}
    elif args.route == "chat-padded":
        route = "/v1/chat/completions"
        body = {"model": MODEL, "messages": [{"role": "user", "content": "Compute 19 * 23. Answer only the integer."}],
                "max_tokens": 32, "temperature": 0, "seed": 17,
                "chat_template_kwargs": {"enable_thinking": False}}
    elif args.route == "chat":
        route = "/v1/chat/completions"
        body = {"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": 1}
    else:
        route = "/tokenize"
        body = {"model": MODEL, "prompt": text, "add_special_tokens": False}
    payload = json.dumps(body, ensure_ascii=False).encode()
    del body, text
    padding = args.mib * 1048576 if args.route == "chat-padded" else 0
    if padding:
        # Legal JSON whitespace: a large accepted HTTP body with a small model
        # context. This is NOT a 64 MiB semantic prompt or a long-context test.
        payload = b" " * padding + payload
    meta = {"route": route, "wire_bytes_per_request": len(payload), "concurrency": args.concurrency,
            "case":args.route,"json_whitespace_padding_bytes":padding,
            "payload_sha256": hashlib.sha256(payload).hexdigest(), "started_epoch": time.time()}
    (args.output / "input.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    barrier = threading.Barrier(args.concurrency)
    def send(index):
        item = {"index": index, "ok": False}
        try:
            request = urllib.request.Request(URL + route, payload, {"Content-Type": "application/json"})
            barrier.wait(timeout=30)
            start = time.monotonic()
            item["sent_epoch"] = time.time()
            try:
                response = urllib.request.urlopen(request, timeout=600)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                item["status"] = response.status
                received, prefix = 0, b""
                while chunk := response.read(1048576):
                    received += len(chunk)
                    if len(prefix) < 65536: prefix += chunk[:65536 - len(prefix)]
            item["response_bytes"] = received
            item["response_prefix"] = prefix[:4096].decode(errors="replace")
            count = re.search(rb'"count"\s*:\s*(\d+)', prefix)
            if count: item["tokenize_count"] = int(count[1])
            item["seconds"] = time.monotonic() - start
            item["ok"] = response.status in ((400, 413) if args.route == "chat" else (200,))
            if args.route == "clef" and response.status == 200:
                parsed = json.loads(prefix)
                item["usage"] = parsed.get("usage")
                item["answers"] = parsed.get("answers")
                item["ok"] = bool(parsed.get("answers")) and parsed["usage"]["input_tokens"] <= 34816
            if args.route == "chat-padded" and response.status == 200:
                parsed = json.loads(prefix)
                item["usage"] = parsed.get("usage")
                item["answer"] = parsed["choices"][0]["message"].get("content", "").strip()
                item["ok"] = item["answer"] == "437"
        except Exception as exc:
            item["error"] = repr(exc)
        finally:
            (args.output / f"request-{index}.json").write_text(json.dumps(item, ensure_ascii=False, indent=2))
        return item
    with cf.ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(send, range(args.concurrency)))
    output = dict(meta, results=results, ok=all(item["ok"] for item in results))
    (args.output / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps({"ok": output["ok"], "statuses": [x.get("status") for x in results],
                      "seconds": [x.get("seconds") for x in results]}), flush=True)
    if not output["ok"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
