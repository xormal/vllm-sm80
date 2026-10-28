#!/usr/bin/env python3
"""Short semantic/seed/tool gates and sampled entropy bounds from top-20 logprobs."""
import argparse
import json
import math
from pathlib import Path
import time
import urllib.request

URL = "http://127.0.0.1:8086"
MODEL = "Qwen3.8-27B-ABLITERATED-candidate"


def post(body):
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as response: return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    cases = [
        ("arithmetic", "Compute 19 * 23. Answer only the integer.", False, "437"),
        ("reasoning", "A box has 7 red and 5 blue balls. Two balls are removed without replacement. What is the probability both are blue? Give the exact fraction.", True, "5/33"),
        ("code", "Write only a short Python function that returns the median of a sorted list of numbers, handling both odd and even lengths.", False, None),
        ("russian", "Почему при суммировании чисел с плавающей точкой порядок операций может влиять на результат? Ответь кратко по-русски.", False, None),
    ]
    for name, prompt, thinking, expected in cases:
        body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "seed": 17, "max_tokens": 1536 if thinking else 384, "logprobs": True, "top_logprobs": 20,
                "chat_template_kwargs": {"enable_thinking": thinking}}
        started = time.monotonic()
        response = post(body)
        choice = response["choices"][0]
        content = choice["message"].get("content") or ""
        entropy = []
        for entry in (choice.get("logprobs") or {}).get("content") or []:
            top = entry.get("top_logprobs") or []
            p = [math.exp(x["logprob"]) for x in top]
            tail = max(0.0, 1.0 - sum(p))
            known = -sum(x * math.log(x) for x in p if x)
            lower = known - (tail * math.log(tail) if tail else 0)
            upper = lower + (tail * math.log(248320 - len(p)) if tail else 0)
            entropy.append({"top20_mass": sum(p), "H_lower_nats": lower, "H_upper_nats": upper,
                            "top1_top2_probability_margin": p[0] - p[1] if len(p) > 1 else None})
        item = {"name": name, "seconds": time.monotonic() - started, "response": response,
                "expected": expected, "expected_present": expected in content.replace(" ", "") if expected else None,
                "entropy": entropy}
        (args.output / (name + ".json")).write_text(json.dumps(item, ensure_ascii=False, indent=2))
        results.append({k:v for k,v in item.items() if k not in ("response", "entropy")})
        print(name, repr(content[:120]), "expected", item["expected_present"], flush=True)
    seed_body = {"model": MODEL, "messages": [{"role": "user", "content": "Придумай четыре необычных названия для научной библиотеки."}],
                 "temperature": .7, "top_p": .9, "seed": 17, "max_tokens": 128,
                 "chat_template_kwargs": {"enable_thinking": False}, "logprobs": True}
    seed = [post(seed_body) for _ in range(2)]
    (args.output / "seed.json").write_text(json.dumps(seed, ensure_ascii=False, indent=2))
    same_seed = seed[0]["choices"][0]["message"] == seed[1]["choices"][0]["message"]
    print("seed repeat messages identical", same_seed, flush=True)
    tool_body = {"model": MODEL, "messages": [{"role": "user", "content": "Call get_weather for Tbilisi, then wait for its result."}],
                 "temperature": 0, "max_tokens": 128, "tool_choice": "auto",
                 "chat_template_kwargs": {"enable_thinking": False},
                 "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city",
                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}
    tool = post(tool_body)
    (args.output / "tool.json").write_text(json.dumps(tool, ensure_ascii=False, indent=2))
    calls = tool["choices"][0]["message"].get("tool_calls") or []
    tool_ok = bool(calls) and calls[0]["function"]["name"] == "get_weather" and json.loads(calls[0]["function"]["arguments"]).get("city", "").lower() == "tbilisi"
    print("tool call parsed correctly", tool_ok, flush=True)
    summary = {"cases": results, "same_seed_messages": same_seed, "tool_ok": tool_ok,
               "entropy_note": "Top-20 bounds, not full entropy; manual code/reasoning review and baseline alignment still required."}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
