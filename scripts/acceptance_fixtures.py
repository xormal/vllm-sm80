#!/usr/bin/env python3
"""Run repository acceptance fixtures on the isolated validation endpoint."""
import argparse
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import time

REPO = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("modes", nargs="+")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = REPO / "sm70/integrations/vllm_sm70/clef/proverka_stenda.py"
    spec = importlib.util.spec_from_file_location("proverka_stenda", source)
    ps = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ps
    spec.loader.exec_module(ps)
    ps.ИМЯ = "Qwen3.8-27B-ABLITERATED-candidate"
    post = ps.post
    def recorded_post(port, path, body, timeout=600):
        if port != 8086:
            raise ValueError("Only isolated port 8086 is allowed")
        started = time.time()
        response, seconds = post(port, path, body, timeout=min(timeout, 600))
        with (args.output / "requests.jsonl").open("a") as log:
            log.write(json.dumps({"started": started, "path": path, "seconds": seconds,
                                  "response": response}, ensure_ascii=False) + "\n")
        if response.get("http_error", 0) >= 500:
            raise RuntimeError(f"Server failure: {response}")
        return response, seconds
    ps.post = recorded_post
    for mode in args.modes:
        output = str(args.output / (mode + ".json"))
        print(f"START {mode} {time.time()}", flush=True)
        if mode == "baza": ps.baza(8086, output)
        elif mode == "clef": ps.clef(8086, output, 2)
        elif mode == "dlinnye": ps.dlinnye(8086, output, 34816)
        elif mode == "clefkart": ps.clefkart(8086, output)
        elif mode == "kartinki": ps.kartinki(8086, output, True)
        elif mode == "igla": ps.igla(8086, 245000, output)
        elif mode in ("recall", "prefix2"):
            sys.argv = [mode, "8086", output, "8", "2"]
            runpy.run_path(str(REPO / f"sm70/tests/server/probe_{mode}.py"), run_name="__main__")
        else: raise ValueError(mode)
        print(f"DONE {mode} {time.time()}", flush=True)


if __name__ == "__main__":
    main()
