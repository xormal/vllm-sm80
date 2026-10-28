#!/usr/bin/env python3
"""Five alternating baseline/candidate pairs, identical warmup and measured workloads."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parent.parent
STATE = REPO.parent / "state/acceptance"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--start-pair", type=int, default=1)
    parser.add_argument("--wait-for", type=Path)
    args = parser.parse_args()
    if args.wait_for:
        deadline = time.monotonic() + 1800
        while not args.wait_for.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Previous workload did not finish: {args.wait_for}")
            time.sleep(1)
        previous = json.loads(args.wait_for.read_text())
        if not previous or not all(x.get("ok") for x in previous):
            raise RuntimeError("Previous workload failed; inspect it before switching variants")
    progress = []
    def save():
        (STATE / "ab-progress.json").write_text(json.dumps(progress, indent=2))
    for pair in range(args.start_pair, args.start_pair + args.pairs):
        for variant in ("baseline", "candidate"):
            label = f"ab-{variant}-{pair}"
            entry = {"pair": pair, "variant": variant, "label": label, "started": time.time(), "phase": "startup"}
            progress.append(entry); save()
            output = STATE / label
            output.mkdir(exist_ok=True)
            with (output / "controller.log").open("w") as log:
                try:
                    subprocess.run(["python3", "-I", "scripts/acceptance_service.py", variant, label], cwd=REPO,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                    def client(phase, script, *argv, allow_failure=False):
                        entry["phase"] = phase; save()
                        print(label, phase, time.time(), flush=True)
                        cmd = ["systemd-run", "--user", "--scope", "--quiet", "--collect",
                               f"--unit=kandidat-abclient-{pair}-{variant}-{phase}", "--slice=kandidat-test.slice",
                               "-p", "MemoryMax=3G", "-p", "MemorySwapMax=0", "-p", "CPUQuota=200%", "-p", "TasksMax=128",
                               "env", "CUDA_VISIBLE_DEVICES=", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
                               "python3", "-I", str(REPO / "scripts" / script), *map(str, argv)]
                        with (output / (phase + ".log")).open("w") as client_log:
                            result = subprocess.run(cmd, cwd=REPO, stdout=client_log, stderr=subprocess.STDOUT)
                        if result.returncode:
                            if not allow_failure:
                                raise subprocess.CalledProcessError(result.returncode, cmd)
                            entry.setdefault("failed_phases", []).append({"phase":phase,"returncode":result.returncode})
                            save()
                    client("quality", "acceptance_fixtures.py", output / "quality", "baza", "clef")
                    client("semantics", "acceptance_semantics.py", output / "semantics")
                    client("warmup", "acceptance_streams.py", output / "warmup", "--cases", "pure1,pure8,mixed34k", "--tokens", "512")
                    client("measure", "acceptance_streams.py", output / "measure", "--cases", "pure1,pure8,mixed34k", "--tokens", "512")
                    if pair == 1:
                        client("state", "acceptance_state.py", output / "state")
                        client("recall", "acceptance_recall_distinct.py", output / "recall-distinct")
                        client("preemption", "acceptance_streams.py", output / "preemption", "--cases", "pure8",
                               "--prompt-tokens", "49147", "--tokens", "2048", allow_failure=True)
                    entry.update(phase="done_with_failures" if entry.get("failed_phases") else "done", finished=time.time())
                except Exception as exc:
                    entry.update(phase="failed", finished=time.time(), error=repr(exc)); save()
                    raise
            save()


if __name__ == "__main__":
    main()
