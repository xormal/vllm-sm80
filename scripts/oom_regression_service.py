#!/usr/bin/env python3
"""Prepared, NOT EXECUTED: manually run ONE phase on an isolated TP2 server.

Never starts/stops/restarts a model; no production endpoint or automatic retries.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from oom_regression_guard import REPO, maintenance_guard, isolated_service_guard, unit_state


PHASES = {
    "sparse196k": ("acceptance_streams.py", "--cases", "pure1", "--prompt-tokens", "196608", "--tokens", "2048"),
    "sparse213k": ("acceptance_streams.py", "--cases", "pure1", "--prompt-tokens", "212992", "--tokens", "2048"),
    "sparse-limit": ("acceptance_streams.py", "--cases", "pure1", "--prompt-tokens", "262140", "--tokens", "4"),
    "mixed-boundary": ("acceptance_streams.py", "--cases", "mixed34k", "--prompt-tokens", "49147", "--tokens", "2048", "--distinct-clef"),
    "queue40": ("acceptance_streams.py", "--cases", "pure40", "--prompt-tokens", "8192", "--tokens", "1024"),
    "mixed-queue": ("acceptance_streams.py", "--cases", "mixedqueue34k", "--prompt-tokens", "8192", "--tokens", "1024", "--distinct-clef"),
    "images": ("acceptance_fixtures.py", "kartinki", "clefkart"),
    "quality": ("acceptance_fixtures.py", "baza", "clef"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("output", type=Path)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--server-log", required=True, type=Path)
    parser.add_argument("--maintenance-window", action="store_true")
    args = parser.parse_args()
    maintenance_guard(args.maintenance_window)
    initial = isolated_service_guard(args.unit)
    log_path = args.server_log.resolve()
    if not log_path.is_relative_to(REPO.parent / "state/acceptance"):
        raise ValueError("Require candidate acceptance server log")
    if initial.get("StandardOutput") != "append:" + str(log_path):
        raise ValueError("Log must match the test unit's StandardOutput=append path")
    log_stat = log_path.stat()
    offset = log_stat.st_size
    if args.output.exists():
        raise RuntimeError("Use a fresh output directory")
    args.output.mkdir(parents=True)
    report = {"status": "running", "phase": args.phase, "initial": initial, "epoch": time.time(),
              "server_log": str(log_path), "server_log_offset": offset,
              "coverage": "HTTP workload only; exact MTP/sparse/MLP shapes require diagnostic trace"}
    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
    def collect_errors():
        if log_path.stat().st_ino != log_stat.st_ino or log_path.stat().st_size < offset:
            raise RuntimeError("Server log rotated/truncated: coverage of errors is incomplete")
        errors = []
        with log_path.open("rb") as server_log:
            server_log.seek(offset)
            for raw in server_log:
                line = raw.decode(errors="replace")
                if any(term in line for term in ("OutOfMemoryError", "CUDA out of memory", "EngineDeadError",
                                                 "illegal memory access", "device-side assert")):
                    errors.append(line.strip())
        return errors
    save()
    command = [sys.executable, str(REPO / "scripts" / PHASES[args.phase][0]),
               str(args.output / "requests"), *PHASES[args.phase][1:]]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    process = None
    start = time.monotonic()
    memory_events_before = None
    try:
        with (args.output / "client.log").open("w") as client_log, (args.output / "samples.jsonl").open("w") as samples:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=client_log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            while True:
                current = unit_state(args.unit)
                if (current.get("ActiveState") != "active" or any(
                        current.get(k) != initial.get(k) for k in ("InvocationID", "MainPID", "NRestarts"))):
                    raise RuntimeError("Server stopped/restarted: do not retry or count recovered requests as pass")
                sample = {"epoch": time.time(), "unit": current}
                group = Path("/sys/fs/cgroup") / current["ControlGroup"].lstrip("/")
                for key in ("memory.current", "memory.peak", "memory.events"):
                    sample[key] = (group / key).read_text().strip()
                events = dict((key, int(value)) for key, value in
                              (line.split() for line in sample["memory.events"].splitlines()))
                if memory_events_before is None:
                    memory_events_before = events
                if any(events.get(key, 0) > memory_events_before.get(key, 0)
                       for key in ("max", "oom", "oom_kill", "oom_group_kill")):
                    report["memory_events_at_failure"] = events
                    raise RuntimeError("Server hit its RAM limit/OOM during the phase")
                sample["gpu01"] = subprocess.check_output(
                    ["nvidia-smi", "-i", "0,1", "--query-gpu=index,memory.total,memory.reserved,memory.used,memory.free,utilization.gpu",
                     "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
                with urllib.request.urlopen("http://127.0.0.1:8086/metrics", timeout=5) as response:
                    sample["metrics"] = response.read().decode()
                samples.write(json.dumps(sample) + "\n")
                samples.flush()
                if process.poll() is not None:
                    if process.returncode:
                        raise RuntimeError(f"Client phase failed: {process.returncode}")
                    break
                if time.monotonic() - start > 1800:
                    raise TimeoutError("Phase wall-time exceeded 1800 seconds")
                time.sleep(1)
        errors = collect_errors()
        report["runtime_errors"] = errors
        if errors:
            raise RuntimeError("Runtime error found even though client phase completed")
        report.update(status="workload_passed_path_coverage_unverified", release_acceptance=False)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        if process is not None and process.poll() is None:
            # Only the test client group, never the model service or its observers.
            import signal
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        try:
            report["runtime_errors"] = collect_errors()
            if report["runtime_errors"]:
                report["status"] = "failed"
        except Exception as exc:
            report.update(status="failed", log_collection_error=repr(exc))
        report["seconds"] = time.monotonic() - start
        save()
        if report["status"] == "failed" and sys.exc_info()[0] is None:
            raise RuntimeError("Final evidence check failed; inspect result.json")


if __name__ == "__main__":
    main()
