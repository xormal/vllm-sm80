#!/usr/bin/env python3
"""Switch only candidate transient services between verified runtime variants."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request

REPO = Path(__file__).resolve().parent.parent
STATE = REPO.parent / "state/acceptance"


def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("baseline", "candidate"))
    parser.add_argument("label")
    args = parser.parse_args()
    if not re.fullmatch("[a-z0-9-]+", args.label): raise ValueError(args.label)
    production = run("systemctl", "--user", "show", "sm80-vllm.service", "-p", "ActiveState", "--value", capture_output=True).stdout.strip()
    if production not in ("inactive", "failed"): raise RuntimeError("Production must remain stopped")
    manifest = json.loads((STATE / "baseline-overlay.json").read_text())
    for name, record in manifest.items():
        digest = hashlib.sha256((STATE / "baseline-overlay" / name).read_bytes()).hexdigest()
        if digest != record["sha256"]: raise RuntimeError(f"Reference changed: {name}")
    old = (STATE / "active-unit").read_text().strip()
    if not re.fullmatch(r"kandidat-(?:validation|acceptance-[a-z0-9-]+)\.service", old):
        raise RuntimeError("Refusing to stop unrelated service")
    run("systemctl", "--user", "stop", old)
    output = STATE / args.label
    output.mkdir(exist_ok=True)
    unit = f"kandidat-acceptance-{args.label}.service"
    cmd = ["systemd-run", "--user", f"--unit={unit}"]
    for prop in ("Type=exec", "Slice=sm80.slice", "Restart=no", "MemoryHigh=20G", "MemoryMax=21G",
                 "MemorySwapMax=0", "CPUQuota=1600%", "TasksMax=4096", "OOMPolicy=kill",
                 "OOMScoreAdjust=900", "TimeoutStopSec=45", "KillMode=control-group",
                 f"WorkingDirectory={REPO}", f"StandardOutput=append:{output}/server.log", "StandardError=inherit"):
        cmd += ["--property=" + prop]
    for env in (f"CANDIDATE_VARIANT={args.variant}", "FA2SM80_BLOCKPIPE_FALLBACK_ROWS=0",
                "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "TORCHINDUCTOR_COMPILE_THREADS=1"):
        cmd += ["--setenv=" + env]
    cmd += [str(REPO / "scripts/serve_acceptance_variant.sh")]
    run(*cmd)
    (STATE / "active-unit").write_text(unit + "\n")
    (output / "launch.json").write_text(json.dumps({"variant": args.variant, "unit": unit,
             "epoch": time.time(), "command": cmd, "reference_overlay": manifest}, indent=2))
    started = time.monotonic()
    while time.monotonic() - started < 900:
        state = run("systemctl", "--user", "show", unit, "-p", "ActiveState", "--value", capture_output=True).stdout.strip()
        if state in ("failed", "inactive"): raise RuntimeError(f"Startup failed: {unit}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8086/health", timeout=1) as response:
                if response.status == 200:
                    print(f"READY {args.variant} {unit} {time.monotonic()-started:.1f}s", flush=True)
                    return
        except OSError: pass
        time.sleep(1)
    raise TimeoutError(unit)


if __name__ == "__main__":
    main()
