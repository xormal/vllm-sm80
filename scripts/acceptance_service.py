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
    parser.add_argument("--fallback-rows", type=int, choices=(0,1024), default=0)
    parser.add_argument("--scheduler-trace", action="store_true",
                        help="Separate diagnostic process; not performance acceptance")
    parser.add_argument("--resume-admission",type=int,choices=(0,1),default=None)
    parser.add_argument("--gran-conv-all",type=int,choices=(0,1),default=0)
    parser.add_argument("--conv-meta-fused",type=int,choices=(0,1),default=0)
    parser.add_argument("--live",action="store_true",
                        help="Authorized live-load trial on 8085 with production model aliases")
    args = parser.parse_args()
    if args.conv_meta_fused and (args.variant != "candidate" or not args.gran_conv_all):
        raise ValueError("Fused conv metadata requires the candidate convolution correction")
    if args.resume_admission is None:
        args.resume_admission = int(args.variant == "candidate")
    if args.live and (args.variant != "candidate" or args.scheduler_trace
                     or args.resume_admission != 1 or args.gran_conv_all != 1
                     or args.fallback_rows != 0 or args.conv_meta_fused != 0):
        raise ValueError("Live trial requires the selected admission1/conv1/chunks0/fused0 profile")
    if args.live:
        import socket
        with socket.socket() as probe:
            probe.settimeout(1)
            if probe.connect_ex(("127.0.0.1",8085)) == 0:
                raise RuntimeError("Port 8085 is already occupied")
    if args.variant == "baseline" and args.gran_conv_all:
        raise ValueError("Reference must retain original convolution metadata")
    if args.variant == "baseline" and args.resume_admission:
        raise ValueError("Reference must retain the pre-fix admission behavior")
    if args.variant == "baseline" and args.fallback_rows:
        raise ValueError("The reference has no chunked fallback")
    if args.scheduler_trace and args.variant != "candidate":
        raise ValueError("Trace is restricted to the candidate diagnostic process")
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
    deadline = time.monotonic() + 15
    while True:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8086/metrics", timeout=3) as response:
                metrics = response.read().decode()
            inflight = sum(float(x) for x in re.findall(
                r'^vllm:num_requests_(?:running|waiting)(?:\{[^}]*\})?\s+([0-9.eE+-]+)', metrics, re.M))
        except OSError:
            state = run("systemctl", "--user", "show", old, "-p", "ActiveState", "--value", capture_output=True).stdout.strip()
            if state not in ("failed", "inactive"):
                raise RuntimeError("Cannot establish whether active validation requests are finished")
            break
        if not inflight:
            break
        if time.monotonic() > deadline:
            raise RuntimeError(f"Refusing to interrupt {inflight} validation requests")
        time.sleep(.5)
    # A stopped transient unit can be collected during a standalone GPU test.
    # The idle/dead check above still applies; only skip an already absent unit.
    load_state = run("systemctl", "--user", "show", old, "-p", "LoadState",
                     "--value", capture_output=True).stdout.strip()
    if load_state != "not-found":
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
    for env in (f"CANDIDATE_VARIANT={args.variant}", f"FA2SM80_BLOCKPIPE_FALLBACK_ROWS={args.fallback_rows}",
                f"FA2SM80_RESUME_ADMISSION={args.resume_admission}",
                f"FA2SM70_GRAN_CONV_ALL={args.gran_conv_all}",
                f"FA2SM80_CONV_META_FUSED={args.conv_meta_fused}",
                "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "TORCHINDUCTOR_COMPILE_THREADS=1"):
        cmd += ["--setenv=" + env]
    if args.scheduler_trace:
        cmd += ["--setenv=CANDIDATE_SCHEDULER_TRACE=" + str(output / "scheduler.jsonl")]
    cmd += [str(REPO / ("scripts/serve_candidate_live.sh" if args.live else
                       "scripts/serve_acceptance_trace.sh" if args.scheduler_trace
                       else "scripts/serve_acceptance_variant.sh"))]
    run(*cmd)
    (STATE / "active-unit").write_text(unit + "\n")
    (output / "launch.json").write_text(json.dumps({"variant": args.variant, "fallback_rows": args.fallback_rows, "unit": unit,
             "epoch": time.time(), "command": cmd, "reference_overlay": manifest,
             "scheduler_trace": args.scheduler_trace,"resume_admission":args.resume_admission,
             "gran_conv_all":args.gran_conv_all,"conv_meta_fused":args.conv_meta_fused,
             "live":args.live,"port":8085 if args.live else 8086,
             "repository_head":run("git","rev-parse","HEAD",capture_output=True).stdout.strip()}, indent=2))
    started = time.monotonic()
    while time.monotonic() - started < 900:
        state = run("systemctl", "--user", "show", unit, "-p", "ActiveState", "--value", capture_output=True).stdout.strip()
        if state in ("failed", "inactive"): raise RuntimeError(f"Startup failed: {unit}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{8085 if args.live else 8086}/health", timeout=1) as response:
                if response.status == 200:
                    startup_log = (output / "server.log").read_text(errors="replace")
                    if ("НАСТРОЙКА НЕ УДАЛАСЬ" in startup_log or
                        not all(f"ранг {rank}/2: поправка на" in startup_log for rank in (0,1))):
                        run("systemctl", "--user", "stop", unit)
                        raise RuntimeError("Clef did not initialize on both TP ranks; reference is invalid")
                    print(f"READY {args.variant} {unit} {time.monotonic()-started:.1f}s", flush=True)
                    return
        except OSError: pass
        time.sleep(1)
    raise TimeoutError(unit)


if __name__ == "__main__":
    main()
