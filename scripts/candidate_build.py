#!/usr/bin/env python3
"""Run a build in a separate capped cgroup; never join production sm80.slice."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO.parent
STATE = ROOT / "state"
GIB = 1024**3


def available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable unavailable")


def main():
    args = sys.argv[1:]
    if args[:1] == ["--"]:
        args = args[1:]
    if not args:
        print("Usage: scripts/candidate_build.py -- COMMAND [ARGS...]", file=sys.stderr)
        return 2
    STATE.mkdir(exist_ok=True)
    with (STATE / "build.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("A candidate build already holds the build lock.", file=sys.stderr)
            return 75
        available = available_memory()
        if available < 16 * GIB:
            print(f"Build not started: MemAvailable={available/GIB:.1f} GiB; need 16 GiB.", file=sys.stderr)
            return 75
        if not Path.home().joinpath(".config/systemd/user/kandidat-build.slice").is_file():
            print("Missing kandidat-build.slice; install configs/kandidat-build.slice first.", file=sys.stderr)
            return 78
        env = os.environ.copy()
        env.update(MAX_JOBS="2", NVCC_THREADS="1", CMAKE_BUILD_PARALLEL_LEVEL="2",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", TORCHINDUCTOR_COMPILE_THREADS="1",
                   CUDA_VISIBLE_DEVICES="", TORCH_CUDA_ARCH_LIST="8.0",
                   MAKEFLAGS="-j2", CMAKE_GENERATOR="Ninja", FA2SM70_BUILD_ONLY="1")
        unit = f"kandidat-build-{os.getpid()}-{time.time_ns()}.scope"
        command = ["systemd-run", "--user", "--scope", "--quiet", "--collect",
                   f"--unit={unit}", "--slice=kandidat-build.slice",
                   "-p", "MemoryHigh=5G", "-p", "MemoryMax=6G",
                   "-p", "MemorySwapMax=0",
                   "-p", "CPUQuota=400%", "-p", "CPUWeight=10",
                   "-p", "TasksMax=256", "-p", "IOWeight=10",
                   "nice", "-n", "15", "ionice", "-c", "3", "choom", "-n", "1000", "--",
                   "bash", "-c", 'source "$1"; shift; exec "$@"', "candidate-build",
                   str(REPO / "scripts/candidate_env.sh"), *args]
        proc = subprocess.Popen(command, cwd=REPO, env=env)
        stopped = False

        def stop_build(_signum=None, _frame=None):
            nonlocal stopped
            stopped = True
            subprocess.run(["systemctl", "--user", "kill", "--kill-whom=all",
                            "--signal=SIGKILL", unit], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False)

        signal.signal(signal.SIGTERM, stop_build)
        signal.signal(signal.SIGINT, stop_build)
        try:
            while proc.poll() is None:
                available = available_memory()
                if available < 8 * GIB:
                    print(f"Stopping candidate build only: MemAvailable={available/GIB:.1f} GiB < 8 GiB.",
                          file=sys.stderr, flush=True)
                    stop_build()
                if stopped:
                    # A signal may arrive before systemd has registered the scope.
                    stop_build()
                time.sleep(0.5)
            result = 137 if stopped else proc.returncode
            if result:
                stop_build()
            return result
        finally:
            if proc.poll() is None:
                stop_build()
                proc.wait()


if __name__ == "__main__":
    sys.exit(main())
