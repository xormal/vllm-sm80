#!/usr/bin/env python3
"""Host-memory reserve watcher; may kill only the named candidate validation unit."""
import json
from pathlib import Path
import re
import subprocess
import time

STATE = Path(__file__).resolve().parents[2] / "state/acceptance"
while True:
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    unit = (STATE / "active-unit").read_text().strip()
    if not re.fullmatch(r"kandidat-(?:validation|acceptance-[a-z0-9-]+)\.service", unit):
        raise RuntimeError("Refusing to watch an unrelated unit")
    cgroup = subprocess.run(["systemctl", "--user", "show", unit, "-p", "ControlGroup", "--value"],
                            capture_output=True, text=True, check=False).stdout.strip()
    sample = {"epoch": time.time(), "mem_available": available, "unit": unit}
    if cgroup:
        root = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
        for name in ("memory.current", "memory.peak", "memory.events", "cpu.stat"):
            try: sample[name] = (root / name).read_text().strip()
            except FileNotFoundError: pass
    if available < 8 * 1024**3:
        sample["action"] = "KILL candidate only: host available memory <8GiB"
        subprocess.run(["systemctl", "--user", "kill", "--kill-whom=all", "--signal=SIGKILL", unit], check=False)
    with (STATE / "host-memory.jsonl").open("a") as f:
        f.write(json.dumps(sample) + "\n")
    time.sleep(.5)
