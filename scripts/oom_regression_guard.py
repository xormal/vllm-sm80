"""Guards for manually scheduled OOM tests. Importing this file does no work."""
from pathlib import Path
import socket
import subprocess

REPO = Path(__file__).resolve().parents[1]
STATE = REPO.parent / "state/acceptance"
LIVE = "kandidat-acceptance-live-20261029.service"


def unit_state(unit):
    text = subprocess.check_output(
        ["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "MainPID",
         "-p", "InvocationID", "-p", "NRestarts", "-p", "Restart", "-p", "ControlGroup",
         "-p", "StandardOutput"],
        text=True, timeout=5)
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def maintenance_guard(confirmed):
    if not confirmed:
        raise RuntimeError("Not scheduled: require --maintenance-window after owner authorizes tests")
    for unit in (LIVE, "sm80-vllm.service"):
        allowed = ("inactive",) if unit == LIVE else ("inactive", "failed")
        if unit_state(unit).get("ActiveState") not in allowed:
            raise RuntimeError(f"Refusing tests while {unit} is running")
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(("127.0.0.1", 8085)) == 0:
            raise RuntimeError("Live port 8085 is occupied; no tests allowed")
    # Both microtests and clients must use the existing bounded build runner.
    rel = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
               if line.startswith("0::"))
    group = Path("/sys/fs/cgroup") / rel.lstrip("/")
    memory = (group / "memory.max").read_text().strip()
    quota, period = (group / "cpu.max").read_text().split()
    if memory == "max" or int(memory) > 6 * 1024**3:
        raise RuntimeError("Run through scripts/candidate_build.py: RAM cap must be <=6 GiB")
    if quota == "max" or int(quota) / int(period) > 4:
        raise RuntimeError("CPU quota must be <=400%")


def unused_gpus_guard():
    # Never query, allocate on, or enumerate processes on GPU2–3.
    for index in (0, 1):
        pids = subprocess.check_output(
            ["nvidia-smi", "-i", str(index), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True, timeout=5).strip()
        if pids:
            raise RuntimeError(f"GPU{index} is occupied or cannot be confirmed empty: {pids}")


def isolated_service_guard(unit):
    if (not unit.startswith("kandidat-acceptance-") or not unit.endswith(".service")
            or unit == LIVE or (STATE / "active-unit").read_text().strip() != unit):
        raise RuntimeError("Require the explicitly named isolated acceptance unit")
    info = unit_state(unit)
    if info.get("ActiveState") != "active" or info.get("Restart") != "no":
        raise RuntimeError("Test service must be active with Restart=no; crashes must not be hidden")
    proc = Path("/proc") / info["MainPID"]
    env = dict(x.split(b"=", 1) for x in (proc / "environ").read_bytes().split(b"\0") if b"=" in x)
    cmd = (proc / "cmdline").read_bytes().split(b"\0")
    if env.get(b"CUDA_VISIBLE_DEVICES") != b"0,1":
        raise RuntimeError("Test server must be restricted to physical GPU0–1")
    for flag, value in ((b"--port", b"8086"), (b"--host", b"127.0.0.1")):
        if flag not in cmd or cmd[cmd.index(flag) + 1] != value:
            raise RuntimeError("Only isolated 127.0.0.1:8086 is allowed")
    return info
