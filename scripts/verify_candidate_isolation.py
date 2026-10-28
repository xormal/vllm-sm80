#!/usr/bin/env python3
"""Read-only audit of preserved production files and the active local candidate."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

REPO = Path(__file__).resolve().parents[1]
STATE = REPO.parent / "state/acceptance"


def output(*args):
    return subprocess.run(args, text=True, capture_output=True, check=True).stdout.strip()


def main():
    baseline = json.loads((REPO / "docs/research/2026-10-28-qwen38/candidate-setup/BASELINE.json").read_text())
    checks = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
              for p,digest in baseline["production_files"].items()}
    original = Path(baseline["source_root"]) / "vllm-sm70"
    status = output("git", "-C", str(original), "status", "--porcelain")
    unit = (STATE / "active-unit").read_text().strip()
    if not re.fullmatch(r"kandidat-(?:validation|acceptance-[a-z0-9-]+)\.service", unit):
        raise ValueError("Unexpected candidate service name")
    properties = output("systemctl", "--user", "show", unit, "-p", "MainPID", "-p", "ActiveState",
                        "-p", "MemoryMax", "-p", "MemorySwapMax", "-p", "CPUQuotaPerSecUSec")
    fields = dict(row.split("=",1) for row in properties.splitlines())
    selected = {}
    if int(fields["MainPID"]):
        environment = Path(f"/proc/{fields['MainPID']}/environ").read_bytes().split(b"\0")
        for item in environment:
            key, _, value = item.partition(b"=")
            if key in (b"CUDA_VISIBLE_DEVICES",b"CANDIDATE_VARIANT",b"FA2SM80_BLOCKPIPE_FALLBACK_ROWS",b"CANDIDATE_SCHEDULER_TRACE"):
                selected[key.decode()] = value.decode()
    ports = output("ss", "-ltnH", "( sport = :8085 or sport = :8086 )")
    result = {"recorded_utc":datetime.now(timezone.utc).isoformat(),
              "production_files_unchanged":checks,"production_git_status":status,
              "production_head":output("git","-C",str(original),"rev-parse","HEAD"),
              "production_service":output("systemctl","--user","show","sm80-vllm.service",
                                          "-p","ActiveState","-p","UnitFileState","-p","ExecStart"),
              "candidate_unit":unit,"candidate_service":fields,"candidate_selected_environment":selected,
              "listeners":ports}
    (STATE / "isolation-current.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    assert all(checks.values()) and not status
    assert result["production_head"] == baseline["baseline_commit"]
    assert not any(row.split()[3].endswith(":8085") for row in ports.splitlines())
    assert all(row.split()[3] == "127.0.0.1:8086" for row in ports.splitlines())
    if selected:
        assert selected.get("CUDA_VISIBLE_DEVICES") == "0,1"


if __name__ == "__main__":
    main()
