#!/usr/bin/env python3
"""Run extra large-wire chat and scheduler diagnosis only after all A/B pairs."""
import json
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
STATE = REPO.parent / "state/acceptance"


def main():
    progress = {"phase":"waiting_for_ab"}
    def save(phase, **extra):
        progress.update(phase=phase,epoch=time.time(),**extra)
        (STATE / "followup-progress.json").write_text(json.dumps(progress,indent=2))
        print(phase,flush=True)
    save("waiting_for_ab")
    deadline = time.monotonic() + 1800
    while True:
        try:
            entries = json.loads((STATE / "ab-progress.json").read_text())
        except json.JSONDecodeError:
            time.sleep(1)
            continue
        if len(entries) == 10 and entries[-1]["phase"] == "done":
            if any(e["phase"] not in ("done","done_with_failures") for e in entries):
                raise RuntimeError("A/B did not complete")
            break
        active = subprocess.run(["systemctl","--user","show","kandidat-ab-controller-v3.service",
                                 "-p","ActiveState","--value"],capture_output=True,text=True,check=True).stdout.strip()
        if active not in ("active","activating") or time.monotonic() > deadline:
            raise RuntimeError("A/B stopped before completing all pairs")
        time.sleep(2)
    if (STATE / "active-unit").read_text().strip() != "kandidat-acceptance-ab-candidate-5.service":
        raise RuntimeError("Unexpected active service after A/B")
    def client(phase, script, output, *args, allow_failure=False):
        save(phase)
        cmd = ["systemd-run","--user","--scope","--quiet","--collect",
               "--unit=kandidat-followup-"+phase,"--slice=kandidat-test.slice",
               "-p","MemoryMax=3G","-p","MemorySwapMax=0","-p","CPUQuota=200%","-p","TasksMax=128",
               "env","CUDA_VISIBLE_DEVICES=","OMP_NUM_THREADS=1","OPENBLAS_NUM_THREADS=1",
               "python3","-I",str(REPO / "scripts" / script),str(output),*args]
        with (STATE / ("followup-"+phase+".log")).open("w") as log:
            result = subprocess.run(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode and not allow_failure:
            raise subprocess.CalledProcessError(result.returncode,cmd)
        return result.returncode
    for count in (1,40):
        client(f"large-chat-{count}","acceptance_large_http.py",STATE/f"post-ab-large-chat-{count}",
               "--route","chat-padded","--concurrency",str(count))
    save("trace_startup")
    with (STATE / "followup-trace-startup.log").open("w") as log:
        subprocess.run(["python3","-I","scripts/acceptance_service.py","candidate","preemption-trace",
                        "--scheduler-trace"],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    code = client("trace-load","acceptance_streams.py",STATE/"preemption-trace/workload",
                  "--cases","pure8","--prompt-tokens","49147","--tokens","2048",
                  "--stream-read-timeout","120",allow_failure=True)
    save("trace_complete",trace_client_returncode=code)


if __name__ == "__main__":
    main()
