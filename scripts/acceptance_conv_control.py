#!/usr/bin/env python3
"""Match the three fused benchmark inputs with the ordinary metadata path."""
import json
from pathlib import Path
import subprocess
import time

REPO=Path(__file__).resolve().parents[1]
STATE=REPO.parent/'state/acceptance'


def main():
    def save(phase):
        (STATE/'conv-control-progress.json').write_text(json.dumps(
            {'phase':phase,'epoch':time.time()},indent=2))
        print(phase,flush=True)
    save('waiting_for_fused_gate')
    deadline=time.monotonic()+3600
    while True:
        p=STATE/'conv-fused-progress.json'
        if p.exists() and json.loads(p.read_text())['phase']=='complete':break
        status=subprocess.check_output(['systemctl','--user','show',
            'kandidat-conv-fused-gate-controller.service','-p','ActiveState','--value'],text=True).strip()
        if status not in ('active','activating') or time.monotonic()>deadline:
            raise RuntimeError('Fused gate did not complete')
        time.sleep(2)
    root=STATE/'conv-meta-control-1'
    root.mkdir(exist_ok=False)
    def start(label,fused):
        save(label+'_startup')
        with (root/(label+'-startup.log')).open('w') as log:
            subprocess.run(['python3','-I','scripts/acceptance_service.py','candidate',label,
                '--gran-conv-all','1','--conv-meta-fused',str(fused)],cwd=REPO,
                stdout=log,stderr=subprocess.STDOUT,check=True)
    def client(phase,repeat):
        save(phase)
        with (root/(phase+'.log')).open('w') as log:
            subprocess.run(['systemd-run','--user','--scope','--quiet','--collect',
                '--unit=kandidat-conv-control-'+phase,'--slice=kandidat-test.slice',
                '-p','MemoryMax=3G','-p','MemorySwapMax=0','-p','CPUQuota=200%',
                '-p','TasksMax=128','env','CUDA_VISIBLE_DEVICES=',
                'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','python3','-I',
                str(REPO/'scripts/acceptance_streams.py'),str(root/phase),
                '--cases','pure1,pure8,mixed34k','--tokens','512','--repeat',str(repeat)],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        start('conv-meta-control-1',0)
        client('warmup',1)
        client('measure',3)
    finally:
        start('conv-meta-final-1',1)
    save('complete')


if __name__=='__main__':main()
