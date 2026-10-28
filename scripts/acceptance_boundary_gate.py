#!/usr/bin/env python3
"""Validate the isolated convolution read/write metadata experiment."""
import json
from pathlib import Path
import subprocess
import time

REPO=Path(__file__).resolve().parents[1]
STATE=REPO.parent/'state/acceptance'
ROOT=STATE/'conv-boundary-1'


def main():
    progress={}
    def save(phase):
        progress.update(phase=phase,epoch=time.time())
        (STATE/'boundary-progress.json').write_text(json.dumps(progress,indent=2))
        print(phase,flush=True)
    save('waiting_for_short_control')
    deadline=time.monotonic()+900
    while not (ROOT/'replay-32/summary.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('short control')
        time.sleep(1)
    short=json.loads((ROOT/'replay-32/summary.json').read_text())
    assert len(short)==3 and all(x['first_code']==x['expected'] for x in short)
    if (STATE/'active-unit').read_text().strip()!='kandidat-acceptance-conv-boundary-1.service':
        raise RuntimeError('Unexpected active service')
    def client(phase,script,*args):
        save(phase)
        command=['systemd-run','--user','--scope','--quiet','--collect',
                 '--unit=kandidat-boundary-'+phase,'--slice=kandidat-test.slice',
                 '-p','MemoryMax=3G','-p','MemorySwapMax=0','-p','CPUQuota=200%',
                 '-p','TasksMax=128','env','CUDA_VISIBLE_DEVICES=',
                 'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','python3','-I',
                 str(REPO/'scripts'/script),str(ROOT/phase),*args]
        with (ROOT/(phase+'.log')).open('w') as log:
            subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    client('preemption-quality','acceptance_preemption_quality.py')
    client('regression','acceptance_streams.py','--cases','pure8','--prompt-tokens','49147','--tokens','2048')
    client('quality','acceptance_fixtures.py','baza','clef')
    client('semantics','acceptance_semantics.py')
    client('state','acceptance_state.py')
    client('recall-distinct','acceptance_recall_distinct.py')
    for phase in ('warmup','measure'):
        client(phase,'acceptance_streams.py','--cases','pure1,pure8,mixed34k','--tokens','512')
    client('queue','acceptance_streams.py','--cases','pure40,mixedqueue34k','--tokens','1024')
    client('long-context','acceptance_fixtures.py','igla')
    save('complete')


if __name__=='__main__':main()
