#!/usr/bin/env python3
"""Regress the admission fix, then validate the plain candidate independently."""
import json
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
STATE = REPO.parent / 'state/acceptance'


def main():
    progress = {}
    def save(phase, **extra):
        progress.update(phase=phase, epoch=time.time(), **extra)
        (STATE/'postfix-progress.json').write_text(json.dumps(progress, indent=2))
        print(phase, flush=True)
    def client(phase, script, output, *args):
        save(phase)
        command = ['systemd-run','--user','--scope','--quiet','--collect',
                   '--unit=kandidat-postfix-'+phase,'--slice=kandidat-test.slice',
                   '-p','MemoryMax=3G','-p','MemorySwapMax=0','-p','CPUQuota=200%',
                   '-p','TasksMax=128','env','CUDA_VISIBLE_DEVICES=',
                   'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','python3','-I',
                   str(REPO/'scripts'/script),str(output),*args]
        with (STATE/('postfix-'+phase+'.log')).open('w') as log:
            subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    save('waiting_for_fixed_trace_startup')
    deadline = time.monotonic()+900
    while 'READY candidate' not in (STATE/'resume-admission-launch.txt').read_text():
        if time.monotonic()>deadline: raise TimeoutError('fixed trace startup')
        time.sleep(1)
    if (STATE/'active-unit').read_text().strip() != 'kandidat-acceptance-resume-admission-1.service':
        raise RuntimeError('Unexpected service before regression')
    client('regression','acceptance_streams.py',STATE/'resume-admission-1/workload',
           '--cases','pure8','--prompt-tokens','49147','--tokens','2048')
    with (STATE/'postfix-trace-summary.log').open('w') as log:
        subprocess.run(['python3','-I','scripts/summarize_scheduler_trace.py',
                        str(STATE/'resume-admission-1/scheduler.jsonl'),'--output',
                        str(STATE/'resume-admission-1/trace-summary.json')],cwd=REPO,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    summary=json.loads((STATE/'resume-admission-1/trace-summary.json').read_text())
    total=sum(summary['preemptions'].values())
    if not 1 <= total <= 8: raise RuntimeError(f'Unexpected preemptions: {total}')
    save('plain_startup',regression_preemptions=total)
    with (STATE/'postfix-plain-startup.log').open('w') as log:
        subprocess.run(['python3','-I','scripts/acceptance_service.py','candidate',
                        'resume-admission-plain'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    root=STATE/'resume-admission-plain'
    client('quality','acceptance_fixtures.py',root/'quality','baza','clef')
    client('semantics','acceptance_semantics.py',root/'semantics')
    client('state','acceptance_state.py',root/'state')
    client('recall','acceptance_recall_distinct.py',root/'recall-distinct')
    for phase in ('warmup','measure'):
        client(phase,'acceptance_streams.py',root/phase,'--cases','pure1,pure8,mixed34k','--tokens','512')
    client('queue','acceptance_streams.py',root/'queue','--cases','pure40,mixedqueue34k','--tokens','1024')
    save('complete')


if __name__=='__main__':
    main()
