#!/usr/bin/env python3
"""Test opt-in temporary MLP chunks, always restore the default candidate."""
import importlib.util
import json
from pathlib import Path
import subprocess
import time

REPO=Path(__file__).resolve().parents[1]
STATE=REPO.parent/'state/acceptance'


def main():
    progress={}
    def save(phase,**extra):
        progress.update(phase=phase,epoch=time.time(),**extra)
        (STATE/'fallback-progress.json').write_text(json.dumps(progress,indent=2))
        print(phase,flush=True)
    save('waiting_for_preemption_quality')
    deadline=time.monotonic()+7200
    quality=STATE/'resume-admission-plain/preemption-quality/summary.json'
    while not quality.exists():
        if time.monotonic()>deadline: raise TimeoutError('postfix/preemption quality')
        state=subprocess.check_output(['systemctl','--user','show',
              'kandidat-preemption-quality-controller.service','-p','ActiveState','--value'],text=True).strip()
        if state not in ('active','activating'): raise RuntimeError('Preemption quality did not finish')
        time.sleep(2)
    checked=json.loads(quality.read_text())
    if checked['correct']!=8 or checked['preemptions']<1:
        raise RuntimeError('Inspect preemption quality before proceeding')
    output=STATE/'fallback-gate'
    output.mkdir(exist_ok=False)
    def start(label,rows):
        save(label+'_startup')
        with (output/(label+'-startup.log')).open('w') as log:
            subprocess.run(['python3','-I','scripts/acceptance_service.py','candidate',label,
                            '--fallback-rows',str(rows)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    def client(phase,script,*args):
        save(phase)
        command=['systemd-run','--user','--scope','--quiet','--collect',
                 '--unit=kandidat-fallback-'+phase,'--slice=kandidat-test.slice',
                 '-p','MemoryMax=3G','-p','MemorySwapMax=0','-p','CPUQuota=200%',
                 '-p','TasksMax=128','env','CUDA_VISIBLE_DEVICES=',
                 'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','python3','-I',
                 str(REPO/'scripts'/script),str(STATE/'fallback-1024'/phase),*args]
        with (output/(phase+'.log')).open('w') as log:
            subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        start('fallback-1024',1024)
        client('quality','acceptance_fixtures.py','baza','clef')
        client('semantics','acceptance_semantics.py')
        client('state','acceptance_state.py')
        client('long-clef','acceptance_fixtures.py','dlinnye','clefkart')
        for phase in ('warmup','measure'):
            client(phase,'acceptance_streams.py','--cases','pure1,pure8,mixed34k','--tokens','512')
        spec=importlib.util.spec_from_file_location('compare',REPO/'scripts/acceptance_compare.py')
        compare=importlib.util.module_from_spec(spec);spec.loader.exec_module(compare)
        result={'quality':compare.compare_quality(STATE/'resume-admission-plain',STATE/'fallback-1024'),
                'extra':compare.compare_extra(STATE/'resume-admission-plain',STATE/'fallback-1024')}
        (output/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        log=(STATE/'fallback-1024/server.log').read_text(errors='replace')
        if 'bounded fallback:' not in log or 'Clef=True' not in log:
            raise RuntimeError('Missing evidence that both fallback and Clef paths ran')
        save('tests_complete')
    finally:
        start('validated-candidate',0)
    save('complete',default_restored=True)


if __name__=='__main__':
    main()
