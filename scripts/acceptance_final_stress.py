#!/usr/bin/env python3
"""Long decode stability and large HTTP bodies after optional-path validation."""
import argparse
import json
from pathlib import Path
import subprocess
import time

REPO=Path(__file__).resolve().parents[1]
STATE=REPO.parent/'state/acceptance'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fused',action='store_true',help='Require completed fused metadata gate')
    parser.add_argument('--matched-control',action='store_true',help='Wait for the matched control and final fused restart')
    parser.add_argument('--torch-final',action='store_true',help='Use the ordinary metadata path after matched comparison')
    args=parser.parse_args()
    if args.matched_control and not args.fused:raise ValueError('--matched-control requires --fused')
    if args.torch_final and args.fused:raise ValueError('Choose one final metadata path')
    gate='conv-control' if args.matched_control or args.torch_final else 'conv-fused' if args.fused else 'fallback'
    controller=('kandidat-conv-control-controller.service' if args.matched_control or args.torch_final else
                'kandidat-conv-fused-gate-controller.service' if args.fused
                else 'kandidat-fallback-gate-v2-controller.service')
    label=('conv-meta-final-torch-1' if args.torch_final else
           'conv-meta-final-1' if args.matched_control else 'conv-meta-fast-1' if args.fused else 'validated-candidate')
    progress={'label':label}
    def save(phase):
        progress.update(phase=phase,epoch=time.time())
        (STATE/'final-stress-progress.json').write_text(json.dumps(progress,indent=2))
        print(phase,flush=True)
    save('waiting_for_'+gate+'_gate')
    deadline=time.monotonic()+7200
    while True:
        p=STATE/(gate+'-progress.json')
        if p.exists() and json.loads(p.read_text()).get('phase')=='complete':break
        if time.monotonic()>deadline:raise TimeoutError(gate+' gate')
        state=subprocess.check_output(['systemctl','--user','show',
             controller,'-p','ActiveState','--value'],text=True).strip()
        if state not in ('active','activating'):raise RuntimeError(gate+' gate did not finish')
        time.sleep(2)
    if (STATE/'active-unit').read_text().strip()!='kandidat-acceptance-'+label+'.service':
        raise RuntimeError('Unexpected final service')
    root=STATE/label
    def client(phase,script,*args):
        save(phase)
        command=['systemd-run','--user','--scope','--quiet','--collect',
                 '--unit=kandidat-final-'+phase,'--slice=kandidat-test.slice',
                 '-p','MemoryMax=3G','-p','MemorySwapMax=0','-p','CPUQuota=200%',
                 '-p','TasksMax=128','env','CUDA_VISIBLE_DEVICES=',
                 'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','python3','-I',
                 str(REPO/'scripts'/script),str(root/phase),*args]
        with (STATE/('final-'+phase+'.log')).open('w') as log:
            subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    client('context-limit','acceptance_streams.py','--cases','pure1','--prompt-tokens','262140','--tokens','4')
    client('context-max1','acceptance_streams.py','--cases','pure1','--prompt-tokens','262143','--tokens','1')
    client('decode32k','acceptance_streams.py','--cases','pure1','--prompt-tokens','49147','--tokens','32768')
    client('large-valid-chat40','acceptance_large_http.py','--route','chat-padded','--concurrency','40')
    client('large-clef','acceptance_large_http.py','--route','clef','--concurrency','1')
    client('large-chat-rejected','acceptance_large_http.py','--route','chat','--concurrency','1')
    client('quality-after-stress','acceptance_fixtures.py','baza','clef')
    save('complete')


if __name__=='__main__':main()
