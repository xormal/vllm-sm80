#!/usr/bin/env python3
"""Wait for host reserve, original-service stop and GPU0/1 driver readiness."""
from pathlib import Path
import subprocess
import time

STATE=Path(__file__).resolve().parents[2]/'state/acceptance'
UNIT='kandidat-acceptance-live-20261029.service'


def main():
    STATE.mkdir(parents=True,exist_ok=True)
    (STATE/'active-unit').write_text(UNIT+'\n')
    while True:
        available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()
                       if x.startswith('MemAvailable:'))
        original=subprocess.check_output(['systemctl','--user','show','sm80-vllm.service',
            '-p','ActiveState','--value'],text=True).strip()
        try:
            gpu=subprocess.run(['nvidia-smi','-i','0,1','--query-gpu=index','--format=csv,noheader'],
                capture_output=True,timeout=5).returncode==0
        except subprocess.TimeoutExpired:gpu=False
        if available>=16*1024**3 and original in ('inactive','failed') and gpu:return
        print(f'Waiting: MemAvailable={available/1024**3:.1f}GiB (need16), original={original}, GPU0/1_ready={gpu}',flush=True)
        time.sleep(10)


if __name__=='__main__':main()
