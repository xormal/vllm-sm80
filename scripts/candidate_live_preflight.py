#!/usr/bin/env python3
"""Wait for host reserve, original-service stop and GPU0/1 driver readiness."""
from pathlib import Path
import subprocess
import time

STATE=Path(__file__).resolve().parents[2]/'state/acceptance'
UNIT='kandidat-acceptance-live-20261029.service'


def remove_orphan_shm(min_age=600):
    """[СРЫВ 04.11] Упавший сервер оставляет в /dev/shm свои сегменты (psm_* очередей vLLM, fa2ar1h_* AR1H): на 04.11
    их скопилось 84 файла, 8.5 ГиБ видимых, 1.2 ГиБ живой памяти хоста. Удаляются только свои файлы, которые никто
    не отображает и не держит открытыми, и не моложе min_age секунд (чужой процесс мог создать и ещё не подключить)."""
    import os
    uid=os.getuid()
    used=set()
    for pid in filter(str.isdigit,os.listdir('/proc')):
        try:
            for line in open(f'/proc/{pid}/maps'):
                if '/dev/shm/' in line:used.add(line.split('/dev/shm/',1)[1].split()[0])
            for fd in os.listdir(f'/proc/{pid}/fd'):
                t=os.readlink(f'/proc/{pid}/fd/{fd}')
                if t.startswith('/dev/shm/'):used.add(t[9:].split()[0])
        except OSError:continue
    gone=freed=0
    for name in os.listdir('/dev/shm'):
        if not name.startswith(('psm_','fa2ar1h_')) or name in used:continue
        p=f'/dev/shm/{name}'
        try:
            st=os.lstat(p)
            if st.st_uid!=uid or not (st.st_mode&0o170000)==0o100000 or time.time()-st.st_mtime<min_age:continue
            os.unlink(p);gone+=1;freed+=st.st_blocks*512
        except OSError:continue
    if gone:print(f'Removed {gone} orphan /dev/shm segments ({freed/2**20:.0f} MiB of host RAM)',flush=True)


def main():
    STATE.mkdir(parents=True,exist_ok=True)
    (STATE/'active-unit').write_text(UNIT+'\n')
    remove_orphan_shm()
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
