#!/usr/bin/env python3
"""Read-only live trial telemetry; no prompts or response bodies are recorded."""
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import re
import subprocess
import time
import urllib.request

STATE=Path(__file__).resolve().parents[2]/'state/acceptance'
UNIT='kandidat-acceptance-live-20261029.service'
OUT=STATE/'live-20261029'
KEYS={'num_requests_running','num_requests_waiting','num_preemptions_total',
      'kv_cache_usage_perc','request_success_total','prompt_tokens_total',
      'generation_tokens_total','spec_decode_num_drafts_total',
      'spec_decode_num_accepted_tokens_total'}


def main():
    OUT.mkdir(exist_ok=True)
    log=logging.getLogger('live-trial')
    log.setLevel(logging.INFO)
    log.addHandler(RotatingFileHandler(OUT/'telemetry.jsonl',maxBytes=8*1024**2,backupCount=2))
    while True:
        fields=subprocess.check_output(['systemctl','--user','show',UNIT,
            '-p','ActiveState','-p','ControlGroup','-p','Result'],text=True)
        fields=dict(line.split('=',1) for line in fields.splitlines())
        sample={'epoch':time.time(),'unit':UNIT,**fields}
        try:
            with urllib.request.urlopen('http://127.0.0.1:8085/metrics',timeout=2) as response:
                metrics=response.read().decode()
            values={}
            for name,value in re.findall(r'^vllm:(\w+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$',metrics,re.M):
                if name in KEYS:values[name]=values.get(name,0)+float(value)
            sample['metrics']=values
        except Exception as exc:
            sample['metrics_error']=str(exc)
        try:
            sample['gpu_csv']=subprocess.check_output(['nvidia-smi','-i','0,1',
                '--query-gpu=index,memory.total,memory.reserved,memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True,timeout=3).strip()
        except Exception as exc:
            sample['gpu_error']=str(exc)
        if fields.get('ControlGroup'):
            root=Path('/sys/fs/cgroup')/fields['ControlGroup'].lstrip('/')
            for name in ('memory.current','memory.peak','memory.events'):
                try:sample[name]=(root/name).read_text().strip()
                except FileNotFoundError:pass
        log.info(json.dumps(sample,ensure_ascii=False))
        (OUT/'latest.json').write_text(json.dumps(sample,ensure_ascii=False,indent=2))
        if fields['ActiveState'] not in ('active','activating'):break
        time.sleep(10)


if __name__=='__main__':main()
