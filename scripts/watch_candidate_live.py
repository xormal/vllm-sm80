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
import importlib.util

spec=importlib.util.spec_from_file_location('candidate_recovery_policy',Path(__file__).with_name('candidate_recovery_policy.py'))
import sys
policy_module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=policy_module
spec.loader.exec_module(policy_module)

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
    policy=policy_module.RecoveryPolicy()
    while True:
        fields=subprocess.check_output(['systemctl','--user','show',UNIT,
            '-p','ActiveState','-p','ControlGroup','-p','Result',
            '-p','InvocationID','-p','ActiveEnterTimestampMonotonic'],text=True)
        fields=dict(line.split('=',1) for line in fields.splitlines())
        sample={'epoch':time.time(),'unit':UNIT,**fields}
        available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()
                       if x.startswith('MemAvailable:'))
        sample['host_available_bytes']=available
        healthy=False
        try:
            with urllib.request.urlopen('http://127.0.0.1:8085/health',timeout=5) as response:
                healthy=response.status==200
        except Exception as exc:sample['health_error']=str(exc)
        sample['healthy']=healthy
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
        now=time.monotonic()
        age=now-int(fields.get('ActiveEnterTimestampMonotonic') or 0)/1e6
        action=policy.decide(now=now,invocation=fields.get('InvocationID',''),
            state=fields['ActiveState'],age=age,healthy=healthy,
            metrics=sample.get('metrics'),available=available,
            reclaimable=int(sample.get('memory.current') or 0))
        if action:
            # Avoid restarting a replacement process or a manually stopped unit.
            current=subprocess.check_output(['systemctl','--user','show',UNIT,
                '-p','InvocationID','-p','ActiveState'],text=True)
            current=dict(x.split('=',1) for x in current.splitlines())
            if all(current[k]==fields[k] for k in ('InvocationID','ActiveState')):
                sample['recovery_action'],sample['recovery_reason']=action
                if action[0]=='start':subprocess.run(['systemctl','--user','reset-failed',UNIT],check=False)
                result=subprocess.run(['systemctl','--user','--no-block',action[0],UNIT],capture_output=True,text=True)
                sample['recovery_returncode']=result.returncode
                if result.returncode:sample['recovery_error']=result.stderr
        log.info(json.dumps(sample,ensure_ascii=False))
        (OUT/'latest.json').write_text(json.dumps(sample,ensure_ascii=False,indent=2))
        time.sleep(10)


if __name__=='__main__':main()
