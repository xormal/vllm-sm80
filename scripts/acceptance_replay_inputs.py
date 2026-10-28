#!/usr/bin/env python3
"""Replay saved exact token inputs sequentially, with a fresh prefix-cache salt."""
import argparse
import json
from pathlib import Path
import re
import time
import urllib.request


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('inputs',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--indices',default='0,3')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for i in map(int,args.indices.split(',')):
        body=json.loads((args.inputs/f'input-{i}.json').read_text())
        prior=json.loads((args.inputs/f'response-{i}.json').read_text())
        body['cache_salt']=f'replay-{i}-{time.time_ns()}'
        started=time.monotonic()
        request=urllib.request.Request('http://127.0.0.1:8086/v1/completions',
                                      json.dumps(body).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=600) as response: result=json.loads(response.read())
        text=result['choices'][0]['text']
        first=re.search(r'КОД:\s*\**\s*(\d{6})',text,re.I)
        last=re.findall(r'ИТОГОВЫЙ\s+КОД:\s*\**\s*(\d{6})',text,re.I)
        item={'index':i,'expected':prior['expected'],'seconds':time.monotonic()-started,
              'first_code':first.group(1) if first else None,'last_code':last[-1] if last else None,
              'same_text_as_concurrent':text==prior['response']['choices'][0]['text'],
              'response':result}
        item['ok']=item['first_code']==item['last_code']==item['expected']
        (args.output/f'response-{i}.json').write_text(json.dumps(item,ensure_ascii=False,indent=2))
        results.append({k:v for k,v in item.items() if k!='response'})
        print(json.dumps(results[-1],ensure_ascii=False),flush=True)
    (args.output/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
