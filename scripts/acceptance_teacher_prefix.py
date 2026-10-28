#!/usr/bin/env python3
"""Compare a long completion with fresh-prefill one-token continuation.

This changes execution path (prefill versus recurrent/MTP decode), so a
mismatch diagnoses path dependence, not by itself a specific faulty kernel.
"""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def post(path,body):
    request=urllib.request.Request('http://127.0.0.1:8086'+path,json.dumps(body).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=600) as response:return json.loads(response.read())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--prefix',default='КОД: 12')
    parser.add_argument('--steps',type=int,default=6)
    args=parser.parse_args()
    assert 1<=args.steps<=16
    args.output.mkdir(parents=True,exist_ok=False)
    body=json.loads(args.input.read_text())
    initial=post('/tokenize',{'model':body['model'],'prompt':args.prefix,'add_special_tokens':False})['tokens']
    prompt=body['prompt']+initial
    text=args.prefix;rows=[]
    for i in range(args.steps):
        request={**body,'prompt':prompt,'max_tokens':1,'ignore_eos':True,'logprobs':20,
                 'return_tokens_as_token_ids':True,'cache_salt':f'teacher-{i}-{time.time_ns()}'}
        started=time.monotonic();result=post('/v1/completions',request)
        choice=result['choices'][0];token=choice['logprobs']['tokens'][0]
        assert token.startswith('token_id:'),token
        row={'step':i,'prompt_tokens':len(prompt),'prefix_text':text,'next_text':choice['text'],
             'seconds':time.monotonic()-started,'response':result}
        rows.append(row)
        (args.output/f'step-{i}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2))
        prompt.append(int(token.split(':')[1]));text+=choice['text']
        print(json.dumps({k:v for k,v in row.items() if k!='response'},ensure_ascii=False),flush=True)
    (args.output/'summary.json').write_text(json.dumps({'text':text,'initial_prefix_tokens':initial,
        'steps':[{k:v for k,v in x.items() if k!='response'} for x in rows]},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
