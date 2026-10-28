#!/usr/bin/env python3
"""Eight exact-length chat-template prompts crossing a KV page boundary.

The model must compute separate codes, retain them during long decode, and
repeat them after a story. This measures functional recovery, not bitwise
batch invariance or general model quality.
"""
import argparse
import concurrent.futures as cf
import importlib.util
import json
from pathlib import Path
import re
import threading
import time
import urllib.request

URL='http://127.0.0.1:8086'
MODEL='Qwen3.8-27B-ABLITERATED-candidate'
REPO=Path(__file__).resolve().parents[1]


def post(path, body):
    request=urllib.request.Request(URL+path,json.dumps(body).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=600) as response:
        return json.loads(response.read())


def metrics():
    with urllib.request.urlopen(URL+'/metrics',timeout=5) as response:
        return response.read().decode()


def preemptions(text):
    return sum(float(x) for x in re.findall(r'^vllm:num_preemptions_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)',text,re.M))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('fixture',REPO/'sm70/integrations/vllm_sm70/clef/proverka_stenda.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    docs=fixture.текст_доков(2000000)
    topics=['реке','маяке','деревне','поезде','обсерватории','острове','мосте','караване']
    bodies=[]
    for i in range(8):
        offset=1009+i*731
        prompt=(f'Независимый документ {i}.\n'+docs[i*107:i*107+180000]+
                f'\nЗадание: вычисли 123456 + {offset}. Результат — твой шестизначный код. '
                'Запиши его первой строкой «КОД: XXXXXX». Затем напиши художественный рассказ '
                f'не меньше 900 слов о {topics[i]}. В конце повтори код отдельной строкой '
                '«ИТОГОВЫЙ КОД: XXXXXX». В середине рассказа код не повторяй.')
        ids=post('/tokenize',{'model':MODEL,'messages':[{'role':'user','content':prompt}],
                             'chat_template_kwargs':{'enable_thinking':False}})['tokens']
        excess=len(ids)-49147
        assert excess>0
        # Remove only middle document tokens, preserving the actual template,
        # unique document header and complete task at the end.
        ids=ids[:512]+ids[512+excess:]
        assert len(ids)==49147
        body={'model':MODEL,'prompt':ids,'temperature':0,'seed':17,
              'max_tokens':4096,'logprobs':1,'cache_salt':f'preempt-quality-{i}'}
        (args.output/f'input-{i}.json').write_text(json.dumps(body))
        bodies.append(body)
    before=metrics();(args.output/'metrics-before.txt').write_text(before)
    barrier=threading.Barrier(8)
    def ask(i):
        item={'index':i,'expected':str(123456+1009+i*731),'prompt_tokens':49147}
        barrier.wait(timeout=30);started=time.monotonic()
        try:
            response=post('/v1/completions',bodies[i]);item['response']=response
            text=response['choices'][0]['text']
            first=re.match(r'\A\s*\**КОД:\s*\**(\d{6})(?!\d)\**[^\S\n]*(?:\n|\Z)',text,re.I)
            last=re.findall(r'(?:^|\n)\s*\**ИТОГОВЫЙ\s+КОД:\s*\**(\d{6})(?!\d)\**\s*\Z',text,re.I)
            words=text.split();tail=words[-400:]
            grams=[tuple(tail[k:k+12]) for k in range(max(0,len(tail)-11))]
            item.update(first_code=first.group(1) if first else None,
                        last_code=last[-1] if last else None,words=len(words),
                        repeated_tail_12grams=len(grams)-len(set(grams)),
                        output_tokens=response['usage']['completion_tokens'],
                        finish_reason=response['choices'][0]['finish_reason'])
            item['ok']=item['first_code']==item['last_code']==item['expected']
        except Exception as exc:
            item.update(error=repr(exc),ok=False)
        item['seconds']=time.monotonic()-started
        (args.output/f'response-{i}.json').write_text(json.dumps(item,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in item.items() if k!='response'},ensure_ascii=False),flush=True)
        return item
    started=time.monotonic()
    with cf.ThreadPoolExecutor(8) as pool: results=list(pool.map(ask,range(8)))
    after=metrics();(args.output/'metrics-after.txt').write_text(after)
    summary={'seconds':time.monotonic()-started,'requests':8,'correct':sum(x['ok'] for x in results),
             'preemptions':preemptions(after)-preemptions(before),
             'results':[{k:v for k,v in x.items() if k!='response'} for x in results],
             'note':'Functional code retention after KV saturation; no bitwise or broad quality claim.'}
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    assert summary['correct']==8, 'Code retention failed'
    assert summary['preemptions']>=1, 'Workload did not exercise preemption'


if __name__=='__main__':
    main()
