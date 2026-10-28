#!/usr/bin/env python3
"""Compare matching prompts, never pool unmatched repeat indices."""
import json
import math
from pathlib import Path
import re
import statistics

STATE=Path(__file__).resolve().parents[2]/'state/acceptance'


def read(path):
    return json.loads(path.read_text())


def draft_delta(path):
    samples=[]
    keys=('spec_decode_num_drafts_total','spec_decode_num_accepted_tokens_total')
    for suffix in ('before','after'):
        data=(path/f'metrics-{suffix}.txt').read_text()
        samples.append({k:sum(float(x) for x in re.findall(
            r'^vllm:'+k+r'\{[^}]*\}\s+([0-9.eE+-]+)',data,re.M)) for k in keys})
    return {k:samples[1][k]-samples[0][k] for k in keys}


def main():
    roots={name:STATE/label for name,label in (
        ('torch','conv-meta-control-1'),('fused','conv-meta-fast-1'))}
    runs={name:{(r['case'],r['repeat']):r for r in read(root/'measure/results.json')}
          for name,root in roots.items()}
    expected={(case,repeat) for case in ('pure1','pure8','mixed34k') for repeat in range(3)}
    assert all(set(x)==expected for x in runs.values()),'Incomplete benchmark'
    report={}
    for case in ('pure1','pure8','mixed34k'):
        pairs=[]
        for repeat in range(3):
            a,b=(runs[name][case,repeat] for name in ('torch','fused'))
            assert a['ok'] and b['ok']
            assert (a['prompt_tokens_each'],a['requested_output_tokens'])==(b['prompt_tokens_each'],b['requested_output_tokens'])
            paths={name:root/'measure'/f'{case}-{repeat}' for name,root in roots.items()}
            same=[read(paths['torch']/f'chat-{i}.json')['tokens']==
                  read(paths['fused']/f'chat-{i}.json')['tokens'] for i in range(a['chat_requests'])]
            pairs.append({'repeat':repeat,'tokens_equal_requests':sum(same),'requests':len(same),
                'torch_tps':a['aggregate_tokens_per_s_including_prefill'],
                'fused_tps':b['aggregate_tokens_per_s_including_prefill'],
                'fused_over_torch':b['aggregate_tokens_per_s_including_prefill']/a['aggregate_tokens_per_s_including_prefill'],
                'torch_gap_p99_s':a['chunk_gap_p99_s'],'fused_gap_p99_s':b['chunk_gap_p99_s'],
                'draft_counter_deltas':{name:draft_delta(path) for name,path in paths.items()}})
        report[case]={'pairs':pairs,'geomean_fused_over_torch':math.exp(statistics.mean(
            math.log(x['fused_over_torch']) for x in pairs))}
    result={'cases':report,'notes':[
        'Each repeat index changes the prompt header; only matching indices are compared.',
        'Three matched inputs, one process per variant, fused then Torch; not independent process replications.',
        'Concurrent HTTP arrival and batch placement are not replayed deterministically.',
        'Different generated tokens or draft acceptance can affect throughput. No global zero-regression claim.']}
    (STATE/'conv-metadata-comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
