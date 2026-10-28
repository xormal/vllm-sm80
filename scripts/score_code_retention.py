#!/usr/bin/env python3
"""Strictly rescore saved first/last code lines without rewriting raw evidence."""
import argparse
import json
from pathlib import Path
import re


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('roots',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    scored=[]
    for root in args.roots:
        rows=[]
        paths=sorted(root.glob('request-*.json'))+sorted(root.glob('response-*.json'))
        for path in paths:
            record=json.loads(path.read_text())
            if 'response' not in record: continue
            choice=record['response']['choices'][0]
            text=(choice.get('message') or {}).get('content') or choice.get('text') or ''
            first=re.match(r'\A\s*\**КОД:\s*\**(\d{6})(?!\d)\**[^\S\n]*(?:\n|\Z)',text,re.I)
            last=re.search(r'(?:^|\n)\s*\**ИТОГОВЫЙ\s+КОД:\s*\**(\d{6})(?!\d)\**\s*\Z',text,re.I)
            a,b=first.group(1) if first else None,last.group(1) if last else None
            rows.append({'file':path.name,'expected':record['expected'],
                         'first_line':text.strip().splitlines()[0],
                         'last_line':text.strip().splitlines()[-1],
                         'first_code':a,'last_code':b,'correct_first':a==record['expected'],
                         'correct_last':b==record['expected'],'correct_both':a==b==record['expected']})
        scored.append({'root':str(root),'requests':len(paths),'scored':len(rows),
                       'correct_first':sum(x['correct_first'] for x in rows),
                       'correct_last':sum(x['correct_last'] for x in rows),
                       'correct_both':sum(x['correct_both'] for x in rows),'rows':rows})
    args.output.write_text(json.dumps(scored,ensure_ascii=False,indent=2))
    print(json.dumps([{k:v for k,v in x.items() if k!='rows'} for x in scored],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
