#!/usr/bin/env python3
"""Long decode recall with separately computed codes in eight concurrent slots."""
import argparse
import concurrent.futures as cf
import importlib.util
import json
from pathlib import Path
import re
import time
import urllib.request

REPO = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("fixture", REPO / "sm70/integrations/vllm_sm70/clef/proverka_stenda.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    docs = fixture.текст_доков(10**7)
    topics = ["реке", "маяке", "деревне", "поезде", "обсерватории", "острове", "мосте", "караване"]
    def ask(i):
        offset = i * 731 + 1009
        expected = str(123456 + offset)
        off = (i * 50021) % (len(docs)-62000)
        prompt = (f"Независимый документ {i}.\n" + docs[off:off+60000] +
                  f"\n\nЗадание: вычисли 123456 + {offset}. Результат — твой шестизначный код. "
                  "Запиши его первой строкой «КОД: XXXXXX». Затем напиши подробный художественный рассказ "
                  f"не меньше 900 слов о {topics[i%8]}. В самом конце отдельной строкой повтори свой код: "
                  "«ИТОГОВЫЙ КОД: XXXXXX». В середине рассказа код не повторяй.")
        body = {"model":"Qwen3.8-27B-ABLITERATED-candidate", "messages":[{"role":"user","content":prompt}],
                "temperature":0, "seed":17, "max_tokens":4096, "chat_template_kwargs":{"enable_thinking":False}}
        item = {"index":i,"expected":expected,"started":time.time()}
        try:
            request = urllib.request.Request("http://127.0.0.1:8086/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"})
            with urllib.request.urlopen(request,timeout=600) as response: result=json.loads(response.read())
            item["response"]=result
            text=result["choices"][0]["message"].get("content") or ""
            first=re.search(r"КОД:\s*\**\s*(\d{6})",text,re.I)
            last=re.findall(r"ИТОГОВЫЙ\s+КОД:\s*\**\s*(\d{6})",text,re.I)
            a,b=first.group(1) if first else None,last[-1] if last else None
            words=text.split(); tail=words[-400:]
            grams=[tuple(tail[k:k+12]) for k in range(max(0,len(tail)-11))]
            item.update(first_code=a,last_code=b,first_equals_expected=a==expected,
                        recall_equal=bool(a and a==b), words=len(words),
                        repeated_tail_12grams=len(grams)-len(set(grams)),
                        stopped=result["choices"][0]["finish_reason"]=="stop")
        except Exception as exc: item["error"]=repr(exc)
        item["seconds"]=time.time()-item["started"]
        (args.output/f"request-{i}.json").write_text(json.dumps(item,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in item.items() if k!="response"},ensure_ascii=False),flush=True)
        return item
    results=[]
    for round_no in range(2):
        with cf.ThreadPoolExecutor(8) as pool: results.extend(pool.map(ask,range(round_no*8,(round_no+1)*8)))
    summary={"requests":len(results),"correct_first":sum(x.get("first_equals_expected",False) for x in results),
             "recall_equal":sum(x.get("recall_equal",False) for x in results),
             "unique_first_codes":len({x.get("first_code") for x in results if x.get("first_code")}),
             "stopped":sum(x.get("stopped",False) for x in results),
             "errors":[x for x in results if "error" in x],
             "note":"Code is computed from input arithmetic; this probes cross-request isolation but is not pure unconstrained self-generated recall."}
    (args.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
