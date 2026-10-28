#!/usr/bin/env python3
"""Read CPU-only scheduler JSONL; retain a small excerpt of repeated preemption."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace",type=Path)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    records = []
    for line in args.trace.read_text().splitlines():
        try: records.append(json.loads(line))
        except json.JSONDecodeError: pass
    preemptions = [r for r in records if r["event"] == "preempt"]
    counts = Counter(r["before"]["id"] for r in preemptions)
    result = {"records":len(records),"preemptions":dict(counts),"requests":{}}
    for rid, count in counts.items():
        first = next(r for r in preemptions if r["before"]["id"]==rid)
        schedules = []
        generated = 0
        for r in records:
            if r["t"] < first["t"]: continue
            if r["event"] == "update":
                generated += r["generated"].get(rid) or 0
            if r["event"] != "schedule": continue
            before = next((q for q in r["before"]["requests"] if q["id"]==rid),None)
            after = next((q for q in r["after"]["requests"] if q["id"]==rid),None)
            if before is None and after is None: continue
            schedules.append({"step":r["step"],"seconds_since_first_preempt":r["t"]-first["t"],
                              "before":before,"after":after,"scheduled":r["scheduled"].get(rid,0),
                              "resumed":rid in r["resumed"],"preempted":rid in r["preempted"],
                              "pools_before":r["before"]["pools"],"pools_after":r["after"]["pools"]})
        result["requests"][rid] = {"preemptions":count,"first_preempt":first,
            "last_preempt":next(r for r in reversed(preemptions) if r["before"]["id"]==rid),
            "generated_tokens_from_first_preempt":generated,
            "scheduled_tokens_from_first_preempt":sum(r["scheduled"] for r in schedules),
            "resume_events":sum(r["resumed"] for r in schedules),
            "first_schedules":schedules[:16],"last_schedules":schedules[-4:]}
    if args.output: args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({"records":len(records),"preemptions":dict(counts),
        "requests":{rid:{k:v for k,v in value.items() if k not in ("first_schedules","last_schedules")}
                    for rid,value in result["requests"].items()}},indent=2))


if __name__ == "__main__":
    main()
