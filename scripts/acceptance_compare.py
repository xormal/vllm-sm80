#!/usr/bin/env python3
"""Summarize saved alternating runs; missing data never constitutes a pass."""
import json
import math
from pathlib import Path
import statistics

STATE = Path(__file__).resolve().parents[2] / "state/acceptance"


def read(path):
    return json.loads(path.read_text())


def numbers(value):
    if isinstance(value, dict):
        return {str(k) + "/" + p: n for k, v in value.items() for p, n in numbers(v).items()}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {"": value}
    return {}


def compare_quality(baseline, candidate):
    b, c = read(baseline / "quality/baza.json"), read(candidate / "quality/baza.json")
    chats = {k: {"text_equal": b[k]["text"] == c[k]["text"], "logprobs_equal": b[k]["lp"] == c[k]["lp"],
                 "max_logprob_delta_if_same_text": max((abs(x-y) for x,y in zip(b[k]["lp"], c[k]["lp"])), default=0)
                 if b[k]["text"] == c[k]["text"] and len(b[k]["lp"]) == len(c[k]["lp"]) else None} for k in b}
    b = {(x["i"], x["k"]): x["resp"]["answers"] for x in read(baseline / "quality/clef.json")}
    c = {(x["i"], x["k"]): x["resp"]["answers"] for x in read(candidate / "quality/clef.json")}
    deltas = [abs(v - numbers(c[k])[p]) for k in b for p, v in numbers(b[k]).items()]
    clef = {"answers_equal": b == c, "max_numeric_delta": max(deltas, default=0),
            "baseline_repeat_equal": all(b[i,0] == b[i,1] for i,_ in b),
            "candidate_repeat_equal": all(c[i,0] == c[i,1] for i,_ in c)}
    semantic = {}
    for name in ("arithmetic", "reasoning", "code", "russian"):
        b, c = (read(p / "semantics" / (name + ".json")) for p in (baseline, candidate))
        bc, cc = b["response"]["choices"][0], c["response"]["choices"][0]
        bl = (bc.get("logprobs") or {}).get("content") or []
        cl = (cc.get("logprobs") or {}).get("content") or []
        aligned = []
        # Include the first divergent output position: its input prefix is
        # still identical. Stop before comparing unrelated later prefixes.
        for i, (x, y) in enumerate(zip(bl, cl)):
            aligned.append({"position": i, "same_token": x["bytes"] == y["bytes"],
                            "same_top20": x.get("top_logprobs") == y.get("top_logprobs"),
                            "H_lower_delta": c["entropy"][i]["H_lower_nats"] - b["entropy"][i]["H_lower_nats"],
                            "H_upper_delta": c["entropy"][i]["H_upper_nats"] - b["entropy"][i]["H_upper_nats"]})
            if x["bytes"] != y["bytes"]: break
        semantic[name] = {"message_equal": bc["message"] == cc["message"], "logprobs_equal": bl == cl,
                          "common_input_prefix_positions": len(aligned),
                          "all_aligned_top20_equal": all(x["same_top20"] for x in aligned) if aligned else None,
                          "max_aligned_entropy_bound_delta": max((max(abs(x["H_lower_delta"]),abs(x["H_upper_delta"])) for x in aligned), default=None),
                          "baseline_expected": b["expected_present"], "candidate_expected": c["expected_present"]}
    summaries = {name: read(p / "semantics/summary.json")
                 for name, p in (("baseline", baseline), ("candidate", candidate))}
    bs, cs = (read(p / "semantics/seed.json") for p in (baseline, candidate))
    return {"chats": chats, "clef": clef, "semantics": semantic,
            "seed": {"baseline_repeat_equal": summaries["baseline"]["same_seed_messages"],
                     "candidate_repeat_equal": summaries["candidate"]["same_seed_messages"],
                     "cross_variant_messages_equal": all(x["choices"][0]["message"] == y["choices"][0]["message"]
                                                         for x,y in zip(bs,cs))},
            "tools": {name: value["tool_ok"] for name,value in summaries.items()}}


def compare_extra(baseline, candidate):
    out = {}
    if all((p / "state/results.json").exists() for p in (baseline, candidate)):
        br, cr = (read(p / "state/results.json") for p in (baseline,candidate))
        states = {}
        for key in br:
            roles = ("before","after") if key == "cancel" else ("first","again") if key == "prefix_max_tokens_1" else ("solo","mixed")
            states[key] = {}
            for role in roles:
                b,c = (r[key][role]["choices"][0]["logprobs"] for r in (br,cr))
                same = b["tokens"] == c["tokens"]
                states[key][role] = {"tokens_equal":same,"logprobs_equal":b["token_logprobs"]==c["token_logprobs"],
                                    "max_logprob_delta_if_same_tokens":max((abs(x-y) for x,y in zip(b["token_logprobs"],c["token_logprobs"]) if x is not None and y is not None),default=0) if same else None}
            for name,r in (("baseline",br),("candidate",cr)):
                a,b = (r[key][role]["choices"][0]["logprobs"] for role in roles)
                states[key][name+"_within_trial_max_logprob_delta"] = max((abs(x-y) for x,y in zip(a["token_logprobs"],b["token_logprobs"]) if x is not None and y is not None),default=0)
        out["states"] = states
    if all((p / "recall-distinct/summary.json").exists() for p in (baseline,candidate)):
        out["recall"] = {"baseline":read(baseline/"recall-distinct/summary.json"),"candidate":read(candidate/"recall-distinct/summary.json"),"requests":[]}
        for i in range(16):
            b,c = (read(p/"recall-distinct"/f"request-{i}.json") for p in (baseline,candidate))
            if "response" in b and "response" in c:
                out["recall"]["requests"].append({"index":i,"same_message":b["response"]["choices"][0]["message"]==c["response"]["choices"][0]["message"],"baseline_words":b["words"],"candidate_words":c["words"],
                    "baseline_repeated_tail_12grams":b["repeated_tail_12grams"],
                    "candidate_repeated_tail_12grams":c["repeated_tail_12grams"]})
    if all((p / "preemption/results.json").exists() for p in (baseline,candidate)):
        out["preemption"] = {"baseline":read(baseline/"preemption/results.json"),"candidate":read(candidate/"preemption/results.json"),"requests":[]}
        for i in range(8):
            b,c = (read(p/"preemption/pure8-0"/f"chat-{i}.json") for p in (baseline,candidate))
            out["preemption"]["requests"].append({"index":i,"tokens_equal":b.get("tokens")==c.get("tokens"),"baseline_ok":b["ok"],"candidate_ok":c["ok"]})
        for name,p in (("baseline",baseline),("candidate",candidate)):
            samples=read(p/"preemption/pure8-0/metrics-during.json")
            counts=[x["num_preemptions_total"] for x in samples if "num_preemptions_total" in x]
            out["preemption"][name+"_preemption_delta"] = max(counts)-min(counts) if counts else None
    return out


def main():
    pairs = []
    perf = {}
    for pair in range(1,6):
        b, c = (STATE / f"ab-{v}-{pair}" for v in ("baseline", "candidate"))
        if not all((p / "measure/results.json").exists() for p in (b,c)): continue
        quality = compare_quality(b,c)
        br, cr = ({x["case"]: x for x in read(p / "measure/results.json")} for p in (b,c))
        if set(br) != {"pure1","pure8","mixed34k"} or set(cr) != set(br): continue
        pairs.append({"pair": pair, "quality": quality, "extra":compare_extra(b,c) if pair==1 else {}})
        for case in br:
            for metric in ("aggregate_tokens_per_s_including_prefill", "chunk_gap_p99_s", "chunk_gap_max_s"):
                key = case + "/" + metric
                perf.setdefault(key, []).append({"pair": pair, "baseline":br[case][metric], "candidate":cr[case][metric],
                                                "workloads_ok":br[case]["ok"] and cr[case]["ok"]})
    performance = {}
    for key, values in perf.items():
        ratios = [math.log(x["candidate"] / x["baseline"]) for x in values]
        entry = {"runs":values,"geomean_candidate_over_baseline":math.exp(statistics.mean(ratios))}
        if len(ratios)==5:
            half = 2.776445105 * statistics.stdev(ratios) / math.sqrt(5)
            entry["paired_logratio_t95_interval"] = [math.exp(statistics.mean(ratios)+s*half) for s in (-1,1)]
        performance[key]=entry
    repeatability = {}
    for variant in ("baseline", "candidate"):
        first = STATE / f"ab-{variant}-1"
        comparisons = []
        for pair in range(2,6):
            other = STATE / f"ab-{variant}-{pair}"
            if (other / "semantics/summary.json").exists():
                comparisons.append({"reference_run":1,"other_run":pair,
                                    "quality":compare_quality(first,other)})
        repeatability[variant] = comparisons
    result = {"completed_pairs":len(pairs), "required_pairs":5, "pairs":pairs, "performance":performance,
              "within_variant_repeatability":repeatability,
              "notes":["Five paired samples; t intervals assume independent approximately normal log ratios.",
                       "Baseline-first order and shared host/training load can cause drift. No release pass inferred automatically.",
                       "Clef numeric values are rounded by the API. Top-20 entropy bounds are not full entropy."]}
    (STATE / "ab-comparison.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({"completed_pairs":len(pairs),"performance":performance},indent=2))


if __name__ == "__main__":
    main()
