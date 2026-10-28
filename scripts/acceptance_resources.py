#!/usr/bin/env python3
"""Summarize sampled host and GPU0/1 resources by acceptance launch window."""
import csv
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

STATE = Path(__file__).resolve().parents[2] / "state/acceptance"


def main():
    units = {}
    for line in (STATE / "host-memory.jsonl").read_text().splitlines():
        try:
            sample = json.loads(line)
        except json.JSONDecodeError:
            continue  # Concurrent writer may not have completed its last line.
        unit = sample["unit"]
        out = units.setdefault(unit, {"samples":0,"first_epoch":sample["epoch"],
                                     "max_memory_current":0,"max_memory_peak":0,
                                     "min_system_available":sample["mem_available"],
                                     "memory_events_max":{},"watcher_actions":[],
                                     "gpu_peak_MiB":{}})
        out["samples"] += 1
        out["last_epoch"] = sample["epoch"]
        out["min_system_available"] = min(out["min_system_available"], sample["mem_available"])
        for name in ("current", "peak"):
            out["max_memory_"+name] = max(out["max_memory_"+name], int(sample.get("memory."+name, 0)))
        for row in sample.get("memory.events", "").splitlines():
            key, value = row.split()
            out["memory_events_max"][key] = max(out["memory_events_max"].get(key,0),int(value))
        if "action" in sample: out["watcher_actions"].append(sample["action"])
    with (STATE / "gpu-memory.csv").open() as source:
        reader = csv.reader(source)
        next(reader)
        for row in reader:
            if len(row) != 4: continue
            timestamp, index, memory, _ = (x.strip() for x in row)
            if index not in ("0", "1"):
                raise ValueError("Resource series contains a prohibited GPU")
            epoch = datetime.strptime(timestamp, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=ZoneInfo("Asia/Tbilisi")).timestamp()
            for unit in units.values():
                if unit["first_epoch"] <= epoch <= unit["last_epoch"]:
                    unit["gpu_peak_MiB"][index] = max(unit["gpu_peak_MiB"].get(index,0),int(memory.split()[0]))
    result = {"units":units,"notes":["Host sampled at 0.5 s; GPU at 1 s. Not allocator peaks.",
              "GPU samples are attributed by host-watch unit windows; the switch boundary can include startup/stop overlap.",
              "memory.peak includes startup and charged file cache; it is not process RSS."]}
    (STATE / "resource-summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
