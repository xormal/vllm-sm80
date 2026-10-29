#!/usr/bin/env python3
"""[ПЛАВНОСТЬ 29.10] Сверка ответов Clef: графобезопасная поправка (кусочный граф has_clef) против прежнего пути.

Короткие записи (шаг <= FA2SM70_CLEF_GRAPH_MAX) по одной, без параллельной нагрузки; режим переключается файлом
FA2SM70_CLEF_GRAPH_FILE ("1" -- граф, "0" -- прежний force_eager). Режимы чередуются 1,0,1,0: сравниваются
повторяемость внутри режима и расхождение между режимами (все числа ответа /v1/systemone).
"""
import argparse
import importlib.util
import json
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def числа(v, путь=""):
    if isinstance(v, bool) or v is None:
        return {}
    if isinstance(v, (int, float)):
        return {путь: float(v)}
    if isinstance(v, dict):
        out = {}
        for k in sorted(v):
            out.update(числа(v[k], f"{путь}/{k}"))
        return out
    if isinstance(v, list):
        out = {}
        for i, x in enumerate(v):
            out.update(числа(x, f"{путь}[{i}]"))
        return out
    return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--url", default="http://127.0.0.1:8086")
    p.add_argument("--toggle-file", type=Path, required=True)
    p.add_argument("--modes", default="1,0,1,0")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    def post(path, body):
        req = urllib.request.Request(a.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())

    spec = importlib.util.spec_from_file_location("fixtures", REPO / "sm70/integrations/vllm_sm70/clef/proverka_stenda.py")
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    docs = fx.текст_доков(200000)
    записи = []
    for i, (n, позиция) in enumerate([(300, "ON"), (340, "OFF"), (380, "ON"), (420, "OFF"), (460, "ON"), (500, "OFF")]):
        записи.append({"model": "clef", "max_length": n,
                       "state": f"Record parity-{i}. The switch is {позиция}.\n" + docs[i * 3001:][:n * 3],
                       "questions": {"switch": {"type": "choice", "instructions": "What is the switch position?",
                                                "criteria": {"on": "The switch is ON.", "off": "The switch is OFF."}}}})
    прогоны = []
    for m in a.modes.split(","):
        a.toggle_file.write_text(m)
        time.sleep(2.5)                                   # переключатель перечитывается раз в секунду
        ответы = [post("/v1/systemone", r) for r in записи]
        прогоны.append({"mode": m, "answers": ответы})
        print(f"режим {m}: {[json.dumps(x.get('answers'), ensure_ascii=False)[:80] for x in ответы[:2]]}", flush=True)
    (a.output / "parity.json").write_text(json.dumps(прогоны, ensure_ascii=False, indent=1))

    def разн(x, y):
        dx, dy = числа(x), числа(y)
        ключи = set(dx) | set(dy)
        return max((abs(dx.get(k, float("nan")) - dy.get(k, float("nan"))) for k in ключи), default=0.0)

    for i in range(len(прогоны)):
        for j in range(i + 1, len(прогоны)):
            d = max(разн(x, y) for x, y in zip(прогоны[i]["answers"], прогоны[j]["answers"]))
            same_choice = all(json.dumps(числа(x)) is not None and
                              (x.get("answers", {}).get("switch", {}).get("choice") ==
                               y.get("answers", {}).get("switch", {}).get("choice"))
                              for x, y in zip(прогоны[i]["answers"], прогоны[j]["answers"]))
            print(f"прогон {i} (режим {прогоны[i]['mode']}) против {j} (режим {прогоны[j]['mode']}): "
                  f"max |разница чисел| = {d:.3e}, выбор совпал: {same_choice}", flush=True)


if __name__ == "__main__":
    main()
