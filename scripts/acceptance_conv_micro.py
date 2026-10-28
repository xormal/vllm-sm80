#!/usr/bin/env python3
"""Wait for the fallback gate, then test metadata on freed physical GPU0."""
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import re

REPO = Path(__file__).resolve().parents[1]
STATE = REPO.parent / 'state/acceptance'


def main():
    deadline = time.monotonic() + 3600
    while True:
        progress = json.loads((STATE / 'fallback-progress.json').read_text())
        if progress['phase'] == 'complete':
            break
        status = subprocess.check_output(['systemctl', '--user', 'show',
            'kandidat-fallback-gate-v2-controller.service', '-p', 'ActiveState',
            '--value'], text=True).strip()
        if status not in ('active', 'activating') or time.monotonic() > deadline:
            raise RuntimeError('Fallback gate did not complete')
        time.sleep(2)
    unit = (STATE / 'active-unit').read_text().strip()
    if unit != 'kandidat-acceptance-validated-candidate.service':
        raise RuntimeError('Unexpected service')
    with urllib.request.urlopen('http://127.0.0.1:8086/metrics', timeout=5) as r:
        metrics = r.read().decode()
    inflight = sum(float(x) for x in re.findall(
        r'^vllm:num_requests_(?:running|waiting)(?:\{[^}]*\})?\s+([0-9.eE+-]+)',
        metrics, re.M))
    if inflight:
        raise RuntimeError(f'Refusing to interrupt {inflight} requests')
    subprocess.run(['systemctl', '--user', 'stop', unit], check=True)
    subprocess.run(['python3', '-I', 'scripts/candidate_build.py', '--',
        'env', 'CUDA_VISIBLE_DEVICES=0',
        'CANDIDATE_CONV_META_RESULT=' + str(STATE / 'conv-metadata-micro.json'),
        str(REPO / '.venv/bin/python'),
        'sm70/tests/test_conv_metadata_cuda.py'], cwd=REPO, check=True)


if __name__ == '__main__':
    main()
