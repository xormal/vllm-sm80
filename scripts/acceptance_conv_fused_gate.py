#!/usr/bin/env python3
"""Model-level acceptance after exact standalone metadata validation."""
import json
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
STATE = REPO.parent / 'state/acceptance'
LABEL = 'conv-meta-fast-1'
ROOT = STATE / LABEL


def main():
    micro = json.loads((STATE / 'conv-metadata-micro.json').read_text())
    assert micro['parity_cases'] == 96 and micro['capture_replay']
    assert not ROOT.exists(), 'Do not overwrite prior validation'
    ROOT.mkdir()
    def save(phase):
        (STATE / 'conv-fused-progress.json').write_text(json.dumps(
            {'phase': phase, 'epoch': time.time(), 'label': LABEL}, indent=2))
        print(phase, flush=True)
    save('startup')
    with (ROOT / 'startup.log').open('w') as log:
        subprocess.run(['python3', '-I', 'scripts/acceptance_service.py',
            'candidate', LABEL, '--gran-conv-all', '1', '--conv-meta-fused', '1'],
            cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    def client(phase, script, *args):
        save(phase)
        command = ['systemd-run', '--user', '--scope', '--quiet', '--collect',
            '--unit=kandidat-fused-' + phase, '--slice=kandidat-test.slice',
            '-p', 'MemoryMax=3G', '-p', 'MemorySwapMax=0', '-p', 'CPUQuota=200%',
            '-p', 'TasksMax=128', 'env', 'CUDA_VISIBLE_DEVICES=',
            'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'python3', '-I',
            str(REPO / 'scripts' / script), str(ROOT / phase), *args]
        with (ROOT / (phase + '.log')).open('w') as log:
            subprocess.run(command, cwd=REPO, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    client('preemption-quality', 'acceptance_preemption_quality.py')
    client('regression', 'acceptance_streams.py', '--cases', 'pure8',
           '--prompt-tokens', '49147', '--tokens', '2048')
    client('quality', 'acceptance_fixtures.py', 'baza', 'clef')
    client('semantics', 'acceptance_semantics.py')
    client('state', 'acceptance_state.py')
    client('recall-distinct', 'acceptance_recall_distinct.py')
    client('warmup', 'acceptance_streams.py', '--cases', 'pure1,pure8,mixed34k', '--tokens', '512')
    client('measure', 'acceptance_streams.py', '--cases', 'pure1,pure8,mixed34k', '--tokens', '512', '--repeat', '3')
    client('queue', 'acceptance_streams.py', '--cases', 'pure40,mixedqueue34k', '--tokens', '1024')
    client('long-context', 'acceptance_fixtures.py', 'igla')
    log = (ROOT / 'server.log').read_text(errors='replace')
    if log.count('Fused convolution metadata active') < 2:
        raise RuntimeError('No evidence of fused metadata on both ranks')
    save('complete')


if __name__ == '__main__':
    main()
