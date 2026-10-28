#!/usr/bin/env bash
# Transient validation service, deliberately separate from agents on port 8085.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(dirname "$REPO")"
if [ "${1:-}" != "--run" ]; then
  printf 'Usage: %s --run\nGPU0-1; http://127.0.0.1:8086; no automatic restart or boot enable.\n' "$0"
  exit 0
fi
state="$(systemctl --user show sm80-vllm.service -p ActiveState --value)"
case "$state" in
  inactive|failed) ;;
  *) printf 'Production is %s; stop it before candidate validation.\n' "$state" >&2; exit 1 ;;
esac
mkdir -p "$ROOT/state/gpu-validation"
exec systemd-run --user --unit=kandidat-validation \
  --description='Candidate validation GPU0-1 localhost8086' \
  --property=Type=exec --property=Slice=sm80.slice --property=Restart=no \
  --property=MemoryHigh=20G --property=MemoryMax=21G --property=MemorySwapMax=0 \
  --property=CPUQuota=1600% --property=TasksMax=4096 \
  --property=OOMPolicy=kill --property=OOMScoreAdjust=900 \
  --property=TimeoutStopSec=45 --property=KillMode=control-group \
  --property="WorkingDirectory=$REPO" \
  --property="StandardOutput=append:$ROOT/state/gpu-validation/server.log" \
  --property=StandardError=inherit \
  --setenv=CARDS=0,1 --setenv=HOST=127.0.0.1 --setenv=PORT=8086 \
  --setenv=FA2SM70_EXCH12=1 \
  --setenv="FA2SM80_BLOCKPIPE_FALLBACK_ROWS=${FA2SM80_BLOCKPIPE_FALLBACK_ROWS:-0}" \
  --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=TORCHINDUCTOR_COMPILE_THREADS=1 \
  "$REPO/scripts/serve_candidate_sm80.sh" --run
