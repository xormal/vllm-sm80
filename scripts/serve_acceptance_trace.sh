#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/candidate_env.sh"
[[ "${CANDIDATE_VARIANT:?}" == candidate ]]
: "${CANDIDATE_SCHEDULER_TRACE:?}"
export PY="$CANDIDATE_REPO/scripts/acceptance_trace_python.sh"
export CARDS=0,1 HOST=127.0.0.1 PORT=8086
export SERVED_NAMES=Qwen3.8-27B-ABLITERATED-candidate
export FA2SM70_CLEF=1 FA2SM80_SEL_DENSE_NB=1024 FA2SM70_EXCH12=1
export VLLM_CACHE_ROOT="$STATE_DIR/acceptance/vllm-candidate-fallback-${FA2SM80_BLOCKPIPE_FALLBACK_ROWS:-0}"
export FA2SM70_CLEF_OUT="$STATE_DIR/acceptance/clef-candidate-trace"
cd "$CANDIDATE_REPO"
exec scripts/serve_qwen38_sm80.sh
