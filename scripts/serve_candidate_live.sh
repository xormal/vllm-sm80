#!/usr/bin/env bash
# Authorized live-load trial. The original production unit remains unchanged.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/candidate_env.sh"
export CANDIDATE_VARIANT=candidate
export PY="$CANDIDATE_REPO/scripts/acceptance_python.sh"
export CARDS=0,1 HOST=0.0.0.0 PORT=8085
export SERVED_NAMES="huihui-qwen3.5-35b-a3b-abliterated-q8_0 gpt-4-0125-preview Qwen3.6-27B-AWQ Qwen3.8-27B-ABLITERATED"
export FA2SM70_CLEF=1 FA2SM80_SEL_DENSE_NB=1024 FA2SM70_EXCH12=1
export FA2SM80_RESUME_ADMISSION=1 FA2SM70_GRAN_CONV_ALL=1
export FA2SM80_BLOCKPIPE_FALLBACK_ROWS=0 FA2SM80_CONV_META_FUSED=0
export FA2SM80_MEMTRACE=900
# [ПЛАВНОСТЬ 30.10] по решению владельца после окон 9-10b: бюджет prefill 2048 токенов на шаг, пока идёт декод
# (без порога очереди), и голова Clef вне критического пути (сборщик с замком захвата графов)
export FA2SM80_PREFILL_BUDGET_DECODING=2048 FA2SM80_PREFILL_BUDGET_BACKLOG=0
export FA2SM70_CLEF_HEAD_ASYNC=1
export VLLM_CACHE_ROOT="$STATE_DIR/acceptance/vllm-candidate-fallback-0-conv-all-1"
export FA2SM70_CLEF_OUT="$STATE_DIR/acceptance/clef-candidate"
cd "$CANDIDATE_REPO"
exec scripts/serve_qwen38_sm80.sh
