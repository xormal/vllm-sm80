#!/usr/bin/env bash
# [ПЛАВНОСТЬ 29.10] A/B-экземпляр в окне обслуживания: профиль и кэши боевого кандидата (serve_candidate_live.sh),
# но 127.0.0.1:8086, бюджет prefill из файла FA2SM80_PREFILL_BUDGET_FILE и учёт шагов FA2SM80_STEP_TIMING.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/candidate_env.sh"
: "${FA2SM80_PREFILL_BUDGET_FILE:?}" "${FA2SM80_STEP_TIMING:?}"
export CANDIDATE_VARIANT=candidate
export PY="$CANDIDATE_REPO/scripts/acceptance_steptime_python.sh"
export CARDS=0,1 HOST=127.0.0.1 PORT="${AB_PORT:-8086}"   # AB_PORT -- порт, о котором агенты-потребители не знают
export SERVED_NAMES="Qwen3.8-27B-ABLITERATED"
export FA2SM70_CLEF=1 FA2SM80_SEL_DENSE_NB=1024 FA2SM70_EXCH12=1
export FA2SM80_RESUME_ADMISSION=1 FA2SM70_GRAN_CONV_ALL=1
export FA2SM80_BLOCKPIPE_FALLBACK_ROWS=0 FA2SM80_CONV_META_FUSED=0
export FA2SM80_MEMTRACE="${FA2SM80_MEMTRACE:-900}"
export VLLM_CACHE_ROOT="$STATE_DIR/acceptance/vllm-candidate-fallback-0-conv-all-1"
export FA2SM70_CLEF_OUT="$STATE_DIR/acceptance/clef-candidate-ab"
cd "$CANDIDATE_REPO"
exec scripts/serve_qwen38_sm80.sh
