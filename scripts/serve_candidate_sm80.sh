#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/candidate_env.sh"
case "${1:---paths}" in
  --paths)
    printf 'repo=%s\npython=%s\nmodel=%s\nstate=%s\nbuild=%s\n' \
      "$CANDIDATE_REPO" "$PY" "$MODEL_DIR" "$STATE_DIR" "$FA2SM70_BUILD_DIR"
    exit 0 ;;
  --dry-run) export DRYRUN=1; export CARDS="${CARDS:-0,1}" ;;
  --run) : "${CARDS:?Set CARDS=0,1 after the production replacement is authorized}"; export DRYRUN=0 ;;
  *) printf 'Usage: %s [--paths|--dry-run|--run]\n' "$0" >&2; exit 2 ;;
esac
if [ "$CARDS" != "0,1" ]; then
  printf 'Candidate supports only CARDS=0,1; GPU2-3 are reserved for training.\n' >&2
  exit 2
fi
export HOST="${HOST:-127.0.0.1}" PORT="${PORT:-8086}"
export FA2SM70_CLEF=1 FA2SM80_SEL_DENSE_NB=1024
# Candidate defaults selected by GPU acceptance. Experimental chunks/fusion
# stay opt-in; the original production launcher and configs are untouched.
export FA2SM80_RESUME_ADMISSION="${FA2SM80_RESUME_ADMISSION:-1}"
export FA2SM70_GRAN_CONV_ALL="${FA2SM70_GRAN_CONV_ALL:-1}"
export FA2SM80_BLOCKPIPE_FALLBACK_ROWS="${FA2SM80_BLOCKPIPE_FALLBACK_ROWS:-0}"
export FA2SM80_CONV_META_FUSED="${FA2SM80_CONV_META_FUSED:-0}"
export VLLM_CACHE_ROOT="$STATE_DIR/vllm-cache-conv-${FA2SM70_GRAN_CONV_ALL}-fused-${FA2SM80_CONV_META_FUSED}-rows-${FA2SM80_BLOCKPIPE_FALLBACK_ROWS}"
export SERVED_NAMES="Qwen3.8-27B-ABLITERATED-candidate"
cd "$CANDIDATE_REPO"
exec scripts/serve_qwen38_sm80.sh
