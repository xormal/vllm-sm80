#!/usr/bin/env bash
# [ПЛАВНОСТЬ 29.10] Только для A/B-экземпляра: учёт времени и состава шагов (fa2sm80_step_timing).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${FA2SM80_STEP_TIMING:?}"
exec "$REPO/.venv/bin/python" "$@" --scheduler-cls fa2sm80_step_timing.StepTimingAsyncScheduler
