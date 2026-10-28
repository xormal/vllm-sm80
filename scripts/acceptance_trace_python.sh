#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CANDIDATE_SCHEDULER_TRACE:?}"
exec "$REPO/.venv/bin/python" "$@" --scheduler-cls fa2sm80_scheduler_trace.TraceAsyncScheduler
