#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${CANDIDATE_VARIANT:?}" in
  baseline) export PYTHONPATH="$(dirname "$REPO")/state/acceptance/baseline-overlay:${PYTHONPATH:-}" ;;
  candidate) ;;
  *) exit 2 ;;
esac
exec "$REPO/.venv/bin/python" "$@"
