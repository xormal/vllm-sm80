#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
export VLLM_VERSION_OVERRIDE=0.0.3+sm80.candidate
exec /usr/bin/python3 "$REPO/scripts/candidate_build.py" -- \
  "$REPO/.venv/bin/python" -m pip install --no-build-isolation --no-deps -e "$REPO" -v
