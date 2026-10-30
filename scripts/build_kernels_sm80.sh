#!/usr/bin/env bash
# Prebuild the JIT CUDA kernels of sm70/fa2_sm70 for NVIDIA A100 (sm_80) -- without a GPU and in exactly the
# environment scripts/serve_qwen38_sm80.sh uses, so the server's first start loads them instead of rebuilding.
#
#   scripts/build_kernels_sm80.sh                          # into ~/.cache/vllm-sm80/build
#   PY=/opt/vllm-sm80/bin/python STATE_DIR=/srv/vllm-sm80 scripts/build_kernels_sm80.sh
#
# Environment:
#   PY             python of the virtualenv with this repo installed   (default <repo>/.venv/bin/python)
#   STATE_DIR      cache root of the server                            (default ~/.cache/vllm-sm80)
#   FA2SM70_BUILD_DIR  kernel build directory                          (default $STATE_DIR/build)
#   CUDA_HOME      CUDA 12.x toolkit                                    (default: nvcc on PATH if 12.x, else /usr/local/cuda-12.*)
#   MAX_JOBS       parallel nvcc jobs per extension                     (default 2; ~2-3 GB RAM each)
#   KERNELS        space-separated builders of fa2_sm70._ext            (default: the production set below)
#   DRYRUN=1       print the resolved environment and the list, build nothing
#
# Why "exactly the same environment": torch's JIT loader rebuilds an extension whenever its compile command
# changes. The command depends on TORCH_CUDA_ARCH_LIST (8.0 adds -DFA2_SM80=1), on CC/CXX (-ccbin), on the include
# paths and on the ninja found on PATH -- a build from a different environment is rebuilt by the server at start.
# The first build of the full set takes 15-30 minutes (CUTLASS prefill ~9 min, Marlin8 ~2 min).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
say() { printf '[kernels] %s\n' "$*"; }
die() { printf '[kernels] ERROR: %s\n' "$*" >&2; exit 1; }

PY="${PY:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || die "no python at $PY (set PY to the virtualenv's python)"
STATE_DIR="${STATE_DIR:-$HOME/.cache/vllm-sm80}"
MAX_JOBS="${MAX_JOBS:-2}"
# The production set: what serve_qwen38_sm80.sh prebuilds, plus Clef and Marlin8 (FA2SM80_W8_MARLIN=1).
# w8a8_sm80_ext is only needed with FA2SM80_W8A8 (off in production).
KERNELS="${KERNELS:-decode_ext w8a16_ext w8a16_sm80_ext gdn_ext gdn_head_ext prefill_ext w12a16_ext exch_ext conv_ext i4_ext cg_sm80_ext clef_ext marlin8_ext}"

# ---- toolchain: the same probe as the launcher (first host gcc that this nvcc accepts for sm_80)
if [ -z "${CUDA_HOME:-}" ]; then
  # the first CUDA 12.x toolkit: nvcc on PATH, then /usr/local/cuda-12.9 / 12.8 / 12 (CUDA 13 is not supported)
  for cand in "$(dirname "$(dirname "$(command -v nvcc 2>/dev/null || echo /nonexistent/bin/nvcc)")")" \
              /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12; do
    if [ -x "$cand/bin/nvcc" ] && "$cand/bin/nvcc" --version 2>/dev/null | grep -q "release 12\."; then
      CUDA_HOME="$cand"; break
    fi
  done
  [ -n "${CUDA_HOME:-}" ] || die "no CUDA 12.x toolkit found: set CUDA_HOME"
fi
[ -x "$CUDA_HOME/bin/nvcc" ] || die "no nvcc in $CUDA_HOME/bin"
"$CUDA_HOME/bin/nvcc" --version | grep -q "release 12\." || die "CUDA 12.x toolkit required ($CUDA_HOME)"
export CUDA_HOME
probe_dir="$(mktemp -d)"; trap 'rm -rf "$probe_dir"' EXIT
printf '#include <cuda_runtime.h>\n__global__ void k(){}\nint main(){return 0;}\n' > "$probe_dir/p.cu"
cc_ok=""
for cand in ${CC:-} /usr/bin/gcc /usr/bin/gcc-14 /usr/bin/gcc-13 /usr/bin/gcc-12 /usr/bin/gcc-11; do
  [ -x "$cand" ] || continue
  if "$CUDA_HOME/bin/nvcc" -ccbin "$cand" -arch=sm_80 -o "$probe_dir/p" "$probe_dir/p.cu" >/dev/null 2>&1; then
    cc_ok="$cand"; break
  fi
done
[ -n "$cc_ok" ] || die "no host gcc accepted by $CUDA_HOME/bin/nvcc for sm_80"
export CC="$cc_ok" CXX="${cc_ok/gcc/g++}" CUDAHOSTCXX="${cc_ok/gcc/g++}"
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS
# absolute: torch runs ninja from the build directory, where a relative .venv/bin is not found and the system
# ninja (another version, other command hashes) would rebuild every extension
PY_BIN="$(cd "$(dirname "$(command -v "$PY")")" && pwd)"
export PATH="$PY_BIN:$CUDA_HOME/bin:$PATH"                # the virtualenv's ninja first, as in the launcher
export FA2SM70_PATH="$REPO/sm70"
export PYTHONPATH="$REPO/sm70/integrations/vllm_sm70:$REPO/sm70${PYTHONPATH:+:$PYTHONPATH}"
export FA2SM70_BUILD_DIR="${FA2SM70_BUILD_DIR:-$STATE_DIR/build}"
# No GPU needed: compile for the explicit architecture (fa2_sm70._ext preflight), never touch a device.
export FA2SM70_BUILD_ONLY=1 CUDA_VISIBLE_DEVICES=""

say "CUDA_HOME=$CUDA_HOME CC=$CC ninja=$(command -v ninja || echo "python -m ninja") MAX_JOBS=$MAX_JOBS"
say "build dir $FA2SM70_BUILD_DIR; kernels: $KERNELS"
if [ "${DRYRUN:-0}" = "1" ]; then
  env | grep -E '^(CUDA_HOME|CC|CXX|CUDAHOSTCXX|TORCH_CUDA_ARCH_LIST|MAX_JOBS|FA2SM70_|PYTHONPATH)=' | sort | sed 's/^/    /'
  exit 0
fi
mkdir -p "$FA2SM70_BUILD_DIR"
"$PY" -u - $KERNELS <<'PYEOF'
import sys, time
import fa2_sm70._ext as e
bad = 0
for n in sys.argv[1:]:
    t = time.time()
    try:
        getattr(e, n)()
        print(f"[kernels]   ok {n} ({time.time() - t:.0f} s)", flush=True)
    except Exception as ex:  # noqa: BLE001
        bad += 1
        print(f"[kernels]   FAILED {n}: {str(ex)[-1500:]}", flush=True)
sys.exit(1 if bad else 0)
PYEOF
say "done"
