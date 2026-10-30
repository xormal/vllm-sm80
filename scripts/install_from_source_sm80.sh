#!/usr/bin/env bash
# Build vllm-sm80 from source into a virtualenv: Python 3.12, CUDA 12.x, NVIDIA Ampere A100 (sm_80).
#
#   scripts/install_from_source_sm80.sh                        # .venv in the repository, caches in ~/.cache/vllm-sm80
#   VENV=/opt/vllm-sm80 STATE_DIR=/srv/vllm-sm80 MAX_JOBS=4 scripts/install_from_source_sm80.sh
#   DRYRUN=1 scripts/install_from_source_sm80.sh               # check the toolchain and print the plan only
#
# Environment:
#   VENV        virtualenv to create or reuse                     (default <repo>/.venv)
#   PYTHON      interpreter used to create it                     (default python3.12)
#   CUDA_HOME   CUDA 12.x toolkit                                 (default: nvcc on PATH if 12.x, else /usr/local/cuda-12.9/12.8/12)
#   MAX_JOBS    parallel nvcc jobs for vLLM (~4-5 GB RAM each:    (default 4)
#               Marlin/CUTLASS units are heavy; MAX_JOBS=6 ran out of a 14 GB cap on 2026-10-23)
#   STATE_DIR   server cache root: tuning tables and kernel builds (default ~/.cache/vllm-sm80)
#   CMAKE_BUILD_TYPE  vLLM build type                             (default RelWithDebInfo, as setup.py)
#   PREBUILD    1 = also prebuild the fa2_sm70 JIT kernels        (default 1; scripts/build_kernels_sm80.sh)
#
# What it does, mirroring the deployed environment:
#   1. pinned packages (requirements-sm80.txt = requirements-sm70.txt: torch 2.9.1+cu128, triton 3.5.1, ...);
#   2. vLLM's C++/CUDA code for sm_80 only, and only the targets the server loads: _C, cumem_allocator and
#      triton_kernels. vLLM's own FlashAttention (FA2/FA3), FlashMLA and the MoE kernels are not built: attention is
#      ours (sm70/fa2_sm70) and the model is dense. The Volta TurboMind kernels compile out on sm_80;
#   3. registers the source tree in the virtualenv (.pth + dist-info vllm 0.0.3+sm80), as on the deployment;
#   4. installs the W8 and CUTLASS-GEMM tuning tables measured on DRIVE A100 PROD (configs/sm80-tune) into
#      STATE_DIR -- without them the first start picks kernel configurations by timing, which is not reproducible;
#      on other GPUs the tables do not match the device name and the first start tunes;
#   5. prebuilds the fa2_sm70 JIT kernels in the launcher's environment (no GPU needed).
# Serve afterwards with: PY=$VENV/bin/python MODEL_DIR=... STATE_DIR=... scripts/serve_qwen38_sm80.sh
#
# vLLM's CMake configuration clones CUTLASS, vllm-flash-attn, FlashMLA, qutlass and triton_kernels into .deps even
# for these targets; offline, point VLLM_CUTLASS_SRC_DIR, VLLM_FLASH_ATTN_SRC_DIR, FLASH_MLA_SRC_DIR,
# QUTLASS_SRC_DIR and TRITON_KERNELS_SRC_DIR at local checkouts (revisions: README of vllm-sm70, "Installation").
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO/.venv}"
PYTHON="${PYTHON:-python3.12}"
MAX_JOBS="${MAX_JOBS:-4}"
STATE_DIR="${STATE_DIR:-$HOME/.cache/vllm-sm80}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
VERSION="0.0.3+sm80"
BUILD_DIR="$REPO/build/sm80"
say() { printf '[install-sm80] %s\n' "$*"; }
die() { printf '[install-sm80] ERROR: %s\n' "$*" >&2; exit 1; }

# ---- 0. toolchain
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
[ -x "$CUDA_HOME/bin/nvcc" ] || die "no nvcc in $CUDA_HOME/bin (set CUDA_HOME to a CUDA 12.x toolkit)"
"$CUDA_HOME/bin/nvcc" --version | grep -q "release 12\." || die "CUDA 12.x toolkit required ($CUDA_HOME)"
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS NVCC_THREADS=1 VLLM_TARGET_DEVICE=cuda
probe_dir="$(mktemp -d)"; trap 'rm -rf "$probe_dir"' EXIT
printf '#include <cuda_runtime.h>\n__global__ void k(){}\nint main(){return 0;}\n' > "$probe_dir/p.cu"
cc_ok=""
for cand in ${CC:-} /usr/bin/gcc /usr/bin/gcc-14 /usr/bin/gcc-13 /usr/bin/gcc-12 /usr/bin/gcc-11; do
  [ -x "$cand" ] || continue
  if "$CUDA_HOME/bin/nvcc" -ccbin "$cand" -arch=sm_80 -o "$probe_dir/p" "$probe_dir/p.cu" >/dev/null 2>&1; then
    cc_ok="$cand"; break
  fi
done
[ -n "$cc_ok" ] || die "no host gcc accepted by nvcc for sm_80"
export CC="$cc_ok" CXX="${cc_ok/gcc/g++}" CUDAHOSTCXX="${cc_ok/gcc/g++}"
say "CUDA_HOME=$CUDA_HOME ($("$CUDA_HOME/bin/nvcc" --version | grep -o 'release [0-9.]*')) CC=$CC MAX_JOBS=$MAX_JOBS"
say "VENV=$VENV STATE_DIR=$STATE_DIR build=$BUILD_TYPE in $BUILD_DIR"
if [ "${DRYRUN:-0}" = "1" ]; then
  say "plan: pip -r requirements-sm80.txt -> cmake ($BUILD_TYPE, sm_80) targets _C cumem_allocator triton_kernels -> install into"
  say "      $REPO/vllm -> register $VERSION in $VENV -> tuning tables -> $STATE_DIR -> kernels (PREBUILD=${PREBUILD:-1})"
  exit 0
fi

# ---- 1. pinned environment
# Never rebuild under a running server: steps 2-3 replace vllm/_C.abi3.so and the registration it imports.
if [ -x "$VENV/bin/python" ] && pgrep -f "$VENV/bin/python" >/dev/null 2>&1 && [ "${FORCE:-0}" != "1" ]; then
  die "a process runs $VENV/bin/python (a server?): stop it or use another VENV (FORCE=1 to override)"
fi
[ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || die "Python 3.12 required in $VENV"
export PATH="$VENV/bin:$PATH"                 # the venv's cmake (>= 3.26) and ninja
"$PY" -m pip install --upgrade pip
say "installing the pinned environment (requirements-sm80.txt)"
"$PY" -m pip install -r "$REPO/requirements-sm80.txt"

# ---- 2. vLLM C++/CUDA for sm_80: configure like setup.py, build only what the server loads
say "configuring vLLM for sm_80"
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
mkdir -p "$BUILD_DIR"
cmake -S "$REPO" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DVLLM_TARGET_DEVICE=cuda \
  -DVLLM_PYTHON_EXECUTABLE="$PY" -DVLLM_PYTHON_PATH="$("$PY" -c 'import sys; print(":".join(sys.path))')" \
  -DFETCHCONTENT_BASE_DIR="${FETCHCONTENT_BASE_DIR:-$REPO/.deps}" -DNVCC_THREADS=1 \
  -DCMAKE_JOB_POOL_COMPILE:STRING=compile -DCMAKE_JOB_POOLS:STRING=compile="$MAX_JOBS" \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" ${CMAKE_ARGS:-}
say "building _C, cumem_allocator, triton_kernels (about an hour with MAX_JOBS=4)"
cmake --build "$BUILD_DIR" -j="$MAX_JOBS" --target _C cumem_allocator triton_kernels
for c in _C cumem_allocator triton_kernels; do
  cmake --install "$BUILD_DIR" --prefix "$REPO" --component "$c"
done

# ---- 3. register the source tree (no pip rebuild: the targets above are all the server needs)
rm -f "$SITE"/vllm*.pth "$SITE"/__editable__*vllm* 2>/dev/null || true
rm -rf "$SITE"/vllm-*.dist-info
printf '%s\n' "$REPO" > "$SITE/vllm_sm80_src.pth"
DI="$SITE/vllm-$VERSION.dist-info"
mkdir -p "$DI"
printf 'Metadata-Version: 2.1\nName: vllm\nVersion: %s\n' "$VERSION" > "$DI/METADATA"
printf 'pip\n' > "$DI/INSTALLER"
printf 'vllm_sm80_src.pth,,\n' > "$DI/RECORD"
"$PY" - <<'EOF'
import importlib.metadata as md, os
import torch, vllm
so = os.path.join(os.path.dirname(vllm.__file__), "_C.abi3.so")
assert os.path.exists(so), so
torch.ops.load_library(so)
print(f"[install-sm80] ok: vllm {md.version('vllm')} from {os.path.dirname(vllm.__file__)}, torch {torch.__version__}, "
      f"CUDA {torch.version.cuda}")
EOF

# ---- 4. tuning tables measured on DRIVE A100 PROD (kept if already present)
mkdir -p "$STATE_DIR/w8tune"
for f in "$REPO"/configs/sm80-tune/w8tune/*.json; do
  [ -e "$STATE_DIR/w8tune/$(basename "$f")" ] || cp "$f" "$STATE_DIR/w8tune/"
done
[ -e "$STATE_DIR/cgtune.json" ] || cp "$REPO/configs/sm80-tune/cgtune.json" "$STATE_DIR/cgtune.json"
say "tuning tables in $STATE_DIR (w8tune/, cgtune.json)"

# ---- 5. JIT kernels
if [ "${PREBUILD:-1}" = "1" ]; then
  PY="$PY" STATE_DIR="$STATE_DIR" CUDA_HOME="$CUDA_HOME" CC="$CC" "$REPO/scripts/build_kernels_sm80.sh"
fi
say "done. Serve with: PY=$PY MODEL_DIR=... STATE_DIR=$STATE_DIR scripts/serve_qwen38_sm80.sh"
