# vLLM-SM80: kernel changes against vLLM-SM70

Base: [`xormal/vllm-sm70`](https://github.com/xormal/vllm-sm70) at `8a59d8fb8` (2026-10-19). Target: 2× NVIDIA A100
PG199 32 GB (`sm_80`, 96 SMs, 1275 MHz, HBM ~1.32 TB/s), tensor parallel 2 over PCIe (GPU0↔GPU1 P2P ~11 GB/s through
the host bridge). Kernels live in `sm70/fa2_sm70/csrc/`, are built by `sm70/fa2_sm70/_ext.py` with
`TORCH_CUDA_ARCH_LIST=8.0` (adds `-DFA2_SM80=1`), and are wired in through `sm70/integrations/vllm_sm70/`.

The sm_70 kernels on an A100 were either broken (empty `WarpMma` specialisations) or emulated (`m8n8k4`), so every hot
path was ported or replaced with native Ampere code: `mma.m16n8k16` / `m16n8k32`, `ldmatrix`, `cp.async`, 16-byte
vector I/O. Unless stated otherwise outputs are bitwise identical to the previous path (checked on the GPU), and
every switch below is an environment lever; the values of the running server are in
[`LOAD_TESTS.md`](LOAD_TESTS.md#configuration).

37 files changed: 14 new, 22 modified, plus `_ext.py` (`git diff --stat 8a59d8fb8 -- sm70/fa2_sm70/csrc`).

## New kernels

| File | What it is | Switch | Result on A100 |
|---|---|---|---|
| `fwd_sm80_i8.cuh` | Prefill attention on the int8 paged KV pool: FA2 pipeline written directly in the GEMM code, d=256, GQA, causal; varlen mode takes its row from a device tile map (one call per layer, graph-safe) | default on sm80; varlen `FA2SM80_ATTN_VARLEN` | 103.5 TFLOP/s (the `volta_fwd_ws` port via sm80 `WarpMma` gave 52) |
| `decode_sm80.cuh` (+ `decode_sm80_ext.cu`, test TU) | Decode attention: int8 KV with a per-(token, head) scale, d=256, GQA + MTP rows; same partial-slice format as the Volta "chimera", so `flash_decode_hmma_paged` is unchanged | default on sm80 (`FA2SM80_DEC_CHIMERA=1` = old path) | 57 µs @ 32K, 281 µs @ 262K |
| `decode_hmma.cuh` | Volta "chimera" decode (K/V tile in shared memory, both GEMMs on tensor cores); kept for A/B | `FA2SM80_DEC_CHIMERA=1` | reference |
| `w8a16_sm80.cu` | GPTQ int8 (g=128, asymmetric) W8A16 for Ampere: decode / small-M kernel, `gemm_large` for mid M (MI=1/2/4, deterministic split-K), dequant + cuBLAS for large M; table-driven tuner cache (no timing-based choice, so numerics do not change between restarts) | default | M=48–64 2–2.5× vs the first port; batch-invariant |
| `w8a16_sw_sm80.cuh` | Decode with A/B swapped (x in B, weight in A: no zero half-tiles at M ≤ 8) | `FA2SM80_W8SW=1` | −5…−11 % kernel time |
| `w8a16_sm80_g3.cuh` | W8A16 v3 for mid/large M with A in fragment order (fewer MOVs in the main loop) | `FA2SM80_W8_G3` (off) | slower than dequant + GEMM on 1024–2048-row chunks |
| `w8a8_sm80.cu` | int8 activations × int8 GPTQ weights on `mma.m16n8k32`, no weight dequant | `FA2SM80_W8A8` (off) | accuracy floor: Clef 33/35 with group-128 activations |
| `marlin8_sm80.cu`, `marlin8_selector.inc` | vLLM's Marlin template instantiated for `kU8` **with zero points** (asymmetric GPTQ), 15 instances, inside our extension; device-wide lock/fp32-reduce scratch reserved before capture (graph-safe) | `FA2SM80_W8_MARLIN=1` | per model pass and rank: M=40 13.2 vs 22.0 ms, M=440 80.3 vs 107.6, M=1024 165 vs 177; M > 1024 goes through dequant + cuBLAS |
| `marlin8_dequant.cuh` | Marlin layout → W [N, K] fp16. v4: 64 columns × 256 k per block, 16-byte weight loads (512 B contiguous per warp, L1 bypass), scales and zero points in one vector load per thread, `PRMT` + `HSUB2` + `HMUL2`, XOR-swizzled shared memory (no bank conflicts), 512 B row stores per warp | `FA2SM80_M8DQ` = 2 / 41 / 42 (default 42 when K % 256 == 0) | all weights of a rank per step: **34.0 ms** (1024 GB/s) vs 89.6 ms for v2 and 49.1 ms for `w8a16_sm80_dequant`; bitwise, incl. subnormal fp16 scales |
| `m8dq_roofline.cu` | Standalone CUDA (no torch) ceiling for the dequant traffic shape: fp16 copy, ideal int8→fp16 stream, write-only fill, and v4 on the same sizes | tool | — |
| `cg_sm80.cu` | fp16 prefill GEMM on CUTLASS 2.x that is **batch-invariant** (no slice-K, k ascending, one `mma.m16n8k16` per step): per-element summation order does not depend on M, tile or row position | `FA2SM80_CG` | deterministic outputs across batch compositions |
| `ar1ll.cuh` | Two-card reduction through pinned host memory with "data + sequence number in one 16-byte write" (no fence + flag round trip) | `FA2SM80_AR_LL=1` | ~13 vs ~22 µs per call; 3M stress operations clean |
| `gdn_rec_v5.cuh` | Gated DeltaNet decode recurrence v5: reduce-scatter shuffles, state-independent prologue, 2 warps per scale group (named barrier), F2I-free int16 pack | `FA2SM80_GDN_V5=1` | 14.9 → 10.3 µs (N=1), bitwise |

## Modified kernels

| File | Change | Switch |
|---|---|---|
| `attn_fwd_cutlass.cu` | Dispatch to `fwd_sm80_i8.cuh` on sm80; prefill attention of all long rows in **one call per layer** with device metadata (`paged_varlen_meta`: tile map + page tables of every row in fixed-capacity buffers), shared by all attention layers of a step — no host sync, graph-safe | `FA2SM80_ATTN_VARLEN` |
| `split_decode.cu` | sm80 dispatch to `decode_sm80.cuh` instead of the chimera | default |
| `exch_i12.cu` | 12-bit TP exchange: unpack-add into a caller-provided output (persistent block-pipeline buffers, `unpack_add_i12_into`, aliasing-safe); copy-engine transport inside a CUDA graph with a device-side sequence number; AR1 whole reduction in one kernel (A100 host path) | `FA2SM70_EXCH12`, `FA2SM80_BLOCKPIPE_BUFS` |
| `clef.cu` | Clef decision head: `tap(h, run_heads)` + `run_pending_heads()` (heads on a side stream, rows collected on the main stream), `req_begin(sync=false)` with a pinned slot buffer + event, row windows for the row pipeline, graph-safe correction (`delta_graph`) | `FA2SM70_CLEF_HEAD_ASYNC` |
| `gdn_scan.cu` | Grid-size SPLIT of the GDN step (measured: N=1 12.74 / 11.82 / 14.14 µs at SPLIT 1/2/4); q/k/v/a/b accepted as strided views; output written straight into the caller's dense buffer (no copy node per GDN layer); varlen scan for the whole step | `FA2SM80_GDN_SPLIT`, `FA2SM80_GDN_ALL_VARLEN` |
| `volta_gdn_rec.cuh`, `volta_gdn_wu.cuh` | sm80 build: all-token input prefetch through `cp.async` before the state read, warps per scale group, per-token strides (drops five `.contiguous()` per layer = 5 graph nodes × 48 GDN layers) | default on sm80 |
| `volta_gdn_kkt.cuh`, `volta_gdn_o.cuh`, `volta_fwd_*.cuh`, `volta_gemm_f16.cuh`, `volta_bwd_block.cuh` | Accumulator-fragment lane mapping through `wmma_row_base` / `wmma_col_base` / `FA2_RSTEP` / `FA2_CHALF`: the Volta `m8n8k4` fragment layout differs from Ampere `m16n8k16`, so the hard-coded Volta lane formulas gave wrong rows/columns on sm80 | build |
| `conv_update.cu` | 16-byte vector I/O (8 fp16 per instruction), all independent loads issued first (MTP step grid is 5 blocks): 8.6 → 4.6 µs; native conv for the full mixed-step graph | default |
| `sparse_select.cu` | Selection block as a template parameter (B=64, D=256), sub-block to shared memory in one `cp.async` burst (14 → 5.4 µs cold); dense mode on the device when visible blocks ≤ `dense_nb` (A100 decode is faster dense) | `FA2SM80_SEL_DENSE_NB` |
| `w8a16_hmma.cuh`, `w12a16.cu`, `w12a16_hmma.cuh` | `FA2_SM80` guards: the Volta K=8 step geometry is compiled out on sm80 (`m16n8k16` needs K % 16 == 0); Volta paths unchanged | build |
| `_ext.py` | Builders for the new extensions (`marlin8_ext`, `w8a8`, `cg_sm80`, decode test TU); `-DFA2_SM80=1` when every target arch ≥ 8.0; per-extension build directories | build |

## Integration changes that the kernels depend on

* **Block pipeline** (`fa2sm80_blockpipe.py`): `out_proj → reduce → add + RMSNorm → gate_up → SiLU → down → reduce` in
  row chunks with reductions on a high-priority side stream; persistent chunk buffers and reduction into the output
  slice (`FA2SM80_BLOCKPIPE_BUFS`); mid-M pipeline with Marlin chunk GEMMs (`FA2SM80_BLOCKPIPE_M8_MIN=256`).
* **Host syncs removed** from the mixed step (~23 per step before): GDN token-index views and prefix slices
  (`FA2SM70_GDN_IDX_VIEW`), `num_computed` on the CPU (`FA2SM70_GDN_NCOMP_CPU`), pinned conv1d metadata ring
  (`FA2SM70_CONV_META_PINNED`), chunk metadata on the device (`FA2SM80_GDN_DEVMETA`), uniform spec-decode rows in
  mixed steps (`FA2SM80_MIX_DEC_UNIFORM`).
* **CUDA graphs**: decode FULL graphs and MTP loop graph for up to 9 requests (`FA2SM80_CG_SIZES` up to 45,
  `FA2SM80_MTP_LOOP_GRAPH_MAXB=9`). vLLM rounds capture sizes **up** to a multiple of k+1 and drops results above the
  largest listed size, so 45 must be listed explicitly (44 is dropped).
* **Scheduler**: decode interleave (`FA2SM80_DECODE_INTERLEAVE=4`: after a step with a prefill chunk during decode,
  four pure-decode steps), prefill budget while decoding (`FA2SM80_PREFILL_BUDGET_DECODING`).

The investigation behind each change (profiles, A/B windows, failed attempts) is in
`docs/research/2026-10-28-qwen38/` (`SMOOTHNESS_WINDOW*.md`, `KERNEL_ARCHITECTURE.md`).
