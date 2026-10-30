# Tuning tables measured on NVIDIA DRIVE A100 PROD (PG199)

Copied from the production state directory on 2026-10-31; `scripts/install_from_source_sm80.sh` installs them into
`STATE_DIR` (kept if already present).

* `w8tune/<device>_K<K>_N<N>_v2.json` — kernel choice per GPTQ-int8 weight shape (`fa2sm80_w8.py`, read from
  `FA2SM80_W8_TUNE_DIR`, default `$STATE_DIR/w8tune`): warps / stages / split-K for each M bucket, the direct-vs-dequant
  route, the mid-M kernel. The choice is made by timing on the first start and then written here, because timing is
  not reproducible between starts or ranks, and a different split-K changes the summation order (greedy texts of two
  starts diverged after 34–107 tokens). With these files both ranks and every start use the same configurations.
* `cgtune.json` — CUTLASS prefill GEMM configuration per (M bucket, N, K) (`fa2sm80_cg.py`, `FA2SM80_CG_TABLE`, default
  `$STATE_DIR/cgtune.json`); the choice does not change numbers (batch-invariant kernel), only speed.

The file name carries the device name reported by CUDA, so on another GPU model these tables are ignored and the
first start tunes and writes its own.
