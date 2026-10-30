# vLLM-SM80

**English** · [Русский](README.ru.md)

**The production vLLM stack for NVIDIA Ampere (A100, `sm_80`).** It serves Qwen3.8-27B (GPTQ-Int8) with a
262,144-token context on two 32 GB cards, plus Cloudflare's Clef decision model on the same weights.

This is the code of the running server: the vLLM fork, the kernel library ported to Ampere and the integration
layer, with the launcher and the lever set. It is a fork of [xormal/vllm-sm70](https://github.com/xormal/vllm-sm70)
(the V100 production stack; common base `8a59d8fb8`, branch `sm70`), which itself forks
[1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) and [vllm-project/vllm](https://github.com/vllm-project/vllm).
A100 production since 2026-10-25; the configuration described here since 2026-10-31.

Model weights and Clef adapters: **[alexxorm/Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef](https://huggingface.co/alexxorm/Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef)** on Hugging Face.

| | |
|---|---|
| **Model** | [Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef](https://huggingface.co/alexxorm/Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef) (int8 GPTQ of [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16), base [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)) + adapters for [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) |
| **Hardware** | 2× NVIDIA DRIVE A100 PROD (PG199) 32 GB, tensor parallel 2 over PCIe 3.0, no NVLink |
| **Context** | 262,144 tokens, int8 KV cache |
| **Modalities** | text, images, video, tool calls (OpenAI-compatible API) |
| **Decisions** | `POST /v1/systemone`: Clef (Jev/SystemOne API) on the same weights, records up to 34,816 tokens with images |
| **Speculation** | native MTP head, k = 4, with n-gram assist |

* **[Kernel changes against vLLM-SM70](docs/sm80/KERNELS.md)** — every new and modified kernel with its switch and
  its A100 result.
* **[Load tests and metrics](docs/sm80/LOAD_TESTS.md)** of the running configuration, raw results in
  [`docs/sm80/results/`](docs/sm80/results/).

---

## Why

On an A100 the standard machinery already runs — FlashAttention 2, Marlin, cuBLAS, CUDA graphs — so nothing here
exists to make the model start, unlike on Volta. This branch is about how far a pair of the weakest A100s can be
pushed for one model. Two targets were set when the port began (2026-10-23):

* **prefill above 5,000 tokens/s**, and
* **single-stream decode above 100 tokens/s**,

without changing what the model computes: the same GPTQ-Int8 weights and int8 KV cache, with every change checked
against the V100 production reference (Clef decisions) and against the previous build (greedy LLM fingerprints).
That rules out the usual shortcuts:

* **DFlash 2 drafting.** DFlash ([paper](https://arxiv.org/abs/2602.06036)) drafts a whole block of tokens with a
  small block-diffusion model conditioned on the target's hidden states; DFlash 2 adds dynamic convolutions and a
  candidate-path selector. For Qwen3.8-27B the authors report acceptance lengths of 4.1–5.5 against 3.7–5.0 for the
  model's own MTP head, and 184–236 tok/s at concurrency 1 on one H200; a third-party test on a single A100 shows
  28.9 → 59.1 tok/s. The authors call it lossless (the target verifies every drafted token), and the published
  results cover the base model only. The decisive point: the drafter reads the target's hidden states and is
  trained on those of `Qwen/Qwen3.8-27B`, so **no DFlash 2 drafter exists for the abliterated weights served
  here** — one would have to be trained (2B parameters), and a mismatched drafter would only lower acceptance. The
  runtime contains DFlash/DFlash2 drafter support (`vllm/v1/spec_decode/dflash*.py`); it is not used. We use the
  model's own MTP head (k = 4) with n-gram assist.
* **8-bit activations (W8A8).** Implemented (`w8a8_sm80.cu`) and measured: Clef reaches 33/35 at best with
  group-128 activation scales against 34–35/35 for the int8-weight path, so it stays off.
* **Sparse prefill of long chunks.** Faster, but measurably changes answers about the end of long prompts.

**Result:** prefill **5.2 K tok/s** on 8K prompts (5,216 alone, 5,222–5,256 with 4–16 concurrent prompts) and
**114–152 tok/s** single-stream decode depending on the text (MTP acceptance), with Clef 34/35 against the V100
reference and LLM fingerprints 160/160. Under a saturated Clef queue, 1–9 chats keep 25–33 tok/s per stream with
the longest pause p99 0.35–0.45 s. Longer prompts prefill slower as attention grows (4.8 K tok/s at 32K and 3.8 K at
131K in the 2026-10-24 measurement).

## These cards in the A100 line

| | A100 PCIe 40 GB | A100 SXM4 40 GB | A100 PCIe 80 GB | A100 SXM4 80 GB | **DRIVE A100 PROD (PG199), here** |
|---|---|---|---|---|---|
| SMs | 108 | 108 | 108 | 108 | **96** |
| Max SM clock | 1410 MHz | 1410 MHz | 1410 MHz | 1410 MHz | **1275 MHz** |
| FP16 tensor, dense | 312 TFLOPS | 312 TFLOPS | 312 TFLOPS | 312 TFLOPS | **248–251 TFLOPS** (cuBLAS reaches 216) |
| Memory | 40 GB HBM2 | 40 GB HBM2 | 80 GB HBM2e | 80 GB HBM2e | **32 GB** HBM2 |
| Memory bandwidth | 1,555 GB/s | 1,555 GB/s | 1,935 GB/s | 2,039 GB/s | **1,438 GB/s** peak, ≈1,270 GB/s measured copy |
| Interconnect | PCIe 4.0, NVLink bridge | NVLink 600 GB/s | PCIe 4.0, NVLink bridge | NVLink 600 GB/s | **no NVLink**; PCIe 4.0 card, link trained at 3.0 x16 in this host |
| Max TDP | 250 W | 400 W | 300 W | 400 W | not reported |

### Against an A100 40 GB, as the hardware reports it

Values for the PG199 come from the driver (`cuDeviceGetAttribute`) and `nvidia-smi -q` on both cards of the
deployment; the A100 40 GB column is the datasheet.

| | A100 40 GB (PCIe) | DRIVE A100 PROD (PG199) | |
|---|---|---|---|
| GPU | GA100, compute capability 8.0 | GA100 (PCI device `20BB`, GPU part `20BB-550-A1`), compute capability 8.0 | same chip and SM |
| SMs / FP32 cores / tensor cores | 108 / 6,912 / 432 | **96 / 6,144 / 384** | −11 % |
| SM clock | 1410 MHz boost | **1275 MHz max, 1260 MHz customer boost**; 1140–1200 MHz observed under our load | −10 % |
| FP16 / BF16 tensor, dense | 312 TFLOPS | **248–251 TFLOPS** | −20 % |
| INT8 tensor, dense | 624 TOPS | **≈500 TOPS** | −20 % |
| FP32 / FP64 | 19.5 / 9.7 TFLOPS | **≈15.7 / 7.8 TFLOPS** | −20 % |
| L2 cache | 40 MB | **32 MB** (persisting up to 20 MB) | −20 % |
| Shared memory / registers per SM | 164 KB / 64 K | 164 KB / 64 K | same |
| Memory | 40 GB HBM2, 5 stacks, 5120-bit, 1215 MHz | **32 GB HBM2, 4 stacks, 4096-bit, 1404 MHz**; 31.55 GiB usable with ECC on | −20 % capacity |
| Peak memory bandwidth | 1,555 GB/s | **1,438 GB/s** (≈1,270 GB/s measured fp16 copy) | −8 % peak |
| PCIe | 4.0 x16 | 4.0 x16 capable; **3.0 x16** trained in this host (slot/riser) | half the host bandwidth |
| NVLink | bridge, 600 GB/s | **none**: GPU0↔GPU1 ≈11 GB/s through the host bridge | |
| Power | 250 W TDP | limit not reported; 126–160 W drawn under our load, shutdown at 110 °C | |
| Board | PCIe card | DRIVE board `900-6G199-0000-C00`, VBIOS 92.00.79.00.01 | |

So the per-SM machinery is the A100's; there is less of it (96 SMs, 32 MB L2), it runs slower (1275 vs 1410 MHz),
with a narrower memory (4 HBM stacks) and no NVLink. Third-party spec aggregators list 108 SMs and 1.87 TB/s for
this board, which does not match what the hardware reports. Against a data-center A100 the pair has about 80 % of
the compute, 70–92 % of the peak memory bandwidth (1,438 against 1,555–2,039 GB/s), 20–60 % less memory per card, and the tensor-parallel exchange of
every layer goes over PCIe 3.0 through the host.

Sources: [NVIDIA A100 80 GB datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf)
(312 TFLOPS, 1,935 / 2,039 GB/s, 300 / 400 W, NVLink bridge), [A100 specifications summary](https://www.thundercompute.com/blog/nvidia-a100-specs-full-guide)
(108 SMs, 1410 MHz, 40 GB variants), [DFlash paper](https://arxiv.org/abs/2602.06036),
[DFlash 2 drafter for Qwen3.8-27B](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2),
[DFlash 2 on a single A100](https://www.mindstudio.ai/blog/dflash-2-speculative-decoding-qwen),
[DFlash/DFlash2 in vLLM](https://docs.nvidia.com/nemo/speech/nightly/speechlm2/vllm_dflash.html). PG199 figures:
driver attributes, `nvidia-smi -q` and measurements on the deployment ([`docs/sm80/results/hardware-pg199.txt`](docs/sm80/results/hardware-pg199.txt)).

## What was done

The Volta kernels were either broken on sm_80 (empty `WarpMma` specialisations) or emulated (`m8n8k4`), so every hot
path was ported or replaced with native Ampere code; outputs are bitwise identical to the previous path unless stated
otherwise. Highlights, details in [docs/sm80/KERNELS.md](docs/sm80/KERNELS.md):

* **Attention:** FA2 prefill on the int8 paged KV pool (103.5 TFLOP/s, one call per layer with device metadata),
  int8 decode attention (57 µs at 32K, 281 µs at 262K).
* **Linear layers:** GPTQ-int8 W8A16 kernels for decode and mid M, vLLM's Marlin instantiated with zero points for
  asymmetric GPTQ (1.1–1.9× at M ≤ 1024), a new dequant from the Marlin layout for large M (34.0 ms per step vs 89.6
  and 49.1 for the previous paths), batch-invariant prefill GEMM.
* **Tensor-parallel exchange without NVLink:** 12-bit compressed exchange, host-memory reduction with data and
  sequence number in one 16-byte write, a block pipeline that overlaps the reductions with the MLP GEMMs.
* **Gated DeltaNet and the mixed step:** ported recurrence, conv and selection kernels, host synchronisations
  removed from the mixed LLM + Clef step, full decode graphs for every batch the state pool admits.
* **Scheduling:** decode interleave and a prefill budget while chats decode, so a Clef queue does not stall chats.

## Measured on the reference deployment

| 2× DRIVE A100 PROD 32 GB, TP2 | |
|---|---|
| Prefill, 8K prompts | 5,216 tok/s alone, 5,222–5,256 tok/s with 4–16 concurrent prompts |
| Decode, 1 / 4 / 8 / 16 streams | 114–152 / 92 / 57–60 / 52 tok/s per stream (9 requests at once at most; 16 queue) |
| Clef alone, 2000-token records | 3,080 / 4,090 input tok/s with 1 / 4 clients (3 slots) |
| Chats under a saturated Clef queue | 4 chats 32–33 tok/s, 8 chats 25–27 tok/s per stream, pause p99 0.35–0.45 s |
| LLM prefill under a saturated Clef queue | 2.8 / 3.7 / 4.2 / 4.7–4.8 K tok/s with 1 / 4 / 8 / 16 prompts, Clef keeps 1.1–3.5 K |
| Accuracy | Clef 34/35 against the V100 reference (max \|Δp\| 0.0251), LLM fingerprints 160/160 at 600 / 2048 / 5000 tokens |

Method, configuration and every table: [docs/sm80/LOAD_TESTS.md](docs/sm80/LOAD_TESTS.md).

## Requirements and build

* 2× NVIDIA A100 (tested: one pair of DRIVE A100 PROD 32 GB), dedicated to the server (fixed memory plan,
  `--gpu-memory-utilization 0.96`).
* CUDA 12.x toolkit (the deployment uses 12.9) and Python 3.12. The kernels in `sm70/fa2_sm70` are JIT-built for
  sm_80 when `TORCH_CUDA_ARCH_LIST=8.0` (adds `-DFA2_SM80=1`).
* The scripted source build in this repository, [`scripts/install_from_source.sh`](scripts/install_from_source.sh),
  is still the sm_70 one (`TORCH_CUDA_ARCH_LIST=7.0`); a scripted sm_80 environment build is not published yet.

## Serving

```bash
PY=.venv/bin/python MODEL_DIR=/models/Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef \
CARDS=0,1 PORT=8000 scripts/serve_qwen38_sm80.sh
```

[`scripts/serve_qwen38_sm80.sh`](scripts/serve_qwen38_sm80.sh) loads [`configs/qwen38-27b-int8-sm80.env`](configs/qwen38-27b-int8-sm80.env)
over [`configs/qwen38-27b-int8-clef.env`](configs/qwen38-27b-int8-clef.env) (variables already set by the caller
win), builds all kernels in one process and starts the OpenAI-compatible server. The production profile on top of
it is [`scripts/serve_candidate_live.sh`](scripts/serve_candidate_live.sh); its levers are listed in
[LOAD_TESTS.md](docs/sm80/LOAD_TESTS.md#configuration). `DRYRUN=1` prints the resolved environment and command.
In this tree the launcher only accepts `CARDS=0,1` (site policy: the other cards of the host belong to another
project).

### Chat, vision, tools

The standard OpenAI-compatible endpoints: `/v1/chat/completions` (text and images up to 20 per request,
video up to 2), `/v1/completions`, `/v1/responses`, tool calling (`qwen3_coder` parser) and reasoning
(`qwen3` parser).

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-27B-ABLITERATED",
  "messages": [{"role": "user", "content": "Explain warp divergence in one paragraph."}]}'
```

### Decisions: `POST /v1/systemone` (Clef)

```bash
curl http://localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "clef",
  "state": "Our checkout started returning errors and orders are blocked.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle the message?",
                   "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
    "urgency":    {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
    "outage":     {"type": "noul", "instructions": "Is a service down?"}}}'
```

Question types: `choice` (options → `choice`, `probabilities`), `score` (ordered list → expected `score`),
`noul` (→ probability of "yes"); `state` is a string or any JSON value; images go in `"images"`. The record runs as
a one-token prefill with Clef's weight correction applied inside the int8 GEMM kernels for its rows only, and the
final hidden states feed Clef's joint schema head (native CUDA). Code:
[`sm70/integrations/vllm_sm70/fa2sm70_clef.py`](sm70/integrations/vllm_sm70/fa2sm70_clef.py),
[`sm70/fa2_sm70/csrc/clef.cu`](sm70/fa2_sm70/csrc/clef.cu). The full API description is in the
[vLLM-SM70 README](https://github.com/xormal/vllm-sm70#decisions-post-v1systemone-clef).

## Repository layout

```
vllm/                     vLLM 0.0.3 (1Cat-vLLM) + the production Python runtime (attention backend, GDN path,
                          MTP / DFlash / n-gram drafters, scheduler budget and interleave, CUDA-graph sizing)
sm70/                     kernel library and integration layer (MIT); builds for sm_70 and, with FA2_SM80, sm_80
  fa2_sm70/csrc/          CUDA kernels: *_sm80.* are the Ampere ports and additions, see docs/sm80/KERNELS.md
  integrations/vllm_sm70/ runtime layer: linear paths, block pipeline, TP exchange, MTP head, Clef
scripts/                  launchers (serve_qwen38_sm80.sh, serve_candidate_live.sh), bench_conc.py, kernel
                          benches, maintenance-window scripts (window*.sh)
configs/                  lever sets (qwen38-27b-int8-sm80.env over qwen38-27b-int8-clef.env), systemd units
docs/sm80/                kernel changes, load tests, raw results
docs/research/            the investigation journal (Russian): profiles, A/B windows, failed attempts
```

Many code comments and the research journal are engineering notes in Russian; they record the measurement behind
each decision.

## Known limitations

* Tested on one pair of DRIVE A100 PROD 32 GB only. Data-center A100s have more SMs, bandwidth and memory, and
  NVLink; the W8 tuning tables, CUDA-graph sizes and the prefill budget were measured on these cards.
* At most 9 requests decode at once (Gated DeltaNet state pool at k = 4); Clef has 3 slots, further records wait.
* Under a saturated Clef queue the prefill budget of 1024 trades 12–23 % of Clef throughput for chat smoothness.
* `/v1/completions` with `echo` and `logprobs` on long prompts allocates full-vocabulary logits per prompt chunk and
  can exhaust memory; do not expose prompt logprobs to untrusted clients.
* With speculation, decode logprobs depend at the ~1e-3 level on the GPU memory layout at startup; bitwise
  comparisons are only meaningful between identical configurations.
* No DFlash/DFlash 2 drafter is trained for these weights.

## License

* vLLM and 1Cat-vLLM code (everything outside `sm70/`): Apache License 2.0, see [LICENSE](LICENSE).
* `sm70/`: MIT, see [sm70/LICENSE](sm70/LICENSE), except the vendored third-party code:
  CUTLASS in `sm70/fa2_src/cutlass` (BSD-3-Clause, see its `LICENSE.txt`), TurboMind headers in
  `sm70/third_party/sm70_turbomind` (Apache-2.0), and vLLM's Marlin template used by `marlin8_sm80.cu`
  (Apache-2.0, from `csrc/quantization/marlin`).
* Model weights are distributed separately under Apache-2.0 (see the model card).

## Acknowledgements

[vLLM](https://github.com/vllm-project/vllm), [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) by 1Cat AI,
[CUTLASS](https://github.com/NVIDIA/cutlass), [Marlin](https://github.com/IST-DASLab/marlin),
[LMDeploy/TurboMind](https://github.com/InternLM/lmdeploy),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (vendored in vLLM), the
[Qwen](https://huggingface.co/Qwen) team, [Blackfrost](https://huggingface.co/Blackfrost-AI), and
[Cloudflare](https://huggingface.co/Cloudflare/clef) for Clef.
