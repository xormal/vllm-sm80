# vLLM-SM80: load tests and metrics of the running configuration

All numbers are from A/B instances in maintenance windows on the production pair (2× A100 PG199 32 GB, TP2, GPU0–1),
model **Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef**, context 262,144, int8 KV, MTP k=4 with n-gram assist, Clef on the
same weights. Raw results are in [`results/`](results/) (keys are Russian, see the glossary at the end); the full
journal of every window is in `docs/research/2026-10-28-qwen38/SMOOTHNESS_WINDOW20_25_2026-10-30.md`.

## Configuration

The running server is `scripts/serve_candidate_live.sh` → `scripts/serve_qwen38_sm80.sh` (lever files
`configs/qwen38-27b-int8-sm80.env` over `configs/qwen38-27b-int8-clef.env`; variables set by the caller win). On
top of the sm80 config it exports:

```bash
export FA2SM70_CLEF=1 FA2SM80_SEL_DENSE_NB=1024 FA2SM70_EXCH12=1
export FA2SM80_RESUME_ADMISSION=1 FA2SM70_GRAN_CONV_ALL=1
export FA2SM80_BLOCKPIPE_FALLBACK_ROWS=0 FA2SM80_CONV_META_FUSED=0
export FA2SM80_PREFILL_BUDGET_DECODING=1024 FA2SM80_PREFILL_BUDGET_BACKLOG=0   # prefill tokens per step while decoding
export FA2SM70_CLEF_HEAD_ASYNC=1 FA2SM70_NOSYNC_ACC2=1 FA2SM70_MEM_TRIM_MODE=off
export FA2SM80_DECODE_INTERLEAVE=4                     # 4 pure-decode steps after a step with a prefill chunk
export FA2SM80_W8_MARLIN=1                             # Marlin8 W8 GEMM (M <= 1024), dequant + cuBLAS above
export FA2SM80_BLOCKPIPE_BUFS=1 FA2SM80_BLOCKPIPE_M8_MIN=256
export FA2SM70_GDN_IDX_VIEW=1 FA2SM70_GDN_NCOMP_CPU=1 FA2SM70_CONV_META_PINNED=1
export FA2SM80_GDN_DEVMETA=1 FA2SM80_GDN_ALL_VARLEN=1 FA2SM80_ATTN_VARLEN=1 FA2SM80_MIX_DEC_UNIFORM=1
export FA2SM80_M8DQ=42                                 # marlin8_dequant v4, 256 k per block
export FA2SM80_CG_SIZES=1,2,4,8,12,16,20,24,28,32,36,40,45   # -> FULL decode graphs for 1..9 requests at k=4
export FA2SM80_MTP_LOOP_GRAPH_MAXB=9
```

`DRYRUN=1 scripts/serve_candidate_live.sh` prints the resolved environment and the vLLM command without starting
anything.

## Method

* **`scripts/bench_conc.py URL OUT LABEL`** (env `CONC`, `PARTS`, `COMBOS`, `SEED=21`, `CLEF_SEC=45`):
  * *decode*: c streams of `/v1/completions`, each with its own 2048-token prompt of real text (seeded corpus slices,
    no prefix-cache hits), 512 output tokens, `ignore_eos`, T=0, streamed with strict SSE checks (error events,
    missing `[DONE]` or a short answer fail the scenario). Steady decode = window from the last first token to the
    first last token; per-stream rate, total rate, TTFT, inter-chunk gaps p50/p99/max (also over full answers);
  * *prefill*: c simultaneous 8192-token prompts, 1 output token: tokens / wall time and TTFT;
  * *Clef*: c clients sending `/v1/systemone` records back to back (≈2000 tokens of text, one choice question) for
    45 s: input tokens/s, records/s, latency p50/p95. Clef has no decode phase;
  * *mix*: decode on c streams while c Clef clients run;
  * *combinations* `COMBOS="L:C,..."`: L LLM streams with C Clef clients, measured twice — decode and prefill —
    with Clef throughput and latency over the same interval. 3 Clef slots: Clef saturates at 4 clients, so C=8 is
    "maximum Clef". At most 9 LLM requests run at once (GDN state pool at k=4); 16 means a queue.
* **Accuracy**, every window: Clef against the V100 reference decisions (`sm70/integrations/vllm_sm70/clef/proverka_stenda.py
  clef` + `clefsr`, 35 decisions, max |Δp|; the reference file is from the V100 production acceptance and is not part
  of this repository), Clef and an LLM fingerprint against the previous sm80 base (`quality_ref.py` from the work
  directory, not part of this repository: 160 greedy tokens at prompt lengths 600 / 2048 / 5000, first divergence
  and |Δlogp|).
* **Step timing**: `FA2SM80_STEP_TIMING=<file>` writes one JSON line per scheduler step (tokens, decode rows, prefill
  and Clef tokens, duration).
* Windows are scripted end to end (`scripts/window26_fixes.sh` … `window29_graphs.sh`): stop the live service, start
  an A/B instance on 127.0.0.1:18493 with the profile + levers, run accuracy and load, restore or deploy, 5-minute
  check of the live service.

Reproduce a run against a running server:

```bash
python scripts/bench_conc.py http://127.0.0.1:8085 out.json mylabel                      # decode, prefill, Clef, mix at c=1/4/8
CONC=1 PARTS=combo COMBOS=4:1,1:8,4:8,8:8,16:8 python scripts/bench_conc.py http://127.0.0.1:8085 combos.json mylabel
```

## Production under live Clef traffic (2026-11-02)

The production server itself, not the stand: the window 29b configuration plus the memory fixes of windows 30–31
(MTP head MLP in row chunks, MTP draft-graph caps), live since 2026-11-01 19:41. Clef runs continuously — 51,246
records in the first 7 h, 1,232 input tokens on average, about 2 records/s — so the server is almost never idle and
these numbers are taken on that live background, not in silence.

Method: [`scripts/live_clef_traffic_probe.py`](../../scripts/live_clef_traffic_probe.py) sends one request at a time
(the load of one client), each after the queue is empty, and records the background during every measurement from
`/metrics`: other requests finished and other prompt tokens per second. The latter includes prefix-cache hits, so it
is an upper bound of the background prefill. Raw results:
[`results/live-20261102-clef-traffic.json`](results/live-20261102-clef-traffic.json).

#### Clef latency by record size (3 rounds, 02:5x)

| record | input tokens | latency median / max s | measurements | background in-tok/s (median) |
|---|---|---|---|---|
| short (two test records) | 300–319 | 0.38 / 0.54 | 6 | 1,588 |
| text, 1,500 chars, 3 questions | 839 | 0.83 / 1.05 | 3 | 2,715 |
| text, 4,000 chars, 3 questions | 1,621 | 0.56 / 0.89 | 3 | 1,010 |
| text, 8,000 chars, 3 questions | 3,496 | 1.48 / 1.79 | 3 | 1,961 |
| 12 questions × 6 options | 3,692 | 0.90 / 1.71 | 3 | 878 |
| text, 30,000 chars, 3 questions | 12,883 | 2.96 / 3.56 | 3 | 506 |

Records up to 1,621 tokens: 11 of 12 under a second, median 0.50 s. Since 2026-11-01 19:41, 5 of 52,391
`/v1/systemone` requests took longer than 60 s (67–73 s, the server logs every request above 60 s); 0 errors 5xx.

#### One chat, 512 tokens, temperature 0 (6 runs)

| decode tok/s | tokens per chunk | gap p50 / p99 ms | TTFT s | background in-tok/s | other requests per run |
|---|---|---|---|---|---|
| 44.8–49.4 (median 48.3) | 3.12–3.18 | 22.2–22.6 / 330–358 | 0.30–0.58 | 1,702–2,160 | 17–25 |

The gap p50 is the pure decode step: 22.4 ms, the same as window 29b (22.2 ms for one request), so the memory fixes
of windows 30–31 did not change it. The p99 gaps are steps that carry a Clef prefill chunk (budget 1024); with
3.18 tokens per step and no Clef the same chat would run at ~142 tok/s.

#### Prefill of a unique prompt (no prefix-cache hits), `max_tokens` 1

| prompt | tok/s per run | median | background in-tok/s | other requests per run |
|---|---|---|---|---|
| 8,192 | 3,819 / 5,480 / 4,458 / 3,982 | 4,220 | 1,319–1,998 | 3–9 |
| 32,768 | 4,270 / 4,260 / 3,528 | 4,260 | 371–1,061 | 4–7 |

An earlier run at 00:53 (same prompts) measured 32K at 5,040–5,110 tok/s with 3–6 Clef records finishing alongside
(background tokens not recorded).

## Results — running configuration (window 29b, 2026-10-31)

Accuracy: Clef vs V100 **34/35** (max |Δp| 0.0251, same as the sm80 base), vs the sm80 base 35/35 (0.0022); LLM
fingerprint **160/160** at 600 / 2048 / 5000 tokens. FULL decode graphs 9/9, PIECEWISE 9/9.

Pure-decode step by number of requests (ms): 1: 22.2 · 2: 26.0 · 3: 28.3 · 4: 31.6 · 5: 34.3 · 6: 38.9 · 7: 44.2 ·
8: 45.8 · **9: 47.5** (81.6 before the 45-token graph, eager).

#### Decode only (no Clef), 2048-token prompts, 512 output tokens

| streams | decode tok/s per stream | decode tok/s total | TTFT s (median) | gap p50 s | tokens per step |
|---|---|---|---|---|---|
| 8 | 57.1 | 515.8 | 3.486 | 0.046 | 2.94 |
| 9 | 53.4 | 548.7 | 2.307 | 0.047 | 2.9 |
| 16 | 51.9 | — | 7.12 | — | 3.08 |

#### LLM × saturated Clef (8 Clef streams), prefill budget 1024

| LLM:Clef | decode tok/s per stream | gap p99 / max s | TTFT 2K s | Clef in-tok/s (decode run) | LLM prefill tok/s (8K prompts) | TTFT 8K s | Clef in-tok/s (prefill run) |
|---|---|---|---|---|---|---|---|
| 9:8 | 33.0 | 0.375 / 0.377 | 3.746 | 2167.0 | 4532.0 | 9.981 | 1699.0 |
| 16:8 | 39.8 | — / — | 6.913 | 1694.0 | 4836.0 | 15.317 | 1095.0 |

## Results — same paths before the dequant v4 and the 9-request graphs (window 27)

Window 27 ran the same E paths with the old Marlin dequant (v2, 89.6 ms per step) and decode graphs up to 8
requests, so prefill-heavy and 9+-request numbers below are lower bounds for the running configuration.

#### LLM alone

| streams | prefill tok/s (8K prompts) | TTFT 8K s (median) | decode tok/s per stream | decode tok/s total | TTFT 2K s |
|---|---|---|---|---|---|
| 1 | 4845.0 | 1.69 | 113.6 | 113.6 | 0.584 |
| 4 | 5222.0 | 3.976 | 91.8 | 352.2 | 1.684 |
| 8 | 5252.0 | 7.065 | 60.3 | 550.1 | 2.734 |
| 16 | 5256.0 | 13.293 | 40.9 | — | 5.53 |

#### Clef alone (records of 2000 tokens, one choice question; Clef is prefill-only)

| streams | input tok/s | records/s | latency p50 / p95 s | errors |
|---|---|---|---|---|
| 1 | 3080.0 | 1.54 | 0.65 / 0.65 | 0 |
| 4 | 4090.0 | 2.05 | 1.95 / 1.97 | 0 |
| 8 | 4093.0 | 2.05 | 3.91 / 3.92 | 0 |

#### LLM × Clef combinations, prefill budget 2048

| LLM:Clef | decode tok/s per stream | gap p99 / max s | Clef in-tok/s (decode run) | LLM prefill tok/s | TTFT 8K s | Clef in-tok/s (prefill run) |
|---|---|---|---|---|---|---|
| 4:1 | 21.0 | 0.611 / 0.616 | 2638.0 | 5139.0 | 4.127 | 1198.0 |
| 1:8 | 19.7 | 0.642 / 0.647 | 3148.0 | 2680.0 | 3.056 | 3428.0 |
| 4:8 | 20.0 | 0.647 / 0.648 | 2936.0 | 3702.0 | 6.413 | 2429.0 |
| 8:8 | 18.0 | 0.657 / 0.663 | 2692.0 | 4265.0 | 9.702 | 1759.0 |
| 16:8 | 27.2 | — / — | 1921.0 | 4689.0 | 15.877 | 1156.0 |

#### LLM × Clef combinations, prefill budget 1024

| LLM:Clef | decode tok/s per stream | gap p99 / max s | Clef in-tok/s (decode run) | LLM prefill tok/s | TTFT 8K s | Clef in-tok/s (prefill run) |
|---|---|---|---|---|---|---|
| 4:1 | 35.3 | 0.338 / 0.358 | 2317.0 | 5025.0 | 4.337 | 1385.0 |
| 1:8 | 33.1 | 0.358 / 0.359 | 2990.0 | 2818.0 | 2.905 | 3528.0 |
| 4:8 | 31.6 | 0.35 / 0.362 | 2700.0 | 3660.0 | 6.408 | 2504.0 |
| 8:8 | 24.7 | 0.445 / 0.47 | 2035.0 | 4218.0 | 9.78 | 1795.0 |
| 16:8 | 29.4 | — / — | 1309.0 | 4669.0 | 15.945 | 1153.0 |

Under Clef the GPU is shared: LLM prefill + Clef ≈ 5.9–6.1 K tok/s together; the budget only applies to steps with
decode, so it barely changes prefill but makes chats 1.4–1.7× faster.

## Before / after (windows 24b and 26)

#### Chats under Clef: c chats + c Clef streams

| c | base (sm80 2026-10-30, budget 2048) | E, budget 2048 | E, budget 1024 |
|---|---|---|---|
| 1 | 21.1 tok/s, p99 0.623 s, Clef 2795.0 | 23.6 tok/s, p99 0.619 s, Clef 2761.0 | — |
| 4 | 20.2 tok/s, p99 0.578 s, Clef 2916.0 | 19.6 tok/s, p99 0.647 s, Clef 2917.0 | 33.2 tok/s, p99 0.385 s, Clef 2563.0 |
| 8 | 17.4 tok/s, p99 0.598 s, Clef 2628.0 | 17.6 tok/s, p99 0.66 s, Clef 2653.0 | 27.4 tok/s, p99 0.358 s, Clef 2205.0 |

`w28-dequant.json` -- ms per step (all weights of one rank): 2: 89.56, 41: 34.01, 42: 33.98, наш: 49.14, копия: 36.92; bitwise: {'2': True, '41': True, '42': True}

`w29-dequant-roofline.json` -- ms per step (all weights of one rank): 2: 89.38, 41: 33.96, 42: 33.96, наш: 49.2, копия: 36.96, i8_в_fp16: 68.28; bitwise: {'2': True, '41': True, '42': True}

Pure decode, tok/s per stream at 1 / 4 / 8 streams: base 136 / 86 / 55, E 134 / 92 / 60 (step 34 / 52 → 32 / 46 ms
at 4 / 8). Prefill 8K: 5166 / 5249 / 5252 → 5201 / 5250 / 5256 tok/s. Allocator retries on GPU0: 481 → 30.

## Marlin dequant (windows 28 and 29)

`w28-dequant.json` -- ms per step (all weights of one rank): 2: 89.56, 41: 34.01, 42: 33.98, наш: 49.14, копия: 36.92; bitwise: {'2': True, '41': True, '42': True}
`w29-dequant-roofline.json` -- ms per step (all weights of one rank): 2: 89.38, 41: 33.96, 42: 33.96, наш: 49.2, копия: 36.96, i8_в_fp16: 68.28; bitwise: {'2': True, '41': True, '42': True}

`2` = v2, `41`/`42` = v4 with 128/256 k per block, `наш` = `w8a16_sm80_dequant` (our layout), `копия` = fp16 copy of the
same size (1:1 read/write, 1270 GB/s), `i8_в_fp16` = torch int8→fp16 cast (not a valid ceiling: 2× slower than v4).
The ceiling for the 1:2 traffic shape is measured by `sm70/fa2_sm70/csrc/m8dq_roofline.cu`.

## Glossary of result keys

| key | meaning |
|---|---|
| `llm_генерация` / `llm_префилл` / `clef` / `смесь` / `сочетания` | decode / prefill / Clef / mix / combinations |
| `декод_на_поток_ток_с`, `декод_всего_ток_с` | decode tok/s per stream, total |
| `ttft_с`, `ttft_с_медиана` | time to first token, s (median) |
| `разрыв_p50_с` / `_p99_с` / `_max_с`, `полн_разрыв_*` | inter-chunk gap in the steady window / over full answers, s |
| `токенов_на_шаг` | accepted tokens per MTP step |
| `запрос_с_медиана`, `запрос_с_max` | full request time, s |
| `префилл_ток_с`, `стена_с` | prefill tok/s, wall time, s |
| `вход_ток_с`, `записей_с`, `задержка_p50_с`, `задержка_p95_с`, `ошибок` | Clef input tok/s, records/s, latency, errors |
| `clef_*` (inside decode/prefill runs) | Clef metrics over the same interval |
| `шаг_мс`, `побитно`, `лучший` | dequant ms per step, bitwise vs ours, best variant |
