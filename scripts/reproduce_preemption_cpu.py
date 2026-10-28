#!/usr/bin/env python3
"""CPU-only scheduler stress with synthetic outputs; no model weights or kernels.

This isolates page admission/recompute behavior. It does not model GDN, MTP
acceptance, async worker state, numerical quality, or real inference throughput.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Run with CUDA_VISIBLE_DEVICES='' for CPU-only reproduction")

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.outputs import ModelRunnerOutput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lookahead", type=int, choices=(0,4), default=0,
                        help="Allocation reserve only; does not simulate MTP outputs")
    parser.add_argument("--admission-probe", action="store_true",
                        help="CPU-only experimental admission check, not a runtime fix")
    parser.add_argument("--runtime", action="store_true", help="Test the actual candidate admission fix")
    parser.add_argument("--expect-recovery", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000:
        raise ValueError("CPU reproduction is bounded to 1..10000 scheduler steps")
    if args.runtime and args.admission_probe:
        raise ValueError("Use the real runtime or the experimental probe, not both")
    os.environ["FA2SM80_RESUME_ADMISSION"] = "1" if args.runtime else "0"
    with tempfile.TemporaryDirectory(prefix="kandidat-cpu-scheduler-") as temporary:
        (Path(temporary) / "config.json").write_text(json.dumps({
            "model_type":"opt", "architectures":["OPTForCausalLM"],
            "hidden_size":8,"ffn_dim":16,"num_hidden_layers":1,"num_attention_heads":1,
            "vocab_size":100,"max_position_embeddings":262144,
            "bos_token_id":0,"eos_token_id":99,"word_embed_proj_dim":8}))
        scheduler = create_scheduler(model=temporary, max_num_seqs=16,
            max_num_batched_tokens=8192, num_blocks=32, block_size=16384,
            max_model_len=262144, skip_tokenizer_init=True, async_scheduling=False)
        scheduler.num_lookahead_tokens = args.lookahead
        if args.admission_probe:
            allocate = scheduler.kv_cache_manager.allocate_slots
            def probe(request, num_new_tokens, **kwargs):
                if request.num_preemptions and request.num_computed_tokens == 0:
                    manager = scheduler.kv_cache_manager
                    cached = kwargs.get("new_computed_blocks") or manager.empty_kv_cache_blocks
                    target = min(request.num_tokens + 1, scheduler.max_model_len)
                    slots = min(target + scheduler.num_lookahead_tokens, scheduler.max_model_len)
                    manager.coordinator.get_num_blocks_to_allocate(
                        request_id=request.request_id, num_tokens=slots,
                        new_computed_blocks=cached.blocks, num_encoder_tokens=0,
                        total_computed_tokens=kwargs.get("num_new_computed_tokens",0),
                        num_tokens_main_model=target)
                    if not manager.coordinator.has_free_blocks_for():
                        return None
                return allocate(request,num_new_tokens,**kwargs)
            scheduler.kv_cache_manager.allocate_slots = probe
        requests = create_requests(8, num_tokens=49147, max_tokens=2048, block_size=16384)
        for request in requests:
            scheduler.add_request(request)
        recompute_tokens = 0
        preemptions = 0
        first_output = {}
        last_output = {}
        for step in range(args.steps):
            output = scheduler.schedule()
            preemptions += len(output.preempted_req_ids)
            ids = list(output.num_scheduled_tokens)
            generated = []
            for rid in ids:
                request = scheduler.requests[rid]
                emitted = [] if request.is_prefill_chunk else [42]
                generated.append(emitted)
                if request.num_preemptions:
                    recompute_tokens += max(0, output.num_scheduled_tokens[rid]-len(emitted))
                if emitted:
                    first_output.setdefault(rid,step)
                    last_output[rid] = step
            scheduler.update_from_output(output, ModelRunnerOutput(
                req_ids=ids, req_id_to_index={rid:i for i,rid in enumerate(ids)},
                sampled_token_ids=generated, logprobs=None, prompt_logprobs_dict={}, pooler_output=[]))
            if all(r.is_finished() for r in requests):
                break
        result = {"steps":step+1,"admission_probe":args.admission_probe,
            "runtime_admission":args.runtime,
            "allocation_lookahead":args.lookahead,
            "all_finished":all(r.is_finished() for r in requests),
            "scheduler_preemptions":preemptions,
            "scheduled_recompute_tokens_after_preemption":recompute_tokens,
            "requests":[{"id":r.request_id,"output_tokens":r.num_output_tokens,
                "preemptions":r.num_preemptions,"first_output_step":first_output.get(r.request_id),
                "last_output_step":last_output.get(r.request_id)} for r in requests]}
        if args.output: args.output.write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
        if args.expect_recovery:
            assert result["all_finished"]
            assert preemptions <= 1, "Repeated preemption returned"
            assert recompute_tokens < 2 * 49147, "Excessive recomputation returned"
            assert all(r.num_output_tokens == 2048 for r in requests)


if __name__ == "__main__":
    main()
