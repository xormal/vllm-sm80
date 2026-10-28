"""Opt-in CPU metadata trace; never selected by the normal candidate launcher.

Select with --scheduler-cls fa2sm80_scheduler_trace.TraceAsyncScheduler and set
CANDIDATE_SCHEDULER_TRACE to a new JSONL path. This changes timing: results from
this process must not be used for performance acceptance. No CUDA tensor reads.
"""
import json
import os
import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class TraceAsyncScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.scheduler_config.async_scheduling:
            raise ValueError("Diagnostic subclass requires the existing async scheduler")
        self._trace = open(os.environ["CANDIDATE_SCHEDULER_TRACE"], "x", buffering=1)
        self._trace_bytes = 0
        self._trace_step = 0
        self._emit("init", max_tokens=self.max_num_scheduled_tokens,
                   lookahead=self.num_lookahead_tokens)

    def _emit(self, event, **fields):
        # Bound diagnostic disk usage; fail open on an I/O failure after startup.
        if self._trace_bytes >= 64 * 1024 * 1024:
            return
        record = json.dumps(dict(event=event, step=self._trace_step,
                                 t=time.monotonic(), **fields), separators=(",", ":")) + "\n"
        try:
            self._trace.write(record)
            self._trace_bytes += len(record.encode())
        except OSError:
            self._trace_bytes = 64 * 1024 * 1024

    @staticmethod
    def _request(request):
        return dict(id=request.request_id, status=request.status.name,
                    prompt=request.num_prompt_tokens, output=request.num_output_tokens,
                    computed=request.num_computed_tokens,
                    placeholders=request.num_output_placeholders,
                    spec=len(request.spec_token_ids), prefill=request.is_prefill_chunk,
                    preemptions=request.num_preemptions)

    def _snapshot(self):
        coordinator = self.kv_cache_manager.coordinator
        return dict(requests=[self._request(r) for r in self.requests.values()],
                    running=[r.request_id for r in self.running],
                    waiting=[r.request_id for r in self.waiting],
                    pools=[dict(total=p.num_gpu_blocks, free=p.get_num_free_blocks())
                           for p in coordinator.block_pools])

    def schedule(self):
        self._trace_step += 1
        before = self._snapshot()
        output = super().schedule()
        self._emit("schedule", before=before, after=self._snapshot(),
                   scheduled=output.num_scheduled_tokens,
                   spec={k: len(v) for k, v in output.scheduled_spec_decode_tokens.items()},
                   resumed=sorted(output.scheduled_cached_reqs.resumed_req_ids),
                   preempted=sorted(output.preempted_req_ids))
        return output

    def _preempt_request(self, request, timestamp):
        before = self._request(request)
        result = super()._preempt_request(request, timestamp)
        self._emit("preempt", before=before, after=self._request(request))
        return result

    def update_from_output(self, scheduler_output, model_runner_output):
        before = self._snapshot()
        # This public runner result contains CPU lists, not device tensors.
        sampled = model_runner_output.sampled_token_ids or []
        counts = {rid: len(sampled[i]) if i < len(sampled) else None
                  for rid, i in model_runner_output.req_id_to_index.items()}
        result = super().update_from_output(scheduler_output, model_runner_output)
        self._emit("update", before=before, after=self._snapshot(),
                   scheduled=scheduler_output.num_scheduled_tokens, generated=counts)
        return result
