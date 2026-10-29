"""[ПЛАВНОСТЬ 29.10] Время и состав каждого шага движка -- для A/B на отдельном экземпляре, не для боя.

Выбор: --scheduler-cls fa2sm80_step_timing.StepTimingAsyncScheduler и FA2SM80_STEP_TIMING=<новый JSONL>.
На шаг одна строка (~120 Б): момент прихода результата шага, интервал от прошлого результата (при async scheduling
это и есть длительность шага), всего токенов, число decode-запросов, prefill-токены, токены/число записей Clef
(max_tokens=1), были ли входы энкодера (картинки). Без чтения CUDA-тензоров; файл ограничен 64 МиБ.
"""
import json
import os
import time

from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class StepTimingAsyncScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._шаги = open(os.environ["FA2SM80_STEP_TIMING"], "x", buffering=1)
        self._шаги_байт = 0
        self._прошлый = None

    def update_from_output(self, scheduler_output, model_runner_output):
        t = time.monotonic()
        dt = None if self._прошлый is None else t - self._прошлый
        self._прошлый = t
        dec = pf = clef = nclef = 0
        spec = scheduler_output.scheduled_spec_decode_tokens
        for rid, n in scheduler_output.num_scheduled_tokens.items():
            r = self.requests.get(rid)
            if r is not None and r.max_tokens == 1:
                clef += n
                nclef += 1
            elif n <= 1 + len(spec.get(rid, ())):
                dec += 1
            else:
                pf += n
        rec = {"t": round(t, 4), "dt": None if dt is None else round(dt, 4),
               "tok": scheduler_output.total_num_scheduled_tokens, "dec": dec, "pf": pf,
               "clef": clef, "nclef": nclef, "mm": bool(scheduler_output.scheduled_encoder_inputs)}
        if self._шаги_байт < 64 * 2**20:
            s = json.dumps(rec, separators=(",", ":")) + "\n"
            try:
                self._шаги.write(s)
                self._шаги_байт += len(s)
            except OSError:
                self._шаги_байт = 64 * 2**20
        return super().update_from_output(scheduler_output, model_runner_output)
