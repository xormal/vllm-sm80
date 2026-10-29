# SPDX-License-Identifier: Apache-2.0
# [ПЛАВНОСТЬ 29.10] Бюджет prefill на шаг при декоде (FA2SM80_PREFILL_BUDGET_DECODING). CPU-тест контракта:
# выключено -- прежнее поведение; включено -- prefill-куски шага не больше бюджета, пока есть декод; decode-токены
# не урезаются; без декода -- полный бюджет; длинный prefill доходит до конца (нет голодания).
import os

import pytest

import vllm.v1.core.sched.scheduler as sched_mod
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

MODEL = os.environ.get("FA2SM80_TEST_MODEL",
                       "/home/alex/devpack_sm70_20261022/model/Qwen3.8-27B-ABLITERATED-GPTQ-Int8-Clef")


def _scheduler(async_scheduling=False):
    return create_scheduler(model=MODEL, max_num_batched_tokens=8192, max_model_len=65536,
                            num_blocks=20000, block_size=16, async_scheduling=async_scheduling)


def _step(scheduler, sampled_for=None):
    """Один шаг: schedule + выход модели (по токену тем, кто дошёл до конца prompt)."""
    out = scheduler.schedule()
    ids = list(out.num_scheduled_tokens)
    reqs = {r.request_id: r for r in scheduler.running}
    sampled = []
    for rid in ids:
        r = reqs.get(rid)
        # schedule() уже сдвинул num_computed_tokens на запланированное
        done_prompt = r is not None and r.num_computed_tokens >= r.num_prompt_tokens
        sampled.append([7] if done_prompt else [])
    mro = ModelRunnerOutput(req_ids=ids, req_id_to_index={rid: i for i, rid in enumerate(ids)},
                            sampled_token_ids=sampled, logprobs=None, prompt_logprobs_dict={}, pooler_output=[])
    scheduler.update_from_output(out, mro)
    return out


def _decoding(scheduler, n):
    """n запросов с коротким prompt, доведённых до декода."""
    reqs = create_requests(n, num_tokens=32, max_tokens=4096, req_ids=[f"chat{i}" for i in range(n)])
    for r in reqs:
        scheduler.add_request(r)
    _step(scheduler)
    assert all(not sched_mod._в_префилле(r) for r in reqs)
    return reqs


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("budget", [0, 512, 2048])
def test_prefill_capped_only_while_decoding(monkeypatch, budget, async_scheduling):
    monkeypatch.setattr(sched_mod, "_PREFILL_BUDGET_DECODING", budget)
    s = _scheduler(async_scheduling)
    chats = _decoding(s, 8)
    long = create_requests(1, num_tokens=30000, max_tokens=1, req_ids=["clef0"])[0]
    s.add_request(long)
    out = _step(s)
    for c in chats:
        assert out.num_scheduled_tokens[c.request_id] == 1      # decode не урезан
    chunk = out.num_scheduled_tokens["clef0"]
    if budget:
        assert chunk == budget
    else:
        assert chunk == 8192 - 8                                 # прежнее: весь остаток бюджета шага


def test_full_budget_without_decode(monkeypatch):
    monkeypatch.setattr(sched_mod, "_PREFILL_BUDGET_DECODING", 512)
    s = _scheduler()
    long = create_requests(1, num_tokens=30000, max_tokens=1, req_ids=["clef0"])[0]
    s.add_request(long)
    out = _step(s)
    assert out.num_scheduled_tokens["clef0"] == 8192


def test_budget_shared_by_several_prefills(monkeypatch):
    monkeypatch.setattr(sched_mod, "_PREFILL_BUDGET_DECODING", 1000)
    s = _scheduler()
    _decoding(s, 4)
    for r in create_requests(3, num_tokens=5000, max_tokens=1, req_ids=["a", "b", "c"]):
        s.add_request(r)
    out = _step(s)
    pref = {k: v for k, v in out.num_scheduled_tokens.items() if k in ("a", "b", "c")}
    assert sum(pref.values()) == 1000 and pref.get("a") == 1000   # FCFS: весь бюджет первому
    out = _step(s)                                                  # «a» уже в running, остальные ждут
    pref = {k: v for k, v in out.num_scheduled_tokens.items() if k in ("a", "b", "c")}
    assert sum(pref.values()) == 1000


def test_long_prefill_finishes_while_decoding(monkeypatch):
    monkeypatch.setattr(sched_mod, "_PREFILL_BUDGET_DECODING", 2048)
    s = _scheduler()
    chats = _decoding(s, 8)
    long = create_requests(1, num_tokens=34816, max_tokens=1, req_ids=["clef0"])[0]
    s.add_request(long)
    steps = 0
    while long.request_id in {r.request_id for r in s.running} or long in list(s.waiting):
        out = _step(s)
        steps += 1
        assert out.num_scheduled_tokens.get("clef0", 0) <= 2048
        for c in chats:
            assert out.num_scheduled_tokens.get(c.request_id) == 1
        assert steps < 100
    assert steps == -(-34816 // 2048)                                # 17 шагов, прогресс каждый шаг
    assert long.is_finished()


def test_budget_file_overrides_env(monkeypatch, tmp_path):
    f = tmp_path / "budget"
    f.write_text("1024")
    monkeypatch.setattr(sched_mod, "_PREFILL_BUDGET_DECODING", 0)
    monkeypatch.setattr(sched_mod, "_PREFILL_BUDGET_FILE", str(f))
    monkeypatch.setattr(sched_mod, "_ФАЙЛ_БЮДЖЕТА", {"t": 0.0, "v": None})
    s = _scheduler()
    _decoding(s, 2)
    s.add_request(create_requests(1, num_tokens=30000, max_tokens=1, req_ids=["clef0"])[0])
    assert _step(s).num_scheduled_tokens["clef0"] == 1024
    f.write_text("oops")                                             # битый файл -> значение окружения (0 = выкл.)
    sched_mod._ФАЙЛ_БЮДЖЕТА["t"] = 0.0
    assert _step(s).num_scheduled_tokens["clef0"] == 8192 - 2


def test_async_final_clef_chunk_in_flight_is_not_decode(monkeypatch):
    """Async scheduling: у записи Clef последний кусок в полёте (placeholder) -- это не декод, бюджет не включается."""
    monkeypatch.setattr(sched_mod, "_PREFILL_BUDGET_DECODING", 512)
    s = create_scheduler(model=MODEL, max_num_batched_tokens=8192, max_model_len=65536,
                         num_blocks=20000, block_size=16, async_scheduling=True)
    a, b = create_requests(2, num_tokens=8000, max_tokens=1, req_ids=["clefA", "clefB"])
    s.add_request(a)
    s.add_request(b)
    out1 = s.schedule()                                  # A целиком (последний кусок), B -- остаток бюджета
    assert out1.num_scheduled_tokens["clefA"] == 8000
    assert a.num_output_placeholders == 1                # A в полёте
    out2 = s.schedule()                                  # следующий шаг до результата первого
    assert "clefA" not in out2.num_scheduled_tokens
    assert out2.num_scheduled_tokens["clefB"] == 8000 - 192   # полный остаток, а не 512


def test_adaptive_budget_only_with_backlog(monkeypatch):
    """Порог несделанного prefill: малая очередь -- полный кусок, большая -- бюджет."""
    monkeypatch.setattr(sched_mod, "_PREFILL_BUDGET_DECODING", 2048)
    monkeypatch.setattr(sched_mod, "_PREFILL_BUDGET_BACKLOG", 12000)
    s = _scheduler()
    _decoding(s, 4)
    s.add_request(create_requests(1, num_tokens=6000, max_tokens=1, req_ids=["small"])[0])
    out = _step(s)
    assert out.num_scheduled_tokens["small"] == 6000                 # 6000 < 12000: без бюджета
    for r in create_requests(3, num_tokens=8000, max_tokens=1, req_ids=["b1", "b2", "b3"]):
        s.add_request(r)
    out = _step(s)                                                   # несделано 24000 >= 12000: бюджет
    assert sum(v for k, v in out.num_scheduled_tokens.items() if k.startswith("b")) == 2048


def test_budget_file_with_threshold(monkeypatch, tmp_path):
    f = tmp_path / "budget"
    f.write_text("1024:50000")
    monkeypatch.setattr(sched_mod, "_PREFILL_BUDGET_FILE", str(f))
    monkeypatch.setattr(sched_mod, "_ФАЙЛ_БЮДЖЕТА", {"t": 0.0, "v": None})
    s = _scheduler()
    _decoding(s, 2)
    s.add_request(create_requests(1, num_tokens=30000, max_tokens=1, req_ids=["clef0"])[0])
    assert _step(s).num_scheduled_tokens["clef0"] == 8192 - 2       # 30000 < 50000: без бюджета
    f.write_text("1024:20000")
    sched_mod._ФАЙЛ_БЮДЖЕТА["t"] = 0.0
    assert _step(s).num_scheduled_tokens["clef0"] == 1024           # осталось 21810 >= 20000: бюджет
