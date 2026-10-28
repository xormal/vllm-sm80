# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Contains helpers related to asynchronous code.

This is similar in concept to the `asyncio` module.
"""

import asyncio
import contextlib
from asyncio import FIRST_COMPLETED, AbstractEventLoop, Future, Task
from collections.abc import AsyncGenerator, Awaitable, Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, TypeVar

from transformers.tokenization_utils_base import BatchEncoding
from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")

# [ПАМЯТЬ 27.10] КУСОЧНАЯ ТОКЕНИЗАЦИЯ БОЛЬШИХ СТРОК (FA2SM80_TOK_CHUNK_CHARS, 0 -- выкл.). Токенизатор HF держит на
# каждый токен служебные структуры (смещения и т. п.): ~485 Б/токен в пике, ~200 МБ на 1 МБ текста. Один /tokenize
# (или чат) на десятки МБ раздувал API-сервер боевого sm80 на 7-9 ГБ (cgroup 14 ГБ -> oom-kill воркера по кругу,
# 26.10). Здесь строка длиннее порога режется ТОЛЬКО после "\n" перед непробельным символом -- это всегда граница
# предразбиения Qwen ("\s*[\r\n]+" -- отдельная часть), BPE её не пересекает, спецтокены переводов строк не
# содержат -> номера ПОБИТОВО те же (сверка на 8 МБ корпуса с трудными случаями: 0 расхождений); номера копятся в
# array('i') (4 Б/токен). 16 МБ текста: пик +2.8 ГБ -> ~0, 23.5 -> 17.4 с. Усечение и незнакомые параметры --
# прежним цельным путём.
import json as _json
import os as _os
import re as _re
from array import array as _array

_КУСОК_ПОРОГ = int(_os.environ.get("FA2SM80_TOK_CHUNK_CHARS", "262144"))
_КУСОК = 1 << 20
_РАЗРЕЗ = _re.compile(r"\n(?=\S)")
_КУСОК_МОЖНО = {"add_special_tokens", "return_attention_mask", "return_token_type_ids"}

# [ПАМЯТЬ 27.10] ЗАПАСНОЙ РАЗРЕЗ -- ПЕРЕД ПРОБЕЛОМ ПОСЛЕ БУКВЫ/ЦИФРЫ (строки без "\n" или с очень длинными строками;
# раньше такой текст шёл цельным путём: ~200 МБ на 1 МБ текста). По регулярке предразбиения Qwen ни одна часть не
# продолжается через пробел после непробельного символа (пробел бывает только ПЕРВЫМ символом части), начало части
# от предыдущего текста не зависит (просмотра назад нет), NFC через пробел не склеивает -> граница, номера те же.
# Только если у токенизатора ровно эта регулярка, NFC (или нет нормализатора) и нет lstrip/rstrip у добавленных токенов.
_РАЗРЕЗ_ВПЕРЁД = _re.compile(r"\n(?=\S)|(?<=\w)(?= )")
_РАЗРЕЗ_ПРОБЕЛ = _re.compile(r"(?<=\w)(?= )")
_QWEN_ПРЕДРАЗБИЕНИЕ = {"type": "Sequence", "pretokenizers": [
    {"type": "Split", "behavior": "Isolated", "invert": False, "pattern": {"Regex": (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
        r"|\s*[\r\n]+|\s+(?!\S)|\s+")}},
    {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}]}
_ПРОБЕЛ_МОЖНО = {}


def _пробел_можно(tokenizer):
    ключ = id(tokenizer)
    if ключ not in _ПРОБЕЛ_МОЖНО:
        try:
            b = tokenizer.backend_tokenizer
            норм = _json.loads(b.normalizer.__getstate__()) if b.normalizer is not None else None
            _ПРОБЕЛ_МОЖНО[ключ] = (
                _json.loads(b.pre_tokenizer.__getstate__()) == _QWEN_ПРЕДРАЗБИЕНИЕ
                and норм in (None, {"type": "NFC"})
                and not any(a.lstrip or a.rstrip for a in tokenizer.added_tokens_decoder.values()))
        except Exception:  # noqa: BLE001
            _ПРОБЕЛ_МОЖНО[ключ] = False
    return _ПРОБЕЛ_МОЖНО[ключ]


def _кусочные_id(tokenizer, text, kw, порог=None, пачка=1, стоп=None, кусок=None):
    """Номера токенов большой строки кусками по границам строк -> array('i'); None -- кусками нельзя.
    Куски (~`кусок` символов) подаются пачками по `пачка` (токенизатор Rust раскладывает пачку по ядрам; пик памяти
    ~ пачка*кусок символов в работе). `стоп` (threading.Event): выставлен -- бросить работу (клиент ушёл), вернуть None."""
    порог = _КУСОК_ПОРОГ if порог is None else порог
    кусок = _КУСОК if кусок is None else кусок
    # [ПАМЯТЬ 27.10] truncation=False и max_length=None -- пустые параметры, но их ВСЕГДА шлёт рендерер
    # (TokenizeParams.get_encode_kwargs): с ними кусочный путь не включался ни разу, большие чаты шли цельным
    # (~485 Б/токен; чат 394.6 МБ -> oom-kill 27.10 13:20). Убираем только пустые значения.
    kw = {k: v for k, v in kw.items()
          if not ((k == "truncation" and v is False) or (k == "max_length" and v is None))}
    if (not порог or not isinstance(text, str) or len(text) <= порог
            or not set(kw) <= _КУСОК_МОЖНО):
        return None
    пробел = _пробел_можно(tokenizer)
    if not пробел and "\n" not in text:
        return None
    обр = tokenizer("a\nb", **kw)          # ключи и служебные токены -- как у цельного вызова
    if list(обр.keys()) and any(k not in ("input_ids", "attention_mask", "token_type_ids") for k in обр.keys()):
        return None
    if kw.get("add_special_tokens", True) and tokenizer("a\nb", add_special_tokens=False)["input_ids"] != list(
            обр["input_ids"]):
        return None                          # токенизатор добавляет служебные токены -- кусками не склеить
    ids = _array("i")
    pos, n = 0, len(text)
    куски = []
    while pos < n:
        end = min(n, pos + кусок)
        if end < n:
            m = None
            for m in _РАЗРЕЗ.finditer(text, pos + кусок // 2, end):
                pass
            if m is None and пробел:
                for m in _РАЗРЕЗ_ПРОБЕЛ.finditer(text, pos + кусок // 2, end):
                    pass
            if m is None:
                m = (_РАЗРЕЗ_ВПЕРЁД if пробел else _РАЗРЕЗ).search(text, end)
            end = m.end() if m is not None else n
        куски.append(text[pos:end])
        pos = end
        if len(куски) >= пачка or pos >= n:
            if стоп is not None and стоп.is_set():
                return None
            if len(куски) == 1:
                ids.extend(tokenizer(куски[0], **kw)["input_ids"])
            else:
                for r in tokenizer(куски, **kw)["input_ids"]:
                    ids.extend(r)
            куски = []
    return ids, обр


# [ПАМЯТЬ 27.10] ОГРОМНЫЕ ТЕКСТЫ (> FA2SM80_TOK_STREAM_CHARS символов) -- в ОДНОМ отдельном потоке: общий поток
# токенизатора не стоит минуту, две огромные токенизации не идут разом (память). Им пользуются потоковый /tokenize
# (api_router) и проверка длины огромного промпта (renderers/protocol.py).
_БОЛЬШОЙ_ПОРОГ = int(_os.environ.get("FA2SM80_TOK_STREAM_CHARS", str(4 * 2**20)))
_БОЛЬШОЙ_ИСП = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fa2-tok-big")


async def большой_текст_ids(tokenizer, text, kw, порог=None):
    """array('i') номеров огромного текста (кусками, в отдельном потоке); None -- текст не длиннее `порог`
    (по умолчанию FA2SM80_TOK_STREAM_CHARS) или кусками нельзя. Куски по 128К символов пачками по 8: 30 МБ текста
    за 8.5 с, пик +0.5 ГБ (по 1М пачками по 4: 14 с, +1.2 ГБ). Отмена (клиент ушёл) останавливает токенизацию."""
    import threading

    порог = _БОЛЬШОЙ_ПОРОГ if порог is None else порог
    if not порог or not isinstance(text, str) or len(text) <= порог:
        return None
    стоп = threading.Event()
    try:
        r = await asyncio.get_running_loop().run_in_executor(
            _БОЛЬШОЙ_ИСП, lambda: _кусочные_id(tokenizer, text, kw, порог=1, пачка=8, стоп=стоп, кусок=1 << 17))
    except asyncio.CancelledError:
        стоп.set()
        raise
    return None if r is None else r[0]


def _tok_кусками(tokenizer, text, kw):
    """Токенизация большой строки кусками по границам строк; None -- кусками нельзя (идти цельным путём)."""
    r = _кусочные_id(tokenizer, text, kw)
    if r is None:
        return None
    ids, обр = r
    out = {"input_ids": ids.tolist()}
    if "attention_mask" in обр:
        out["attention_mask"] = [1] * len(ids)
    if "token_type_ids" in обр:
        out["token_type_ids"] = [0] * len(ids)
    return BatchEncoding(out)


def _tok(tokenizer, text, kw):
    r = _tok_кусками(tokenizer, text, kw)
    return r if r is not None else tokenizer(text, **kw)


class AsyncMicrobatchTokenizer:
    """Asynchronous tokenizer with micro-batching.

    Pulls pending encode/decode requests from a queue and batches them
    up to reduce overhead. A single-thread ThreadPoolExecutor is used
    so the event loop stays responsive.
    """

    def __init__(
        self,
        tokenizer,
        max_batch_size: int = 32,
        batch_wait_timeout_s: float = 0.002,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s

        self._loop = asyncio.get_running_loop()
        self._queues: dict[
            tuple,
            asyncio.Queue[tuple[str, dict, Future] | tuple[list[int], Future]],
        ] = {}
        self._batcher_tasks: list[Task] = []

        # Single-thread executor for blocking tokenizer calls.
        self._executor = ThreadPoolExecutor(max_workers=1)

    # === Public async API ===
    async def __call__(self, prompt, **kwargs) -> BatchEncoding:
        result_future: Future = self._loop.create_future()
        key = self._queue_key("encode", kwargs)
        queue = self._get_queue(self._loop, key)
        await queue.put((prompt, kwargs, result_future))
        return await result_future

    async def encode(self, prompt, **kwargs) -> list[int]:
        return (await self(prompt, **kwargs)).input_ids

    async def decode(self, token_ids, **kwargs) -> str:
        result_future: Future = self._loop.create_future()
        key = self._queue_key("decode", kwargs)
        queue = self._get_queue(self._loop, key)
        await queue.put((token_ids, result_future))
        return await result_future

    # === Internal helpers ===
    def _get_queue(
        self, loop: asyncio.AbstractEventLoop, key: tuple
    ) -> asyncio.Queue[tuple[str, dict, Future] | tuple[list[int], Future]]:
        """Get the request queue for the given operation key, creating a new
        queue and batcher task if needed."""
        queue = self._queues.get(key)
        if queue is None:
            self._queues[key] = queue = asyncio.Queue()
            if key[0] == "encode":
                can_batch = key[1] != "other"
                coro = self._batch_encode_loop(queue, can_batch)
            else:
                assert key[0] == "decode", f"Unknown operation type: {key[0]}."
                coro = self._batch_decode_loop(queue)
            self._batcher_tasks.append(loop.create_task(coro))
        return queue

    async def _batch_encode_loop(self, queue: asyncio.Queue, can_batch: bool):
        """Batch incoming encode requests for efficiency."""
        while True:
            prompt, kwargs, result_future = await queue.get()
            prompts = [prompt]
            kwargs_list = [kwargs]
            result_futures = [result_future]
            deadline = self._loop.time() + self.batch_wait_timeout_s

            while len(prompts) < self.max_batch_size:
                timeout = deadline - self._loop.time()
                if timeout <= 0:
                    break
                try:
                    prompt, kwargs, result_future = await asyncio.wait_for(
                        queue.get(), timeout
                    )
                    prompts.append(prompt)
                    result_futures.append(result_future)
                    if not can_batch:
                        kwargs_list.append(kwargs)
                except asyncio.TimeoutError:
                    break

            try:
                # If every request uses identical kwargs we can run a single
                # batched tokenizer call for a big speed-up.
                if can_batch and len(prompts) > 1 and not (
                    _КУСОК_ПОРОГ and any(isinstance(p, str) and len(p) > _КУСОК_ПОРОГ for p in prompts)
                ):
                    batch_encode_fn = partial(self.tokenizer, prompts, **kwargs)
                    results = await self._loop.run_in_executor(
                        self._executor, batch_encode_fn
                    )

                    for i, fut in enumerate(result_futures):
                        if not fut.done():
                            data = {k: v[i] for k, v in results.items()}
                            fut.set_result(BatchEncoding(data))
                else:
                    if can_batch:   # одни и те же kwargs у всех (в kwargs_list лежит только первый)
                        kwargs_list = [kwargs] * len(prompts)
                    encode_fn = lambda prompts=prompts, kwargs=kwargs_list: [
                        _tok(self.tokenizer, p, kw) for p, kw in zip(prompts, kwargs)
                    ]
                    results = await self._loop.run_in_executor(
                        self._executor, encode_fn
                    )

                    for fut, res in zip(result_futures, results):
                        if not fut.done():
                            fut.set_result(res)
            except Exception as e:
                for fut in result_futures:
                    if not fut.done():
                        fut.set_exception(e)

    async def _batch_decode_loop(self, queue: asyncio.Queue):
        """Batch incoming decode requests for efficiency."""
        while True:
            token_ids, result_future = await queue.get()
            token_ids_list = [token_ids]
            result_futures = [result_future]
            deadline = self._loop.time() + self.batch_wait_timeout_s

            while len(token_ids_list) < self.max_batch_size:
                timeout = deadline - self._loop.time()
                if timeout <= 0:
                    break
                try:
                    token_ids, result_future = await asyncio.wait_for(
                        queue.get(), timeout
                    )
                    token_ids_list.append(token_ids)
                    result_futures.append(result_future)
                except asyncio.TimeoutError:
                    break

            try:
                # Perform a single batched decode call for all requests
                results = await self._loop.run_in_executor(
                    self._executor, self.tokenizer.batch_decode, token_ids_list
                )
                for fut, res in zip(result_futures, results):
                    if not fut.done():
                        fut.set_result(res)
            except Exception as e:
                for fut in result_futures:
                    if not fut.done():
                        fut.set_exception(e)

    def _queue_key(self, op: str, kwargs: dict) -> tuple:
        """
        Return a normalized key describing operation + kwargs.

        - `add_special_tokens`: {True/False}
        - `truncation`: {True/False}
          - If `truncation` is False (`max_length` is None),
            returns a key for a can_batch queue.
          - If `truncation` is True and `max_length` is None or equals
            `tokenizer.model_max_length`, returns a key for a can_batch queue.
          - Otherwise, returns a key for a cannot_batch queue.

        Examples:
          - Decode: ("decode",)
          - Encode typical:
            ("encode", add_special_tokens, bool_truncation, max_length_label)
          - Fallback: ("encode", "other")
        """

        if op == "decode":
            return ("decode",)

        add_special_tokens = kwargs.get("add_special_tokens", True)
        truncation = kwargs.get("truncation", False)
        max_length = kwargs.get("max_length")

        if not truncation:
            return "encode", add_special_tokens, False, None

        model_max = getattr(self.tokenizer, "model_max_length", None)
        if max_length is None or (model_max is not None and max_length == model_max):
            return "encode", add_special_tokens, True, "model_max"

        return "encode", "other"

    def __del__(self):
        if (
            (tasks := getattr(self, "_batcher_tasks", None))
            and (loop := getattr(self, "_loop", None))
            and not loop.is_closed()
        ):

            def cancel_tasks():
                for task in tasks:
                    task.cancel()

            loop.call_soon_threadsafe(cancel_tasks)


def cancel_task_threadsafe(task: Task):
    if task and not task.done():
        run_in_loop(task.get_loop(), task.cancel)


def make_async(
    func: Callable[P, T],
    executor: Executor | None = None,
) -> Callable[P, Awaitable[T]]:
    """
    Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.
    """

    def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Future[T]:
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        return loop.run_in_executor(executor=executor, func=p_func)

    return _async_wrapper


def run_in_loop(loop: AbstractEventLoop, function: Callable, *args):
    if in_loop(loop):
        function(*args)
    elif not loop.is_closed():
        loop.call_soon_threadsafe(function, *args)


def in_loop(event_loop: AbstractEventLoop) -> bool:
    try:
        return asyncio.get_running_loop() == event_loop
    except RuntimeError:
        return False


# A hack to pass mypy
if TYPE_CHECKING:

    def anext(it: AsyncGenerator[T, None]):
        return it.__anext__()


async def merge_async_iterators(
    *iterators: AsyncGenerator[T, None],
) -> AsyncGenerator[tuple[int, T], None]:
    """Merge multiple asynchronous iterators into a single iterator.

    This method handle the case where some iterators finish before others.
    When it yields, it yields a tuple (i, item) where i is the index of the
    iterator that yields the item.
    """
    if len(iterators) == 1:
        # Fast-path single iterator case.
        async for item in iterators[0]:
            yield 0, item
        return

    loop = asyncio.get_running_loop()

    awaits = {loop.create_task(anext(it)): (i, it) for i, it in enumerate(iterators)}
    try:
        while awaits:
            done, _ = await asyncio.wait(awaits.keys(), return_when=FIRST_COMPLETED)
            for d in done:
                pair = awaits.pop(d)
                try:
                    item = await d
                    i, it = pair
                    awaits[loop.create_task(anext(it))] = pair
                    yield i, item
                except StopAsyncIteration:
                    pass
    finally:
        # Cancel any remaining iterators
        for f, (_, it) in awaits.items():
            with contextlib.suppress(BaseException):
                f.cancel()
                await it.aclose()


async def collect_from_async_generator(iterator: AsyncGenerator[T, None]) -> list[T]:
    """Collect all items from an async generator into a list."""
    items = []
    async for item in iterator:
        items.append(item)
    return items
