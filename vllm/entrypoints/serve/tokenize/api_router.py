# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from typing_extensions import assert_never

from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
)
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.serve.tokenize.protocol import (
    DetokenizeRequest,
    DetokenizeResponse,
    TokenizeCompletionRequest,
    TokenizeRequest,
    TokenizeResponse,
)
from vllm.entrypoints.serve.tokenize.serving import OpenAIServingTokenization
from vllm.entrypoints.utils import (
    with_cancellation,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def tokenization(request: Request) -> OpenAIServingTokenization:
    return request.app.state.openai_serving_tokenization


router = APIRouter()

import asyncio as _aio_tok
import json as _json_tok
import os as _os_tok
import threading as _thr_tok
import time as _time_tok

_ПОТОК_ПОРОГ = int(_os_tok.environ.get("FA2SM80_TOK_STREAM_CHARS", str(4 * 2**20)))


def _вернуть_память():
    # [ПАМЯТЬ 27.10] После огромного /tokenize glibc держит освобождённое у себя (API 1.2 -> 4.5 ГБ после запроса на
    # 495 МБ); malloc_trim(0) отдаёт свободные страницы системе (офлайн, чат 152 МБ: 1670 -> 1216 МБ за 63 мс).
    # Только после этого пути, раз на огромный запрос; обычные запросы не затрагивает.
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass


def _json_потоком(ids, max_model_len, строки=None, шаг=1 << 20):
    """Тот же JSON, что JSONResponse(TokenizeResponse(...).model_dump()) (json.dumps, ensure_ascii=False,
    separators=(',', ':')), по частям. строки -- tokenizer.convert_ids_to_tokens (return_token_strs) или None."""
    try:
        yield f'{{"count":{len(ids)},"max_model_len":{max_model_len},"tokens":['.encode()
        for i in range(0, len(ids), шаг):
            yield ((b"," if i else b"") + ",".join(map(str, ids[i:i + шаг])).encode())
        if строки is None:
            yield b'],"token_strs":null}'
            return
        yield b'],"token_strs":['
        for i in range(0, len(ids), шаг):
            yield ((b"," if i else b"") + ",".join(
                _json_tok.dumps(t, ensure_ascii=False) for t in строки(ids[i:i + шаг].tolist())).encode())
        yield b"]}"
    finally:
        # через 5 с после ответа (или ухода клиента): запрос, тело и номера к тому времени освобождены
        _thr_tok.Timer(5.0, _вернуть_память).start()


def _объём_чата(messages):
    """Символов текста в сообщениях; -1 -- есть нетекстовые части (картинки и т. п.) -- только прежним путём."""
    n = 0
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if isinstance(c, str):
            n += len(c)
        elif isinstance(c, list):
            for part in c:
                if not isinstance(part, dict) or part.get("type", "text") != "text":
                    return -1
                n += len(part.get("text") or "")
    return n


def _ответ_ошибкой(err):
    return JSONResponse(content=err.model_dump(), status_code=err.error.code)


async def _большой_tokenize(handler, request):
    """[ПАМЯТЬ 27.10] /tokenize огромного текста потоком; None -- идти прежним путём (handler.create_tokenize)."""
    if isinstance(request, TokenizeCompletionRequest):
        объём = len(request.prompt) if isinstance(request.prompt, str) else 0
        форма = "completion"
    else:
        объём = _объём_чата(request.messages)
        форма = "chat"
    if объём <= _ПОТОК_ПОРОГ:
        return None
    err = await handler._check_model(request)
    if err is not None:
        return _ответ_ошибкой(err)
    renderer = handler.renderer
    tok_params = request.build_tok_params(handler.model_config)
    if форма == "completion":
        prompt = {"prompt": request.prompt}
    else:
        # как OpenAIServingTokenization.create_tokenize -> OpenAIServing._preprocess_chat, тот же рендер шаблона
        import jinja2

        from vllm.renderers import merge_kwargs
        from vllm.tokenizers.mistral import MistralTokenizer

        tool_dicts = None if request.tools is None else [tool.model_dump() for tool in request.tools]
        err = handler._validate_chat_template(
            request_chat_template=request.chat_template,
            chat_template_kwargs=request.chat_template_kwargs,
            trust_request_chat_template=handler.trust_request_chat_template,
        )
        if err is not None:
            return _ответ_ошибкой(err)
        chat_params = request.build_chat_params(
            handler.chat_template, handler.chat_template_content_format
        ).with_defaults(merge_kwargs(None, dict(
            tools=tool_dicts, tokenize=isinstance(renderer.tokenizer, MistralTokenizer))))
        try:
            _, prompt = await renderer.render_messages_async(request.messages, chat_params)
        except (ValueError, TypeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return _ответ_ошибкой(handler.create_error_response(f"{e} {e.__cause__}"))
        if set(prompt) != {"prompt"}:
            logger.warning("ТОКЕНИЗАЦИЯ БОЛЬШАЯ chat: %d символов, прежним путём (рендер дал %s)", объём, sorted(prompt))
            return None
    prompt = tok_params.apply_pre_tokenization(renderer.tokenizer, prompt)
    kw = tok_params.get_encode_kwargs()
    if kw.pop("truncation") or kw.pop("max_length") is not None:
        logger.warning("ТОКЕНИЗАЦИЯ БОЛЬШАЯ %s: %d символов, прежним путём (усечение)", форма, объём)
        return None
    from vllm.utils.async_utils import большой_текст_ids

    text = prompt["prompt"]
    tokenizer = renderer.get_tokenizer()
    t0 = _time_tok.time()
    logger.warning("ТОКЕНИЗАЦИЯ БОЛЬШАЯ %s: %d символов текста, потоком%s", форма, len(text),
                   " (+token_strs)" if request.return_token_strs else "")
    try:
        ids = await большой_текст_ids(tokenizer, text, kw, порог=1)   # огромность уже решена выше
    except _aio_tok.CancelledError:      # клиент ушёл -- токенизация остановлена
        logger.warning("ТОКЕНИЗАЦИЯ БОЛЬШАЯ %s: клиент ушёл через %.0f с, брошено", форма, _time_tok.time() - t0)
        raise
    if ids is None:
        logger.warning("ТОКЕНИЗАЦИЯ БОЛЬШАЯ %s: кусками нельзя, прежним путём", форма)
        return None
    logger.warning("ТОКЕНИЗАЦИЯ БОЛЬШАЯ %s: %d токенов за %.1f с, отдаю потоком", форма, len(ids),
                   _time_tok.time() - t0)
    from fastapi.responses import StreamingResponse

    return StreamingResponse(
        _json_потоком(ids, handler.max_model_len, tokenizer.convert_ids_to_tokens if request.return_token_strs else None),
        media_type="application/json")


@router.post(
    "/tokenize",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def tokenize(request: TokenizeRequest, raw_request: Request):
    handler = tokenization(raw_request)
    # [ПАМЯТЬ 27.10] ОГРОМНЫЙ ТЕКСТ -- ПОТОКОМ. Клиент шлёт /tokenize на ~494 МБ (~200 млн токенов): обычный путь
    # держит список int (~7 ГБ), его копии (input_ids, pydantic, model_dump) и json.dumps всего ответа -- API-сервер
    # до 12 ГБ, oom-kill по кругу (26-27.10). Здесь: номера кусками по границам частей в array('i') (4 Б/токен) в
    # ОТДЕЛЬНОМ потоке (общий поток токенизатора не стоит минуты), ответ -- потоком, байт в байт тот же JSON, что у
    # JSONResponse(TokenizeResponse(...).model_dump()). Обе формы: prompt и messages (тот же рендер шаблона, что у
    # create_tokenize), с return_token_strs и без. Прежним путём: картинки в сообщениях, усечение, рендер не в текст.
    # Запрос в журнал запросов (_log_inputs) здесь не пишется. FA2SM80_TOK_STREAM_CHARS=0 -- выключить.
    if _ПОТОК_ПОРОГ:
        try:
            ответ = await _большой_tokenize(handler, request)
        except _aio_tok.CancelledError:
            raise
        except Exception as e:
            logger.exception("ТОКЕНИЗАЦИЯ БОЛЬШАЯ: ошибка")
            ответ = _ответ_ошибкой(handler.create_error_response(e))
        if ответ is not None:
            return ответ

    try:
        generator = await handler.create_tokenize(request, raw_request)
    except Exception as e:
        # [FA2/SM70 29.08] СТАТУС ОТВЕТА ОБЯЗАН СОВПАДАТЬ С ТЕЛОМ.
        # Здесь возвращался ОБЪЕКТ ErrorResponse, а не JSONResponse -- и FastAPI отдавал его
        # со статусом по умолчанию, то есть **HTTP 200 с телом ошибки**. Соседняя ветка того
        # же роутера (`isinstance(generator, ErrorResponse)`) делает это правильно.
        # Отказ по отчёту падений §8-тер.2: недоступный image_url -> 200 OK и
        # {"error": {..., "code": 500}}. Любая совместимая с OpenAI библиотека на статусе 200
        # исключения не поднимет и `choices` не найдёт -- клиент видит «пустой ответ».
        _ошибка = handler.create_error_response(e)
        return JSONResponse(
            content=_ошибка.model_dump(),
            status_code=getattr(getattr(_ошибка, "error", None), "code", 500) or 500,
        )

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    elif isinstance(generator, TokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post(
    "/detokenize",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def detokenize(request: DetokenizeRequest, raw_request: Request):
    handler = tokenization(raw_request)

    try:
        generator = await handler.create_detokenize(request, raw_request)
    except OverflowError as e:
        raise RequestValidationError(errors=[str(e)]) from e
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    elif isinstance(generator, DetokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


def attach_router(app: FastAPI):
    if getattr(app.state.args, "enable_tokenizer_info_endpoint", False):
        """Conditionally register the tokenizer info endpoint if enabled."""

        @router.get("/tokenizer_info")
        async def get_tokenizer_info(raw_request: Request):
            """Get comprehensive tokenizer information."""
            result = await tokenization(raw_request).get_tokenizer_info()
            return JSONResponse(
                content=result.model_dump(),
                status_code=result.error.code
                if isinstance(result, ErrorResponse)
                else 200,
            )

    app.include_router(router)
