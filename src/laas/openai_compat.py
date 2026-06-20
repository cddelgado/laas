from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterable

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .errors import openai_error
from .manager import ModelManager, ModelNotDownloadedError
from .multimodal import normalize_chat_messages, normalize_content_parts
from .schemas import ChatCompletionRequest, CompletionRequest, ModelList, OpenAIModel, ResponseRequest
from .tools import normalize_tools_for_responses, parse_tool_calls, remove_tool_call_markup


def build_openai_router(manager: ModelManager) -> APIRouter:
    router = APIRouter(prefix="/v1")

    @router.get("/models", response_model=ModelList)
    def list_models() -> ModelList:
        return ModelList(
            data=[
                OpenAIModel(
                    id=manager.settings.model_id,
                    created=1712966400,
                    owned_by="google-local-gguf",
                )
            ]
        )

    @router.get("/models/{model_id}")
    def retrieve_model(model_id: str) -> dict[str, Any]:
        if model_id != manager.settings.model_id:
            raise openai_error(404, f"The model '{model_id}' does not exist", param="model", code="model_not_found")
        return OpenAIModel(id=model_id, created=1712966400, owned_by="google-local-gguf").model_dump()

    @router.post("/chat/completions")
    def create_chat_completion(request: ChatCompletionRequest) -> Any:
        _assert_model(request.model, manager)
        _validate_capabilities(request, manager)
        messages = normalize_chat_messages([message.model_dump(exclude_none=True) for message in request.messages])
        tools = normalize_tools_for_responses(request.tools)
        backend = _get_backend(manager)
        result = backend.chat_completion(
            messages=messages,
            model=manager.settings.model_id,
            tools=tools,
            tool_choice=request.tool_choice,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.requested_max_tokens,
            stream=request.stream,
            extra_params=_chat_sampling_params(request),
        )
        if request.stream:
            return _sse(result)
        return _normalize_chat_response(result, manager.settings.model_id, tools)

    @router.post("/completions")
    def create_completion(request: CompletionRequest) -> Any:
        _assert_model(request.model, manager)
        prompt = request.prompt[0] if isinstance(request.prompt, list) else request.prompt
        result = _get_backend(manager).completion(
            prompt=prompt,
            model=manager.settings.model_id,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stream=request.stream,
            extra_params=_completion_sampling_params(request),
        )
        if request.stream:
            return _sse(result)
        return _normalize_completion_response(result, manager.settings.model_id)

    @router.post("/responses")
    def create_response(request: ResponseRequest) -> Any:
        _assert_model(request.model, manager)
        messages = _responses_input_to_messages(request)
        chat_request = ChatCompletionRequest(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_completion_tokens=request.max_output_tokens,
            stream=request.stream,
            tools=normalize_tools_for_responses(request.tools),
            tool_choice=request.tool_choice,
            response_format=_response_text_format(request.text),
            stop=request.stop,
            seed=request.seed,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            repeat_penalty=request.repeat_penalty,
            top_k=request.top_k,
            min_p=request.min_p,
            typical_p=request.typical_p,
            tfs_z=request.tfs_z,
            mirostat_mode=request.mirostat_mode,
            mirostat_tau=request.mirostat_tau,
            mirostat_eta=request.mirostat_eta,
            logit_bias=request.logit_bias,
            logprobs=request.logprobs,
            top_logprobs=request.top_logprobs,
        )
        _validate_capabilities(chat_request, manager)
        result = _get_backend(manager).chat_completion(
            messages=normalize_chat_messages([message.model_dump(exclude_none=True) for message in chat_request.messages]),
            model=manager.settings.model_id,
            tools=chat_request.tools,
            tool_choice=chat_request.tool_choice,
            temperature=chat_request.temperature,
            top_p=chat_request.top_p,
            max_tokens=chat_request.requested_max_tokens,
            stream=request.stream,
            extra_params=_chat_sampling_params(chat_request),
        )
        if request.stream:
            return _sse(_responses_stream(result, manager.settings.model_id))
        chat_response = _normalize_chat_response(result, manager.settings.model_id, chat_request.tools)
        return _chat_to_response(chat_response, request, manager.settings.model_id)

    return router


def _assert_model(requested_model: str | None, manager: ModelManager) -> None:
    if requested_model and requested_model != manager.settings.model_id:
        raise openai_error(
            404,
            f"The model '{requested_model}' is not loaded. Loaded/configured model is '{manager.settings.model_id}'.",
            param="model",
            code="model_not_found",
        )


def _get_backend(manager: ModelManager):
    try:
        return manager.backend
    except ModelNotDownloadedError as exc:
        raise openai_error(
            409,
            f"The configured {exc.asset} is not downloaded. Call POST /v1/local/models/download and "
            "POST /v1/local/models/load before inference, or set LAAS_AUTO_DOWNLOAD=true to allow "
            "LAAS to download missing model assets during load.",
            type_="invalid_request_error",
            param=exc.asset,
            code="model_not_downloaded",
        ) from exc


def _validate_capabilities(request: ChatCompletionRequest, manager: ModelManager) -> None:
    if request.tools and not manager.capabilities.tool_calls:
        raise openai_error(400, "the loaded model does not support tool calls", param="tools")
    for message in request.messages:
        content = message.content
        if not isinstance(content, list):
            continue
        for part in content:
            part_type = part.get("type") if isinstance(part, dict) else None
            if part_type in {"image_url", "input_image"} and not manager.capabilities.vision:
                raise openai_error(400, "the loaded model does not support image inputs", param="messages")
            if part_type in {"video_url", "input_video"} and not manager.capabilities.video:
                raise openai_error(400, "the loaded model does not support video inputs", param="messages")
            if part_type in {"audio", "input_audio"} and not manager.capabilities.audio_input:
                raise openai_error(400, "the loaded model does not support audio inputs", param="messages")


def _chat_sampling_params(request: ChatCompletionRequest) -> dict[str, Any]:
    return _present_params(
        request,
        [
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "repeat_penalty",
            "top_k",
            "min_p",
            "typical_p",
            "tfs_z",
            "mirostat_mode",
            "mirostat_tau",
            "mirostat_eta",
            "response_format",
            "logit_bias",
            "logprobs",
            "top_logprobs",
        ],
    )


def _completion_sampling_params(request: CompletionRequest) -> dict[str, Any]:
    return _present_params(
        request,
        [
            "suffix",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "repeat_penalty",
            "top_k",
            "min_p",
            "typical_p",
            "tfs_z",
            "mirostat_mode",
            "mirostat_tau",
            "mirostat_eta",
            "logit_bias",
            "logprobs",
            "echo",
        ],
    )


def _present_params(request: Any, fields: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for field in fields:
        value = getattr(request, field)
        if value is not None:
            params[field] = value
    return params


def _response_text_format(text: dict[str, Any] | None) -> dict[str, Any] | None:
    if not text:
        return None
    response_format = text.get("format", text)
    return response_format if isinstance(response_format, dict) else None


def _responses_input_to_messages(request: ResponseRequest) -> list[Any]:
    messages: list[dict[str, Any]] = []
    if request.instructions:
        messages.append({"role": "system", "content": request.instructions})

    if isinstance(request.input, str):
        messages.append({"role": "user", "content": request.input})
    elif isinstance(request.input, list):
        for item in request.input:
            if not isinstance(item, dict):
                raise openai_error(400, "response input items must be objects", param="input")
            item_type = item.get("type")
            if item_type == "message" or "role" in item:
                role = item.get("role", "user")
                content = item.get("content", "")
                if isinstance(content, list):
                    content = normalize_content_parts(content)
                messages.append({"role": role, "content": content})
            elif item_type in {"input_text", "input_image", "input_video", "input_audio"}:
                messages.append({"role": "user", "content": normalize_content_parts([item])})
            elif item_type == "function_call_output":
                messages.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": item.get("output", "")})
            else:
                raise openai_error(400, f"unsupported response input item type: {item_type}", param="input")
    else:
        raise openai_error(400, "input must be a string or array", param="input")

    from .schemas import ChatMessage

    return [ChatMessage(**message) for message in messages]


def _normalize_chat_response(result: Any, model_id: str, tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise openai_error(500, "backend returned an invalid response", type_="server_error")
    result.setdefault("id", f"chatcmpl_{uuid.uuid4().hex}")
    result.setdefault("object", "chat.completion")
    result.setdefault("created", int(time.time()))
    result["model"] = model_id

    for choice in result.get("choices", []):
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and "tool_calls" not in message:
            tool_calls = parse_tool_calls(content, tools)
            if tool_calls:
                message["tool_calls"] = tool_calls
                message["content"] = remove_tool_call_markup(content) or None
                choice["finish_reason"] = "tool_calls"
        choice["message"] = message
    result.setdefault("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    return result


def _normalize_completion_response(result: Any, model_id: str) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise openai_error(500, "backend returned an invalid response", type_="server_error")
    result.setdefault("id", f"cmpl_{uuid.uuid4().hex}")
    result.setdefault("object", "text_completion")
    result.setdefault("created", int(time.time()))
    result["model"] = model_id
    result.setdefault("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    return result


def _chat_to_response(chat_response: dict[str, Any], request: ResponseRequest, model_id: str) -> dict[str, Any]:
    choice = chat_response["choices"][0]
    message = choice["message"]
    output: list[dict[str, Any]] = []

    for call in message.get("tool_calls") or []:
        output.append(
            {
                "type": "function_call",
                "id": call["id"],
                "call_id": call["id"],
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
                "status": "completed",
            }
        )

    if message.get("content"):
        output.append(
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": message["content"], "annotations": []}],
            }
        )

    output_text = "".join(
        part["text"]
        for item in output
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )
    now = int(time.time())
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": now,
        "status": "completed",
        "model": model_id,
        "output": output,
        "output_text": output_text,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "temperature": request.temperature,
        "tool_choice": request.tool_choice,
        "tools": request.tools or [],
        "top_p": request.top_p,
        "usage": {
            "input_tokens": chat_response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": chat_response.get("usage", {}).get("completion_tokens", 0),
            "total_tokens": chat_response.get("usage", {}).get("total_tokens", 0),
        },
    }


def _sse(chunks: Any) -> StreamingResponse:
    def events() -> Iterable[str]:
        for chunk in chunks:
            yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def _responses_stream(chunks: Any, model_id: str) -> Iterable[dict[str, Any]]:
    response_id = f"resp_{uuid.uuid4().hex}"
    yield {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "model": model_id,
            "output": [],
        },
    }
    for chunk in chunks:
        delta = ""
        choices = chunk.get("choices", []) if isinstance(chunk, dict) else []
        if choices:
            delta = choices[0].get("delta", {}).get("content") or choices[0].get("text", "")
        if delta:
            yield {
                "type": "response.output_text.delta",
                "response_id": response_id,
                "delta": delta,
            }
    yield {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": model_id,
            "output": [],
        },
    }
