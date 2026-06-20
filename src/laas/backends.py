from __future__ import annotations

import time
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from .tools import parse_tool_calls, remove_tool_call_markup


class InferenceBackend(ABC):
    @abstractmethod
    def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        stream: bool,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | Iterable[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def completion(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        stream: bool,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | Iterable[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class LlamaCppBackend(InferenceBackend):
    def __init__(
        self,
        *,
        model_path: Path,
        n_ctx: int,
        n_gpu_layers: int,
        n_threads: int | None,
        verbose: bool,
        mmproj_path: Path | None = None,
    ) -> None:
        try:
            from llama_cpp import Llama
        except Exception as exc:
            raise RuntimeError("llama-cpp-python is required: pip install -e .[llama]") from exc

        kwargs: dict[str, Any] = {
            "model_path": str(model_path),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": verbose,
        }
        if n_threads:
            kwargs["n_threads"] = n_threads
        if mmproj_path:
            _add_mmproj_kwargs(
                Llama,
                kwargs,
                mmproj_path,
                verbose=verbose,
                use_gpu=n_gpu_layers != 0,
            )
        self._llm = Llama(**kwargs)

    def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        stream: bool,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | Iterable[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature if temperature is not None else 1.0,
            "top_p": top_p if top_p is not None else 1.0,
            "stream": stream,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        kwargs.update(extra_params or {})
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        return self._llm.create_chat_completion(**kwargs)

    def completion(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        stream: bool,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | Iterable[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "temperature": temperature if temperature is not None else 1.0,
            "top_p": top_p if top_p is not None else 1.0,
            "stream": stream,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        kwargs.update(extra_params or {})
        return self._llm.create_completion(**kwargs)

    def close(self) -> None:
        self._llm = None


class EchoBackend(InferenceBackend):
    """Tiny backend used by tests and explicit dry-run wiring."""

    def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        stream: bool,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | Iterable[dict[str, Any]]:
        content = _last_user_text(messages) or "ok"
        if tools and tool_choice not in {"none", None} and "call_tool" in content:
            tool = tools[0].get("function", tools[0])
            content = f'<tool_call>{{"name":"{tool.get("name", "tool")}","arguments":{{}}}}</tool_call>'

        message: dict[str, Any] = {"role": "assistant", "content": content}
        tool_calls = parse_tool_calls(content, tools)
        if tool_calls:
            message["content"] = remove_tool_call_markup(content) or None
            message["tool_calls"] = tool_calls

        result = {
            "id": "chatcmpl_echo",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        if not stream:
            return result
        return iter(
            [
                {
                    "id": "chatcmpl_echo",
                    "object": "chat.completion.chunk",
                    "created": result["created"],
                    "model": model,
                    "choices": [{"index": 0, "delta": message, "finish_reason": None}],
                },
                {
                    "id": "chatcmpl_echo",
                    "object": "chat.completion.chunk",
                    "created": result["created"],
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
        )

    def completion(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        stream: bool,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | Iterable[dict[str, Any]]:
        result = {
            "id": "cmpl_echo",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "text": prompt, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        if not stream:
            return result
        return iter(
            [
                {
                    "id": "cmpl_echo",
                    "object": "text_completion",
                    "created": result["created"],
                    "model": model,
                    "choices": [{"index": 0, "text": prompt, "finish_reason": "stop"}],
                }
            ]
        )

    def close(self) -> None:
        return None


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [part.get("text", "") for part in content if part.get("type") in {"text", "input_text"}]
            return " ".join(texts)
    return ""


def _add_mmproj_kwargs(
    llama_cls: Any,
    kwargs: dict[str, Any],
    mmproj_path: Path,
    *,
    verbose: bool = False,
    use_gpu: bool = True,
    chat_handler_cls: Any | None = None,
) -> None:
    signature = inspect.signature(llama_cls)
    parameters = signature.parameters
    if "mmproj" in parameters:
        kwargs["mmproj"] = str(mmproj_path)
        return
    if "mmproj_path" in parameters:
        kwargs["mmproj_path"] = str(mmproj_path)
        return
    if "clip_model_path" in parameters:
        kwargs["clip_model_path"] = str(mmproj_path)
        return
    if "chat_handler" in parameters:
        if chat_handler_cls is None:
            try:
                from llama_cpp.llama_chat_format import Gemma4ChatHandler as chat_handler_cls
            except Exception as exc:
                raise RuntimeError(
                    "The configured model requires a multimodal projector, but the installed "
                    "llama-cpp-python package does not expose Gemma4ChatHandler."
                ) from exc
        kwargs["chat_handler"] = chat_handler_cls(
            clip_model_path=str(mmproj_path),
            verbose=verbose,
            use_gpu=use_gpu,
        )
        return
    raise RuntimeError(
        "The configured model requires a multimodal projector, but the installed "
        "llama-cpp-python Llama constructor does not expose mmproj/mmproj_path/clip_model_path/chat_handler. "
        "Install a llama-cpp-python build with multimodal projector support or set "
        "LAAS_MMPROJ_REQUIRED=false for text-only use."
    )
