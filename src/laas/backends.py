from __future__ import annotations

import time
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from .native import configure_native_dll_directories
from .tools import parse_tool_calls, remove_tool_call_markup, selected_tool_name, tool_name


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
        n_gpu_layers: int | None,
        n_threads: int | None,
        n_threads_batch: int | None,
        n_batch: int | None,
        n_ubatch: int | None,
        flash_attn: bool,
        offload_kqv: bool,
        op_offload: bool | None,
        swa_full: bool | None,
        speculative_decoding: bool,
        speculative_mode: str,
        speculative_max_ngram_size: int,
        speculative_num_pred_tokens: int,
        verbose: bool,
        mmproj_path: Path | None = None,
    ) -> None:
        configure_native_dll_directories()
        try:
            from llama_cpp import Llama
        except Exception as exc:
            raise RuntimeError("llama-cpp-python is required: pip install -e .[llama]") from exc

        kwargs: dict[str, Any] = {
            "model_path": str(model_path),
            "n_ctx": n_ctx,
            "verbose": verbose,
        }
        _add_supported_constructor_kwargs(
            Llama,
            kwargs,
            {
                "n_gpu_layers": n_gpu_layers,
                "n_threads": n_threads,
                "n_threads_batch": n_threads_batch,
                "n_batch": n_batch,
                "n_ubatch": n_ubatch,
                "flash_attn": flash_attn,
                "offload_kqv": offload_kqv,
                "op_offload": op_offload,
                "swa_full": swa_full,
            },
        )
        if speculative_decoding:
            _add_speculative_kwargs(
                Llama,
                kwargs,
                mode=speculative_mode,
                max_ngram_size=speculative_max_ngram_size,
                num_pred_tokens=speculative_num_pred_tokens,
            )
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
        if tools and tool_choice != "none" and "call_tool" in content:
            chosen_name = selected_tool_name(tool_choice)
            selected_tool = next(
                (tool for tool in tools if chosen_name and tool_name(tool) == chosen_name),
                tools[0],
            )
            tool = selected_tool.get("function", selected_tool)
            content = f'<tool_call>{{"name":"{tool.get("name", "tool")}","arguments":{{}}}}</tool_call>'

        message: dict[str, Any] = {"role": "assistant", "content": content}
        tool_calls = parse_tool_calls(content, tools, tool_choice)
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


def _add_supported_constructor_kwargs(
    llama_cls: Any,
    kwargs: dict[str, Any],
    candidates: dict[str, Any],
) -> None:
    parameters = inspect.signature(llama_cls).parameters
    for name, value in candidates.items():
        if value is None:
            continue
        if name in parameters:
            kwargs[name] = value


def _add_speculative_kwargs(
    llama_cls: Any,
    kwargs: dict[str, Any],
    *,
    mode: str,
    max_ngram_size: int,
    num_pred_tokens: int,
) -> None:
    parameters = inspect.signature(llama_cls).parameters
    if "draft_model" not in parameters:
        raise RuntimeError("Installed llama-cpp-python does not expose draft_model speculative decoding support.")
    if mode != "prompt_lookup":
        raise RuntimeError(
            "Only prompt_lookup speculative decoding is supported through llama-cpp-python in LAAS. "
            "External Gemma MTP GGUF draft models are tracked as assets but are not exposed by this binding."
        )
    try:
        from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
    except Exception as exc:
        raise RuntimeError("Installed llama-cpp-python does not expose LlamaPromptLookupDecoding.") from exc
    kwargs["draft_model"] = LlamaPromptLookupDecoding(
        max_ngram_size=max_ngram_size,
        num_pred_tokens=num_pred_tokens,
    )
