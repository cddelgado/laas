from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelCapabilities(BaseModel):
    text: bool = True
    tool_calls: bool = True
    vision: bool = True
    video: bool = True
    audio_input: bool = True
    reasoning: bool = True
    text_output: bool = True
    image_output: bool = False
    audio_output: bool = False


class LocalModelStatus(BaseModel):
    configured_model: str
    loaded_model: str | None
    is_loaded: bool
    model_path: str
    downloaded: bool
    capabilities: ModelCapabilities
    idle_unload_seconds: int
    last_used_at: float | None = None


class DownloadModelRequest(BaseModel):
    model_id: str | None = None
    hf_repo_id: str | None = None
    filename: str | None = None


class LoadModelRequest(BaseModel):
    model_id: str | None = None
    hf_repo_id: str | None = None
    filename: str | None = None
    download_if_missing: bool = True


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = "auto"
    response_format: dict[str, Any] | None = None

    @property
    def requested_max_tokens(self) -> int | None:
        return self.max_completion_tokens if self.max_completion_tokens is not None else self.max_tokens


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    prompt: str | list[str]
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    max_tokens: int | None = 16
    stream: bool = False


class ResponseRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: Any
    instructions: str | None = None
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    max_output_tokens: int | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = "auto"
    text: dict[str, Any] | None = None


class OpenAIModel(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "local"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[OpenAIModel]


class SettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str | None = None
    port: int | None = None
    model_dir: str | None = None
    model_id: str | None = None
    hf_repo_id: str | None = None
    hf_filename: str | None = None
    auto_load: bool | None = None
    auto_download: bool | None = None
    n_ctx: int | None = Field(default=None, gt=0)
    n_gpu_layers: int | None = None
    n_threads: int | None = Field(default=None, gt=0)
    idle_unload_seconds: int | None = Field(default=None, ge=0)
