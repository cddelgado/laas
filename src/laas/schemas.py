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
    mmproj_path: str | None = None
    mmproj_downloaded: bool = False
    mmproj_required: bool = True
    capabilities: ModelCapabilities
    idle_unload_seconds: int
    last_used_at: float | None = None


class LocalAudioStatus(BaseModel):
    configured_model: str
    loaded_model: str | None
    is_loaded: bool
    model_path: str
    model_downloaded: bool
    voices_path: str
    voices_downloaded: bool
    default_voice: str
    default_lang: str
    supported_formats: list[str]
    ffmpeg_path: str | None = None
    ffmpeg_available: bool = False
    idle_unload_seconds: int
    last_used_at: float | None = None


class DownloadAudioRequest(BaseModel):
    model_id: str | None = None
    hf_repo_id: str | None = None
    model_filename: str | None = None
    voices_filename: str | None = None


class LoadAudioRequest(BaseModel):
    model_id: str | None = None
    hf_repo_id: str | None = None
    model_filename: str | None = None
    voices_filename: str | None = None
    download_if_missing: bool = True


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: str
    voice: str | None = None
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    lang: str | None = None
    is_phonemes: bool = False
    trim: bool = True


class DownloadModelRequest(BaseModel):
    model_id: str | None = None
    hf_repo_id: str | None = None
    filename: str | None = None
    include_mmproj: bool = True


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
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repeat_penalty: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    typical_p: float | None = None
    tfs_z: float | None = None
    mirostat_mode: int | None = None
    mirostat_tau: float | None = None
    mirostat_eta: float | None = None
    logit_bias: dict[int, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
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
    suffix: str | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repeat_penalty: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    typical_p: float | None = None
    tfs_z: float | None = None
    mirostat_mode: int | None = None
    mirostat_tau: float | None = None
    mirostat_eta: float | None = None
    logit_bias: dict[int, float] | None = None
    logprobs: int | None = None
    echo: bool | None = None
    stream: bool = False


class ResponseRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: Any
    instructions: str | None = None
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    max_output_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repeat_penalty: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    typical_p: float | None = None
    tfs_z: float | None = None
    mirostat_mode: int | None = None
    mirostat_tau: float | None = None
    mirostat_eta: float | None = None
    logit_bias: dict[int, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
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
    mmproj_repo_id: str | None = None
    mmproj_filename: str | None = None
    mmproj_required: bool | None = None
    auto_load: bool | None = None
    auto_download: bool | None = None
    n_ctx: int | None = Field(default=None, gt=0)
    n_gpu_layers: int | None = None
    n_threads: int | None = Field(default=None, gt=0)
    idle_unload_seconds: int | None = Field(default=None, ge=0)
    tts_model_id: str | None = None
    tts_hf_repo_id: str | None = None
    tts_model_filename: str | None = None
    tts_voices_filename: str | None = None
    tts_default_voice: str | None = None
    tts_default_lang: str | None = None
    tts_auto_load: bool | None = None
    tts_auto_download: bool | None = None
    tts_idle_unload_seconds: int | None = Field(default=None, ge=0)
    tts_ffmpeg_path: str | None = None
