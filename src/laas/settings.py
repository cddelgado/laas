from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_SETTINGS_FILE = Path(".laas/settings.json")


def default_model_dir() -> Path:
    if sys.platform.startswith("win"):
        return Path(r"D:\AI\Models")
    return Path.home() / "AI" / "Models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LAAS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8000
    model_dir: Path = Field(default_factory=default_model_dir)
    model_id: str = "gemma-4-e4b-it-q4_k_m"
    hf_repo_id: str = "ggml-org/gemma-4-E4B-it-GGUF"
    hf_filename: str = "gemma-4-E4B-it-Q4_K_M.gguf"
    mmproj_repo_id: str | None = None
    mmproj_filename: str | None = "mmproj-gemma-4-E4B-it-Q8_0.gguf"
    mmproj_required: bool = True
    auto_load: bool = False
    auto_download: bool = False
    n_ctx: int = 32768
    n_gpu_layers: int = -1
    n_threads: int | None = None
    verbose_llama: bool = False
    idle_unload_seconds: int = 900
    tts_model_id: str = "kokoro-82m"
    tts_hf_repo_id: str = "fastrtc/kokoro-onnx"
    tts_model_filename: str = "kokoro-v1.0.onnx"
    tts_voices_filename: str = "voices-v1.0.bin"
    tts_default_voice: str = "af_heart"
    tts_default_lang: str = "en-us"
    tts_auto_load: bool = False
    tts_auto_download: bool = False
    tts_idle_unload_seconds: int = 900
    tts_ffmpeg_path: str = "ffmpeg"
    stt_model_id: str = "whisper-small"
    stt_hf_repo_id: str = "ggerganov/whisper.cpp"
    stt_model_filename: str = "ggml-small.bin"
    stt_default_language: str | None = None
    stt_n_threads: int | None = None
    stt_auto_load: bool = False
    stt_auto_download: bool = False
    stt_idle_unload_seconds: int = 900
    voice_auto_load: bool = False
    voice_auto_download: bool = False
    embedding_model_id: str = "laas-hash-embedding"
    embedding_dimensions: int = 384
    image_model_id: str = "sdxl-turbo"
    image_hf_repo_id: str = "stabilityai/sdxl-turbo"
    image_default_size: str = "768x768"
    image_num_inference_steps: int = 2
    image_guidance_scale: float = 0.0
    image_default_response_format: str = "b64_json"
    image_output_dir: Path | None = None
    image_output_retention_seconds: int = 86400
    image_auto_load: bool = False
    image_auto_download: bool = True
    image_idle_unload_seconds: int = 900
    image_device: str = "auto"
    image_torch_dtype: str = "float16"
    image_variation_default_size: str = "512x512"
    image_variation_num_inference_steps: int = 4
    image_variation_guidance_scale: float = 0.0
    image_variation_strength: float = 0.55
    image_variation_prompt: str = "a high quality variation of the provided image, same subject, similar composition"
    image_edit_model_id: str = "sd-1.5-inpainting"
    image_edit_hf_repo_id: str = "stable-diffusion-v1-5/stable-diffusion-inpainting"
    image_edit_default_size: str = "512x512"
    image_edit_num_inference_steps: int = 25
    image_edit_guidance_scale: float = 7.5
    image_edit_strength: float = 0.8
    image_edit_padding_mask_crop: int | None = 32
    image_edit_composite_blur_radius: int = 4
    image_edit_auto_load: bool = False
    image_edit_auto_download: bool = True
    image_edit_idle_unload_seconds: int = 900
    settings_file: Path = DEFAULT_SETTINGS_FILE

    @field_validator("image_output_dir", mode="before")
    @classmethod
    def _empty_path_is_none(cls, value: Any) -> Any:
        if value == "":
            return None
        return value

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.hf_repo_id.replace("/", "__") / self.hf_filename

    @property
    def resolved_mmproj_repo_id(self) -> str:
        return self.mmproj_repo_id or self.hf_repo_id

    @property
    def mmproj_path(self) -> Path | None:
        if not self.mmproj_filename:
            return None
        return self.model_dir / self.resolved_mmproj_repo_id.replace("/", "__") / self.mmproj_filename

    @property
    def tts_model_path(self) -> Path:
        return self.model_dir / self.tts_hf_repo_id.replace("/", "__") / self.tts_model_filename

    @property
    def tts_voices_path(self) -> Path:
        return self.model_dir / self.tts_hf_repo_id.replace("/", "__") / self.tts_voices_filename

    @property
    def stt_model_path(self) -> Path:
        return self.model_dir / self.stt_hf_repo_id.replace("/", "__") / self.stt_model_filename

    @property
    def image_model_path(self) -> Path:
        return self.model_dir / self.image_hf_repo_id.replace("/", "__")

    @property
    def image_edit_model_path(self) -> Path:
        return self.model_dir / self.image_edit_hf_repo_id.replace("/", "__")

    @property
    def resolved_image_output_dir(self) -> Path:
        return self.image_output_dir or (self.model_dir / "outputs" / "images")

    def public_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "model_dir": str(self.model_dir),
            "model_id": self.model_id,
            "hf_repo_id": self.hf_repo_id,
            "hf_filename": self.hf_filename,
            "mmproj_repo_id": self.mmproj_repo_id,
            "mmproj_filename": self.mmproj_filename,
            "mmproj_required": self.mmproj_required,
            "auto_load": self.auto_load,
            "auto_download": self.auto_download,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "n_threads": self.n_threads,
            "idle_unload_seconds": self.idle_unload_seconds,
            "tts_model_id": self.tts_model_id,
            "tts_hf_repo_id": self.tts_hf_repo_id,
            "tts_model_filename": self.tts_model_filename,
            "tts_voices_filename": self.tts_voices_filename,
            "tts_default_voice": self.tts_default_voice,
            "tts_default_lang": self.tts_default_lang,
            "tts_auto_load": self.tts_auto_load,
            "tts_auto_download": self.tts_auto_download,
            "tts_idle_unload_seconds": self.tts_idle_unload_seconds,
            "tts_ffmpeg_path": self.tts_ffmpeg_path,
            "stt_model_id": self.stt_model_id,
            "stt_hf_repo_id": self.stt_hf_repo_id,
            "stt_model_filename": self.stt_model_filename,
            "stt_default_language": self.stt_default_language,
            "stt_n_threads": self.stt_n_threads,
            "stt_auto_load": self.stt_auto_load,
            "stt_auto_download": self.stt_auto_download,
            "stt_idle_unload_seconds": self.stt_idle_unload_seconds,
            "voice_auto_load": self.voice_auto_load,
            "voice_auto_download": self.voice_auto_download,
            "embedding_model_id": self.embedding_model_id,
            "embedding_dimensions": self.embedding_dimensions,
            "image_model_id": self.image_model_id,
            "image_hf_repo_id": self.image_hf_repo_id,
            "image_default_size": self.image_default_size,
            "image_num_inference_steps": self.image_num_inference_steps,
            "image_guidance_scale": self.image_guidance_scale,
            "image_default_response_format": self.image_default_response_format,
            "image_output_dir": str(self.image_output_dir) if self.image_output_dir else None,
            "image_output_retention_seconds": self.image_output_retention_seconds,
            "image_auto_load": self.image_auto_load,
            "image_auto_download": self.image_auto_download,
            "image_idle_unload_seconds": self.image_idle_unload_seconds,
            "image_device": self.image_device,
            "image_torch_dtype": self.image_torch_dtype,
            "image_variation_default_size": self.image_variation_default_size,
            "image_variation_num_inference_steps": self.image_variation_num_inference_steps,
            "image_variation_guidance_scale": self.image_variation_guidance_scale,
            "image_variation_strength": self.image_variation_strength,
            "image_variation_prompt": self.image_variation_prompt,
            "image_edit_model_id": self.image_edit_model_id,
            "image_edit_hf_repo_id": self.image_edit_hf_repo_id,
            "image_edit_default_size": self.image_edit_default_size,
            "image_edit_num_inference_steps": self.image_edit_num_inference_steps,
            "image_edit_guidance_scale": self.image_edit_guidance_scale,
            "image_edit_strength": self.image_edit_strength,
            "image_edit_padding_mask_crop": self.image_edit_padding_mask_crop,
            "image_edit_composite_blur_radius": self.image_edit_composite_blur_radius,
            "image_edit_auto_load": self.image_edit_auto_load,
            "image_edit_auto_download": self.image_edit_auto_download,
            "image_edit_idle_unload_seconds": self.image_edit_idle_unload_seconds,
        }


def load_settings() -> Settings:
    base = Settings()
    if not base.settings_file.exists():
        return base

    with base.settings_file.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return Settings(**payload)


def save_settings(settings: Settings, updates: dict[str, Any]) -> Settings:
    accepted = set(settings.public_dict())
    unknown = set(updates) - accepted
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise ValueError(f"unknown setting(s): {unknown_list}")

    next_payload = settings.public_dict() | updates
    settings.settings_file.parent.mkdir(parents=True, exist_ok=True)
    with settings.settings_file.open("w", encoding="utf-8") as fh:
        json.dump(next_payload, fh, indent=2)
        fh.write("\n")
    return Settings(**next_payload)
