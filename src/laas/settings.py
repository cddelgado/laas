from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import Field
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
    settings_file: Path = DEFAULT_SETTINGS_FILE

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
