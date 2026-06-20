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
    auto_load: bool = False
    n_ctx: int = 32768
    n_gpu_layers: int = -1
    n_threads: int | None = None
    verbose_llama: bool = False
    idle_unload_seconds: int = 900
    settings_file: Path = DEFAULT_SETTINGS_FILE

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.hf_repo_id.replace("/", "__") / self.hf_filename

    def public_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "model_dir": str(self.model_dir),
            "model_id": self.model_id,
            "hf_repo_id": self.hf_repo_id,
            "hf_filename": self.hf_filename,
            "auto_load": self.auto_load,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "n_threads": self.n_threads,
            "idle_unload_seconds": self.idle_unload_seconds,
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
