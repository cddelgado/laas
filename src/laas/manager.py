from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from huggingface_hub import hf_hub_download

from .backends import InferenceBackend, LlamaCppBackend
from .schemas import LocalModelStatus, ModelCapabilities
from .settings import Settings

BackendFactory = Callable[[Path, Settings], InferenceBackend]


class ModelNotDownloadedError(RuntimeError):
    def __init__(self, model_path: Path, asset: str = "model") -> None:
        self.model_path = model_path
        self.asset = asset
        super().__init__(f"{asset} file is not downloaded: {model_path}")


class ModelManager:
    def __init__(self, settings: Settings, backend_factory: BackendFactory | None = None) -> None:
        self.settings = settings
        self._backend_factory = backend_factory or self._default_backend_factory
        self._backend: InferenceBackend | None = None
        self._loaded_model: str | None = None
        self._last_used_at: float | None = None
        self._lock = threading.RLock()

    @property
    def capabilities(self) -> ModelCapabilities:
        return capabilities_from_settings(self.settings)

    @property
    def loaded_model(self) -> str | None:
        return self._loaded_model

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    @property
    def backend(self) -> InferenceBackend:
        with self._lock:
            self._unload_if_idle_locked()
            if self._backend is None:
                self.load(download_if_missing=self.settings.auto_download)
            assert self._backend is not None
            self._last_used_at = time.time()
            return self._backend

    def status(self) -> LocalModelStatus:
        path = self.settings.model_path
        mmproj_path = self.settings.mmproj_path
        with self._lock:
            self._unload_if_idle_locked()
            return LocalModelStatus(
                configured_model=self.settings.model_id,
                loaded_model=self._loaded_model,
                is_loaded=self._backend is not None,
                model_path=str(path),
                downloaded=path.exists(),
                mmproj_path=str(mmproj_path) if mmproj_path else None,
                mmproj_downloaded=bool(mmproj_path and mmproj_path.exists()),
                mmproj_required=self.settings.mmproj_required,
                capabilities=self.capabilities,
                idle_unload_seconds=self.settings.idle_unload_seconds,
                last_used_at=self._last_used_at,
            )

    def download_file(self, *, repo_id: str, filename: str) -> Path:
        local_dir = self.settings.model_dir / repo_id.replace("/", "__")
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
        )
        return Path(downloaded)

    def download(self, *, hf_repo_id: str | None = None, filename: str | None = None) -> Path:
        repo_id = hf_repo_id or self.settings.hf_repo_id
        target_filename = filename or self.settings.hf_filename
        return self.download_file(repo_id=repo_id, filename=target_filename)

    def download_mmproj(self) -> Path | None:
        if not self.settings.mmproj_filename:
            return None
        return self.download_file(
            repo_id=self.settings.resolved_mmproj_repo_id,
            filename=self.settings.mmproj_filename,
        )

    def download_configured_assets(self, *, include_mmproj: bool = True) -> list[Path]:
        downloaded = [self.download()]
        if include_mmproj and self.settings.mmproj_filename:
            mmproj = self.download_mmproj()
            if mmproj:
                downloaded.append(mmproj)
        return downloaded

    def missing_required_paths(self) -> list[tuple[str, Path]]:
        missing: list[tuple[str, Path]] = []
        if not self.settings.model_path.exists():
            missing.append(("model", self.settings.model_path))
        mmproj_path = self.settings.mmproj_path
        if self.settings.mmproj_required and mmproj_path and not mmproj_path.exists():
            missing.append(("mmproj", mmproj_path))
        return missing

    def _ensure_required_assets(self, *, download_if_missing: bool) -> None:
        missing = self.missing_required_paths()
        if not missing:
            return
        if not download_if_missing:
            asset, path = missing[0]
            raise ModelNotDownloadedError(path, asset=asset)

        for asset, _path in missing:
            if asset == "model":
                self.download()
            elif asset == "mmproj":
                self.download_mmproj()

        remaining = self.missing_required_paths()
        if remaining:
            asset, path = remaining[0]
            raise ModelNotDownloadedError(path, asset=asset)

    def load(
        self,
        *,
        model_id: str | None = None,
        hf_repo_id: str | None = None,
        filename: str | None = None,
        download_if_missing: bool = True,
    ) -> LocalModelStatus:
        with self._lock:
            desired_model = model_id or self.settings.model_id
            if self._backend is not None and self._loaded_model == desired_model:
                self._last_used_at = time.time()
                return self.status()

            if self._backend is not None:
                self.unload()

            if hf_repo_id:
                self.settings.hf_repo_id = hf_repo_id
            if filename:
                self.settings.hf_filename = filename

            self._ensure_required_assets(download_if_missing=download_if_missing)
            model_path = self.settings.model_path

            self._backend = self._backend_factory(model_path, self.settings)
            self._loaded_model = desired_model
            self._last_used_at = time.time()
            return self.status()

    def unload(self) -> LocalModelStatus:
        with self._lock:
            if self._backend is not None:
                self._backend.close()
            self._backend = None
            self._loaded_model = None
            self._last_used_at = None
            return self.status()

    def touch(self) -> None:
        self._last_used_at = time.time()

    def _unload_if_idle_locked(self) -> None:
        if self._backend is None or self.settings.idle_unload_seconds <= 0 or self._last_used_at is None:
            return
        if time.time() - self._last_used_at > self.settings.idle_unload_seconds:
            self._backend.close()
            self._backend = None
            self._loaded_model = None
            self._last_used_at = None

    @staticmethod
    def _default_backend_factory(model_path: Path, settings: Settings) -> InferenceBackend:
        return LlamaCppBackend(
            model_path=model_path,
            n_ctx=settings.n_ctx,
            n_gpu_layers=settings.n_gpu_layers,
            n_threads=settings.n_threads,
            verbose=settings.verbose_llama,
            mmproj_path=settings.mmproj_path if settings.mmproj_required else None,
        )


def capabilities_from_settings(settings: Settings) -> ModelCapabilities:
    multimodal_projector_configured = bool(settings.mmproj_required and settings.mmproj_filename)
    return ModelCapabilities(
        vision=multimodal_projector_configured,
        video=multimodal_projector_configured,
        audio_input=settings.llm_audio_input_enabled,
    )
