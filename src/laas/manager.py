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
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        super().__init__(f"model file is not downloaded: {model_path}")


class ModelManager:
    def __init__(self, settings: Settings, backend_factory: BackendFactory | None = None) -> None:
        self.settings = settings
        self.capabilities = ModelCapabilities()
        self._backend_factory = backend_factory or self._default_backend_factory
        self._backend: InferenceBackend | None = None
        self._loaded_model: str | None = None
        self._last_used_at: float | None = None
        self._lock = threading.RLock()

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
        with self._lock:
            self._unload_if_idle_locked()
            return LocalModelStatus(
                configured_model=self.settings.model_id,
                loaded_model=self._loaded_model,
                is_loaded=self._backend is not None,
                model_path=str(path),
                downloaded=path.exists(),
                capabilities=self.capabilities,
                idle_unload_seconds=self.settings.idle_unload_seconds,
                last_used_at=self._last_used_at,
            )

    def download(self, *, hf_repo_id: str | None = None, filename: str | None = None) -> Path:
        repo_id = hf_repo_id or self.settings.hf_repo_id
        target_filename = filename or self.settings.hf_filename
        local_dir = self.settings.model_dir / repo_id.replace("/", "__")
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=target_filename,
            local_dir=local_dir,
        )
        return Path(downloaded)

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

            model_path = self.settings.model_path
            if not model_path.exists():
                if not download_if_missing:
                    raise ModelNotDownloadedError(model_path)
                model_path = self.download(hf_repo_id=hf_repo_id, filename=filename)

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
        )
