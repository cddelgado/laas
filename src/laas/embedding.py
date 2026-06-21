from __future__ import annotations

import base64
import hashlib
import math
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import snapshot_download

from .schemas import LocalEmbeddingStatus
from .settings import Settings


class EmbeddingNotDownloadedError(RuntimeError):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.asset = "embedding_model"
        super().__init__(f"embedding model is not downloaded: {path}")


class EmbeddingBackend:
    @property
    def dimensions(self) -> int:
        raise NotImplementedError

    def embed(self, inputs: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class HashEmbeddingBackend(EmbeddingBackend):
    def __init__(self, *, dimensions: int) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, inputs: list[str]) -> list[list[float]]:
        return [_hash_embedding(value, self._dimensions) for value in inputs]

    def close(self) -> None:
        return None


class SentenceTransformerEmbeddingBackend(EmbeddingBackend):
    def __init__(self, *, model_path: Path, device: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError("sentence-transformers is required: pip install -e .[embeddings]") from exc

        kwargs: dict[str, Any] = {}
        if device != "auto":
            kwargs["device"] = device
        self._model = SentenceTransformer(str(model_path), **kwargs)
        get_dimensions = getattr(self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension)
        self._dimensions = int(get_dimensions() or 0)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, inputs: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            inputs,
            normalize_embeddings=True,
            convert_to_numpy=False,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]

    def close(self) -> None:
        self._model = None


EmbeddingBackendFactory = Callable[[Path, Settings], EmbeddingBackend]


class EmbeddingManager:
    def __init__(self, settings: Settings, backend_factory: EmbeddingBackendFactory | None = None) -> None:
        self.settings = settings
        self._backend_factory = backend_factory or self._default_backend_factory
        self._backend: EmbeddingBackend | None = None
        self._loaded_model: str | None = None
        self._last_used_at: float | None = None
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    @property
    def backend(self) -> EmbeddingBackend:
        with self._lock:
            self._unload_if_idle_locked()
            if self._backend is None:
                self.load(download_if_missing=self.settings.embedding_auto_download)
            assert self._backend is not None
            self._last_used_at = time.time()
            return self._backend

    def status(self) -> LocalEmbeddingStatus:
        with self._lock:
            self._unload_if_idle_locked()
            return LocalEmbeddingStatus(
                configured_model=self.settings.embedding_model_id,
                loaded_model=self._loaded_model,
                is_loaded=self._backend is not None,
                model_path=str(self.settings.embedding_model_path),
                downloaded=self.downloaded,
                hf_repo_id=self.settings.embedding_hf_repo_id,
                dimensions=self.settings.embedding_dimensions,
                device=self.settings.embedding_device,
                idle_unload_seconds=self.settings.embedding_idle_unload_seconds,
                last_used_at=self._last_used_at,
            )

    @property
    def downloaded(self) -> bool:
        path = self.settings.embedding_model_path
        return path.exists() and any(path.iterdir())

    def download(self) -> Path:
        local_dir = self.settings.embedding_model_path
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded = snapshot_download(repo_id=self.settings.embedding_hf_repo_id, local_dir=local_dir)
        return Path(downloaded)

    def load(
        self,
        *,
        model_id: str | None = None,
        hf_repo_id: str | None = None,
        download_if_missing: bool = True,
    ) -> LocalEmbeddingStatus:
        with self._lock:
            desired_model = model_id or self.settings.embedding_model_id
            if self._backend is not None and self._loaded_model == desired_model:
                self._last_used_at = time.time()
                return self.status()

            if self._backend is not None:
                self.unload()

            if hf_repo_id:
                self.settings.embedding_hf_repo_id = hf_repo_id

            if not self.downloaded:
                if not download_if_missing:
                    raise EmbeddingNotDownloadedError(self.settings.embedding_model_path)
                self.download()

            if not self.downloaded:
                raise EmbeddingNotDownloadedError(self.settings.embedding_model_path)

            self._backend = self._backend_factory(self.settings.embedding_model_path, self.settings)
            self._loaded_model = desired_model
            self._last_used_at = time.time()
            return self.status()

    def unload(self) -> LocalEmbeddingStatus:
        with self._lock:
            if self._backend is not None:
                self._backend.close()
            self._backend = None
            self._loaded_model = None
            self._last_used_at = None
            return self.status()

    def embed(self, inputs: list[str], *, dimensions: int | None = None) -> list[list[float]]:
        vectors = self.backend.embed(inputs)
        target_dimensions = dimensions or self.settings.embedding_dimensions
        shaped = [_shape_embedding(vector, target_dimensions) for vector in vectors]
        self._last_used_at = time.time()
        return shaped

    def _unload_if_idle_locked(self) -> None:
        if (
            self._backend is None
            or self.settings.embedding_idle_unload_seconds <= 0
            or self._last_used_at is None
        ):
            return
        if time.time() - self._last_used_at > self.settings.embedding_idle_unload_seconds:
            self._backend.close()
            self._backend = None
            self._loaded_model = None
            self._last_used_at = None

    @staticmethod
    def _default_backend_factory(model_path: Path, settings: Settings) -> EmbeddingBackend:
        return SentenceTransformerEmbeddingBackend(model_path=model_path, device=settings.embedding_device)


def encode_embedding(vector: list[float], encoding_format: str) -> list[float] | str:
    if encoding_format == "float":
        return vector
    packed = b"".join(struct.pack("<f", value) for value in vector)
    return base64.b64encode(packed).decode("ascii")


def estimate_tokens(value: str) -> int:
    return max(1, len(value.split()))


def _hash_embedding(value: str, dimensions: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < dimensions:
        digest = hashlib.sha256(f"{value}\0{counter}".encode("utf-8")).digest()
        for offset in range(0, len(digest), 4):
            integer = int.from_bytes(digest[offset : offset + 4], "big", signed=False)
            values.append((integer / 0xFFFFFFFF) * 2.0 - 1.0)
            if len(values) == dimensions:
                break
        counter += 1
    return _normalize(values)


def _shape_embedding(vector: list[float], dimensions: int) -> list[float]:
    if dimensions <= len(vector):
        return _normalize(vector[:dimensions])
    return _normalize([*vector, *([0.0] * (dimensions - len(vector)))])


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(item * item for item in vector)) or 1.0
    return [item / magnitude for item in vector]
