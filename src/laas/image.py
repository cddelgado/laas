from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download

from .schemas import LocalImageStatus
from .settings import Settings


@dataclass
class GeneratedImage:
    content: bytes
    media_type: str = "image/png"


class ImageNotDownloadedError(RuntimeError):
    def __init__(self, path: Path, asset: str = "image_model") -> None:
        self.path = path
        self.asset = asset
        super().__init__(f"{asset} is not downloaded: {path}")


class ImageBackend:
    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
    ) -> GeneratedImage:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class DiffusersImageBackend(ImageBackend):
    def __init__(self, *, model_path: Path, settings: Settings) -> None:
        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except Exception as exc:
            raise RuntimeError("diffusers image support is required: pip install -e .[image]") from exc

        dtype = _torch_dtype(torch, settings.image_torch_dtype)
        self._device = _resolve_device(torch, settings.image_device)
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
        )
        if self._device == "cuda":
            self._pipe.to("cuda")
        elif self._device == "mps":
            self._pipe.to("mps")
        else:
            self._pipe.to("cpu")

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
    ) -> GeneratedImage:
        generator = None
        if seed is not None:
            try:
                import torch

                generator = torch.Generator(device=self._device).manual_seed(seed)
            except Exception:
                generator = None
        output = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        image = output.images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return GeneratedImage(content=buffer.getvalue(), media_type="image/png")

    def close(self) -> None:
        self._pipe = None


ImageBackendFactory = Callable[[Path, Settings], ImageBackend]


class ImageManager:
    def __init__(self, settings: Settings, backend_factory: ImageBackendFactory | None = None) -> None:
        self.settings = settings
        self._backend_factory = backend_factory or self._default_backend_factory
        self._backend: ImageBackend | None = None
        self._loaded_model: str | None = None
        self._last_used_at: float | None = None
        self._download_in_progress = False
        self._download_started_at: float | None = None
        self._download_finished_at: float | None = None
        self._last_download_error: str | None = None
        self._lock = threading.RLock()
        self._download_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    @property
    def backend(self) -> ImageBackend:
        with self._lock:
            self._unload_if_idle_locked()
            if self._backend is None:
                self.load(download_if_missing=self.settings.image_auto_download)
            assert self._backend is not None
            self._last_used_at = time.time()
            return self._backend

    def status(self) -> LocalImageStatus:
        with self._lock:
            self._unload_if_idle_locked()
            return LocalImageStatus(
                configured_model=self.settings.image_model_id,
                loaded_model=self._loaded_model,
                is_loaded=self._backend is not None,
                model_path=str(self.settings.image_model_path),
                downloaded=self._is_downloaded(),
                default_size=self.settings.image_default_size,
                num_inference_steps=self.settings.image_num_inference_steps,
                guidance_scale=self.settings.image_guidance_scale,
                device=self.settings.image_device,
                torch_dtype=self.settings.image_torch_dtype,
                idle_unload_seconds=self.settings.image_idle_unload_seconds,
                last_used_at=self._last_used_at,
                download_in_progress=self._download_in_progress,
                download_started_at=self._download_started_at,
                download_finished_at=self._download_finished_at,
                last_download_error=self._last_download_error,
            )

    def download(self, *, hf_repo_id: str | None = None) -> Path:
        with self._download_lock:
            if hf_repo_id:
                self.settings.image_hf_repo_id = hf_repo_id
            local_dir = self.settings.image_model_path
            local_dir.mkdir(parents=True, exist_ok=True)
            with self._lock:
                self._download_in_progress = True
                self._download_started_at = time.time()
                self._download_finished_at = None
                self._last_download_error = None
            print(
                f"Downloading image model snapshot {self.settings.image_hf_repo_id} to {local_dir}...",
                flush=True,
            )
            try:
                downloaded = snapshot_download(repo_id=self.settings.image_hf_repo_id, local_dir=local_dir)
            except Exception as exc:
                with self._lock:
                    self._download_in_progress = False
                    self._download_finished_at = time.time()
                    self._last_download_error = str(exc)
                print(f"Image model download failed: {exc}", flush=True)
                raise
            with self._lock:
                self._download_in_progress = False
                self._download_finished_at = time.time()
            print(f"Image model snapshot ready at {downloaded}", flush=True)
            return Path(downloaded)

    def load(
        self,
        *,
        model_id: str | None = None,
        hf_repo_id: str | None = None,
        download_if_missing: bool = True,
    ) -> LocalImageStatus:
        needs_download = False
        desired_model = model_id or self.settings.image_model_id
        with self._lock:
            if self._backend is not None and self._loaded_model == desired_model:
                self._last_used_at = time.time()
                return self.status()

            if self._backend is not None:
                self._backend.close()
                self._backend = None
                self._loaded_model = None
                self._last_used_at = None

            if hf_repo_id:
                self.settings.image_hf_repo_id = hf_repo_id

            if not self._is_downloaded():
                if not download_if_missing:
                    raise ImageNotDownloadedError(self.settings.image_model_path)
                needs_download = True

        if needs_download:
            self.download()

        with self._lock:
            if self._backend is not None and self._loaded_model == desired_model:
                self._last_used_at = time.time()
                return self.status()

            if self._backend is not None:
                self._backend.close()
                self._backend = None
                self._loaded_model = None
                self._last_used_at = None

            if not self._is_downloaded():
                raise ImageNotDownloadedError(self.settings.image_model_path)

            self._backend = self._backend_factory(self.settings.image_model_path, self.settings)
            self._loaded_model = desired_model
            self._last_used_at = time.time()
            return self.status()

    def unload(self) -> LocalImageStatus:
        with self._lock:
            if self._backend is not None:
                self._backend.close()
            self._backend = None
            self._loaded_model = None
            self._last_used_at = None
            return self.status()

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
    ) -> GeneratedImage:
        image = self.backend.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        self._last_used_at = time.time()
        return image

    def _is_downloaded(self) -> bool:
        return (self.settings.image_model_path / "model_index.json").exists()

    def _unload_if_idle_locked(self) -> None:
        if (
            self._backend is None
            or self.settings.image_idle_unload_seconds <= 0
            or self._last_used_at is None
        ):
            return
        if time.time() - self._last_used_at > self.settings.image_idle_unload_seconds:
            self._backend.close()
            self._backend = None
            self._loaded_model = None
            self._last_used_at = None

    @staticmethod
    def _default_backend_factory(model_path: Path, settings: Settings) -> ImageBackend:
        return DiffusersImageBackend(model_path=model_path, settings=settings)


def parse_image_size(size: str) -> tuple[int, int]:
    try:
        raw_width, raw_height = size.lower().split("x", maxsplit=1)
        width = int(raw_width)
        height = int(raw_height)
    except Exception as exc:
        raise ValueError("size must use WIDTHxHEIGHT format") from exc
    if width <= 0 or height <= 0:
        raise ValueError("size dimensions must be positive")
    return width, height


def _resolve_device(torch_module, configured: str) -> str:
    if configured != "auto":
        return configured
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _torch_dtype(torch_module, configured: str):
    mapping = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    try:
        return mapping[configured.lower()]
    except KeyError as exc:
        raise RuntimeError(f"unsupported image_torch_dtype: {configured}") from exc
