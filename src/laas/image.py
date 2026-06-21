from __future__ import annotations

import io
import inspect
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download

from .schemas import LocalImageEditStatus, LocalImageStatus
from .settings import Settings

IMAGE_OUTPUT_FORMATS = {"png", "jpeg", "webp"}
IMAGE_OUTPUT_MEDIA_TYPES = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


@dataclass
class GeneratedImage:
    content: bytes
    media_type: str = "image/png"


class ImageNotDownloadedError(RuntimeError):
    def __init__(self, path: Path, asset: str = "image_model") -> None:
        self.path = path
        self.asset = asset
        super().__init__(f"{asset} is not downloaded: {path}")


class ImageParameterError(ValueError):
    def __init__(self, message: str, *, param: str) -> None:
        self.param = param
        super().__init__(message)


@dataclass
class ImageGenerationOptions:
    prompt: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float


@dataclass
class ImageEditOptions:
    prompt: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    strength: float


@dataclass
class ImageVariationOptions:
    prompt: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    strength: float


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

    def variation(
        self,
        *,
        prompt: str,
        image,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
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

        self.settings = settings
        self._model_path = model_path
        dtype = _torch_dtype(torch, settings.image_torch_dtype)
        self._torch_dtype = dtype
        self._device = _resolve_device(torch, settings.image_device)
        self._load_kwargs = {"torch_dtype": dtype}
        if settings.image_torch_dtype.lower() in {"float16", "fp16"}:
            self._load_kwargs["variant"] = "fp16"
        self._pipe = AutoPipelineForText2Image.from_pretrained(str(model_path), **self._load_kwargs)
        self._image_to_image_pipe = None
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

    def variation(
        self,
        *,
        prompt: str,
        image,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        seed: int | None,
    ) -> GeneratedImage:
        generator = None
        if seed is not None:
            try:
                import torch

                generator = torch.Generator(device=self._device).manual_seed(seed)
            except Exception:
                generator = None
        pipe = self._img2img_pipe()
        output = pipe(
            prompt=prompt,
            image=image.resize((width, height)),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator,
        )
        result = output.images[0]
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        return GeneratedImage(content=buffer.getvalue(), media_type="image/png")

    def close(self) -> None:
        self._image_to_image_pipe = None
        self._pipe = None

    def _img2img_pipe(self):
        if self._image_to_image_pipe is not None:
            return self._image_to_image_pipe
        try:
            from diffusers import AutoPipelineForImage2Image
        except Exception as exc:
            raise RuntimeError("diffusers image variation support is required: pip install -e .[image]") from exc

        if hasattr(AutoPipelineForImage2Image, "from_pipe"):
            pipe = AutoPipelineForImage2Image.from_pipe(self._pipe)
        else:
            pipe = AutoPipelineForImage2Image.from_pretrained(str(self._model_path), **self._load_kwargs)
        if self._device == "cuda":
            pipe.to("cuda")
        elif self._device == "mps":
            pipe.to("mps")
        else:
            pipe.to("cpu")
        self._image_to_image_pipe = pipe
        return pipe


class ImageEditBackend:
    def edit(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        image,
        mask_image,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        seed: int | None,
    ) -> GeneratedImage:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class DiffusersImageEditBackend(ImageEditBackend):
    def __init__(self, *, model_path: Path, settings: Settings) -> None:
        try:
            import torch
            from diffusers import AutoPipelineForInpainting
        except Exception as exc:
            raise RuntimeError("diffusers image edit support is required: pip install -e .[image]") from exc

        self.settings = settings
        dtype = _torch_dtype(torch, settings.image_torch_dtype)
        self._device = _resolve_device(torch, settings.image_device)
        load_kwargs = {"torch_dtype": dtype}
        if settings.image_torch_dtype.lower() in {"float16", "fp16"}:
            load_kwargs["variant"] = "fp16"
        self._pipe = AutoPipelineForInpainting.from_pretrained(str(model_path), **load_kwargs)
        if self._device == "cuda":
            self._pipe.to("cuda")
        elif self._device == "mps":
            self._pipe.to("mps")
        else:
            self._pipe.to("cpu")
        self._supports_padding_mask_crop = "padding_mask_crop" in inspect.signature(self._pipe.__call__).parameters

    def edit(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        image,
        mask_image,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        seed: int | None,
    ) -> GeneratedImage:
        generator = None
        if seed is not None:
            try:
                import torch

                generator = torch.Generator(device=self._device).manual_seed(seed)
            except Exception:
                generator = None
        pipe_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "mask_image": mask_image,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "generator": generator,
        }
        if self._supports_padding_mask_crop and self.settings.image_edit_padding_mask_crop is not None:
            pipe_kwargs["padding_mask_crop"] = self.settings.image_edit_padding_mask_crop
        output = self._pipe(**pipe_kwargs)
        result = output.images[0].convert("RGB").resize((width, height))
        base = image.convert("RGB").resize((width, height))
        mask = mask_image.convert("L").resize((width, height))
        try:
            from PIL import Image, ImageFilter

            blur_radius = max(0, self.settings.image_edit_composite_blur_radius)
            if blur_radius:
                mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            result = Image.composite(result, base, mask)
        except Exception:
            pass
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        return GeneratedImage(content=buffer.getvalue(), media_type="image/png")

    def close(self) -> None:
        self._pipe = None


ImageBackendFactory = Callable[[Path, Settings], ImageBackend]
ImageEditBackendFactory = Callable[[Path, Settings], ImageEditBackend]


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
        self._active_jobs = 0
        self._current_operation: str | None = None
        self._last_job_started_at: float | None = None
        self._last_job_finished_at: float | None = None
        self._last_job_error: str | None = None
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
        if not self._lock.acquire(blocking=False):
            return self._status_snapshot(downloaded=self._is_downloaded())
        try:
            self._unload_if_idle_locked()
            return self._status_snapshot(downloaded=self._is_downloaded())
        finally:
            self._lock.release()

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
        self._start_job("generation")
        try:
            image = self.backend.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        except Exception as exc:
            self._finish_job(error=exc)
            raise
        self._finish_job()
        return image

    def variation(
        self,
        *,
        prompt: str,
        image,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        seed: int | None,
    ) -> GeneratedImage:
        self._start_job("variation")
        try:
            varied = self.backend.variation(
                prompt=prompt,
                image=image,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                seed=seed,
            )
        except Exception as exc:
            self._finish_job(error=exc)
            raise
        self._finish_job()
        return varied

    def _is_downloaded(self) -> bool:
        return (self.settings.image_model_path / "model_index.json").exists()

    def _status_snapshot(self, *, downloaded: bool) -> LocalImageStatus:
        return LocalImageStatus(
            configured_model=self.settings.image_model_id,
            loaded_model=self._loaded_model,
            is_loaded=self._backend is not None,
            model_path=str(self.settings.image_model_path),
            downloaded=downloaded,
            default_size=self.settings.image_default_size,
            num_inference_steps=self.settings.image_num_inference_steps,
            guidance_scale=self.settings.image_guidance_scale,
            device=self.settings.image_device,
            torch_dtype=self.settings.image_torch_dtype,
            output_dir=str(self.settings.resolved_image_output_dir),
            output_retention_seconds=self.settings.image_output_retention_seconds,
            idle_unload_seconds=self.settings.image_idle_unload_seconds,
            last_used_at=self._last_used_at,
            download_in_progress=self._download_in_progress,
            download_started_at=self._download_started_at,
            download_finished_at=self._download_finished_at,
            last_download_error=self._last_download_error,
            active_jobs=self._active_jobs,
            current_operation=self._current_operation,
            last_job_started_at=self._last_job_started_at,
            last_job_finished_at=self._last_job_finished_at,
            last_job_error=self._last_job_error,
        )

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

    def _start_job(self, operation: str) -> None:
        with self._lock:
            now = time.time()
            self._active_jobs += 1
            self._current_operation = operation
            self._last_job_started_at = now
            self._last_job_error = None

    def _finish_job(self, *, error: BaseException | None = None) -> None:
        with self._lock:
            self._active_jobs = max(0, self._active_jobs - 1)
            self._current_operation = None if self._active_jobs == 0 else self._current_operation
            self._last_job_finished_at = time.time()
            self._last_job_error = str(error) if error is not None else None
            self._last_used_at = self._last_job_finished_at

    @staticmethod
    def _default_backend_factory(model_path: Path, settings: Settings) -> ImageBackend:
        return DiffusersImageBackend(model_path=model_path, settings=settings)


class ImageEditManager:
    def __init__(self, settings: Settings, backend_factory: ImageEditBackendFactory | None = None) -> None:
        self.settings = settings
        self._backend_factory = backend_factory or self._default_backend_factory
        self._backend: ImageEditBackend | None = None
        self._loaded_model: str | None = None
        self._last_used_at: float | None = None
        self._download_in_progress = False
        self._download_started_at: float | None = None
        self._download_finished_at: float | None = None
        self._last_download_error: str | None = None
        self._active_jobs = 0
        self._current_operation: str | None = None
        self._last_job_started_at: float | None = None
        self._last_job_finished_at: float | None = None
        self._last_job_error: str | None = None
        self._lock = threading.RLock()
        self._download_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    @property
    def backend(self) -> ImageEditBackend:
        with self._lock:
            self._unload_if_idle_locked()
            if self._backend is None:
                self.load(download_if_missing=self.settings.image_edit_auto_download)
            assert self._backend is not None
            self._last_used_at = time.time()
            return self._backend

    def status(self) -> LocalImageEditStatus:
        if not self._lock.acquire(blocking=False):
            return self._status_snapshot(downloaded=self._is_downloaded())
        try:
            self._unload_if_idle_locked()
            return self._status_snapshot(downloaded=self._is_downloaded())
        finally:
            self._lock.release()

    def download(self, *, hf_repo_id: str | None = None) -> Path:
        with self._download_lock:
            if hf_repo_id:
                self.settings.image_edit_hf_repo_id = hf_repo_id
            local_dir = self.settings.image_edit_model_path
            local_dir.mkdir(parents=True, exist_ok=True)
            with self._lock:
                self._download_in_progress = True
                self._download_started_at = time.time()
                self._download_finished_at = None
                self._last_download_error = None
            print(
                f"Downloading image edit model snapshot {self.settings.image_edit_hf_repo_id} to {local_dir}...",
                flush=True,
            )
            try:
                downloaded = snapshot_download(
                    repo_id=self.settings.image_edit_hf_repo_id,
                    local_dir=local_dir,
                    allow_patterns=[
                        "model_index.json",
                        "config.json",
                        "feature_extractor/*",
                        "scheduler/*",
                        "safety_checker/config.json",
                        "safety_checker/model.fp16.safetensors",
                        "text_encoder/config.json",
                        "text_encoder/model.fp16.safetensors",
                        "tokenizer/*",
                        "unet/config.json",
                        "unet/diffusion_pytorch_model.fp16.safetensors",
                        "vae/config.json",
                        "vae/diffusion_pytorch_model.fp16.safetensors",
                    ],
                )
            except Exception as exc:
                with self._lock:
                    self._download_in_progress = False
                    self._download_finished_at = time.time()
                    self._last_download_error = str(exc)
                print(f"Image edit model download failed: {exc}", flush=True)
                raise
            with self._lock:
                self._download_in_progress = False
                self._download_finished_at = time.time()
            print(f"Image edit model snapshot ready at {downloaded}", flush=True)
            return Path(downloaded)

    def load(
        self,
        *,
        model_id: str | None = None,
        hf_repo_id: str | None = None,
        download_if_missing: bool = True,
    ) -> LocalImageEditStatus:
        needs_download = False
        desired_model = model_id or self.settings.image_edit_model_id
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
                self.settings.image_edit_hf_repo_id = hf_repo_id

            if not self._is_downloaded():
                if not download_if_missing:
                    raise ImageNotDownloadedError(self.settings.image_edit_model_path, asset="image_edit_model")
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
                raise ImageNotDownloadedError(self.settings.image_edit_model_path, asset="image_edit_model")

            self._backend = self._backend_factory(self.settings.image_edit_model_path, self.settings)
            self._loaded_model = desired_model
            self._last_used_at = time.time()
            return self.status()

    def unload(self) -> LocalImageEditStatus:
        with self._lock:
            if self._backend is not None:
                self._backend.close()
            self._backend = None
            self._loaded_model = None
            self._last_used_at = None
            return self.status()

    def edit(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        image,
        mask_image,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        seed: int | None,
    ) -> GeneratedImage:
        self._start_job("edit")
        try:
            edited = self.backend.edit(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask_image,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                seed=seed,
            )
        except Exception as exc:
            self._finish_job(error=exc)
            raise
        self._finish_job()
        return edited

    def _is_downloaded(self) -> bool:
        return (self.settings.image_edit_model_path / "model_index.json").exists()

    def _status_snapshot(self, *, downloaded: bool) -> LocalImageEditStatus:
        return LocalImageEditStatus(
            configured_model=self.settings.image_edit_model_id,
            loaded_model=self._loaded_model,
            is_loaded=self._backend is not None,
            model_path=str(self.settings.image_edit_model_path),
            downloaded=downloaded,
            default_size=self.settings.image_edit_default_size,
            num_inference_steps=self.settings.image_edit_num_inference_steps,
            guidance_scale=self.settings.image_edit_guidance_scale,
            strength=self.settings.image_edit_strength,
            padding_mask_crop=self.settings.image_edit_padding_mask_crop,
            composite_blur_radius=self.settings.image_edit_composite_blur_radius,
            device=self.settings.image_device,
            torch_dtype=self.settings.image_torch_dtype,
            output_dir=str(self.settings.resolved_image_output_dir),
            output_retention_seconds=self.settings.image_output_retention_seconds,
            idle_unload_seconds=self.settings.image_edit_idle_unload_seconds,
            last_used_at=self._last_used_at,
            download_in_progress=self._download_in_progress,
            download_started_at=self._download_started_at,
            download_finished_at=self._download_finished_at,
            last_download_error=self._last_download_error,
            active_jobs=self._active_jobs,
            current_operation=self._current_operation,
            last_job_started_at=self._last_job_started_at,
            last_job_finished_at=self._last_job_finished_at,
            last_job_error=self._last_job_error,
        )

    def _unload_if_idle_locked(self) -> None:
        if (
            self._backend is None
            or self.settings.image_edit_idle_unload_seconds <= 0
            or self._last_used_at is None
        ):
            return
        if time.time() - self._last_used_at > self.settings.image_edit_idle_unload_seconds:
            self._backend.close()
            self._backend = None
            self._loaded_model = None
            self._last_used_at = None

    def _start_job(self, operation: str) -> None:
        with self._lock:
            now = time.time()
            self._active_jobs += 1
            self._current_operation = operation
            self._last_job_started_at = now
            self._last_job_error = None

    def _finish_job(self, *, error: BaseException | None = None) -> None:
        with self._lock:
            self._active_jobs = max(0, self._active_jobs - 1)
            self._current_operation = None if self._active_jobs == 0 else self._current_operation
            self._last_job_finished_at = time.time()
            self._last_job_error = str(error) if error is not None else None
            self._last_used_at = self._last_job_finished_at

    @staticmethod
    def _default_backend_factory(model_path: Path, settings: Settings) -> ImageEditBackend:
        return DiffusersImageEditBackend(model_path=model_path, settings=settings)


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


def normalize_image_generation_options(
    *,
    prompt: str,
    size: str,
    default_steps: int,
    default_guidance_scale: float,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    quality: str | None = None,
    style: str | None = None,
    background: str | None = None,
    moderation: str | None = None,
) -> ImageGenerationOptions:
    if quality not in {None, "auto", "standard", "hd", "low", "medium", "high"}:
        raise ImageParameterError("unsupported quality value", param="quality")
    if style not in {None, "auto", "vivid", "natural"}:
        raise ImageParameterError("unsupported style value", param="style")
    if background not in {None, "auto", "transparent", "opaque"}:
        raise ImageParameterError("unsupported background value", param="background")
    if background == "transparent":
        raise ImageParameterError(
            "background=transparent is not supported by the local SDXL Turbo backend",
            param="background",
        )
    if moderation not in {None, "auto", "low"}:
        raise ImageParameterError("unsupported moderation value", param="moderation")

    width, height = parse_image_size(size)
    steps = num_inference_steps or default_steps
    if num_inference_steps is None and quality in {"hd", "high"}:
        steps = max(steps, 4)
    resolved_guidance = default_guidance_scale if guidance_scale is None else guidance_scale

    translated_prompt = prompt
    if style == "vivid":
        translated_prompt = f"{translated_prompt}, vivid color, high contrast, dramatic detail"
    elif style == "natural":
        translated_prompt = f"{translated_prompt}, natural color, realistic lighting"

    return ImageGenerationOptions(
        prompt=translated_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=resolved_guidance,
    )


def normalize_image_edit_options(
    *,
    prompt: str,
    size: str,
    default_steps: int,
    default_guidance_scale: float,
    default_strength: float,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    strength: float | None = None,
    quality: str | None = None,
    background: str | None = None,
    input_fidelity: str | None = None,
    moderation: str | None = None,
) -> ImageEditOptions:
    if quality not in {None, "auto", "standard", "low", "medium", "high"}:
        raise ImageParameterError("unsupported quality value", param="quality")
    if background == "transparent":
        raise ImageParameterError("background=transparent is not supported by the local inpainting backend", param="background")
    if background not in {None, "auto", "opaque"}:
        raise ImageParameterError("unsupported background value", param="background")
    if input_fidelity not in {None, "low", "high"}:
        raise ImageParameterError("unsupported input_fidelity value", param="input_fidelity")
    if moderation not in {None, "auto", "low"}:
        raise ImageParameterError("unsupported moderation value", param="moderation")

    width, height = parse_image_size(size)
    steps = num_inference_steps or default_steps
    if num_inference_steps is None and quality == "high":
        steps = max(steps, 35)
    resolved_guidance = default_guidance_scale if guidance_scale is None else guidance_scale
    resolved_strength = default_strength if strength is None else strength
    if not 0 < resolved_strength <= 1:
        raise ImageParameterError("strength must be greater than 0 and less than or equal to 1", param="strength")
    if input_fidelity == "high":
        resolved_strength = min(resolved_strength, 0.65)

    return ImageEditOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=resolved_guidance,
        strength=resolved_strength,
    )


def normalize_image_variation_options(
    *,
    size: str | None,
    default_size: str,
    default_prompt: str,
    default_steps: int,
    default_guidance_scale: float,
    default_strength: float,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    strength: float | None = None,
) -> ImageVariationOptions:
    resolved_size = size if size and size != "auto" else default_size
    if resolved_size not in {"256x256", "512x512", "1024x1024"}:
        raise ImageParameterError("size must be one of 256x256, 512x512, or 1024x1024", param="size")
    width, height = parse_image_size(resolved_size)
    resolved_strength = default_strength if strength is None else strength
    if not 0 < resolved_strength <= 1:
        raise ImageParameterError("strength must be greater than 0 and less than or equal to 1", param="strength")
    return ImageVariationOptions(
        prompt=default_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps or default_steps,
        guidance_scale=default_guidance_scale if guidance_scale is None else guidance_scale,
        strength=resolved_strength,
    )


def prepare_inpaint_inputs(*, image_bytes: bytes, mask_bytes: bytes | None, width: int, height: int):
    try:
        from PIL import Image, ImageOps
    except Exception as exc:
        raise RuntimeError("pillow is required for local image editing: pip install -e .[image]") from exc

    try:
        base = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGBA")
    except Exception as exc:
        raise ImageParameterError("image must be a valid PNG or JPEG file", param="image") from exc

    if mask_bytes:
        try:
            raw_mask = ImageOps.exif_transpose(Image.open(io.BytesIO(mask_bytes))).convert("RGBA")
        except Exception as exc:
            raise ImageParameterError("mask must be a valid PNG or JPEG file", param="mask") from exc
        if raw_mask.getextrema()[3] != (255, 255):
            alpha = raw_mask.getchannel("A")
            mask = Image.eval(alpha, lambda value: 255 - value)
        else:
            mask = raw_mask.convert("L")
    else:
        alpha = base.getchannel("A")
        if alpha.getextrema() == (255, 255):
            raise ImageParameterError("mask is required unless image has transparent pixels", param="mask")
        mask = Image.eval(alpha, lambda value: 255 - value)

    rgb_base = Image.new("RGB", base.size, (255, 255, 255))
    rgb_base.paste(base.convert("RGB"), mask=base.getchannel("A"))
    return (
        rgb_base.resize((width, height)),
        mask.convert("L").resize((width, height)),
    )


def prepare_variation_input(*, image_bytes: bytes, width: int, height: int):
    if len(image_bytes) > 4 * 1024 * 1024:
        raise ImageParameterError("image must be less than 4MB", param="image")
    try:
        from PIL import Image, ImageOps
    except Exception as exc:
        raise RuntimeError("pillow is required for local image variations: pip install -e .[image]") from exc

    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            if opened.format != "PNG":
                raise ImageParameterError("image must be a PNG file", param="image")
            source = ImageOps.exif_transpose(opened).convert("RGB")
    except ImageParameterError:
        raise
    except Exception as exc:
        raise ImageParameterError("image must be a valid PNG file", param="image") from exc

    if source.width != source.height:
        raise ImageParameterError("image must be square", param="image")
    return source.resize((width, height))


def normalize_image_output_format(output_format: str | None) -> str:
    resolved = (output_format or "png").lower()
    if resolved not in IMAGE_OUTPUT_FORMATS:
        raise ImageParameterError("output_format must be png, jpeg, or webp", param="output_format")
    return resolved


def encode_image_output(*, content: bytes, output_format: str, output_compression: int | None = None) -> GeneratedImage:
    output_format = normalize_image_output_format(output_format)
    if output_compression is not None and not 0 <= output_compression <= 100:
        raise ImageParameterError("output_compression must be between 0 and 100", param="output_compression")
    if output_format == "png" and output_compression is None:
        return GeneratedImage(content=content, media_type="image/png")

    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("pillow is required for local image encoding: pip install -e .[image]") from exc

    try:
        image = Image.open(io.BytesIO(content))
    except Exception as exc:
        raise RuntimeError("backend returned an invalid image") from exc

    buffer = io.BytesIO()
    if output_format == "png":
        image.save(buffer, format="PNG")
    elif output_format == "jpeg":
        quality = 95 if output_compression is None else max(1, output_compression)
        image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    else:
        quality = 95 if output_compression is None else max(1, output_compression)
        image.save(buffer, format="WEBP", quality=quality)
    return GeneratedImage(content=buffer.getvalue(), media_type=IMAGE_OUTPUT_MEDIA_TYPES[output_format])


def image_extension_for_media_type(media_type: str) -> str:
    if media_type == "image/jpeg":
        return ".jpeg"
    if media_type == "image/webp":
        return ".webp"
    return ".png"


def save_image_output(*, content: bytes, output_dir: Path, image_id: str | None = None, media_type: str = "image/png") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = image_extension_for_media_type(media_type)
    filename = image_id or f"img-{int(time.time())}-{secrets.token_hex(8)}{extension}"
    if Path(filename).suffix.lower() not in {".png", ".jpeg", ".jpg", ".webp"}:
        filename = f"{filename}{extension}"
    path = output_dir / Path(filename).name
    path.write_bytes(content)
    return path


def cleanup_image_outputs(*, output_dir: Path, retention_seconds: int) -> None:
    if retention_seconds <= 0 or not output_dir.exists():
        return
    cutoff = time.time() - retention_seconds
    for path in list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpeg")) + list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.webp")):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


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
