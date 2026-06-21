from __future__ import annotations

import io
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download

from .schemas import LocalImageEditStatus, LocalImageStatus
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
        output = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator,
        )
        result = output.images[0].convert("RGB").resize((width, height))
        base = image.convert("RGB").resize((width, height))
        mask = mask_image.convert("L").resize((width, height))
        try:
            from PIL import Image

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
                output_dir=str(self.settings.resolved_image_output_dir),
                output_retention_seconds=self.settings.image_output_retention_seconds,
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
        with self._lock:
            self._unload_if_idle_locked()
            return LocalImageEditStatus(
                configured_model=self.settings.image_edit_model_id,
                loaded_model=self._loaded_model,
                is_loaded=self._backend is not None,
                model_path=str(self.settings.image_edit_model_path),
                downloaded=self._is_downloaded(),
                default_size=self.settings.image_edit_default_size,
                num_inference_steps=self.settings.image_edit_num_inference_steps,
                guidance_scale=self.settings.image_edit_guidance_scale,
                strength=self.settings.image_edit_strength,
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
            )

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
        self._last_used_at = time.time()
        return edited

    def _is_downloaded(self) -> bool:
        return (self.settings.image_edit_model_path / "model_index.json").exists()

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


def save_image_output(*, content: bytes, output_dir: Path, image_id: str | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = image_id or f"img-{int(time.time())}-{secrets.token_hex(8)}.png"
    if not filename.endswith(".png"):
        filename = f"{filename}.png"
    path = output_dir / Path(filename).name
    path.write_bytes(content)
    return path


def cleanup_image_outputs(*, output_dir: Path, retention_seconds: int) -> None:
    if retention_seconds <= 0 or not output_dir.exists():
        return
    cutoff = time.time() - retention_seconds
    for path in output_dir.glob("*.png"):
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
