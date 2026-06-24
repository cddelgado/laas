from __future__ import annotations

import base64
import io
import mimetypes
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from huggingface_hub import hf_hub_download, snapshot_download

from .schemas import LocalVideoGenerationStatus
from .settings import Settings


@dataclass
class GeneratedVideo:
    content: bytes
    media_type: str = "video/mp4"


@dataclass
class VideoGenerationOptions:
    prompt: str
    width: int
    height: int
    seconds: float
    fps: int
    num_inference_steps: int
    guidance_scale: float


class VideoNotDownloadedError(RuntimeError):
    def __init__(self, path: Path, *, asset: str = "model") -> None:
        super().__init__(f"Video generation asset is not downloaded: {path}")
        self.path = path
        self.asset = asset


class VideoParameterError(ValueError):
    def __init__(self, message: str, *, param: str) -> None:
        super().__init__(message)
        self.param = param


class VideoBackend:
    def generate(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        width: int,
        height: int,
        seconds: float,
        fps: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
    ) -> GeneratedVideo:
        raise NotImplementedError

    def close(self) -> None:
        pass


VideoBackendFactory = Callable[[Path, Settings], VideoBackend]


def ensure_video_export_dependencies() -> None:
    try:
        import imageio  # noqa: F401
        import imageio_ffmpeg  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Video export requires imageio and imageio-ffmpeg. Install requirements-image.txt."
        ) from exc


WAN_DIFFUSERS_ALLOW_PATTERNS = (
    "model_index.json",
    "scheduler/*",
    "text_encoder/config.json",
    "tokenizer/*",
    "transformer/config.json",
    "vae/*",
)

WAN_DIFFUSERS_BASE_REQUIRED_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "vae/config.json",
)

WAN_DUAL_EXPERT_EXTRA_REQUIRED_FILES = ("transformer_2/config.json",)


class DiffusersWanVideoBackend(VideoBackend):
    def __init__(self, *, model_path: Path, settings: Settings) -> None:
        self.model_path = model_path
        self.settings = settings
        self._pipe = None
        self._device = None

    def _ensure_pipe(self):
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
            from diffusers import (
                AutoencoderKLWan,
                GGUFQuantizationConfig,
                WanImageToVideoPipeline,
                WanTransformer3DModel,
            )
            from transformers import UMT5EncoderModel
        except ImportError as exc:
            raise RuntimeError(
                "Wan video generation requires the image/video dependencies: diffusers, transformers, torch, "
                "pillow, safetensors, and gguf. Install the image extra or requirements-image.txt."
            ) from exc

        device = self._resolve_device(torch)
        dtype = self._resolve_dtype(torch, device=device)
        base_path = self.settings.video_generation_diffusers_model_path
        quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)
        text_encoder = UMT5EncoderModel.from_pretrained(
            str(base_path / "text_encoder"),
            gguf_file=str(self.settings.video_generation_text_encoder_path),
            torch_dtype=dtype,
        )

        architecture = normalized_video_architecture(self.settings)
        transformer = WanTransformer3DModel.from_single_file(
            str(video_transformer_path(self.settings, role="transformer")),
            quantization_config=quantization_config,
            config=str(base_path),
            subfolder="transformer",
            torch_dtype=dtype,
        )
        if hasattr(transformer, "config"):
            transformer.config.image_dim = None
        transformer_2 = None
        if architecture == "dual":
            transformer_2 = WanTransformer3DModel.from_single_file(
                str(video_transformer_path(self.settings, role="transformer_2")),
                quantization_config=quantization_config,
                config=str(base_path),
                subfolder="transformer_2",
                torch_dtype=dtype,
            )
            if hasattr(transformer_2, "config"):
                transformer_2.config.image_dim = None
        vae = AutoencoderKLWan.from_pretrained(
            str(base_path),
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            str(base_path),
            transformer=transformer,
            transformer_2=transformer_2,
            text_encoder=text_encoder,
            vae=vae,
            boundary_ratio=self.settings.video_generation_boundary_ratio,
            torch_dtype=dtype,
        )
        if device == "cuda" and self.settings.video_generation_enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        self._pipe = pipe
        self._device = device
        return pipe

    def generate(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        width: int,
        height: int,
        seconds: float,
        fps: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
    ) -> GeneratedVideo:
        ensure_video_export_dependencies()
        try:
            import torch
            from diffusers.utils import export_to_video
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Wan video generation dependencies are not installed") from exc

        pipe = self._ensure_pipe()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((width, height))
        num_frames = frames_for_duration(seconds=seconds, fps=fps)
        generator = None
        if seed is not None:
            generator_device = self._device if self._device in {"cuda", "cpu"} else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(seed)
        output = pipe(
            image=image,
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_scale_2=(
                self.settings.video_generation_guidance_scale_2
                if self.settings.video_generation_guidance_scale_2 is not None
                else guidance_scale
            ),
            generator=generator,
        ).frames[0]
        output_dir = self.settings.resolved_video_generation_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{uuid.uuid4().hex}.mp4"
        export_to_video(output, str(path), fps=fps)
        content = path.read_bytes()
        try:
            path.unlink()
        except OSError:
            pass
        return GeneratedVideo(content=content, media_type="video/mp4")

    def close(self) -> None:
        self._pipe = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _resolve_device(self, torch) -> str:
        requested = self.settings.video_generation_device.lower()
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self, torch, *, device: str):
        requested = self.settings.video_generation_torch_dtype.lower()
        if requested == "float32":
            return torch.float32
        if requested == "float16":
            return torch.float16
        if requested == "bfloat16":
            return torch.bfloat16
        if device == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32


def normalized_video_architecture(settings: Settings) -> str:
    architecture = settings.video_generation_architecture.lower().strip()
    if architecture in {"single", "ti2v", "dense"}:
        return "single"
    if architecture in {"dual", "moe", "a14b"}:
        return "dual"
    raise RuntimeError(f"unsupported video_generation_architecture: {settings.video_generation_architecture}")


def video_diffusers_required_files(settings: Settings) -> tuple[str, ...]:
    files = list(WAN_DIFFUSERS_BASE_REQUIRED_FILES)
    if normalized_video_architecture(settings) == "dual":
        files.extend(WAN_DUAL_EXPERT_EXTRA_REQUIRED_FILES)
    return tuple(files)


def video_transformer_path(settings: Settings, *, role: str) -> Path:
    architecture = normalized_video_architecture(settings)
    if architecture == "single":
        if not settings.video_generation_transformer_filename:
            raise VideoNotDownloadedError(settings.video_generation_model_path, asset="transformer")
        return settings.video_generation_transformer_path
    if role == "transformer":
        if not settings.video_generation_high_noise_filename:
            raise VideoNotDownloadedError(settings.video_generation_model_path, asset="high_noise")
        return settings.video_generation_high_noise_path
    if role == "transformer_2":
        if not settings.video_generation_low_noise_filename:
            raise VideoNotDownloadedError(settings.video_generation_model_path, asset="low_noise")
        return settings.video_generation_low_noise_path
    raise RuntimeError(f"unsupported transformer role: {role}")


class VideoManager:
    def __init__(self, settings: Settings, backend_factory: VideoBackendFactory | None = None) -> None:
        self.settings = settings
        self._backend_factory = backend_factory or self._default_backend_factory
        self._backend: VideoBackend | None = None
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
    def backend(self) -> VideoBackend:
        with self._lock:
            self._unload_if_idle_locked()
            if self._backend is None:
                self.load(download_if_missing=self.settings.video_generation_auto_download)
            assert self._backend is not None
            self._last_used_at = time.time()
            return self._backend

    def status(self) -> LocalVideoGenerationStatus:
        if not self._lock.acquire(blocking=False):
            return self._status_snapshot(downloaded=self._is_downloaded())
        try:
            self._unload_if_idle_locked()
            return self._status_snapshot(downloaded=self._is_downloaded())
        finally:
            self._lock.release()

    def download(
        self,
        *,
        architecture: str | None = None,
        hf_repo_id: str | None = None,
        diffusers_hf_repo_id: str | None = None,
        text_encoder_hf_repo_id: str | None = None,
        transformer_filename: str | None = None,
        text_encoder_filename: str | None = None,
        high_noise_filename: str | None = None,
        low_noise_filename: str | None = None,
        vae_filename: str | None = None,
    ) -> Path:
        with self._download_lock:
            if architecture:
                self.settings.video_generation_architecture = architecture
            if hf_repo_id:
                self.settings.video_generation_hf_repo_id = hf_repo_id
            if diffusers_hf_repo_id:
                self.settings.video_generation_diffusers_hf_repo_id = diffusers_hf_repo_id
            if text_encoder_hf_repo_id:
                self.settings.video_generation_text_encoder_hf_repo_id = text_encoder_hf_repo_id
            if transformer_filename:
                self.settings.video_generation_transformer_filename = transformer_filename
            if text_encoder_filename:
                self.settings.video_generation_text_encoder_filename = text_encoder_filename
            if high_noise_filename:
                self.settings.video_generation_high_noise_filename = high_noise_filename
            if low_noise_filename:
                self.settings.video_generation_low_noise_filename = low_noise_filename
            if vae_filename:
                self.settings.video_generation_vae_filename = vae_filename

            local_dir = self.settings.video_generation_model_path
            diffusers_dir = self.settings.video_generation_diffusers_model_path
            local_dir.mkdir(parents=True, exist_ok=True)
            diffusers_dir.mkdir(parents=True, exist_ok=True)
            with self._lock:
                self._download_in_progress = True
                self._download_started_at = time.time()
                self._download_finished_at = None
                self._last_download_error = None

            print(
                f"Downloading video generation assets {self.settings.video_generation_hf_repo_id} to {local_dir}...",
                flush=True,
            )
            try:
                for filename in self._asset_filenames():
                    hf_hub_download(
                        repo_id=self.settings.video_generation_hf_repo_id,
                        filename=filename,
                        local_dir=local_dir,
                    )
                if self.settings.video_generation_text_encoder_filename:
                    hf_hub_download(
                        repo_id=self.settings.video_generation_text_encoder_hf_repo_id,
                        filename=self.settings.video_generation_text_encoder_filename,
                        local_dir=self.settings.video_generation_text_encoder_model_path,
                    )
                snapshot_download(
                    repo_id=self.settings.video_generation_diffusers_hf_repo_id,
                    local_dir=diffusers_dir,
                    allow_patterns=list(WAN_DIFFUSERS_ALLOW_PATTERNS),
                )
            except Exception as exc:
                with self._lock:
                    self._download_in_progress = False
                    self._download_finished_at = time.time()
                    self._last_download_error = str(exc)
                print(f"Video generation asset download failed: {exc}", flush=True)
                raise

            with self._lock:
                self._download_in_progress = False
                self._download_finished_at = time.time()
            print(f"Video generation assets ready at {local_dir} and {diffusers_dir}", flush=True)
            return local_dir

    def load(
        self,
        *,
        model_id: str | None = None,
        architecture: str | None = None,
        hf_repo_id: str | None = None,
        diffusers_hf_repo_id: str | None = None,
        text_encoder_hf_repo_id: str | None = None,
        transformer_filename: str | None = None,
        text_encoder_filename: str | None = None,
        high_noise_filename: str | None = None,
        low_noise_filename: str | None = None,
        vae_filename: str | None = None,
        download_if_missing: bool = True,
    ) -> LocalVideoGenerationStatus:
        needs_download = False
        desired_model = model_id or self.settings.video_generation_model_id
        with self._lock:
            if self._backend is not None and self._loaded_model == desired_model:
                self._last_used_at = time.time()
                return self.status()

            self._close_locked()
            if architecture:
                self.settings.video_generation_architecture = architecture
            if hf_repo_id:
                self.settings.video_generation_hf_repo_id = hf_repo_id
            if diffusers_hf_repo_id:
                self.settings.video_generation_diffusers_hf_repo_id = diffusers_hf_repo_id
            if text_encoder_hf_repo_id:
                self.settings.video_generation_text_encoder_hf_repo_id = text_encoder_hf_repo_id
            if transformer_filename:
                self.settings.video_generation_transformer_filename = transformer_filename
            if text_encoder_filename:
                self.settings.video_generation_text_encoder_filename = text_encoder_filename
            if high_noise_filename:
                self.settings.video_generation_high_noise_filename = high_noise_filename
            if low_noise_filename:
                self.settings.video_generation_low_noise_filename = low_noise_filename
            if vae_filename:
                self.settings.video_generation_vae_filename = vae_filename

            if not self._is_downloaded():
                if not download_if_missing:
                    path, asset = self._first_missing_asset()
                    raise VideoNotDownloadedError(path, asset=asset)
                needs_download = True

        if needs_download:
            self.download()

        with self._lock:
            if self._backend is not None and self._loaded_model == desired_model:
                self._last_used_at = time.time()
                return self.status()

            self._close_locked()
            if not self._is_downloaded():
                path, asset = self._first_missing_asset()
                raise VideoNotDownloadedError(path, asset=asset)

            self._backend = self._backend_factory(self.settings.video_generation_model_path, self.settings)
            self._loaded_model = desired_model
            self._last_used_at = time.time()
            return self.status()

    def unload(self) -> LocalVideoGenerationStatus:
        with self._lock:
            self._close_locked()
            return self.status()

    def generate(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        width: int,
        height: int,
        seconds: float,
        fps: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
    ) -> GeneratedVideo:
        self._start_job("generation")
        try:
            video = self.backend.generate(
                prompt=prompt,
                image_bytes=image_bytes,
                width=width,
                height=height,
                seconds=seconds,
                fps=fps,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        except Exception as exc:
            self._finish_job(error=exc)
            raise
        self._finish_job()
        return video

    def _is_downloaded(self) -> bool:
        return all(path.exists() for path in self._asset_paths()) and self._text_encoder_downloaded() and all(
            (self.settings.video_generation_diffusers_model_path / filename).exists()
            for filename in video_diffusers_required_files(self.settings)
        )

    def _asset_filenames(self) -> tuple[str, ...]:
        names = []
        if normalized_video_architecture(self.settings) == "dual":
            names.extend(
                [
                    self.settings.video_generation_high_noise_filename,
                    self.settings.video_generation_low_noise_filename,
                ]
            )
        else:
            names.append(self.settings.video_generation_transformer_filename)
        if self.settings.video_generation_vae_filename:
            names.append(self.settings.video_generation_vae_filename)
        return tuple(name for name in names if name)

    def _asset_paths(self) -> tuple[Path, ...]:
        paths = []
        if normalized_video_architecture(self.settings) == "dual":
            paths.extend(
                [
                    self.settings.video_generation_high_noise_path,
                    self.settings.video_generation_low_noise_path,
                ]
            )
        else:
            paths.append(self.settings.video_generation_transformer_path)
        if self.settings.video_generation_vae_filename:
            paths.append(self.settings.video_generation_vae_path)
        return tuple(paths)

    def _text_encoder_downloaded(self) -> bool:
        if not self.settings.video_generation_text_encoder_filename:
            return False
        return self.settings.video_generation_text_encoder_path.exists()

    def _first_missing_asset(self) -> tuple[Path, str]:
        if normalized_video_architecture(self.settings) == "dual":
            assets = [
                (self.settings.video_generation_high_noise_path, "high_noise"),
                (self.settings.video_generation_low_noise_path, "low_noise"),
            ]
        else:
            assets = [(self.settings.video_generation_transformer_path, "transformer")]
        if self.settings.video_generation_vae_filename:
            assets.append((self.settings.video_generation_vae_path, "vae"))
        for path, name in assets:
            if not path.exists():
                return path, name
        if not self._text_encoder_downloaded():
            return self.settings.video_generation_text_encoder_path, "text_encoder"
        for filename in video_diffusers_required_files(self.settings):
            path = self.settings.video_generation_diffusers_model_path / filename
            if not path.exists():
                return path, "diffusers_base"
        return self.settings.video_generation_model_path, "model"

    def _status_snapshot(self, *, downloaded: bool) -> LocalVideoGenerationStatus:
        return LocalVideoGenerationStatus(
            configured_model=self.settings.video_generation_model_id,
            loaded_model=self._loaded_model,
            is_loaded=self._backend is not None,
            architecture=normalized_video_architecture(self.settings),
            model_path=str(self.settings.video_generation_model_path),
            diffusers_model_path=str(self.settings.video_generation_diffusers_model_path),
            downloaded=downloaded,
            hf_repo_id=self.settings.video_generation_hf_repo_id,
            diffusers_hf_repo_id=self.settings.video_generation_diffusers_hf_repo_id,
            text_encoder_hf_repo_id=self.settings.video_generation_text_encoder_hf_repo_id,
            transformer_filename=self.settings.video_generation_transformer_filename,
            text_encoder_filename=self.settings.video_generation_text_encoder_filename,
            high_noise_filename=self.settings.video_generation_high_noise_filename,
            low_noise_filename=self.settings.video_generation_low_noise_filename,
            vae_filename=self.settings.video_generation_vae_filename,
            transformer_path=(
                str(self.settings.video_generation_transformer_path)
                if self.settings.video_generation_transformer_filename
                else None
            ),
            text_encoder_path=(
                str(self.settings.video_generation_text_encoder_path)
                if self.settings.video_generation_text_encoder_filename
                else None
            ),
            high_noise_path=(
                str(self.settings.video_generation_high_noise_path)
                if self.settings.video_generation_high_noise_filename
                else None
            ),
            low_noise_path=(
                str(self.settings.video_generation_low_noise_path)
                if self.settings.video_generation_low_noise_filename
                else None
            ),
            vae_path=str(self.settings.video_generation_vae_path) if self.settings.video_generation_vae_filename else None,
            default_size=self.settings.video_generation_default_size,
            default_seconds=self.settings.video_generation_default_seconds,
            default_fps=self.settings.video_generation_default_fps,
            num_inference_steps=self.settings.video_generation_num_inference_steps,
            guidance_scale=self.settings.video_generation_guidance_scale,
            guidance_scale_2=self.settings.video_generation_guidance_scale_2,
            boundary_ratio=self.settings.video_generation_boundary_ratio,
            device=self.settings.video_generation_device,
            torch_dtype=self.settings.video_generation_torch_dtype,
            enable_model_cpu_offload=self.settings.video_generation_enable_model_cpu_offload,
            output_dir=str(self.settings.resolved_video_generation_output_dir),
            output_retention_seconds=self.settings.video_generation_output_retention_seconds,
            idle_unload_seconds=self.settings.video_generation_idle_unload_seconds,
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

    def _close_locked(self) -> None:
        if self._backend is not None:
            self._backend.close()
        self._backend = None
        self._loaded_model = None
        self._last_used_at = None

    def _unload_if_idle_locked(self) -> None:
        if (
            self._backend is None
            or self.settings.video_generation_idle_unload_seconds <= 0
            or self._last_used_at is None
        ):
            return
        if time.time() - self._last_used_at > self.settings.video_generation_idle_unload_seconds:
            self._close_locked()

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
    def _default_backend_factory(model_path: Path, settings: Settings) -> VideoBackend:
        return DiffusersWanVideoBackend(model_path=model_path, settings=settings)


def parse_video_size(size: str) -> tuple[int, int]:
    try:
        raw_width, raw_height = size.lower().split("x", maxsplit=1)
        width = int(raw_width)
        height = int(raw_height)
    except Exception as exc:
        raise ValueError("size must use WIDTHxHEIGHT format") from exc
    if width <= 0 or height <= 0:
        raise ValueError("size dimensions must be positive")
    return width, height


def normalize_video_generation_options(
    *,
    prompt: str,
    size: str,
    seconds: float | None,
    fps: int | None,
    default_seconds: float,
    default_fps: int,
    default_steps: int,
    default_guidance_scale: float,
    num_inference_steps: int | None,
    guidance_scale: float | None,
) -> VideoGenerationOptions:
    width, height = parse_video_size(size)
    resolved_seconds = default_seconds if seconds is None else seconds
    resolved_fps = default_fps if fps is None else fps
    if resolved_seconds <= 0 or resolved_seconds > 30:
        raise VideoParameterError("seconds must be greater than 0 and no more than 30", param="seconds")
    if resolved_fps <= 0 or resolved_fps > 60:
        raise VideoParameterError("fps must be greater than 0 and no more than 60", param="fps")
    return VideoGenerationOptions(
        prompt=prompt,
        width=width,
        height=height,
        seconds=resolved_seconds,
        fps=resolved_fps,
        num_inference_steps=num_inference_steps or default_steps,
        guidance_scale=default_guidance_scale if guidance_scale is None else guidance_scale,
    )


def frames_for_duration(*, seconds: float, fps: int) -> int:
    frames = max(1, int(round(seconds * fps)) + 1)
    return ((frames - 1) // 4) * 4 + 1


def encode_video_output(video: GeneratedVideo) -> str:
    return base64.b64encode(video.content).decode("ascii")


def save_video_output(*, content: bytes, output_dir: Path, media_type: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = mimetypes.guess_extension(media_type) or ".mp4"
    if suffix == ".m4v":
        suffix = ".mp4"
    path = output_dir / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(content)
    return path


def cleanup_video_outputs(*, output_dir: Path, retention_seconds: int) -> None:
    if retention_seconds <= 0 or not output_dir.exists():
        return
    cutoff = time.time() - retention_seconds
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue
