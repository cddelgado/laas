from __future__ import annotations

import base64
import mimetypes
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from huggingface_hub import hf_hub_download

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
        hf_repo_id: str | None = None,
        high_noise_filename: str | None = None,
        low_noise_filename: str | None = None,
        vae_filename: str | None = None,
    ) -> Path:
        with self._download_lock:
            if hf_repo_id:
                self.settings.video_generation_hf_repo_id = hf_repo_id
            if high_noise_filename:
                self.settings.video_generation_high_noise_filename = high_noise_filename
            if low_noise_filename:
                self.settings.video_generation_low_noise_filename = low_noise_filename
            if vae_filename:
                self.settings.video_generation_vae_filename = vae_filename

            local_dir = self.settings.video_generation_model_path
            local_dir.mkdir(parents=True, exist_ok=True)
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
            print(f"Video generation assets ready at {local_dir}", flush=True)
            return local_dir

    def load(
        self,
        *,
        model_id: str | None = None,
        hf_repo_id: str | None = None,
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
            if hf_repo_id:
                self.settings.video_generation_hf_repo_id = hf_repo_id
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
        return all(path.exists() for path in self._asset_paths())

    def _asset_filenames(self) -> tuple[str, str, str]:
        return (
            self.settings.video_generation_high_noise_filename,
            self.settings.video_generation_low_noise_filename,
            self.settings.video_generation_vae_filename,
        )

    def _asset_paths(self) -> tuple[Path, Path, Path]:
        return (
            self.settings.video_generation_high_noise_path,
            self.settings.video_generation_low_noise_path,
            self.settings.video_generation_vae_path,
        )

    def _first_missing_asset(self) -> tuple[Path, str]:
        assets = (
            (self.settings.video_generation_high_noise_path, "high_noise"),
            (self.settings.video_generation_low_noise_path, "low_noise"),
            (self.settings.video_generation_vae_path, "vae"),
        )
        for path, name in assets:
            if not path.exists():
                return path, name
        return self.settings.video_generation_model_path, "model"

    def _status_snapshot(self, *, downloaded: bool) -> LocalVideoGenerationStatus:
        return LocalVideoGenerationStatus(
            configured_model=self.settings.video_generation_model_id,
            loaded_model=self._loaded_model,
            is_loaded=self._backend is not None,
            model_path=str(self.settings.video_generation_model_path),
            downloaded=downloaded,
            hf_repo_id=self.settings.video_generation_hf_repo_id,
            high_noise_filename=self.settings.video_generation_high_noise_filename,
            low_noise_filename=self.settings.video_generation_low_noise_filename,
            vae_filename=self.settings.video_generation_vae_filename,
            high_noise_path=str(self.settings.video_generation_high_noise_path),
            low_noise_path=str(self.settings.video_generation_low_noise_path),
            vae_path=str(self.settings.video_generation_vae_path),
            default_size=self.settings.video_generation_default_size,
            default_seconds=self.settings.video_generation_default_seconds,
            default_fps=self.settings.video_generation_default_fps,
            num_inference_steps=self.settings.video_generation_num_inference_steps,
            guidance_scale=self.settings.video_generation_guidance_scale,
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
        _ = model_path, settings
        raise RuntimeError(
            "Wan2.2 GGUF video assets are configured, but no local video runner backend is available yet. "
            "Use a VideoManager backend_factory integration such as ComfyUI-GGUF or a future native Wan runner."
        )


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
