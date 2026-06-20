from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from huggingface_hub import hf_hub_download

from .schemas import LocalTranscriptionStatus
from .settings import Settings


@dataclass
class TranscriptionSegment:
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float | None = None


@dataclass
class TranscriptionResult:
    text: str
    segments: list[TranscriptionSegment]
    language: str | None = None
    duration: float | None = None


class TranscriptionNotDownloadedError(RuntimeError):
    def __init__(self, path: Path, asset: str = "stt_model") -> None:
        self.path = path
        self.asset = asset
        super().__init__(f"{asset} file is not downloaded: {path}")


class TranscriptionBackend:
    def transcribe(
        self,
        *,
        media_path: Path,
        language: str | None,
        prompt: str | None,
        temperature: float | None,
        translate: bool,
    ) -> TranscriptionResult:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class WhisperCppBackend(TranscriptionBackend):
    def __init__(self, *, model_path: Path, n_threads: int | None = None, ffmpeg_path: str = "ffmpeg") -> None:
        try:
            from pywhispercpp.model import Model
        except Exception as exc:
            raise RuntimeError("pywhispercpp is required: pip install -e .[stt]") from exc

        params = {"print_progress": False, "print_realtime": False, "print_timestamps": False}
        if n_threads:
            params["n_threads"] = n_threads
        self._model = Model(str(model_path), redirect_whispercpp_logs_to=None, **params)
        self._ffmpeg_path = ffmpeg_path

    def transcribe(
        self,
        *,
        media_path: Path,
        language: str | None,
        prompt: str | None,
        temperature: float | None,
        translate: bool,
    ) -> TranscriptionResult:
        params: dict[str, object] = {
            "translate": translate,
            "print_progress": False,
            "print_realtime": False,
            "print_timestamps": False,
        }
        if language:
            params["language"] = language
        if prompt:
            params["initial_prompt"] = prompt
        if temperature is not None:
            params["temperature"] = temperature

        prepared_path = _prepare_media_for_whisper(media_path, ffmpeg_path=self._ffmpeg_path)
        try:
            raw_segments = self._model.transcribe(str(prepared_path), **params)
        finally:
            if prepared_path != media_path:
                prepared_path.unlink(missing_ok=True)
        segments = [_segment_from_whisper(index, segment) for index, segment in enumerate(raw_segments)]
        return TranscriptionResult(
            text="".join(segment.text for segment in segments).strip(),
            segments=segments,
            language=language,
            duration=segments[-1].end if segments else 0.0,
        )

    def close(self) -> None:
        self._model = None


TranscriptionBackendFactory = Callable[[Path, Settings], TranscriptionBackend]


class TranscriptionManager:
    def __init__(self, settings: Settings, backend_factory: TranscriptionBackendFactory | None = None) -> None:
        self.settings = settings
        self._backend_factory = backend_factory or self._default_backend_factory
        self._backend: TranscriptionBackend | None = None
        self._loaded_model: str | None = None
        self._last_used_at: float | None = None
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    @property
    def backend(self) -> TranscriptionBackend:
        with self._lock:
            self._unload_if_idle_locked()
            if self._backend is None:
                self.load(download_if_missing=self.settings.stt_auto_download)
            assert self._backend is not None
            self._last_used_at = time.time()
            return self._backend

    def status(self) -> LocalTranscriptionStatus:
        with self._lock:
            self._unload_if_idle_locked()
            return LocalTranscriptionStatus(
                configured_model=self.settings.stt_model_id,
                loaded_model=self._loaded_model,
                is_loaded=self._backend is not None,
                model_path=str(self.settings.stt_model_path),
                downloaded=self.settings.stt_model_path.exists(),
                default_language=self.settings.stt_default_language,
                n_threads=self.settings.stt_n_threads,
                idle_unload_seconds=self.settings.stt_idle_unload_seconds,
                last_used_at=self._last_used_at,
            )

    def download_file(self, *, repo_id: str, filename: str) -> Path:
        local_dir = self.settings.model_dir / repo_id.replace("/", "__")
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        return Path(downloaded)

    def download(self, *, hf_repo_id: str | None = None, filename: str | None = None) -> Path:
        return self.download_file(
            repo_id=hf_repo_id or self.settings.stt_hf_repo_id,
            filename=filename or self.settings.stt_model_filename,
        )

    def missing_required_paths(self) -> list[tuple[str, Path]]:
        if not self.settings.stt_model_path.exists():
            return [("stt_model", self.settings.stt_model_path)]
        return []

    def load(
        self,
        *,
        model_id: str | None = None,
        hf_repo_id: str | None = None,
        filename: str | None = None,
        download_if_missing: bool = True,
    ) -> LocalTranscriptionStatus:
        with self._lock:
            desired_model = model_id or self.settings.stt_model_id
            if self._backend is not None and self._loaded_model == desired_model:
                self._last_used_at = time.time()
                return self.status()

            if self._backend is not None:
                self.unload()

            if hf_repo_id:
                self.settings.stt_hf_repo_id = hf_repo_id
            if filename:
                self.settings.stt_model_filename = filename

            missing = self.missing_required_paths()
            if missing and not download_if_missing:
                asset, path = missing[0]
                raise TranscriptionNotDownloadedError(path, asset=asset)
            if missing:
                self.download()

            remaining = self.missing_required_paths()
            if remaining:
                asset, path = remaining[0]
                raise TranscriptionNotDownloadedError(path, asset=asset)

            self._backend = self._backend_factory(self.settings.stt_model_path, self.settings)
            self._loaded_model = desired_model
            self._last_used_at = time.time()
            return self.status()

    def unload(self) -> LocalTranscriptionStatus:
        with self._lock:
            if self._backend is not None:
                self._backend.close()
            self._backend = None
            self._loaded_model = None
            self._last_used_at = None
            return self.status()

    def transcribe(
        self,
        *,
        media_path: Path,
        language: str | None,
        prompt: str | None,
        temperature: float | None,
        translate: bool,
    ) -> TranscriptionResult:
        result = self.backend.transcribe(
            media_path=media_path,
            language=language or self.settings.stt_default_language,
            prompt=prompt,
            temperature=temperature,
            translate=translate,
        )
        self._last_used_at = time.time()
        return result

    def _unload_if_idle_locked(self) -> None:
        if (
            self._backend is None
            or self.settings.stt_idle_unload_seconds <= 0
            or self._last_used_at is None
        ):
            return
        if time.time() - self._last_used_at > self.settings.stt_idle_unload_seconds:
            self._backend.close()
            self._backend = None
            self._loaded_model = None
            self._last_used_at = None

    @staticmethod
    def _default_backend_factory(model_path: Path, settings: Settings) -> TranscriptionBackend:
        return WhisperCppBackend(
            model_path=model_path,
            n_threads=settings.stt_n_threads,
            ffmpeg_path=settings.tts_ffmpeg_path,
        )


def transcription_to_response(result: TranscriptionResult, response_format: str, *, task: str) -> dict | str:
    fmt = response_format or "json"
    if fmt == "json":
        return {"text": result.text}
    if fmt == "verbose_json":
        return {
            "task": task,
            "language": result.language,
            "duration": result.duration,
            "text": result.text,
            "segments": [
                {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "avg_logprob": segment.avg_logprob,
                }
                for segment in result.segments
            ],
        }
    if fmt == "text":
        return result.text
    if fmt == "srt":
        return _segments_to_srt(result.segments)
    if fmt == "vtt":
        return _segments_to_vtt(result.segments)
    raise ValueError(f"unsupported transcription response_format: {response_format}")


def _segment_from_whisper(index: int, segment: object) -> TranscriptionSegment:
    probability = getattr(segment, "probability", math.nan)
    return TranscriptionSegment(
        id=index,
        start=getattr(segment, "t0") / 100.0,
        end=getattr(segment, "t1") / 100.0,
        text=getattr(segment, "text").strip(),
        avg_logprob=None if math.isnan(probability) else math.log(max(probability, 1e-12)),
    )


def _prepare_media_for_whisper(media_path: Path, *, ffmpeg_path: str) -> Path:
    executable = _resolve_ffmpeg_path(ffmpeg_path)
    if not executable:
        return media_path

    fd, raw_path = tempfile.mkstemp(suffix=".wav")
    os_path = Path(raw_path)
    os.close(fd)
    try:
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(media_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(os_path),
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception:
        os_path.unlink(missing_ok=True)
        return media_path
    return os_path


def _resolve_ffmpeg_path(ffmpeg_path: str) -> str | None:
    configured = Path(ffmpeg_path)
    if configured.is_file():
        return str(configured)
    return shutil.which(ffmpeg_path)


def _segments_to_srt(segments: list[TranscriptionSegment]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n{_format_srt_time(segment.start)} --> {_format_srt_time(segment.end)}\n{segment.text}\n"
        )
    return "\n".join(blocks).strip() + ("\n" if blocks else "")


def _segments_to_vtt(segments: list[TranscriptionSegment]) -> str:
    lines = ["WEBVTT", ""]
    for segment in segments:
        lines.extend([f"{_format_vtt_time(segment.start)} --> {_format_vtt_time(segment.end)}", segment.text, ""])
    return "\n".join(lines)


def _format_srt_time(seconds: float) -> str:
    hours, minutes, whole_seconds, milliseconds = _split_time(seconds)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02},{milliseconds:03}"


def _format_vtt_time(seconds: float) -> str:
    hours, minutes, whole_seconds, milliseconds = _split_time(seconds)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02}.{milliseconds:03}"


def _split_time(seconds: float) -> tuple[int, int, int, int]:
    total_ms = max(0, int(round(seconds * 1000)))
    milliseconds = total_ms % 1000
    total_seconds = total_ms // 1000
    whole_seconds = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return hours, minutes, whole_seconds, milliseconds
