from __future__ import annotations

import io
import json
import shutil
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import hf_hub_download

from .schemas import LocalAudioStatus
from .settings import Settings


@dataclass
class SynthesizedSpeech:
    samples: Any
    sample_rate: int


class AudioNotDownloadedError(RuntimeError):
    def __init__(self, path: Path, asset: str) -> None:
        self.path = path
        self.asset = asset
        super().__init__(f"{asset} file is not downloaded: {path}")


class AudioEncoderMissingError(RuntimeError):
    def __init__(self, response_format: str, encoder: str) -> None:
        self.response_format = response_format
        self.encoder = encoder
        super().__init__(f"{response_format} output requires {encoder}, but it was not found")


class AudioEncodingError(RuntimeError):
    def __init__(self, response_format: str, message: str) -> None:
        self.response_format = response_format
        super().__init__(f"failed to encode {response_format} audio: {message}")


class AudioBackend:
    def synthesize(
        self,
        *,
        text: str,
        voice: str,
        speed: float,
        lang: str,
        is_phonemes: bool,
        trim: bool,
    ) -> SynthesizedSpeech:
        raise NotImplementedError

    def voices(self) -> list[str]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class KokoroOnnxBackend(AudioBackend):
    def __init__(self, *, model_path: Path, voices_path: Path) -> None:
        try:
            from kokoro_onnx import Kokoro
        except Exception as exc:
            raise RuntimeError("kokoro-onnx is required: pip install -e .[tts]") from exc

        self._tts = Kokoro(str(model_path), str(voices_path))

    def synthesize(
        self,
        *,
        text: str,
        voice: str,
        speed: float,
        lang: str,
        is_phonemes: bool,
        trim: bool,
    ) -> SynthesizedSpeech:
        samples, sample_rate = self._tts.create(
            text,
            voice=voice,
            speed=speed,
            lang=lang,
            is_phonemes=is_phonemes,
            trim=trim,
        )
        return SynthesizedSpeech(samples=samples, sample_rate=sample_rate)

    def voices(self) -> list[str]:
        voices = self._tts.get_voices()
        if isinstance(voices, dict):
            return sorted(str(name) for name in voices)
        return sorted(str(name) for name in voices)

    def close(self) -> None:
        self._tts = None


AudioBackendFactory = Callable[[Path, Path, Settings], AudioBackend]


class AudioManager:
    def __init__(self, settings: Settings, backend_factory: AudioBackendFactory | None = None) -> None:
        self.settings = settings
        self._backend_factory = backend_factory or self._default_backend_factory
        self._backend: AudioBackend | None = None
        self._loaded_model: str | None = None
        self._last_used_at: float | None = None
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    @property
    def backend(self) -> AudioBackend:
        with self._lock:
            self._unload_if_idle_locked()
            if self._backend is None:
                self.load(download_if_missing=self.settings.tts_auto_download)
            assert self._backend is not None
            self._last_used_at = time.time()
            return self._backend

    def status(self) -> LocalAudioStatus:
        with self._lock:
            self._unload_if_idle_locked()
            return LocalAudioStatus(
                configured_model=self.settings.tts_model_id,
                loaded_model=self._loaded_model,
                is_loaded=self._backend is not None,
                model_path=str(self.settings.tts_model_path),
                model_downloaded=self.settings.tts_model_path.exists(),
                voices_path=str(self.settings.tts_voices_path),
                voices_downloaded=self.settings.tts_voices_path.exists(),
                default_voice=self.settings.tts_default_voice,
                default_lang=self.settings.tts_default_lang,
                supported_formats=available_audio_formats(self.settings.tts_ffmpeg_path),
                ffmpeg_path=resolve_ffmpeg_path(self.settings.tts_ffmpeg_path),
                ffmpeg_available=resolve_ffmpeg_path(self.settings.tts_ffmpeg_path) is not None,
                idle_unload_seconds=self.settings.tts_idle_unload_seconds,
                last_used_at=self._last_used_at,
            )

    def download_file(self, *, repo_id: str, filename: str) -> Path:
        local_dir = self.settings.model_dir / repo_id.replace("/", "__")
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        return Path(downloaded)

    def download_configured_assets(self) -> list[Path]:
        return [
            self.download_file(repo_id=self.settings.tts_hf_repo_id, filename=self.settings.tts_model_filename),
            self.download_file(repo_id=self.settings.tts_hf_repo_id, filename=self.settings.tts_voices_filename),
        ]

    def missing_required_paths(self) -> list[tuple[str, Path]]:
        missing: list[tuple[str, Path]] = []
        if not self.settings.tts_model_path.exists():
            missing.append(("tts_model", self.settings.tts_model_path))
        if not self.settings.tts_voices_path.exists():
            missing.append(("tts_voices", self.settings.tts_voices_path))
        return missing

    def load(
        self,
        *,
        model_id: str | None = None,
        hf_repo_id: str | None = None,
        model_filename: str | None = None,
        voices_filename: str | None = None,
        download_if_missing: bool = True,
    ) -> LocalAudioStatus:
        with self._lock:
            desired_model = model_id or self.settings.tts_model_id
            if self._backend is not None and self._loaded_model == desired_model:
                self._last_used_at = time.time()
                return self.status()

            if self._backend is not None:
                self.unload()

            if hf_repo_id:
                self.settings.tts_hf_repo_id = hf_repo_id
            if model_filename:
                self.settings.tts_model_filename = model_filename
            if voices_filename:
                self.settings.tts_voices_filename = voices_filename

            missing = self.missing_required_paths()
            if missing and not download_if_missing:
                asset, path = missing[0]
                raise AudioNotDownloadedError(path, asset=asset)
            if missing:
                self.download_configured_assets()

            remaining = self.missing_required_paths()
            if remaining:
                asset, path = remaining[0]
                raise AudioNotDownloadedError(path, asset=asset)

            self._backend = self._backend_factory(
                self.settings.tts_model_path,
                self.settings.tts_voices_path,
                self.settings,
            )
            self._loaded_model = desired_model
            self._last_used_at = time.time()
            return self.status()

    def unload(self) -> LocalAudioStatus:
        with self._lock:
            if self._backend is not None:
                self._backend.close()
            self._backend = None
            self._loaded_model = None
            self._last_used_at = None
            return self.status()

    def synthesize(
        self,
        *,
        text: str,
        voice: str | None,
        speed: float,
        lang: str | None,
        is_phonemes: bool,
        trim: bool,
    ) -> SynthesizedSpeech:
        speech = self.backend.synthesize(
            text=text,
            voice=resolve_voice(voice or self.settings.tts_default_voice),
            speed=speed,
            lang=lang or self.settings.tts_default_lang,
            is_phonemes=is_phonemes,
            trim=trim,
        )
        self._last_used_at = time.time()
        return speech

    def voices(self) -> list[str]:
        with self._lock:
            if self._backend is not None:
                return self._backend.voices()
            return voices_from_file(self.settings.tts_voices_path)

    def _unload_if_idle_locked(self) -> None:
        if (
            self._backend is None
            or self.settings.tts_idle_unload_seconds <= 0
            or self._last_used_at is None
        ):
            return
        if time.time() - self._last_used_at > self.settings.tts_idle_unload_seconds:
            self._backend.close()
            self._backend = None
            self._loaded_model = None
            self._last_used_at = None

    @staticmethod
    def _default_backend_factory(model_path: Path, voices_path: Path, settings: Settings) -> AudioBackend:
        return KokoroOnnxBackend(model_path=model_path, voices_path=voices_path)


OPENAI_VOICE_ALIASES = {
    "alloy": "af_alloy",
    "ash": "am_echo",
    "ballad": "bm_fable",
    "coral": "af_bella",
    "echo": "am_echo",
    "fable": "bm_fable",
    "nova": "af_nova",
    "onyx": "am_onyx",
    "sage": "af_sarah",
    "shimmer": "af_sky",
}


def resolve_voice(voice: str) -> str:
    return OPENAI_VOICE_ALIASES.get(voice, voice)


def voices_from_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            return sorted(str(name) for name in payload)
        if isinstance(payload, list):
            return sorted(str(item) for item in payload)
    try:
        import numpy as np
    except Exception:
        return []
    try:
        payload = np.load(path)
    except Exception:
        return []
    if hasattr(payload, "files"):
        return sorted(str(name) for name in payload.files)
    return []


def encode_audio(
    samples: Any,
    sample_rate: int,
    response_format: str,
    *,
    ffmpeg_path: str = "ffmpeg",
) -> tuple[bytes, str]:
    fmt = response_format.lower()
    if fmt == "pcm":
        return _encode_pcm16(samples), "audio/pcm"
    if fmt in {"aac", "opus"}:
        return _encode_with_ffmpeg(samples, sample_rate, fmt, ffmpeg_path=ffmpeg_path)

    soundfile_format = {
        "mp3": ("MP3", "MPEG_LAYER_III", "audio/mpeg"),
        "wav": ("WAV", None, "audio/wav"),
        "flac": ("FLAC", None, "audio/flac"),
    }.get(fmt)
    if soundfile_format is None:
        raise ValueError(f"unsupported audio response_format: {response_format}")

    sf_format, subtype, media_type = soundfile_format
    try:
        import soundfile as sf
    except Exception as exc:
        raise RuntimeError("soundfile is required for encoded audio output: pip install -e .[tts]") from exc

    buffer = io.BytesIO()
    kwargs: dict[str, Any] = {"format": sf_format}
    if subtype:
        kwargs["subtype"] = subtype
    sf.write(buffer, _as_mono_float32(samples), sample_rate, **kwargs)
    return buffer.getvalue(), media_type


def available_audio_formats(ffmpeg_path: str = "ffmpeg") -> list[str]:
    formats = {"pcm"}
    try:
        import soundfile as sf

        available = sf.available_formats()
        if "MP3" in available:
            formats.add("mp3")
        if "WAV" in available:
            formats.add("wav")
        if "FLAC" in available:
            formats.add("flac")
    except Exception:
        pass
    if resolve_ffmpeg_path(ffmpeg_path):
        formats.update({"aac", "opus"})
    return sorted(formats)


def resolve_ffmpeg_path(ffmpeg_path: str = "ffmpeg") -> str | None:
    configured = Path(ffmpeg_path)
    if configured.is_file():
        return str(configured)
    return shutil.which(ffmpeg_path)


def _encode_with_ffmpeg(samples: Any, sample_rate: int, response_format: str, *, ffmpeg_path: str) -> tuple[bytes, str]:
    executable = resolve_ffmpeg_path(ffmpeg_path)
    if not executable:
        raise AudioEncoderMissingError(response_format, "ffmpeg")

    if response_format == "opus":
        output_args = ["-f", "ogg", "-c:a", "libopus", "-b:a", "64k"]
        media_type = "audio/ogg"
    elif response_format == "aac":
        output_args = ["-f", "adts", "-c:a", "aac", "-b:a", "128k"]
        media_type = "audio/aac"
    else:
        raise ValueError(f"unsupported audio response_format: {response_format}")

    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        *output_args,
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        input=_encode_pcm16(samples),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip() or f"ffmpeg exited {result.returncode}"
        raise AudioEncodingError(response_format, message)
    return result.stdout, media_type


def _encode_pcm16(samples: Any) -> bytes:
    audio = _as_mono_float32(samples)
    return b"".join(struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32767.0)) for sample in audio)


def _as_mono_float32(samples: Any) -> Any:
    if hasattr(samples, "reshape") and hasattr(samples, "tolist"):
        return samples.reshape(-1).tolist()
    if isinstance(samples, (bytes, bytearray)):
        raise TypeError("audio samples must be numeric, not bytes")
    if isinstance(samples, list | tuple):
        flattened: list[float] = []
        for sample in samples:
            if isinstance(sample, list | tuple):
                flattened.extend(float(value) for value in sample)
            else:
                flattened.append(float(sample))
        return flattened
    return [float(sample) for sample in samples]
