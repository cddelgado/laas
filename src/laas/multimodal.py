from __future__ import annotations

import base64
import mimetypes
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import openai_error


@dataclass(frozen=True)
class VideoExtractionConfig:
    max_frames: int = 8
    sample_fps: float = 0.5
    max_seconds: float = 60.0
    frame_size: int = 768


@dataclass(frozen=True)
class MaterializedVideo:
    path: Path
    temporary: bool = False


def normalize_chat_messages(
    messages: list[dict[str, Any]],
    *,
    video_config: VideoExtractionConfig | None = None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if isinstance(content, list):
            item["content"] = normalize_content_parts(content, video_config=video_config)
        normalized.append(item)
    return normalized


def normalize_content_parts(
    parts: list[dict[str, Any]],
    *,
    video_config: VideoExtractionConfig | None = None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for part in parts:
        part_type = part.get("type")
        if part_type in {"text", "input_text"}:
            normalized.append({"type": "text", "text": part.get("text", "")})
        elif part_type in {"image_url", "input_image"}:
            normalized.append(_normalize_image_part(part))
        elif part_type in {"video_url", "input_video"}:
            normalized.extend(_normalize_video_part(part, video_config=video_config))
        elif part_type in {"audio", "input_audio"}:
            normalized.append(_normalize_audio_part(part))
        else:
            normalized.append(part)
    return normalized


def _normalize_image_part(part: dict[str, Any]) -> dict[str, Any]:
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        return {"type": "image_url", "image_url": image_url}
    if isinstance(image_url, str):
        return {"type": "image_url", "image_url": {"url": image_url}}
    file_id = part.get("file_id")
    if file_id:
        raise openai_error(400, "file_id image inputs are not available in local mode", param="input")
    raise openai_error(400, "image inputs require image_url", param="input")


def _normalize_audio_part(part: dict[str, Any]) -> dict[str, Any]:
    audio = part.get("input_audio") or part.get("audio") or part
    if not isinstance(audio, dict):
        raise openai_error(400, "audio inputs require input_audio data and format", param="input", code="invalid_audio")
    data = audio.get("data")
    if not isinstance(data, str) or not data:
        raise openai_error(400, "audio inputs require base64 data in local mode", param="input", code="invalid_audio")
    _decode_base64(data, "audio", code="invalid_audio")
    audio_format = audio.get("format")
    if not isinstance(audio_format, str) or audio_format.lower() not in {"wav", "mp3"}:
        raise openai_error(
            400,
            "audio input format must be 'wav' or 'mp3'",
            param="input_audio.format",
            code="invalid_audio_format",
        )
    normalized = dict(audio)
    normalized["format"] = audio_format.lower()
    return {"type": "input_audio", "input_audio": normalized}


def _normalize_video_part(
    part: dict[str, Any],
    *,
    video_config: VideoExtractionConfig | None = None,
) -> list[dict[str, Any]]:
    video = part.get("video_url") or part.get("input_video")
    frames = part.get("frames")
    if isinstance(video, dict) and not frames:
        frames = video.get("frames")
    if isinstance(frames, list) and frames:
        return [_normalize_video_frame(frame) for frame in frames]

    video_url = _video_source_to_url(video)
    if video_url is None:
        raise openai_error(400, "video inputs require video_url, input_video data, or frames", param="input")

    frame_urls = extract_video_frame_data_urls(video_url, config=video_config)
    return [{"type": "image_url", "image_url": {"url": url}} for url in frame_urls]


def _normalize_video_frame(frame: Any) -> dict[str, Any]:
    if isinstance(frame, str):
        return _normalize_image_part({"type": "input_image", "image_url": frame})
    if isinstance(frame, dict):
        if "image_url" in frame:
            return _normalize_image_part({"type": "input_image", "image_url": frame["image_url"]})
        if "url" in frame:
            return _normalize_image_part({"type": "input_image", "image_url": frame})
    raise openai_error(400, "video frames must be image URL strings or objects", param="input.frames")


def _video_source_to_url(video: Any) -> str | None:
    if isinstance(video, str):
        return video
    if not isinstance(video, dict):
        return None

    url = video.get("url") or video.get("video_url")
    if isinstance(url, str):
        return url

    data = video.get("data") or video.get("file_data")
    if not isinstance(data, str) or not data:
        if video.get("file_id"):
            raise openai_error(400, "file_id video inputs are not available in local mode", param="input")
        return None
    if data.startswith("data:"):
        return data

    _decode_base64(data, "video", code="invalid_video")
    video_format = video.get("format") or "mp4"
    if not isinstance(video_format, str):
        raise openai_error(400, "video format must be a string", param="input_video.format", code="invalid_video")
    return f"data:video/{video_format.lower().lstrip('.')};base64,{data}"


def extract_video_frame_data_urls(
    video_url: str,
    max_frames: int | None = None,
    *,
    config: VideoExtractionConfig | None = None,
) -> list[str]:
    active_config = config or VideoExtractionConfig()
    if max_frames is not None:
        active_config = VideoExtractionConfig(
            max_frames=max_frames,
            sample_fps=active_config.sample_fps,
            max_seconds=active_config.max_seconds,
            frame_size=active_config.frame_size,
        )
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise openai_error(
            501,
            "video frame extraction requires the optional 'video' extra: pip install -e .[video]",
            param="input",
            code="video_backend_missing",
        ) from exc

    materialized = _materialize_video(video_url)
    capture = cv2.VideoCapture(str(materialized.path))
    if not capture.isOpened():
        raise openai_error(400, "could not open video input", param="input")

    frames: list[str] = []
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        for index in _sample_frame_indices(frame_count, fps, active_config):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            frame = _resize_frame(cv2, frame, active_config.frame_size)
            success, buffer = cv2.imencode(".jpg", frame)
            if success:
                encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
                frames.append(f"data:image/jpeg;base64,{encoded}")
    finally:
        capture.release()
        if materialized.temporary:
            try:
                os.unlink(materialized.path)
            except OSError:
                pass

    if not frames:
        raise openai_error(400, "could not extract video frames", param="input")
    return frames


def _sample_frame_indices(frame_count: int, fps: float, config: VideoExtractionConfig) -> list[int]:
    max_frames = max(1, config.max_frames)
    if frame_count <= 0:
        step = max(1, int(round(fps / config.sample_fps))) if fps > 0 else 30
        return [index * step for index in range(max_frames)]

    effective_frame_count = frame_count
    if fps > 0 and config.max_seconds > 0:
        effective_frame_count = min(frame_count, max(1, int(fps * config.max_seconds)))

    if fps > 0 and config.sample_fps > 0:
        sample_step = max(1, int(round(fps / config.sample_fps)))
        candidates = list(range(0, effective_frame_count, sample_step))
    else:
        candidates = list(range(effective_frame_count))
    if not candidates:
        candidates = [0]

    if len(candidates) <= max_frames:
        return candidates
    if max_frames == 1:
        return [candidates[0]]

    selected: list[int] = []
    for offset in range(max_frames):
        candidate_index = round(offset * (len(candidates) - 1) / (max_frames - 1))
        selected.append(candidates[candidate_index])
    return sorted(dict.fromkeys(selected))


def _resize_frame(cv2: Any, frame: Any, max_size: int) -> Any:
    shape = getattr(frame, "shape", None)
    if not shape or len(shape) < 2 or max_size <= 0:
        return frame
    height, width = int(shape[0]), int(shape[1])
    longest = max(height, width)
    if longest <= max_size:
        return frame
    scale = max_size / longest
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(frame, size, interpolation=getattr(cv2, "INTER_AREA", 1))


def _materialize_video(video_url: str) -> MaterializedVideo:
    if video_url.startswith("data:"):
        try:
            header, encoded = video_url.split(",", 1)
        except ValueError as exc:
            raise openai_error(400, "invalid video data URL", param="input", code="invalid_video") from exc
        suffix = ".mp4"
        mime = header.split(";", 1)[0].removeprefix("data:")
        guessed = mimetypes.guess_extension(mime)
        if guessed:
            suffix = guessed
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            temp.write(_decode_base64(encoded, "video", code="invalid_video"))
        finally:
            temp.close()
        return MaterializedVideo(Path(temp.name), temporary=True)

    if video_url.startswith(("http://", "https://")):
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_url).suffix or ".mp4")
        temp.close()
        urllib.request.urlretrieve(video_url, temp.name)
        return MaterializedVideo(Path(temp.name), temporary=True)

    path = Path(video_url)
    if not path.exists():
        raise openai_error(400, f"video path does not exist: {video_url}", param="input")
    return MaterializedVideo(path)


def _decode_base64(value: str, label: str, *, code: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:
        raise openai_error(400, f"{label} data must be valid base64", param="input", code=code) from exc
