from __future__ import annotations

import base64
import mimetypes
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from .errors import openai_error


def normalize_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if isinstance(content, list):
            item["content"] = normalize_content_parts(content)
        normalized.append(item)
    return normalized


def normalize_content_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for part in parts:
        part_type = part.get("type")
        if part_type in {"text", "input_text"}:
            normalized.append({"type": "text", "text": part.get("text", "")})
        elif part_type in {"image_url", "input_image"}:
            normalized.append(_normalize_image_part(part))
        elif part_type in {"video_url", "input_video"}:
            normalized.extend(_normalize_video_part(part))
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
    data = audio.get("data") if isinstance(audio, dict) else None
    if not data:
        raise openai_error(400, "audio inputs require base64 data in local mode", param="input")
    return {"type": "input_audio", "input_audio": audio}


def _normalize_video_part(part: dict[str, Any]) -> list[dict[str, Any]]:
    frames = part.get("frames")
    if isinstance(frames, list) and frames:
        return [_normalize_image_part({"type": "input_image", "image_url": frame}) for frame in frames]

    video_url = part.get("video_url") or part.get("input_video")
    if isinstance(video_url, dict):
        video_url = video_url.get("url")
    if not isinstance(video_url, str):
        raise openai_error(400, "video inputs require video_url or frames", param="input")

    frame_urls = extract_video_frame_data_urls(video_url)
    return [{"type": "image_url", "image_url": {"url": url}} for url in frame_urls]


def extract_video_frame_data_urls(video_url: str, max_frames: int = 8) -> list[str]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise openai_error(
            501,
            "video frame extraction requires the optional 'video' extra: pip install -e .[video]",
            param="input",
            code="video_backend_missing",
        ) from exc

    video_path = _materialize_video(video_url)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise openai_error(400, "could not open video input", param="input")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, frame_count // max_frames) if frame_count else 30
    frames: list[str] = []
    index = 0
    while len(frames) < max_frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            break
        success, buffer = cv2.imencode(".jpg", frame)
        if success:
            encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
            frames.append(f"data:image/jpeg;base64,{encoded}")
        index += step
    capture.release()

    if not frames:
        raise openai_error(400, "could not extract video frames", param="input")
    return frames


def _materialize_video(video_url: str) -> Path:
    if video_url.startswith("data:"):
        header, encoded = video_url.split(",", 1)
        suffix = ".mp4"
        mime = header.split(";", 1)[0].removeprefix("data:")
        guessed = mimetypes.guess_extension(mime)
        if guessed:
            suffix = guessed
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp.write(base64.b64decode(encoded))
        temp.close()
        return Path(temp.name)

    if video_url.startswith(("http://", "https://")):
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_url).suffix or ".mp4")
        temp.close()
        urllib.request.urlretrieve(video_url, temp.name)
        return Path(temp.name)

    path = Path(video_url)
    if not path.exists():
        raise openai_error(400, f"video path does not exist: {video_url}", param="input")
    return path
