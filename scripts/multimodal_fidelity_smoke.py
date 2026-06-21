from __future__ import annotations

import argparse
import base64
import io
import tempfile
import wave
from pathlib import Path
from typing import Any

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test LAAS Gemma multimodal fidelity against a running server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="LAAS server URL, with or without /v1.")
    parser.add_argument("--model", default="gemma-4-e4b-it-q4_k_m")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--expect-audio-input-supported",
        action="store_true",
        help="Expect Chat Completions input_audio to succeed instead of returning the default unsupported error.",
    )
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout) as client:
        client.get("/health").raise_for_status()
        print_capabilities(client)
        run_image_input(client, args.model)
        run_video_input(client, args.model)
        run_audio_input_truth_test(client, args.model, expect_supported=args.expect_audio_input_supported)

    print("Multimodal fidelity smoke completed.")
    return 0


def print_capabilities(client: httpx.Client) -> None:
    response = client.get("/v1/local/capabilities")
    response.raise_for_status()
    print(f"capabilities: {response.json()}")


def run_image_input(client: httpx.Client, model: str) -> None:
    image_url = make_test_image_data_url()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer in three words or fewer: what is the dominant color?"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": 32,
    }
    response = client.post("/v1/chat/completions", json=payload)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"].get("content", "")
    print(f"image input: {content!r}")


def run_video_input(client: httpx.Client, model: str) -> None:
    video_path = make_test_video()
    try:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer in three words or fewer: what dominant color appears?"},
                        {"type": "input_video", "input_video": {"url": str(video_path)}},
                    ],
                }
            ],
            "max_tokens": 32,
        }
        response = client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"].get("content", "")
        print(f"video input: {content!r}")
    finally:
        video_path.unlink(missing_ok=True)


def run_audio_input_truth_test(client: httpx.Client, model: str, *, expect_supported: bool) -> None:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe or describe this audio."},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(make_silent_wav_bytes()).decode("ascii"),
                            "format": "wav",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 32,
    }
    response = client.post("/v1/chat/completions", json=payload)
    if expect_supported:
        response.raise_for_status()
        content = response.json()["choices"][0]["message"].get("content", "")
        print(f"audio input: {content!r}")
        return

    if response.status_code != 400:
        raise AssertionError(f"expected input_audio rejection, got {response.status_code}: {response.text}")
    error = response.json().get("detail", {}).get("error", {})
    if error.get("param") != "messages":
        raise AssertionError(f"expected messages capability error, got: {error}")
    print(f"audio input: rejected as unsupported ({error.get('message')})")


def make_test_image_data_url() -> str:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required for multimodal smoke image generation.") from exc

    image = Image.new("RGB", (256, 256), (220, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 80, 176, 176), fill=(245, 245, 245))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def make_test_video() -> Path:
    try:
        import cv2  # type: ignore
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("opencv-python and numpy are required for multimodal smoke video generation.") from exc

    path = Path(tempfile.gettempdir()) / "laas-multimodal-smoke.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (128, 128))
    if not writer.isOpened():
        raise RuntimeError("failed to create temporary MP4 for video input smoke")
    try:
        for _index in range(6):
            frame = np.zeros((128, 128, 3), dtype=np.uint8)
            frame[:, :] = (0, 0, 220)
            writer.write(frame)
    finally:
        writer.release()
    return path


def make_silent_wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)
    return buffer.getvalue()


if __name__ == "__main__":
    raise SystemExit(main())
