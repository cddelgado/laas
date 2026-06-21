from __future__ import annotations

import argparse
import io
import tempfile
import wave
from pathlib import Path

from openai import OpenAI


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test LAAS through the official OpenAI Python client.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="LAAS server URL, with or without /v1.")
    parser.add_argument("--api-key", default="laas-local", help="Dummy API key accepted by local OpenAI clients.")
    parser.add_argument("--text-model", default="gemma-4-e4b-it-q4_k_m")
    parser.add_argument("--embedding-model", default="bge-small-en-v1.5")
    parser.add_argument("--image-model", default="sdxl-turbo")
    parser.add_argument("--image-edit-model", default="sd-1.5-inpainting")
    parser.add_argument("--tts-model", default="tts-1")
    parser.add_argument("--stt-model", default="whisper-1")
    parser.add_argument("--prompt", default="a small brass table lamp, realistic lighting")
    parser.add_argument("--include-image", action="store_true", help="Also call /v1/images/generations.")
    parser.add_argument("--include-image-edit", action="store_true", help="Also call /v1/images/edits.")
    parser.add_argument("--include-voice", action="store_true", help="Also call TTS and transcription endpoints.")
    args = parser.parse_args()

    client = OpenAI(base_url=openai_base_url(args.base_url), api_key=args.api_key)
    run_text_stack(client, args)
    if args.include_image:
        run_image_generation(client, args)
    if args.include_image_edit:
        run_image_edit(client, args)
    if args.include_voice:
        run_voice_stack(client, args)
    print("OpenAI client smoke completed.")
    return 0


def openai_base_url(value: str) -> str:
    value = value.rstrip("/")
    return value if value.endswith("/v1") else f"{value}/v1"


def run_text_stack(client: OpenAI, args: argparse.Namespace) -> None:
    models = client.models.list()
    assert any(model.id == args.text_model for model in models.data), f"{args.text_model} not listed"
    print(f"models.list: {len(models.data)} models")

    chat = client.chat.completions.create(
        model=args.text_model,
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=32,
    )
    assert chat.choices[0].message.content is not None
    print(f"chat.completions: {chat.choices[0].finish_reason}")

    tool_chat = client.chat.completions.create(
        model=args.text_model,
        messages=[{"role": "user", "content": "call_tool"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        max_tokens=64,
    )
    calls = tool_chat.choices[0].message.tool_calls or []
    assert calls and calls[0].function.name == "get_weather"
    print(f"chat.completions tool call: {calls[0].function.name}")

    response = client.responses.create(model=args.text_model, input="hello", max_output_tokens=32)
    assert response.output_text is not None
    print(f"responses.create: {response.status}")

    embedding = client.embeddings.create(model=args.embedding_model, input="alpha", dimensions=8)
    assert len(embedding.data[0].embedding) == 8
    print(f"embeddings.create: {len(embedding.data[0].embedding)} dimensions")


def run_image_generation(client: OpenAI, args: argparse.Namespace) -> None:
    image = client.images.generate(
        model=args.image_model,
        prompt=args.prompt,
        size="512x512",
        response_format="b64_json",
        n=1,
    )
    assert image.data and image.data[0].b64_json
    print("images.generate: received b64_json")


def run_image_edit(client: OpenAI, args: argparse.Namespace) -> None:
    base_path, mask_path = make_inpaint_pngs()
    try:
        with base_path.open("rb") as image, mask_path.open("rb") as mask:
            edited = client.images.edit(
                model=args.image_edit_model,
                image=image,
                mask=mask,
                prompt=args.prompt,
                size="512x512",
                response_format="b64_json",
                n=1,
            )
    finally:
        base_path.unlink(missing_ok=True)
        mask_path.unlink(missing_ok=True)
    assert edited.data and edited.data[0].b64_json
    print("images.edit: received b64_json")


def make_inpaint_pngs() -> tuple[Path, Path]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required for --include-image-edit; install LAAS image dependencies.") from exc

    temp_dir = Path(tempfile.gettempdir())
    base_path = temp_dir / "laas-openai-smoke-base.png"
    mask_path = temp_dir / "laas-openai-smoke-mask.png"

    base = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(base)
    draw.rectangle((128, 160, 384, 360), fill=(185, 160, 120))
    draw.rectangle((230, 210, 282, 330), fill=(245, 245, 245))
    base.save(base_path)

    mask = Image.new("L", (512, 512), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((220, 200, 292, 340), fill=255)
    mask.save(mask_path)
    return base_path, mask_path


def run_voice_stack(client: OpenAI, args: argparse.Namespace) -> None:
    speech = client.audio.speech.create(
        model=args.tts_model,
        voice="alloy",
        input="hello from LAAS",
        response_format="wav",
    )
    assert read_binary_response(speech)
    print("audio.speech.create: received audio bytes")

    wav_path = make_silent_wav()
    try:
        with wav_path.open("rb") as audio:
            transcription = client.audio.transcriptions.create(
                model=args.stt_model,
                file=audio,
                response_format="json",
            )
    finally:
        wav_path.unlink(missing_ok=True)
    assert hasattr(transcription, "text")
    print("audio.transcriptions.create: received text")


def read_binary_response(response: object) -> bytes:
    if hasattr(response, "read"):
        return response.read()
    if isinstance(response, bytes | bytearray):
        return bytes(response)
    raise TypeError(f"Unsupported binary response type: {type(response)!r}")


def make_silent_wav() -> Path:
    path = Path(tempfile.gettempdir()) / "laas-openai-smoke-silence.wav"
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)
    path.write_bytes(buffer.getvalue())
    return path


if __name__ == "__main__":
    raise SystemExit(main())
