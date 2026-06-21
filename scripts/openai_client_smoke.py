from __future__ import annotations

import argparse
import json
import io
import tempfile
import urllib.request
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
    parser.add_argument("--include-storage", action="store_true", help="Also call files, vector stores, batches, and moderations.")
    args = parser.parse_args()

    base_url = openai_base_url(args.base_url)
    client = OpenAI(base_url=base_url, api_key=args.api_key)
    run_text_stack(client, args)
    if args.include_storage:
        run_storage_stack(client, base_url, args)
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


def run_storage_stack(client: OpenAI, base_url: str, args: argparse.Namespace) -> None:
    notes_path = Path(tempfile.gettempdir()) / "laas-openai-smoke-notes.md"
    batch_path = Path(tempfile.gettempdir()) / "laas-openai-smoke-batch.jsonl"
    uploaded_id: str | None = None
    batch_input_id: str | None = None
    batch_output_id: str | None = None
    store_id: str | None = None
    notes_path.write_text("Vulkan setup requires a compatible GPU runtime package.\n", encoding="utf-8")
    batch_path.write_text(
        json.dumps(
            {
                "custom_id": "embedding-one",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"input": "alpha", "dimensions": 4},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        with notes_path.open("rb") as fh:
            uploaded = client.files.create(file=fh, purpose="assistants")
        assert uploaded.id
        uploaded_id = uploaded.id
        print(f"files.create: {uploaded.id}")

        vector_stores = getattr(client, "vector_stores", None) or getattr(getattr(client, "beta", None), "vector_stores", None)
        assert vector_stores is not None, "OpenAI SDK does not expose vector_stores"
        store = vector_stores.create(name="laas-smoke")
        assert store.id
        store_id = store.id
        vector_stores.files.create(vector_store_id=store.id, file_id=uploaded.id)
        print(f"vector_stores.files.create: {store.id}")

        chat = client.chat.completions.create(
            model=args.text_model,
            messages=[{"role": "user", "content": "What does Vulkan setup require?"}],
            tools=[{"type": "file_search", "vector_store_ids": [store.id]}],
            max_tokens=64,
        )
        assert getattr(chat, "laas_file_search", None) or chat.model_extra.get("laas_file_search")
        print("chat.completions file_search: received retrieval metadata")

        response = client.responses.create(
            model=args.text_model,
            input="What does Vulkan setup require?",
            tools=[{"type": "file_search", "vector_store_ids": [store.id]}],
            max_output_tokens=64,
        )
        assert response.model_extra.get("laas_file_search")
        print("responses.create file_search: received retrieval metadata")

        moderation = client.moderations.create(input="hello")
        assert moderation.results[0].flagged is False
        print("moderations.create: ok")

        with batch_path.open("rb") as fh:
            batch_input = client.files.create(file=fh, purpose="batch")
        batch_input_id = batch_input.id
        batch = client.batches.create(input_file_id=batch_input.id, endpoint="/v1/embeddings", completion_window="24h")
        assert batch.output_file_id
        batch_output_id = batch.output_file_id
        print(f"batches.create: {batch.status}")

        local_search = post_json(base_url, f"/local/vector_stores/{store.id}/search", {"query": "Vulkan setup", "limit": 1})
        assert local_search["data"]
        print("local.vector_stores.search: ok")
    finally:
        if store_id:
            delete_json(base_url, f"/vector_stores/{store_id}")
        for file_id in (uploaded_id, batch_input_id, batch_output_id):
            if file_id:
                delete_json(base_url, f"/files/{file_id}")
        notes_path.unlink(missing_ok=True)
        batch_path.unlink(missing_ok=True)


def post_json(base_url: str, path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer laas-local"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def delete_json(base_url: str, path: str) -> dict | None:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        headers={"Authorization": "Bearer laas-local"},
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


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
