from __future__ import annotations

import base64
import json
import os
import sqlite3
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from laas.compat_check import run_compat_check
from laas.diagnostics import collect_diagnostics
from laas.app import create_app
from laas.backends import EchoBackend, _add_mmproj_kwargs, _add_speculative_kwargs, _add_supported_constructor_kwargs
from laas.embedding import EmbeddingManager, HashEmbeddingBackend
from laas.image import (
    DiffusersImageEditBackend,
    GeneratedImage,
    ImageBackend,
    ImageEditBackend,
    ImageEditManager,
    ImageManager,
)
from laas.main import build_parser, confirm_missing_model_downloads, missing_configured_model_paths
from laas.main import main as laas_main
from laas.manager import ModelManager
from laas.openai_compat import _normalize_chat_response, _normalize_completion_response, _video_config
from laas.settings import Settings, default_model_dir
from laas.tools import parse_tool_calls, remove_tool_call_markup
from laas.transcription import (
    TranscriptionBackend,
    TranscriptionManager,
    TranscriptionResult,
    TranscriptionSegment,
    transcription_to_response,
)
from laas.tts import AudioBackend, AudioEncoderMissingError, AudioManager, SynthesizedSpeech, encode_audio, resolve_voice
from laas.video import DiffusersWanVideoBackend, GeneratedVideo, VideoBackend, VideoManager, frames_for_duration

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)
OPENAI_COMPAT_FIXTURES = Path(__file__).parent / "fixtures" / "openai_compat"


def make_embedding_manager(settings: Settings, *, write_model: bool = True) -> EmbeddingManager:
    if write_model:
        settings.embedding_model_path.mkdir(parents=True, exist_ok=True)
        (settings.embedding_model_path / "config.json").write_text("{}", encoding="utf-8")
    return EmbeddingManager(
        settings,
        backend_factory=lambda model_path, active_settings: HashEmbeddingBackend(
            dimensions=active_settings.embedding_dimensions
        ),
    )


def make_client(tmp_path: Path, *, write_model: bool = True, auto_download: bool = False) -> TestClient:
    settings = Settings(
        model_dir=tmp_path,
        file_storage_dir=tmp_path / "file-storage",
        settings_file=tmp_path / "settings.json",
        idle_unload_seconds=0,
        auto_download=auto_download,
    )

    def backend_factory(model_path: Path, active_settings: Settings) -> EchoBackend:
        return EchoBackend()

    manager = ModelManager(settings, backend_factory=backend_factory)
    embedding_manager = make_embedding_manager(settings)
    if write_model:
        (settings.model_path.parent).mkdir(parents=True, exist_ok=True)
        settings.model_path.write_bytes(b"test-model")
        if settings.mmproj_path:
            settings.mmproj_path.write_bytes(b"test-mmproj")
    return TestClient(create_app(settings=settings, manager=manager, embedding_manager=embedding_manager))


def make_client_with_backend(
    tmp_path: Path,
    backend: EchoBackend,
    *,
    llm_audio_input_enabled: bool = False,
) -> TestClient:
    settings = Settings(
        model_dir=tmp_path,
        file_storage_dir=tmp_path / "file-storage",
        settings_file=tmp_path / "settings.json",
        idle_unload_seconds=0,
        llm_audio_input_enabled=llm_audio_input_enabled,
    )
    manager = ModelManager(settings, backend_factory=lambda model_path, active_settings: backend)
    embedding_manager = make_embedding_manager(settings)
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"test-model")
    if settings.mmproj_path:
        settings.mmproj_path.write_bytes(b"test-mmproj")
    return TestClient(create_app(settings=settings, manager=manager, embedding_manager=embedding_manager))


def make_embedding_client(
    tmp_path: Path,
    *,
    write_model: bool = True,
    auto_download: bool = True,
) -> TestClient:
    settings = Settings(
        model_dir=tmp_path,
        file_storage_dir=tmp_path / "file-storage",
        settings_file=tmp_path / "settings.json",
        idle_unload_seconds=0,
        embedding_idle_unload_seconds=0,
        embedding_auto_download=auto_download,
    )
    manager = ModelManager(settings, backend_factory=lambda model_path, active_settings: EchoBackend())
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"test-model")
    if settings.mmproj_path:
        settings.mmproj_path.write_bytes(b"test-mmproj")
    embedding_manager = make_embedding_manager(settings, write_model=write_model)
    return TestClient(create_app(settings=settings, manager=manager, embedding_manager=embedding_manager))


def sse_payloads(response) -> list[Any]:
    payloads: list[Any] = []
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        payloads.append("[DONE]" if data == "[DONE]" else json.loads(data))
    return payloads


class CapturingBackend(EchoBackend):
    def __init__(self) -> None:
        self.chat_params: dict[str, object] = {}
        self.completion_params: dict[str, object] = {}

    def chat_completion(self, **kwargs):
        self.chat_params = kwargs.get("extra_params") or {}
        return super().chat_completion(**kwargs)

    def completion(self, **kwargs):
        self.completion_params = kwargs.get("extra_params") or {}
        return super().completion(**kwargs)


class MessageCapturingBackend(EchoBackend):
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs["messages"])
        return super().chat_completion(**kwargs)


class SplitStreamingBackend(EchoBackend):
    def chat_completion(self, **kwargs):
        if not kwargs.get("stream"):
            return super().chat_completion(**kwargs)
        return iter(
            [
                {
                    "id": "chatcmpl_split",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": kwargs["model"],
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "split "}, "finish_reason": None}],
                },
                {
                    "id": "chatcmpl_split",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": kwargs["model"],
                    "choices": [{"index": 0, "delta": {"content": "stream"}, "finish_reason": None}],
                },
                {
                    "id": "chatcmpl_split",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": kwargs["model"],
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
        )


class StreamingToolBackend(EchoBackend):
    def chat_completion(self, **kwargs):
        if not kwargs.get("stream"):
            return super().chat_completion(**kwargs)
        return iter(
            [
                {
                    "id": "chatcmpl_raw",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": r"D:\AI\Models\gemma.gguf",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": '<|tool_call>call:get_weather{location:<|"|>Chi'},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl_raw",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": r"D:\AI\Models\gemma.gguf",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": 'cago<|"|>}<tool_call|>'},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl_raw",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": r"D:\AI\Models\gemma.gguf",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
        )


class FakeAudioBackend(AudioBackend):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def synthesize(self, **kwargs) -> SynthesizedSpeech:
        self.calls.append(kwargs)
        return SynthesizedSpeech(samples=[0.0, 0.5, -0.5], sample_rate=24000)

    def voices(self) -> list[str]:
        return ["af", "af_alloy"]

    def close(self) -> None:
        self.closed = True


class FakeTranscriptionBackend(TranscriptionBackend):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def transcribe(self, **kwargs) -> TranscriptionResult:
        self.calls.append(kwargs)
        assert Path(kwargs["media_path"]).exists()
        return TranscriptionResult(
            text="hello from whisper",
            language=kwargs.get("language") or "en",
            duration=1.25,
            segments=[TranscriptionSegment(id=0, start=0.0, end=1.25, text="hello from whisper")],
        )

    def close(self) -> None:
        self.closed = True


class FakeImageBackend(ImageBackend):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def generate(self, **kwargs) -> GeneratedImage:
        self.calls.append(kwargs)
        return GeneratedImage(content=b"fake-png", media_type="image/png")

    def variation(self, **kwargs) -> GeneratedImage:
        self.calls.append(kwargs)
        return GeneratedImage(content=PNG_1X1, media_type="image/png")

    def close(self) -> None:
        self.closed = True


class FakeImageEditBackend(ImageEditBackend):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def edit(self, **kwargs) -> GeneratedImage:
        self.calls.append(kwargs)
        return GeneratedImage(content=b"fake-edited-png", media_type="image/png")

    def close(self) -> None:
        self.closed = True


class FakeVideoBackend(VideoBackend):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def generate(self, **kwargs) -> GeneratedVideo:
        self.calls.append(kwargs)
        return GeneratedVideo(content=b"fake-mp4", media_type="video/mp4")

    def close(self) -> None:
        self.closed = True


def test_diffusers_image_edit_backend_passes_padding_crop(tmp_path: Path) -> None:
    from PIL import Image

    class FakePipe:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return type("Output", (), {"images": [Image.new("RGB", (8, 8), (0, 0, 0))]})()

    settings = Settings(
        model_dir=tmp_path,
        file_storage_dir=tmp_path / "file-storage",
        settings_file=tmp_path / "settings.json",
        image_edit_padding_mask_crop=24,
        image_edit_composite_blur_radius=0,
    )
    pipe = FakePipe()
    backend = object.__new__(DiffusersImageEditBackend)
    backend.settings = settings
    backend._device = "cpu"
    backend._pipe = pipe
    backend._supports_padding_mask_crop = True

    result = backend.edit(
        prompt="add lamp",
        negative_prompt=None,
        image=Image.new("RGB", (8, 8), (255, 255, 255)),
        mask_image=Image.new("L", (8, 8), 255),
        width=8,
        height=8,
        num_inference_steps=1,
        guidance_scale=7.5,
        strength=1.0,
        seed=None,
    )

    assert pipe.calls[0]["padding_mask_crop"] == 24
    assert result.media_type == "image/png"


def make_audio_client(
    tmp_path: Path,
    *,
    write_assets: bool = True,
    ffmpeg_path: str = "ffmpeg",
) -> tuple[TestClient, FakeAudioBackend]:
    settings = Settings(
        model_dir=tmp_path,
        file_storage_dir=tmp_path / "file-storage",
        settings_file=tmp_path / "settings.json",
        idle_unload_seconds=0,
        tts_idle_unload_seconds=0,
        tts_voices_filename="voices.json",
        tts_ffmpeg_path=ffmpeg_path,
    )
    text_manager = ModelManager(settings, backend_factory=lambda model_path, active_settings: EchoBackend())
    backend = FakeAudioBackend()
    audio_manager = AudioManager(settings, backend_factory=lambda model_path, voices_path, active_settings: backend)

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"model")
    if settings.mmproj_path:
        settings.mmproj_path.write_bytes(b"mmproj")
    if write_assets:
        settings.tts_model_path.parent.mkdir(parents=True, exist_ok=True)
        settings.tts_model_path.write_bytes(b"tts-model")
        settings.tts_voices_path.write_text('{"af": [], "af_alloy": []}', encoding="utf-8")

    client = TestClient(create_app(settings=settings, manager=text_manager, audio_manager=audio_manager))
    return client, backend


def make_image_client(
    tmp_path: Path,
    *,
    write_model: bool = True,
    auto_download: bool = False,
) -> tuple[TestClient, FakeImageBackend]:
    settings = Settings(
        model_dir=tmp_path,
        file_storage_dir=tmp_path / "file-storage",
        settings_file=tmp_path / "settings.json",
        idle_unload_seconds=0,
        image_idle_unload_seconds=0,
        image_auto_download=auto_download,
    )
    text_manager = ModelManager(settings, backend_factory=lambda model_path, active_settings: EchoBackend())
    image_backend = FakeImageBackend()
    image_manager = ImageManager(settings, backend_factory=lambda model_path, active_settings: image_backend)

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"model")
    if settings.mmproj_path:
        settings.mmproj_path.write_bytes(b"mmproj")
    if write_model:
        settings.image_model_path.mkdir(parents=True, exist_ok=True)
        (settings.image_model_path / "model_index.json").write_text("{}", encoding="utf-8")

    client = TestClient(create_app(settings=settings, manager=text_manager, image_manager=image_manager))
    return client, image_backend


def make_video_client(
    tmp_path: Path,
    *,
    write_model: bool = True,
    auto_download: bool = False,
) -> tuple[TestClient, FakeVideoBackend]:
    settings = Settings(
        model_dir=tmp_path,
        file_storage_dir=tmp_path / "file-storage",
        settings_file=tmp_path / "settings.json",
        idle_unload_seconds=0,
        video_generation_idle_unload_seconds=0,
        video_generation_auto_download=auto_download,
    )
    text_manager = ModelManager(settings, backend_factory=lambda model_path, active_settings: EchoBackend())
    video_backend = FakeVideoBackend()
    video_manager = VideoManager(settings, backend_factory=lambda model_path, active_settings: video_backend)

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"model")
    if settings.mmproj_path:
        settings.mmproj_path.write_bytes(b"mmproj")
    if write_model:
        settings.video_generation_high_noise_path.parent.mkdir(parents=True, exist_ok=True)
        settings.video_generation_low_noise_path.parent.mkdir(parents=True, exist_ok=True)
        settings.video_generation_vae_path.parent.mkdir(parents=True, exist_ok=True)
        settings.video_generation_high_noise_path.write_bytes(b"high")
        settings.video_generation_low_noise_path.write_bytes(b"low")
        settings.video_generation_vae_path.write_bytes(b"vae")
        for filename in (
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors.index.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "transformer/config.json",
            "transformer_2/config.json",
            "vae/config.json",
        ):
            path = settings.video_generation_diffusers_model_path / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")

    client = TestClient(create_app(settings=settings, manager=text_manager, video_manager=video_manager))
    return client, video_backend


def make_image_edit_client(
    tmp_path: Path,
    *,
    write_model: bool = True,
    auto_download: bool = False,
) -> tuple[TestClient, FakeImageEditBackend]:
    settings = Settings(
        model_dir=tmp_path,
        file_storage_dir=tmp_path / "file-storage",
        settings_file=tmp_path / "settings.json",
        idle_unload_seconds=0,
        image_edit_idle_unload_seconds=0,
        image_edit_auto_download=auto_download,
    )
    text_manager = ModelManager(settings, backend_factory=lambda model_path, active_settings: EchoBackend())
    image_edit_backend = FakeImageEditBackend()
    image_edit_manager = ImageEditManager(
        settings,
        backend_factory=lambda model_path, active_settings: image_edit_backend,
    )

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"model")
    if settings.mmproj_path:
        settings.mmproj_path.write_bytes(b"mmproj")
    if write_model:
        settings.image_edit_model_path.mkdir(parents=True, exist_ok=True)
        (settings.image_edit_model_path / "model_index.json").write_text("{}", encoding="utf-8")

    client = TestClient(create_app(settings=settings, manager=text_manager, image_edit_manager=image_edit_manager))
    return client, image_edit_backend


def make_voice_client(
    tmp_path: Path,
    *,
    write_audio_assets: bool = True,
    write_transcription_model: bool = True,
    text_backend: EchoBackend | None = None,
) -> tuple[TestClient, FakeAudioBackend, FakeTranscriptionBackend]:
    settings = Settings(
        model_dir=tmp_path,
        file_storage_dir=tmp_path / "file-storage",
        settings_file=tmp_path / "settings.json",
        idle_unload_seconds=0,
        tts_idle_unload_seconds=0,
        stt_idle_unload_seconds=0,
        tts_voices_filename="voices.json",
    )
    text_manager = ModelManager(settings, backend_factory=lambda model_path, active_settings: text_backend or EchoBackend())
    audio_backend = FakeAudioBackend()
    transcription_backend = FakeTranscriptionBackend()
    audio_manager = AudioManager(settings, backend_factory=lambda model_path, voices_path, active_settings: audio_backend)
    transcription_manager = TranscriptionManager(
        settings,
        backend_factory=lambda model_path, active_settings: transcription_backend,
    )

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"model")
    if settings.mmproj_path:
        settings.mmproj_path.write_bytes(b"mmproj")
    if write_audio_assets:
        settings.tts_model_path.parent.mkdir(parents=True, exist_ok=True)
        settings.tts_model_path.write_bytes(b"tts-model")
        settings.tts_voices_path.write_text('{"af": [], "af_alloy": []}', encoding="utf-8")
    if write_transcription_model:
        settings.stt_model_path.parent.mkdir(parents=True, exist_ok=True)
        settings.stt_model_path.write_bytes(b"stt-model")

    client = TestClient(
        create_app(
            settings=settings,
            manager=text_manager,
            audio_manager=audio_manager,
            transcription_manager=transcription_manager,
        )
    )
    return client, audio_backend, transcription_backend


def golden_fixture_names() -> list[str]:
    return sorted(path.name for path in OPENAI_COMPAT_FIXTURES.glob("*.json"))


def golden_client(stack: str, tmp_path: Path) -> TestClient:
    if stack == "text":
        return make_client(tmp_path)
    if stack == "image":
        client, _backend = make_image_client(tmp_path)
        return client
    if stack == "image_edit":
        client, _backend = make_image_edit_client(tmp_path)
        return client
    if stack == "audio":
        client, _backend = make_audio_client(tmp_path)
        return client
    if stack == "voice":
        client, _audio_backend, _transcription_backend = make_voice_client(tmp_path)
        return client
    raise AssertionError(f"Unknown golden fixture stack: {stack}")


def send_golden_request(client: TestClient, fixture: dict[str, Any]):
    request = fixture["request"]
    method = request["method"]
    path = request["path"]
    if method == "GET":
        return client.get(path)
    if method != "POST":
        raise AssertionError(f"Unsupported golden fixture method: {method}")
    if "files" in request:
        files = {
            item["field"]: (
                item["filename"],
                base64.b64decode(item["content_base64"]),
                item["content_type"],
            )
            for item in request["files"]
        }
        return client.post(path, data=request.get("form", {}), files=files)
    return client.post(path, json=request.get("json", {}))


def value_at_path(payload: Any, path: str) -> Any:
    value = payload
    for part in path.split("."):
        if isinstance(value, list):
            value = value[int(part)]
        else:
            value = value[part]
    return value


def assert_golden_response(response, fixture: dict[str, Any]) -> None:
    expected = fixture["expect"]
    assert response.status_code == expected["status_code"]

    for header in expected.get("headers", []):
        assert response.headers[header["name"]] == header["equals"]

    if "body_base64" in expected:
        assert response.content == base64.b64decode(expected["body_base64"])

    if any(key in expected for key in ("paths", "lengths", "base64")):
        payload = response.json()
        for assertion in expected.get("paths", []):
            value = value_at_path(payload, assertion["path"])
            if "equals" in assertion:
                assert value == assertion["equals"]
            if "prefix" in assertion:
                assert isinstance(value, str)
                assert value.startswith(assertion["prefix"])
        for assertion in expected.get("lengths", []):
            assert len(value_at_path(payload, assertion["path"])) == assertion["equals"]
        for assertion in expected.get("base64", []):
            value = value_at_path(payload, assertion["path"])
            assert base64.b64decode(value) == assertion["equals"].encode("utf-8")


@pytest.mark.parametrize("fixture_name", golden_fixture_names())
def test_openai_compat_golden_fixture(fixture_name: str, tmp_path: Path, monkeypatch) -> None:
    fixture = json.loads((OPENAI_COMPAT_FIXTURES / fixture_name).read_text(encoding="utf-8"))
    if fixture.get("patch") == "inpaint_inputs":
        monkeypatch.setattr("laas.app.prepare_inpaint_inputs", lambda **kwargs: ("base-image", "mask-image"))
    response = send_golden_request(golden_client(fixture["stack"], tmp_path), fixture)
    assert_golden_response(response, fixture)


def live_smoke_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def live_smoke_base_url() -> str:
    return os.environ.get("LAAS_SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def live_smoke_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {os.environ.get('LAAS_SMOKE_API_KEY', 'laas-local')}"}


def live_download_if_missing() -> bool:
    return live_smoke_enabled("LAAS_SMOKE_DOWNLOAD_IF_MISSING")


def assert_live_ok(response, label: str) -> None:
    assert response.status_code < 400, f"{label} failed: {response.status_code} {response.text}"


def silent_wav_bytes() -> bytes:
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)
    return buffer.getvalue()


def test_live_smoke_text_stack() -> None:
    if not live_smoke_enabled("LAAS_LIVE_SMOKE"):
        pytest.skip("set LAAS_LIVE_SMOKE=true to run against a live LAAS server")

    import httpx

    with httpx.Client(base_url=live_smoke_base_url(), headers=live_smoke_headers(), timeout=120.0) as client:
        assert_live_ok(client.get("/v1/models"), "models.list")
        assert_live_ok(
            client.post("/v1/local/models/load", json={"download_if_missing": live_download_if_missing()}),
            "local.models.load",
        )
        assert_live_ok(
            client.post("/v1/local/embeddings/load", json={"download_if_missing": live_download_if_missing()}),
            "local.embeddings.load",
        )
        chat = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}]})
        assert_live_ok(chat, "chat.completions")
        assert chat.json()["object"] == "chat.completion"

        response = client.post("/v1/responses", json={"input": "hello"})
        assert_live_ok(response, "responses")
        assert response.json()["object"] == "response"

        embeddings = client.post("/v1/embeddings", json={"input": "alpha", "dimensions": 8})
        assert_live_ok(embeddings, "embeddings")
        assert embeddings.json()["object"] == "list"


def test_live_smoke_image_stack() -> None:
    if not live_smoke_enabled("LAAS_LIVE_SMOKE_IMAGES"):
        pytest.skip("set LAAS_LIVE_SMOKE_IMAGES=true to run image smoke tests against a live LAAS server")

    import httpx

    with httpx.Client(base_url=live_smoke_base_url(), headers=live_smoke_headers(), timeout=600.0) as client:
        assert_live_ok(
            client.post("/v1/local/images/load", json={"download_if_missing": live_download_if_missing()}),
            "local.images.load",
        )
        image = client.post(
            "/v1/images/generations",
            json={
                "prompt": "a small brass table lamp, realistic lighting",
                "size": "512x512",
                "response_format": "b64_json",
                "n": 1,
                "seed": 42,
            },
        )
        assert_live_ok(image, "images.generations")
        assert base64.b64decode(image.json()["data"][0]["b64_json"])


def test_live_smoke_voice_stack() -> None:
    if not live_smoke_enabled("LAAS_LIVE_SMOKE_VOICE"):
        pytest.skip("set LAAS_LIVE_SMOKE_VOICE=true to run voice smoke tests against a live LAAS server")

    import httpx

    with httpx.Client(base_url=live_smoke_base_url(), headers=live_smoke_headers(), timeout=300.0) as client:
        assert_live_ok(
            client.post("/v1/local/voice/load", json={"download_if_missing": live_download_if_missing()}),
            "local.voice.load",
        )
        speech = client.post(
            "/v1/audio/speech",
            json={"model": "tts-1", "input": "hello from LAAS", "voice": "alloy", "response_format": "wav"},
        )
        assert_live_ok(speech, "audio.speech")
        assert speech.content

        transcription = client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1", "response_format": "json"},
            files={"file": ("silence.wav", silent_wav_bytes(), "audio/wav")},
        )
        assert_live_ok(transcription, "audio.transcriptions")
        assert "text" in transcription.json()


def test_models_and_local_status(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    models = client.get("/v1/models").json()
    assert models["object"] == "list"
    assert models["data"][0]["id"] == "gemma-4-e4b-it-q4_k_m"
    assert any(model["id"] == "bge-small-en-v1.5" for model in models["data"])
    assert any(model["id"] == "sdxl-turbo" for model in models["data"])
    assert any(model["id"] == "sd-1.5-inpainting" for model in models["data"])

    embedding_model = client.get("/v1/models/bge-small-en-v1.5").json()
    assert embedding_model["id"] == "bge-small-en-v1.5"
    image_model = client.get("/v1/models/sdxl-turbo").json()
    assert image_model["id"] == "sdxl-turbo"
    image_edit_model = client.get("/v1/models/sd-1.5-inpainting").json()
    assert image_edit_model["id"] == "sd-1.5-inpainting"

    status = client.get("/v1/local/models/status").json()
    assert status["configured_model"] == "gemma-4-e4b-it-q4_k_m"
    assert status["downloaded"] is True
    assert status["mmproj_downloaded"] is True
    assert status["mmproj_required"] is True
    assert status["is_loaded"] is False
    assert status["capabilities"]["vision"] is True
    assert status["capabilities"]["video"] is True
    assert status["capabilities"]["audio_input"] is False

    image_status = client.get("/v1/local/images/status/all").json()
    assert image_status["generation"]["configured_model"] == "sdxl-turbo"
    assert image_status["edit"]["configured_model"] == "sd-1.5-inpainting"


def test_diagnostics_endpoint_reports_runtime_and_actions(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        model_dir=tmp_path,
        settings_file=tmp_path / "settings.json",
        tts_ffmpeg_path="definitely-missing-ffmpeg",
    )
    client = make_client(tmp_path)
    client.app.state.settings.tts_ffmpeg_path = settings.tts_ffmpeg_path

    response = client.get("/v1/local/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "local.diagnostics"
    assert payload["python"]["executable"]
    assert payload["settings"]["model_dir"] == str(tmp_path)
    assert "llama_cpp" in payload["packages"]
    assert payload["ffmpeg"]["available"] is False
    assert any(action["component"] == "ffmpeg" for action in payload["actions"])


def test_collect_diagnostics_reports_missing_optional_import(monkeypatch, tmp_path: Path) -> None:
    import importlib.util

    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name == "llama_cpp":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    report = collect_diagnostics(Settings(model_dir=tmp_path, settings_file=tmp_path / "settings.json"))

    assert report["packages"]["llama_cpp"]["available"] is False
    assert any(action["component"] == "llama_cpp" for action in report["actions"])


def test_cli_diagnose_prints_json(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    laas_main(["diagnose"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["object"] == "local.diagnostics"
    assert "packages" in payload


def test_compatibility_matrix_and_unsupported_openai_endpoints(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    matrix = client.get("/v1/local/compatibility")
    assert matrix.status_code == 200
    payload = matrix.json()
    assert payload["object"] == "local.compatibility_matrix"
    assert any(item["surface"] == "Embeddings" and item["status"] == "supported" for item in payload["data"])
    assert any(item["surface"] == "Files" and item["status"] == "supported" for item in payload["data"])
    assert any(item["surface"] == "Vector Stores" and item["status"] == "supported" for item in payload["data"])
    assert any(item["surface"] == "Batches" and item["status"] == "supported" for item in payload["data"])
    assert any(item["surface"] == "Moderations" and item["status"] == "supported" for item in payload["data"])

    files = client.get("/v1/files")
    assert files.status_code == 200
    assert files.json()["data"] == []

    batch = client.post("/v1/batches", json={})
    assert batch.status_code == 400
    assert batch.json()["detail"]["error"]["param"] == "input_file_id"

    fine_tuning = client.post("/v1/fine_tuning/jobs", json={})
    assert fine_tuning.status_code == 501
    assert fine_tuning.json()["detail"]["error"]["param"] == "endpoint"


def test_files_api_persists_lists_serves_and_deletes(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    status = client.get("/v1/local/files/status").json()
    assert status["root"] == str(tmp_path / "file-storage")

    created = client.post(
        "/v1/files",
        data={"purpose": "assistants"},
        files={"file": ("notes.txt", b"alpha beta gamma", "text/plain")},
    )
    assert created.status_code == 200
    file_payload = created.json()
    assert file_payload["object"] == "file"
    assert file_payload["filename"] == "notes.txt"
    assert file_payload["bytes"] == len(b"alpha beta gamma")

    listed = client.get("/v1/files").json()
    assert listed["object"] == "list"
    assert listed["data"][0]["id"] == file_payload["id"]

    retrieved = client.get(f"/v1/files/{file_payload['id']}").json()
    assert retrieved["filename"] == "notes.txt"

    content = client.get(f"/v1/files/{file_payload['id']}/content")
    assert content.status_code == 200
    assert content.content == b"alpha beta gamma"

    deleted = client.delete(f"/v1/files/{file_payload['id']}").json()
    assert deleted == {"id": file_payload["id"], "object": "file.deleted", "deleted": True}
    missing = client.get(f"/v1/files/{file_payload['id']}")
    assert missing.status_code == 404


def test_storage_prune_dry_run_and_preserves_referenced_files(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    stale = client.post(
        "/v1/files",
        data={"purpose": "assistants"},
        files={"file": ("old.txt", b"stale content", "text/plain")},
    ).json()
    referenced = client.post(
        "/v1/files",
        data={"purpose": "assistants"},
        files={"file": ("referenced.txt", b"referenced vector content", "text/plain")},
    ).json()
    store = client.post("/v1/vector_stores", json={"name": "docs"}).json()
    client.post(f"/v1/vector_stores/{store['id']}/files", json={"file_id": referenced["id"]})

    old_timestamp = int(time.time()) - 181 * 86400
    with sqlite3.connect(tmp_path / "file-storage" / "laas.sqlite3") as con:
        con.execute("update files set created_at = ?", (old_timestamp,))

    dry_run = client.post("/v1/local/storage/prune", json={"older_than_days": 180, "dry_run": True}).json()
    assert dry_run["dry_run"] is True
    assert dry_run["counts"]["files"] == 1
    assert dry_run["files"][0]["id"] == stale["id"]
    assert client.get(f"/v1/files/{stale['id']}").status_code == 200

    pruned = client.post("/v1/local/storage/prune", json={"older_than_days": 180}).json()
    assert pruned["counts"]["files"] == 1
    assert client.get(f"/v1/files/{stale['id']}").status_code == 404
    assert client.get(f"/v1/files/{referenced['id']}").status_code == 200

    vacuum = client.post("/v1/local/storage/vacuum")
    assert vacuum.status_code == 200
    assert vacuum.json()["object"] == "local.storage_vacuum"


def test_vector_stores_attach_index_search_and_delete(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    file_response = client.post(
        "/v1/files",
        data={"purpose": "assistants"},
        files={
            "file": (
                "manual.md",
                b"Vulkan setup uses a GPU runtime. Banana bread belongs in a kitchen.",
                "text/markdown",
            )
        },
    ).json()
    store = client.post("/v1/vector_stores", json={"name": "docs", "metadata": {"suite": "test"}}).json()
    assert store["object"] == "vector_store"
    assert store["name"] == "docs"

    attached = client.post(
        f"/v1/vector_stores/{store['id']}/files",
        json={"file_id": file_response["id"]},
    )
    assert attached.status_code == 200
    assert attached.json()["status"] == "completed"

    files = client.get(f"/v1/vector_stores/{store['id']}/files").json()
    assert files["data"][0]["id"] == file_response["id"]

    refreshed = client.get(f"/v1/vector_stores/{store['id']}").json()
    assert refreshed["file_counts"]["completed"] == 1
    assert refreshed["metadata"] == {"suite": "test"}

    search = client.post(
        f"/v1/local/vector_stores/{store['id']}/search",
        json={"query": "Vulkan GPU setup", "limit": 2},
    )
    assert search.status_code == 200
    result = search.json()["data"][0]
    assert result["file_id"] == file_response["id"]
    assert "Vulkan setup" in result["text"]
    assert isinstance(result["score"], float)

    detached = client.delete(f"/v1/vector_stores/{store['id']}/files/{file_response['id']}").json()
    assert detached["deleted"] is True
    assert client.get(f"/v1/vector_stores/{store['id']}").json()["file_counts"]["total"] == 0

    deleted = client.delete(f"/v1/vector_stores/{store['id']}").json()
    assert deleted == {"id": store["id"], "object": "vector_store.deleted", "deleted": True}


def test_vector_store_async_indexing_status(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    file_response = client.post(
        "/v1/files",
        data={"purpose": "assistants"},
        files={"file": ("manual.md", b"async indexing text", "text/markdown")},
    ).json()
    store = client.post("/v1/vector_stores", json={"name": "docs"}).json()

    attached = client.post(
        f"/v1/vector_stores/{store['id']}/files",
        json={"file_id": file_response["id"], "wait": False},
    )

    assert attached.status_code == 200
    assert attached.json()["status"] in {"in_progress", "completed"}
    assert attached.json()["job_id"].startswith("job_")
    status = client.get(f"/v1/local/vector_stores/{store['id']}/indexing/status")
    assert status.status_code == 200
    assert status.json()["file_counts"]["total"] == 1
    job = client.get(f"/v1/local/jobs/{attached.json()['job_id']}")
    assert job.status_code == 200
    assert job.json()["kind"] == "vector_store.index"


def test_file_search_injects_context_and_returns_metadata(tmp_path: Path) -> None:
    backend = MessageCapturingBackend()
    client = make_client_with_backend(tmp_path, backend)
    file_response = client.post(
        "/v1/files",
        data={"purpose": "assistants"},
        files={"file": ("manual.md", b"Vulkan requires the GPU runtime package.", "text/markdown")},
    ).json()
    store = client.post("/v1/vector_stores", json={"name": "docs"}).json()
    client.post(f"/v1/vector_stores/{store['id']}/files", json={"file_id": file_response["id"]})

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "How do I configure Vulkan?"}],
            "tools": [{"type": "file_search", "vector_store_ids": [store["id"]]}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["laas_file_search"]["results"][0]["file_id"] == file_response["id"]
    assert "Vulkan requires" in backend.calls[0][0]["content"]


def test_responses_file_search_returns_metadata(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    file_response = client.post(
        "/v1/files",
        data={"purpose": "assistants"},
        files={"file": ("manual.md", b"ROCm setup needs a compatible wheel.", "text/markdown")},
    ).json()
    store = client.post("/v1/vector_stores", json={"name": "docs"}).json()
    client.post(f"/v1/vector_stores/{store['id']}/files", json={"file_id": file_response["id"]})

    response = client.post(
        "/v1/responses",
        json={
            "input": "What does ROCm setup need?",
            "tools": [{"type": "file_search", "vector_store_ids": [store["id"]]}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["laas_file_search"]["results"][0]["file_id"] == file_response["id"]
    annotations = payload["output"][0]["content"][0]["annotations"]
    assert annotations[0]["type"] == "file_citation"
    assert annotations[0]["file_id"] == file_response["id"]


def test_vector_store_extracts_html_text(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    file_response = client.post(
        "/v1/files",
        data={"purpose": "assistants"},
        files={"file": ("page.html", b"<html><script>nope</script><h1>Vulkan Guide</h1></html>", "text/html")},
    ).json()
    store = client.post("/v1/vector_stores", json={"name": "docs"}).json()
    client.post(f"/v1/vector_stores/{store['id']}/files", json={"file_id": file_response["id"]})

    search = client.post(f"/v1/local/vector_stores/{store['id']}/search", json={"query": "Vulkan", "limit": 1})

    assert search.status_code == 200
    assert "Vulkan Guide" in search.json()["data"][0]["text"]
    assert "nope" not in search.json()["data"][0]["text"]


def test_batches_embeddings_jsonl_output_file(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    line = json.dumps({"custom_id": "one", "method": "POST", "url": "/v1/embeddings", "body": {"input": "alpha", "dimensions": 4}})
    input_file = client.post(
        "/v1/files",
        data={"purpose": "batch"},
        files={"file": ("batch.jsonl", (line + "\n").encode("utf-8"), "application/jsonl")},
    ).json()

    batch = client.post(
        "/v1/batches",
        json={"input_file_id": input_file["id"], "endpoint": "/v1/embeddings", "completion_window": "24h"},
    )

    assert batch.status_code == 200
    payload = batch.json()
    assert payload["object"] == "batch"
    assert payload["status"] == "completed"
    assert payload["request_counts"] == {"total": 1, "completed": 1, "failed": 0}
    output = client.get(f"/v1/files/{payload['output_file_id']}/content")
    assert output.status_code == 200
    row = json.loads(output.text.splitlines()[0])
    assert row["custom_id"] == "one"
    assert row["response"]["body"]["data"][0]["object"] == "embedding"

    persisted_client = make_client(tmp_path)
    persisted = persisted_client.get(f"/v1/batches/{payload['id']}")
    assert persisted.status_code == 200
    assert persisted.json()["output_file_id"] == payload["output_file_id"]

    jobs = client.get("/v1/local/jobs").json()
    assert any(job["kind"] == "batch" and job["metadata"].get("batch_id") == payload["id"] for job in jobs["data"])


def test_moderations_rule_endpoint(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.post("/v1/moderations", json={"input": ["hello", "I will kill you"]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["flagged"] is False
    assert payload["results"][1]["flagged"] is True
    assert payload["results"][1]["categories"]["violence"] is True


def test_compat_check_report(monkeypatch) -> None:
    import laas.compat_check as compat_check

    calls = []

    def fake_request(url: str, *, method: str, body: dict | None, timeout: float):
        assert url.startswith("http://testserver")
        calls.append((method, url, body))
        return 200, {"ok": True}

    monkeypatch.setattr(compat_check, "_request", fake_request)
    report = run_compat_check("http://testserver")

    assert report["object"] == "local.compat_check"
    assert report["ok"] is True
    assert any(item["name"] == "moderations.create" for item in report["results"])
    assert any(item["name"] == "realtime.sessions.create" for item in report["results"])
    assert any(call[1].endswith("/v1/realtime/sessions") for call in calls)


def test_load_chat_completion_and_unload(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    loaded = client.post("/v1/local/models/load", json={}).json()
    assert loaded["is_loaded"] is True

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma-4-e4b-it-q4_k_m",
            "messages": [{"role": "user", "content": "hello"}],
        },
    ).json()
    assert response["object"] == "chat.completion"
    assert response["choices"][0]["message"]["content"] == "hello"

    unloaded = client.post("/v1/local/models/unload").json()
    assert unloaded["is_loaded"] is False


def test_embeddings_endpoint_supports_float_and_base64(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.post(
        "/v1/embeddings",
        json={"model": "bge-small-en-v1.5", "input": ["alpha", "beta"], "dimensions": 8},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["model"] == "bge-small-en-v1.5"
    assert len(payload["data"]) == 2
    assert payload["data"][0]["object"] == "embedding"
    assert payload["data"][0]["index"] == 0
    assert len(payload["data"][0]["embedding"]) == 8
    assert payload["usage"]["prompt_tokens"] == 2

    encoded = client.post(
        "/v1/embeddings",
        json={"input": "alpha", "dimensions": 4, "encoding_format": "base64"},
    ).json()
    assert isinstance(encoded["data"][0]["embedding"], str)
    assert len(base64.b64decode(encoded["data"][0]["embedding"])) == 16


def test_embeddings_endpoint_validates_inputs(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    empty = client.post("/v1/embeddings", json={"input": ""})
    assert empty.status_code == 400
    unknown_model = client.post("/v1/embeddings", json={"model": "missing", "input": "hello"})
    assert unknown_model.status_code == 404


def test_embedding_status_load_and_unload(tmp_path: Path) -> None:
    client = make_embedding_client(tmp_path)

    status = client.get("/v1/local/embeddings/status").json()
    assert status["configured_model"] == "bge-small-en-v1.5"
    assert status["hf_repo_id"] == "BAAI/bge-small-en-v1.5"
    assert status["downloaded"] is True
    assert status["is_loaded"] is False
    assert status["dimensions"] == 384

    loaded = client.post("/v1/local/embeddings/load", json={}).json()
    assert loaded["is_loaded"] is True
    assert loaded["loaded_model"] == "bge-small-en-v1.5"

    response = client.post("/v1/embeddings", json={"input": "alpha", "dimensions": 8})
    assert response.status_code == 200
    assert len(response.json()["data"][0]["embedding"]) == 8

    unloaded = client.post("/v1/local/embeddings/unload").json()
    assert unloaded["is_loaded"] is False


def test_embedding_missing_model_requires_download(tmp_path: Path) -> None:
    client = make_embedding_client(tmp_path, write_model=False, auto_download=False)

    load_response = client.post("/v1/local/embeddings/load", json={"download_if_missing": False})
    assert load_response.status_code == 409
    assert load_response.json()["detail"]["error"]["code"] == "embedding_model_not_downloaded"

    embedding_response = client.post("/v1/embeddings", json={"input": "alpha"})
    assert embedding_response.status_code == 409
    assert embedding_response.json()["detail"]["error"]["code"] == "embedding_model_not_downloaded"


def test_embedding_download_endpoint_fetches_snapshot(tmp_path: Path, monkeypatch) -> None:
    client = make_embedding_client(tmp_path, write_model=False, auto_download=False)

    def fake_snapshot_download(*, repo_id, local_dir, **kwargs):
        path = Path(local_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        return str(path)

    monkeypatch.setattr("laas.embedding.snapshot_download", fake_snapshot_download)
    response = client.post("/v1/local/embeddings/download", json={})

    assert response.status_code == 200
    assert response.json()["model_id"] == "bge-small-en-v1.5"
    assert response.json()["downloaded"] is True
    assert client.get("/v1/local/embeddings/status").json()["downloaded"] is True


def test_embedding_auto_downloads_for_openai_client_path(tmp_path: Path, monkeypatch) -> None:
    client = make_embedding_client(tmp_path, write_model=False, auto_download=True)

    def fake_snapshot_download(*, repo_id, local_dir, **kwargs):
        path = Path(local_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        return str(path)

    monkeypatch.setattr("laas.embedding.snapshot_download", fake_snapshot_download)
    response = client.post("/v1/embeddings", json={"input": "alpha", "dimensions": 8})

    assert response.status_code == 200
    assert len(response.json()["data"][0]["embedding"]) == 8
    status = client.get("/v1/local/embeddings/status").json()
    assert status["downloaded"] is True
    assert status["is_loaded"] is True


def test_image_generation_status_load_generate_and_unload(tmp_path: Path) -> None:
    client, backend = make_image_client(tmp_path)

    status = client.get("/v1/local/images/status").json()
    assert status["configured_model"] == "sdxl-turbo"
    assert status["downloaded"] is True
    assert status["is_loaded"] is False
    assert status["output_dir"] == str(tmp_path / "outputs" / "images")
    assert status["output_retention_seconds"] == 86400

    loaded = client.post("/v1/local/images/load", json={}).json()
    assert loaded["is_loaded"] is True

    response = client.post(
        "/v1/images/generations",
        json={
            "model": "sdxl-turbo",
            "prompt": "a tiny robot repairing a neon sign",
            "size": "512x384",
            "response_format": "b64_json",
            "num_inference_steps": 3,
            "guidance_scale": 0.5,
            "seed": 42,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "created" in payload
    assert base64.b64decode(payload["data"][0]["b64_json"]) == b"fake-png"
    assert payload["data"][0]["revised_prompt"] == "a tiny robot repairing a neon sign"
    assert backend.calls[0]["prompt"] == "a tiny robot repairing a neon sign"
    assert backend.calls[0]["width"] == 512
    assert backend.calls[0]["height"] == 384
    assert backend.calls[0]["num_inference_steps"] == 3
    assert backend.calls[0]["guidance_scale"] == 0.5
    assert backend.calls[0]["seed"] == 42

    unloaded = client.post("/v1/local/images/unload", json={}).json()
    assert unloaded["is_loaded"] is False
    assert backend.closed is True


def test_image_generation_supports_url_response_and_multiple_outputs(tmp_path: Path) -> None:
    client, backend = make_image_client(tmp_path)

    response = client.post(
        "/v1/images/generations",
        json={
            "prompt": "a quiet workshop",
            "n": 2,
            "response_format": "url",
            "quality": "high",
            "style": "natural",
            "background": "opaque",
            "moderation": "auto",
            "seed": 100,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["data"]) == 2
    assert backend.calls[0]["seed"] == 100
    assert backend.calls[1]["seed"] == 101
    assert backend.calls[0]["num_inference_steps"] == 4
    assert backend.calls[0]["prompt"] == "a quiet workshop, natural color, realistic lighting"
    assert payload["data"][0]["revised_prompt"] == "a quiet workshop, natural color, realistic lighting"

    first_image = client.get(payload["data"][0]["url"])
    second_image = client.get(payload["data"][1]["url"])
    assert first_image.status_code == 200
    assert first_image.headers["content-type"] == "image/png"
    assert first_image.content == b"fake-png"
    assert second_image.status_code == 200
    assert len(list((tmp_path / "outputs" / "images").glob("*.png"))) == 2


def test_image_variation_supports_url_response_and_output_format(tmp_path: Path) -> None:
    client, backend = make_image_client(tmp_path)

    response = client.post(
        "/v1/images/variations",
        data={
            "model": "sdxl-turbo",
            "n": "2",
            "response_format": "url",
            "size": "512x512",
            "output_format": "webp",
            "output_compression": "80",
            "seed": "80",
        },
        files={"image": ("source.png", PNG_1X1, "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["data"]) == 2
    assert backend.calls[0]["prompt"] == "a high quality variation of the provided image, same subject, similar composition"
    assert backend.calls[0]["width"] == 512
    assert backend.calls[0]["height"] == 512
    assert backend.calls[0]["num_inference_steps"] == 4
    assert backend.calls[0]["strength"] == 0.55
    assert backend.calls[0]["seed"] == 80
    assert backend.calls[1]["seed"] == 81

    first_image = client.get(payload["data"][0]["url"])
    assert first_image.status_code == 200
    assert first_image.headers["content-type"] == "image/webp"
    assert payload["data"][0]["url"].endswith(".webp")
    assert len(list((tmp_path / "outputs" / "images").glob("*.webp"))) == 2


def test_image_variation_rejects_non_square_or_non_png_input(tmp_path: Path) -> None:
    client, _backend = make_image_client(tmp_path)

    bad_png = client.post(
        "/v1/images/variations",
        data={"size": "512x512"},
        files={"image": ("source.jpg", b"not-png", "image/jpeg")},
    )
    assert bad_png.status_code == 400
    assert bad_png.json()["detail"]["error"]["param"] == "image"

    bad_size = client.post(
        "/v1/images/variations",
        data={"size": "768x768"},
        files={"image": ("source.png", PNG_1X1, "image/png")},
    )
    assert bad_size.status_code == 400
    assert bad_size.json()["detail"]["error"]["param"] == "size"


def test_image_variation_backend_errors_are_json(tmp_path: Path) -> None:
    client, backend = make_image_client(tmp_path)

    def fail_variation(**kwargs) -> GeneratedImage:
        raise Exception("diffusers exploded")

    backend.variation = fail_variation  # type: ignore[method-assign]
    response = client.post(
        "/v1/images/variations",
        data={"size": "512x512"},
        files={"image": ("source.png", PNG_1X1, "image/png")},
    )

    assert response.status_code == 500
    assert response.json()["detail"]["error"]["code"] == "image_variation_failed"


def test_image_generation_rejects_unsupported_options(tmp_path: Path) -> None:
    client, _backend = make_image_client(tmp_path)

    unknown_model = client.post("/v1/images/generations", json={"model": "missing", "prompt": "hello"})
    assert unknown_model.status_code == 404

    transparent_response = client.post(
        "/v1/images/generations",
        json={"prompt": "hello", "background": "transparent"},
    )
    assert transparent_response.status_code == 400
    assert transparent_response.json()["detail"]["error"]["param"] == "background"

    bad_style = client.post(
        "/v1/images/generations",
        json={"prompt": "hello", "style": "cubist"},
    )
    assert bad_style.status_code == 400
    assert bad_style.json()["detail"]["error"]["param"] == "style"


def test_image_generation_missing_model_requires_download(tmp_path: Path) -> None:
    client, _backend = make_image_client(tmp_path, write_model=False, auto_download=False)

    load_response = client.post("/v1/local/images/load", json={"download_if_missing": False})
    assert load_response.status_code == 409
    assert load_response.json()["detail"]["error"]["code"] == "image_model_not_downloaded"

    generation_response = client.post("/v1/images/generations", json={"prompt": "hello"})
    assert generation_response.status_code == 409
    assert generation_response.json()["detail"]["error"]["code"] == "image_model_not_downloaded"


def test_image_generation_auto_downloads_for_openai_client_path(tmp_path: Path, monkeypatch) -> None:
    client, backend = make_image_client(tmp_path, write_model=False, auto_download=True)

    def fake_snapshot_download(*, repo_id, local_dir, **kwargs):
        path = Path(local_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model_index.json").write_text("{}", encoding="utf-8")
        return str(path)

    monkeypatch.setattr("laas.image.snapshot_download", fake_snapshot_download)
    response = client.post("/v1/images/generations", json={"prompt": "hello"})

    assert response.status_code == 200
    assert base64.b64decode(response.json()["data"][0]["b64_json"]) == b"fake-png"
    assert backend.calls[0]["prompt"] == "hello"
    status = client.get("/v1/local/images/status").json()
    assert status["downloaded"] is True
    assert status["download_in_progress"] is False
    assert status["download_started_at"] is not None
    assert status["download_finished_at"] is not None


def test_video_generation_status_load_generate_and_unload(tmp_path: Path) -> None:
    client, backend = make_video_client(tmp_path)

    status = client.get("/v1/local/videos/status").json()
    assert status["configured_model"] == "wan2.2-i2v-q3"
    assert status["downloaded"] is True
    assert status["is_loaded"] is False
    assert status["hf_repo_id"] == "QuantStack/Wan2.2-I2V-A14B-GGUF"
    assert status["diffusers_hf_repo_id"] == "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    assert status["high_noise_filename"] == "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q3_K_M.gguf"
    assert status["low_noise_filename"] == "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q3_K_M.gguf"
    assert status["vae_filename"] == "VAE/Wan2.1_VAE.safetensors"
    assert status["device"] == "auto"
    assert status["torch_dtype"] == "auto"
    assert status["guidance_scale_2"] is None
    assert status["boundary_ratio"] == 0.9

    loaded = client.post("/v1/local/videos/load", json={}).json()
    assert loaded["is_loaded"] is True

    response = client.post(
        "/v1/videos/generations",
        data={
            "model": "wan2.2-i2v-q3",
            "prompt": "a brass table lamp glowing in a quiet room",
            "size": "640x360",
            "response_format": "b64_json",
            "seconds": "3",
            "fps": "12",
            "num_inference_steps": "6",
            "guidance_scale": "1.5",
            "seed": "42",
        },
        files={"image": ("frame.png", PNG_1X1, "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert base64.b64decode(payload["data"][0]["b64_json"]) == b"fake-mp4"
    assert backend.calls[0]["prompt"] == "a brass table lamp glowing in a quiet room"
    assert backend.calls[0]["width"] == 640
    assert backend.calls[0]["height"] == 360
    assert backend.calls[0]["seconds"] == 3
    assert backend.calls[0]["fps"] == 12
    assert backend.calls[0]["num_inference_steps"] == 6
    assert backend.calls[0]["guidance_scale"] == 1.5
    assert backend.calls[0]["seed"] == 42
    assert backend.calls[0]["image_bytes"] == PNG_1X1

    unloaded = client.post("/v1/local/videos/unload", json={}).json()
    assert unloaded["is_loaded"] is False
    assert backend.closed is True


def test_video_generation_supports_url_response(tmp_path: Path) -> None:
    client, _backend = make_video_client(tmp_path)

    response = client.post(
        "/v1/videos/generations",
        data={
            "prompt": "a slow camera push across a workbench",
            "response_format": "url",
        },
        files={"image": ("frame.png", PNG_1X1, "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    video_response = client.get(payload["data"][0]["url"])
    assert video_response.status_code == 200
    assert video_response.headers["content-type"] == "video/mp4"
    assert video_response.content == b"fake-mp4"
    assert len(list((tmp_path / "outputs" / "videos").glob("*.mp4"))) == 1


def test_video_generation_rejects_unsupported_options(tmp_path: Path) -> None:
    client, _backend = make_video_client(tmp_path)

    unknown_model = client.post(
        "/v1/videos/generations",
        data={"model": "missing", "prompt": "hello"},
        files={"image": ("frame.png", PNG_1X1, "image/png")},
    )
    assert unknown_model.status_code == 404

    bad_n = client.post(
        "/v1/videos/generations",
        data={"prompt": "hello", "n": "2"},
        files={"image": ("frame.png", PNG_1X1, "image/png")},
    )
    assert bad_n.status_code == 400
    assert bad_n.json()["detail"]["error"]["param"] == "n"

    bad_size = client.post(
        "/v1/videos/generations",
        data={"prompt": "hello", "size": "big"},
        files={"image": ("frame.png", PNG_1X1, "image/png")},
    )
    assert bad_size.status_code == 400
    assert bad_size.json()["detail"]["error"]["param"] == "size"


def test_video_generation_missing_model_requires_download(tmp_path: Path) -> None:
    client, _backend = make_video_client(tmp_path, write_model=False, auto_download=False)

    load_response = client.post("/v1/local/videos/load", json={"download_if_missing": False})
    assert load_response.status_code == 409
    assert load_response.json()["detail"]["error"]["code"] == "video_model_not_downloaded"
    assert load_response.json()["detail"]["error"]["param"] == "high_noise"

    generation_response = client.post(
        "/v1/videos/generations",
        data={"prompt": "hello"},
        files={"image": ("frame.png", PNG_1X1, "image/png")},
    )
    assert generation_response.status_code == 409
    assert generation_response.json()["detail"]["error"]["code"] == "video_model_not_downloaded"


def test_video_generation_auto_downloads_configured_assets(tmp_path: Path, monkeypatch) -> None:
    client, backend = make_video_client(tmp_path, write_model=False, auto_download=True)

    def fake_hf_hub_download(*, repo_id, filename, local_dir, **kwargs):
        _ = repo_id, kwargs
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"asset")
        return str(path)

    def fake_snapshot_download(*, repo_id, local_dir, allow_patterns, **kwargs):
        _ = repo_id, allow_patterns, kwargs
        path = Path(local_dir)
        for filename in (
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors.index.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "transformer/config.json",
            "transformer_2/config.json",
            "vae/config.json",
        ):
            file_path = path / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("{}", encoding="utf-8")
        return str(path)

    monkeypatch.setattr("laas.video.hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr("laas.video.snapshot_download", fake_snapshot_download)
    response = client.post(
        "/v1/videos/generations",
        data={"prompt": "hello"},
        files={"image": ("frame.png", PNG_1X1, "image/png")},
    )

    assert response.status_code == 200
    assert base64.b64decode(response.json()["data"][0]["b64_json"]) == b"fake-mp4"
    assert backend.calls[0]["prompt"] == "hello"
    status = client.get("/v1/local/videos/status").json()
    assert status["downloaded"] is True
    assert status["download_in_progress"] is False
    assert status["download_started_at"] is not None
    assert status["download_finished_at"] is not None


def test_video_generation_frame_count_matches_wan_temporal_stride() -> None:
    assert frames_for_duration(seconds=4.0, fps=16) == 65
    assert frames_for_duration(seconds=5.0, fps=16) == 81


def test_video_default_backend_is_native_diffusers(tmp_path: Path) -> None:
    settings = Settings(model_dir=tmp_path, settings_file=tmp_path / "settings.json")
    backend = VideoManager._default_backend_factory(settings.video_generation_model_path, settings)
    assert isinstance(backend, DiffusersWanVideoBackend)


def test_image_auto_download_status_is_observable_during_load(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        model_dir=tmp_path,
        settings_file=tmp_path / "settings.json",
        image_idle_unload_seconds=0,
    )
    image_backend = FakeImageBackend()
    image_manager = ImageManager(settings, backend_factory=lambda model_path, active_settings: image_backend)
    started = threading.Event()
    release = threading.Event()

    def fake_snapshot_download(*, repo_id, local_dir, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        path = Path(local_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model_index.json").write_text("{}", encoding="utf-8")
        return str(path)

    monkeypatch.setattr("laas.image.snapshot_download", fake_snapshot_download)

    error: list[BaseException] = []

    def load_model() -> None:
        try:
            image_manager.load(download_if_missing=True)
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=load_model)
    thread.start()
    assert started.wait(timeout=5)

    status = image_manager.status()
    assert status.download_in_progress is True
    assert status.downloaded is False
    assert status.download_started_at is not None
    assert status.download_finished_at is None

    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert error == []
    assert image_manager.status().is_loaded is True


def test_image_status_does_not_block_when_manager_is_busy(tmp_path: Path) -> None:
    settings = Settings(model_dir=tmp_path, settings_file=tmp_path / "settings.json")
    image_manager = ImageManager(settings, backend_factory=lambda model_path, active_settings: FakeImageBackend())
    settings.image_model_path.mkdir(parents=True, exist_ok=True)
    (settings.image_model_path / "model_index.json").write_text("{}", encoding="utf-8")

    image_manager._start_job("variation")
    image_manager._lock.acquire()
    try:
        status = image_manager.status()
    finally:
        image_manager._lock.release()
        image_manager._finish_job()

    assert status.active_jobs == 1
    assert status.current_operation == "variation"
    assert status.downloaded is True


def test_image_download_endpoint_fetches_snapshot(tmp_path: Path, monkeypatch) -> None:
    client, _backend = make_image_client(tmp_path, write_model=False)

    def fake_snapshot_download(*, repo_id, local_dir, **kwargs):
        path = Path(local_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model_index.json").write_text("{}", encoding="utf-8")
        return str(path)

    monkeypatch.setattr("laas.image.snapshot_download", fake_snapshot_download)
    response = client.post("/v1/local/images/download", json={})

    assert response.status_code == 200
    assert response.json()["downloaded"] is True
    assert response.json()["model_id"] == "sdxl-turbo"


def test_image_edit_status_load_edit_and_unload(tmp_path: Path, monkeypatch) -> None:
    client, backend = make_image_edit_client(tmp_path)

    status = client.get("/v1/local/images/edit/status").json()
    assert status["configured_model"] == "sd-1.5-inpainting"
    assert status["downloaded"] is True
    assert status["is_loaded"] is False
    assert status["default_size"] == "512x512"
    assert status["strength"] == 0.8

    loaded = client.post("/v1/local/images/edit/load", json={}).json()
    assert loaded["is_loaded"] is True

    monkeypatch.setattr("laas.app.prepare_inpaint_inputs", lambda **kwargs: ("base-image", "mask-image"))
    response = client.post(
        "/v1/images/edits",
        data={
            "model": "sd-1.5-inpainting",
            "prompt": "add a brass lamp",
            "size": "512x512",
            "response_format": "b64_json",
            "quality": "high",
            "input_fidelity": "high",
            "negative_prompt": "blurry",
            "strength": "0.9",
            "seed": "10",
        },
        files={
            "image": ("base.png", b"base-bytes", "image/png"),
            "mask": ("mask.png", b"mask-bytes", "image/png"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert base64.b64decode(payload["data"][0]["b64_json"]) == b"fake-edited-png"
    assert backend.calls[0]["prompt"] == "add a brass lamp"
    assert backend.calls[0]["negative_prompt"] == "blurry"
    assert backend.calls[0]["image"] == "base-image"
    assert backend.calls[0]["mask_image"] == "mask-image"
    assert backend.calls[0]["width"] == 512
    assert backend.calls[0]["height"] == 512
    assert backend.calls[0]["num_inference_steps"] == 35
    assert backend.calls[0]["guidance_scale"] == 7.5
    assert backend.calls[0]["strength"] == 0.65
    assert backend.calls[0]["seed"] == 10

    unloaded = client.post("/v1/local/images/edit/unload", json={}).json()
    assert unloaded["is_loaded"] is False
    assert backend.closed is True


def test_unload_all_image_models_unloads_generation_and_edit(tmp_path: Path) -> None:
    settings = Settings(
        model_dir=tmp_path,
        settings_file=tmp_path / "settings.json",
        idle_unload_seconds=0,
        image_idle_unload_seconds=0,
        image_edit_idle_unload_seconds=0,
    )
    text_manager = ModelManager(settings, backend_factory=lambda model_path, active_settings: EchoBackend())
    image_backend = FakeImageBackend()
    image_edit_backend = FakeImageEditBackend()
    image_manager = ImageManager(settings, backend_factory=lambda model_path, active_settings: image_backend)
    image_edit_manager = ImageEditManager(settings, backend_factory=lambda model_path, active_settings: image_edit_backend)

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"model")
    if settings.mmproj_path:
        settings.mmproj_path.write_bytes(b"mmproj")
    settings.image_model_path.mkdir(parents=True, exist_ok=True)
    (settings.image_model_path / "model_index.json").write_text("{}", encoding="utf-8")
    settings.image_edit_model_path.mkdir(parents=True, exist_ok=True)
    (settings.image_edit_model_path / "model_index.json").write_text("{}", encoding="utf-8")

    client = TestClient(
        create_app(
            settings=settings,
            manager=text_manager,
            image_manager=image_manager,
            image_edit_manager=image_edit_manager,
        )
    )
    assert client.post("/v1/local/images/load", json={}).json()["is_loaded"] is True
    assert client.post("/v1/local/images/edit/load", json={}).json()["is_loaded"] is True

    response = client.post("/v1/local/images/unload/all")
    assert response.status_code == 200
    payload = response.json()
    assert payload["is_loaded"] is False
    assert payload["generation"]["is_loaded"] is False
    assert payload["edit"]["is_loaded"] is False
    assert image_backend.closed is True
    assert image_edit_backend.closed is True


def test_image_exclusive_load_unloads_other_image_pipeline(tmp_path: Path) -> None:
    settings = Settings(
        model_dir=tmp_path,
        settings_file=tmp_path / "settings.json",
        idle_unload_seconds=0,
        image_idle_unload_seconds=0,
        image_edit_idle_unload_seconds=0,
        image_exclusive_load=True,
    )
    text_manager = ModelManager(settings, backend_factory=lambda model_path, active_settings: EchoBackend())
    image_backend = FakeImageBackend()
    image_edit_backend = FakeImageEditBackend()
    image_manager = ImageManager(settings, backend_factory=lambda model_path, active_settings: image_backend)
    image_edit_manager = ImageEditManager(settings, backend_factory=lambda model_path, active_settings: image_edit_backend)

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"model")
    if settings.mmproj_path:
        settings.mmproj_path.write_bytes(b"mmproj")
    settings.image_model_path.mkdir(parents=True, exist_ok=True)
    (settings.image_model_path / "model_index.json").write_text("{}", encoding="utf-8")
    settings.image_edit_model_path.mkdir(parents=True, exist_ok=True)
    (settings.image_edit_model_path / "model_index.json").write_text("{}", encoding="utf-8")

    client = TestClient(
        create_app(
            settings=settings,
            manager=text_manager,
            image_manager=image_manager,
            image_edit_manager=image_edit_manager,
        )
    )

    assert client.post("/v1/local/images/edit/load", json={}).json()["is_loaded"] is True
    assert client.post("/v1/local/images/load", json={}).json()["is_loaded"] is True
    assert image_edit_backend.closed is True
    assert client.get("/v1/local/images/edit/status").json()["is_loaded"] is False

    assert client.post("/v1/local/images/edit/load", json={}).json()["is_loaded"] is True
    assert image_backend.closed is True
    assert client.get("/v1/local/images/status").json()["is_loaded"] is False


def test_unload_all_local_models_unloads_text_and_image_stacks(tmp_path: Path) -> None:
    settings = Settings(
        model_dir=tmp_path,
        settings_file=tmp_path / "settings.json",
        idle_unload_seconds=0,
        image_idle_unload_seconds=0,
        image_edit_idle_unload_seconds=0,
        video_generation_idle_unload_seconds=0,
        image_exclusive_load=False,
    )
    text_manager = ModelManager(settings, backend_factory=lambda model_path, active_settings: EchoBackend())
    image_backend = FakeImageBackend()
    image_edit_backend = FakeImageEditBackend()
    video_backend = FakeVideoBackend()
    image_manager = ImageManager(settings, backend_factory=lambda model_path, active_settings: image_backend)
    image_edit_manager = ImageEditManager(settings, backend_factory=lambda model_path, active_settings: image_edit_backend)
    video_manager = VideoManager(settings, backend_factory=lambda model_path, active_settings: video_backend)

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"model")
    if settings.mmproj_path:
        settings.mmproj_path.write_bytes(b"mmproj")
    settings.image_model_path.mkdir(parents=True, exist_ok=True)
    (settings.image_model_path / "model_index.json").write_text("{}", encoding="utf-8")
    settings.image_edit_model_path.mkdir(parents=True, exist_ok=True)
    (settings.image_edit_model_path / "model_index.json").write_text("{}", encoding="utf-8")
    settings.video_generation_high_noise_path.parent.mkdir(parents=True, exist_ok=True)
    settings.video_generation_low_noise_path.parent.mkdir(parents=True, exist_ok=True)
    settings.video_generation_vae_path.parent.mkdir(parents=True, exist_ok=True)
    settings.video_generation_high_noise_path.write_bytes(b"high")
    settings.video_generation_low_noise_path.write_bytes(b"low")
    settings.video_generation_vae_path.write_bytes(b"vae")
    for filename in (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "transformer_2/config.json",
        "vae/config.json",
    ):
        path = settings.video_generation_diffusers_model_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    client = TestClient(
        create_app(
            settings=settings,
            manager=text_manager,
            image_manager=image_manager,
            image_edit_manager=image_edit_manager,
            video_manager=video_manager,
        )
    )

    assert client.post("/v1/local/models/load", json={}).json()["is_loaded"] is True
    assert client.post("/v1/local/images/load", json={}).json()["is_loaded"] is True
    assert client.post("/v1/local/images/edit/load", json={}).json()["is_loaded"] is True
    assert client.post("/v1/local/videos/load", json={}).json()["is_loaded"] is True

    response = client.post("/v1/local/unload/all")
    assert response.status_code == 200
    payload = response.json()
    assert payload["is_loaded"] is False
    assert payload["text"]["is_loaded"] is False
    assert payload["images"]["generation"]["is_loaded"] is False
    assert payload["images"]["edit"]["is_loaded"] is False
    assert payload["video"]["is_loaded"] is False
    assert image_backend.closed is True
    assert image_edit_backend.closed is True
    assert video_backend.closed is True


def test_image_edit_supports_url_response_and_multiple_outputs(tmp_path: Path, monkeypatch) -> None:
    client, backend = make_image_edit_client(tmp_path)

    monkeypatch.setattr("laas.app.prepare_inpaint_inputs", lambda **kwargs: ("base-image", "mask-image"))
    response = client.post(
        "/v1/images/edits",
        data={
            "prompt": "replace the window with a painting",
            "n": "2",
            "response_format": "url",
            "seed": "50",
        },
        files={
            "image": ("base.png", b"base-bytes", "image/png"),
            "mask": ("mask.png", b"mask-bytes", "image/png"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["data"]) == 2
    assert backend.calls[0]["seed"] == 50
    assert backend.calls[1]["seed"] == 51
    first_image = client.get(payload["data"][0]["url"])
    assert first_image.status_code == 200
    assert first_image.headers["content-type"] == "image/png"
    assert first_image.content == b"fake-edited-png"
    assert len(list((tmp_path / "outputs" / "images").glob("*.png"))) == 2


def test_image_edit_rejects_unsupported_options(tmp_path: Path) -> None:
    client, _backend = make_image_edit_client(tmp_path)

    unknown_model = client.post(
        "/v1/images/edits",
        data={"model": "missing", "prompt": "hello"},
        files={"image": ("base.png", b"base-bytes", "image/png")},
    )
    assert unknown_model.status_code == 404

    transparent = client.post(
        "/v1/images/edits",
        data={"prompt": "hello", "background": "transparent"},
        files={"image": ("base.png", b"base-bytes", "image/png")},
    )
    assert transparent.status_code == 400
    assert transparent.json()["detail"]["error"]["param"] == "background"

    output_format = client.post(
        "/v1/images/edits",
        data={"prompt": "hello", "output_format": "gif"},
        files={"image": ("base.png", b"base-bytes", "image/png")},
    )
    assert output_format.status_code == 400
    assert output_format.json()["detail"]["error"]["param"] == "output_format"


def test_image_edit_missing_model_requires_download(tmp_path: Path, monkeypatch) -> None:
    client, _backend = make_image_edit_client(tmp_path, write_model=False, auto_download=False)

    load_response = client.post("/v1/local/images/edit/load", json={"download_if_missing": False})
    assert load_response.status_code == 409
    assert load_response.json()["detail"]["error"]["code"] == "image_edit_model_not_downloaded"

    monkeypatch.setattr("laas.app.prepare_inpaint_inputs", lambda **kwargs: ("base-image", "mask-image"))
    edit_response = client.post(
        "/v1/images/edits",
        data={"prompt": "hello"},
        files={
            "image": ("base.png", b"base-bytes", "image/png"),
            "mask": ("mask.png", b"mask-bytes", "image/png"),
        },
    )
    assert edit_response.status_code == 409
    assert edit_response.json()["detail"]["error"]["code"] == "image_edit_model_not_downloaded"


def test_image_edit_auto_downloads_for_openai_client_path(tmp_path: Path, monkeypatch) -> None:
    client, backend = make_image_edit_client(tmp_path, write_model=False, auto_download=True)

    def fake_snapshot_download(*, repo_id, local_dir, **kwargs):
        path = Path(local_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model_index.json").write_text("{}", encoding="utf-8")
        return str(path)

    monkeypatch.setattr("laas.image.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("laas.app.prepare_inpaint_inputs", lambda **kwargs: ("base-image", "mask-image"))
    response = client.post(
        "/v1/images/edits",
        data={"prompt": "hello"},
        files={
            "image": ("base.png", b"base-bytes", "image/png"),
            "mask": ("mask.png", b"mask-bytes", "image/png"),
        },
    )

    assert response.status_code == 200
    assert base64.b64decode(response.json()["data"][0]["b64_json"]) == b"fake-edited-png"
    assert backend.calls[0]["prompt"] == "hello"
    status = client.get("/v1/local/images/edit/status").json()
    assert status["downloaded"] is True
    assert status["download_in_progress"] is False
    assert status["download_started_at"] is not None
    assert status["download_finished_at"] is not None


def test_image_edit_download_endpoint_fetches_snapshot(tmp_path: Path, monkeypatch) -> None:
    client, _backend = make_image_edit_client(tmp_path, write_model=False)

    def fake_snapshot_download(*, repo_id, local_dir, **kwargs):
        path = Path(local_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model_index.json").write_text("{}", encoding="utf-8")
        return str(path)

    monkeypatch.setattr("laas.image.snapshot_download", fake_snapshot_download)
    response = client.post("/v1/local/images/edit/download", json={})

    assert response.status_code == 200
    assert response.json()["downloaded"] is True
    assert response.json()["model_id"] == "sd-1.5-inpainting"


def test_tool_call_translation(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "call_tool"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
    ).json()
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_chat_completion_golden_response_shape(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gemma-4-e4b-it-q4_k_m", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"].startswith("chatcmpl")
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "gemma-4-e4b-it-q4_k_m"
    assert payload["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }
    ]
    assert payload["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_chat_completion_tool_choice_object_selects_declared_function(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "call_tool"}],
            "tool_choice": {"type": "function", "function": {"name": "lookup_time"}},
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
                },
                {
                    "type": "function",
                    "function": {"name": "lookup_time", "parameters": {"type": "object", "properties": {}}},
                },
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"] == {"name": "lookup_time", "arguments": "{}"}


def test_chat_completion_tool_choice_none_suppresses_tool_translation(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "call_tool"}],
            "tool_choice": "none",
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
                }
            ],
        },
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message == {"role": "assistant", "content": "call_tool"}


def test_chat_completion_tool_choice_required_allows_declared_function(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "call_tool"}],
            "tool_choice": "required",
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_chat_completion_unknown_tool_choice_is_openai_error(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "call_tool"}],
            "tool_choice": {"type": "function", "function": {"name": "missing_tool"}},
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
                }
            ],
        },
    )

    assert response.status_code == 400
    error = response.json()["detail"]["error"]
    assert error["param"] == "tool_choice"
    assert "missing_tool" in error["message"]


def test_chat_completion_streaming_text_is_normalized(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
    )

    payloads = sse_payloads(response)
    assert payloads[-1] == "[DONE]"
    chunks = payloads[:-1]
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["model"] == "gemma-4-e4b-it-q4_k_m"
    assert chunks[0]["choices"][0]["delta"]["content"] == "hello"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_streaming_gemma_tool_markup_becomes_tool_call_delta(tmp_path: Path) -> None:
    client = make_client_with_backend(tmp_path, StreamingToolBackend())

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "call_tool"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
    )

    payloads = sse_payloads(response)
    chunks = payloads[:-1]
    serialized = json.dumps(chunks)
    assert r"D:\\AI\\Models" not in serialized
    assert "<|tool_call" not in serialized
    assert all(chunk["model"] == "gemma-4-e4b-it-q4_k_m" for chunk in chunks)

    tool_delta = next(chunk["choices"][0]["delta"] for chunk in chunks if "tool_calls" in chunk["choices"][0]["delta"])
    tool_call = tool_delta["tool_calls"][0]
    assert tool_call["function"]["name"] == "get_weather"
    assert tool_call["function"]["arguments"] == '{"location":"Chicago"}'
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_responses_streaming_tool_calls_use_responses_events(tmp_path: Path) -> None:
    client = make_client_with_backend(tmp_path, StreamingToolBackend())

    response = client.post(
        "/v1/responses",
        json={
            "input": "call_tool",
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        },
    )

    payloads = sse_payloads(response)
    events = payloads[:-1]
    assert events[0]["type"] == "response.created"
    assert any(event["type"] == "response.output_item.added" for event in events)
    argument_delta = next(event for event in events if event["type"] == "response.function_call_arguments.delta")
    assert argument_delta["delta"] == '{"location":"Chicago"}'
    completed = next(event for event in events if event["type"] == "response.completed")
    assert completed["response"]["model"] == "gemma-4-e4b-it-q4_k_m"
    assert completed["response"]["output"][0]["name"] == "get_weather"


def test_responses_api_function_call_golden_shape(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/v1/responses",
        json={
            "model": "gemma-4-e4b-it-q4_k_m",
            "input": "call_tool",
            "tool_choice": {"type": "function", "name": "get_weather"},
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"].startswith("resp_")
    assert payload["object"] == "response"
    assert payload["status"] == "completed"
    assert payload["model"] == "gemma-4-e4b-it-q4_k_m"
    assert payload["output_text"] == ""
    assert payload["usage"] == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    output = payload["output"]
    assert len(output) == 1
    assert output[0]["type"] == "function_call"
    assert output[0]["call_id"].startswith("call_")
    assert output[0]["name"] == "get_weather"
    assert output[0]["arguments"] == "{}"
    assert output[0]["status"] == "completed"


def test_chat_completion_sampling_params_are_forwarded(tmp_path: Path) -> None:
    settings = Settings(model_dir=tmp_path, settings_file=tmp_path / "settings.json")
    backend = CapturingBackend()
    manager = ModelManager(settings, backend_factory=lambda model_path, active_settings: backend)
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"model")
    if settings.mmproj_path:
        settings.mmproj_path.write_bytes(b"mmproj")
    client = TestClient(create_app(settings=settings, manager=manager))

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "stop": ["<end>"],
            "seed": 123,
            "top_k": 32,
            "min_p": 0.1,
            "repeat_penalty": 1.05,
            "response_format": {"type": "json_object"},
        },
    )

    assert response.status_code == 200
    assert backend.chat_params == {
        "stop": ["<end>"],
        "seed": 123,
        "repeat_penalty": 1.05,
        "top_k": 32,
        "min_p": 0.1,
        "response_format": {"type": "json_object"},
    }


def test_completion_sampling_params_are_forwarded(tmp_path: Path) -> None:
    settings = Settings(model_dir=tmp_path, settings_file=tmp_path / "settings.json")
    backend = CapturingBackend()
    manager = ModelManager(settings, backend_factory=lambda model_path, active_settings: backend)
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"model")
    if settings.mmproj_path:
        settings.mmproj_path.write_bytes(b"mmproj")
    client = TestClient(create_app(settings=settings, manager=manager))

    response = client.post(
        "/v1/completions",
        json={
            "prompt": "hello",
            "suffix": "done",
            "stop": "END",
            "seed": 456,
            "top_k": 16,
            "typical_p": 0.9,
            "echo": True,
        },
    )

    assert response.status_code == 200
    assert backend.completion_params == {
        "suffix": "done",
        "stop": "END",
        "seed": 456,
        "top_k": 16,
        "typical_p": 0.9,
        "echo": True,
    }


def test_responses_sampling_params_are_forwarded(tmp_path: Path) -> None:
    settings = Settings(model_dir=tmp_path, settings_file=tmp_path / "settings.json")
    backend = CapturingBackend()
    manager = ModelManager(settings, backend_factory=lambda model_path, active_settings: backend)
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"model")
    if settings.mmproj_path:
        settings.mmproj_path.write_bytes(b"mmproj")
    client = TestClient(create_app(settings=settings, manager=manager))

    response = client.post(
        "/v1/responses",
        json={
            "input": "hello",
            "stop": "END",
            "seed": 789,
            "top_k": 8,
            "text": {"format": {"type": "json_object"}},
        },
    )

    assert response.status_code == 200
    assert backend.chat_params == {
        "stop": "END",
        "seed": 789,
        "top_k": 8,
        "response_format": {"type": "json_object"},
    }


def test_gemma_native_tool_call_translation() -> None:
    text = '<|tool_call>call:get_weather{location:<|"|>Chicago<|"|>}<tool_call|>'
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]

    calls = parse_tool_calls(text, tools)

    assert calls[0]["function"]["name"] == "get_weather"
    assert calls[0]["function"]["arguments"] == '{"location":"Chicago"}'
    assert remove_tool_call_markup(text) == ""


def test_tool_call_translation_handles_nested_json_arguments() -> None:
    text = (
        'prefix <tool_call>{"name":"get_weather","arguments":{"location":{"city":"Chicago"},'
        '"units":"fahrenheit"}}</tool_call> suffix'
    )
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]

    calls = parse_tool_calls(text, tools)

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert calls[0]["function"]["arguments"] == '{"location":{"city":"Chicago"},"units":"fahrenheit"}'
    assert remove_tool_call_markup(text) == "prefix  suffix"


def test_tool_call_translation_filters_unknown_and_unselected_tools() -> None:
    text = (
        '<tool_call>{"name":"get_weather","arguments":{}}</tool_call>'
        '<tool_call>{"name":"lookup_time","arguments":{}}</tool_call>'
        '<tool_call>{"name":"delete_file","arguments":{}}</tool_call>'
    )
    tools = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "lookup_time", "parameters": {"type": "object"}}},
    ]

    calls = parse_tool_calls(text, tools, {"type": "function", "function": {"name": "lookup_time"}})

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "lookup_time"
    assert parse_tool_calls(text, tools, "none") == []


def test_malformed_tool_markup_remains_text_without_crashing() -> None:
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]

    chat = _normalize_chat_response(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '<tool_call>{"name":"get_weather","arguments":'},
                    "finish_reason": "stop",
                }
            ]
        },
        "gemma-4-e4b-it-q4_k_m",
        tools,
    )

    message = chat["choices"][0]["message"]
    assert message["content"] == '<tool_call>{"name":"get_weather","arguments":'
    assert "tool_calls" not in message
    assert chat["choices"][0]["finish_reason"] == "stop"


def test_responses_api_text_and_image_parts(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/v1/responses",
        json={
            "model": "gemma-4-e4b-it-q4_k_m",
            "instructions": "Be terse.",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "what is here?"},
                        {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                    ],
                }
            ],
        },
    ).json()
    assert response["object"] == "response"
    assert response["status"] == "completed"
    assert "what is here?" in response["output_text"]


def test_video_frames_translate_to_image_parts(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/v1/responses",
        json={
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "summarize video"},
                        {
                            "type": "input_video",
                            "frames": ["data:image/jpeg;base64,AA==", "data:image/jpeg;base64,AA=="],
                        },
                    ],
                }
            ]
        },
    ).json()
    assert response["status"] == "completed"
    assert "summarize video" in response["output_text"]


def test_chat_completion_rejects_input_audio_by_default(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(b"fake wav").decode("ascii"),
                                "format": "wav",
                            },
                        }
                    ],
                }
            ]
        },
    )

    assert response.status_code == 400
    error = response.json()["detail"]["error"]
    assert error["param"] == "messages"
    assert "audio inputs" in error["message"]


def test_chat_completion_input_audio_validates_and_reaches_backend_when_enabled(tmp_path: Path) -> None:
    backend = MessageCapturingBackend()
    client = make_client_with_backend(tmp_path, backend, llm_audio_input_enabled=True)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this sound"},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(b"fake wav").decode("ascii"),
                                "format": "WAV",
                            },
                        },
                    ],
                }
            ]
        },
    )

    assert response.status_code == 200
    audio_part = backend.calls[-1][0]["content"][1]
    assert audio_part == {
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(b"fake wav").decode("ascii"),
            "format": "wav",
        },
    }


@pytest.mark.parametrize(
    ("audio", "code"),
    [
        ({"data": base64.b64encode(b"fake").decode("ascii")}, "invalid_audio_format"),
        ({"data": "not base64", "format": "wav"}, "invalid_audio"),
        ({"data": base64.b64encode(b"fake").decode("ascii"), "format": "flac"}, "invalid_audio_format"),
    ],
)
def test_chat_completion_rejects_invalid_input_audio(tmp_path: Path, audio: dict[str, str], code: str) -> None:
    client = make_client_with_backend(tmp_path, EchoBackend(), llm_audio_input_enabled=True)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": audio}]}]},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == code


def test_chat_completion_rejects_native_audio_output_modality(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "say hello"}],
            "modalities": ["text", "audio"],
            "audio": {"voice": "alloy", "format": "wav"},
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "unsupported_audio_output"


def test_video_url_extraction_uses_sampling_settings(tmp_path: Path, monkeypatch) -> None:
    from laas.multimodal import extract_video_frame_data_urls

    positions: list[int] = []

    class FakeBuffer:
        def __init__(self, index: int) -> None:
            self.index = index

        def tobytes(self) -> bytes:
            return f"frame-{self.index}".encode("ascii")

    class FakeCapture:
        def __init__(self, path: str) -> None:
            self.path = path
            self.index = 0
            self.opened = True

        def isOpened(self) -> bool:
            return self.opened

        def get(self, prop: int) -> float:
            if prop == fake_cv2.CAP_PROP_FRAME_COUNT:
                return 100
            if prop == fake_cv2.CAP_PROP_FPS:
                return 10
            return 0

        def set(self, prop: int, value: int) -> bool:
            if prop == fake_cv2.CAP_PROP_POS_FRAMES:
                self.index = int(value)
                positions.append(self.index)
            return True

        def read(self):
            return True, self.index

        def release(self) -> None:
            self.opened = False

    class FakeCv2:
        CAP_PROP_FRAME_COUNT = 7
        CAP_PROP_FPS = 5
        CAP_PROP_POS_FRAMES = 1

        def VideoCapture(self, path: str) -> FakeCapture:
            return FakeCapture(path)

        def imencode(self, extension: str, frame: int):
            return True, FakeBuffer(frame)

    fake_cv2 = FakeCv2()
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    settings = Settings(
        model_dir=tmp_path,
        settings_file=tmp_path / "settings.json",
        video_max_frames=3,
        video_sample_fps=1.0,
        video_max_seconds=4.0,
    )
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video")

    frames = extract_video_frame_data_urls(str(video_path), config=_video_config(settings))

    assert positions == [0, 20, 30]
    assert [base64.b64decode(frame.split(",", 1)[1]) for frame in frames] == [b"frame-0", b"frame-20", b"frame-30"]


def test_responses_store_retrieve_delete_and_previous_response_id(tmp_path: Path) -> None:
    backend = MessageCapturingBackend()
    client = make_client_with_backend(tmp_path, backend)

    first = client.post("/v1/responses", json={"input": "first"}).json()
    assert first["previous_response_id"] is None
    assert first["store"] is True

    retrieved = client.get(f"/v1/responses/{first['id']}").json()
    assert retrieved["id"] == first["id"]
    input_items = client.get(f"/v1/responses/{first['id']}/input_items").json()
    assert input_items["object"] == "list"
    assert input_items["data"][0]["text"] == "first"

    second = client.post(
        "/v1/responses",
        json={"input": "second", "previous_response_id": first["id"]},
    ).json()
    assert second["previous_response_id"] == first["id"]
    assert any(message.get("role") == "assistant" and message.get("content") == "first" for message in backend.calls[-1])

    deleted = client.delete(f"/v1/responses/{first['id']}").json()
    assert deleted == {"id": first["id"], "object": "response.deleted", "deleted": True}
    assert client.get(f"/v1/responses/{first['id']}").status_code == 404


def test_responses_store_false_is_not_retrievable(tmp_path: Path) -> None:
    client = make_client(tmp_path)

    response = client.post("/v1/responses", json={"input": "temporary", "store": False}).json()

    assert response["store"] is False
    assert client.get(f"/v1/responses/{response['id']}").status_code == 404


def test_audio_status_voices_speech_and_unload(tmp_path: Path) -> None:
    client, backend = make_audio_client(tmp_path)

    status = client.get("/v1/local/audio/status").json()
    assert status["configured_model"] == "kokoro-82m"
    assert status["model_downloaded"] is True
    assert status["voices_downloaded"] is True
    assert "pcm" in status["supported_formats"]

    voices = client.get("/v1/local/audio/voices").json()
    assert [voice["id"] for voice in voices["data"]] == ["af", "af_alloy"]

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": "hello from kokoro",
            "voice": "alloy",
            "response_format": "pcm",
            "speed": 1.25,
            "lang": "en-us",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/pcm"
    assert response.headers["x-laas-audio-sample-rate"] == "24000"
    assert response.content == b"\x00\x00\xff?\x01\xc0"
    assert backend.calls[0]["text"] == "hello from kokoro"
    assert backend.calls[0]["voice"] == "af_alloy"
    assert backend.calls[0]["speed"] == 1.25

    unloaded = client.post("/v1/local/audio/unload", json={}).json()
    assert unloaded["is_loaded"] is False
    assert backend.closed is True


def test_audio_missing_assets_require_download(tmp_path: Path) -> None:
    client, _backend = make_audio_client(tmp_path, write_assets=False)

    response = client.post("/v1/local/audio/load", json={"download_if_missing": False})

    assert response.status_code == 409
    assert response.json()["detail"]["error"]["code"] == "audio_not_downloaded"


def test_audio_download_endpoint_fetches_model_and_voices(tmp_path: Path, monkeypatch) -> None:
    client, _backend = make_audio_client(tmp_path, write_assets=False)

    def fake_download(*, repo_id, filename, local_dir):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"downloaded")
        return path

    monkeypatch.setattr("laas.tts.hf_hub_download", fake_download)
    response = client.post("/v1/local/audio/download", json={})

    assert response.status_code == 200
    assert response.json()["downloaded"] is True
    assert len(response.json()["paths"]) == 2


def test_audio_rejects_unknown_model_and_reports_missing_encoder(tmp_path: Path) -> None:
    client, _backend = make_audio_client(tmp_path, ffmpeg_path="definitely-missing-ffmpeg")

    unknown_model = client.post(
        "/v1/audio/speech",
        json={"model": "not-kokoro", "input": "hello", "response_format": "pcm"},
    )
    assert unknown_model.status_code == 404

    missing_encoder = client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": "hello", "response_format": "opus"},
    )
    assert missing_encoder.status_code == 503
    assert missing_encoder.json()["detail"]["error"]["code"] == "audio_encoder_missing"


def test_audio_encode_reports_missing_ffmpeg() -> None:
    try:
        encode_audio([0.0, 0.1], 24000, "aac", ffmpeg_path="definitely-missing-ffmpeg")
    except AudioEncoderMissingError as exc:
        assert exc.response_format == "aac"
        assert exc.encoder == "ffmpeg"
    else:
        raise AssertionError("Expected AudioEncoderMissingError")


def test_openai_voice_aliases_map_to_kokoro_ids() -> None:
    assert resolve_voice("alloy") == "af_alloy"
    assert resolve_voice("af_heart") == "af_heart"


def test_transcription_endpoint_accepts_file_and_returns_verbose_json(tmp_path: Path) -> None:
    client, _audio_backend, transcription_backend = make_voice_client(tmp_path)

    loaded = client.post("/v1/local/transcription/load", json={}).json()
    assert loaded["is_loaded"] is True

    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": "whisper-1",
            "language": "en",
            "prompt": "domain words",
            "response_format": "verbose_json",
            "temperature": "0",
        },
        files={"file": ("sample.wav", b"fake audio bytes", "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task"] == "transcribe"
    assert payload["text"] == "hello from whisper"
    assert payload["segments"][0]["start"] == 0.0
    assert transcription_backend.calls[0]["language"] == "en"
    assert transcription_backend.calls[0]["prompt"] == "domain words"
    assert transcription_backend.calls[0]["translate"] is False


def test_transcription_timestamp_granularities(tmp_path: Path) -> None:
    client, _audio_backend, _transcription_backend = make_voice_client(tmp_path)

    response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        },
        files={"file": ("sample.wav", b"fake audio bytes", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["segments"][0]["start"] == 0.0

    word_response = client.post(
        "/v1/audio/transcriptions",
        data={
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        },
        files={"file": ("sample.wav", b"fake audio bytes", "audio/wav")},
    )
    assert word_response.status_code == 400
    assert word_response.json()["detail"]["error"]["param"] == "response_format"


def test_translation_endpoint_and_text_output(tmp_path: Path) -> None:
    client, _audio_backend, transcription_backend = make_voice_client(tmp_path)

    response = client.post(
        "/v1/audio/translations",
        data={"model": "whisper-1", "response_format": "text"},
        files={"file": ("sample.wav", b"fake audio bytes", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.text == "hello from whisper"
    assert transcription_backend.calls[0]["translate"] is True


def test_transcription_srt_and_vtt_formatting() -> None:
    result = TranscriptionResult(
        text="hello",
        language="en",
        duration=1.25,
        segments=[TranscriptionSegment(id=0, start=0.0, end=1.25, text="hello")],
    )

    assert "00:00:00,000 --> 00:00:01,250" in transcription_to_response(result, "srt", task="transcribe")
    assert "WEBVTT" in transcription_to_response(result, "vtt", task="transcribe")
    assert "00:00:00.000 --> 00:00:01.250" in transcription_to_response(result, "vtt", task="transcribe")


def test_transcription_missing_model_requires_download(tmp_path: Path) -> None:
    client, _audio_backend, _transcription_backend = make_voice_client(tmp_path, write_transcription_model=False)

    response = client.post("/v1/local/transcription/load", json={"download_if_missing": False})

    assert response.status_code == 409
    assert response.json()["detail"]["error"]["code"] == "transcription_not_downloaded"


def test_voice_stack_loads_and_unloads_tts_and_transcription(tmp_path: Path) -> None:
    client, audio_backend, transcription_backend = make_voice_client(tmp_path)

    loaded = client.post("/v1/local/voice/load", json={}).json()
    assert loaded["is_loaded"] is True
    assert loaded["tts"]["is_loaded"] is True
    assert loaded["transcription"]["is_loaded"] is True

    status = client.get("/v1/local/voice/status").json()
    assert status["is_loaded"] is True

    unloaded = client.post("/v1/local/voice/unload", json={}).json()
    assert unloaded["is_loaded"] is False
    assert audio_backend.closed is True
    assert transcription_backend.closed is True


def test_voice_session_turn_transcribes_generates_and_synthesizes(tmp_path: Path) -> None:
    client, audio_backend, transcription_backend = make_voice_client(tmp_path)

    created = client.post(
        "/v1/local/voice/sessions",
        json={"instructions": "Be brief.", "voice": "alloy", "response_format": "pcm"},
    )
    assert created.status_code == 200
    session = created.json()
    assert session["status"] == "active"
    assert session["turn_count"] == 0

    turn_response = client.post(
        f"/v1/local/voice/sessions/{session['id']}/turns",
        files={"file": ("sample.wav", b"fake audio bytes", "audio/wav")},
    )

    assert turn_response.status_code == 200
    turn = turn_response.json()
    assert turn["session_id"] == session["id"]
    assert turn["transcript"]["text"] == "hello from whisper"
    assert turn["response"]["text"] == "hello from whisper"
    assert base64.b64decode(turn["audio"]["data"]) == b"\x00\x00\xff?\x01\xc0"
    assert turn["audio"]["format"] == "pcm"
    assert audio_backend.calls[0]["text"] == "hello from whisper"
    assert audio_backend.calls[0]["voice"] == "af_alloy"
    assert transcription_backend.calls[0]["translate"] is False

    current = client.get(f"/v1/local/voice/sessions/{session['id']}").json()
    assert current["turn_count"] == 1
    assert current["turns"][0]["id"] == turn["id"]

    ended = client.delete(f"/v1/local/voice/sessions/{session['id']}").json()
    assert ended["status"] == "ended"
    assert client.get(f"/v1/local/voice/sessions/{session['id']}").status_code == 404


def test_voice_session_realtime_websocket_buffer_commit(tmp_path: Path) -> None:
    client, audio_backend, transcription_backend = make_voice_client(tmp_path)
    session = client.post(
        "/v1/local/voice/sessions",
        json={"voice": "alloy", "response_format": "pcm"},
    ).json()

    with client.websocket_connect(f"/v1/local/voice/sessions/{session['id']}/realtime") as websocket:
        created = websocket.receive_json()
        assert created["type"] == "session.created"

        websocket.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(b"fake audio bytes").decode("ascii"),
            }
        )
        appended = websocket.receive_json()
        assert appended["type"] == "input_audio_buffer.appended"
        assert appended["buffer_bytes"] == len(b"fake audio bytes")

        websocket.send_json({"type": "input_audio_buffer.commit", "filename": "sample.wav"})
        completed = websocket.receive_json()
        assert completed["type"] == "response.completed"
        assert completed["turn"]["transcript"]["text"] == "hello from whisper"
        assert completed["turn"]["response"]["text"] == "hello from whisper"
        assert base64.b64decode(completed["turn"]["audio"]["data"]) == b"\x00\x00\xff?\x01\xc0"

        websocket.send_json({"type": "response.cancel"})
        assert websocket.receive_json()["type"] == "response.cancelled"

        websocket.send_json({"type": "session.close"})
        closed = websocket.receive_json()
        assert closed["type"] == "session.closed"

    assert audio_backend.calls[0]["voice"] == "af_alloy"
    assert transcription_backend.calls[0]["translate"] is False
    assert client.get(f"/v1/local/voice/sessions/{session['id']}").status_code == 404


def test_voice_session_realtime_session_update_and_response_create(tmp_path: Path) -> None:
    client, audio_backend, transcription_backend = make_voice_client(tmp_path)
    session = client.post(
        "/v1/local/voice/sessions",
        json={"voice": "alloy", "response_format": "pcm"},
    ).json()

    with client.websocket_connect(f"/v1/local/voice/sessions/{session['id']}/realtime") as websocket:
        assert websocket.receive_json()["type"] == "session.created"

        websocket.send_json(
            {
                "type": "session.update",
                "session": {
                    "instructions": "Answer briefly.",
                    "voice": "af",
                    "response_format": "wav",
                    "language": "en",
                },
            }
        )
        updated = websocket.receive_json()
        assert updated["type"] == "session.updated"
        assert updated["session"]["voice"] == "af"
        assert updated["session"]["response_format"] == "wav"

        websocket.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(b"fake audio bytes").decode("ascii"),
            }
        )
        assert websocket.receive_json()["type"] == "input_audio_buffer.appended"

        websocket.send_json({"type": "response.create", "filename": "sample.wav"})
        completed = websocket.receive_json()
        assert completed["type"] == "response.completed"
        assert completed["turn"]["audio"]["format"] == "wav"
        assert completed["turn"]["transcript"]["language"] == "en"

        websocket.send_json({"type": "session.close"})
        assert websocket.receive_json()["type"] == "session.closed"

    assert audio_backend.calls[0]["voice"] == "af"
    assert transcription_backend.calls[0]["language"] == "en"


def test_openai_realtime_session_endpoint_wraps_voice_stack(tmp_path: Path) -> None:
    client, audio_backend, transcription_backend = make_voice_client(tmp_path)
    created = client.post(
        "/v1/realtime/sessions",
        json={
            "voice": "alloy",
            "response_format": "pcm",
            "instructions": "Be concise.",
            "modalities": ["text", "audio"],
            "input_audio_format": "wav",
            "output_audio_format": "pcm",
            "turn_detection": {"type": "server_vad", "threshold": 0.5},
        },
    )
    assert created.status_code == 200
    session = created.json()
    assert session["object"] == "realtime.session"
    assert session["modalities"] == ["text", "audio"]
    assert session["instructions"] == "Be concise."
    assert session["input_audio_format"] == "wav"
    assert session["output_audio_format"] == "pcm"
    assert session["turn_detection"] == {"type": "server_vad", "threshold": 0.5}

    with client.websocket_connect(f"/v1/realtime/sessions/{session['id']}") as websocket:
        created_event = websocket.receive_json()
        assert created_event["type"] == "session.created"
        assert created_event["session"]["object"] == "realtime.session"

        websocket.send_json(
            {
                "type": "session.update",
                "session": {
                    "voice": "af",
                    "response_format": "wav",
                    "language": "en",
                    "modalities": ["audio", "text"],
                    "input_audio_format": "pcm",
                    "turn_detection": None,
                },
            }
        )
        updated = websocket.receive_json()
        assert updated["type"] == "session.updated"
        assert updated["session"]["voice"] == "af"
        assert updated["session"]["output_audio_format"] == "wav"
        assert updated["session"]["input_audio_format"] == "pcm"
        assert updated["session"]["turn_detection"] is None

        websocket.send_json({"type": "conversation.item.clear"})
        unsupported = websocket.receive_json()
        assert unsupported["type"] == "error"
        assert unsupported["error"]["code"] == "unsupported_event"

        websocket.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(b"fake audio bytes").decode("ascii"),
            }
        )
        assert websocket.receive_json()["type"] == "input_audio_buffer.appended"

        websocket.send_json({"type": "response.create", "filename": "sample.wav"})
        response_created = websocket.receive_json()
        assert response_created["type"] == "response.created"
        response_id = response_created["response"]["id"]

        item_added = websocket.receive_json()
        assert item_added["type"] == "response.output_item.added"
        assert item_added["response_id"] == response_id
        item_id = item_added["item"]["id"]

        text_delta = websocket.receive_json()
        assert text_delta["type"] == "response.output_text.delta"
        assert text_delta["response_id"] == response_id
        assert text_delta["item_id"] == item_id
        assert text_delta["delta"] == "hello from whisper"

        text_done = websocket.receive_json()
        assert text_done["type"] == "response.output_text.done"
        assert text_done["response_id"] == response_id
        assert text_done["item_id"] == item_id
        assert text_done["text"] == "hello from whisper"

        audio_chunks = []
        while True:
            audio_event = websocket.receive_json()
            if audio_event["type"] == "response.audio.delta":
                assert audio_event["response_id"] == response_id
                assert audio_event["item_id"] == item_id
                assert audio_event["format"] == "wav"
                audio_chunks.append(base64.b64decode(audio_event["delta"]))
                continue
            assert audio_event["type"] == "response.audio.done"
            assert audio_event["response_id"] == response_id
            assert audio_event["item_id"] == item_id
            break

        item_done = websocket.receive_json()
        assert item_done["type"] == "response.output_item.done"
        assert item_done["response_id"] == response_id
        assert item_done["item"]["id"] == item_id

        completed = websocket.receive_json()
        assert completed["type"] == "response.completed"
        assert completed["response"]["id"] == response_id
        assert completed["response"]["object"] == "realtime.response"
        assert completed["response"]["status"] == "completed"
        content = completed["response"]["output"][0]["content"]
        assert content[0]["type"] == "output_text"
        assert content[1]["type"] == "output_audio"
        assert content[1]["format"] == "wav"
        assert b"".join(audio_chunks) == base64.b64decode(content[1]["audio"])
        assert completed["laas_turn"]["transcript"]["language"] == "en"

        websocket.send_json({"type": "response.cancel"})
        cancelled = websocket.receive_json()
        assert cancelled["type"] == "response.cancelled"
        assert cancelled["response_id"] == response_id
        assert cancelled["status"] == "cancelled"

        websocket.send_json({"type": "session.close"})
        closed = websocket.receive_json()
        assert closed["type"] == "session.closed"
        assert closed["session"]["object"] == "realtime.session"

    assert audio_backend.calls[0]["voice"] == "af"
    assert transcription_backend.calls[0]["language"] == "en"
    assert client.get(f"/v1/local/voice/sessions/{session['id']}").status_code == 404


def test_openai_realtime_conversation_item_create_text_only_response(tmp_path: Path) -> None:
    client, audio_backend, transcription_backend = make_voice_client(tmp_path)
    session = client.post(
        "/v1/realtime/sessions",
        json={"voice": "alloy", "response_format": "pcm", "instructions": "Be concise."},
    ).json()

    with client.websocket_connect(f"/v1/realtime/sessions/{session['id']}") as websocket:
        assert websocket.receive_json()["type"] == "session.created"

        websocket.send_json(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "text only realtime question"}],
                },
            }
        )
        created_item = websocket.receive_json()
        assert created_item["type"] == "conversation.item.created"
        assert created_item["item"]["role"] == "user"
        assert created_item["item"]["content"][0]["text"] == "text only realtime question"

        websocket.send_json({"type": "response.create"})
        response_created = websocket.receive_json()
        assert response_created["type"] == "response.created"
        response_id = response_created["response"]["id"]

        item_added = websocket.receive_json()
        assert item_added["type"] == "response.output_item.added"
        item_id = item_added["item"]["id"]

        text_delta = websocket.receive_json()
        assert text_delta["type"] == "response.output_text.delta"
        assert text_delta["delta"] == "text only realtime question"

        text_done = websocket.receive_json()
        assert text_done["type"] == "response.output_text.done"
        assert text_done["text"] == "text only realtime question"

        audio_chunks = []
        while True:
            event = websocket.receive_json()
            if event["type"] == "response.audio.delta":
                assert event["response_id"] == response_id
                assert event["item_id"] == item_id
                audio_chunks.append(base64.b64decode(event["delta"]))
                continue
            assert event["type"] == "response.audio.done"
            break

        item_done = websocket.receive_json()
        assert item_done["type"] == "response.output_item.done"
        assert item_done["item"]["id"] == item_id

        completed = websocket.receive_json()
        assert completed["type"] == "response.completed"
        assert completed["response"]["id"] == response_id
        assert completed["laas_turn"]["transcript"] is None
        content = completed["response"]["output"][0]["content"]
        assert content[0]["text"] == "text only realtime question"
        assert b"".join(audio_chunks) == base64.b64decode(content[1]["audio"])

        websocket.send_json({"type": "session.close"})
        assert websocket.receive_json()["type"] == "session.closed"

    assert transcription_backend.calls == []
    assert audio_backend.calls[0]["text"] == "text only realtime question"


def test_openai_realtime_conversation_item_retrieve_delete_truncate(tmp_path: Path) -> None:
    client, audio_backend, transcription_backend = make_voice_client(tmp_path)
    session = client.post("/v1/realtime/sessions", json={"voice": "alloy", "response_format": "pcm"}).json()

    with client.websocket_connect(f"/v1/realtime/sessions/{session['id']}") as websocket:
        assert websocket.receive_json()["type"] == "session.created"

        websocket.send_json(
            {
                "type": "conversation.item.create",
                "item": {
                    "id": "item_keep",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "keep this text"}],
                },
            }
        )
        assert websocket.receive_json()["type"] == "conversation.item.created"

        websocket.send_json(
            {
                "type": "conversation.item.create",
                "item": {
                    "id": "item_delete",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "delete this text"}],
                },
            }
        )
        assert websocket.receive_json()["type"] == "conversation.item.created"

        websocket.send_json({"type": "conversation.item.retrieve", "item_id": "item_keep"})
        retrieved = websocket.receive_json()
        assert retrieved["type"] == "conversation.item.retrieved"
        assert retrieved["item"]["id"] == "item_keep"

        websocket.send_json({"type": "conversation.item.truncate", "item_id": "item_keep", "text_end_index": 4})
        truncated = websocket.receive_json()
        assert truncated["type"] == "conversation.item.truncated"
        assert truncated["item"]["content"][0]["text"] == "keep"

        websocket.send_json({"type": "conversation.item.delete", "item_id": "item_delete"})
        deleted = websocket.receive_json()
        assert deleted["type"] == "conversation.item.deleted"
        assert deleted["item_id"] == "item_delete"

        websocket.send_json({"type": "conversation.item.retrieve", "item_id": "item_delete"})
        missing = websocket.receive_json()
        assert missing["type"] == "error"
        assert missing["error"]["code"] == "item_not_found"

        websocket.send_json({"type": "response.create"})
        assert websocket.receive_json()["type"] == "response.created"
        assert websocket.receive_json()["type"] == "response.output_item.added"
        text_delta = websocket.receive_json()
        assert text_delta["type"] == "response.output_text.delta"
        assert text_delta["delta"] == "keep"
        while True:
            event = websocket.receive_json()
            if event["type"] == "response.completed":
                break

        websocket.send_json({"type": "session.close"})
        assert websocket.receive_json()["type"] == "session.closed"

    assert transcription_backend.calls == []
    assert audio_backend.calls[0]["text"] == "keep"


def test_openai_realtime_conversation_item_create_rejects_audio_content(tmp_path: Path) -> None:
    client, _audio_backend, _transcription_backend = make_voice_client(tmp_path)
    session = client.post("/v1/realtime/sessions", json={"voice": "alloy"}).json()

    with client.websocket_connect(f"/v1/realtime/sessions/{session['id']}") as websocket:
        assert websocket.receive_json()["type"] == "session.created"
        websocket.send_json(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_audio", "audio": "abc"}],
                },
            }
        )
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == "invalid_conversation_item"
        assert "input_audio_buffer.append" in error["error"]["message"]


def test_openai_realtime_server_vad_auto_commits_pcm_audio(tmp_path: Path) -> None:
    client, audio_backend, transcription_backend = make_voice_client(tmp_path)
    session = client.post(
        "/v1/realtime/sessions",
        json={
            "voice": "alloy",
            "response_format": "pcm",
            "input_audio_format": "pcm",
            "turn_detection": {"type": "server_vad", "threshold": 0.5, "silence_duration_ms": 60, "frame_ms": 30},
        },
    ).json()
    speech = struct.pack("<" + "h" * 2400, *([12000] * 2400))
    silence = struct.pack("<" + "h" * 2400, *([0] * 2400))

    with client.websocket_connect(f"/v1/realtime/sessions/{session['id']}") as websocket:
        assert websocket.receive_json()["type"] == "session.created"

        websocket.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(speech).decode("ascii")})
        assert websocket.receive_json()["type"] == "input_audio_buffer.appended"
        speech_started = websocket.receive_json()
        assert speech_started["type"] == "input_audio_buffer.speech_started"

        websocket.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(silence).decode("ascii")})
        assert websocket.receive_json()["type"] == "input_audio_buffer.appended"
        speech_stopped = websocket.receive_json()
        assert speech_stopped["type"] == "input_audio_buffer.speech_stopped"
        committed = websocket.receive_json()
        assert committed["type"] == "input_audio_buffer.committed"
        assert committed["buffer_bytes"] == len(speech) + len(silence)
        assert websocket.receive_json()["type"] == "response.created"

        while True:
            event = websocket.receive_json()
            if event["type"] == "response.completed":
                assert event["laas_turn"]["transcript"]["text"] == "hello from whisper"
                break

        websocket.send_json({"type": "session.close"})
        assert websocket.receive_json()["type"] == "session.closed"

    assert transcription_backend.calls[0]["translate"] is False
    assert audio_backend.calls[0]["text"] == "hello from whisper"


def test_openai_realtime_uses_backend_text_stream_deltas(tmp_path: Path) -> None:
    client, audio_backend, transcription_backend = make_voice_client(tmp_path, text_backend=SplitStreamingBackend())
    session = client.post("/v1/realtime/sessions", json={"voice": "alloy", "response_format": "pcm"}).json()

    with client.websocket_connect(f"/v1/realtime/sessions/{session['id']}") as websocket:
        assert websocket.receive_json()["type"] == "session.created"
        websocket.send_json(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "ignored by split backend"}],
                },
            }
        )
        assert websocket.receive_json()["type"] == "conversation.item.created"
        websocket.send_json({"type": "response.create"})
        assert websocket.receive_json()["type"] == "response.created"
        assert websocket.receive_json()["type"] == "response.output_item.added"
        first_delta = websocket.receive_json()
        second_delta = websocket.receive_json()
        assert first_delta["type"] == "response.output_text.delta"
        assert first_delta["delta"] == "split "
        assert second_delta["type"] == "response.output_text.delta"
        assert second_delta["delta"] == "stream"
        done = websocket.receive_json()
        assert done["type"] == "response.output_text.done"
        assert done["text"] == "split stream"

        while True:
            event = websocket.receive_json()
            if event["type"] == "response.completed":
                assert event["laas_turn"]["response"]["text"] == "split stream"
                break

    assert transcription_backend.calls == []
    assert audio_backend.calls[0]["text"] == "split stream"


def test_patch_model_directory_setting(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    target = tmp_path / "models"
    response = client.patch("/v1/local/settings", json={"model_dir": str(target)}).json()
    assert response["model_dir"] == str(target)


def test_blank_image_output_dir_uses_default(tmp_path: Path) -> None:
    settings = Settings(
        model_dir=tmp_path,
        settings_file=tmp_path / "settings.json",
        image_output_dir="",
    )

    assert settings.image_output_dir is None
    assert settings.resolved_image_output_dir == tmp_path / "outputs" / "images"


def test_missing_model_requires_manual_download_or_auto_download(tmp_path: Path, monkeypatch) -> None:
    client = make_client(tmp_path, write_model=False)

    load_response = client.post("/v1/local/models/load", json={"download_if_missing": False})
    assert load_response.status_code == 409
    assert load_response.json()["detail"]["error"]["code"] == "model_not_downloaded"

    chat_response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert chat_response.status_code == 409
    assert chat_response.json()["detail"]["error"]["code"] == "model_not_downloaded"


def test_load_downloads_missing_model_when_allowed(tmp_path: Path, monkeypatch) -> None:
    client = make_client(tmp_path, write_model=False)

    def fake_download(*, repo_id, filename, local_dir):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"downloaded")
        return path

    monkeypatch.setattr("laas.manager.hf_hub_download", fake_download)
    response = client.post("/v1/local/models/load", json={})
    assert response.status_code == 200
    assert response.json()["is_loaded"] is True


def test_inference_auto_downloads_when_enabled(tmp_path: Path, monkeypatch) -> None:
    client = make_client(tmp_path, write_model=False, auto_download=True)

    def fake_download(*, repo_id, filename, local_dir):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"downloaded")
        return path

    monkeypatch.setattr("laas.manager.hf_hub_download", fake_download)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hello"


def test_default_model_dir_is_platform_specific(monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "win32")
    assert default_model_dir() == Path(r"D:\AI\Models")

    monkeypatch.setattr("sys.platform", "linux")
    assert default_model_dir() == Path.home() / "AI" / "Models"


def test_cli_parser_accepts_host_port_and_reload() -> None:
    args = build_parser().parse_args(["--host", "0.0.0.0", "--port", "9000", "--reload"])
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.reload is True


def test_missing_configured_model_paths(tmp_path: Path) -> None:
    settings = Settings(model_dir=tmp_path, settings_file=tmp_path / "settings.json")
    assert missing_configured_model_paths(settings) == [settings.model_path, settings.mmproj_path]

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"model")
    if settings.mmproj_path:
        settings.mmproj_path.write_bytes(b"mmproj")
    assert missing_configured_model_paths(settings) == []


def test_confirm_missing_model_download_decline(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(model_dir=tmp_path, settings_file=tmp_path / "settings.json")
    messages: list[str] = []

    downloaded = confirm_missing_model_downloads(
        settings,
        input_fn=lambda prompt: "n",
        output_fn=messages.append,
        prompt=True,
    )

    assert downloaded == []
    assert any("Skipping model download" in message for message in messages)


def test_confirm_missing_model_download_accept(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(model_dir=tmp_path, settings_file=tmp_path / "settings.json")

    def fake_download(*, repo_id, filename, local_dir):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"downloaded")
        return path

    monkeypatch.setattr("laas.manager.hf_hub_download", fake_download)
    downloaded = confirm_missing_model_downloads(
        settings,
        input_fn=lambda prompt: "yes",
        output_fn=lambda message: None,
        prompt=True,
    )

    assert downloaded == [settings.model_path, settings.mmproj_path]


def test_backend_mmproj_kwargs_are_mapped(tmp_path: Path) -> None:
    class SupportsMmproj:
        def __init__(self, model_path: str, mmproj: str) -> None:
            pass

    kwargs = {"model_path": "model.gguf"}
    _add_mmproj_kwargs(SupportsMmproj, kwargs, tmp_path / "mmproj.gguf")
    assert kwargs["mmproj"] == str(tmp_path / "mmproj.gguf")


def test_backend_mmproj_kwargs_use_gemma4_chat_handler(tmp_path: Path) -> None:
    class SupportsChatHandler:
        def __init__(self, model_path: str, chat_handler: object | None = None) -> None:
            pass

    class FakeGemma4ChatHandler:
        def __init__(self, clip_model_path: str, verbose: bool, use_gpu: bool) -> None:
            self.clip_model_path = clip_model_path
            self.verbose = verbose
            self.use_gpu = use_gpu

    kwargs = {"model_path": "model.gguf"}
    _add_mmproj_kwargs(
        SupportsChatHandler,
        kwargs,
        tmp_path / "mmproj.gguf",
        verbose=True,
        use_gpu=False,
        chat_handler_cls=FakeGemma4ChatHandler,
    )

    handler = kwargs["chat_handler"]
    assert isinstance(handler, FakeGemma4ChatHandler)
    assert handler.clip_model_path == str(tmp_path / "mmproj.gguf")
    assert handler.verbose is True
    assert handler.use_gpu is False


def test_backend_mmproj_kwargs_fail_without_support(tmp_path: Path) -> None:
    class TextOnly:
        def __init__(self, model_path: str) -> None:
            pass

    try:
        _add_mmproj_kwargs(TextOnly, {"model_path": "model.gguf"}, tmp_path / "mmproj.gguf")
    except RuntimeError as exc:
        assert "multimodal projector" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_backend_supported_constructor_kwargs_are_mapped() -> None:
    class SupportsBatch:
        def __init__(
            self,
            model_path: str,
            n_gpu_layers: int,
            n_batch: int,
            n_ubatch: int,
            n_threads_batch: int,
            flash_attn: bool,
            offload_kqv: bool,
        ) -> None:
            pass

    kwargs = {"model_path": "model.gguf"}
    _add_supported_constructor_kwargs(
        SupportsBatch,
        kwargs,
        {
            "n_gpu_layers": None,
            "n_batch": 512,
            "n_ubatch": 256,
            "n_threads_batch": 8,
            "flash_attn": True,
            "offload_kqv": True,
            "swa_full": None,
        },
    )

    assert "n_gpu_layers" not in kwargs
    assert kwargs["n_batch"] == 512
    assert kwargs["n_ubatch"] == 256
    assert kwargs["n_threads_batch"] == 8
    assert kwargs["flash_attn"] is True
    assert kwargs["offload_kqv"] is True
    assert "swa_full" not in kwargs


def test_backend_speculative_kwargs_reject_external_mtp_mode() -> None:
    class SupportsDraft:
        def __init__(self, model_path: str, draft_model: object | None = None) -> None:
            pass

    try:
        _add_speculative_kwargs(
            SupportsDraft,
            {"model_path": "model.gguf"},
            mode="mtp",
            max_ngram_size=2,
            num_pred_tokens=10,
        )
    except RuntimeError as exc:
        assert "External Gemma MTP GGUF" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_backend_response_model_paths_are_normalized() -> None:
    chat = _normalize_chat_response(
        {"model": r"D:\AI\Models\model.gguf", "choices": [{"message": {"content": "ok"}}]},
        "gemma-4-e4b-it-q4_k_m",
        None,
    )
    completion = _normalize_completion_response(
        {"model": r"D:\AI\Models\model.gguf", "choices": [{"text": "ok"}]},
        "gemma-4-e4b-it-q4_k_m",
    )

    assert chat["model"] == "gemma-4-e4b-it-q4_k_m"
    assert completion["model"] == "gemma-4-e4b-it-q4_k_m"
