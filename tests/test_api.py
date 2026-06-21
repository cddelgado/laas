from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from laas.app import create_app
from laas.backends import EchoBackend, _add_mmproj_kwargs
from laas.image import GeneratedImage, ImageBackend, ImageManager
from laas.main import build_parser, confirm_missing_model_downloads, missing_configured_model_paths
from laas.manager import ModelManager
from laas.openai_compat import _normalize_chat_response, _normalize_completion_response
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


def make_client(tmp_path: Path, *, write_model: bool = True, auto_download: bool = False) -> TestClient:
    settings = Settings(
        model_dir=tmp_path,
        settings_file=tmp_path / "settings.json",
        idle_unload_seconds=0,
        auto_download=auto_download,
    )

    def backend_factory(model_path: Path, active_settings: Settings) -> EchoBackend:
        return EchoBackend()

    manager = ModelManager(settings, backend_factory=backend_factory)
    if write_model:
        (settings.model_path.parent).mkdir(parents=True, exist_ok=True)
        settings.model_path.write_bytes(b"test-model")
        if settings.mmproj_path:
            settings.mmproj_path.write_bytes(b"test-mmproj")
    return TestClient(create_app(settings=settings, manager=manager))


def make_client_with_backend(tmp_path: Path, backend: EchoBackend) -> TestClient:
    settings = Settings(model_dir=tmp_path, settings_file=tmp_path / "settings.json", idle_unload_seconds=0)
    manager = ModelManager(settings, backend_factory=lambda model_path, active_settings: backend)
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"test-model")
    if settings.mmproj_path:
        settings.mmproj_path.write_bytes(b"test-mmproj")
    return TestClient(create_app(settings=settings, manager=manager))


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

    def close(self) -> None:
        self.closed = True


def make_audio_client(
    tmp_path: Path,
    *,
    write_assets: bool = True,
    ffmpeg_path: str = "ffmpeg",
) -> tuple[TestClient, FakeAudioBackend]:
    settings = Settings(
        model_dir=tmp_path,
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
) -> tuple[TestClient, FakeImageBackend]:
    settings = Settings(
        model_dir=tmp_path,
        settings_file=tmp_path / "settings.json",
        idle_unload_seconds=0,
        image_idle_unload_seconds=0,
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


def make_voice_client(
    tmp_path: Path,
    *,
    write_audio_assets: bool = True,
    write_transcription_model: bool = True,
) -> tuple[TestClient, FakeAudioBackend, FakeTranscriptionBackend]:
    settings = Settings(
        model_dir=tmp_path,
        settings_file=tmp_path / "settings.json",
        idle_unload_seconds=0,
        tts_idle_unload_seconds=0,
        stt_idle_unload_seconds=0,
        tts_voices_filename="voices.json",
    )
    text_manager = ModelManager(settings, backend_factory=lambda model_path, active_settings: EchoBackend())
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


def test_models_and_local_status(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    models = client.get("/v1/models").json()
    assert models["object"] == "list"
    assert models["data"][0]["id"] == "gemma-4-e4b-it-q4_k_m"
    assert any(model["id"] == "laas-hash-embedding" for model in models["data"])
    assert any(model["id"] == "sdxl-turbo" for model in models["data"])

    embedding_model = client.get("/v1/models/laas-hash-embedding").json()
    assert embedding_model["id"] == "laas-hash-embedding"
    image_model = client.get("/v1/models/sdxl-turbo").json()
    assert image_model["id"] == "sdxl-turbo"

    status = client.get("/v1/local/models/status").json()
    assert status["configured_model"] == "gemma-4-e4b-it-q4_k_m"
    assert status["downloaded"] is True
    assert status["mmproj_downloaded"] is True
    assert status["is_loaded"] is False


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
        json={"model": "laas-hash-embedding", "input": ["alpha", "beta"], "dimensions": 8},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["model"] == "laas-hash-embedding"
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


def test_image_generation_status_load_generate_and_unload(tmp_path: Path) -> None:
    client, backend = make_image_client(tmp_path)

    status = client.get("/v1/local/images/status").json()
    assert status["configured_model"] == "sdxl-turbo"
    assert status["downloaded"] is True
    assert status["is_loaded"] is False

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


def test_image_generation_rejects_unsupported_options(tmp_path: Path) -> None:
    client, _backend = make_image_client(tmp_path)

    unknown_model = client.post("/v1/images/generations", json={"model": "missing", "prompt": "hello"})
    assert unknown_model.status_code == 404

    multiple = client.post("/v1/images/generations", json={"prompt": "hello", "n": 2})
    assert multiple.status_code == 400
    assert multiple.json()["detail"]["error"]["param"] == "n"

    url_response = client.post(
        "/v1/images/generations",
        json={"prompt": "hello", "response_format": "url"},
    )
    assert url_response.status_code == 400
    assert url_response.json()["detail"]["error"]["param"] == "response_format"


def test_image_generation_missing_model_requires_download(tmp_path: Path) -> None:
    client, _backend = make_image_client(tmp_path, write_model=False)

    load_response = client.post("/v1/local/images/load", json={"download_if_missing": False})
    assert load_response.status_code == 409
    assert load_response.json()["detail"]["error"]["code"] == "image_model_not_downloaded"

    generation_response = client.post("/v1/images/generations", json={"prompt": "hello"})
    assert generation_response.status_code == 409
    assert generation_response.json()["detail"]["error"]["code"] == "image_model_not_downloaded"


def test_image_download_endpoint_fetches_snapshot(tmp_path: Path, monkeypatch) -> None:
    client, _backend = make_image_client(tmp_path, write_model=False)

    def fake_snapshot_download(*, repo_id, local_dir):
        path = Path(local_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model_index.json").write_text("{}", encoding="utf-8")
        return str(path)

    monkeypatch.setattr("laas.image.snapshot_download", fake_snapshot_download)
    response = client.post("/v1/local/images/download", json={})

    assert response.status_code == 200
    assert response.json()["downloaded"] is True
    assert response.json()["model_id"] == "sdxl-turbo"


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
    assert "wav" in status["supported_formats"]

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


def test_patch_model_directory_setting(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    target = tmp_path / "models"
    response = client.patch("/v1/local/settings", json={"model_dir": str(target)}).json()
    assert response["model_dir"] == str(target)


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
