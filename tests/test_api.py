from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from laas.app import create_app
from laas.backends import EchoBackend
from laas.manager import ModelManager
from laas.settings import Settings, default_model_dir


def make_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        model_dir=tmp_path,
        settings_file=tmp_path / "settings.json",
        idle_unload_seconds=0,
    )

    def backend_factory(model_path: Path, active_settings: Settings) -> EchoBackend:
        return EchoBackend()

    manager = ModelManager(settings, backend_factory=backend_factory)
    (settings.model_path.parent).mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"test-model")
    return TestClient(create_app(settings=settings, manager=manager))


def test_models_and_local_status(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    models = client.get("/v1/models").json()
    assert models["object"] == "list"
    assert models["data"][0]["id"] == "gemma-4-e4b-it-q4_k_m"

    status = client.get("/v1/local/models/status").json()
    assert status["configured_model"] == "gemma-4-e4b-it-q4_k_m"
    assert status["downloaded"] is True
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


def test_patch_model_directory_setting(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    target = tmp_path / "models"
    response = client.patch("/v1/local/settings", json={"model_dir": str(target)}).json()
    assert response["model_dir"] == str(target)


def test_default_model_dir_is_platform_specific(monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "win32")
    assert default_model_dir() == Path(r"D:\AI\Models")

    monkeypatch.setattr("sys.platform", "linux")
    assert default_model_dir() == Path.home() / "AI" / "Models"
