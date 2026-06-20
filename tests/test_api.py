from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from laas.app import create_app
from laas.backends import EchoBackend, _add_mmproj_kwargs
from laas.main import build_parser, confirm_missing_model_downloads, missing_configured_model_paths
from laas.manager import ModelManager
from laas.settings import Settings, default_model_dir


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


def test_models_and_local_status(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    models = client.get("/v1/models").json()
    assert models["object"] == "list"
    assert models["data"][0]["id"] == "gemma-4-e4b-it-q4_k_m"

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
