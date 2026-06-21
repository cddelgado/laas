from __future__ import annotations

import time
import threading
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from laas.app import create_app
from laas.backends import EchoBackend, InferenceBackend
from laas.concurrency import ConcurrencyCoordinator
from laas.image import GeneratedImage, ImageBackend, ImageManager
from laas.manager import ModelManager
from laas.settings import Settings


class BlockingEchoBackend(EchoBackend):
    """Echo backend that blocks for a given time to simulate long-running inference."""

    def __init__(self, delay: float = 0.5, events: list[str] | None = None) -> None:
        super().__init__()
        self.delay = delay
        self.events = events if events is not None else []

    def chat_completion(self, *args, **kwargs):
        self.events.append("chat_start")
        time.sleep(self.delay)
        self.events.append("chat_end")
        return super().chat_completion(*args, **kwargs)


class BlockingImageBackend(ImageBackend):
    """Image backend that blocks for a given time to simulate long-running image gen."""

    def __init__(self, delay: float = 0.5, events: list[str] | None = None) -> None:
        self.delay = delay
        self.events = events if events is not None else []

    def generate(self, *args, **kwargs) -> GeneratedImage:
        self.events.append("image_start")
        time.sleep(self.delay)
        self.events.append("image_end")
        return GeneratedImage(content=b"fake-image-bytes", media_type="image/png")

    def variation(self, *args, **kwargs) -> GeneratedImage:
        return self.generate(*args, **kwargs)

    def close(self) -> None:
        pass


def test_concurrency_serialization_and_swapping(tmp_path: Path) -> None:
    # Setup settings
    settings = Settings(
        model_dir=tmp_path,
        settings_file=tmp_path / "settings.json",
        image_exclusive_load=True,
        mmproj_required=False,
    )
    # Write dummy model paths
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"dummy")
    settings.image_model_path.mkdir(parents=True, exist_ok=True)
    (settings.image_model_path / "model_index.json").write_text("{}", encoding="utf-8")

    # Shared events list
    events: list[str] = []

    # Backends
    llm_backend = BlockingEchoBackend(delay=0.6, events=events)
    img_backend = BlockingImageBackend(delay=0.6, events=events)

    # Managers
    manager = ModelManager(
        settings,
        backend_factory=lambda model_path, active_settings: llm_backend,
    )
    image_manager = ImageManager(
        settings,
        backend_factory=lambda model_path, active_settings: img_backend,
    )

    app = create_app(settings=settings, manager=manager, image_manager=image_manager)
    client = TestClient(app)

    # We will trigger a Chat Completion request and an Image Generation request concurrently.
    # Because they are both heavy models, they should be serialized:
    # chat starts -> chat ends -> image starts -> image ends.

    def run_chat():
        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    def run_image():
        # Give chat a tiny head-start to ensure it acquires the lock first
        time.sleep(0.15)
        client.post(
            "/v1/images/generations",
            json={"prompt": "neon sign", "n": 1, "size": "512x512"},
        )

    t1 = threading.Thread(target=run_chat)
    t2 = threading.Thread(target=run_chat)  # two concurrent chats should also be serialized
    t3 = threading.Thread(target=run_image)

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    # Verify that:
    # 1. Models swapped correctly: only one manager was loaded at the end.
    # 2. Sequential execution occurred.
    # Note: Because they are serialized, one chat runs, then the other, then the image.
    # We shouldn't see overlapping starts/ends where a start occurs before the previous ends.
    assert len(events) == 6
    # Let's check that the first start was followed by an end before the next start:
    assert events[1] in ("chat_end", "image_end")
    assert events[3] in ("chat_end", "image_end")
    assert events[5] in ("chat_end", "image_end")


def test_cuda_cache_cleared_on_swap(tmp_path: Path, monkeypatch) -> None:
    cache_cleared = False

    class FakeTorchCuda:
        def is_available(self) -> bool:
            return True

        def empty_cache(self) -> None:
            nonlocal cache_cleared
            cache_cleared = True

    import sys
    import types
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = FakeTorchCuda()  # type: ignore[attr-defined]
    sys.modules["torch"] = fake_torch

    # Create coordinator and register dummy managers
    coordinator = ConcurrencyCoordinator()

    class DummyManager:
        def __init__(self, name: str) -> None:
            self.name = name
            self.is_loaded = True

        def unload(self) -> None:
            self.is_loaded = False

        def load(self) -> None:
            self.is_loaded = True

    mgr_llm = DummyManager("llm")
    mgr_img = DummyManager("image")

    coordinator.register_manager("llm", mgr_llm)
    coordinator.register_manager("image", mgr_img)

    assert mgr_llm.is_loaded is True
    assert mgr_img.is_loaded is True

    # Swap to LLM
    coordinator.acquire("llm")
    try:
        # It should have unloaded the image manager
        assert mgr_img.is_loaded is False
        assert mgr_llm.is_loaded is True
        assert cache_cleared is True
    finally:
        coordinator.release("llm")


def test_manual_unload_waits_for_active_inference(tmp_path: Path) -> None:
    settings = Settings(
        model_dir=tmp_path,
        settings_file=tmp_path / "settings.json",
        mmproj_required=False,
    )
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.model_path.write_bytes(b"dummy")

    events: list[str] = []
    chat_started = threading.Event()

    class SlowClosableBackend(EchoBackend):
        def chat_completion(self, *args, **kwargs):
            events.append("chat_start")
            chat_started.set()
            time.sleep(0.4)
            events.append("chat_end")
            return super().chat_completion(*args, **kwargs)

        def close(self) -> None:
            events.append("close")
            return super().close()

    backend = SlowClosableBackend()
    manager = ModelManager(settings, backend_factory=lambda model_path, active_settings: backend)
    client = TestClient(create_app(settings=settings, manager=manager))

    def run_chat() -> None:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hold the model"}]},
        )
        assert response.status_code == 200

    def run_unload() -> None:
        assert chat_started.wait(timeout=2)
        response = client.post("/v1/local/models/unload")
        assert response.status_code == 200

    chat_thread = threading.Thread(target=run_chat)
    unload_thread = threading.Thread(target=run_unload)

    chat_thread.start()
    unload_thread.start()
    chat_thread.join(timeout=5)
    unload_thread.join(timeout=5)

    assert not chat_thread.is_alive()
    assert not unload_thread.is_alive()
    assert events == ["chat_start", "chat_end", "close"]
