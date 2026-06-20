from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from .errors import openai_error
from .manager import ModelManager, ModelNotDownloadedError
from .openai_compat import build_openai_router
from .schemas import DownloadModelRequest, LoadModelRequest, SettingsPatch
from .settings import Settings, load_settings, save_settings


def create_app(settings: Settings | None = None, manager: ModelManager | None = None) -> FastAPI:
    active_settings = settings or load_settings()
    active_manager = manager or ModelManager(active_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if active_settings.auto_load:
            try:
                active_manager.load(download_if_missing=active_settings.auto_download)
            except ModelNotDownloadedError:
                pass
        yield

    app = FastAPI(
        title="LAAS",
        version="0.1.0",
        description="OpenAI-compatible local API host for Gemma 4 GGUF models.",
        lifespan=lifespan,
    )
    app.state.settings = active_settings
    app.state.manager = active_manager

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model_loaded": active_manager.is_loaded}

    @app.get("/v1/local/settings")
    def get_settings() -> dict[str, Any]:
        return active_settings.public_dict()

    @app.patch("/v1/local/settings")
    def patch_settings(patch: SettingsPatch) -> dict[str, Any]:
        updates = patch.model_dump(exclude_none=True)
        try:
            next_settings = save_settings(active_settings, updates)
        except ValueError as exc:
            raise openai_error(400, str(exc), param="settings") from exc
        active_settings.__dict__.update(next_settings.__dict__)
        return active_settings.public_dict()

    @app.get("/v1/local/models/status")
    def model_status() -> dict[str, Any]:
        return active_manager.status().model_dump()

    @app.get("/v1/local/capabilities")
    def capabilities() -> dict[str, Any]:
        return active_manager.capabilities.model_dump()

    @app.post("/v1/local/models/download")
    def download_model(request: DownloadModelRequest) -> dict[str, Any]:
        path = active_manager.download(hf_repo_id=request.hf_repo_id, filename=request.filename)
        return {"model_id": request.model_id or active_settings.model_id, "path": str(path), "downloaded": True}

    @app.post("/v1/local/models/load")
    def load_model(request: LoadModelRequest) -> dict[str, Any]:
        try:
            return active_manager.load(
                model_id=request.model_id,
                hf_repo_id=request.hf_repo_id,
                filename=request.filename,
                download_if_missing=request.download_if_missing,
            ).model_dump()
        except ModelNotDownloadedError as exc:
            raise openai_error(
                409,
                "The configured model is not downloaded. Call POST /v1/local/models/download first, "
                "or retry /v1/local/models/load with download_if_missing=true.",
                type_="invalid_request_error",
                param="model",
                code="model_not_downloaded",
            ) from exc
        except RuntimeError as exc:
            raise openai_error(503, str(exc), type_="server_error", code="backend_missing") from exc

    @app.post("/v1/local/models/unload")
    def unload_model() -> dict[str, Any]:
        return active_manager.unload().model_dump()

    app.include_router(build_openai_router(active_manager))
    return app


app = create_app()
