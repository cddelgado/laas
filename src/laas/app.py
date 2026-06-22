from __future__ import annotations

import base64
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse

from .diagnostics import collect_diagnostics
from .embedding import EmbeddingManager, EmbeddingNotDownloadedError
from .errors import openai_error
from .image import (
    ImageEditManager,
    ImageManager,
    ImageNotDownloadedError,
    ImageParameterError,
    cleanup_image_outputs,
    encode_image_output,
    normalize_image_edit_options,
    normalize_image_generation_options,
    normalize_image_output_format,
    normalize_image_variation_options,
    prepare_inpaint_inputs,
    prepare_variation_input,
    save_image_output,
)
from .manager import ModelManager, ModelNotDownloadedError
from .openai_compat import COMPATIBILITY_MATRIX, _normalize_chat_response, build_openai_router
from .concurrency import ConcurrencyCoordinator
from .schemas import (
    CreateVoiceSessionRequest,
    DownloadAudioRequest,
    DownloadEmbeddingRequest,
    DownloadImageRequest,
    DownloadModelRequest,
    DownloadTranscriptionRequest,
    LoadEmbeddingRequest,
    ImageGenerationRequest,
    LoadAudioRequest,
    LoadImageRequest,
    LoadModelRequest,
    LoadTranscriptionRequest,
    LoadVoiceStackRequest,
    LocalVoiceStackStatus,
    SettingsPatch,
    SpeechRequest,
)
from .settings import Settings, load_settings, save_settings
from .transcription import (
    TranscriptionManager,
    TranscriptionNotDownloadedError,
    transcription_to_response,
)
from .tts import AudioEncoderMissingError, AudioEncodingError, AudioManager, AudioNotDownloadedError, encode_audio


def create_app(
    settings: Settings | None = None,
    manager: ModelManager | None = None,
    audio_manager: AudioManager | None = None,
    transcription_manager: TranscriptionManager | None = None,
    embedding_manager: EmbeddingManager | None = None,
    image_manager: ImageManager | None = None,
    image_edit_manager: ImageEditManager | None = None,
) -> FastAPI:
    active_settings = settings or load_settings()
    active_manager = manager or ModelManager(active_settings)
    active_audio_manager = audio_manager or AudioManager(active_settings)
    active_transcription_manager = transcription_manager or TranscriptionManager(active_settings)
    active_embedding_manager = embedding_manager or EmbeddingManager(active_settings)
    active_image_manager = image_manager or ImageManager(active_settings)
    active_image_edit_manager = image_edit_manager or ImageEditManager(active_settings)

    coordinator = ConcurrencyCoordinator()
    coordinator.register_manager("llm", active_manager)
    coordinator.register_manager("image", active_image_manager)
    coordinator.register_manager("image_edit", active_image_edit_manager)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if active_settings.voice_auto_load:
            try:
                active_audio_manager.load(download_if_missing=active_settings.voice_auto_download)
                active_transcription_manager.load(download_if_missing=active_settings.voice_auto_download)
            except (AudioNotDownloadedError, TranscriptionNotDownloadedError):
                pass
        if active_settings.auto_load:
            try:
                active_manager.load(download_if_missing=active_settings.auto_download)
            except ModelNotDownloadedError:
                pass
        if active_settings.tts_auto_load:
            try:
                active_audio_manager.load(download_if_missing=active_settings.tts_auto_download)
            except AudioNotDownloadedError:
                pass
        if active_settings.stt_auto_load:
            try:
                active_transcription_manager.load(download_if_missing=active_settings.stt_auto_download)
            except TranscriptionNotDownloadedError:
                pass
        if active_settings.embedding_auto_load:
            try:
                active_embedding_manager.load(download_if_missing=active_settings.embedding_auto_download)
            except EmbeddingNotDownloadedError:
                pass
        if active_settings.image_auto_load:
            try:
                active_image_manager.load(download_if_missing=active_settings.image_auto_download)
            except ImageNotDownloadedError:
                pass
        if active_settings.image_edit_auto_load:
            try:
                active_image_edit_manager.load(download_if_missing=active_settings.image_edit_auto_download)
            except ImageNotDownloadedError:
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
    app.state.audio_manager = active_audio_manager
    app.state.transcription_manager = active_transcription_manager
    app.state.embedding_manager = active_embedding_manager
    app.state.image_manager = active_image_manager
    app.state.image_edit_manager = active_image_edit_manager
    app.state.coordinator = coordinator
    app.state.voice_sessions = {}

    def image_response_item(
        *,
        request: Request,
        image,
        response_format: str,
        output_format: str,
        output_compression: int | None,
        revised_prompt: str | None = None,
    ) -> dict[str, Any]:
        encoded = encode_image_output(
            content=image.content,
            output_format=output_format,
            output_compression=output_compression,
        )
        item: dict[str, Any] = {}
        if revised_prompt is not None:
            item["revised_prompt"] = revised_prompt
        if response_format == "b64_json":
            item["b64_json"] = base64.b64encode(encoded.content).decode("ascii")
        else:
            path = save_image_output(
                content=encoded.content,
                output_dir=active_settings.resolved_image_output_dir,
                media_type=encoded.media_type,
            )
            item["url"] = str(request.url_for("get_local_image_file", filename=path.name))
        return item

    def prepare_image_generation_slot() -> None:
        if active_settings.image_exclusive_load and active_image_edit_manager.is_loaded:
            active_image_edit_manager.unload()

    def prepare_image_edit_slot() -> None:
        if active_settings.image_exclusive_load and active_image_manager.is_loaded:
            active_image_manager.unload()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": active_manager.is_loaded,
            "audio_model_loaded": active_audio_manager.is_loaded,
            "transcription_model_loaded": active_transcription_manager.is_loaded,
            "embedding_model_loaded": active_embedding_manager.is_loaded,
            "voice_stack_loaded": active_audio_manager.is_loaded and active_transcription_manager.is_loaded,
            "image_model_loaded": active_image_manager.is_loaded,
            "image_edit_model_loaded": active_image_edit_manager.is_loaded,
        }

    @app.get("/v1/local/settings")
    def get_settings() -> dict[str, Any]:
        return active_settings.public_dict()

    @app.get("/v1/local/diagnostics")
    def diagnostics() -> dict[str, Any]:
        return collect_diagnostics(active_settings)

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

    @app.get("/v1/local/compatibility")
    def compatibility() -> dict[str, Any]:
        return {"object": "local.compatibility_matrix", "data": COMPATIBILITY_MATRIX}

    @app.get("/v1/local/concurrency/status")
    def concurrency_status() -> dict[str, Any]:
        return coordinator.status()

    @app.post("/v1/local/models/download")
    def download_model(request: DownloadModelRequest) -> dict[str, Any]:
        if request.hf_repo_id or request.filename:
            path = active_manager.download(hf_repo_id=request.hf_repo_id, filename=request.filename)
            paths = [path]
            if request.include_mmproj and not request.filename:
                mmproj = active_manager.download_mmproj()
                if mmproj:
                    paths.append(mmproj)
        else:
            paths = active_manager.download_configured_assets(include_mmproj=request.include_mmproj)
        return {
            "model_id": request.model_id or active_settings.model_id,
            "paths": [str(path) for path in paths],
            "path": str(paths[0]),
            "downloaded": True,
        }

    @app.post("/v1/local/models/load")
    def load_model(request: LoadModelRequest) -> dict[str, Any]:
        with coordinator.maintenance("llm"):
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
                    f"The configured {exc.asset} is not downloaded. Call POST /v1/local/models/download first, "
                    "or retry POST /v1/local/models/load with download_if_missing=true.",
                    type_="invalid_request_error",
                    param=exc.asset,
                    code="model_not_downloaded",
                ) from exc
            except RuntimeError as exc:
                raise openai_error(503, str(exc), type_="server_error", code="backend_missing") from exc

    @app.post("/v1/local/models/unload")
    def unload_model() -> dict[str, Any]:
        with coordinator.maintenance():
            return active_manager.unload().model_dump()

    def _unload_all_image_models_internal() -> dict[str, Any]:
        generation = active_image_manager.unload().model_dump()
        edit = active_image_edit_manager.unload().model_dump()
        return {
            "generation": generation,
            "edit": edit,
            "is_loaded": generation["is_loaded"] or edit["is_loaded"],
        }

    @app.post("/v1/local/unload/all")
    def unload_all_local_models() -> dict[str, Any]:
        with coordinator.maintenance():
            text = active_manager.unload().model_dump()
            audio = active_audio_manager.unload().model_dump()
            transcription = active_transcription_manager.unload().model_dump()
            embeddings = active_embedding_manager.unload().model_dump()
            images = _unload_all_image_models_internal()
            coordinator.clear_accelerator_cache()
            return {
                "text": text,
                "audio": audio,
                "transcription": transcription,
                "embeddings": embeddings,
                "images": images,
                "is_loaded": (
                    text["is_loaded"]
                    or audio["is_loaded"]
                    or transcription["is_loaded"]
                    or embeddings["is_loaded"]
                    or images["is_loaded"]
                ),
            }

    @app.get("/v1/local/embeddings/status")
    def embedding_status() -> dict[str, Any]:
        return active_embedding_manager.status().model_dump()

    @app.post("/v1/local/embeddings/download")
    def download_embedding_model(request: DownloadEmbeddingRequest) -> dict[str, Any]:
        if request.hf_repo_id:
            active_settings.embedding_hf_repo_id = request.hf_repo_id
        path = active_embedding_manager.download()
        return {
            "model_id": request.model_id or active_settings.embedding_model_id,
            "path": str(path),
            "downloaded": True,
        }

    @app.post("/v1/local/embeddings/load")
    def load_embedding_model(request: LoadEmbeddingRequest) -> dict[str, Any]:
        try:
            return active_embedding_manager.load(
                model_id=request.model_id,
                hf_repo_id=request.hf_repo_id,
                download_if_missing=request.download_if_missing,
            ).model_dump()
        except EmbeddingNotDownloadedError as exc:
            raise openai_error(
                409,
                "The configured embedding model is not downloaded. Call POST /v1/local/embeddings/download first, "
                "or retry POST /v1/local/embeddings/load with download_if_missing=true.",
                type_="invalid_request_error",
                param=exc.asset,
                code="embedding_model_not_downloaded",
            ) from exc
        except RuntimeError as exc:
            raise openai_error(503, str(exc), type_="server_error", code="embedding_backend_missing") from exc

    @app.post("/v1/local/embeddings/unload")
    def unload_embedding_model() -> dict[str, Any]:
        return active_embedding_manager.unload().model_dump()

    @app.get("/v1/local/images/status")
    def image_status() -> dict[str, Any]:
        return active_image_manager.status().model_dump()

    @app.get("/v1/local/images/status/all")
    def image_status_all() -> dict[str, Any]:
        return {
            "generation": active_image_manager.status().model_dump(),
            "edit": active_image_edit_manager.status().model_dump(),
        }

    @app.post("/v1/local/images/download")
    def download_image_model(request: DownloadImageRequest) -> dict[str, Any]:
        if request.hf_repo_id:
            active_settings.image_hf_repo_id = request.hf_repo_id
        path = active_image_manager.download()
        return {
            "model_id": request.model_id or active_settings.image_model_id,
            "path": str(path),
            "downloaded": True,
        }

    @app.post("/v1/local/images/load")
    def load_image_model(request: LoadImageRequest) -> dict[str, Any]:
        with coordinator.maintenance("image"):
            try:
                prepare_image_generation_slot()
                return active_image_manager.load(
                    model_id=request.model_id,
                    hf_repo_id=request.hf_repo_id,
                    download_if_missing=request.download_if_missing,
                ).model_dump()
            except ImageNotDownloadedError as exc:
                raise openai_error(
                    409,
                    "The configured image model is not downloaded. Call POST /v1/local/images/download first, "
                    "or retry POST /v1/local/images/load with download_if_missing=true.",
                    type_="invalid_request_error",
                    param=exc.asset,
                    code="image_model_not_downloaded",
                ) from exc
            except RuntimeError as exc:
                raise openai_error(503, str(exc), type_="server_error", code="image_backend_missing") from exc

    @app.post("/v1/local/images/unload")
    def unload_image_model() -> dict[str, Any]:
        with coordinator.maintenance():
            return active_image_manager.unload().model_dump()

    @app.post("/v1/local/images/unload/all")
    def unload_all_image_models() -> dict[str, Any]:
        with coordinator.maintenance():
            res = _unload_all_image_models_internal()
            coordinator.clear_accelerator_cache()
            return res

    @app.get("/v1/local/images/edit/status")
    def image_edit_status() -> dict[str, Any]:
        return active_image_edit_manager.status().model_dump()

    @app.post("/v1/local/images/edit/download")
    def download_image_edit_model(request: DownloadImageRequest) -> dict[str, Any]:
        if request.hf_repo_id:
            active_settings.image_edit_hf_repo_id = request.hf_repo_id
        path = active_image_edit_manager.download()
        return {
            "model_id": request.model_id or active_settings.image_edit_model_id,
            "path": str(path),
            "downloaded": True,
        }

    @app.post("/v1/local/images/edit/load")
    def load_image_edit_model(request: LoadImageRequest) -> dict[str, Any]:
        with coordinator.maintenance("image_edit"):
            try:
                prepare_image_edit_slot()
                return active_image_edit_manager.load(
                    model_id=request.model_id,
                    hf_repo_id=request.hf_repo_id,
                    download_if_missing=request.download_if_missing,
                ).model_dump()
            except ImageNotDownloadedError as exc:
                raise openai_error(
                    409,
                    "The configured image edit model is not downloaded. Call POST /v1/local/images/edit/download first, "
                    "or retry POST /v1/local/images/edit/load with download_if_missing=true.",
                    type_="invalid_request_error",
                    param=exc.asset,
                    code="image_edit_model_not_downloaded",
                ) from exc
            except RuntimeError as exc:
                raise openai_error(503, str(exc), type_="server_error", code="image_edit_backend_missing") from exc

    @app.post("/v1/local/images/edit/unload")
    def unload_image_edit_model() -> dict[str, Any]:
        with coordinator.maintenance():
            return active_image_edit_manager.unload().model_dump()

    @app.get("/v1/local/files/images/{filename}")
    def get_local_image_file(filename: str) -> FileResponse:
        safe_name = Path(filename).name
        suffix = Path(safe_name).suffix.lower()
        media_type = {
            ".png": "image/png",
            ".jpeg": "image/jpeg",
            ".jpg": "image/jpeg",
            ".webp": "image/webp",
        }.get(suffix)
        if safe_name != filename or media_type is None:
            raise openai_error(404, "Image output not found", code="not_found")
        path = active_settings.resolved_image_output_dir / safe_name
        if not path.exists() or not path.is_file():
            raise openai_error(404, "Image output not found", code="not_found")
        return FileResponse(path, media_type=media_type, filename=safe_name)

    @app.get("/v1/local/audio/status")
    def audio_status() -> dict[str, Any]:
        return active_audio_manager.status().model_dump()

    @app.get("/v1/local/audio/voices")
    def audio_voices() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": voice, "object": "voice"} for voice in active_audio_manager.voices()],
        }

    @app.post("/v1/local/audio/download")
    def download_audio(request: DownloadAudioRequest) -> dict[str, Any]:
        if request.hf_repo_id:
            active_settings.tts_hf_repo_id = request.hf_repo_id
        if request.model_filename:
            active_settings.tts_model_filename = request.model_filename
        if request.voices_filename:
            active_settings.tts_voices_filename = request.voices_filename
        paths = active_audio_manager.download_configured_assets()
        return {
            "model_id": request.model_id or active_settings.tts_model_id,
            "paths": [str(path) for path in paths],
            "downloaded": True,
        }

    @app.post("/v1/local/audio/load")
    def load_audio(request: LoadAudioRequest) -> dict[str, Any]:
        try:
            return active_audio_manager.load(
                model_id=request.model_id,
                hf_repo_id=request.hf_repo_id,
                model_filename=request.model_filename,
                voices_filename=request.voices_filename,
                download_if_missing=request.download_if_missing,
            ).model_dump()
        except AudioNotDownloadedError as exc:
            raise openai_error(
                409,
                f"The configured {exc.asset} is not downloaded. Call POST /v1/local/audio/download first, "
                "or retry POST /v1/local/audio/load with download_if_missing=true.",
                type_="invalid_request_error",
                param=exc.asset,
                code="audio_not_downloaded",
            ) from exc
        except (RuntimeError, ValueError) as exc:
            raise openai_error(503, str(exc), type_="server_error", code="audio_backend_missing") from exc

    @app.post("/v1/local/audio/unload")
    def unload_audio() -> dict[str, Any]:
        return active_audio_manager.unload().model_dump()

    @app.get("/v1/local/transcription/status")
    def transcription_status() -> dict[str, Any]:
        return active_transcription_manager.status().model_dump()

    @app.post("/v1/local/transcription/download")
    def download_transcription(request: DownloadTranscriptionRequest) -> dict[str, Any]:
        if request.hf_repo_id:
            active_settings.stt_hf_repo_id = request.hf_repo_id
        if request.filename:
            active_settings.stt_model_filename = request.filename
        path = active_transcription_manager.download()
        return {
            "model_id": request.model_id or active_settings.stt_model_id,
            "path": str(path),
            "downloaded": True,
        }

    @app.post("/v1/local/transcription/load")
    def load_transcription(request: LoadTranscriptionRequest) -> dict[str, Any]:
        try:
            return active_transcription_manager.load(
                model_id=request.model_id,
                hf_repo_id=request.hf_repo_id,
                filename=request.filename,
                download_if_missing=request.download_if_missing,
            ).model_dump()
        except TranscriptionNotDownloadedError as exc:
            raise openai_error(
                409,
                f"The configured {exc.asset} is not downloaded. Call POST /v1/local/transcription/download first, "
                "or retry POST /v1/local/transcription/load with download_if_missing=true.",
                type_="invalid_request_error",
                param=exc.asset,
                code="transcription_not_downloaded",
            ) from exc
        except RuntimeError as exc:
            raise openai_error(503, str(exc), type_="server_error", code="transcription_backend_missing") from exc

    @app.post("/v1/local/transcription/unload")
    def unload_transcription() -> dict[str, Any]:
        return active_transcription_manager.unload().model_dump()

    @app.get("/v1/local/voice/status")
    def voice_status() -> dict[str, Any]:
        return LocalVoiceStackStatus(
            tts=active_audio_manager.status(),
            transcription=active_transcription_manager.status(),
            is_loaded=active_audio_manager.is_loaded and active_transcription_manager.is_loaded,
        ).model_dump()

    @app.post("/v1/local/voice/download")
    def download_voice_stack() -> dict[str, Any]:
        audio_paths = active_audio_manager.download_configured_assets()
        transcription_path = active_transcription_manager.download()
        return {
            "downloaded": True,
            "paths": [str(path) for path in [*audio_paths, transcription_path]],
        }

    @app.post("/v1/local/voice/load")
    def load_voice_stack(request: LoadVoiceStackRequest) -> dict[str, Any]:
        try:
            audio_status = active_audio_manager.load(download_if_missing=request.download_if_missing)
            transcription_status = active_transcription_manager.load(download_if_missing=request.download_if_missing)
            return LocalVoiceStackStatus(
                tts=audio_status,
                transcription=transcription_status,
                is_loaded=active_audio_manager.is_loaded and active_transcription_manager.is_loaded,
            ).model_dump()
        except AudioNotDownloadedError as exc:
            raise openai_error(
                409,
                f"The configured {exc.asset} is not downloaded. Call POST /v1/local/voice/download first.",
                type_="invalid_request_error",
                param=exc.asset,
                code="audio_not_downloaded",
            ) from exc
        except TranscriptionNotDownloadedError as exc:
            raise openai_error(
                409,
                f"The configured {exc.asset} is not downloaded. Call POST /v1/local/voice/download first.",
                type_="invalid_request_error",
                param=exc.asset,
                code="transcription_not_downloaded",
            ) from exc
        except RuntimeError as exc:
            raise openai_error(503, str(exc), type_="server_error", code="voice_backend_missing") from exc

    @app.post("/v1/local/voice/unload")
    def unload_voice_stack() -> dict[str, Any]:
        audio_status = active_audio_manager.unload()
        transcription_status = active_transcription_manager.unload()
        app.state.voice_sessions.clear()
        return LocalVoiceStackStatus(
            tts=audio_status,
            transcription=transcription_status,
            is_loaded=False,
        ).model_dump()

    def _create_voice_session_record(request: CreateVoiceSessionRequest) -> dict[str, Any]:
        if request.model and request.model != active_settings.model_id:
            raise openai_error(
                404,
                f"The model '{request.model}' is not loaded. Loaded/configured model is '{active_settings.model_id}'.",
                param="model",
                code="model_not_found",
            )
        try:
            active_manager.load(download_if_missing=request.download_if_missing)
            active_audio_manager.load(download_if_missing=request.download_if_missing)
            active_transcription_manager.load(download_if_missing=request.download_if_missing)
        except ModelNotDownloadedError as exc:
            raise openai_error(409, f"The configured {exc.asset} is not downloaded.", param=exc.asset, code="model_not_downloaded") from exc
        except AudioNotDownloadedError as exc:
            raise openai_error(409, f"The configured {exc.asset} is not downloaded.", param=exc.asset, code="audio_not_downloaded") from exc
        except TranscriptionNotDownloadedError as exc:
            raise openai_error(
                409,
                f"The configured {exc.asset} is not downloaded.",
                param=exc.asset,
                code="transcription_not_downloaded",
            ) from exc
        except RuntimeError as exc:
            raise openai_error(503, str(exc), type_="server_error", code="voice_backend_missing") from exc

        session_id = f"vs_{uuid.uuid4().hex}"
        now = int(time.time())
        messages = []
        if request.instructions:
            messages.append({"role": "system", "content": request.instructions})
        extra = request.model_extra or {}
        modalities = _normalize_realtime_modalities(extra.get("modalities"))
        input_audio_format = _normalize_realtime_audio_format(extra.get("input_audio_format"), default="wav")
        output_audio_format = _normalize_realtime_audio_format(
            extra.get("output_audio_format"),
            default=request.response_format,
        )
        session = {
            "id": session_id,
            "object": "local.voice.session",
            "created_at": now,
            "updated_at": now,
            "status": "active",
            "model": active_settings.model_id,
            "voice": request.voice,
            "response_format": output_audio_format,
            "modalities": modalities,
            "input_audio_format": input_audio_format,
            "output_audio_format": output_audio_format,
            "turn_detection": extra.get("turn_detection"),
            "language": request.language,
            "prompt": request.prompt,
            "temperature": request.temperature,
            "speed": request.speed,
            "lang": request.lang,
            "messages": messages,
            "conversation_items": [],
            "active_response_id": None,
            "last_response_id": None,
            "cancelled_response_ids": [],
            "turns": [],
        }
        app.state.voice_sessions[session_id] = session
        return session

    @app.post("/v1/local/voice/sessions")
    def create_voice_session(request: CreateVoiceSessionRequest) -> dict[str, Any]:
        session = _create_voice_session_record(request)
        return _public_voice_session(session)

    @app.post("/v1/realtime/sessions")
    def create_realtime_session(request: CreateVoiceSessionRequest) -> dict[str, Any]:
        session = _create_voice_session_record(request)
        return _public_openai_realtime_session(session)

    @app.get("/v1/local/voice/sessions/{session_id}")
    def get_voice_session(session_id: str) -> dict[str, Any]:
        session = _get_voice_session(session_id, app.state.voice_sessions)
        return {**_public_voice_session(session), "turns": session["turns"]}

    @app.delete("/v1/local/voice/sessions/{session_id}")
    def delete_voice_session(session_id: str) -> dict[str, Any]:
        session = _get_voice_session(session_id, app.state.voice_sessions)
        session["status"] = "ended"
        session["updated_at"] = int(time.time())
        app.state.voice_sessions.pop(session_id, None)
        return _public_voice_session(session)

    def _normalize_realtime_modalities(value: Any) -> list[str]:
        if value is None:
            return ["audio", "text"]
        if not isinstance(value, list) or not value:
            raise openai_error(400, "modalities must be a non-empty array", param="modalities")
        normalized = []
        for modality in value:
            if modality not in {"audio", "text"}:
                raise openai_error(400, f"unsupported realtime modality: {modality}", param="modalities")
            if modality not in normalized:
                normalized.append(modality)
        return normalized

    def _normalize_realtime_audio_format(value: Any, *, default: str) -> str:
        if value is None:
            return default
        if value not in {"pcm", "wav", "mp3", "flac", "opus", "aac"}:
            raise openai_error(400, f"unsupported realtime audio format: {value}", param="audio_format")
        return str(value)

    def _backend_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in message.items() if key != "item_id"}
            for message in messages
        ]

    def _run_voice_response(
        session: dict[str, Any],
        *,
        user_message: dict[str, Any] | None = None,
        transcript_payload: dict[str, Any] | None = None,
        response_format: str | None = None,
        voice: str | None = None,
        speed: float | None = None,
    ) -> dict[str, Any]:
        messages = [*session["messages"]]
        if user_message is not None:
            messages.append(user_message)
        chat_result = active_manager.backend.chat_completion(
            messages=_backend_messages(messages),
            model=active_settings.model_id,
            tools=None,
            tool_choice="none",
            temperature=1.0,
            top_p=1.0,
            max_tokens=None,
            stream=False,
            extra_params={},
        )
        chat_response = _normalize_chat_response(chat_result, active_settings.model_id, None)
        assistant_text = chat_response["choices"][0]["message"].get("content") or ""
        assistant_message = {"role": "assistant", "content": assistant_text}
        speech = active_audio_manager.synthesize(
            text=assistant_text,
            voice=voice if voice is not None else session.get("voice"),
            speed=speed if speed is not None else session.get("speed"),
            lang=session.get("lang"),
            is_phonemes=False,
            trim=True,
        )
        requested_format = response_format or session.get("response_format") or "pcm"
        audio_content, media_type = encode_audio(
            speech.samples,
            speech.sample_rate,
            requested_format,
            ffmpeg_path=active_settings.tts_ffmpeg_path,
        )

        next_messages = [*session["messages"]]
        if user_message is not None:
            next_messages.append(user_message)
        next_messages.append(assistant_message)
        session["messages"] = next_messages
        session["updated_at"] = int(time.time())
        turn = {
            "id": f"vturn_{uuid.uuid4().hex}",
            "object": "local.voice.turn",
            "session_id": session["id"],
            "created_at": session["updated_at"],
            "transcript": transcript_payload,
            "response": {"text": assistant_text},
            "audio": {
                "data": base64.b64encode(audio_content).decode("ascii"),
                "format": requested_format,
                "media_type": media_type,
                "sample_rate": speech.sample_rate,
            },
        }
        session["turns"].append(turn)
        return turn

    def _run_voice_turn(
        session: dict[str, Any],
        media_path: Path,
        *,
        response_format: str | None = None,
        voice: str | None = None,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        speed: float | None = None,
    ) -> dict[str, Any]:
        transcript = active_transcription_manager.transcribe(
            media_path=media_path,
            language=language if language is not None else session.get("language"),
            prompt=prompt if prompt is not None else session.get("prompt"),
            temperature=temperature if temperature is not None else session.get("temperature"),
            translate=False,
        )
        user_message = {"role": "user", "content": transcript.text}
        return _run_voice_response(
            session,
            user_message=user_message,
            transcript_payload={
                "text": transcript.text,
                "language": transcript.language,
                "duration": transcript.duration,
                "segments": [
                    {
                        "id": segment.id,
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                    }
                    for segment in transcript.segments
                ],
            },
            response_format=response_format,
            voice=voice,
            speed=speed,
        )

    @app.post("/v1/local/voice/sessions/{session_id}/turns")
    async def create_voice_turn(
        session_id: str,
        file: UploadFile = File(...),
        response_format: str | None = Form(None),
        voice: str | None = Form(None),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
        temperature: float | None = Form(None),
        speed: float | None = Form(None),
    ) -> dict[str, Any]:
        session = _get_voice_session(session_id, app.state.voice_sessions)
        media_path = await _upload_to_temp_file(file)
        try:
            return _run_voice_turn(
                session,
                media_path,
                response_format=response_format,
                voice=voice,
                language=language,
                prompt=prompt,
                temperature=temperature,
                speed=speed,
            )
        except TranscriptionNotDownloadedError as exc:
            raise openai_error(409, f"The configured {exc.asset} is not downloaded.", param=exc.asset, code="transcription_not_downloaded") from exc
        except AudioNotDownloadedError as exc:
            raise openai_error(409, f"The configured {exc.asset} is not downloaded.", param=exc.asset, code="audio_not_downloaded") from exc
        except ModelNotDownloadedError as exc:
            raise openai_error(409, f"The configured {exc.asset} is not downloaded.", param=exc.asset, code="model_not_downloaded") from exc
        except ValueError as exc:
            raise openai_error(400, str(exc), type_="invalid_request_error") from exc
        except AudioEncoderMissingError as exc:
            raise openai_error(503, str(exc), type_="server_error", param="response_format", code="audio_encoder_missing") from exc
        except AudioEncodingError as exc:
            raise openai_error(500, str(exc), type_="server_error", param="response_format", code="audio_encoding_failed") from exc
        except RuntimeError as exc:
            raise openai_error(503, str(exc), type_="server_error", code="voice_backend_missing") from exc
        finally:
            media_path.unlink(missing_ok=True)

    def _update_voice_session_from_realtime_event(session: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
        session_patch = event.get("session") if isinstance(event.get("session"), dict) else event
        allowed = [
            "instructions",
            "voice",
            "response_format",
            "output_audio_format",
            "input_audio_format",
            "modalities",
            "turn_detection",
            "language",
            "prompt",
            "temperature",
            "speed",
            "lang",
        ]
        if "instructions" in session_patch:
            instructions = session_patch.get("instructions")
            session["messages"] = [{"role": "system", "content": instructions}] if instructions else []
            session["conversation_items"] = []
        for key in [field for field in allowed if field != "instructions"]:
            if key in session_patch:
                if key == "modalities":
                    session[key] = _normalize_realtime_modalities(session_patch[key])
                elif key in {"input_audio_format", "output_audio_format"}:
                    session[key] = _normalize_realtime_audio_format(session_patch[key], default="wav")
                    if key == "output_audio_format":
                        session["response_format"] = session[key]
                elif key == "response_format":
                    session[key] = _normalize_realtime_audio_format(session_patch[key], default="pcm")
                    session["output_audio_format"] = session[key]
                else:
                    session[key] = session_patch[key]
        session["updated_at"] = int(time.time())
        return _public_voice_session(session)

    def _public_openai_realtime_session(session: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": session["id"],
            "object": "realtime.session",
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "status": session["status"],
            "model": session["model"],
            "modalities": session.get("modalities", ["audio", "text"]),
            "instructions": session["messages"][0]["content"] if session["messages"] else None,
            "voice": session["voice"],
            "input_audio_format": session.get("input_audio_format", "wav"),
            "output_audio_format": session.get("output_audio_format", session["response_format"]),
            "temperature": session["temperature"],
            "turn_detection": session.get("turn_detection"),
            "tools": [],
        }

    def _public_session_for_realtime(session: dict[str, Any], *, openai_shape: bool) -> dict[str, Any]:
        return _public_openai_realtime_session(session) if openai_shape else _public_voice_session(session)

    def _realtime_text_from_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            raise ValueError("conversation item content must be a string or an array")
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                raise ValueError("conversation item content parts must be objects")
            part_type = part.get("type")
            if part_type in {"input_text", "text", "output_text", "input_text_delta", "response.output_text"}:
                parts.append(str(part.get("text") or part.get("delta") or ""))
                continue
            if part_type in {None, "message"} and "content" in part:
                parts.append(_realtime_text_from_content(part.get("content")))
                continue
            if part_type in {"input_audio", "audio"}:
                raise ValueError("conversation.item.create audio content is not supported; use input_audio_buffer.append")
            raise ValueError(f"unsupported conversation item content type: {part_type}")
        return "".join(parts)

    def _realtime_item_id_from_event(event: dict[str, Any]) -> str:
        item_id = event.get("item_id")
        if not item_id and isinstance(event.get("item"), dict):
            item_id = event["item"].get("id")
        if not item_id:
            raise ValueError("event requires item_id")
        return str(item_id)

    def _find_realtime_conversation_item(session: dict[str, Any], item_id: str) -> dict[str, Any] | None:
        return next((item for item in session["conversation_items"] if item.get("id") == item_id), None)

    def _remove_message_for_realtime_item(session: dict[str, Any], item_id: str) -> None:
        session["messages"] = [message for message in session["messages"] if message.get("item_id") != item_id]

    def _replace_message_for_realtime_item(session: dict[str, Any], item_id: str, text: str) -> None:
        for message in session["messages"]:
            if message.get("item_id") == item_id:
                message["content"] = text
                return

    def _append_realtime_conversation_item(session: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
        raw_item = event.get("item")
        if not isinstance(raw_item, dict):
            raise ValueError("conversation.item.create requires an item object")
        item_type = raw_item.get("type", "message")
        if item_type != "message":
            raise ValueError(f"unsupported conversation item type: {item_type}")
        role = raw_item.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError("conversation item message role must be system, user, or assistant")
        text = _realtime_text_from_content(raw_item.get("content", ""))
        if not text:
            raise ValueError("conversation item message content cannot be empty")
        item_id = str(raw_item.get("id") or f"item_{uuid.uuid4().hex}")
        item = {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "role": role,
            "content": [{"type": "text", "text": text}],
        }
        session["conversation_items"].append(item)
        session["messages"].append({"role": role, "content": text, "item_id": item_id})
        session["updated_at"] = int(time.time())
        return item

    def _retrieve_realtime_conversation_item(session: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
        item_id = _realtime_item_id_from_event(event)
        item = _find_realtime_conversation_item(session, item_id)
        if item is None:
            raise LookupError(f"conversation item '{item_id}' does not exist")
        return item

    def _delete_realtime_conversation_item(session: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
        item_id = _realtime_item_id_from_event(event)
        item = _find_realtime_conversation_item(session, item_id)
        if item is None:
            raise LookupError(f"conversation item '{item_id}' does not exist")
        session["conversation_items"] = [existing for existing in session["conversation_items"] if existing.get("id") != item_id]
        _remove_message_for_realtime_item(session, item_id)
        session["updated_at"] = int(time.time())
        return item

    def _truncate_realtime_conversation_item(session: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
        item_id = _realtime_item_id_from_event(event)
        item = _find_realtime_conversation_item(session, item_id)
        if item is None:
            raise LookupError(f"conversation item '{item_id}' does not exist")
        content = item.get("content") or []
        content_index = int(event.get("content_index", 0) or 0)
        if content_index < 0 or content_index >= len(content):
            raise ValueError("content_index is out of range")
        part = content[content_index]
        if not isinstance(part, dict):
            raise ValueError("conversation item content part is invalid")
        if part.get("type") in {"text", "input_text", "output_text"}:
            if "text" in event:
                new_text = str(event.get("text") or "")
            elif "text_end_index" in event:
                new_text = str(part.get("text") or "")[: max(0, int(event.get("text_end_index") or 0))]
            elif "audio_end_ms" in event:
                new_text = str(part.get("text") or "")
            else:
                new_text = ""
            part["text"] = new_text
            _replace_message_for_realtime_item(session, item_id, new_text)
        session["updated_at"] = int(time.time())
        return item

    def _can_create_text_only_realtime_response(session: dict[str, Any]) -> bool:
        return any(message.get("role") == "user" for message in session.get("messages", []))

    def _completed_realtime_event(
        *,
        session_id: str,
        turn_payload: dict[str, Any],
        openai_shape: bool,
        response_id: str | None = None,
        item_id: str | None = None,
    ) -> dict[str, Any]:
        if not openai_shape:
            return {"type": "response.completed", "session_id": session_id, "turn": turn_payload}
        return {
            "type": "response.completed",
            "response": _openai_realtime_response(
                turn_payload,
                response_id=response_id or f"resp_{uuid.uuid4().hex}",
                item_id=item_id or f"item_{uuid.uuid4().hex}",
                status="completed",
            ),
            "laas_turn": turn_payload,
        }

    def _openai_realtime_response(
        turn_payload: dict[str, Any],
        *,
        response_id: str,
        item_id: str,
        status: str,
    ) -> dict[str, Any]:
        return {
            "id": response_id,
            "object": "realtime.response",
            "status": status,
            "output": [
                {
                    "id": item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "role": "assistant",
                    "status": status,
                    "content": [
                        {"type": "output_text", "text": turn_payload["response"]["text"]},
                        {
                            "type": "output_audio",
                            "audio": turn_payload["audio"]["data"],
                            "format": turn_payload["audio"]["format"],
                            "media_type": turn_payload["audio"]["media_type"],
                            "sample_rate": turn_payload["audio"]["sample_rate"],
                        },
                    ],
                }
            ],
        }

    def _openai_realtime_audio_delta_chunks(turn_payload: dict[str, Any], chunk_size: int = 16 * 1024) -> list[str]:
        audio_bytes = base64.b64decode(turn_payload["audio"]["data"])
        if not audio_bytes:
            return []
        return [
            base64.b64encode(audio_bytes[index : index + chunk_size]).decode("ascii")
            for index in range(0, len(audio_bytes), chunk_size)
        ]

    async def _send_openai_realtime_turn_events(
        websocket: WebSocket,
        turn_payload: dict[str, Any],
        *,
        response_id: str,
    ) -> None:
        item_id = f"item_{uuid.uuid4().hex}"
        await websocket.send_json(
            {
                "type": "response.output_item.added",
                "response_id": response_id,
                "output_index": 0,
                "item": {
                    "id": item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
            }
        )
        text = turn_payload["response"]["text"]
        if text:
            await websocket.send_json(
                {
                    "type": "response.output_text.delta",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text,
                }
            )
            await websocket.send_json(
                {
                    "type": "response.output_text.done",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": text,
                }
            )
        for chunk in _openai_realtime_audio_delta_chunks(turn_payload):
            await websocket.send_json(
                {
                    "type": "response.audio.delta",
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 1,
                    "delta": chunk,
                    "format": turn_payload["audio"]["format"],
                    "media_type": turn_payload["audio"]["media_type"],
                    "sample_rate": turn_payload["audio"]["sample_rate"],
                }
            )
        await websocket.send_json(
            {
                "type": "response.audio.done",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 1,
            }
        )
        completed_item = _openai_realtime_response(
            turn_payload,
            response_id=response_id,
            item_id=item_id,
            status="completed",
        )["output"][0]
        await websocket.send_json(
            {
                "type": "response.output_item.done",
                "response_id": response_id,
                "output_index": 0,
                "item": completed_item,
            }
        )
        await websocket.send_json(
            _completed_realtime_event(
                session_id=turn_payload["session_id"],
                turn_payload=turn_payload,
                openai_shape=True,
                response_id=response_id,
                item_id=item_id,
            )
        )

    async def _run_realtime_voice_turn(
        *,
        websocket: WebSocket,
        session: dict[str, Any],
        session_id: str,
        audio_bytes: bytes,
        event: dict[str, Any],
        openai_shape: bool = False,
    ) -> None:
        if not audio_bytes:
            await websocket.send_json({"type": "error", "error": {"message": "audio buffer is empty", "code": "empty_audio"}})
            return
        media_path = _bytes_to_temp_file(audio_bytes, filename=event.get("filename"))
        turn_payload: dict[str, Any] | None = None
        response_id = f"resp_{uuid.uuid4().hex}" if openai_shape else None
        if response_id:
            session["active_response_id"] = response_id
            session["last_response_id"] = response_id
        if openai_shape:
            await websocket.send_json(
                {
                    "type": "response.created",
                    "response": {
                        "id": response_id,
                        "object": "realtime.response",
                        "status": "in_progress",
                        "output": [],
                    },
                }
            )
        try:
            turn_payload = _run_voice_turn(
                session,
                media_path,
                response_format=event.get("response_format"),
                voice=event.get("voice"),
                language=event.get("language"),
                prompt=event.get("prompt"),
                temperature=event.get("temperature"),
                speed=event.get("speed"),
            )
        except Exception as exc:
            await websocket.send_json({"type": "error", "error": {"message": str(exc), "code": "voice_turn_failed"}})
        finally:
            media_path.unlink(missing_ok=True)
        if turn_payload is not None:
            if openai_shape:
                assert response_id is not None
                await _send_openai_realtime_turn_events(websocket, turn_payload, response_id=response_id)
                session["active_response_id"] = None
                return
            await websocket.send_json(
                _completed_realtime_event(
                    session_id=session_id,
                    turn_payload=turn_payload,
                    openai_shape=openai_shape,
                )
            )

    async def _run_realtime_text_response(
        *,
        websocket: WebSocket,
        session: dict[str, Any],
        event: dict[str, Any],
        openai_shape: bool,
    ) -> None:
        if not _can_create_text_only_realtime_response(session):
            await websocket.send_json({"type": "error", "error": {"message": "audio buffer is empty", "code": "empty_audio"}})
            return
        response_id = f"resp_{uuid.uuid4().hex}" if openai_shape else None
        if response_id:
            session["active_response_id"] = response_id
            session["last_response_id"] = response_id
        if openai_shape:
            await websocket.send_json(
                {
                    "type": "response.created",
                    "response": {
                        "id": response_id,
                        "object": "realtime.response",
                        "status": "in_progress",
                        "output": [],
                    },
                }
            )
        try:
            turn_payload = _run_voice_response(
                session,
                transcript_payload=None,
                response_format=event.get("response_format"),
                voice=event.get("voice"),
                speed=event.get("speed"),
            )
        except Exception as exc:
            await websocket.send_json({"type": "error", "error": {"message": str(exc), "code": "voice_turn_failed"}})
            return
        if openai_shape:
            assert response_id is not None
            await _send_openai_realtime_turn_events(websocket, turn_payload, response_id=response_id)
            session["active_response_id"] = None
            return
        await websocket.send_json(
            _completed_realtime_event(
                session_id=session["id"],
                turn_payload=turn_payload,
                openai_shape=False,
            )
        )

    @app.websocket("/v1/local/voice/sessions/{session_id}/realtime")
    async def realtime_voice_session(session_id: str, websocket: WebSocket) -> None:
        await _realtime_voice_websocket(session_id=session_id, websocket=websocket, openai_shape=False)

    @app.websocket("/v1/realtime/sessions/{session_id}")
    async def openai_realtime_session(session_id: str, websocket: WebSocket) -> None:
        await _realtime_voice_websocket(session_id=session_id, websocket=websocket, openai_shape=True)

    async def _realtime_voice_websocket(session_id: str, websocket: WebSocket, *, openai_shape: bool) -> None:
        await websocket.accept()
        session = app.state.voice_sessions.get(session_id)
        if not session:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": {"message": f"The voice session '{session_id}' does not exist", "code": "not_found"},
                }
            )
            await websocket.close(code=1008)
            return

        audio_buffer = bytearray()
        await websocket.send_json(
            {"type": "session.created", "session": _public_session_for_realtime(session, openai_shape=openai_shape)}
        )
        try:
            while True:
                event = await websocket.receive_json()
                event_type = event.get("type")
                if event_type == "session.update":
                    _update_voice_session_from_realtime_event(session, event)
                    session_payload = _public_session_for_realtime(session, openai_shape=openai_shape)
                    await websocket.send_json({"type": "session.updated", "session": session_payload})
                    continue
                if event_type in {"session.close", "close"}:
                    session["status"] = "ended"
                    session["updated_at"] = int(time.time())
                    app.state.voice_sessions.pop(session_id, None)
                    await websocket.send_json(
                        {"type": "session.closed", "session": _public_session_for_realtime(session, openai_shape=openai_shape)}
                    )
                    await websocket.close()
                    return
                if event_type == "response.cancel":
                    response_id = event.get("response_id") or session.get("active_response_id") or session.get("last_response_id")
                    if response_id:
                        session["cancelled_response_ids"].append(response_id)
                    await websocket.send_json(
                        {
                            "type": "response.cancelled",
                            "session_id": session_id,
                            "response_id": response_id,
                            "status": "cancelled" if response_id else "no_active_response",
                        }
                    )
                    continue
                if event_type == "input_audio_buffer.clear":
                    audio_buffer.clear()
                    await websocket.send_json({"type": "input_audio_buffer.cleared", "session_id": session_id})
                    continue
                if event_type == "conversation.item.create":
                    try:
                        item = _append_realtime_conversation_item(session, event)
                    except ValueError as exc:
                        await websocket.send_json(
                            {"type": "error", "error": {"message": str(exc), "code": "invalid_conversation_item"}}
                        )
                        continue
                    await websocket.send_json(
                        {
                            "type": "conversation.item.created",
                            "previous_item_id": event.get("previous_item_id"),
                            "item": item,
                        }
                    )
                    continue
                if event_type == "conversation.item.retrieve":
                    try:
                        item = _retrieve_realtime_conversation_item(session, event)
                    except (LookupError, ValueError) as exc:
                        await websocket.send_json(
                            {"type": "error", "error": {"message": str(exc), "code": "item_not_found"}}
                        )
                        continue
                    await websocket.send_json({"type": "conversation.item.retrieved", "item": item})
                    continue
                if event_type == "conversation.item.delete":
                    try:
                        item = _delete_realtime_conversation_item(session, event)
                    except (LookupError, ValueError) as exc:
                        await websocket.send_json(
                            {"type": "error", "error": {"message": str(exc), "code": "item_not_found"}}
                        )
                        continue
                    await websocket.send_json(
                        {"type": "conversation.item.deleted", "item_id": item["id"], "item": item}
                    )
                    continue
                if event_type == "conversation.item.truncate":
                    try:
                        item = _truncate_realtime_conversation_item(session, event)
                    except LookupError as exc:
                        await websocket.send_json(
                            {"type": "error", "error": {"message": str(exc), "code": "item_not_found"}}
                        )
                        continue
                    except ValueError as exc:
                        await websocket.send_json(
                            {"type": "error", "error": {"message": str(exc), "code": "invalid_conversation_item"}}
                        )
                        continue
                    await websocket.send_json({"type": "conversation.item.truncated", "item": item})
                    continue
                if event_type == "input_audio_buffer.append":
                    try:
                        audio_buffer.extend(base64.b64decode(event.get("audio", ""), validate=True))
                    except Exception:
                        await websocket.send_json(
                            {"type": "error", "error": {"message": "audio must be base64 encoded", "code": "invalid_audio"}}
                        )
                        continue
                    await websocket.send_json(
                        {
                            "type": "input_audio_buffer.appended",
                            "session_id": session_id,
                            "buffer_bytes": len(audio_buffer),
                        }
                    )
                    continue
                if event_type in {"input_audio_buffer.commit", "voice.turn", "response.create"}:
                    if event_type == "voice.turn":
                        try:
                            audio_bytes = base64.b64decode(event.get("audio", ""), validate=True)
                        except Exception:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "error": {"message": "audio must be base64 encoded", "code": "invalid_audio"},
                                }
                            )
                            continue
                    else:
                        audio_bytes = bytes(audio_buffer)
                        audio_buffer.clear()
                    if not audio_bytes and event_type == "response.create":
                        await _run_realtime_text_response(
                            websocket=websocket,
                            session=session,
                            event=event,
                            openai_shape=openai_shape,
                        )
                        continue
                    await _run_realtime_voice_turn(
                        websocket=websocket,
                        session=session,
                        session_id=session_id,
                        audio_bytes=audio_bytes,
                        event=event,
                        openai_shape=openai_shape,
                    )
                    continue

                await websocket.send_json(
                    {"type": "error", "error": {"message": f"unsupported event type: {event_type}", "code": "unsupported_event"}}
                )
        except WebSocketDisconnect:
            return

    @app.post("/v1/images/generations")
    def create_image_generation(request: Request, payload: ImageGenerationRequest) -> dict[str, Any]:
        response_format = payload.response_format or active_settings.image_default_response_format
        if response_format not in {"b64_json", "url"}:
            raise openai_error(400, "response_format must be b64_json or url", param="response_format")
        if payload.model and payload.model != active_settings.image_model_id:
            raise openai_error(
                404,
                f"The image model '{payload.model}' is not available. Configured image model is '{active_settings.image_model_id}'.",
                param="model",
                code="model_not_found",
            )
        try:
            with coordinator.execute("image"):
                prepare_image_generation_slot()
                output_format = normalize_image_output_format(payload.output_format)
                options = normalize_image_generation_options(
                    prompt=payload.prompt,
                    size=payload.size or active_settings.image_default_size,
                    default_steps=active_settings.image_num_inference_steps,
                    default_guidance_scale=active_settings.image_guidance_scale,
                    num_inference_steps=payload.num_inference_steps,
                    guidance_scale=payload.guidance_scale,
                    quality=payload.quality,
                    style=payload.style,
                    background=payload.background,
                    moderation=payload.moderation,
                )
                cleanup_image_outputs(
                    output_dir=active_settings.resolved_image_output_dir,
                    retention_seconds=active_settings.image_output_retention_seconds,
                )
                data = []
                for index in range(payload.n):
                    seed = payload.seed + index if payload.seed is not None else None
                    image = active_image_manager.generate(
                        prompt=options.prompt,
                        negative_prompt=payload.negative_prompt,
                        width=options.width,
                        height=options.height,
                        num_inference_steps=options.num_inference_steps,
                        guidance_scale=options.guidance_scale,
                        seed=seed,
                    )
                    data.append(
                        image_response_item(
                            request=request,
                            image=image,
                            response_format=response_format,
                            output_format=output_format,
                            output_compression=payload.output_compression,
                            revised_prompt=options.prompt,
                        )
                    )
        except ImageNotDownloadedError as exc:
            raise openai_error(
                409,
                "The configured image model is not downloaded. Call POST /v1/local/images/download first.",
                type_="invalid_request_error",
                param=exc.asset,
                code="image_model_not_downloaded",
            ) from exc
        except ImageParameterError as exc:
            raise openai_error(400, str(exc), type_="invalid_request_error", param=exc.param) from exc
        except ValueError as exc:
            raise openai_error(400, str(exc), type_="invalid_request_error", param="size") from exc
        except RuntimeError as exc:
            raise openai_error(503, str(exc), type_="server_error", code="image_backend_missing") from exc
        except Exception as exc:
            raise openai_error(500, str(exc), type_="server_error", code="image_generation_failed") from exc

        return {
            "created": int(time.time()),
            "data": data,
        }

    @app.post("/v1/images/variations")
    def create_image_variation(
        request: Request,
        image: UploadFile = File(...),
        model: str | None = Form(None),
        n: int = Form(1),
        size: str | None = Form(None),
        response_format: str | None = Form(None),
        user: str | None = Form(None),
        seed: int | None = Form(None),
        output_format: str | None = Form(None),
        output_compression: int | None = Form(None),
        num_inference_steps: int | None = Form(None),
        guidance_scale: float | None = Form(None),
        strength: float | None = Form(None),
    ) -> dict[str, Any]:
        _ = user
        response_format = response_format or active_settings.image_default_response_format
        if response_format not in {"b64_json", "url"}:
            raise openai_error(400, "response_format must be b64_json or url", param="response_format")
        if output_compression is not None and not 0 <= output_compression <= 100:
            raise openai_error(400, "output_compression must be between 0 and 100", param="output_compression")
        if n < 1 or n > 10:
            raise openai_error(400, "n must be between 1 and 10", param="n")
        if model and model not in {active_settings.image_model_id, "dall-e-2"}:
            raise openai_error(
                404,
                f"The image variation model '{model}' is not available. Configured image model is "
                f"'{active_settings.image_model_id}'.",
                param="model",
                code="model_not_found",
            )
        try:
            with coordinator.execute("image"):
                prepare_image_generation_slot()
                output_format = normalize_image_output_format(output_format)
                options = normalize_image_variation_options(
                    size=size,
                    default_size=active_settings.image_variation_default_size,
                    default_prompt=active_settings.image_variation_prompt,
                    default_steps=active_settings.image_variation_num_inference_steps,
                    default_guidance_scale=active_settings.image_variation_guidance_scale,
                    default_strength=active_settings.image_variation_strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                )
                source_image = prepare_variation_input(
                    image_bytes=image.file.read(),
                    width=options.width,
                    height=options.height,
                )
                cleanup_image_outputs(
                    output_dir=active_settings.resolved_image_output_dir,
                    retention_seconds=active_settings.image_output_retention_seconds,
                )
                data = []
                for index in range(n):
                    request_seed = seed + index if seed is not None else None
                    varied = active_image_manager.variation(
                        prompt=options.prompt,
                        image=source_image,
                        width=options.width,
                        height=options.height,
                        num_inference_steps=options.num_inference_steps,
                        guidance_scale=options.guidance_scale,
                        strength=options.strength,
                        seed=request_seed,
                    )
                    data.append(
                        image_response_item(
                            request=request,
                            image=varied,
                            response_format=response_format,
                            output_format=output_format,
                            output_compression=output_compression,
                        )
                    )
        except ImageNotDownloadedError as exc:
            raise openai_error(
                409,
                "The configured image model is not downloaded. Call POST /v1/local/images/download first.",
                type_="invalid_request_error",
                param=exc.asset,
                code="image_model_not_downloaded",
            ) from exc
        except ImageParameterError as exc:
            raise openai_error(400, str(exc), type_="invalid_request_error", param=exc.param) from exc
        except ValueError as exc:
            raise openai_error(400, str(exc), type_="invalid_request_error", param="size") from exc
        except RuntimeError as exc:
            raise openai_error(503, str(exc), type_="server_error", code="image_variation_backend_missing") from exc
        except Exception as exc:
            raise openai_error(500, str(exc), type_="server_error", code="image_variation_failed") from exc

        return {
            "created": int(time.time()),
            "data": data,
        }

    @app.post("/v1/images/edits")
    def create_image_edit(
        request: Request,
        prompt: str = Form(...),
        image: UploadFile = File(...),
        mask: UploadFile | None = File(None),
        model: str | None = Form(None),
        n: int = Form(1),
        size: str | None = Form(None),
        response_format: str | None = Form(None),
        user: str | None = Form(None),
        negative_prompt: str | None = Form(None),
        num_inference_steps: int | None = Form(None),
        guidance_scale: float | None = Form(None),
        strength: float | None = Form(None),
        seed: int | None = Form(None),
        quality: str | None = Form(None),
        background: str | None = Form(None),
        input_fidelity: str | None = Form(None),
        moderation: str | None = Form(None),
        output_format: str | None = Form(None),
        output_compression: int | None = Form(None),
    ) -> dict[str, Any]:
        _ = user
        response_format = response_format or active_settings.image_default_response_format
        if response_format not in {"b64_json", "url"}:
            raise openai_error(400, "response_format must be b64_json or url", param="response_format")
        if output_compression is not None and not 0 <= output_compression <= 100:
            raise openai_error(400, "output_compression must be between 0 and 100", param="output_compression")
        if n < 1:
            raise openai_error(400, "n must be greater than or equal to 1", param="n")
        if model and model != active_settings.image_edit_model_id:
            raise openai_error(
                404,
                f"The image edit model '{model}' is not available. Configured image edit model is "
                f"'{active_settings.image_edit_model_id}'.",
                param="model",
                code="model_not_found",
            )
        try:
            with coordinator.execute("image_edit"):
                prepare_image_edit_slot()
                resolved_output_format = normalize_image_output_format(output_format)
                options = normalize_image_edit_options(
                    prompt=prompt,
                    size=size if size and size != "auto" else active_settings.image_edit_default_size,
                    default_steps=active_settings.image_edit_num_inference_steps,
                    default_guidance_scale=active_settings.image_edit_guidance_scale,
                    default_strength=active_settings.image_edit_strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    quality=quality,
                    background=background,
                    input_fidelity=input_fidelity,
                    moderation=moderation,
                )
                image_bytes = image.file.read()
                mask_bytes = mask.file.read() if mask is not None else None
                base_image, mask_image = prepare_inpaint_inputs(
                    image_bytes=image_bytes,
                    mask_bytes=mask_bytes,
                    width=options.width,
                    height=options.height,
                )
                cleanup_image_outputs(
                    output_dir=active_settings.resolved_image_output_dir,
                    retention_seconds=active_settings.image_output_retention_seconds,
                )
                data = []
                for index in range(n):
                    request_seed = seed + index if seed is not None else None
                    edited = active_image_edit_manager.edit(
                        prompt=options.prompt,
                        negative_prompt=negative_prompt,
                        image=base_image,
                        mask_image=mask_image,
                        width=options.width,
                        height=options.height,
                        num_inference_steps=options.num_inference_steps,
                        guidance_scale=options.guidance_scale,
                        strength=options.strength,
                        seed=request_seed,
                    )
                    data.append(
                        image_response_item(
                            request=request,
                            image=edited,
                            response_format=response_format,
                            output_format=resolved_output_format,
                            output_compression=output_compression,
                            revised_prompt=options.prompt,
                        )
                    )
        except ImageNotDownloadedError as exc:
            raise openai_error(
                409,
                "The configured image edit model is not downloaded. Call POST /v1/local/images/edit/download first.",
                type_="invalid_request_error",
                param=exc.asset,
                code="image_edit_model_not_downloaded",
            ) from exc
        except ImageParameterError as exc:
            raise openai_error(400, str(exc), type_="invalid_request_error", param=exc.param) from exc
        except ValueError as exc:
            raise openai_error(400, str(exc), type_="invalid_request_error", param="size") from exc
        except RuntimeError as exc:
            raise openai_error(503, str(exc), type_="server_error", code="image_edit_backend_missing") from exc
        except Exception as exc:
            raise openai_error(500, str(exc), type_="server_error", code="image_edit_failed") from exc

        return {
            "created": int(time.time()),
            "data": data,
        }

    @app.post("/v1/audio/speech")
    def create_speech(request: SpeechRequest) -> Response:
        if request.model and request.model not in {active_settings.tts_model_id, "tts-1", "tts-1-hd", "kokoro"}:
            raise openai_error(
                404,
                f"The audio model '{request.model}' is not available. Configured audio model is '{active_settings.tts_model_id}'.",
                param="model",
                code="model_not_found",
            )
        try:
            speech = active_audio_manager.synthesize(
                text=request.input,
                voice=request.voice,
                speed=request.speed,
                lang=request.lang,
                is_phonemes=request.is_phonemes,
                trim=request.trim,
            )
        except AudioNotDownloadedError as exc:
            raise openai_error(
                409,
                f"The configured {exc.asset} is not downloaded. Call POST /v1/local/audio/download first.",
                type_="invalid_request_error",
                param=exc.asset,
                code="audio_not_downloaded",
            ) from exc
        except AssertionError as exc:
            raise openai_error(400, str(exc), type_="invalid_request_error", param="voice") from exc
        except RuntimeError as exc:
            raise openai_error(503, str(exc), type_="server_error", code="audio_backend_missing") from exc

        try:
            content, media_type = encode_audio(
                speech.samples,
                speech.sample_rate,
                request.response_format,
                ffmpeg_path=active_settings.tts_ffmpeg_path,
            )
        except ValueError as exc:
            raise openai_error(400, str(exc), type_="invalid_request_error", param="response_format") from exc
        except AudioEncoderMissingError as exc:
            raise openai_error(
                503,
                str(exc),
                type_="server_error",
                param="response_format",
                code="audio_encoder_missing",
            ) from exc
        except AudioEncodingError as exc:
            raise openai_error(
                500,
                str(exc),
                type_="server_error",
                param="response_format",
                code="audio_encoding_failed",
            ) from exc
        except RuntimeError as exc:
            raise openai_error(503, str(exc), type_="server_error", code="audio_backend_missing") from exc

        return Response(
            content=content,
            media_type=media_type,
            headers={
                "X-LAAS-Audio-Model": active_settings.tts_model_id,
                "X-LAAS-Audio-Sample-Rate": str(speech.sample_rate),
            },
        )

    @app.post("/v1/audio/transcriptions")
    async def create_transcription(
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
        response_format: str = Form("json"),
        temperature: float | None = Form(0.0),
        timestamp_granularities: list[str] | None = Form(None, alias="timestamp_granularities[]"),
    ) -> Any:
        return await _create_transcription_response(
            file=file,
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            translate=False,
            task="transcribe",
            timestamp_granularities=timestamp_granularities,
        )

    @app.post("/v1/audio/translations")
    async def create_translation(
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),
        prompt: str | None = Form(None),
        response_format: str = Form("json"),
        temperature: float | None = Form(0.0),
    ) -> Any:
        return await _create_transcription_response(
            file=file,
            model=model,
            language=None,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            translate=True,
            task="translate",
        )

    async def _create_transcription_response(
        *,
        file: UploadFile,
        model: str,
        language: str | None,
        prompt: str | None,
        response_format: str,
        temperature: float | None,
        translate: bool,
        task: str,
        timestamp_granularities: list[str] | None = None,
    ) -> Any:
        if model not in {active_settings.stt_model_id, "whisper-1"}:
            raise openai_error(
                404,
                f"The transcription model '{model}' is not available. Configured transcription model is '{active_settings.stt_model_id}'.",
                param="model",
                code="model_not_found",
            )
        media_path = await _upload_to_temp_file(file)
        try:
            result = active_transcription_manager.transcribe(
                media_path=media_path,
                language=language,
                prompt=prompt,
                temperature=temperature,
                translate=translate,
            )
            payload = transcription_to_response(
                result,
                response_format,
                task=task,
                timestamp_granularities=timestamp_granularities,
            )
        except TranscriptionNotDownloadedError as exc:
            raise openai_error(
                409,
                f"The configured {exc.asset} is not downloaded. Call POST /v1/local/transcription/download first.",
                type_="invalid_request_error",
                param=exc.asset,
                code="transcription_not_downloaded",
            ) from exc
        except ValueError as exc:
            raise openai_error(400, str(exc), type_="invalid_request_error", param="response_format") from exc
        except RuntimeError as exc:
            raise openai_error(503, str(exc), type_="server_error", code="transcription_backend_missing") from exc
        finally:
            media_path.unlink(missing_ok=True)

        if response_format == "text":
            return PlainTextResponse(str(payload))
        if response_format == "srt":
            return PlainTextResponse(str(payload), media_type="application/x-subrip")
        if response_format == "vtt":
            return PlainTextResponse(str(payload), media_type="text/vtt")
        return payload

    app.state.coordinator = coordinator
    app.include_router(build_openai_router(active_manager, active_embedding_manager, coordinator))
    return app


async def _upload_to_temp_file(file: UploadFile) -> Path:
    content = await file.read()
    if not content:
        raise openai_error(400, "uploaded audio file is empty", param="file")
    suffix = Path(file.filename or "").suffix or ".audio"
    fd, raw_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
    except Exception:
        os.close(fd)
        raise
    return Path(raw_path)


def _bytes_to_temp_file(content: bytes, *, filename: str | None = None) -> Path:
    if not content:
        raise ValueError("audio buffer is empty")
    suffix = Path(filename or "").suffix or ".audio"
    fd, raw_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
    except Exception:
        os.close(fd)
        raise
    return Path(raw_path)


def _get_voice_session(session_id: str, sessions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    session = sessions.get(session_id)
    if not session:
        raise openai_error(404, f"The voice session '{session_id}' does not exist", param="session_id", code="not_found")
    return session


def _public_voice_session(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": session["id"],
        "object": session["object"],
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "status": session["status"],
        "model": session["model"],
        "voice": session["voice"],
        "response_format": session["response_format"],
        "turn_count": len(session["turns"]),
    }


app = create_app()
