from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Response, UploadFile
from fastapi.responses import PlainTextResponse

from .errors import openai_error
from .manager import ModelManager, ModelNotDownloadedError
from .openai_compat import build_openai_router
from .schemas import (
    DownloadAudioRequest,
    DownloadModelRequest,
    DownloadTranscriptionRequest,
    LoadAudioRequest,
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
) -> FastAPI:
    active_settings = settings or load_settings()
    active_manager = manager or ModelManager(active_settings)
    active_audio_manager = audio_manager or AudioManager(active_settings)
    active_transcription_manager = transcription_manager or TranscriptionManager(active_settings)

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

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": active_manager.is_loaded,
            "audio_model_loaded": active_audio_manager.is_loaded,
            "transcription_model_loaded": active_transcription_manager.is_loaded,
            "voice_stack_loaded": active_audio_manager.is_loaded and active_transcription_manager.is_loaded,
        }

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
        return active_manager.unload().model_dump()

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
        return LocalVoiceStackStatus(
            tts=audio_status,
            transcription=transcription_status,
            is_loaded=False,
        ).model_dump()

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
            payload = transcription_to_response(result, response_format, task=task)
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

    app.include_router(build_openai_router(active_manager))
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


app = create_app()
