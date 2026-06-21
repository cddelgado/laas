from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterable

from fastapi import APIRouter, Body, File, Form, Query, Response, UploadFile
from fastapi.responses import StreamingResponse

from .embedding import EmbeddingManager, EmbeddingNotDownloadedError, encode_embedding, estimate_tokens
from .errors import openai_error
from .manager import ModelManager, ModelNotDownloadedError
from .multimodal import VideoExtractionConfig, normalize_chat_messages, normalize_content_parts
from .schemas import ChatCompletionRequest, CompletionRequest, EmbeddingRequest, ModelList, OpenAIModel, ResponseRequest
from .settings import Settings
from .storage import LocalFileNotFoundError, LocalStorage, VectorStoreFileNotFoundError, VectorStoreNotFoundError
from .tools import normalize_tools_for_responses, parse_tool_calls, remove_tool_call_markup, validate_tool_choice
from .concurrency import ConcurrencyCoordinator


COMPATIBILITY_MATRIX: list[dict[str, Any]] = [
    {
        "surface": "Models",
        "status": "supported",
        "endpoints": ["GET /v1/models", "GET /v1/models/{model_id}"],
        "notes": "Lists configured local text, embedding, image, and image edit model IDs.",
    },
    {
        "surface": "Chat Completions",
        "status": "supported",
        "endpoints": ["POST /v1/chat/completions"],
        "notes": "Supports text, multimodal content normalization, streaming, and translated Gemma tool calls.",
    },
    {
        "surface": "Completions",
        "status": "supported",
        "endpoints": ["POST /v1/completions"],
        "notes": "Legacy text completion compatibility over the local llama.cpp backend.",
    },
    {
        "surface": "Responses",
        "status": "supported",
        "endpoints": [
            "POST /v1/responses",
            "GET /v1/responses/{response_id}",
            "DELETE /v1/responses/{response_id}",
            "GET /v1/responses/{response_id}/input_items",
        ],
        "notes": "Local in-memory response storage with text and function-call output normalization.",
    },
    {
        "surface": "Embeddings",
        "status": "supported",
        "endpoints": ["POST /v1/embeddings"],
        "notes": "Uses the configured local Sentence Transformers embedding backend.",
    },
    {
        "surface": "Images",
        "status": "supported",
        "endpoints": ["POST /v1/images/generations", "POST /v1/images/variations", "POST /v1/images/edits"],
        "notes": "Local Diffusers-backed generation, variation, and inpainting/edit compatibility.",
    },
    {
        "surface": "Audio",
        "status": "supported",
        "endpoints": ["POST /v1/audio/speech", "POST /v1/audio/transcriptions", "POST /v1/audio/translations"],
        "notes": "Local Kokoro TTS and whisper.cpp-compatible transcription/translation stack.",
    },
    {
        "surface": "Files",
        "status": "supported",
        "endpoints": [
            "POST /v1/files",
            "GET /v1/files",
            "GET /v1/files/{file_id}",
            "GET /v1/files/{file_id}/content",
            "DELETE /v1/files/{file_id}",
        ],
        "notes": "Local on-disk file storage with SQLite metadata.",
    },
    {
        "surface": "Vector Stores",
        "status": "supported",
        "endpoints": [
            "POST /v1/vector_stores",
            "GET /v1/vector_stores",
            "GET /v1/vector_stores/{vector_store_id}",
            "DELETE /v1/vector_stores/{vector_store_id}",
            "POST /v1/vector_stores/{vector_store_id}/files",
            "GET /v1/vector_stores/{vector_store_id}/files",
            "GET /v1/vector_stores/{vector_store_id}/files/{file_id}",
            "DELETE /v1/vector_stores/{vector_store_id}/files/{file_id}",
            "POST /v1/local/vector_stores/{vector_store_id}/search",
        ],
        "notes": "Local SQLite metadata and chunk store with embedding-backed cosine search.",
    },
    {
        "surface": "Uploads, Batches, Fine-tuning, Moderations",
        "status": "unsupported",
        "endpoints": [
            "/v1/uploads",
            "/v1/batches",
            "/v1/fine_tuning/jobs",
            "/v1/moderations",
        ],
        "notes": "These cloud/account or hosted-storage APIs are not implemented by the local inference host.",
    },
    {
        "surface": "Administration, Containers, Skills, ChatKit, Realtime",
        "status": "not_applicable",
        "endpoints": [],
        "notes": "Not registered. These require OpenAI-hosted account, organization, realtime, or cloud resources.",
    },
]


def build_openai_router(
    manager: ModelManager,
    embedding_manager: EmbeddingManager,
    coordinator: ConcurrencyCoordinator | None = None,
    storage: LocalStorage | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/v1")
    response_store: dict[str, dict[str, Any]] = {}
    active_storage = storage or LocalStorage(manager.settings, embedding_manager)

    @router.get("/models", response_model=ModelList)
    def list_models() -> ModelList:
        return ModelList(
            data=[
                OpenAIModel(
                    id=manager.settings.model_id,
                    created=1712966400,
                    owned_by="google-local-gguf",
                ),
                OpenAIModel(
                    id=embedding_manager.settings.embedding_model_id,
                    created=1712966400,
                    owned_by="sentence-transformers-local",
                ),
                OpenAIModel(
                    id=manager.settings.image_model_id,
                    created=1712966400,
                    owned_by="stability-local-diffusers",
                ),
                OpenAIModel(
                    id=manager.settings.image_edit_model_id,
                    created=1712966400,
                    owned_by="stability-local-diffusers",
                ),
            ]
        )

    @router.get("/models/{model_id}")
    def retrieve_model(model_id: str) -> dict[str, Any]:
        known_models = {
            manager.settings.model_id,
            embedding_manager.settings.embedding_model_id,
            manager.settings.image_model_id,
            manager.settings.image_edit_model_id,
        }
        if model_id not in known_models:
            raise openai_error(404, f"The model '{model_id}' does not exist", param="model", code="model_not_found")
        if model_id == manager.settings.model_id:
            owned_by = "google-local-gguf"
        elif model_id == manager.settings.image_model_id:
            owned_by = "stability-local-diffusers"
        elif model_id == manager.settings.image_edit_model_id:
            owned_by = "stability-local-diffusers"
        else:
            owned_by = "sentence-transformers-local"
        return OpenAIModel(id=model_id, created=1712966400, owned_by=owned_by).model_dump()

    @router.post("/chat/completions")
    def create_chat_completion(request: ChatCompletionRequest) -> Any:
        _assert_model(request.model, manager)
        _validate_capabilities(request, manager)
        messages = normalize_chat_messages(
            [message.model_dump(exclude_none=True) for message in request.messages],
            video_config=_video_config(manager.settings),
        )
        tools = normalize_tools_for_responses(request.tools)
        acquired = False
        try:
            if coordinator:
                if request.stream:
                    coordinator.acquire("llm")
                    acquired = True
                else:
                    with coordinator.execute("llm"):
                        return _chat_completion_with_backend(request, manager, messages, tools)
            else:
                return _chat_completion_with_backend(request, manager, messages, tools)
            backend = _get_backend(manager)
            result = backend.chat_completion(
                messages=messages,
                model=manager.settings.model_id,
                tools=tools,
                tool_choice=request.tool_choice,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.requested_max_tokens,
                stream=request.stream,
                extra_params=_chat_sampling_params(request),
            )
            if request.stream:
                chunks = _normalize_chat_stream(result, manager.settings.model_id, tools, request.tool_choice)
                if coordinator:
                    chunks = coordinator.wrap_stream("llm", chunks)
                    acquired = False
                return _sse(chunks)
            return _normalize_chat_response(result, manager.settings.model_id, tools, request.tool_choice)
        except ModelNotDownloadedError as exc:
            raise openai_error(
                409,
                f"The configured {exc.asset} is not downloaded. Call POST /v1/local/models/download and "
                "POST /v1/local/models/load before inference, or set LAAS_AUTO_DOWNLOAD=true to allow "
                "LAAS to download missing model assets during load.",
                type_="invalid_request_error",
                param=exc.asset,
                code="model_not_downloaded",
            ) from exc
        except Exception:
            raise
        finally:
            if coordinator and acquired:
                coordinator.release("llm")

    @router.post("/completions")
    def create_completion(request: CompletionRequest) -> Any:
        _assert_model(request.model, manager)
        prompt = request.prompt[0] if isinstance(request.prompt, list) else request.prompt
        acquired = False
        try:
            if coordinator:
                if request.stream:
                    coordinator.acquire("llm")
                    acquired = True
                else:
                    with coordinator.execute("llm"):
                        return _completion_with_backend(request, manager, prompt)
            else:
                return _completion_with_backend(request, manager, prompt)
            result = _get_backend(manager).completion(
                prompt=prompt,
                model=manager.settings.model_id,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stream=request.stream,
                extra_params=_completion_sampling_params(request),
            )
            if request.stream:
                chunks = _normalize_completion_stream(result, manager.settings.model_id)
                if coordinator:
                    chunks = coordinator.wrap_stream("llm", chunks)
                    acquired = False
                return _sse(chunks)
            return _normalize_completion_response(result, manager.settings.model_id)
        except ModelNotDownloadedError as exc:
            raise openai_error(
                409,
                f"The configured {exc.asset} is not downloaded. Call POST /v1/local/models/download and "
                "POST /v1/local/models/load before inference, or set LAAS_AUTO_DOWNLOAD=true to allow "
                "LAAS to download missing model assets during load.",
                type_="invalid_request_error",
                param=exc.asset,
                code="model_not_downloaded",
            ) from exc
        except Exception:
            raise
        finally:
            if coordinator and acquired:
                coordinator.release("llm")

    @router.post("/responses")
    def create_response(request: ResponseRequest) -> Any:
        _assert_model(request.model, manager)
        previous_messages: list[dict[str, Any]] = []
        if request.previous_response_id:
            previous = response_store.get(request.previous_response_id)
            if not previous:
                raise openai_error(
                    404,
                    f"The response '{request.previous_response_id}' does not exist",
                    param="previous_response_id",
                    code="response_not_found",
                )
            previous_messages = _response_to_messages(previous["response"])
        messages = [*previous_messages, *_responses_input_to_messages(request, manager.settings)]
        chat_request = ChatCompletionRequest(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_completion_tokens=request.max_output_tokens,
            stream=request.stream,
            tools=normalize_tools_for_responses(request.tools),
            tool_choice=request.tool_choice,
            response_format=_response_text_format(request.text),
            stop=request.stop,
            seed=request.seed,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            repeat_penalty=request.repeat_penalty,
            top_k=request.top_k,
            min_p=request.min_p,
            typical_p=request.typical_p,
            tfs_z=request.tfs_z,
            mirostat_mode=request.mirostat_mode,
            mirostat_tau=request.mirostat_tau,
            mirostat_eta=request.mirostat_eta,
            logit_bias=request.logit_bias,
            logprobs=request.logprobs,
            top_logprobs=request.top_logprobs,
        )
        _validate_capabilities(chat_request, manager)
        normalized_messages = normalize_chat_messages(
            [message.model_dump(exclude_none=True) for message in chat_request.messages],
            video_config=_video_config(manager.settings),
        )
        acquired = False
        try:
            if coordinator:
                if request.stream:
                    coordinator.acquire("llm")
                    acquired = True
                else:
                    with coordinator.execute("llm"):
                        return _response_with_backend(request, chat_request, manager, normalized_messages, response_store)
            else:
                return _response_with_backend(request, chat_request, manager, normalized_messages, response_store)
            result = _get_backend(manager).chat_completion(
                messages=normalized_messages,
                model=manager.settings.model_id,
                tools=chat_request.tools,
                tool_choice=chat_request.tool_choice,
                temperature=chat_request.temperature,
                top_p=chat_request.top_p,
                max_tokens=chat_request.requested_max_tokens,
                stream=request.stream,
                extra_params=_chat_sampling_params(chat_request),
            )
            if request.stream:
                chat_chunks = _normalize_chat_stream(
                    result,
                    manager.settings.model_id,
                    chat_request.tools,
                    chat_request.tool_choice,
                )
                chunks = _responses_stream(chat_chunks, manager.settings.model_id)
                if coordinator:
                    chunks = coordinator.wrap_stream("llm", chunks)
                    acquired = False
                return _sse(chunks)
            chat_response = _normalize_chat_response(result, manager.settings.model_id, chat_request.tools, chat_request.tool_choice)
            response = _chat_to_response(chat_response, request, manager.settings.model_id)
            if request.store:
                response_store[response["id"]] = {
                    "response": response,
                    "input": _response_input_items_for_storage(request),
                }
            return response
        except ModelNotDownloadedError as exc:
            raise openai_error(
                409,
                f"The configured {exc.asset} is not downloaded. Call POST /v1/local/models/download and "
                "POST /v1/local/models/load before inference, or set LAAS_AUTO_DOWNLOAD=true to allow "
                "LAAS to download missing model assets during load.",
                type_="invalid_request_error",
                param=exc.asset,
                code="model_not_downloaded",
            ) from exc
        except Exception:
            raise
        finally:
            if coordinator and acquired:
                coordinator.release("llm")

    @router.get("/responses/{response_id}")
    def retrieve_response(response_id: str) -> dict[str, Any]:
        stored = response_store.get(response_id)
        if not stored:
            raise openai_error(404, f"The response '{response_id}' does not exist", code="response_not_found")
        return stored["response"]

    @router.delete("/responses/{response_id}")
    def delete_response(response_id: str) -> dict[str, Any]:
        if response_id not in response_store:
            raise openai_error(404, f"The response '{response_id}' does not exist", code="response_not_found")
        del response_store[response_id]
        return {"id": response_id, "object": "response.deleted", "deleted": True}

    @router.get("/responses/{response_id}/input_items")
    def list_response_input_items(response_id: str) -> dict[str, Any]:
        stored = response_store.get(response_id)
        if not stored:
            raise openai_error(404, f"The response '{response_id}' does not exist", code="response_not_found")
        return {
            "object": "list",
            "data": stored["input"],
            "first_id": stored["input"][0].get("id") if stored["input"] else None,
            "last_id": stored["input"][-1].get("id") if stored["input"] else None,
            "has_more": False,
        }

    @router.post("/embeddings")
    def create_embedding(request: EmbeddingRequest) -> dict[str, Any]:
        model_id = request.model or embedding_manager.settings.embedding_model_id
        if model_id != embedding_manager.settings.embedding_model_id:
            raise openai_error(404, f"The embedding model '{model_id}' does not exist", param="model", code="model_not_found")
        inputs = _normalize_embedding_inputs(request.input)
        dimensions = request.dimensions or embedding_manager.settings.embedding_dimensions
        try:
            vectors = embedding_manager.embed(inputs, dimensions=dimensions)
        except EmbeddingNotDownloadedError as exc:
            raise openai_error(
                409,
                "The configured embedding model is not downloaded. Call POST /v1/local/embeddings/download and "
                "POST /v1/local/embeddings/load first, or set LAAS_EMBEDDING_AUTO_DOWNLOAD=true to allow "
                "LAAS to download missing embedding assets during load.",
                type_="invalid_request_error",
                param=exc.asset,
                code="embedding_model_not_downloaded",
            ) from exc
        except RuntimeError as exc:
            raise openai_error(503, str(exc), type_="server_error", code="embedding_backend_missing") from exc
        data = [
            {
                "object": "embedding",
                "index": index,
                "embedding": encode_embedding(vector, request.encoding_format),
            }
            for index, vector in enumerate(vectors)
        ]
        prompt_tokens = sum(estimate_tokens(value) for value in inputs)
        return {
            "object": "list",
            "data": data,
            "model": model_id,
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        }

    @router.get("/local/files/status")
    def local_files_status() -> dict[str, Any]:
        return active_storage.status()

    @router.post("/files")
    async def create_file(
        file: UploadFile = File(...),
        purpose: str = Form("assistants"),
    ) -> dict[str, Any]:
        try:
            return active_storage.create_file(
                filename=file.filename or "upload.bin",
                content=await file.read(),
                purpose=purpose,
                mime_type=file.content_type,
            )
        except ValueError as exc:
            raise openai_error(400, str(exc), param="file") from exc

    @router.get("/files")
    def list_files(purpose: str | None = Query(None)) -> dict[str, Any]:
        return {"object": "list", "data": active_storage.list_files(purpose=purpose)}

    @router.get("/files/{file_id}")
    def retrieve_file(file_id: str) -> dict[str, Any]:
        try:
            return active_storage.get_file(file_id)
        except LocalFileNotFoundError as exc:
            raise openai_error(404, f"The file '{file_id}' does not exist", code="file_not_found") from exc

    @router.delete("/files/{file_id}")
    def delete_file(file_id: str) -> dict[str, Any]:
        try:
            return active_storage.delete_file(file_id)
        except LocalFileNotFoundError as exc:
            raise openai_error(404, f"The file '{file_id}' does not exist", code="file_not_found") from exc

    @router.get("/files/{file_id}/content")
    def retrieve_file_content(file_id: str) -> Response:
        try:
            metadata = active_storage.get_file(file_id)
            path = active_storage.file_path(file_id)
        except LocalFileNotFoundError as exc:
            raise openai_error(404, f"The file '{file_id}' does not exist", code="file_not_found") from exc
        return Response(
            content=path.read_bytes(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{metadata["filename"]}"'},
        )

    @router.post("/vector_stores")
    def create_vector_store(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        return active_storage.create_vector_store(name=payload.get("name"), metadata=metadata)

    @router.get("/vector_stores")
    def list_vector_stores() -> dict[str, Any]:
        return {"object": "list", "data": active_storage.list_vector_stores(), "has_more": False}

    @router.get("/vector_stores/{vector_store_id}")
    def retrieve_vector_store(vector_store_id: str) -> dict[str, Any]:
        try:
            return active_storage.get_vector_store(vector_store_id)
        except VectorStoreNotFoundError as exc:
            raise openai_error(
                404,
                f"The vector store '{vector_store_id}' does not exist",
                code="vector_store_not_found",
            ) from exc

    @router.delete("/vector_stores/{vector_store_id}")
    def delete_vector_store(vector_store_id: str) -> dict[str, Any]:
        try:
            return active_storage.delete_vector_store(vector_store_id)
        except VectorStoreNotFoundError as exc:
            raise openai_error(
                404,
                f"The vector store '{vector_store_id}' does not exist",
                code="vector_store_not_found",
            ) from exc

    @router.post("/vector_stores/{vector_store_id}/files")
    def create_vector_store_file(vector_store_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        file_id = payload.get("file_id")
        if not isinstance(file_id, str) or not file_id:
            raise openai_error(400, "file_id is required", param="file_id")
        try:
            return active_storage.attach_file(vector_store_id=vector_store_id, file_id=file_id)
        except LocalFileNotFoundError as exc:
            raise openai_error(404, f"The file '{file_id}' does not exist", param="file_id", code="file_not_found") from exc
        except VectorStoreNotFoundError as exc:
            raise openai_error(
                404,
                f"The vector store '{vector_store_id}' does not exist",
                code="vector_store_not_found",
            ) from exc
        except EmbeddingNotDownloadedError as exc:
            raise openai_error(
                409,
                "The configured embedding model is not downloaded. Call POST /v1/local/embeddings/download first.",
                type_="invalid_request_error",
                param=exc.asset,
                code="embedding_model_not_downloaded",
            ) from exc
        except ValueError as exc:
            raise openai_error(400, str(exc), type_="invalid_request_error", param="file") from exc
        except RuntimeError as exc:
            raise openai_error(503, str(exc), type_="server_error", code="embedding_backend_missing") from exc

    @router.get("/vector_stores/{vector_store_id}/files")
    def list_vector_store_files(vector_store_id: str) -> dict[str, Any]:
        try:
            return {"object": "list", "data": active_storage.list_vector_store_files(vector_store_id), "has_more": False}
        except VectorStoreNotFoundError as exc:
            raise openai_error(
                404,
                f"The vector store '{vector_store_id}' does not exist",
                code="vector_store_not_found",
            ) from exc

    @router.get("/vector_stores/{vector_store_id}/files/{file_id}")
    def retrieve_vector_store_file(vector_store_id: str, file_id: str) -> dict[str, Any]:
        try:
            return active_storage.get_vector_store_file(vector_store_id=vector_store_id, file_id=file_id)
        except VectorStoreFileNotFoundError as exc:
            raise openai_error(404, f"The vector store file '{file_id}' does not exist", code="not_found") from exc

    @router.delete("/vector_stores/{vector_store_id}/files/{file_id}")
    def delete_vector_store_file(vector_store_id: str, file_id: str) -> dict[str, Any]:
        try:
            return active_storage.delete_vector_store_file(vector_store_id=vector_store_id, file_id=file_id)
        except VectorStoreFileNotFoundError as exc:
            raise openai_error(404, f"The vector store file '{file_id}' does not exist", code="not_found") from exc

    @router.post("/local/vector_stores/{vector_store_id}/search")
    def search_vector_store(vector_store_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        query = payload.get("query")
        if not isinstance(query, str):
            raise openai_error(400, "query is required", param="query")
        limit = int(payload.get("limit") or 8)
        try:
            return active_storage.search_vector_store(vector_store_id=vector_store_id, query=query, limit=limit)
        except VectorStoreNotFoundError as exc:
            raise openai_error(
                404,
                f"The vector store '{vector_store_id}' does not exist",
                code="vector_store_not_found",
            ) from exc
        except EmbeddingNotDownloadedError as exc:
            raise openai_error(
                409,
                "The configured embedding model is not downloaded. Call POST /v1/local/embeddings/download first.",
                type_="invalid_request_error",
                param=exc.asset,
                code="embedding_model_not_downloaded",
            ) from exc
        except ValueError as exc:
            raise openai_error(400, str(exc), type_="invalid_request_error", param="query") from exc
        except RuntimeError as exc:
            raise openai_error(503, str(exc), type_="server_error", code="embedding_backend_missing") from exc

    register_unsupported_routes(router)
    return router


def register_unsupported_routes(router: APIRouter) -> None:
    unsupported_routes = [
        ("/uploads", ["POST"], "Uploads"),
        ("/uploads/{upload_id}", ["GET", "POST", "DELETE"], "Uploads"),
        ("/batches", ["GET", "POST"], "Batches"),
        ("/batches/{batch_id}", ["GET"], "Batches"),
        ("/batches/{batch_id}/cancel", ["POST"], "Batches"),
        ("/fine_tuning/jobs", ["GET", "POST"], "Fine-tuning"),
        ("/fine_tuning/jobs/{job_id}", ["GET"], "Fine-tuning"),
        ("/fine_tuning/jobs/{job_id}/cancel", ["POST"], "Fine-tuning"),
        ("/moderations", ["POST"], "Moderations"),
    ]
    for path, methods, surface in unsupported_routes:
        router.add_api_route(
            path,
            _unsupported_route(surface),
            methods=methods,
            include_in_schema=False,
        )


def _unsupported_route(surface: str):
    def route() -> None:
        raise openai_error(
            501,
            f"The OpenAI {surface} API is not implemented by LAAS. This local host supports local inference endpoints only.",
            type_="invalid_request_error",
            param="endpoint",
            code="unsupported_endpoint",
        )

    return route


def _chat_completion_with_backend(
    request: ChatCompletionRequest,
    manager: ModelManager,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    result = _get_backend(manager).chat_completion(
        messages=messages,
        model=manager.settings.model_id,
        tools=tools,
        tool_choice=request.tool_choice,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.requested_max_tokens,
        stream=False,
        extra_params=_chat_sampling_params(request),
    )
    return _normalize_chat_response(result, manager.settings.model_id, tools, request.tool_choice)


def _completion_with_backend(
    request: CompletionRequest,
    manager: ModelManager,
    prompt: str,
) -> dict[str, Any]:
    result = _get_backend(manager).completion(
        prompt=prompt,
        model=manager.settings.model_id,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stream=False,
        extra_params=_completion_sampling_params(request),
    )
    return _normalize_completion_response(result, manager.settings.model_id)


def _response_with_backend(
    request: ResponseRequest,
    chat_request: ChatCompletionRequest,
    manager: ModelManager,
    normalized_messages: list[dict[str, Any]],
    response_store: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result = _get_backend(manager).chat_completion(
        messages=normalized_messages,
        model=manager.settings.model_id,
        tools=chat_request.tools,
        tool_choice=chat_request.tool_choice,
        temperature=chat_request.temperature,
        top_p=chat_request.top_p,
        max_tokens=chat_request.requested_max_tokens,
        stream=False,
        extra_params=_chat_sampling_params(chat_request),
    )
    chat_response = _normalize_chat_response(result, manager.settings.model_id, chat_request.tools, chat_request.tool_choice)
    response = _chat_to_response(chat_response, request, manager.settings.model_id)
    if request.store:
        response_store[response["id"]] = {
            "response": response,
            "input": _response_input_items_for_storage(request),
        }
    return response


def _assert_model(requested_model: str | None, manager: ModelManager) -> None:
    if requested_model and requested_model != manager.settings.model_id:
        raise openai_error(
            404,
            f"The model '{requested_model}' is not loaded. Loaded/configured model is '{manager.settings.model_id}'.",
            param="model",
            code="model_not_found",
        )


def _get_backend(manager: ModelManager):
    try:
        return manager.backend
    except ModelNotDownloadedError as exc:
        raise openai_error(
            409,
            f"The configured {exc.asset} is not downloaded. Call POST /v1/local/models/download and "
            "POST /v1/local/models/load before inference, or set LAAS_AUTO_DOWNLOAD=true to allow "
            "LAAS to download missing model assets during load.",
            type_="invalid_request_error",
            param=exc.asset,
            code="model_not_downloaded",
        ) from exc


def _validate_capabilities(request: ChatCompletionRequest, manager: ModelManager) -> None:
    if request.tools and not manager.capabilities.tool_calls:
        raise openai_error(400, "the loaded model does not support tool calls", param="tools")
    if request.modalities:
        unsupported_modalities = sorted(set(request.modalities) - {"text", "audio"})
        if unsupported_modalities:
            raise openai_error(
                400,
                f"unsupported response modalities: {', '.join(unsupported_modalities)}",
                param="modalities",
                code="unsupported_modality",
            )
        if "audio" in request.modalities and not manager.capabilities.audio_output:
            raise openai_error(
                400,
                "the loaded model does not support native audio output through Chat Completions; use /v1/audio/speech",
                param="modalities",
                code="unsupported_audio_output",
            )
    if request.audio and (not request.modalities or "audio" not in request.modalities):
        raise openai_error(
            400,
            "audio output options require modalities to include 'audio'",
            param="audio",
            code="invalid_audio_output",
        )
    try:
        validate_tool_choice(request.tool_choice, request.tools)
    except ValueError as exc:
        raise openai_error(400, str(exc), param="tool_choice") from exc
    for message in request.messages:
        content = message.content
        if not isinstance(content, list):
            continue
        for part in content:
            part_type = part.get("type") if isinstance(part, dict) else None
            if part_type in {"image_url", "input_image"} and not manager.capabilities.vision:
                raise openai_error(400, "the loaded model does not support image inputs", param="messages")
            if part_type in {"video_url", "input_video"} and not manager.capabilities.video:
                raise openai_error(400, "the loaded model does not support video inputs", param="messages")
            if part_type in {"audio", "input_audio"} and not manager.capabilities.audio_input:
                raise openai_error(400, "the loaded model does not support audio inputs", param="messages")


def _chat_sampling_params(request: ChatCompletionRequest) -> dict[str, Any]:
    return _present_params(
        request,
        [
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "repeat_penalty",
            "top_k",
            "min_p",
            "typical_p",
            "tfs_z",
            "mirostat_mode",
            "mirostat_tau",
            "mirostat_eta",
            "response_format",
            "logit_bias",
            "logprobs",
            "top_logprobs",
        ],
    )


def _completion_sampling_params(request: CompletionRequest) -> dict[str, Any]:
    return _present_params(
        request,
        [
            "suffix",
            "stop",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "repeat_penalty",
            "top_k",
            "min_p",
            "typical_p",
            "tfs_z",
            "mirostat_mode",
            "mirostat_tau",
            "mirostat_eta",
            "logit_bias",
            "logprobs",
            "echo",
        ],
    )


def _present_params(request: Any, fields: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for field in fields:
        value = getattr(request, field)
        if value is not None:
            params[field] = value
    return params


def _response_text_format(text: dict[str, Any] | None) -> dict[str, Any] | None:
    if not text:
        return None
    response_format = text.get("format", text)
    return response_format if isinstance(response_format, dict) else None


def _video_config(settings: Settings) -> VideoExtractionConfig:
    return VideoExtractionConfig(
        max_frames=settings.video_max_frames,
        sample_fps=settings.video_sample_fps,
        max_seconds=settings.video_max_seconds,
        frame_size=settings.video_frame_size,
    )


def _responses_input_to_messages(request: ResponseRequest, settings: Settings) -> list[Any]:
    messages: list[dict[str, Any]] = []
    video_config = _video_config(settings)
    if request.instructions:
        messages.append({"role": "system", "content": request.instructions})

    if isinstance(request.input, str):
        messages.append({"role": "user", "content": request.input})
    elif isinstance(request.input, list):
        for item in request.input:
            if not isinstance(item, dict):
                raise openai_error(400, "response input items must be objects", param="input")
            item_type = item.get("type")
            if item_type == "message" or "role" in item:
                role = item.get("role", "user")
                content = item.get("content", "")
                if isinstance(content, list):
                    content = normalize_content_parts(content, video_config=video_config)
                messages.append({"role": role, "content": content})
            elif item_type in {"input_text", "input_image", "input_video", "input_audio"}:
                messages.append({"role": "user", "content": normalize_content_parts([item], video_config=video_config)})
            elif item_type == "function_call_output":
                messages.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": item.get("output", "")})
            else:
                raise openai_error(400, f"unsupported response input item type: {item_type}", param="input")
    else:
        raise openai_error(400, "input must be a string or array", param="input")

    from .schemas import ChatMessage

    return [ChatMessage(**message) for message in messages]


def _response_input_items_for_storage(request: ResponseRequest) -> list[dict[str, Any]]:
    raw_items = request.input if isinstance(request.input, list) else [{"type": "input_text", "text": request.input}]
    items: list[dict[str, Any]] = []
    if request.instructions:
        items.append({"id": f"item_{uuid.uuid4().hex}", "type": "message", "role": "system", "content": request.instructions})
    for item in raw_items:
        if isinstance(item, dict):
            stored = dict(item)
        else:
            stored = {"type": "input_text", "text": str(item)}
        stored.setdefault("id", f"item_{uuid.uuid4().hex}")
        items.append(stored)
    return items


def _response_to_messages(response: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in response.get("output", []):
        item_type = item.get("type")
        if item_type == "message":
            text = "".join(
                part.get("text", "")
                for part in item.get("content", [])
                if isinstance(part, dict) and part.get("type") == "output_text"
            )
            if text:
                messages.append({"role": item.get("role", "assistant"), "content": text})
        elif item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.get("call_id") or item.get("id"),
                            "type": "function",
                            "function": {
                                "name": item.get("name"),
                                "arguments": item.get("arguments", ""),
                            },
                        }
                    ],
                }
            )
    return messages


def _normalize_chat_response(
    result: Any,
    model_id: str,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any = "auto",
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise openai_error(500, "backend returned an invalid response", type_="server_error")
    result.setdefault("id", f"chatcmpl_{uuid.uuid4().hex}")
    result.setdefault("object", "chat.completion")
    result.setdefault("created", int(time.time()))
    result["model"] = model_id

    for choice in result.get("choices", []):
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and "tool_calls" not in message:
            tool_calls = parse_tool_calls(content, tools, tool_choice)
            if tool_calls:
                message["tool_calls"] = tool_calls
                message["content"] = remove_tool_call_markup(content) or None
                choice["finish_reason"] = "tool_calls"
        choice["message"] = message
    result.setdefault("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    return result


def _normalize_completion_response(result: Any, model_id: str) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise openai_error(500, "backend returned an invalid response", type_="server_error")
    result.setdefault("id", f"cmpl_{uuid.uuid4().hex}")
    result.setdefault("object", "text_completion")
    result.setdefault("created", int(time.time()))
    result["model"] = model_id
    result.setdefault("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    return result


def _normalize_chat_stream(
    chunks: Any,
    model_id: str,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any = "auto",
) -> Iterable[dict[str, Any]]:
    stream_id = f"chatcmpl_{uuid.uuid4().hex}"
    created = int(time.time())
    content_buffer: list[str] = []
    final_finish_reason: str | None = None

    if tools:
        pending_tool_calls: list[dict[str, Any]] = []
        for raw_chunk in chunks:
            chunk = _normalized_stream_chunk(raw_chunk, model_id, "chat.completion.chunk", stream_id, created)
            stream_id = chunk["id"]
            created = chunk["created"]
            choices = chunk.get("choices", [])
            for choice in choices:
                final_finish_reason = choice.get("finish_reason") or final_finish_reason
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content_buffer.append(delta["content"])
                pending_tool_calls.extend(delta.get("tool_calls") or [])

        content = "".join(content_buffer)
        parsed_tool_calls = parse_tool_calls(content, tools, tool_choice)
        tool_calls = pending_tool_calls or parsed_tool_calls
        visible_content = remove_tool_call_markup(content) if parsed_tool_calls else content
        if visible_content:
            yield _chat_stream_chunk(
                stream_id,
                created,
                model_id,
                {"role": "assistant", "content": visible_content},
                finish_reason=None,
            )
        elif tool_calls:
            yield _chat_stream_chunk(stream_id, created, model_id, {"role": "assistant"}, finish_reason=None)
        for index, call in enumerate(tool_calls):
            yield _chat_stream_chunk(
                stream_id,
                created,
                model_id,
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call["id"],
                            "type": call.get("type", "function"),
                            "function": {
                                "name": call.get("function", {}).get("name"),
                                "arguments": call.get("function", {}).get("arguments", ""),
                            },
                        }
                    ]
                },
                finish_reason=None,
            )
        yield _chat_stream_chunk(
            stream_id,
            created,
            model_id,
            {},
            finish_reason="tool_calls" if tool_calls else final_finish_reason or "stop",
        )
        return

    for raw_chunk in chunks:
        yield _normalized_stream_chunk(raw_chunk, model_id, "chat.completion.chunk", stream_id, created)


def _normalize_completion_stream(chunks: Any, model_id: str) -> Iterable[dict[str, Any]]:
    stream_id = f"cmpl_{uuid.uuid4().hex}"
    created = int(time.time())
    for raw_chunk in chunks:
        yield _normalized_stream_chunk(raw_chunk, model_id, "text_completion", stream_id, created)


def _normalized_stream_chunk(
    raw_chunk: Any,
    model_id: str,
    object_name: str,
    fallback_id: str,
    fallback_created: int,
) -> dict[str, Any]:
    if not isinstance(raw_chunk, dict):
        raw_chunk = {"choices": [{"index": 0, "delta": {"content": str(raw_chunk)}, "finish_reason": None}]}
    chunk = dict(raw_chunk)
    chunk["id"] = chunk.get("id") or fallback_id
    chunk["object"] = object_name
    chunk["created"] = chunk.get("created") or fallback_created
    chunk["model"] = model_id
    chunk["choices"] = [_normalize_stream_choice(choice) for choice in chunk.get("choices", [])]
    return chunk


def _normalize_stream_choice(choice: Any) -> dict[str, Any]:
    if not isinstance(choice, dict):
        choice = {"index": 0, "delta": {"content": str(choice)}, "finish_reason": None}
    normalized = dict(choice)
    if "delta" not in normalized and "text" in normalized:
        normalized["delta"] = {"content": normalized.pop("text")}
    delta = normalized.get("delta")
    if isinstance(delta, dict):
        normalized["delta"] = _normalize_stream_delta(delta)
    else:
        normalized["delta"] = {}
    normalized.setdefault("index", 0)
    normalized.setdefault("finish_reason", None)
    return normalized


def _normalize_stream_delta(delta: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(delta)
    if "message" in normalized and isinstance(normalized["message"], dict):
        normalized.update(normalized.pop("message"))
    if "content" in normalized and normalized["content"] is None:
        normalized.pop("content")
    if "tool_calls" in normalized:
        normalized["tool_calls"] = [
            _normalize_stream_tool_call(index, call)
            for index, call in enumerate(normalized.get("tool_calls") or [])
            if isinstance(call, dict)
        ]
    return normalized


def _normalize_stream_tool_call(index: int, call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    return {
        "index": call.get("index", index),
        "id": call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
        "type": call.get("type", "function"),
        "function": {
            "name": function.get("name"),
            "arguments": function.get("arguments", ""),
        },
    }


def _chat_stream_chunk(
    stream_id: str,
    created: int,
    model_id: str,
    delta: dict[str, Any],
    *,
    finish_reason: str | None,
) -> dict[str, Any]:
    return {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _chat_to_response(chat_response: dict[str, Any], request: ResponseRequest, model_id: str) -> dict[str, Any]:
    choice = chat_response["choices"][0]
    message = choice["message"]
    output: list[dict[str, Any]] = []

    for call in message.get("tool_calls") or []:
        output.append(
            {
                "type": "function_call",
                "id": call["id"],
                "call_id": call["id"],
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
                "status": "completed",
            }
        )

    if message.get("content"):
        output.append(
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": message["content"], "annotations": []}],
            }
        )

    output_text = "".join(
        part["text"]
        for item in output
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )
    now = int(time.time())
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": now,
        "status": "completed",
        "model": model_id,
        "output": output,
        "output_text": output_text,
        "parallel_tool_calls": True,
        "previous_response_id": request.previous_response_id,
        "store": request.store,
        "temperature": request.temperature,
        "tool_choice": request.tool_choice,
        "tools": request.tools or [],
        "top_p": request.top_p,
        "usage": {
            "input_tokens": chat_response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": chat_response.get("usage", {}).get("completion_tokens", 0),
            "total_tokens": chat_response.get("usage", {}).get("total_tokens", 0),
        },
    }


def _normalize_embedding_inputs(value: Any) -> list[str]:
    if isinstance(value, str):
        if not value:
            raise openai_error(400, "input cannot be an empty string", param="input")
        return [value]
    if not isinstance(value, list) or not value:
        raise openai_error(400, "input must be a string or non-empty array", param="input")
    if all(isinstance(item, str) for item in value):
        if any(item == "" for item in value):
            raise openai_error(400, "input array cannot contain empty strings", param="input")
        return list(value)
    if all(isinstance(item, int) for item in value):
        return [" ".join(str(item) for item in value)]
    if all(isinstance(item, list) and all(isinstance(token, int) for token in item) for item in value):
        return [" ".join(str(token) for token in item) for item in value]
    raise openai_error(400, "input must be a string, array of strings, token array, or array of token arrays", param="input")


def _sse(chunks: Any) -> StreamingResponse:
    def events() -> Iterable[str]:
        for chunk in chunks:
            yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def _responses_stream(chunks: Any, model_id: str) -> Iterable[dict[str, Any]]:
    response_id = f"resp_{uuid.uuid4().hex}"
    output: list[dict[str, Any]] = []
    text_item_id: str | None = None
    tool_items: dict[int, dict[str, Any]] = {}
    yield {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "model": model_id,
            "output": [],
        },
    }
    for chunk in chunks:
        choices = chunk.get("choices", []) if isinstance(chunk, dict) else []
        for choice in choices:
            delta = choice.get("delta", {})
            content_delta = delta.get("content") or choice.get("text", "")
            if content_delta:
                if text_item_id is None:
                    text_item_id = f"msg_{uuid.uuid4().hex}"
                    output.append(
                        {
                            "type": "message",
                            "id": text_item_id,
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "", "annotations": []}],
                        }
                    )
                    yield {
                        "type": "response.output_item.added",
                        "response_id": response_id,
                        "output_index": len(output) - 1,
                        "item": output[-1],
                    }
                output[-1]["content"][0]["text"] += content_delta
                yield {"type": "response.output_text.delta", "response_id": response_id, "delta": content_delta}

            for tool_call in delta.get("tool_calls") or []:
                index = int(tool_call.get("index", len(tool_items)))
                function = tool_call.get("function", {})
                if index not in tool_items:
                    call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}"
                    item = {
                        "type": "function_call",
                        "id": call_id,
                        "call_id": call_id,
                        "name": function.get("name") or "",
                        "arguments": "",
                        "status": "in_progress",
                    }
                    tool_items[index] = item
                    output.append(item)
                    yield {
                        "type": "response.output_item.added",
                        "response_id": response_id,
                        "output_index": len(output) - 1,
                        "item": item,
                    }
                item = tool_items[index]
                if function.get("name"):
                    item["name"] = function["name"]
                arguments_delta = function.get("arguments") or ""
                if arguments_delta:
                    item["arguments"] += arguments_delta
                    yield {
                        "type": "response.function_call_arguments.delta",
                        "response_id": response_id,
                        "item_id": item["id"],
                        "output_index": output.index(item),
                        "delta": arguments_delta,
                    }

            if choice.get("finish_reason") == "tool_calls":
                for item in tool_items.values():
                    item["status"] = "completed"
                    yield {
                        "type": "response.output_item.done",
                        "response_id": response_id,
                        "output_index": output.index(item),
                        "item": item,
                    }
            elif choice.get("finish_reason") == "stop" and text_item_id and output:
                output[-1]["status"] = "completed"
                yield {
                    "type": "response.output_item.done",
                    "response_id": response_id,
                    "output_index": len(output) - 1,
                    "item": output[-1],
                }
    for item in output:
        item["status"] = "completed"
    yield {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": model_id,
            "output": output,
        },
    }
