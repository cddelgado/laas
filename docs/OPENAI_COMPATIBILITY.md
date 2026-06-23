# OpenAI Compatibility Matrix

LAAS is an OpenAI-compatible local inference host. It implements the endpoints
that can run against local model files and returns explicit unsupported errors
for selected cloud/account APIs.

Reference scope: OpenAI's API reference groups current APIs into inference
surfaces such as Responses, Chat Completions, Images, Audio, Embeddings, Files,
Vector Stores, and Models, plus hosted/cloud surfaces such as Uploads, Batches,
Fine-tuning, Realtime, Containers, Skills, and Administration.

## Supported

| Surface | LAAS endpoints | Notes |
| --- | --- | --- |
| Models | `GET /v1/models`, `GET /v1/models/{model_id}` | Lists configured local text, embedding, image, image edit, and video generation models. |
| Chat Completions | `POST /v1/chat/completions` | Local Gemma chat with streaming, multimodal content normalization, bounded video-to-frame translation, and Gemma tool-call translation. OpenAI-shaped `input_audio` is rejected by default unless explicitly enabled for a backend that proves native support. |
| Completions | `POST /v1/completions` | Legacy text completion compatibility over the local llama.cpp backend. |
| Responses | `POST /v1/responses`, `GET /v1/responses/{id}`, `DELETE /v1/responses/{id}`, `GET /v1/responses/{id}/input_items` | Local in-memory response storage with text and function-call output normalization. |
| Embeddings | `POST /v1/embeddings` | Local Sentence Transformers backend, defaulting to `bge-small-en-v1.5`. |
| Files | `POST /v1/files`, `GET /v1/files`, `GET /v1/files/{id}`, `GET /v1/files/{id}/content`, `DELETE /v1/files/{id}` | Local file bytes on disk with SQLite metadata. |
| Vector Stores | `POST /v1/vector_stores`, `GET /v1/vector_stores`, `GET /v1/vector_stores/{id}`, `DELETE /v1/vector_stores/{id}`, `POST/GET /v1/vector_stores/{id}/files`, `GET/DELETE /v1/vector_stores/{id}/files/{file_id}`, `POST /v1/local/vector_stores/{id}/search`, `GET /v1/local/vector_stores/{id}/indexing/status` | Local file indexing and cosine search over embedding chunks. Chat Completions and Responses can use `tools: [{"type":"file_search","vector_store_ids":[...]}]`; Responses include file citation annotations. |
| Batches | `POST /v1/batches`, `GET /v1/batches`, `GET /v1/batches/{id}`, `POST /v1/batches/{id}/cancel` | SQLite-persisted local JSONL batch runner, currently for `/v1/embeddings`. |
| Moderations | `POST /v1/moderations` | Deterministic local rule-backed compatibility endpoint. |
| Local Jobs | `GET /v1/local/jobs`, `GET /v1/local/jobs/{id}` | Local status records for async vector indexing and batch work. |
| Local Storage Maintenance | `GET /v1/local/storage/status`, `POST /v1/local/storage/prune`, `POST /v1/local/storage/vacuum` | SQLite/file-storage usage, 180-day unused prune policy, and database vacuum. |
| Images | `POST /v1/images/generations`, `POST /v1/images/variations`, `POST /v1/images/edits` | Local Diffusers generation, variation, and inpainting/edit compatibility. |
| Video Generation | `POST /v1/videos/generations` | Local OpenAI-shaped image-to-video surface for the configured Wan2.2 I2V Q3 assets using a native Diffusers runner. Downloads GGUF HighNoise/LowNoise transformers plus required Diffusers-side tokenizer, text encoder, scheduler, VAE, and configs. |
| Audio | `POST /v1/audio/speech`, `POST /v1/audio/transcriptions`, `POST /v1/audio/translations` | Local Kokoro TTS and whisper.cpp-compatible STT. |
| Local Voice Realtime | `POST /v1/realtime/sessions`, `WS /v1/realtime/sessions/{session_id}`, `WS /v1/local/voice/sessions/{session_id}/realtime` | OpenAI-shaped local realtime wrapper plus the stable LAAS local transport over Kokoro, Whisper, and Gemma. Supports text `conversation.item.create/retrieve/delete/truncate`, session config round-tripping, backend text stream deltas, built-in PCM/WAV server VAD, and chunked response events. See [REALTIME.md](REALTIME.md). |

## Multimodal Notes

Chat Completions audio input follows OpenAI's `input_audio` content part shape:
base64 `data` plus `format` of `wav` or `mp3`, but LAAS rejects it by default
because the current local Gemma/llama.cpp MTMD handler only proves image input
support. LAAS does not silently fall back to Whisper transcription.

Video input is a LAAS/Gemma compatibility extension. `input_video` accepts
explicit image `frames`, local paths, HTTP(S) URLs, data URLs, or inline base64
video data. LAAS extracts a bounded, deterministic set of JPEG frames controlled
by `video_max_frames`, `video_sample_fps`, `video_max_seconds`, and
`video_frame_size`.

Native Chat Completions audio output through `modalities: ["audio"]` is
explicitly rejected because the local Gemma text backend does not produce
OpenAI-style audio response objects. Use `POST /v1/audio/speech` for Kokoro TTS.

See [GEMMA_MULTIMODAL.md](GEMMA_MULTIMODAL.md) for the detailed support matrix
and live fidelity smoke.

## Unsupported But Registered

These routes return OpenAI-shaped `501` errors with
`code: "unsupported_endpoint"` so generic clients get a predictable response.

| Surface | Registered routes |
| --- | --- |
| Uploads | `POST /v1/uploads`, `GET/POST/DELETE /v1/uploads/{upload_id}` |
| Fine-tuning | `GET/POST /v1/fine_tuning/jobs`, `GET /v1/fine_tuning/jobs/{job_id}`, `POST /v1/fine_tuning/jobs/{job_id}/cancel` |

## Not Applicable

LAAS does not register Administration, Containers, Skills, ChatKit, or hosted
OpenAI Realtime cloud APIs such as ephemeral hosted sessions or WebRTC relay
resources. Those surfaces require OpenAI-hosted account, organization, session,
or cloud execution resources. LAAS does provide local realtime WebSocket routes
documented in [REALTIME.md](REALTIME.md).

## Inspect Programmatically

Use:

```bash
curl http://127.0.0.1:8000/v1/local/compatibility
```

or PowerShell:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/local/compatibility
```
