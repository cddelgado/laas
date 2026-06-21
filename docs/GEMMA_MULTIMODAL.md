# Gemma Multimodal Support Matrix

LAAS accepts OpenAI-shaped multimodal requests, but local behavior is limited by
the installed `llama-cpp-python` chat handler and the configured Gemma projector.
This page documents what is accepted, transformed, and live-smoked.

## Current Backend Finding

The installed `llama-cpp-python` package exposes `Gemma4ChatHandler` through the
MTMD projector path. Its handler recognizes `image_url` content parts. It does
not recognize `input_audio` or native video parts. LAAS therefore treats video
as an image-frame translation layer and rejects LLM-native audio input by
default.

## Matrix

| Input or output | OpenAI-style shape | LAAS behavior | Backend path | Default capability |
| --- | --- | --- | --- | --- |
| Text input | `{"type":"text","text":"..."}` | Forwarded as text. | Gemma chat template. | `text=true` |
| Image input | `{"type":"image_url","image_url":{"url":"..."}}` | Normalized and forwarded. | Gemma MTMD/projector image path. | `vision=true` when `LAAS_MMPROJ_REQUIRED=true` and a projector is configured. |
| Video input | `{"type":"input_video", ...}` | Extracted into bounded JPEG image frames, then forwarded as `image_url` parts. | Gemma MTMD/projector image path after LAAS frame extraction. | `video=true` when image/projector support is configured. |
| Audio input to LLM | `{"type":"input_audio","input_audio":{"data":"...","format":"wav"}}` | Rejected by default with an OpenAI-shaped capability error. | Not forwarded unless `LAAS_LLM_AUDIO_INPUT_ENABLED=true`. | `audio_input=false` |
| Speech-to-text | `/v1/audio/transcriptions` | Accepted as explicit STT. | Whisper.cpp-compatible local stack. | Separate Audio API, not LLM context input. |
| Text-to-speech | `/v1/audio/speech` | Accepted as explicit TTS. | Kokoro local stack. | Separate Audio API, not native LLM audio output. |
| Audio output from Chat Completions | `modalities:["audio"]` | Rejected; use `/v1/audio/speech`. | Not produced by Gemma text backend. | `audio_output=false` |

## Live Audit

Start LAAS:

```powershell
laas
```

Run the multimodal audit:

```powershell
.\.venv\Scripts\python.exe scripts\multimodal_fidelity_smoke.py --base-url http://127.0.0.1:8000
```

The smoke:

- Sends a generated PNG through `image_url`.
- Generates a tiny MP4, sends it through `input_video`, and verifies LAAS can
  extract and forward frames.
- Sends OpenAI-style `input_audio` and expects the default unsupported-capability
  response.

To experiment with a future backend that supports native LLM audio input, set
`LAAS_LLM_AUDIO_INPUT_ENABLED=true` and run:

```powershell
.\.venv\Scripts\python.exe scripts\multimodal_fidelity_smoke.py --expect-audio-input-supported
```

Do not enable that setting for normal use unless the local backend has been
verified to consume audio content parts correctly. Whisper transcription remains
the supported speech-to-text path.
