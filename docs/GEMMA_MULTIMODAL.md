# Gemma Multimodal Support Matrix

LAAS accepts OpenAI-shaped multimodal requests, but local behavior is limited by
the installed `llama-cpp-python` chat handler and the configured Gemma model.
This page documents what is accepted, transformed, and live-smoked.

## Current Backend Finding

The default Gemma 4 E4B GGUF requires the configured projector file,
`mmproj-gemma-4-E4B-it-Q8_0.gguf`, for image and video-frame inputs. LAAS still
treats video as an image-frame translation layer because the OpenAI API shape
and local llama.cpp handlers are more reliable with bounded image frames than
arbitrary video blobs.

## Matrix

| Input or output | OpenAI-style shape | LAAS behavior | Backend path | Default capability |
| --- | --- | --- | --- | --- |
| Text input | `{"type":"text","text":"..."}` | Forwarded as text. | Gemma chat template. | `text=true` |
| Image input | `{"type":"image_url","image_url":{"url":"..."}}` | Normalized and forwarded. | Gemma 4 projector-backed chat path. | `vision=true` |
| Video input | `{"type":"input_video", ...}` | Extracted into bounded JPEG image frames, then forwarded as `image_url` parts. | Gemma 4 projector-backed image path after LAAS frame extraction. | `video=true` |
| Audio input to LLM | `{"type":"input_audio","input_audio":{"data":"...","format":"wav"}}` | Rejected by default; use explicit STT unless a verified backend supports direct LLM audio input. | Not enabled for the default E4B profile. | `audio_input=false` |
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
