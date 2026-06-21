# laas

The Local AI API Stack is a Python OpenAI-compatible API host for local GGUF
models. The initial target is Gemma 4 E4B Instruct with the Q4_K_M GGUF quant,
loaded from `ggml-org/gemma-4-E4B-it-GGUF`.

## What is implemented

- `GET /v1/models`
- `GET /v1/models/{model_id}`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/responses`
- `GET /v1/responses/{response_id}`
- `DELETE /v1/responses/{response_id}`
- `GET /v1/responses/{response_id}/input_items`
- `POST /v1/embeddings`
- `POST /v1/audio/speech`
- `GET /v1/local/settings`
- `PATCH /v1/local/settings`
- `GET /v1/local/models/status`
- `POST /v1/local/models/download`
- `POST /v1/local/models/load`
- `POST /v1/local/models/unload`
- `GET /v1/local/audio/status`
- `GET /v1/local/audio/voices`
- `POST /v1/local/audio/download`
- `POST /v1/local/audio/load`
- `POST /v1/local/audio/unload`
- `GET /v1/local/transcription/status`
- `POST /v1/local/transcription/download`
- `POST /v1/local/transcription/load`
- `POST /v1/local/transcription/unload`
- `GET /v1/local/voice/status`
- `POST /v1/local/voice/download`
- `POST /v1/local/voice/load`
- `POST /v1/local/voice/unload`
- `POST /v1/local/voice/sessions`
- `GET /v1/local/voice/sessions/{session_id}`
- `DELETE /v1/local/voice/sessions/{session_id}`
- `POST /v1/local/voice/sessions/{session_id}/turns`
- `WS /v1/local/voice/sessions/{session_id}/realtime`
- `GET /v1/local/capabilities`

The OpenAI-compatible endpoints accept OpenAI-style text, tool calls, image
parts, and Responses API inputs. Gemma video input is translated to sampled
image frames. If the request already supplies `frames`, LAAS uses them directly;
otherwise install the optional video extra so OpenCV can extract frames.

## Install

Detailed environment, wheel, and install-order guidance is in
[docs/INSTALL.md](docs/INSTALL.md).

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Then install the `llama-cpp-python` wheel that matches your hardware. CPU-only:

```bash
python -m pip install -r requirements-llama-cpu.txt
```

NVIDIA CUDA example for PowerShell:

```powershell
python -m pip install llama-cpp-python `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

NVIDIA CUDA example for macOS/Linux shells:

```bash
python -m pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

ROCm and Vulkan options are documented in [docs/INSTALL.md](docs/INSTALL.md).
Use the wheel index that matches your machine. The upstream `llama-cpp-python`
project documents the current pre-built wheel indexes at
<https://github.com/abetlen/llama-cpp-python#supported-backends>.

For the full local voice stack, Kokoro TTS plus whisper.cpp STT:

```bash
python -m pip install -r requirements-voice.txt
```

On Windows PowerShell:

```powershell
python -m pip install -r requirements-voice.txt
```

The equivalent `pyproject.toml` extra is `python -m pip install -e ".[voice]"`.

## Configure

By default, LAAS downloads models to:

```text
Windows: D:\AI\Models
macOS/Linux: ~/AI/Models
```

Override with `.env`, environment variables, or the local settings endpoint:

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

```http
PATCH /v1/local/settings
{
  "model_dir": "D:\\AI\\Models"
}
```

The default Gemma model settings are:

```text
LAAS_MODEL_ID=gemma-4-e4b-it-q4_k_m
LAAS_HF_REPO_ID=ggml-org/gemma-4-E4B-it-GGUF
LAAS_HF_FILENAME=gemma-4-E4B-it-Q4_K_M.gguf
LAAS_MMPROJ_FILENAME=mmproj-gemma-4-E4B-it-Q8_0.gguf
LAAS_MMPROJ_REQUIRED=true
LAAS_AUTO_DOWNLOAD=false
```

The default Kokoro TTS settings are:

```text
LAAS_TTS_MODEL_ID=kokoro-82m
LAAS_TTS_HF_REPO_ID=fastrtc/kokoro-onnx
LAAS_TTS_MODEL_FILENAME=kokoro-v1.0.onnx
LAAS_TTS_VOICES_FILENAME=voices-v1.0.bin
LAAS_TTS_DEFAULT_VOICE=af_heart
LAAS_TTS_DEFAULT_LANG=en-us
LAAS_TTS_AUTO_DOWNLOAD=false
LAAS_TTS_FFMPEG_PATH=ffmpeg
LAAS_STT_MODEL_ID=whisper-small
LAAS_STT_HF_REPO_ID=ggerganov/whisper.cpp
LAAS_STT_MODEL_FILENAME=ggml-small.bin
LAAS_STT_AUTO_DOWNLOAD=false
LAAS_VOICE_AUTO_LOAD=false
LAAS_VOICE_AUTO_DOWNLOAD=false
LAAS_EMBEDDING_MODEL_ID=laas-hash-embedding
LAAS_EMBEDDING_DIMENSIONS=384
```

## Run

Windows PowerShell:

```powershell
laas
```

The command is `laas`: lowercase `l-a-a-s`. It is not `lass`.

or, without relying on `PATH`:

```powershell
.\.venv\Scripts\laas.exe
```

or, through the active Python interpreter:

```powershell
python -m laas.main
```

Show CLI options:

```powershell
laas --help
```

Do not run `lass`, which is a misspelling. Do not run `.\laas` unless you have
created a `laas` file in the repository root. In PowerShell, `.\laas` means
"run a local file named `laas`"; it does not look up the installed
`.venv\Scripts\laas.exe` console command.

macOS/Linux:

```bash
laas
```

The command is `laas`: lowercase `l-a-a-s`. It is not `lass`.

Direct uvicorn alternative for any platform:

```powershell
python -m uvicorn laas.app:app --host 127.0.0.1 --port 8000
```

When launched through `laas`, startup checks the configured model path before
the server starts. If the model or projector file is missing, LAAS prints the
model id, Hugging Face repo, filenames, and target paths, then asks whether to
download the missing assets.

To confirm from the prompt, answer `y` or `yes`.

To download without prompting:

```powershell
laas --yes-download
```

To skip the startup prompt:

```powershell
laas --no-download-prompt
```

Direct `uvicorn` launches do not ask interactive questions. They are intended
for service/process-manager use.

After the server starts, use another terminal to check whether the model is
present:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/local/models/status
```

Manual download and load. The download endpoint fetches the configured main GGUF
and projector by default:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/models/download `
  -ContentType "application/json" `
  -Body "{}"

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/models/load `
  -ContentType "application/json" `
  -Body "{}"
```

macOS/Linux:

```bash
curl -X POST http://127.0.0.1:8000/v1/local/models/download \
  -H "Content-Type: application/json" \
  -d "{}"

curl -X POST http://127.0.0.1:8000/v1/local/models/load \
  -H "Content-Type: application/json" \
  -d "{}"
```

If you skip the explicit download step, `POST /v1/local/models/load` downloads
the configured model when the file is missing. Inference requests do not trigger
a model download by default. If the model is missing, inference returns
`model_not_downloaded` with instructions to call the local download/load
endpoints.

Set `LAAS_AUTO_DOWNLOAD=true` only if you want LAAS to download a missing model
during auto-load or first inference. With the default `LAAS_AUTO_DOWNLOAD=false`,
downloads happen only after an explicit local download/load request.

Gemma 4 multimodal requests require the projector. The default Q4 main model is
paired with the repo's Q8 projector because the repo does not publish a Q4
projector. For text-only experiments, set `LAAS_MMPROJ_REQUIRED=false`.

Unload it when you are done:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/models/unload
```

macOS/Linux:

```bash
curl -X POST http://127.0.0.1:8000/v1/local/models/unload
```

LAAS also unloads the active model after `LAAS_IDLE_UNLOAD_SECONDS` seconds of
inactivity. Set it to `0` to disable idle unloading.

## Local Voice Stack

Install the voice extra, start `laas`, then download and load Kokoro plus
Whisper small together:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/voice/download `
  -ContentType "application/json" `
  -Body "{}"

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/voice/load `
  -ContentType "application/json" `
  -Body "{}"
```

Generate WAV output:

```powershell
$body = @{
  model = "tts-1"
  input = "Hello from local Kokoro."
  voice = "af_heart"
  response_format = "wav"
  speed = 1.0
} | ConvertTo-Json

Invoke-WebRequest -Method Post -Uri http://127.0.0.1:8000/v1/audio/speech `
  -ContentType "application/json" `
  -Body $body `
  -OutFile .\kokoro.wav
```

macOS/Linux:

```bash
curl -X POST http://127.0.0.1:8000/v1/local/voice/download \
  -H "Content-Type: application/json" \
  -d "{}"

curl -X POST http://127.0.0.1:8000/v1/local/voice/load \
  -H "Content-Type: application/json" \
  -d "{}"

curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello from local Kokoro.","voice":"af_heart","response_format":"wav"}' \
  --output kokoro.wav
```

`/v1/audio/speech` accepts OpenAI voice aliases where practical, for example
`alloy` maps to Kokoro's `af_alloy`. Direct Kokoro voice ids are also accepted.
The `mp3`, `wav`, `flac`, and raw signed 16-bit little endian `pcm` formats are
encoded in-process. `opus` and `aac` are encoded with FFmpeg when it is
available.

Install FFmpeg to enable OpenAI-compatible `opus` and `aac` output:

```powershell
winget install Gyan.FFmpeg
```

macOS/Linux:

```bash
brew install ffmpeg
# or: sudo apt install ffmpeg
```

If `ffmpeg` is not on `PATH`, set `LAAS_TTS_FFMPEG_PATH` to the executable. The
audio status endpoint reports `supported_formats`, `ffmpeg_path`, and
`ffmpeg_available` so clients can decide which formats to request.

Transcribe the generated file with the OpenAI-compatible transcription endpoint.
This Python example works on Windows, macOS, and Linux:

```python
from pathlib import Path

import requests

audio_path = Path("kokoro.wav")
with audio_path.open("rb") as fh:
    response = requests.post(
        "http://127.0.0.1:8000/v1/audio/transcriptions",
        data={
            "model": "whisper-1",
            "response_format": "verbose_json",
            "language": "en",
        },
        files={"file": (audio_path.name, fh, "audio/wav")},
    )
response.raise_for_status()
print(response.json()["text"])
```

`POST /v1/audio/translations` accepts the same multipart upload shape and asks
Whisper to translate speech to English. Supported transcription response formats
are `json`, `text`, `srt`, `verbose_json`, and `vtt`.
For `verbose_json` transcription requests, `timestamp_granularities[]=segment`
is supported. `timestamp_granularities[]=word` returns an explicit compatibility
error because the whisper.cpp backend used here does not expose word timestamps.

For a full local voice turn, create a voice session and send audio to the turn
endpoint. LAAS transcribes the audio with Whisper, sends the transcript to the
loaded text model, synthesizes the assistant response with Kokoro, and returns
the audio as base64 in the requested format:

```python
import base64
from pathlib import Path

import requests

base_url = "http://127.0.0.1:8000"

session = requests.post(
    f"{base_url}/v1/local/voice/sessions",
    json={
        "instructions": "Answer briefly.",
        "voice": "alloy",
        "response_format": "wav",
    },
)
session.raise_for_status()
session_id = session.json()["id"]

audio_path = Path("question.wav")
with audio_path.open("rb") as fh:
    turn = requests.post(
        f"{base_url}/v1/local/voice/sessions/{session_id}/turns",
        files={"file": (audio_path.name, fh, "audio/wav")},
    )
turn.raise_for_status()
payload = turn.json()

print("You said:", payload["transcript"]["text"])
print("Assistant:", payload["response"]["text"])
Path("answer.wav").write_bytes(base64.b64decode(payload["audio"]["data"]))

requests.delete(f"{base_url}/v1/local/voice/sessions/{session_id}").raise_for_status()
```

The realtime WebSocket transport is available at
`/v1/local/voice/sessions/{session_id}/realtime`. After connecting, clients can
send one of two audio shapes:

```json
{"type":"input_audio_buffer.append","audio":"<base64 audio bytes>"}
{"type":"input_audio_buffer.commit","filename":"question.wav"}
```

or a one-shot turn:

```json
{"type":"voice.turn","audio":"<base64 audio bytes>","filename":"question.wav"}
```

The server replies with `response.completed` containing the same turn payload as
the HTTP endpoint. It also accepts `input_audio_buffer.clear`,
`response.cancel`, and `session.close` control events.

Unload the full voice stack when you are done:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/voice/unload
```

Kokoro unloads after `LAAS_TTS_IDLE_UNLOAD_SECONDS` seconds of inactivity.
Whisper unloads after `LAAS_STT_IDLE_UNLOAD_SECONDS` seconds. Set either to `0`
to disable idle unloading for that side of the voice stack.

## Use With OpenAI Clients

Point an OpenAI client at the local API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")

response = client.chat.completions.create(
    model="gemma-4-e4b-it-q4_k_m",
    messages=[{"role": "user", "content": "Say hello from Gemma."}],
)
print(response.choices[0].message.content)
```

Responses are stored in memory by default, so clients can retrieve them or chain
local context with `previous_response_id`:

```python
first = client.responses.create(model="gemma-4-e4b-it-q4_k_m", input="One sentence.")
second = client.responses.create(
    model="gemma-4-e4b-it-q4_k_m",
    previous_response_id=first.id,
    input="Now add one more.",
)
```

Set `store=False` for one-off responses that should not be retrievable. Stored
responses are process-local and disappear when the server restarts.

The local embeddings endpoint exposes `laas-hash-embedding`. It is deterministic
and OpenAI-shape-compatible for local development, but it is not a semantic
embedding model:

```python
embedding = client.embeddings.create(
    model="laas-hash-embedding",
    input=["alpha", "beta"],
    dimensions=128,
)
print(len(embedding.data[0].embedding))
```

## Notes

Gemma 4 E4B is exposed as a text-output model with text, tool-call, image,
video-as-frames, audio-input, and reasoning capabilities. LAAS validates request
capabilities before sending prompts to the backend and preserves OpenAI response
shapes where practical for local inference.
