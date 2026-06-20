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

For Kokoro text-to-speech:

```bash
python -m pip install -r requirements-tts.txt
```

On Windows PowerShell:

```powershell
python -m pip install -r requirements-tts.txt
```

The equivalent `pyproject.toml` extra is `python -m pip install -e ".[tts]"`.

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

## Kokoro Text-to-Speech

Install the TTS extra, start `laas`, then download and load Kokoro:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/audio/download `
  -ContentType "application/json" `
  -Body "{}"

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/audio/load `
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
curl -X POST http://127.0.0.1:8000/v1/local/audio/download \
  -H "Content-Type: application/json" \
  -d "{}"

curl -X POST http://127.0.0.1:8000/v1/local/audio/load \
  -H "Content-Type: application/json" \
  -d "{}"

curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello from local Kokoro.","voice":"af_heart","response_format":"wav"}' \
  --output kokoro.wav
```

`/v1/audio/speech` accepts OpenAI voice aliases where practical, for example
`alloy` maps to Kokoro's `af_alloy`. Direct Kokoro voice ids are also accepted.
Supported output formats are `mp3`, `wav`, `flac`, and raw signed 16-bit little
endian `pcm`; `opus` and `aac` currently return a validation error.

Unload Kokoro when you are done:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/audio/unload
```

Kokoro also unloads after `LAAS_TTS_IDLE_UNLOAD_SECONDS` seconds of inactivity.
Set it to `0` to disable TTS idle unloading.

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

## Notes

Gemma 4 E4B is exposed as a text-output model with text, tool-call, image,
video-as-frames, audio-input, and reasoning capabilities. LAAS validates request
capabilities before sending prompts to the backend and preserves OpenAI response
shapes where practical for local inference.
