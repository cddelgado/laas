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
- `GET /v1/local/settings`
- `PATCH /v1/local/settings`
- `GET /v1/local/models/status`
- `POST /v1/local/models/download`
- `POST /v1/local/models/load`
- `POST /v1/local/models/unload`
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
```

## Run

Windows PowerShell:

```powershell
laas
```

or, without relying on `PATH`:

```powershell
.\.venv\Scripts\laas.exe
```

Show CLI options:

```powershell
laas --help
```

Do not run `.\laas` unless you have created a `laas` file in the repository
root. In PowerShell, `.\laas` means "run a local file named `laas`"; it does not
look up the installed `.venv\Scripts\laas.exe` console command.

macOS/Linux:

```bash
laas
```

Direct uvicorn alternative for any platform:

```powershell
python -m uvicorn laas.app:app --host 127.0.0.1 --port 8000
```

Download and load the model:

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
the configured model when the file is missing. If `LAAS_AUTO_LOAD=true`, the
same download-then-load path runs when the server starts. With the default
`LAAS_AUTO_LOAD=false`, server startup does not download a model.

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
