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

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[llama,video,dev]"
```

If `llama-cpp-python` needs a hardware-specific build, install the correct wheel
for your CUDA/Metal/CPU setup before running LAAS.

## Configure

By default, LAAS downloads models to:

```text
D:\AI\Models
```

Override with `.env`, environment variables, or the local settings endpoint:

```powershell
Copy-Item .env.example .env
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

```powershell
laas
```

or:

```powershell
uvicorn laas.app:app --host 127.0.0.1 --port 8000
```

Download and load the model:

```powershell
curl -X POST http://127.0.0.1:8000/v1/local/models/download `
  -H "Content-Type: application/json" `
  -d "{}"

curl -X POST http://127.0.0.1:8000/v1/local/models/load `
  -H "Content-Type: application/json" `
  -d "{}"
```

Unload it when you are done:

```powershell
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
