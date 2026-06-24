# laas

The Local AI API Stack is a Python OpenAI-compatible API host for local GGUF
models. The default target is Gemma 4 E4B Instruct with the Q4_K_M GGUF quant
and the matching Q8 projector, loaded from `ggml-org/gemma-4-E4B-it-GGUF`.

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
- `POST /v1/files`
- `GET /v1/files`
- `GET /v1/files/{file_id}`
- `GET /v1/files/{file_id}/content`
- `DELETE /v1/files/{file_id}`
- `POST /v1/vector_stores`
- `GET /v1/vector_stores`
- `GET /v1/vector_stores/{vector_store_id}`
- `DELETE /v1/vector_stores/{vector_store_id}`
- `POST /v1/vector_stores/{vector_store_id}/files`
- `GET /v1/vector_stores/{vector_store_id}/files`
- `GET /v1/vector_stores/{vector_store_id}/files/{file_id}`
- `DELETE /v1/vector_stores/{vector_store_id}/files/{file_id}`
- `POST /v1/batches`
- `GET /v1/batches`
- `GET /v1/batches/{batch_id}`
- `POST /v1/batches/{batch_id}/cancel`
- `POST /v1/moderations`
- `POST /v1/images/generations`
- `POST /v1/images/variations`
- `POST /v1/images/edits`
- `POST /v1/videos/generations`
- `POST /v1/audio/speech`
- `POST /v1/realtime/sessions`
- `GET /v1/local/settings`
- `PATCH /v1/local/settings`
- `GET /v1/local/diagnostics`
- `GET /v1/local/compatibility`
- `GET /v1/local/models/status`
- `POST /v1/local/models/download`
- `POST /v1/local/models/load`
- `POST /v1/local/models/unload`
- `GET /v1/local/images/status`
- `POST /v1/local/images/download`
- `POST /v1/local/images/load`
- `POST /v1/local/images/unload`
- `GET /v1/local/images/edit/status`
- `POST /v1/local/images/edit/download`
- `POST /v1/local/images/edit/load`
- `POST /v1/local/images/edit/unload`
- `GET /v1/local/videos/status`
- `POST /v1/local/videos/download`
- `POST /v1/local/videos/load`
- `POST /v1/local/videos/unload`
- `GET /v1/local/audio/status`
- `GET /v1/local/audio/voices`
- `POST /v1/local/audio/download`
- `POST /v1/local/audio/load`
- `POST /v1/local/audio/unload`
- `GET /v1/local/transcription/status`
- `POST /v1/local/transcription/download`
- `POST /v1/local/transcription/load`
- `POST /v1/local/transcription/unload`
- `GET /v1/local/embeddings/status`
- `POST /v1/local/embeddings/download`
- `POST /v1/local/embeddings/load`
- `POST /v1/local/embeddings/unload`
- `GET /v1/local/files/status`
- `GET /v1/local/storage/status`
- `POST /v1/local/storage/prune`
- `POST /v1/local/storage/vacuum`
- `POST /v1/local/vector_stores/{vector_store_id}/search`
- `GET /v1/local/vector_stores/{vector_store_id}/indexing/status`
- `GET /v1/local/jobs`
- `GET /v1/local/jobs/{job_id}`
- `GET /v1/local/voice/status`
- `POST /v1/local/voice/download`
- `POST /v1/local/voice/load`
- `POST /v1/local/voice/unload`
- `POST /v1/local/voice/sessions`
- `GET /v1/local/voice/sessions/{session_id}`
- `DELETE /v1/local/voice/sessions/{session_id}`
- `POST /v1/local/voice/sessions/{session_id}/turns`
- `WS /v1/local/voice/sessions/{session_id}/realtime`
- `WS /v1/realtime/sessions/{session_id}`
- `GET /v1/local/capabilities`
- `GET /v1/local/concurrency/status`

The OpenAI-compatible endpoints accept OpenAI-style text, tool calls, image
parts, and Responses API inputs. Gemma video input is translated to sampled
image frames. The default Gemma 4 E4B profile requires the configured projector
for image and video-frame input. If the request already supplies video `frames`,
LAAS uses them directly; otherwise install the optional video extra so OpenCV
can extract frames.

## Install

Detailed environment, wheel, and install-order guidance is in
[docs/INSTALL.md](docs/INSTALL.md).

OpenAI endpoint support is tracked in
[docs/OPENAI_COMPATIBILITY.md](docs/OPENAI_COMPATIBILITY.md).

Gemma multimodal support and live audit commands are tracked in
[docs/GEMMA_MULTIMODAL.md](docs/GEMMA_MULTIMODAL.md).

Local realtime voice compatibility is documented in
[docs/REALTIME.md](docs/REALTIME.md).

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
python -m pip install --upgrade --force-reinstall --no-cache-dir -r requirements-llama-cuda.txt
```

NVIDIA CUDA example for macOS/Linux shells:

```bash
python -m pip install --upgrade --force-reinstall --no-cache-dir -r requirements-llama-cuda.txt
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

For local SDXL Turbo image generation, install PyTorch and TorchVision wheels
from the same PyTorch index for your OS/GPU first, then install the image
dependencies. TorchVision is needed by Transformers image processors; if it is
missing, image generation can still work but the server logs CLIP/SigLIP
fallback warnings.

For NVIDIA GPUs, choose the newest PyTorch CUDA wheel that is less than or
equal to the CUDA version reported by `nvidia-smi`. The CUDA Toolkit command
`nvcc` is not required for prebuilt PyTorch wheels.

Windows PowerShell:

```powershell
nvidia-smi
```

If `nvidia-smi` reports CUDA 12.8 or newer, install the CUDA 12.8 wheels:

```powershell
python -m pip install --force-reinstall `
  --index-url https://download.pytorch.org/whl/cu128 `
  torch torchvision
```

CPU-only Windows example:

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

Then install the Diffusers-side dependencies:

```powershell
python -m pip install -r requirements-image.txt
```

The equivalent `pyproject.toml` extra is `python -m pip install -e ".[image]"`.

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
LAAS_N_CTX=32768
LAAS_N_GPU_LAYERS=-1
LAAS_N_BATCH=512
LAAS_N_UBATCH=512
LAAS_FLASH_ATTN=true
LAAS_OFFLOAD_KQV=true
LAAS_SPECULATIVE_DECODING=false
LAAS_SPECULATIVE_MODE=prompt_lookup
LAAS_MTP_FILENAME=
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
LAAS_EMBEDDING_MODEL_ID=bge-small-en-v1.5
LAAS_EMBEDDING_HF_REPO_ID=BAAI/bge-small-en-v1.5
LAAS_EMBEDDING_DIMENSIONS=384
LAAS_EMBEDDING_AUTO_LOAD=false
LAAS_EMBEDDING_AUTO_DOWNLOAD=true
LAAS_EMBEDDING_IDLE_UNLOAD_SECONDS=900
LAAS_EMBEDDING_DEVICE=auto
LAAS_IMAGE_MODEL_ID=sdxl-turbo
LAAS_IMAGE_HF_REPO_ID=stabilityai/sdxl-turbo
LAAS_IMAGE_DEFAULT_SIZE=768x768
LAAS_IMAGE_NUM_INFERENCE_STEPS=2
LAAS_IMAGE_GUIDANCE_SCALE=0.0
LAAS_IMAGE_DEFAULT_RESPONSE_FORMAT=b64_json
LAAS_IMAGE_OUTPUT_DIR=
LAAS_IMAGE_OUTPUT_RETENTION_SECONDS=86400
LAAS_IMAGE_AUTO_LOAD=false
LAAS_IMAGE_AUTO_DOWNLOAD=true
LAAS_IMAGE_EXCLUSIVE_LOAD=true
LAAS_IMAGE_VARIATION_DEFAULT_SIZE=512x512
LAAS_IMAGE_VARIATION_NUM_INFERENCE_STEPS=4
LAAS_IMAGE_VARIATION_GUIDANCE_SCALE=0.0
LAAS_IMAGE_VARIATION_STRENGTH=0.55
LAAS_IMAGE_EDIT_MODEL_ID=sd-1.5-inpainting
LAAS_IMAGE_EDIT_HF_REPO_ID=stable-diffusion-v1-5/stable-diffusion-inpainting
LAAS_IMAGE_EDIT_DEFAULT_SIZE=512x512
LAAS_IMAGE_EDIT_NUM_INFERENCE_STEPS=25
LAAS_IMAGE_EDIT_GUIDANCE_SCALE=7.5
LAAS_IMAGE_EDIT_STRENGTH=0.8
LAAS_IMAGE_EDIT_PADDING_MASK_CROP=32
LAAS_IMAGE_EDIT_COMPOSITE_BLUR_RADIUS=4
LAAS_IMAGE_EDIT_AUTO_LOAD=false
LAAS_IMAGE_EDIT_AUTO_DOWNLOAD=true
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

Check installation and optional dependency status without starting the server:

```powershell
laas diagnose
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

When launched through `laas`, startup checks the configured required model path
before the server starts. If the model file is missing, LAAS prints the model
id, Hugging Face repo, filename, and target path, then asks whether to download
the missing assets.

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
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/local/diagnostics
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/local/compatibility
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
the configured model and projector when required files are missing. Inference
requests do not trigger a model download by default. If required files are
missing, inference returns `model_not_downloaded` with instructions to call the
local download/load endpoints.

Set `LAAS_AUTO_DOWNLOAD=true` only if you want LAAS to download a missing model
during auto-load or first inference. With the default `LAAS_AUTO_DOWNLOAD=false`,
downloads happen only after an explicit local download/load request.

Gemma 4 E4B requires a projector for multimodal requests. The default projector
is `mmproj-gemma-4-E4B-it-Q8_0.gguf`.

The default context profile is tuned for maximum practical context on the
reference RTX 3060 Ti setup:

```text
LAAS_N_CTX=32768
LAAS_N_GPU_LAYERS=-1
LAAS_N_BATCH=512
LAAS_N_UBATCH=512
```

Use `scripts/tune_gemma4_context.py` to find the largest context that remains
snappy on another machine.

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

## Local Image Generation

The first local image backend is SDXL Turbo through Diffusers. Install the image
dependencies, start `laas`, then point any OpenAI-compatible image client at
`http://127.0.0.1:8000/v1`. `POST /v1/images/generations` downloads the
configured image snapshot on first use when `LAAS_IMAGE_AUTO_DOWNLOAD=true`,
which is the default, then loads it and generates the image.

Generate image URLs:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")

image = client.images.generate(
    model="sdxl-turbo",
    prompt="a cinematic photo of a tiny robot repairing a neon sign",
    size="768x768",
    n=2,
    response_format="url",
    quality="high",
    style="vivid",
)
print(image.data[0].url)
```

While the first request is downloading, check progress from another terminal:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/local/images/status
```

macOS/Linux:

```bash
curl http://127.0.0.1:8000/v1/local/images/status
```

The status response includes `download_in_progress`, `download_started_at`,
`download_finished_at`, `last_download_error`, `output_dir`, and
`output_retention_seconds`. The server console also logs when the Hugging Face
snapshot download starts, finishes, or fails.

Manual download and load endpoints are still available for prewarming or for
setups that choose `LAAS_IMAGE_AUTO_DOWNLOAD=false`:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/images/download `
  -ContentType "application/json" `
  -Body "{}"

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/images/load `
  -ContentType "application/json" `
  -Body "{}"
```

macOS/Linux:

```bash
curl -X POST http://127.0.0.1:8000/v1/local/images/download \
  -H "Content-Type: application/json" \
  -d "{}"

curl -X POST http://127.0.0.1:8000/v1/local/images/load \
  -H "Content-Type: application/json" \
  -d "{}"
```

The generation endpoint supports `response_format=b64_json`, `response_format=url`,
and `n >= 1`. URL outputs are saved under `LAAS_IMAGE_OUTPUT_DIR`, or
`<LAAS_MODEL_DIR>/outputs/images` by default, and served from
`/v1/local/files/images/{filename}`. Old outputs are removed opportunistically
according to `LAAS_IMAGE_OUTPUT_RETENTION_SECONDS`.

OpenAI image parameters are translated for SDXL Turbo where possible:
`quality=high`/`hd` increases the default step count when
`num_inference_steps` is not supplied, `style=vivid|natural` adds a prompt hint,
and `background=opaque|auto` plus `moderation=auto|low` are accepted for client
compatibility. `background=transparent` returns a clear unsupported-parameter
error because SDXL Turbo does not generate alpha-channel transparent PNGs.
SDXL Turbo unloads after
`LAAS_IMAGE_IDLE_UNLOAD_SECONDS` seconds of inactivity; set it to `0` to keep it
loaded.

`POST /v1/images/variations` is implemented as a local SDXL Turbo img2img
translation of OpenAI's DALL-E-style variations endpoint. It accepts multipart
form data with a square PNG `image`, plus `n`, `size`, `response_format`, and
`user`. Local-only tuning fields `seed`, `strength`, `guidance_scale`, and
`num_inference_steps` are also accepted. Since the OpenAI variations endpoint
does not include a prompt, LAAS uses `LAAS_IMAGE_VARIATION_PROMPT` as the local
img2img prompt.

Image edits use `stable-diffusion-v1-5/stable-diffusion-inpainting` through
Diffusers. The edit model has separate load/download/unload lifecycle endpoints
so it does not need to stay in memory with SDXL Turbo:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/images/edit/download `
  -ContentType "application/json" `
  -Body "{}"

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/images/edit/load `
  -ContentType "application/json" `
  -Body "{}"
```

OpenAI-compatible edit requests use multipart form data with `image`, `prompt`,
and optional `mask`. Diffusers masks use white pixels for the area to repaint
and black pixels for the area to preserve. If the uploaded mask has transparent
pixels, LAAS treats transparent pixels as the edit area. If no mask is provided,
the source image must have transparency so LAAS can derive the edit area from
alpha.

For SD 1.5 inpainting, prompt for the whole finished scene rather than only the
object inside the mask. A loose object envelope usually works better than a
tight silhouette, but rectangular masks can still show a changed wall or
background patch. `LAAS_IMAGE_EDIT_PADDING_MASK_CROP` gives Diffusers extra crop
context around the mask when the installed pipeline supports it, and
`LAAS_IMAGE_EDIT_COMPOSITE_BLUR_RADIUS` softens the final mask edge.

Use the helper script to create a mask and preview the edit area:

```powershell
python .\scripts\make_inpaint_mask.py `
  --image .\base.png `
  --mask .\mask.png `
  --preview .\mask-preview.png `
  --rect 210,35,341,211
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")

edited = client.images.edit(
    model="sd-1.5-inpainting",
    image=open("base.png", "rb"),
    mask=open("mask.png", "rb"),
    prompt="replace the empty wall with a framed landscape painting",
    size="512x512",
    response_format="url",
)
print(edited.data[0].url)
```

The edit endpoint supports `response_format=b64_json`, `response_format=url`,
`n >= 1`, `negative_prompt`, `strength`, `guidance_scale`,
`num_inference_steps`, `seed`, `quality`, `input_fidelity`, `background`, and
`moderation`. Image generation, edits, and variations support
`output_format=png|jpeg|webp` and `output_compression` for JPEG/WebP.

Use `GET /v1/local/images/status/all` to inspect generation and edit model
status together, including active image jobs and the last image job error.
Use `POST /v1/local/images/unload/all` to unload both image pipelines at once.
By default, `LAAS_IMAGE_EXCLUSIVE_LOAD=true` means loading or using generation
and variation unloads the image edit pipeline first, and loading or using image
edits unloads the generation/variation pipeline first. This keeps SDXL Turbo and
SD 1.5 inpainting from sitting in memory together unless you explicitly disable
exclusive loading.

## Local Video Generation

`POST /v1/videos/generations` is implemented as a local image-to-video API
surface for the configured `wan2.2-ti2v-5b-turbo-q3_k_m` model. The native
runner uses Diffusers' Wan pipeline with a single Q3_K_M GGUF transformer from
`hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF`, a quantized UMT5 encoder GGUF from
`city96/umt5-xxl-encoder-gguf`, plus tokenizer, scheduler, VAE, and component
configs from `Wan-AI/Wan2.2-TI2V-5B-Diffusers`. LAAS downloads the required GGUF
files and the required small Diffusers-side components, not the full transformer
or text-encoder safetensor shards from the base repo.

The endpoint accepts multipart form data with `prompt`, `image`, and optional
`size`, `seconds`, `fps`, `num_inference_steps`, `guidance_scale`, `seed`, and
`response_format`. `response_format` can be `b64_json` or `url`; URL outputs are
stored under `<LAAS_MODEL_DIR>/outputs/videos` by default and served from
`/v1/local/files/videos/{filename}`.

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/videos/download `
  -Body "{}" -ContentType "application/json"

$response = Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/videos/generations `
  -Form @{
    prompt = "a brass table lamp glowing in a quiet room"
    image = Get-Item .\frame.png
    size = "832x480"
    seconds = "2"
    response_format = "b64_json"
  }
```

Install `requirements-image.txt` first. The native runner requires `diffusers`,
`transformers`, `torch`, `pillow`, `safetensors`, and `gguf`. On constrained
GPUs, keep `LAAS_VIDEO_GENERATION_ENABLE_MODEL_CPU_OFFLOAD=true` and start with
short clips. The old A14B I2V dual-expert profile is still configurable, but it
is no longer the default because it pins too much memory and can thrash 8GB
systems.

Live smoke against a running server:

```powershell
.\.venv\Scripts\python.exe scripts\video_generation_smoke.py --output .\wan-video-smoke.mp4 --seconds 2 --steps 4
```

Use `POST /v1/local/unload/all` to unload the text model, voice stack, STT model,
image pipelines, and video generation model in one request.

## VRAM Concurrency Coordination

LAAS includes a thread-safe VRAM Concurrency Coordinator to serialize heavy GPU-bound resources (LLM completions, SDXL Turbo image generation, SD 1.5 inpainting, and video generation). It automatically:
- Serializes concurrent GPU-bound requests to prevent CUDA Out-of-Memory (OOM) errors and KV cache corruption.
- Safely unloads conflicting heavy models and clears PyTorch CUDA memory cache on swaps.
- Stream-wraps completions to hold the serialization lock until the client finishes reading.
- Bypasses lightweight CPU-bound endpoints (Kokoro TTS, Whisper STT) to keep them concurrently accessible.

Inspect coordinator state while requests are running:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/local/concurrency/status
```

Run a live concurrency smoke against a running server:

```powershell
.\.venv\Scripts\python.exe scripts\concurrency_smoke.py --include-image-edit
```

For detailed design and configuration, see [docs/CONCURRENCY.md](docs/CONCURRENCY.md).

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

The stable LAAS realtime WebSocket transport is available at
`/v1/local/voice/sessions/{session_id}/realtime`. An OpenAI-shaped wrapper is
also available through `POST /v1/realtime/sessions` followed by
`WS /v1/realtime/sessions/{session_id}`. Both routes use the same local voice
runtime. After connecting, clients can send buffered audio:

```json
{"type":"input_audio_buffer.append","audio":"<base64 audio bytes>"}
{"type":"input_audio_buffer.commit","filename":"question.wav"}
```

or a one-shot local turn:

```json
{"type":"voice.turn","audio":"<base64 audio bytes>","filename":"question.wav"}
```

OpenAI-style text conversation items are also accepted on the
`/v1/realtime/sessions/{session_id}` route:

```json
{
  "type": "conversation.item.create",
  "item": {
    "type": "message",
    "role": "user",
    "content": [{"type": "input_text", "text": "Answer this without audio."}]
  }
}
```

After one or more text items, send `{"type":"response.create"}`. LAAS answers
from the accumulated realtime conversation, skips Whisper because no audio was
provided, and still synthesizes the assistant response with Kokoro.
The OpenAI-shaped realtime route also supports practical text item controls:
`conversation.item.retrieve`, `conversation.item.delete`, and
`conversation.item.truncate`.

`POST /v1/realtime/sessions` and `session.update` accept local-compatible
`modalities`, `input_audio_format`, `output_audio_format`, `response_format`,
and `turn_detection` fields. `turn_detection` is stored and returned for client
compatibility; `{"type":"server_vad"}` enables a built-in energy detector for
PCM/WAV input and can auto-commit after trailing silence.

The local route replies with `response.completed` containing the same turn
payload as the HTTP endpoint. The OpenAI-shaped route replies with a
`realtime.response` object and includes the full local payload under
`laas_turn`. It also emits `response.created`, output item events,
`response.output_text.delta`, and `response.audio.delta` before the final
completion event. Text deltas come from backend streaming when available. With
the current Kokoro backend, audio deltas are chunked from the completed TTS
buffer rather than produced by native streaming synthesis.
Both routes accept `session.update`, `input_audio_buffer.clear`,
`response.cancel`, and `session.close` control events.

Run a live realtime voice smoke against a server with the voice stack loaded:

```powershell
python .\scripts\realtime_voice_smoke.py --base-url http://127.0.0.1:8000 --output .\realtime-smoke-output.wav
python .\scripts\realtime_voice_smoke.py --base-url http://127.0.0.1:8000 --text-only --output .\realtime-text-output.wav
```

macOS/Linux:

```bash
python scripts/realtime_voice_smoke.py --base-url http://127.0.0.1:8000 --output realtime-smoke-output.wav
python scripts/realtime_voice_smoke.py --base-url http://127.0.0.1:8000 --text-only --output realtime-text-output.wav
```

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

The local embeddings endpoint exposes `bge-small-en-v1.5` by default using
`BAAI/bge-small-en-v1.5` through Sentence Transformers. It downloads to
`LAAS_MODEL_DIR` like the other local model stacks and unloads after
`LAAS_EMBEDDING_IDLE_UNLOAD_SECONDS` seconds of inactivity.

```python
embedding = client.embeddings.create(
    model="bge-small-en-v1.5",
    input=["alpha", "beta"],
    dimensions=128,
)
print(len(embedding.data[0].embedding))
```

Manual embedding lifecycle endpoints:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/local/embeddings/status
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/embeddings/download -Body "{}" -ContentType "application/json"
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/embeddings/load -Body "{}" -ContentType "application/json"
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/embeddings/unload
```

Files and vector stores are local. On Windows, uploaded file bytes and SQLite
metadata default to `D:\AI\FileStorage`; on macOS/Linux they default to
`~/AI/FileStorage`. Override with `LAAS_FILE_STORAGE_DIR`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")

uploaded = client.files.create(
    file=open("notes.md", "rb"),
    purpose="assistants",
)
store = client.vector_stores.create(name="local-docs")
client.vector_stores.files.create(
    vector_store_id=store.id,
    file_id=uploaded.id,
)
```

OpenAI's hosted File Search API does not define a direct local search route, so
LAAS adds one:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/v1/local/vector_stores/vs_your_store_id/search `
  -ContentType "application/json" `
  -Body '{"query":"how do I configure Vulkan?","limit":8}'
```

Chat Completions and Responses can also use local vector stores through a
`file_search` tool. LAAS searches the configured stores, injects the retrieved
snippets as local context, and returns the matches in `laas_file_search`:

```python
response = client.chat.completions.create(
    model="gemma-4-e4b-it-q4_k_m",
    messages=[{"role": "user", "content": "How do I configure Vulkan?"}],
    tools=[{"type": "file_search", "vector_store_ids": [store.id]}],
)
print(response.choices[0].message.content)
```

Vector store file attachment indexes synchronously by default. To return
immediately and poll status:

```python
requests.post(
    f"{base_url}/vector_stores/{store_id}/files",
    json={"file_id": file_id, "wait": False},
).raise_for_status()

requests.get(
    f"{base_url}/local/vector_stores/{store_id}/indexing/status",
).raise_for_status()
```

The first local batch implementation supports JSONL requests for
`/v1/embeddings`. Upload a batch input file with purpose `batch`, then call
`POST /v1/batches`; the output is written as another local file. Batch records
are persisted in the same SQLite file as Files and Vector Stores.

Async vector indexing and local batches also create records under
`GET /v1/local/jobs`. The `/v1/moderations` endpoint is implemented as
deterministic local rules for compatibility, not as a replacement for hosted
moderation classifiers.

Local storage auto-prunes unused files, terminal jobs, and terminal batch
records older than `LAAS_STORAGE_PRUNE_UNUSED_DAYS`, which defaults to `180`.
Files referenced by vector stores or active/recent batches are preserved. Review
with a dry run:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/v1/local/storage/prune `
  -ContentType "application/json" `
  -Body '{"dry_run":true,"older_than_days":180}'
```

## Compatibility Testing

Golden OpenAI-compatible request/response fixtures live under
`tests/fixtures/openai_compat`.

Run the fixture suite:

```bash
python -m pytest tests/test_api.py -k openai_compat_golden_fixture
```

Run the official OpenAI Python client smoke script against a running LAAS
server:

```bash
python scripts/openai_client_smoke.py --base-url http://127.0.0.1:8000
```

Optional heavier checks:

```bash
python scripts/openai_client_smoke.py --base-url http://127.0.0.1:8000 --include-image
python scripts/openai_client_smoke.py --base-url http://127.0.0.1:8000 --include-image --include-image-edit --include-voice
python scripts/openai_client_smoke.py --base-url http://127.0.0.1:8000 --include-storage
python scripts/realtime_voice_smoke.py --base-url http://127.0.0.1:8000
python scripts/multimodal_fidelity_smoke.py --base-url http://127.0.0.1:8000
```

Run a lightweight endpoint compatibility probe against a running server:

```bash
laas compat-check --base-url http://127.0.0.1:8000
```

The compatibility probe includes `POST /v1/realtime/sessions`. On a fresh
install, missing local voice assets still count as a predictable registered
route response.

Release validation is tracked in [docs/RELEASE.md](docs/RELEASE.md).

Live pytest smoke tests are disabled unless the relevant environment variables
are set: `LAAS_LIVE_SMOKE=true`, `LAAS_LIVE_SMOKE_IMAGES=true`, or
`LAAS_LIVE_SMOKE_VOICE=true`. See [docs/INSTALL.md](docs/INSTALL.md) for the
full commands and download-safety switch.

## Notes

Gemma 4 E4B is exposed as a text-output model with text, tool-call, image,
video-as-frames, and reasoning capabilities. LAAS
validates request capabilities before sending prompts to the backend and
preserves OpenAI response shapes where practical for local inference.
