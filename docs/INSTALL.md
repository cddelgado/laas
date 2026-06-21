# Installation

LAAS is a Python package and a FastAPI service. Use a virtual environment,
install the API host first, then install the `llama-cpp-python` backend wheel
that matches your machine.

The `laas` console command and `python -m uvicorn laas.app:app ...` both use the
Python environment they are launched from. If `llama-cpp-python` is installed in
a different environment, model loading will fail.

The upstream `llama-cpp-python` project documents the current backend wheel
indexes here:

<https://github.com/abetlen/llama-cpp-python#supported-backends>

## 1. Create an Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 2. Install the API Host

Normal install:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Development and tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Equivalent `pyproject.toml` extras install:

```bash
python -m pip install -e ".[dev]"
```

The requirements files are present so contributors and packagers can see the
dependency sets without parsing `pyproject.toml`.

## 3. Install a llama.cpp Backend

`llama-cpp-python` is separate from the base install because the correct wheel
depends on OS, Python version, GPU vendor, driver/runtime, and acceleration
backend.

Install exactly one backend wheel first. If you change backend, reinstall
`llama-cpp-python` with `--upgrade --force-reinstall --no-cache-dir`.

### CPU

Most portable, slowest:

```bash
python -m pip install -r requirements-llama-cpu.txt
```

Explicit command:

```bash
python -m pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

PowerShell explicit command:

```powershell
python -m pip install llama-cpp-python `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### NVIDIA CUDA

Pick the wheel index matching your CUDA runtime. As of the upstream README
checked for this doc, CUDA indexes include:

- `cu118`
- `cu121`
- `cu122`
- `cu123`
- `cu124`
- `cu125`
- `cu130`
- `cu132`

Example for CUDA 12.4:

```bash
python -m pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

PowerShell:

```powershell
python -m pip install llama-cpp-python `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

### Apple Metal

Prebuilt Metal wheel:

```bash
python -m pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```

Source build fallback:

```bash
CMAKE_ARGS="-DGGML_METAL=on" python -m pip install llama-cpp-python
```

On Apple Silicon, use an arm64 Python build. An x86_64 Python can install but
will build/run the wrong architecture.

### AMD ROCm on Linux

Prebuilt ROCm wheel:

```bash
python -m pip install -r requirements-llama-rocm.txt
```

Explicit command:

```bash
python -m pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/rocm72
```

Source build fallback:

```bash
CMAKE_ARGS="-DGGML_HIP=on" python -m pip install llama-cpp-python
```

### AMD HIP Radeon on Windows

Prebuilt HIP Radeon wheel:

```powershell
python -m pip install -r requirements-llama-hip-radeon.txt
```

Explicit command:

```powershell
python -m pip install llama-cpp-python `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/hip-radeon
```

### Vulkan on Linux or Windows

Vulkan is useful when CUDA/ROCm/Metal is not the right fit and a Vulkan-capable
GPU stack is available.

Prebuilt Vulkan wheel:

```bash
python -m pip install -r requirements-llama-vulkan.txt
```

Explicit command:

```bash
python -m pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/vulkan
```

PowerShell explicit command:

```powershell
python -m pip install llama-cpp-python `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/vulkan
```

Source build fallback:

```bash
CMAKE_ARGS="-DGGML_VULKAN=on" python -m pip install llama-cpp-python
```

PowerShell source build fallback:

```powershell
$env:CMAKE_ARGS = "-DGGML_VULKAN=on"
python -m pip install llama-cpp-python
```

### Verify the Backend

Run this from the same activated environment:

```bash
python -c "import llama_cpp; print(llama_cpp.__version__)"
```

If a GPU backend installs but inference runs on CPU, reinstall the wheel with
the correct backend index and confirm the OS driver/runtime can load that
backend. LAAS passes `n_gpu_layers=-1` by default, so it attempts to offload all
possible layers when the installed backend supports GPU offload.

## 4. Optional Video Frame Extraction

LAAS accepts video inputs by translating video to image frames before sending the
request to Gemma. If the client sends `frames`, OpenCV is not needed. If the
client sends a video file path, URL, or data URL, install:

```bash
python -m pip install -r requirements-video.txt
```

or:

```bash
python -m pip install -e ".[video]"
```

## 5. Optional Local Image Generation

The SDXL Turbo image backend uses PyTorch and Diffusers. Install PyTorch and
TorchVision wheels from the same PyTorch index for your OS/GPU first, then
install LAAS image dependencies. TorchVision is needed by Transformers image
processors; if it is missing, image generation can still work but the server
logs CLIP/SigLIP fallback warnings.

### Choosing a PyTorch CUDA wheel

For prebuilt PyTorch wheels, the CUDA version is the runtime bundled with the
wheel. You do not need the CUDA Toolkit or `nvcc` unless you are compiling CUDA
extensions.

On NVIDIA systems, check the driver-supported CUDA runtime:

Windows PowerShell:

```powershell
nvidia-smi
```

macOS/Linux shell:

```bash
nvidia-smi
```

Use the newest PyTorch CUDA wheel index that is less than or equal to the CUDA
version reported by `nvidia-smi`. If `nvidia-smi` reports CUDA 12.8 or newer,
use `cu128`. If it reports CUDA 12.6, use `cu126`. If it reports only CUDA
12.4, use `cu124` with a PyTorch version that still publishes that wheel.

Current common CUDA 12 choices:

```text
nvidia-smi reports 12.8 or newer -> https://download.pytorch.org/whl/cu128
nvidia-smi reports 12.6          -> https://download.pytorch.org/whl/cu126
nvidia-smi reports 12.4          -> https://download.pytorch.org/whl/cu124
No NVIDIA GPU or no CUDA needed  -> https://download.pytorch.org/whl/cpu
```

Use the PyTorch install selector as the source of truth when this list ages:
<https://pytorch.org/get-started/locally/>.

If you are switching an existing environment from CPU wheels to CUDA wheels,
use `--force-reinstall`. Otherwise pip may keep an already-installed CPU wheel
with the same public version.

CUDA 12.8 Windows example:

```powershell
python -m pip install --force-reinstall `
  --index-url https://download.pytorch.org/whl/cu128 `
  torch torchvision
```

CUDA 12.6 Windows example:

```powershell
python -m pip install --force-reinstall `
  --index-url https://download.pytorch.org/whl/cu126 `
  torch torchvision
```

Verify the install:

```powershell
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

The CUDA build is working when the Torch and TorchVision versions include a
matching `+cu...` suffix and `torch.cuda.is_available()` prints `True`.

CPU-only Windows example:

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

Then install the Diffusers-side dependencies:

```bash
python -m pip install -r requirements-image.txt
```

PowerShell:

```powershell
python -m pip install -r requirements-image.txt
```

Equivalent `pyproject.toml` extra:

```bash
python -m pip install -e ".[image]"
```

Default assets:

```text
LAAS_IMAGE_MODEL_ID=sdxl-turbo
LAAS_IMAGE_HF_REPO_ID=stabilityai/sdxl-turbo
LAAS_IMAGE_DEFAULT_SIZE=768x768
LAAS_IMAGE_NUM_INFERENCE_STEPS=2
LAAS_IMAGE_GUIDANCE_SCALE=0.0
LAAS_IMAGE_DEFAULT_RESPONSE_FORMAT=b64_json
LAAS_IMAGE_OUTPUT_DIR=
LAAS_IMAGE_OUTPUT_RETENTION_SECONDS=86400
LAAS_IMAGE_AUTO_DOWNLOAD=true
LAAS_IMAGE_EDIT_MODEL_ID=sd-1.5-inpainting
LAAS_IMAGE_EDIT_HF_REPO_ID=stable-diffusion-v1-5/stable-diffusion-inpainting
LAAS_IMAGE_EDIT_DEFAULT_SIZE=512x512
LAAS_IMAGE_EDIT_NUM_INFERENCE_STEPS=25
LAAS_IMAGE_EDIT_GUIDANCE_SCALE=7.5
LAAS_IMAGE_EDIT_STRENGTH=0.8
LAAS_IMAGE_EDIT_PADDING_MASK_CROP=32
LAAS_IMAGE_EDIT_COMPOSITE_BLUR_RADIUS=4
LAAS_IMAGE_EDIT_AUTO_DOWNLOAD=true
```

The image snapshots use the same `LAAS_MODEL_DIR` root as the GGUF model. The
backend downloads Diffusers snapshot directories rather than single model
files. By default, the OpenAI-compatible `POST /v1/images/generations` and
`POST /v1/images/edits` endpoints download their configured snapshots on first
use, load them, and then return images.
Use `GET /v1/local/images/status` from another terminal to inspect
`download_in_progress`, `download_started_at`, `download_finished_at`, and
`last_download_error` while the first request is running.

`POST /v1/images/generations` supports `response_format=b64_json`,
`response_format=url`, and `n >= 1`. URL outputs are saved under
`LAAS_IMAGE_OUTPUT_DIR`, or `<LAAS_MODEL_DIR>/outputs/images` when unset, and
served from `/v1/local/files/images/{filename}`. LAAS removes old output PNGs
opportunistically according to `LAAS_IMAGE_OUTPUT_RETENTION_SECONDS`.

OpenAI image parameters are translated for SDXL Turbo where possible:

- `quality=high` or `quality=hd` increases the default step count when
  `num_inference_steps` is not supplied.
- `style=vivid` or `style=natural` appends a small style hint to the prompt.
- `background=auto` and `background=opaque` are accepted.
- `moderation=auto` and `moderation=low` are accepted for client compatibility;
  LAAS does not add a local image moderation model.
- `background=transparent` returns an unsupported-parameter error because SDXL
  Turbo does not generate transparent PNGs.

`POST /v1/images/edits` uses `stable-diffusion-v1-5/stable-diffusion-inpainting`
by default. It accepts multipart form data with `image`, `prompt`, and optional
`mask`. Diffusers masks use white pixels for the area to repaint and black
pixels for the area to preserve. If the uploaded mask has transparent pixels,
LAAS treats transparent pixels as the edit area. If no mask is provided, the
source image must have transparency so LAAS can derive the edit area from alpha.

SD 1.5 inpainting works best when the prompt describes the full completed image,
not only the object being inserted. Use a loose mask around the intended object
and give the model enough room to draw edges and shadows. Very tight masks tend
to get healed back into the background, while rectangular masks can leave a
visible changed wall or background patch. `LAAS_IMAGE_EDIT_PADDING_MASK_CROP`
passes Diffusers extra crop context around the mask when supported, and
`LAAS_IMAGE_EDIT_COMPOSITE_BLUR_RADIUS` softens the final mask edge.

Local edit lifecycle endpoints:

```text
GET  /v1/local/images/edit/status
POST /v1/local/images/edit/download
POST /v1/local/images/edit/load
POST /v1/local/images/edit/unload
```

The edit endpoint supports `response_format=b64_json`, `response_format=url`,
`n >= 1`, `negative_prompt`, `strength`, `guidance_scale`,
`num_inference_steps`, `seed`, `quality`, `input_fidelity`, `background`, and
`moderation`.

## 6. Optional Local Voice Stack

The full local voice stack uses Kokoro TTS plus whisper.cpp STT:

```bash
python -m pip install -r requirements-voice.txt
```

PowerShell:

```powershell
python -m pip install -r requirements-voice.txt
```

Equivalent `pyproject.toml` extra:

```bash
python -m pip install -e ".[voice]"
```

Default assets:

```text
LAAS_TTS_HF_REPO_ID=fastrtc/kokoro-onnx
LAAS_TTS_MODEL_FILENAME=kokoro-v1.0.onnx
LAAS_TTS_VOICES_FILENAME=voices-v1.0.bin
LAAS_STT_HF_REPO_ID=ggerganov/whisper.cpp
LAAS_STT_MODEL_FILENAME=ggml-small.bin
```

These files use the same `LAAS_MODEL_DIR` root as the GGUF model.

FFmpeg is optional but required for OpenAI-compatible `aac` and `opus` speech
outputs. Without FFmpeg, LAAS still supports `mp3`, `wav`, `flac`, and `pcm`.

Windows:

```powershell
winget install Gyan.FFmpeg
```

macOS:

```bash
brew install ffmpeg
```

Ubuntu/Debian:

```bash
sudo apt install ffmpeg
```

If the executable is not on `PATH`, set:

```text
LAAS_TTS_FFMPEG_PATH=C:\path\to\ffmpeg.exe
```

## 7. Configure Model Storage

Built-in defaults:

```text
Windows: D:\AI\Models
macOS/Linux: ~/AI/Models
```

The Windows default is intentional for the original development workstation so
large models do not land on the OS drive. On macOS/Linux, set `LAAS_MODEL_DIR`
if `~/AI/Models` is not where you want model files.

Configuration order:

1. Environment variables such as `LAAS_MODEL_DIR`.
2. `.env` in the repo root.
3. `.laas/settings.json`, written by `PATCH /v1/local/settings`.
4. Built-in defaults.

Windows `.env` example:

```text
LAAS_MODEL_DIR=D:\AI\Models
LAAS_MODEL_ID=gemma-4-e4b-it-q4_k_m
LAAS_HF_REPO_ID=ggml-org/gemma-4-E4B-it-GGUF
LAAS_HF_FILENAME=gemma-4-E4B-it-Q4_K_M.gguf
LAAS_MMPROJ_FILENAME=mmproj-gemma-4-E4B-it-Q8_0.gguf
LAAS_MMPROJ_REQUIRED=true
LAAS_AUTO_LOAD=false
LAAS_AUTO_DOWNLOAD=false
LAAS_IDLE_UNLOAD_SECONDS=900
LAAS_TTS_MODEL_ID=kokoro-82m
LAAS_TTS_HF_REPO_ID=fastrtc/kokoro-onnx
LAAS_TTS_MODEL_FILENAME=kokoro-v1.0.onnx
LAAS_TTS_VOICES_FILENAME=voices-v1.0.bin
LAAS_TTS_DEFAULT_VOICE=af_heart
LAAS_TTS_AUTO_LOAD=false
LAAS_TTS_AUTO_DOWNLOAD=false
LAAS_TTS_IDLE_UNLOAD_SECONDS=900
LAAS_TTS_FFMPEG_PATH=ffmpeg
LAAS_STT_MODEL_ID=whisper-small
LAAS_STT_HF_REPO_ID=ggerganov/whisper.cpp
LAAS_STT_MODEL_FILENAME=ggml-small.bin
LAAS_STT_DEFAULT_LANGUAGE=
LAAS_STT_AUTO_LOAD=false
LAAS_STT_AUTO_DOWNLOAD=false
LAAS_STT_IDLE_UNLOAD_SECONDS=900
LAAS_VOICE_AUTO_LOAD=false
LAAS_VOICE_AUTO_DOWNLOAD=false
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
LAAS_IMAGE_IDLE_UNLOAD_SECONDS=900
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
LAAS_IMAGE_EDIT_IDLE_UNLOAD_SECONDS=900
```

macOS/Linux `.env` example:

```text
LAAS_MODEL_DIR=/mnt/ai/models
LAAS_MODEL_ID=gemma-4-e4b-it-q4_k_m
LAAS_HF_REPO_ID=ggml-org/gemma-4-E4B-it-GGUF
LAAS_HF_FILENAME=gemma-4-E4B-it-Q4_K_M.gguf
LAAS_MMPROJ_FILENAME=mmproj-gemma-4-E4B-it-Q8_0.gguf
LAAS_MMPROJ_REQUIRED=true
LAAS_AUTO_LOAD=false
LAAS_AUTO_DOWNLOAD=false
LAAS_IDLE_UNLOAD_SECONDS=900
LAAS_TTS_MODEL_ID=kokoro-82m
LAAS_TTS_HF_REPO_ID=fastrtc/kokoro-onnx
LAAS_TTS_MODEL_FILENAME=kokoro-v1.0.onnx
LAAS_TTS_VOICES_FILENAME=voices-v1.0.bin
LAAS_TTS_DEFAULT_VOICE=af_heart
LAAS_TTS_AUTO_LOAD=false
LAAS_TTS_AUTO_DOWNLOAD=false
LAAS_TTS_IDLE_UNLOAD_SECONDS=900
LAAS_TTS_FFMPEG_PATH=ffmpeg
LAAS_STT_MODEL_ID=whisper-small
LAAS_STT_HF_REPO_ID=ggerganov/whisper.cpp
LAAS_STT_MODEL_FILENAME=ggml-small.bin
LAAS_STT_DEFAULT_LANGUAGE=
LAAS_STT_AUTO_LOAD=false
LAAS_STT_AUTO_DOWNLOAD=false
LAAS_STT_IDLE_UNLOAD_SECONDS=900
LAAS_VOICE_AUTO_LOAD=false
LAAS_VOICE_AUTO_DOWNLOAD=false
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
LAAS_IMAGE_IDLE_UNLOAD_SECONDS=900
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
LAAS_IMAGE_EDIT_IDLE_UNLOAD_SECONDS=900
```

## 8. Download/Load Behavior

LAAS uses `huggingface-hub` to download the configured GGUF.

The model is downloaded when either:

- `POST /v1/local/models/download` is called. By default this downloads both
  the main GGUF and `LAAS_MMPROJ_FILENAME`.
- `POST /v1/local/models/load` is called and the configured model file is
  missing, unless the request body sets `download_if_missing=false`.
- The server starts with `LAAS_AUTO_LOAD=true`, the configured model file is
  missing, and `LAAS_AUTO_DOWNLOAD=true`.
- An inference endpoint is called while the model is unloaded; LAAS attempts to
  load the model, and that load path downloads the file only when
  `LAAS_AUTO_DOWNLOAD=true`.

By default, `LAAS_AUTO_LOAD=false` and `LAAS_AUTO_DOWNLOAD=false`. Starting the
API does not download or load the model until you explicitly ask it to.
Inference requests return `model_not_downloaded` when the model file is missing.

Gemma 4 multimodal requests require a projector. The default Q4 main model uses
`mmproj-gemma-4-E4B-it-Q8_0.gguf` because the repo currently publishes Q8 and
bf16 projectors, not a Q4 projector. Set `LAAS_MMPROJ_REQUIRED=false` only for
text-only runs.

Check status first:

Windows PowerShell:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/local/models/status
```

macOS/Linux:

```bash
curl http://127.0.0.1:8000/v1/local/models/status
```

Manual download is the confirmation step. Run it only after you are ready for
LAAS to fetch the configured GGUF and projector into `LAAS_MODEL_DIR`.

Windows PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/models/download `
  -ContentType "application/json" `
  -Body "{}"

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/models/load `
  -ContentType "application/json" `
  -Body "{}"
```

To require a previous download and fail instead of downloading during load:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/models/load `
  -ContentType "application/json" `
  -Body '{"download_if_missing": false}'
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

To require a previous download and fail instead of downloading during load:

```bash
curl -X POST http://127.0.0.1:8000/v1/local/models/load \
  -H "Content-Type: application/json" \
  -d '{"download_if_missing": false}'
```

To opt into missing-model downloads during startup auto-load or first inference:

```text
LAAS_AUTO_DOWNLOAD=true
```

## 9. Run

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

Direct uvicorn mode for any platform:

```bash
python -m uvicorn laas.app:app --host 127.0.0.1 --port 8000
```

Development reload mode:

```bash
python -m uvicorn laas.app:app --host 127.0.0.1 --port 8000 --reload
```

When launched through `laas`, startup checks the configured model path before
the server starts. If the main model or projector file is missing, LAAS prints
the model id, Hugging Face repo, filenames, and target paths, then asks whether
to download the missing assets. Answer `y` or `yes` to confirm the download.

To download without prompting:

```bash
laas --yes-download
```

To skip the startup prompt:

```bash
laas --no-download-prompt
```

Direct `uvicorn` launches do not ask interactive questions. They are intended
for service/process-manager use. Use `/v1/local/models/status` and
`/v1/local/models/download` for manual control in that mode.

## 10. Verify

Windows PowerShell:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/models
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/local/models/status
```

macOS/Linux:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/local/models/status
```

Run tests:

```bash
python -m pytest
```

## 11. Unload

Windows PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/local/models/unload
```

macOS/Linux:

```bash
curl -X POST http://127.0.0.1:8000/v1/local/models/unload
```

LAAS also unloads the active model after `LAAS_IDLE_UNLOAD_SECONDS` seconds of
inactivity. Set `LAAS_IDLE_UNLOAD_SECONDS=0` to disable idle unloading.
