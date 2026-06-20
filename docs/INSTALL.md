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

## 5. Configure Model Storage

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
LAAS_AUTO_LOAD=false
LAAS_AUTO_DOWNLOAD=false
LAAS_IDLE_UNLOAD_SECONDS=900
```

macOS/Linux `.env` example:

```text
LAAS_MODEL_DIR=/mnt/ai/models
LAAS_MODEL_ID=gemma-4-e4b-it-q4_k_m
LAAS_HF_REPO_ID=ggml-org/gemma-4-E4B-it-GGUF
LAAS_HF_FILENAME=gemma-4-E4B-it-Q4_K_M.gguf
LAAS_AUTO_LOAD=false
LAAS_AUTO_DOWNLOAD=false
LAAS_IDLE_UNLOAD_SECONDS=900
```

## 6. Download/Load Behavior

LAAS uses `huggingface-hub` to download the configured GGUF.

The model is downloaded when either:

- `POST /v1/local/models/download` is called.
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
LAAS to fetch the configured GGUF into `LAAS_MODEL_DIR`.

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

## 7. Run

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

## 8. Verify

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

## 9. Unload

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
