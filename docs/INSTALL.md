# Installation

LAAS is a Python package and a FastAPI service. Keep it in a virtual
environment, install the API host first, then install the llama.cpp backend wheel
that matches your machine.

The `laas` console command and `python -m uvicorn laas.app:app ...` both use the
Python environment they are launched from. If `llama-cpp-python` is installed in
a different environment, model loading will fail.

## 1. Create an environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

## 2. Install the API host

For normal use:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

For development and tests:

```powershell
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

The equivalent modern packaging command is:

```powershell
python -m pip install -e ".[dev]"
```

The requirements files are present so contributors and packagers can see the
dependency sets without needing to parse `pyproject.toml`.

## 3. Install a llama.cpp backend wheel

`llama-cpp-python` is deliberately separate from the base install because the
right wheel depends on CPU/GPU hardware, Python version, CUDA version, and OS.

The upstream project documents pre-built CPU, CUDA, Metal, ROCm, Vulkan, and
HIP Radeon wheels in the llama-cpp-python README:

<https://github.com/abetlen/llama-cpp-python#supported-backends>

### CPU wheel

This is the most portable option and the slowest option:

```powershell
python -m pip install -r requirements-llama-cpu.txt
```

Equivalent explicit command:

```powershell
python -m pip install llama-cpp-python `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### NVIDIA CUDA wheel

Pick the wheel index matching your CUDA runtime. As of the upstream README
checked while writing this, CUDA wheel indexes include:

- `cu118`
- `cu121`
- `cu122`
- `cu123`
- `cu124`
- `cu125`
- `cu130`
- `cu132`

Example for CUDA 12.4:

```powershell
python -m pip install llama-cpp-python `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

Example for CUDA 12.1:

```powershell
python -m pip install llama-cpp-python `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

Then verify the wheel imports from the same environment:

```powershell
python -c "import llama_cpp; print(llama_cpp.__version__)"
```

If the model loads but all layers run on CPU, reinstall the wheel with the
correct CUDA index and confirm your NVIDIA driver/runtime can load that CUDA
version. LAAS passes `n_gpu_layers=-1` by default, so it will try to offload all
possible layers when the installed backend supports GPU offload.

### Build from source

Use this only when no wheel matches your machine:

```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
$env:FORCE_CMAKE = "1"
python -m pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```

For CPU/OpenBLAS, Metal, Vulkan, ROCm, and HIP Radeon build flags, use the
upstream llama-cpp-python backend instructions.

## 4. Optional video frame extraction

LAAS accepts video inputs by translating video to image frames before sending the
request to Gemma. If the client sends `frames`, OpenCV is not needed. If the
client sends a video file path, URL, or data URL, install:

```powershell
python -m pip install -r requirements-video.txt
```

or:

```powershell
python -m pip install -e ".[video]"
```

## 5. Configure model storage

The default model directory is:

```text
D:\AI\Models
```

That default is intentional so large models do not land on the OS drive.

Configuration order:

1. Environment variables such as `LAAS_MODEL_DIR`.
2. `.env` in the repo root.
3. `.laas/settings.json`, written by `PATCH /v1/local/settings`.
4. Built-in defaults.

Example `.env`:

```text
LAAS_MODEL_DIR=D:\AI\Models
LAAS_MODEL_ID=gemma-4-e4b-it-q4_k_m
LAAS_HF_REPO_ID=ggml-org/gemma-4-E4B-it-GGUF
LAAS_HF_FILENAME=gemma-4-E4B-it-Q4_K_M.gguf
LAAS_AUTO_LOAD=false
LAAS_IDLE_UNLOAD_SECONDS=900
```

## 6. Download/load behavior

LAAS uses `huggingface-hub` to download the configured GGUF.

The model is downloaded when either:

- `POST /v1/local/models/download` is called, or
- `POST /v1/local/models/load` is called and the configured model file is
  missing, or
- the server starts with `LAAS_AUTO_LOAD=true` and the configured model file is
  missing.

By default, `LAAS_AUTO_LOAD=false`. That means starting the API does not
download or load the model until you explicitly ask it to.

Download and load manually:

```powershell
curl -X POST http://127.0.0.1:8000/v1/local/models/download `
  -H "Content-Type: application/json" `
  -d "{}"

curl -X POST http://127.0.0.1:8000/v1/local/models/load `
  -H "Content-Type: application/json" `
  -d "{}"
```

If the model is not loaded and a client calls an inference endpoint, LAAS will
attempt to load it. If the model file is missing, that load path also downloads
the model first.

## 7. Run

Normal mode:

```powershell
laas
```

Direct uvicorn mode:

```powershell
python -m uvicorn laas.app:app --host 127.0.0.1 --port 8000
```

Development reload mode:

```powershell
python -m uvicorn laas.app:app --host 127.0.0.1 --port 8000 --reload
```

## 8. Verify

```powershell
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/local/models/status
```

Run tests:

```powershell
python -m pytest
```

## 9. Unload

Unload explicitly:

```powershell
curl -X POST http://127.0.0.1:8000/v1/local/models/unload
```

LAAS also unloads the active model after `LAAS_IDLE_UNLOAD_SECONDS` seconds of
inactivity. Set `LAAS_IDLE_UNLOAD_SECONDS=0` to disable idle unloading.
