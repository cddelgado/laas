from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .native import configure_native_dll_directories
from .settings import Settings
from .tts import resolve_ffmpeg_path


OPTIONAL_PACKAGES = {
    "llama_cpp": {
        "distribution": "llama-cpp-python",
        "install": "Install one llama-cpp-python wheel from the platform-specific instructions.",
    },
    "sentence_transformers": {
        "distribution": "sentence-transformers",
        "install": "Install embeddings support with: python -m pip install -r requirements-embeddings.txt",
    },
    "torch": {
        "distribution": "torch",
        "install": "Install image/embedding acceleration dependencies that match your CUDA/ROCm/CPU runtime.",
    },
    "torchvision": {
        "distribution": "torchvision",
        "install": "Install torchvision from the same PyTorch wheel index as torch.",
    },
    "diffusers": {
        "distribution": "diffusers",
        "install": "Install image support with: python -m pip install -r requirements-image.txt",
    },
    "kokoro_onnx": {
        "distribution": "kokoro-onnx",
        "install": "Install TTS support with: python -m pip install -r requirements-tts.txt",
    },
    "pywhispercpp": {
        "distribution": "pywhispercpp",
        "install": "Install STT support with: python -m pip install -r requirements-stt.txt",
    },
    "cv2": {
        "distribution": "opencv-python",
        "install": "Install video frame extraction support with: python -m pip install -r requirements-video.txt",
    },
}


def collect_diagnostics(settings: Settings) -> dict[str, Any]:
    packages = {name: package_status(name, spec["distribution"]) for name, spec in OPTIONAL_PACKAGES.items()}
    report = {
        "object": "local.diagnostics",
        "python": python_status(),
        "platform": platform_status(),
        "settings": settings_status(settings),
        "packages": packages,
        "ffmpeg": ffmpeg_status(settings.tts_ffmpeg_path),
        "torch": torch_status(packages["torch"]["available"]),
        "llama_cpp": llama_cpp_status(packages["llama_cpp"]["available"]),
        "models": model_path_status(settings),
    }
    report["actions"] = diagnostic_actions(report)
    return report


def python_status() -> dict[str, Any]:
    return {
        "version": platform.python_version(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "venv_active": sys.prefix != sys.base_prefix,
    }


def platform_status() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def settings_status(settings: Settings) -> dict[str, Any]:
    return {
        "model_dir": str(settings.model_dir),
        "settings_file": str(settings.settings_file),
        "host": settings.host,
        "port": settings.port,
    }


def package_status(module_name: str, distribution_name: str) -> dict[str, Any]:
    status: dict[str, Any] = {
        "module": module_name,
        "distribution": distribution_name,
        "available": False,
        "version": None,
        "error": None,
    }
    try:
        spec = importlib.util.find_spec(module_name)
        status["available"] = spec is not None
    except Exception as exc:
        status["error"] = str(exc)
        return status
    try:
        status["version"] = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        status["version"] = None
    return status


def ffmpeg_status(ffmpeg_path: str) -> dict[str, Any]:
    resolved = resolve_ffmpeg_path(ffmpeg_path) or shutil.which(ffmpeg_path)
    status: dict[str, Any] = {
        "configured": ffmpeg_path,
        "available": resolved is not None,
        "path": resolved,
        "version": None,
        "error": None,
    }
    if not resolved:
        return status
    try:
        completed = subprocess.run(
            [resolved, "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first_line = (completed.stdout or completed.stderr).splitlines()[0]
        status["version"] = first_line
    except Exception as exc:
        status["error"] = str(exc)
    return status


def torch_status(torch_available: bool) -> dict[str, Any]:
    status: dict[str, Any] = {
        "available": torch_available,
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
        "error": None,
    }
    if not torch_available:
        return status
    try:
        torch = importlib.import_module("torch")
        status["cuda_available"] = bool(torch.cuda.is_available())
        status["cuda_version"] = getattr(torch.version, "cuda", None)
        status["device_count"] = int(torch.cuda.device_count()) if status["cuda_available"] else 0
        status["devices"] = [torch.cuda.get_device_name(index) for index in range(status["device_count"])]
    except Exception as exc:
        status["error"] = str(exc)
    return status


def llama_cpp_status(llama_cpp_available: bool) -> dict[str, Any]:
    status: dict[str, Any] = {
        "available": llama_cpp_available,
        "version": None,
        "n_batch": False,
        "n_ubatch": False,
        "n_threads_batch": False,
        "draft_model": False,
        "prompt_lookup_speculative": False,
        "external_mtp_draft_model": False,
        "flash_attn": False,
        "offload_kqv": False,
        "op_offload": False,
        "swa_full": False,
        "added_dll_directories": [],
        "error": None,
    }
    if not llama_cpp_available:
        return status
    try:
        import inspect

        added_dll_directories = configure_native_dll_directories()
        llama_cpp = importlib.import_module("llama_cpp")
        llama_cls = getattr(llama_cpp, "Llama")
        parameters = inspect.signature(llama_cls).parameters
        status["version"] = getattr(llama_cpp, "__version__", importlib.metadata.version("llama-cpp-python"))
        status["added_dll_directories"] = added_dll_directories
        for name in [
            "n_batch",
            "n_ubatch",
            "n_threads_batch",
            "draft_model",
            "flash_attn",
            "offload_kqv",
            "op_offload",
            "swa_full",
        ]:
            status[name] = name in parameters
        try:
            importlib.import_module("llama_cpp.llama_speculative").LlamaPromptLookupDecoding
            status["prompt_lookup_speculative"] = True
        except Exception:
            status["prompt_lookup_speculative"] = False
    except Exception as exc:
        status["error"] = str(exc)
    return status


def model_path_status(settings: Settings) -> dict[str, Any]:
    return {
        "text": file_model_status(settings.model_id, settings.model_path),
        "mmproj": file_model_status(settings.mmproj_filename or "", settings.mmproj_path),
        "mtp": file_model_status(settings.mtp_filename or "", settings.mtp_path),
        "embeddings": directory_model_status(settings.embedding_model_id, settings.embedding_model_path),
        "images": directory_model_status(settings.image_model_id, settings.image_model_path),
        "image_edit": directory_model_status(settings.image_edit_model_id, settings.image_edit_model_path),
        "tts_model": file_model_status(settings.tts_model_id, settings.tts_model_path),
        "tts_voices": file_model_status(settings.tts_voices_filename, settings.tts_voices_path),
        "transcription": file_model_status(settings.stt_model_id, settings.stt_model_path),
    }


def file_model_status(model_id: str, path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"model_id": model_id, "path": None, "exists": False}
    return {"model_id": model_id, "path": str(path), "exists": path.exists(), "kind": "file"}


def directory_model_status(model_id: str, path: Path) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "path": str(path),
        "exists": path.exists() and any(path.iterdir()),
        "kind": "directory",
    }


def diagnostic_actions(report: dict[str, Any]) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    for module_name, package in report["packages"].items():
        if not package["available"]:
            actions.append({"component": module_name, "message": OPTIONAL_PACKAGES[module_name]["install"]})
    if not report["ffmpeg"]["available"]:
        actions.append({"component": "ffmpeg", "message": "Install FFmpeg and ensure it is on PATH or set LAAS_TTS_FFMPEG_PATH."})
    for name, status in report["models"].items():
        if not status["exists"]:
            actions.append({"component": f"model:{name}", "message": f"Missing configured model asset at {status['path']}."})
    return actions
