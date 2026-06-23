from __future__ import annotations

import os
import site
import sys
from pathlib import Path

_DLL_DIRECTORY_HANDLES: list[object] = []
_ADDED_DLL_DIRECTORIES: set[str] = set()


def configure_native_dll_directories() -> list[str]:
    """Register common venv native-library folders before importing binary wheels."""
    if not sys.platform.startswith("win") or not hasattr(os, "add_dll_directory"):
        return []

    candidates: list[Path] = []
    for base in site.getsitepackages():
        site_path = Path(base)
        candidates.extend(
            [
                site_path / "llama_cpp" / "lib",
                site_path / "torch" / "lib",
                site_path / "nvidia" / "cublas" / "bin",
                site_path / "nvidia" / "cuda_runtime" / "bin",
                site_path / "nvidia" / "cuda_nvrtc" / "bin",
            ]
        )

    added: list[str] = []
    existing = {str(path).lower() for path in _current_path_entries()}
    for candidate in candidates:
        if not candidate.exists():
            continue
        normalized = str(candidate.resolve())
        key = normalized.lower()
        if key in existing or key in _ADDED_DLL_DIRECTORIES:
            continue
        try:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(normalized))
            os.environ["PATH"] = normalized + os.pathsep + os.environ.get("PATH", "")
            _ADDED_DLL_DIRECTORIES.add(key)
            added.append(normalized)
        except OSError:
            continue
    return added


def _current_path_entries() -> list[Path]:
    paths: list[Path] = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry:
            paths.append(Path(entry))
    return paths
