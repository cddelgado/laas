from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import site
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "ggml-org/gemma-4-E4B-it-GGUF"
DEFAULT_FILENAME = "gemma-4-E4B-it-Q4_K_M.gguf"
DEFAULT_MMPROJ_FILENAME = "mmproj-gemma-4-E4B-it-Q8_0.gguf"
DEFAULT_CONTEXTS = "8192,16384,32768,49152,65536,98304,131072"
DEFAULT_BATCHES = "1024,512,256,128"
DEFAULT_GPU_LAYERS = "all"
DEFAULT_PROMPT = (
    "Write a concise technical explanation of how to balance context length, KV cache, "
    "GPU layer offload, and batch size for local LLM inference. Use concrete tradeoffs."
)
_DLL_DIRECTORY_HANDLES: list[object] = []
_ADDED_DLL_DIRECTORIES: set[str] = set()


def default_model_dir() -> Path:
    if sys.platform.startswith("win"):
        return Path(r"D:\AI\Models")
    return Path.home() / "AI" / "Models"


def default_output_dir() -> Path:
    if sys.platform.startswith("win"):
        return Path(r"D:\AI\Benchmarks")
    return Path.home() / "AI" / "Benchmarks"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Find the largest practical Gemma 4 E4B context that remains snappy on this machine. "
            "The tuner tests context, batch, ubatch, and GPU-layer profiles with "
            "llama-cpp-python and prints a copy/paste LAAS .env recommendation."
        )
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--mmproj-filename", default=DEFAULT_MMPROJ_FILENAME)
    parser.add_argument(
        "--with-mmproj",
        dest="with_mmproj",
        action="store_true",
        default=True,
        help="Load the configured projector during tuning to account for multimodal memory overhead.",
    )
    parser.add_argument(
        "--without-mmproj",
        dest="with_mmproj",
        action="store_false",
        help="Tune text-only context without loading the configured projector.",
    )
    parser.add_argument("--model-dir", type=Path, default=default_model_dir())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--contexts", default=DEFAULT_CONTEXTS, help="Comma-separated context sizes to test.")
    parser.add_argument("--batches", default=DEFAULT_BATCHES, help="Comma-separated n_batch/n_ubatch values to test.")
    parser.add_argument(
        "--gpu-layers",
        default=DEFAULT_GPU_LAYERS,
        help="Comma-separated layer settings: all, auto, 40, 32, 24, 16, 8, 0.",
    )
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--min-tokens-per-second", type=float, default=20.0)
    parser.add_argument("--max-load-seconds", type=float, default=60.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--flash-attn", dest="flash_attn", action="store_true", default=True)
    parser.add_argument("--no-flash-attn", dest="flash_attn", action="store_false")
    parser.add_argument("--offload-kqv", dest="offload_kqv", action="store_true", default=True)
    parser.add_argument("--no-offload-kqv", dest="offload_kqv", action="store_false")
    parser.add_argument("--op-offload", dest="op_offload", action="store_true", default=None)
    parser.add_argument("--no-op-offload", dest="op_offload", action="store_false")
    parser.add_argument("--swa-full", dest="swa_full", action="store_true", default=None)
    parser.add_argument("--no-swa-full", dest="swa_full", action="store_false")
    parser.add_argument(
        "--stop-after-first-slow",
        action="store_true",
        help="Stop once a context has no profile meeting the speed/load threshold.",
    )
    parser.add_argument("--verbose-llama", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"laas-gemma4-context-tune-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"

    def emit(payload: dict[str, Any]) -> None:
        line = json.dumps(payload, default=str)
        print(line)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    emit(
        {
            "event": "context_tune_start",
            "repo_id": args.repo_id,
            "filename": args.filename,
            "mmproj_filename": args.mmproj_filename,
            "with_mmproj": args.with_mmproj,
            "model_dir": str(args.model_dir),
            "output_path": str(output_path),
            "contexts": parse_int_list(args.contexts),
            "batches": parse_int_list(args.batches),
            "gpu_layers": [value.strip() for value in args.gpu_layers.split(",") if value.strip()],
            "min_tokens_per_second": args.min_tokens_per_second,
        }
    )

    added_dll_dirs = configure_native_dll_directories()
    if added_dll_dirs:
        emit({"event": "dll_directories_added", "paths": added_dll_dirs})

    try:
        from llama_cpp import Llama
    except Exception as exc:
        emit({"event": "context_tune_failed", "error": f"llama-cpp-python import failed: {exc}"})
        return 1

    model_path = resolve_asset(
        repo_id=args.repo_id,
        filename=args.filename,
        model_dir=args.model_dir,
        skip_download=args.skip_download,
    )
    emit({"event": "asset_ready", "model_path": str(model_path), "model_bytes": model_path.stat().st_size})
    mmproj_path = None
    if args.mmproj_filename:
        mmproj_path = resolve_asset(
            repo_id=args.repo_id,
            filename=args.mmproj_filename,
            model_dir=args.model_dir,
            skip_download=args.skip_download,
        )
        emit(
            {
                "event": "asset_ready",
                "asset": "mmproj",
                "mmproj_path": str(mmproj_path),
                "mmproj_bytes": mmproj_path.stat().st_size,
                "loaded_during_tune": args.with_mmproj,
            }
        )

    contexts = sorted(parse_int_list(args.contexts))
    batches = parse_int_list(args.batches)
    gpu_layer_values = [parse_gpu_layer_value(value) for value in args.gpu_layers.split(",") if value.strip()]

    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_available: dict[str, Any] | None = None
    saw_slow_at_full_context = False

    for n_ctx in contexts:
        context_had_snappy_profile = False
        for n_gpu_layers in gpu_layer_values:
            for batch in batches:
                result = run_profile(
                    llama_cls=Llama,
                    model_path=model_path,
                    mmproj_path=mmproj_path if args.with_mmproj else None,
                    n_ctx=n_ctx,
                    n_batch=batch,
                    n_ubatch=batch,
                    n_gpu_layers=n_gpu_layers,
                    args=args,
                )
                results.append(result)
                emit(result)

                if not result.get("ok"):
                    continue
                if is_better_profile(result, best_available):
                    best_available = result
                if float(result["load_seconds"]) > args.max_load_seconds:
                    continue
                if float(result["avg_completion_tokens_per_second"]) < args.min_tokens_per_second:
                    continue

                context_had_snappy_profile = True
                if is_better_profile(result, best):
                    best = result

        if not context_had_snappy_profile:
            saw_slow_at_full_context = True
            emit(
                {
                    "event": "context_not_snappy",
                    "n_ctx": n_ctx,
                    "min_tokens_per_second": args.min_tokens_per_second,
                    "note": "No tested profile for this context met the speed/load thresholds.",
                }
            )
            if args.stop_after_first_slow:
                break

    recommendation = build_recommendation(best, best_available, output_path, args)
    emit(recommendation)
    emit(
        {
            "event": "context_tune_done",
            "tested_profiles": len(results),
            "found_recommendation": best is not None,
            "stopped_after_slow_context": saw_slow_at_full_context and args.stop_after_first_slow,
            "output_path": str(output_path),
        }
    )
    return 0 if best else 2


def run_profile(
    *,
    llama_cls: Any,
    model_path: Path,
    mmproj_path: Path | None,
    n_ctx: int,
    n_batch: int,
    n_ubatch: int,
    n_gpu_layers: int | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    kwargs = llama_constructor_kwargs(
        llama_cls=llama_cls,
        model_path=model_path,
        mmproj_path=mmproj_path,
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_ubatch=n_ubatch,
        n_gpu_layers=n_gpu_layers,
        args=args,
    )
    profile = {
        "event": "context_profile",
        "n_ctx": n_ctx,
        "n_batch": n_batch,
        "n_ubatch": n_ubatch,
        "n_gpu_layers": n_gpu_layers,
        "kwargs": scrub_kwargs(kwargs),
    }

    llm = None
    try:
        started = time.perf_counter()
        llm = llama_cls(**kwargs)
        load_seconds = time.perf_counter() - started

        measurements: list[dict[str, Any]] = []
        for run_index in range(args.runs):
            run_started = time.perf_counter()
            response = llm.create_chat_completion(
                messages=[{"role": "user", "content": args.prompt}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            seconds = time.perf_counter() - run_started
            usage = response.get("usage", {})
            completion_tokens = int(usage.get("completion_tokens") or 0)
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            measurements.append(
                {
                    "run": run_index + 1,
                    "seconds": round(seconds, 3),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "completion_tokens_per_second": round(completion_tokens / seconds, 3) if seconds > 0 else 0.0,
                }
            )

        rates = [item["completion_tokens_per_second"] for item in measurements if item["completion_tokens"]]
        profile.update(
            {
                "ok": True,
                "load_seconds": round(load_seconds, 3),
                "runs": measurements,
                "avg_completion_tokens_per_second": round(sum(rates) / len(rates), 3) if rates else 0.0,
                "best_completion_tokens_per_second": round(max(rates), 3) if rates else 0.0,
                "min_completion_tokens_per_second": round(min(rates), 3) if rates else 0.0,
            }
        )
    except Exception as exc:
        profile.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    finally:
        if llm is not None:
            del llm
        gc.collect()
        try_empty_torch_cache()
    return profile


def llama_constructor_kwargs(
    *,
    llama_cls: Any,
    model_path: Path,
    mmproj_path: Path | None,
    n_ctx: int,
    n_batch: int,
    n_ubatch: int,
    n_gpu_layers: int | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": n_ctx,
        "verbose": args.verbose_llama,
    }
    if n_gpu_layers is not None:
        kwargs["n_gpu_layers"] = n_gpu_layers
    add_if_supported(llama_cls, kwargs, "n_batch", n_batch)
    add_if_supported(llama_cls, kwargs, "n_ubatch", n_ubatch)
    add_if_supported(llama_cls, kwargs, "flash_attn", args.flash_attn)
    add_if_supported(llama_cls, kwargs, "offload_kqv", args.offload_kqv)
    add_if_supported(llama_cls, kwargs, "op_offload", args.op_offload)
    add_if_supported(llama_cls, kwargs, "swa_full", args.swa_full)
    if mmproj_path:
        add_mmproj_kwargs(llama_cls, kwargs, mmproj_path, verbose=args.verbose_llama, use_gpu=n_gpu_layers != 0)
    return kwargs


def is_better_profile(candidate: dict[str, Any], current: dict[str, Any] | None) -> bool:
    if current is None:
        return True
    candidate_key = (
        int(candidate["n_ctx"]),
        float(candidate["avg_completion_tokens_per_second"]),
        int(candidate["n_batch"]),
    )
    current_key = (
        int(current["n_ctx"]),
        float(current["avg_completion_tokens_per_second"]),
        int(current["n_batch"]),
    )
    return candidate_key > current_key


def build_recommendation(
    best: dict[str, Any] | None,
    best_available: dict[str, Any] | None,
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if best is None:
        payload = {
            "event": "context_tune_recommendation",
            "ok": False,
            "output_path": str(output_path),
            "note": "No tested profile met the requested speed and load thresholds.",
        }
        if best_available is not None:
            payload["best_available_profile"] = summarize_profile(best_available)
        return payload

    env = {
        "LAAS_N_CTX": str(best["n_ctx"]),
        "LAAS_N_GPU_LAYERS": gpu_layer_env_value(best["n_gpu_layers"]),
        "LAAS_N_BATCH": str(best["n_batch"]),
        "LAAS_N_UBATCH": str(best["n_ubatch"]),
        "LAAS_FLASH_ATTN": str(args.flash_attn).lower(),
        "LAAS_OFFLOAD_KQV": str(args.offload_kqv).lower(),
        "LAAS_SPECULATIVE_DECODING": "false",
    }
    if args.mmproj_filename:
        env["LAAS_MMPROJ_FILENAME"] = args.mmproj_filename
        env["LAAS_MMPROJ_REQUIRED"] = "true"
    return {
        "event": "context_tune_recommendation",
        "ok": True,
        "profile": summarize_profile(best),
        "env": env,
        "env_text": "\n".join(f"{key}={value}" for key, value in env.items()),
        "output_path": str(output_path),
    }


def summarize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_ctx": profile["n_ctx"],
        "n_gpu_layers": profile["n_gpu_layers"],
        "n_batch": profile["n_batch"],
        "n_ubatch": profile["n_ubatch"],
        "load_seconds": profile["load_seconds"],
        "avg_completion_tokens_per_second": profile["avg_completion_tokens_per_second"],
        "min_completion_tokens_per_second": profile["min_completion_tokens_per_second"],
    }


def resolve_asset(*, repo_id: str, filename: str, model_dir: Path, skip_download: bool) -> Path:
    local_dir = model_dir / repo_id.replace("/", "__")
    local_path = local_dir / filename
    if local_path.exists():
        return local_path
    if skip_download:
        raise FileNotFoundError(local_path)

    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir))


def configure_native_dll_directories() -> list[str]:
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
    existing = {entry.lower() for entry in os.environ.get("PATH", "").split(os.pathsep) if entry}
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


def parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one integer value.")
    return parsed


def parse_gpu_layer_value(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    normalized = value.strip().lower()
    if normalized in {"", "auto", "default", "none"}:
        return None
    if normalized in {"all", "full", "-1"}:
        return -1
    return int(normalized)


def gpu_layer_env_value(value: Any) -> str:
    if value is None:
        return ""
    if int(value) == -1:
        return "-1"
    return str(value)


def add_if_supported(llama_cls: Any, kwargs: dict[str, Any], name: str, value: Any) -> None:
    if value is None:
        return
    if name in inspect.signature(llama_cls).parameters:
        kwargs[name] = value


def add_mmproj_kwargs(
    llama_cls: Any,
    kwargs: dict[str, Any],
    mmproj_path: Path,
    *,
    verbose: bool,
    use_gpu: bool,
) -> None:
    parameters = inspect.signature(llama_cls).parameters
    if "mmproj" in parameters:
        kwargs["mmproj"] = str(mmproj_path)
        return
    if "mmproj_path" in parameters:
        kwargs["mmproj_path"] = str(mmproj_path)
        return
    if "clip_model_path" in parameters:
        kwargs["clip_model_path"] = str(mmproj_path)
        return
    if "chat_handler" in parameters:
        from llama_cpp.llama_chat_format import Gemma4ChatHandler

        kwargs["chat_handler"] = Gemma4ChatHandler(
            clip_model_path=str(mmproj_path),
            verbose=verbose,
            use_gpu=use_gpu,
        )
        return
    raise RuntimeError("Installed llama-cpp-python does not expose Gemma 4 projector support.")


def scrub_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    scrubbed = dict(kwargs)
    if "chat_handler" in scrubbed:
        scrubbed["chat_handler"] = type(scrubbed["chat_handler"]).__name__
    return scrubbed


def try_empty_torch_cache() -> None:
    try:
        import torch
    except Exception:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
