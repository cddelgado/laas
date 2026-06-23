from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import site
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "unsloth/gemma-4-12b-it-GGUF"
DEFAULT_Q4_FILENAME = "gemma-4-12b-it-Q4_K_M.gguf"
DEFAULT_MTP_Q8_FILENAME = "MTP/gemma-4-12b-it-Q8_0-MTP.gguf"
DEFAULT_MMPROJ_FILENAME = "mmproj-F16.gguf"
_DLL_DIRECTORY_HANDLES: list[object] = []
_ADDED_DLL_DIRECTORIES: set[str] = set()


def default_model_dir() -> Path:
    if sys.platform.startswith("win"):
        return Path(r"D:\AI\Models")
    return Path.home() / "AI" / "Models"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download and smoke-test Gemma 4 12B Unified Q4_K_M and the Q8_0 MTP GGUF "
            "with llama-cpp-python. Gemma 4 12B Unified is encoder-free, so no projector "
            "is loaded unless --with-mmproj is set."
        )
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--model-dir", type=Path, default=default_model_dir())
    parser.add_argument("--q4-filename", default=DEFAULT_Q4_FILENAME)
    parser.add_argument("--mtp-filename", default=DEFAULT_MTP_Q8_FILENAME)
    parser.add_argument("--mmproj-filename", default=DEFAULT_MMPROJ_FILENAME)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--with-mmproj", action="store_true")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument(
        "--n-gpu-layers",
        default="auto",
        help="GPU layer setting. Use 'auto' to omit the parameter, 'all' for -1, or an integer.",
    )
    parser.add_argument(
        "--gpu-layer-candidates",
        default=None,
        help="Comma-separated n_gpu_layers values to try for the Q4 model, for example auto,all,40,32,24,16,8,0.",
    )
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--n-ubatch", type=int, default=512)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--n-threads-batch", type=int, default=None)
    parser.add_argument("--flash-attn", dest="flash_attn", action="store_true", default=True)
    parser.add_argument("--no-flash-attn", dest="flash_attn", action="store_false")
    parser.add_argument("--offload-kqv", dest="offload_kqv", action="store_true", default=True)
    parser.add_argument("--no-offload-kqv", dest="offload_kqv", action="store_false")
    parser.add_argument("--op-offload", dest="op_offload", action="store_true", default=None)
    parser.add_argument("--no-op-offload", dest="op_offload", action="store_false")
    parser.add_argument("--swa-full", dest="swa_full", action="store_true", default=None)
    parser.add_argument("--no-swa-full", dest="swa_full", action="store_false")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--prompt", default="Reply with exactly one short sentence: Gemma 4 12B is loaded.")
    parser.add_argument("--benchmark", action="store_true", help="Run baseline and prompt-lookup speculative tok/s checks.")
    parser.add_argument("--benchmark-runs", type=int, default=3)
    parser.add_argument("--benchmark-max-tokens", type=int, default=128)
    parser.add_argument(
        "--benchmark-modes",
        default="baseline,prompt_lookup",
        help="Comma-separated benchmark modes: baseline,prompt_lookup.",
    )
    parser.add_argument(
        "--benchmark-prompt",
        default=(
            "Write a compact numbered list of practical tips for running local LLM inference. "
            "Use repeated phrasing: tip title, one sentence, tip title, one sentence. "
            "Continue until the answer is complete."
        ),
    )
    parser.add_argument("--speculative-max-ngram-size", type=int, default=2)
    parser.add_argument("--speculative-num-pred-tokens", type=int, default=10)
    parser.add_argument("--verbose-llama", action="store_true")
    parser.add_argument(
        "--test-mtp-standalone",
        action="store_true",
        help="Also try to load the MTP GGUF as a primary model. This is expected to fail for MTP draft assets.",
    )
    args = parser.parse_args()

    print(json.dumps({"event": "probe_start", "repo_id": args.repo_id, "model_dir": str(args.model_dir)}))
    added_dll_directories = configure_native_dll_directories()
    if added_dll_directories:
        print(json.dumps({"event": "dll_directories_added", "paths": added_dll_directories}))
    support = llama_support()
    print(json.dumps({"event": "llama_cpp_support", **support}, default=str))

    failures = 0
    try:
        run_q4_probe(args)
    except Exception as exc:
        failures += 1
        print_probe_failure("q4_k_m", args.q4_filename, exc)
    finally:
        gc.collect()
        try_empty_torch_cache()

    try:
        mtp_path = resolve_asset(
            repo_id=args.repo_id,
            filename=args.mtp_filename,
            model_dir=args.model_dir,
            skip_download=args.skip_download,
        )
        print(
            json.dumps(
                {
                    "event": "mtp_asset_ready",
                    "label": "mtp_q8_0",
                    "path": str(mtp_path),
                    "bytes": mtp_path.stat().st_size,
                    "python_binding_external_mtp_supported": False,
                    "note": (
                        "The current llama-cpp-python binding exposes prompt-lookup speculative decoding, "
                        "but does not expose a Python LlamaDraftModel wrapper for an external MTP GGUF."
                    ),
                }
            )
        )
        if args.test_mtp_standalone:
            run_probe("mtp_q8_0_standalone", args.mtp_filename, args, parse_gpu_layer_value(args.n_gpu_layers))
        else:
            print(
                json.dumps(
                    {
                        "event": "mtp_standalone_skipped",
                        "label": "mtp_q8_0",
                        "reason": "MTP GGUF is an auxiliary/draft asset, not the primary chat model.",
                    }
                )
            )
    except Exception as exc:
        failures += 1
        print_probe_failure("mtp_q8_0", args.mtp_filename, exc)
    finally:
        gc.collect()
        try_empty_torch_cache()

    print(json.dumps({"event": "probe_done", "failures": failures}))
    return 1 if failures else 0


def run_q4_probe(args: argparse.Namespace) -> None:
    candidates = gpu_layer_candidates(args)
    last_exc: Exception | None = None
    for n_gpu_layers in candidates:
        try:
            run_probe("q4_k_m", args.q4_filename, args, n_gpu_layers)
            if args.benchmark:
                run_benchmarks(args, n_gpu_layers)
            print(json.dumps({"event": "q4_probe_selected", "n_gpu_layers": n_gpu_layers}))
            return
        except Exception as exc:
            last_exc = exc
            print_probe_failure("q4_k_m", args.q4_filename, exc, n_gpu_layers=n_gpu_layers)
            gc.collect()
            try_empty_torch_cache()
    assert last_exc is not None
    raise RuntimeError(f"Q4_K_M failed for all n_gpu_layers candidates: {candidates}") from last_exc


def run_probe(label: str, filename: str, args: argparse.Namespace, n_gpu_layers: int | None) -> None:
    print(json.dumps({"event": "asset_prepare_start", "label": label, "filename": filename}))
    model_path = resolve_asset(
        repo_id=args.repo_id,
        filename=filename,
        model_dir=args.model_dir,
        skip_download=args.skip_download,
    )
    mmproj_path = None
    if args.with_mmproj:
        mmproj_path = resolve_asset(
            repo_id=args.repo_id,
            filename=args.mmproj_filename,
            model_dir=args.model_dir,
            skip_download=args.skip_download,
        )
    print(
        json.dumps(
            {
                "event": "asset_prepare_done",
                "label": label,
                "model_path": str(model_path),
                "model_bytes": model_path.stat().st_size,
                "mmproj_path": str(mmproj_path) if mmproj_path else None,
                "mmproj_bytes": mmproj_path.stat().st_size if mmproj_path else None,
            }
        )
    )

    configure_native_dll_directories()
    from llama_cpp import Llama

    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": args.n_ctx,
        "verbose": args.verbose_llama,
    }
    if n_gpu_layers is not None:
        kwargs["n_gpu_layers"] = n_gpu_layers
    add_if_supported(Llama, kwargs, "n_batch", args.n_batch)
    add_if_supported(Llama, kwargs, "n_ubatch", args.n_ubatch)
    add_if_supported(Llama, kwargs, "n_threads", args.n_threads)
    add_if_supported(Llama, kwargs, "n_threads_batch", args.n_threads_batch)
    add_if_supported(Llama, kwargs, "flash_attn", args.flash_attn)
    add_if_supported(Llama, kwargs, "offload_kqv", args.offload_kqv)
    add_if_supported(Llama, kwargs, "op_offload", args.op_offload)
    add_if_supported(Llama, kwargs, "swa_full", args.swa_full)
    if mmproj_path:
        add_mmproj_kwargs(Llama, kwargs, mmproj_path, verbose=args.verbose_llama, use_gpu=args.n_gpu_layers != 0)

    load_started = time.perf_counter()
    print(json.dumps({"event": "load_start", "label": label, "kwargs": scrub_kwargs(kwargs)}))
    llm = Llama(**kwargs)
    load_seconds = time.perf_counter() - load_started
    print(json.dumps({"event": "load_done", "label": label, "seconds": round(load_seconds, 3)}))

    infer_started = time.perf_counter()
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    infer_seconds = time.perf_counter() - infer_started
    text = response["choices"][0]["message"].get("content", "")
    usage = response.get("usage", {})
    print(
        json.dumps(
            {
                "event": "inference_done",
                "label": label,
                "seconds": round(infer_seconds, 3),
                "text": text,
                "usage": usage,
            }
        )
    )

    del llm
    gc.collect()
    try_empty_torch_cache()
    print(json.dumps({"event": "unloaded", "label": label}))


def run_benchmarks(args: argparse.Namespace, n_gpu_layers: int | None) -> None:
    modes = [mode.strip() for mode in args.benchmark_modes.split(",") if mode.strip()]
    for mode in modes:
        run_benchmark_mode(args, n_gpu_layers, mode)
        gc.collect()
        try_empty_torch_cache()


def run_benchmark_mode(args: argparse.Namespace, n_gpu_layers: int | None, mode: str) -> None:
    model_path = resolve_asset(
        repo_id=args.repo_id,
        filename=args.q4_filename,
        model_dir=args.model_dir,
        skip_download=args.skip_download,
    )

    configure_native_dll_directories()
    from llama_cpp import Llama

    kwargs = llama_constructor_kwargs(Llama, model_path, args, n_gpu_layers)
    if mode == "prompt_lookup":
        add_prompt_lookup_speculative_kwargs(
            kwargs,
            max_ngram_size=args.speculative_max_ngram_size,
            num_pred_tokens=args.speculative_num_pred_tokens,
        )
    elif mode != "baseline":
        print(json.dumps({"event": "benchmark_skipped", "mode": mode, "reason": "unsupported benchmark mode"}))
        return

    print(json.dumps({"event": "benchmark_load_start", "mode": mode, "kwargs": scrub_kwargs(kwargs)}))
    load_started = time.perf_counter()
    llm = Llama(**kwargs)
    print(json.dumps({"event": "benchmark_load_done", "mode": mode, "seconds": round(time.perf_counter() - load_started, 3)}))

    measurements: list[dict[str, Any]] = []
    for run_index in range(args.benchmark_runs):
        started = time.perf_counter()
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": args.benchmark_prompt}],
            max_tokens=args.benchmark_max_tokens,
            temperature=args.temperature,
        )
        seconds = time.perf_counter() - started
        usage = response.get("usage", {})
        completion_tokens = int(usage.get("completion_tokens") or 0)
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        tokens_per_second = completion_tokens / seconds if seconds > 0 else 0.0
        text = response["choices"][0]["message"].get("content", "")
        measurement = {
            "event": "benchmark_run",
            "mode": mode,
            "run": run_index + 1,
            "seconds": round(seconds, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "completion_tokens_per_second": round(tokens_per_second, 3),
            "text_preview": text[:160],
        }
        measurements.append(measurement)
        print(json.dumps(measurement))

    rates = [item["completion_tokens_per_second"] for item in measurements if item["completion_tokens"]]
    summary = {
        "event": "benchmark_summary",
        "mode": mode,
        "runs": len(measurements),
        "avg_completion_tokens_per_second": round(sum(rates) / len(rates), 3) if rates else 0.0,
        "best_completion_tokens_per_second": round(max(rates), 3) if rates else 0.0,
        "min_completion_tokens_per_second": round(min(rates), 3) if rates else 0.0,
    }
    print(json.dumps(summary))

    del llm
    gc.collect()
    try_empty_torch_cache()
    print(json.dumps({"event": "benchmark_unloaded", "mode": mode}))


def llama_constructor_kwargs(
    llama_cls: Any,
    model_path: Path,
    args: argparse.Namespace,
    n_gpu_layers: int | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": args.n_ctx,
        "verbose": args.verbose_llama,
    }
    if n_gpu_layers is not None:
        kwargs["n_gpu_layers"] = n_gpu_layers
    add_if_supported(llama_cls, kwargs, "n_batch", args.n_batch)
    add_if_supported(llama_cls, kwargs, "n_ubatch", args.n_ubatch)
    add_if_supported(llama_cls, kwargs, "n_threads", args.n_threads)
    add_if_supported(llama_cls, kwargs, "n_threads_batch", args.n_threads_batch)
    add_if_supported(llama_cls, kwargs, "flash_attn", args.flash_attn)
    add_if_supported(llama_cls, kwargs, "offload_kqv", args.offload_kqv)
    add_if_supported(llama_cls, kwargs, "op_offload", args.op_offload)
    add_if_supported(llama_cls, kwargs, "swa_full", args.swa_full)
    return kwargs


def add_prompt_lookup_speculative_kwargs(
    kwargs: dict[str, Any],
    *,
    max_ngram_size: int,
    num_pred_tokens: int,
) -> None:
    from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

    kwargs["draft_model"] = LlamaPromptLookupDecoding(
        max_ngram_size=max_ngram_size,
        num_pred_tokens=num_pred_tokens,
    )


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


def gpu_layer_candidates(args: argparse.Namespace) -> list[int | None]:
    if args.gpu_layer_candidates:
        return [parse_gpu_layer_value(value) for value in args.gpu_layer_candidates.split(",") if value.strip()]
    env_value = os.environ.get("LAAS_GEMMA4_GPU_LAYER_CANDIDATES")
    if env_value:
        return [parse_gpu_layer_value(value) for value in env_value.split(",") if value.strip()]
    return [parse_gpu_layer_value(args.n_gpu_layers)]


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


def llama_support() -> dict[str, Any]:
    try:
        configure_native_dll_directories()
        import llama_cpp
        from llama_cpp import Llama
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    signature = inspect.signature(Llama)
    parameters = signature.parameters
    support: dict[str, Any] = {
        "available": True,
        "version": getattr(llama_cpp, "__version__", "unknown"),
        "n_batch": "n_batch" in parameters,
        "n_ubatch": "n_ubatch" in parameters,
        "n_threads_batch": "n_threads_batch" in parameters,
        "draft_model": "draft_model" in parameters,
        "flash_attn": "flash_attn" in parameters,
        "offload_kqv": "offload_kqv" in parameters,
        "op_offload": "op_offload" in parameters,
        "swa_full": "swa_full" in parameters,
        "mmproj": "mmproj" in parameters,
        "mmproj_path": "mmproj_path" in parameters,
        "clip_model_path": "clip_model_path" in parameters,
        "chat_handler": "chat_handler" in parameters,
    }
    try:
        from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

        support["prompt_lookup_speculative"] = True
        support["prompt_lookup_signature"] = str(inspect.signature(LlamaPromptLookupDecoding))
    except Exception as exc:
        support["prompt_lookup_speculative"] = False
        support["prompt_lookup_error"] = str(exc)
    return support


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


def print_probe_failure(label: str, filename: str, exc: Exception, *, n_gpu_layers: int | None = None) -> None:
    payload: dict[str, Any] = {
        "event": "probe_failed",
        "label": label,
        "filename": filename,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    if n_gpu_layers is not None:
        payload["n_gpu_layers"] = n_gpu_layers
    print(json.dumps(payload))


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
    scrubbed: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key == "chat_handler":
            scrubbed[key] = type(value).__name__
        elif key == "draft_model":
            scrubbed[key] = type(value).__name__
        else:
            scrubbed[key] = value
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
