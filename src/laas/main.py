from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import uvicorn

from . import __version__
from .compat_check import run_compat_check
from .diagnostics import collect_diagnostics
from .manager import ModelManager
from .settings import load_settings
from .settings import Settings

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laas",
        description="Run the LAAS OpenAI-compatible local model API server.",
    )
    parser.add_argument("--host", help="Host interface to bind. Defaults to configured LAAS_HOST.")
    parser.add_argument("--port", type=int, help="Port to bind. Defaults to configured LAAS_PORT.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload mode for development.")
    parser.add_argument(
        "--yes-download",
        action="store_true",
        help="Download missing configured model files during startup without prompting.",
    )
    parser.add_argument(
        "--no-download-prompt",
        action="store_true",
        help="Skip the interactive missing-model download prompt during startup.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["diagnose", "compat-check"],
        help="Run a local command instead of starting the API server.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL for compat-check.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def missing_configured_model_paths(settings: Settings) -> list[Path]:
    """Return missing model paths. Kept as a list for future multi-model config."""
    return [path for _asset, path in ModelManager(settings).missing_required_paths()]


def confirm_missing_model_downloads(
    settings: Settings,
    *,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
    assume_yes: bool = False,
    prompt: bool = True,
) -> list[Path]:
    missing_assets = ModelManager(settings).missing_required_paths()
    missing_paths = [path for _asset, path in missing_assets]
    if not missing_paths:
        return []

    output_fn("Configured model assets are missing:")
    output_fn(f"  model:    {settings.model_id}")
    output_fn(f"  repo:     {settings.hf_repo_id}")
    output_fn(f"  filename: {settings.hf_filename}")
    output_fn(f"  path:     {settings.model_path} ({'present' if settings.model_path.exists() else 'missing'})")
    if settings.mmproj_filename:
        mmproj_path = settings.mmproj_path
        output_fn(f"  mmproj:   {settings.mmproj_filename}")
        output_fn(f"  mmrepo:   {settings.resolved_mmproj_repo_id}")
        output_fn(f"  mmpath:   {mmproj_path} ({'present' if mmproj_path and mmproj_path.exists() else 'missing'})")
    if settings.mtp_filename:
        mtp_path = settings.mtp_path
        output_fn(f"  mtp:      {settings.mtp_filename}")
        output_fn(f"  mtppath:  {mtp_path} ({'present' if mtp_path and mtp_path.exists() else 'missing'})")

    should_download = assume_yes
    if not should_download and prompt:
        answer = input_fn("Download missing model assets now? [y/N] ").strip().lower()
        should_download = answer in {"y", "yes"}

    if not should_download:
        output_fn("Skipping model download. The API will start, but inference will return model_not_downloaded until the model is downloaded.")
        output_fn("Download later with POST /v1/local/models/download, or restart with --yes-download.")
        return []

    output_fn("Downloading missing configured model assets...")
    downloaded = ModelManager(settings).download_configured_assets(include_mmproj=True, include_mtp=True)
    for path in downloaded:
        output_fn(f"Downloaded: {path}")
    return downloaded


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    if args.command == "diagnose":
        print(json.dumps(collect_diagnostics(settings), indent=2))
        return
    if args.command == "compat-check":
        report = run_compat_check(args.base_url)
        print(json.dumps(report, indent=2))
        if not report["ok"]:
            raise SystemExit(1)
        return
    confirm_missing_model_downloads(
        settings,
        assume_yes=args.yes_download,
        prompt=not args.no_download_prompt and sys.stdin.isatty(),
    )
    uvicorn.run(
        "laas.app:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
