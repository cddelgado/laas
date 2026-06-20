from __future__ import annotations

import argparse

import uvicorn

from . import __version__
from .settings import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laas",
        description="Run the LAAS OpenAI-compatible local model API server.",
    )
    parser.add_argument("--host", help="Host interface to bind. Defaults to configured LAAS_HOST.")
    parser.add_argument("--port", type=int, help="Port to bind. Defaults to configured LAAS_PORT.")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload mode for development.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    uvicorn.run(
        "laas.app:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
