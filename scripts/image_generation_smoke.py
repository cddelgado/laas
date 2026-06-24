from __future__ import annotations

import argparse
import base64
import json
import urllib.error
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a live LAAS image generation smoke test.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="LAAS server base URL.")
    parser.add_argument("--output", type=Path, default=Path("sdxl-smoke.png"), help="Output image path.")
    parser.add_argument("--model", default="sdxl-turbo")
    parser.add_argument("--prompt", default="a cinematic photo of a tiny robot repairing a neon sign")
    parser.add_argument("--size", default="768x768")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--quality", default=None)
    parser.add_argument("--style", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    args = parser.parse_args()

    body: dict[str, object] = {
        "model": args.model,
        "prompt": args.prompt,
        "size": args.size,
        "n": args.n,
        "response_format": "b64_json",
        "seed": args.seed,
    }
    if args.quality:
        body["quality"] = args.quality
    if args.style:
        body["style"] = args.style
    if args.steps is not None:
        body["num_inference_steps"] = args.steps
    if args.guidance_scale is not None:
        body["guidance_scale"] = args.guidance_scale

    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/images/generations",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(json.dumps({"ok": False, "status": exc.code, "error": error_body}, indent=2))
        return 1
    except urllib.error.URLError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1

    content = base64.b64decode(data["data"][0]["b64_json"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    print(json.dumps({"ok": True, "output": str(args.output), "bytes": len(content)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
