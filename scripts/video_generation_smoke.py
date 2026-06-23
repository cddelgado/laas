from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a live LAAS Wan image-to-video smoke test.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="LAAS server base URL.")
    parser.add_argument("--image", type=Path, default=None, help="Input image. If omitted, a simple PNG is generated.")
    parser.add_argument("--output", type=Path, default=Path("wan-video-smoke.mp4"), help="Output MP4 path.")
    parser.add_argument("--prompt", default="a brass table lamp glowing softly in a quiet room")
    parser.add_argument("--size", default="832x480")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image_path = args.image or _write_default_image(Path(".laas") / "wan-smoke-frame.png")
    payload = _multipart(
        fields={
            "prompt": args.prompt,
            "size": args.size,
            "seconds": str(args.seconds),
            "fps": str(args.fps),
            "num_inference_steps": str(args.steps),
            "seed": str(args.seed),
            "response_format": "b64_json",
        },
        files={"image": image_path},
    )
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/videos/generations",
        data=payload["body"],
        headers={"Content-Type": payload["content_type"]},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(json.dumps({"ok": False, "status": exc.code, "error": body}, indent=2))
        return 1

    content = base64.b64decode(data["data"][0]["b64_json"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    print(json.dumps({"ok": True, "output": str(args.output), "bytes": len(content)}, indent=2))
    return 0


def _write_default_image(path: Path) -> Path:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (832, 480), (244, 239, 229))
    draw = ImageDraw.Draw(image)
    draw.rectangle((340, 160, 500, 300), fill=(174, 116, 58), outline=(90, 60, 35), width=6)
    draw.rectangle((365, 120, 475, 160), fill=(235, 211, 142), outline=(90, 60, 35), width=4)
    draw.line((420, 300, 420, 390), fill=(90, 60, 35), width=8)
    draw.ellipse((360, 385, 480, 425), fill=(90, 60, 35))
    image.save(path)
    return path


def _multipart(*, fields: dict[str, str], files: dict[str, Path]) -> dict[str, Any]:
    boundary = f"----laas-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("ascii"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    for key, path in files.items():
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                (
                    f'Content-Disposition: form-data; name="{key}"; filename="{path.name}"\r\n'
                    f"Content-Type: {content_type}\r\n\r\n"
                ).encode("ascii"),
                path.read_bytes(),
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    return {
        "body": b"".join(chunks),
        "content_type": f"multipart/form-data; boundary={boundary}",
    }


if __name__ == "__main__":
    raise SystemExit(main())
