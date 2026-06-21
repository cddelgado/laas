from __future__ import annotations

import argparse
import concurrent.futures
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test LAAS heavy-model concurrency against a running server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="LAAS server URL, with or without /v1.")
    parser.add_argument("--text-model", default="gemma-4-e4b-it-q4_k_m")
    parser.add_argument("--image-model", default="sdxl-turbo")
    parser.add_argument("--image-edit-model", default="sd-1.5-inpainting")
    parser.add_argument("--prompt", default="a small brass table lamp, realistic lighting")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--include-image-edit", action="store_true", help="Also run /v1/images/edits.")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    with httpx.Client(base_url=base_url, timeout=args.timeout) as client:
        assert_status_ok(client)
        print_status(client, "initial")

        tasks = [
            ("chat", lambda: run_chat(client, args.text_model)),
            ("image", lambda: run_image_generation(client, args.image_model, args.prompt)),
        ]
        if args.include_image_edit:
            tasks.append(("image_edit", lambda: run_image_edit(client, args.image_edit_model, args.prompt)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {executor.submit(task): name for name, task in tasks}
            saw_active = poll_status_until_done(client, futures)
            for future, name in futures.items():
                result = future.result()
                print(f"{name}: {result}")

        print_status(client, "final")
        if not saw_active:
            print("warning: requests completed before status polling observed an active job")

    print("Concurrency smoke completed.")
    return 0


def assert_status_ok(client: httpx.Client) -> None:
    response = client.get("/health")
    response.raise_for_status()
    status = response.json()
    if status.get("status") != "ok":
        raise RuntimeError(f"unexpected health response: {status}")


def run_chat(client: httpx.Client, model: str) -> str:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Reply with one short sentence."}],
            "max_tokens": 32,
        },
    )
    response.raise_for_status()
    payload = response.json()
    finish_reason = payload["choices"][0].get("finish_reason")
    return f"finish_reason={finish_reason}"


def run_image_generation(client: httpx.Client, model: str, prompt: str) -> str:
    response = client.post(
        "/v1/images/generations",
        json={
            "model": model,
            "prompt": prompt,
            "size": "512x512",
            "response_format": "b64_json",
            "n": 1,
        },
    )
    response.raise_for_status()
    payload = response.json()
    return f"images={len(payload.get('data', []))}"


def run_image_edit(client: httpx.Client, model: str, prompt: str) -> str:
    base_path, mask_path = make_inpaint_pngs()
    try:
        with base_path.open("rb") as image, mask_path.open("rb") as mask:
            response = client.post(
                "/v1/images/edits",
                data={
                    "model": model,
                    "prompt": prompt,
                    "size": "512x512",
                    "response_format": "b64_json",
                    "n": "1",
                },
                files={
                    "image": ("base.png", image, "image/png"),
                    "mask": ("mask.png", mask, "image/png"),
                },
            )
    finally:
        base_path.unlink(missing_ok=True)
        mask_path.unlink(missing_ok=True)
    response.raise_for_status()
    payload = response.json()
    return f"edits={len(payload.get('data', []))}"


def poll_status_until_done(
    client: httpx.Client,
    futures: dict[concurrent.futures.Future[Any], str],
) -> bool:
    saw_active = False
    while any(not future.done() for future in futures):
        status = fetch_concurrency_status(client)
        total_active = int(status.get("total_active_jobs", 0))
        if total_active:
            saw_active = True
        print_compact_status(status)
        time.sleep(0.5)
    return saw_active


def print_status(client: httpx.Client, label: str) -> None:
    status = fetch_concurrency_status(client)
    print(f"{label}:")
    print_compact_status(status)


def fetch_concurrency_status(client: httpx.Client) -> dict[str, Any]:
    response = client.get("/v1/local/concurrency/status")
    response.raise_for_status()
    return response.json()


def print_compact_status(status: dict[str, Any]) -> None:
    active_jobs = status.get("active_jobs", {})
    loaded = {
        name: resource.get("is_loaded")
        for name, resource in status.get("resources", {}).items()
        if isinstance(resource, dict)
    }
    print(
        "  active_resource="
        f"{status.get('active_resource')} total_active_jobs={status.get('total_active_jobs')} "
        f"active_jobs={active_jobs} loaded={loaded}"
    )


def make_inpaint_pngs() -> tuple[Path, Path]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required for --include-image-edit; install image dependencies first.") from exc

    temp_dir = Path(tempfile.gettempdir())
    base_path = temp_dir / "laas-concurrency-smoke-base.png"
    mask_path = temp_dir / "laas-concurrency-smoke-mask.png"

    base = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(base)
    draw.rectangle((128, 160, 384, 360), fill=(185, 160, 120))
    draw.rectangle((230, 210, 282, 330), fill=(245, 245, 245))
    base.save(base_path)

    mask = Image.new("L", (512, 512), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((220, 200, 292, 340), fill=255)
    mask.save(mask_path)
    return base_path, mask_path


if __name__ == "__main__":
    raise SystemExit(main())
