from __future__ import annotations

import argparse
import asyncio
import base64
import json
import urllib.request
from pathlib import Path
from urllib.parse import urlparse, urlunparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test LAAS OpenAI-shaped realtime voice events.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="LAAS server URL, with or without /v1.")
    parser.add_argument("--api-key", default="laas-local", help="Dummy API key accepted by local OpenAI clients.")
    parser.add_argument("--tts-model", default="tts-1")
    parser.add_argument("--voice", default="alloy")
    parser.add_argument("--response-format", default="wav", choices=["pcm", "wav", "mp3", "flac", "opus", "aac"])
    parser.add_argument("--input-text", default="Say hello from the realtime smoke test.")
    parser.add_argument("--instructions", default="Answer in one short sentence.")
    parser.add_argument("--output", default="realtime-smoke-output.wav", help="Path for returned assistant audio.")
    args = parser.parse_args()

    base_url = openai_base_url(args.base_url)
    input_audio = synthesize_input_audio(base_url, args)
    session = post_json(
        base_url,
        "/realtime/sessions",
        {
            "instructions": args.instructions,
            "voice": args.voice,
            "response_format": args.response_format,
        },
        api_key=args.api_key,
    )
    try:
        asyncio.run(run_realtime_turn(base_url, session["id"], input_audio, Path(args.output)))
    finally:
        close_session(base_url, session["id"], api_key=args.api_key)
    print(f"realtime voice smoke completed: {args.output}")
    return 0


def openai_base_url(value: str) -> str:
    value = value.rstrip("/")
    return value if value.endswith("/v1") else f"{value}/v1"


def websocket_url(base_url: str, path: str) -> str:
    parsed = urlparse(f"{base_url.rstrip('/')}{path}")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))


def synthesize_input_audio(base_url: str, args: argparse.Namespace) -> bytes:
    payload = {
        "model": args.tts_model,
        "voice": args.voice,
        "input": args.input_text,
        "response_format": "wav",
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/audio/speech",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        content = response.read()
    if not content:
        raise RuntimeError("/v1/audio/speech returned empty audio")
    return content


def post_json(base_url: str, path: str, payload: dict, *, api_key: str) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def close_session(base_url: str, session_id: str, *, api_key: str) -> None:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/local/voice/sessions/{session_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        method="DELETE",
    )
    try:
        urllib.request.urlopen(request, timeout=10).close()
    except Exception:
        pass


async def run_realtime_turn(base_url: str, session_id: str, input_audio: bytes, output_path: Path) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("websockets is required for this smoke test; install uvicorn[standard].") from exc

    ws_url = websocket_url(base_url, f"/realtime/sessions/{session_id}")
    audio_chunks: list[bytes] = []
    final_audio: bytes | None = None
    response_id: str | None = None
    item_id: str | None = None
    saw_types: list[str] = []

    async with websockets.connect(ws_url, open_timeout=30, close_timeout=10) as websocket:
        created = json.loads(await websocket.recv())
        assert_event(created, "session.created")
        assert created["session"]["object"] == "realtime.session"

        await websocket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(input_audio).decode("ascii"),
                }
            )
        )
        assert_event(json.loads(await websocket.recv()), "input_audio_buffer.appended")

        await websocket.send(json.dumps({"type": "response.create", "filename": "realtime-smoke-input.wav"}))
        while True:
            event = json.loads(await websocket.recv())
            event_type = event.get("type")
            saw_types.append(event_type)
            if event_type == "response.created":
                response_id = event["response"]["id"]
                continue
            if event_type == "response.output_item.added":
                assert event["response_id"] == response_id
                item_id = event["item"]["id"]
                continue
            if event_type == "response.output_text.delta":
                assert event["response_id"] == response_id
                assert event["item_id"] == item_id
                assert isinstance(event.get("delta"), str)
                continue
            if event_type == "response.output_text.done":
                assert event["response_id"] == response_id
                assert event["item_id"] == item_id
                continue
            if event_type == "response.audio.delta":
                assert event["response_id"] == response_id
                assert event["item_id"] == item_id
                audio_chunks.append(base64.b64decode(event["delta"]))
                continue
            if event_type == "response.audio.done":
                assert event["response_id"] == response_id
                assert event["item_id"] == item_id
                continue
            if event_type == "response.output_item.done":
                assert event["response_id"] == response_id
                assert event["item"]["id"] == item_id
                continue
            if event_type == "response.completed":
                assert event["response"]["id"] == response_id
                content = event["response"]["output"][0]["content"]
                final_audio = base64.b64decode(content[1]["audio"])
                break
            raise AssertionError(f"Unexpected realtime event: {event}")

        await websocket.send(json.dumps({"type": "session.close"}))
        assert_event(json.loads(await websocket.recv()), "session.closed")

    required = {
        "response.created",
        "response.output_item.added",
        "response.audio.delta",
        "response.audio.done",
        "response.output_item.done",
        "response.completed",
    }
    missing = required - set(saw_types)
    if missing:
        raise AssertionError(f"Missing realtime event types: {sorted(missing)}")
    streamed_audio = b"".join(audio_chunks)
    if not streamed_audio:
        raise AssertionError("Realtime response did not include audio deltas")
    if final_audio is not None and streamed_audio != final_audio:
        raise AssertionError("Audio deltas did not reconstruct final response audio")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(streamed_audio)


def assert_event(event: dict, expected_type: str) -> None:
    actual = event.get("type")
    if actual != expected_type:
        raise AssertionError(f"Expected {expected_type}, received {actual}: {event}")


if __name__ == "__main__":
    raise SystemExit(main())
