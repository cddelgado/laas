from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def run_compat_check(base_url: str, *, timeout: float = 10.0) -> dict[str, Any]:
    base = base_url.rstrip("/")
    probes = [
        ("health", "GET", "/health", None, 200),
        ("models.list", "GET", "/v1/models", None, 200),
        ("compatibility", "GET", "/v1/local/compatibility", None, 200),
        ("files.list", "GET", "/v1/files", None, 200),
        ("vector_stores.list", "GET", "/v1/vector_stores", None, 200),
        ("batches.list", "GET", "/v1/batches", None, 200),
        ("moderations.create", "POST", "/v1/moderations", {"input": "hello"}, 200),
    ]
    results = []
    for name, method, path, body, expected_status in probes:
        status, payload = _request(base + path, method=method, body=body, timeout=timeout)
        results.append(
            {
                "name": name,
                "method": method,
                "path": path,
                "status_code": status,
                "ok": status == expected_status,
                "expected_status": expected_status,
                "error": payload if status >= 400 else None,
            }
        )
    return {
        "object": "local.compat_check",
        "base_url": base,
        "ok": all(item["ok"] for item in results),
        "results": results,
    }


def _request(url: str, *, method: str, body: dict[str, Any] | None, timeout: float) -> tuple[int, Any]:
    data = None
    headers = {"Authorization": "Bearer laas-local"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read()
            return response.status, _decode(content)
    except urllib.error.HTTPError as exc:
        return exc.code, _decode(exc.read())
    except Exception as exc:
        return 0, {"message": str(exc)}


def _decode(content: bytes) -> Any:
    if not content:
        return None
    try:
        return json.loads(content.decode("utf-8"))
    except Exception:
        return content.decode("utf-8", errors="replace")
