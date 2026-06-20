from __future__ import annotations

import json
import re
import uuid
from typing import Any


TOOL_TAG_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
GEMMA_TOOL_TAG_RE = re.compile(
    r"<\|tool_call\|?>\s*call:([A-Za-z0-9_-]+)\s*(\{.*?\})\s*<\|?tool_call\|>",
    re.DOTALL,
)


def normalize_tools_for_responses(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return tools
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function":
            if "function" in tool:
                normalized.append(tool)
            else:
                normalized.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.get("name"),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                        },
                    }
                )
        else:
            normalized.append(tool)
    return normalized


def parse_tool_calls(text: str, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not text or not tools:
        return []

    parsed: list[dict[str, Any]] = []
    for match in TOOL_TAG_RE.finditer(text):
        call = _tool_call_from_json(match.group(1))
        if call:
            parsed.append(call)
    for match in GEMMA_TOOL_TAG_RE.finditer(text):
        call = _tool_call_from_gemma(match.group(1), match.group(2))
        if call:
            parsed.append(call)

    if parsed:
        return parsed

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        call = _tool_call_from_json(stripped)
        if call:
            return [call]
    return []


def remove_tool_call_markup(text: str) -> str:
    text = TOOL_TAG_RE.sub("", text)
    text = GEMMA_TOOL_TAG_RE.sub("", text)
    return text.strip()


def _tool_call_from_json(raw: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if "function" in payload:
        name = payload["function"].get("name")
        arguments = payload["function"].get("arguments", {})
    else:
        name = payload.get("name")
        arguments = payload.get("arguments", {})

    if not name:
        return None
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, separators=(",", ":"))

    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def _tool_call_from_gemma(name: str, raw_arguments: str) -> dict[str, Any] | None:
    normalized = raw_arguments.replace('<|"|>', '"')
    normalized = re.sub(r"([,{]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)", r'\1"\2"\3', normalized)
    try:
        arguments: Any = json.loads(normalized)
    except json.JSONDecodeError:
        arguments = {"_raw": raw_arguments}

    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, separators=(",", ":")),
        },
    }
