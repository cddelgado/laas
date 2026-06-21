from __future__ import annotations

import json
import re
import uuid
from typing import Any


TOOL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
GEMMA_TOOL_TAG_RE = re.compile(
    r"<\|tool_call\|?>\s*(.*?)\s*(?:<\|/tool_call\|>|<\|?tool_call\|?>)",
    re.DOTALL,
)
GEMMA_CALL_RE = re.compile(r"^\s*call:([A-Za-z0-9_-]+)\s*(.*)\s*$", re.DOTALL)


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


def tool_name(tool: dict[str, Any]) -> str | None:
    function = tool.get("function")
    if isinstance(function, dict):
        name = function.get("name")
    else:
        name = tool.get("name")
    return name if isinstance(name, str) and name else None


def tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    return {name for tool in tools or [] if (name := tool_name(tool))}


def selected_tool_name(tool_choice: Any) -> str | None:
    if not isinstance(tool_choice, dict):
        return None
    function = tool_choice.get("function")
    if isinstance(function, dict):
        name = function.get("name")
    else:
        name = tool_choice.get("name")
    return name if isinstance(name, str) and name else None


def validate_tool_choice(tool_choice: Any, tools: list[dict[str, Any]] | None) -> None:
    names = tool_names(tools)
    if tool_choice in (None, "auto", "none"):
        return
    if tool_choice == "required":
        if not names:
            raise ValueError("tool_choice 'required' requires at least one tool")
        return
    if isinstance(tool_choice, str):
        raise ValueError("tool_choice must be 'none', 'auto', 'required', or a function tool choice object")
    selected = selected_tool_name(tool_choice)
    if not selected:
        raise ValueError("function tool_choice must include a function name")
    if selected not in names:
        raise ValueError(f"tool_choice references unknown function '{selected}'")


def parse_tool_calls(
    text: str,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any = "auto",
) -> list[dict[str, Any]]:
    if not text or not tools:
        return []

    allowed_names = _allowed_tool_names(tools, tool_choice)
    if not allowed_names:
        return []

    parsed: list[dict[str, Any]] = []
    for match in TOOL_TAG_RE.finditer(text):
        parsed.extend(_tool_calls_from_payload(match.group(1)))
        parsed.extend(_tool_calls_from_gemma_body(match.group(1)))
    for match in GEMMA_TOOL_TAG_RE.finditer(text):
        parsed.extend(_tool_calls_from_payload(match.group(1)))
        parsed.extend(_tool_calls_from_gemma_body(match.group(1)))

    if not parsed:
        parsed.extend(_tool_calls_from_payload(text.strip()))

    return [call for call in parsed if call.get("function", {}).get("name") in allowed_names]


def remove_tool_call_markup(text: str) -> str:
    text = TOOL_TAG_RE.sub("", text)
    text = GEMMA_TOOL_TAG_RE.sub("", text)
    return text.strip()


def _allowed_tool_names(tools: list[dict[str, Any]], tool_choice: Any) -> set[str]:
    names = tool_names(tools)
    if tool_choice == "none":
        return set()
    selected = selected_tool_name(tool_choice)
    if selected:
        return {selected} & names
    return names


def _tool_calls_from_payload(raw: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(payload, list):
        calls: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict) and (call := _normalize_tool_call(item)):
                calls.append(call)
        return calls
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("tool_calls"), list):
        calls: list[dict[str, Any]] = []
        for item in payload["tool_calls"]:
            if isinstance(item, dict) and (call := _normalize_tool_call(item)):
                calls.append(call)
        return calls
    if isinstance(payload.get("function_call"), dict):
        call = _normalize_tool_call(payload["function_call"])
        return [call] if call else []
    call = _normalize_tool_call(payload)
    return [call] if call else []


def _tool_calls_from_gemma_body(body: str) -> list[dict[str, Any]]:
    match = GEMMA_CALL_RE.match(body)
    if not match:
        return []
    raw_arguments = _first_balanced_object(match.group(2))
    if raw_arguments is None:
        raw_arguments = match.group(2).strip() or "{}"
    call = _tool_call_from_gemma(match.group(1), raw_arguments)
    return [call] if call else []


def _normalize_tool_call(payload: dict[str, Any]) -> dict[str, Any] | None:
    if "function" in payload and isinstance(payload["function"], dict):
        name = payload["function"].get("name")
        arguments = payload["function"].get("arguments", {})
    else:
        name = payload.get("name")
        arguments = payload.get("arguments", {})

    if not isinstance(name, str) or not name:
        return None
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, separators=(",", ":"))

    return {
        "id": payload.get("id") or f"call_{uuid.uuid4().hex[:24]}",
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


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
