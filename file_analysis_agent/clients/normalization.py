"""Defensive normalization helpers for OpenAI-shaped provider responses."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from file_analysis_agent.agent.models import LLMResponse, ToolCall, Usage
from file_analysis_agent.errors import ClientError


def normalize_chat_response(response: Any) -> LLMResponse:
    choices = _value(response, "choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        raise ClientError("chat completion response has no choices")
    choice = choices[0]
    message = _value(choice, "message")
    if message is None:
        raise ClientError("chat completion response is missing the assistant message")
    calls = _normalize_tool_calls(_value(message, "tool_calls"))
    legacy_call = _value(message, "function_call")
    if not calls and legacy_call is not None:
        calls = (_normalize_tool_call(legacy_call),)
    return LLMResponse(
        content=_text(_value(message, "content")),
        tool_calls=calls,
        finish_reason=_string(_value(choice, "finish_reason")),
        usage=_normalize_usage(_value(response, "usage")),
        raw_response=response,
    )


def normalize_responses_response(response: Any) -> LLMResponse:
    output = _value(response, "output")
    if output is None:
        output = []
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
        raise ClientError("responses response output is malformed")
    text_parts: list[str] = []
    calls: list[ToolCall] = []
    for item in output:
        item_type = _string(_value(item, "type"))
        if item_type == "function_call":
            call_id = _string(_value(item, "call_id")) or _string(_value(item, "id"))
            name = _string(_value(item, "name"))
            arguments = _arguments(_value(item, "arguments"))
            if not call_id or not name or arguments is None:
                raise ClientError("responses function call is missing id, name, or arguments")
            calls.append(ToolCall(call_id, name, arguments))
            continue
        if item_type in {"message", "output_text", "text"}:
            content = _value(item, "content")
            value = _text(content) or _text(_value(item, "text"))
            if value:
                text_parts.append(value)
    if not text_parts:
        output_text = _text(_value(response, "output_text"))
        if output_text:
            text_parts.append(output_text)
    return LLMResponse(
        content="".join(text_parts) or None,
        tool_calls=tuple(calls),
        finish_reason=_string(_value(response, "status"))
        or _string(_value(response, "finish_reason")),
        usage=_normalize_usage(_value(response, "usage")),
        raw_response=response,
    )


def _normalize_tool_calls(value: Any) -> tuple[ToolCall, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ClientError("chat completion tool_calls is malformed")
    return tuple(_normalize_tool_call(item) for item in value)


def _normalize_tool_call(value: Any) -> ToolCall:
    call_id = _string(_value(value, "id"))
    function = _value(value, "function") or value
    name = _string(_value(function, "name"))
    arguments = _arguments(_value(function, "arguments"))
    if not call_id or not name or arguments is None:
        raise ClientError("tool call is missing a function name or arguments")
    return ToolCall(call_id, name, arguments)


def _normalize_usage(value: Any) -> Usage | None:
    if value is None:
        return None
    return Usage(
        prompt_tokens=_integer(
            _value(value, "prompt_tokens")
            if _value(value, "prompt_tokens") is not None
            else _value(value, "input_tokens"),
            "prompt_tokens",
        ),
        completion_tokens=_integer(
            _value(value, "completion_tokens")
            if _value(value, "completion_tokens") is not None
            else _value(value, "output_tokens"),
            "completion_tokens",
        ),
        total_tokens=_integer(_value(value, "total_tokens"), "total_tokens"),
    )


def _arguments(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts: list[str] = []
        for item in value:
            text = _text(_value(item, "text") if _value(item, "text") is not None else item)
            if text:
                parts.append(text)
        return "".join(parts) or None
    if isinstance(value, Mapping):
        for key in ("text", "value", "content", "output_text"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
        return None
    return str(value) if isinstance(value, (int, float)) else None


def _value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _integer(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ClientError(f"provider usage field {field} is not an integer")
