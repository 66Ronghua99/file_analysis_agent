"""JSON-schema-backed tool registration and dispatch."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from file_analysis_agent.agent.models import Observation, ToolCall
from file_analysis_agent.resources_loader import load_tool_schemas

ToolHandler = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    schema: Mapping[str, Any]
    handler: ToolHandler


class ToolRegistry:
    """Expose only explicitly registered, schema-validated callables."""

    def __init__(self, tools: list[RegisteredTool] | None = None) -> None:
        self._tools = {tool.name: tool for tool in (tools or [])}

    @classmethod
    def from_schemas(
        cls,
        schemas: Sequence[Mapping[str, Any]],
        handlers: Mapping[str, ToolHandler],
    ) -> ToolRegistry:
        registered: list[RegisteredTool] = []
        for schema in schemas:
            name = _schema_name(schema)
            handler = handlers.get(name)
            if handler is not None:
                registered.append(RegisteredTool(name, schema, handler))
        return cls(registered)

    @classmethod
    def filesystem(
        cls,
        filesystem_tools: Any,
        *,
        call_subagent: ToolHandler | None = None,
        prompt_dir: Any = None,
    ) -> ToolRegistry:
        schemas = load_tool_schemas(prompt_dir)
        handlers: dict[str, ToolHandler] = {
            "list_dir": lambda args: filesystem_tools.list_dir(**args),
            "search_text": lambda args: filesystem_tools.search_text(**args),
            "read_file": lambda args: filesystem_tools.read_file(**args),
        }
        if call_subagent is not None:
            handlers["call_subagent"] = call_subagent
        return cls.from_schemas(schemas, handlers)

    @property
    def schemas(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(tool.schema for tool in self._tools.values())

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def dispatch(self, call: ToolCall) -> Observation:
        """Validate and invoke one call, returning all failures as observations."""

        tool = self._tools.get(call.name)
        if tool is None:
            return Observation(call.id, call.name, f"[Error] Unknown Tool: {call.name}")
        try:
            arguments = json.loads(call.arguments)
        except (TypeError, json.JSONDecodeError) as exc:
            return Observation(call.id, call.name, f"[Error] Invalid JSON Arguments: {exc}")
        if not isinstance(arguments, dict):
            return Observation(
                call.id, call.name, "[Error] Invalid Arguments: expected a JSON object"
            )
        validation_error = _validate_schema(arguments, tool.schema)
        if validation_error is not None:
            return Observation(call.id, call.name, f"[Error] Invalid Arguments: {validation_error}")
        try:
            result = tool.handler(arguments)
        except Exception as exc:  # Boundary: preserve model-visible diagnostic context.
            result = f"[Error] Tool Execution Failed: {type(exc).__name__}: {exc}"
        return Observation(call.id, call.name, str(result))


def _schema_name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
        raise ValueError("tool schema must contain function.name")
    name = function["name"]
    if not isinstance(name, str):
        raise ValueError("tool schema function.name must be a string")
    return name


def _validate_schema(value: dict[str, Any], schema: Mapping[str, Any]) -> str | None:
    function = schema.get("function")
    parameters = function.get("parameters") if isinstance(function, Mapping) else None
    if not isinstance(parameters, Mapping):
        return "tool schema has no object parameters"
    if parameters.get("type") not in (None, "object"):
        return "tool schema parameters must be an object"
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        return "tool schema properties are malformed"
    for name in required:
        if isinstance(name, str) and name not in value:
            return f"missing required field {name!r}"
    if parameters.get("additionalProperties") is False:
        unknown = sorted(set(value) - set(properties))
        if unknown:
            return f"unknown field(s): {', '.join(unknown)}"
    for name, field_schema in properties.items():
        if name not in value or not isinstance(field_schema, Mapping):
            continue
        error = _validate_field(value[name], field_schema, name)
        if error is not None:
            return error
    return None


def _validate_field(value: Any, schema: Mapping[str, Any], name: str) -> str | None:
    expected = schema.get("type")
    type_ok = {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
    }
    if isinstance(expected, str) and not type_ok.get(expected, True):
        return f"field {name!r} must be {expected}"
    if isinstance(value, int) and isinstance(schema.get("minimum"), (int, float)):
        if value < schema["minimum"]:
            return f"field {name!r} must be at least {schema['minimum']}"
    if isinstance(schema.get("enum"), list) and value not in schema["enum"]:
        return f"field {name!r} has an unsupported value"
    return None
