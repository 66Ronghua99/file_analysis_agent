"""Provider-independent messages and tool-call models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """A normalized function call emitted by a model."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Usage:
    """Optional provider usage counters."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class Message:
    """A message in the normalized conversation history."""

    role: str
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the message to an OpenAI-compatible shape."""

        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True)
class CompletionRequest:
    """A complete provider-neutral completion request."""

    messages: tuple[Message, ...]
    tools: tuple[Mapping[str, Any], ...]
    model: str
    api_mode: str = "chat_completion"
    temperature: float | None = None

    def message_dicts(self) -> list[dict[str, Any]]:
        return [message.to_dict() for message in self.messages]


@dataclass(frozen=True)
class LLMResponse:
    """A normalized response returned by any supported client."""

    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    finish_reason: str | None = None
    usage: Usage | None = None
    raw_response: Any = None


@dataclass(frozen=True)
class Observation:
    """A tool result that is both model-visible and history-preserving."""

    tool_call_id: str
    name: str
    content: str

    def as_message(self) -> Message:
        return Message(
            role="tool",
            content=self.content,
            tool_call_id=self.tool_call_id,
            name=self.name,
        )
