"""OpenAI-compatible adapters for OpenRouter and externally managed vLLM."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from file_analysis_agent.agent.models import CompletionRequest, LLMResponse, Message
from file_analysis_agent.clients.normalization import (
    normalize_chat_response,
    normalize_responses_response,
)
from file_analysis_agent.errors import ClientError, OptionalDependencyError


class OpenAICompatibleClient:
    """Use the selected OpenAI-compatible API mode without provider branches in the loop."""

    def __init__(
        self,
        model: str,
        *,
        provider: str = "openrouter",
        base_url: str | None = None,
        api_key: str | None = None,
        api_mode: str = "chat_completion",
        sdk_client: Any = None,
    ) -> None:
        if api_mode not in {"chat_completion", "responses"}:
            raise ClientError(f"unsupported api mode: {api_mode}")
        if provider not in {"openrouter", "vllm"}:
            raise ClientError(f"unsupported OpenAI-compatible provider: {provider}")
        self.model = model
        self.provider = provider
        self.api_mode = api_mode
        self._client = (
            sdk_client
            if sdk_client is not None
            else self._build_client(provider, base_url, api_key)
        )

    def complete(self, request: CompletionRequest) -> LLMResponse:
        if request.api_mode != self.api_mode:
            raise ClientError(
                f"client is configured for {self.api_mode}, request selected {request.api_mode}"
            )
        if request.model != self.model:
            raise ClientError(
                f"client model {self.model!r} does not match request model {request.model!r}"
            )
        try:
            if self.api_mode == "chat_completion":
                response = self._client.chat.completions.create(**self._chat_payload(request))
                return normalize_chat_response(response)
            response = self._client.responses.create(**self._responses_payload(request))
            return normalize_responses_response(response)
        except ClientError:
            raise
        except Exception as exc:
            raise ClientError(
                f"{self.provider} request failed: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _build_client(provider: str, base_url: str | None, api_key: str | None) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise OptionalDependencyError(
                "the openai package is required for OpenRouter and vLLM adapters; "
                "install file-analysis-agent[openai]"
            ) from exc
        if provider == "openrouter":
            base_url = base_url or "https://openrouter.ai/api/v1"
            api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        else:
            api_key = api_key or os.getenv("OPENAI_API_KEY") or "not-needed"
        if not api_key:
            raise ClientError("an API key is required for the OpenRouter provider")
        return OpenAI(api_key=api_key, base_url=base_url)

    def _chat_payload(self, request: CompletionRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": request.message_dicts(),
            "tools": [dict(tool) for tool in request.tools],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        return payload

    def _responses_payload(self, request: CompletionRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "input": _responses_input(request.messages),
            "tools": [_responses_tool(tool) for tool in request.tools],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        return payload


def _responses_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    function = tool.get("function")
    if not isinstance(function, Mapping):
        raise ClientError("function tool schema is malformed")
    name = function.get("name")
    if not isinstance(name, str):
        raise ClientError("function tool schema is missing a name")
    result: dict[str, Any] = {"type": "function", "name": name}
    for key in ("description", "parameters", "strict"):
        if key in function:
            result[key] = function[key]
    return result


def _responses_input(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    """Translate normalized history into Responses input items."""

    result: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool" and message.tool_call_id is not None:
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content or "",
                }
            )
            continue
        if message.role == "assistant" and message.tool_calls:
            if message.content:
                result.append({"role": "assistant", "content": message.content})
            result.extend(
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                for call in message.tool_calls
            )
            continue
        result.append({"role": message.role, "content": message.content})
    return result
