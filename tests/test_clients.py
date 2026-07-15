from __future__ import annotations

from typing import Any

import pytest

from file_analysis_agent.agent.models import CompletionRequest, Message
from file_analysis_agent.clients.litellm_client import LiteLLMClient
from file_analysis_agent.clients.normalization import (
    normalize_chat_response,
    normalize_responses_response,
)
from file_analysis_agent.clients.openai_compatible import OpenAICompatibleClient
from file_analysis_agent.errors import ClientError


def _request(mode: str = "chat_completion") -> CompletionRequest:
    return CompletionRequest(
        messages=(Message(role="user", content="inspect"),),
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "read",
                    "parameters": {"type": "object"},
                },
            },
        ),
        model="demo",
        api_mode=mode,
    )


class FakeChatCompletions:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


class FakeSDK:
    def __init__(self, response: Any) -> None:
        self.chat = type("Chat", (), {"completions": FakeChatCompletions(response)})()
        self.responses = type("Responses", (), {"create": lambda _self, **_kwargs: response})()


def test_chat_completion_normalizes_multiple_tool_calls() -> None:
    response = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": "I will inspect both files.",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "list_dir", "arguments": "{}"}},
                        {
                            "id": "c2",
                            "function": {
                                "name": "read_file",
                                "arguments": {"filepath": "a.txt"},
                            },
                        },
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }

    normalized = normalize_chat_response(response)

    assert normalized.content == "I will inspect both files."
    assert [call.id for call in normalized.tool_calls] == ["c1", "c2"]
    assert normalized.tool_calls[1].arguments == '{"filepath":"a.txt"}'
    assert normalized.usage is not None and normalized.usage.total_tokens == 14


def test_openai_compatible_adapter_keeps_chat_payload_and_mode() -> None:
    sdk = FakeSDK({"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]})
    client = OpenAICompatibleClient(
        "demo", provider="vllm", api_mode="chat_completion", sdk_client=sdk
    )

    result = client.complete(_request())

    assert result.content == "done"
    assert sdk.chat.completions.calls[0]["model"] == "demo"
    assert sdk.chat.completions.calls[0]["messages"][0]["content"] == "inspect"


def test_responses_normalization_matches_internal_shape() -> None:
    response = {
        "status": "completed",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "done"}]},
            {"type": "function_call", "call_id": "c1", "name": "list_dir", "arguments": "{}"},
        ],
    }

    normalized = normalize_responses_response(response)

    assert normalized.content == "done"
    assert normalized.tool_calls[0].id == "c1"
    assert normalized.finish_reason == "completed"


def test_litellm_adapter_normalizes_fake_module() -> None:
    class FakeLiteLLM:
        @staticmethod
        def completion(**_kwargs: Any) -> Any:
            return {"choices": [{"message": {"content": "lite"}, "finish_reason": "stop"}]}

    client = LiteLLMClient("demo", module=FakeLiteLLM())

    assert client.complete(_request()).content == "lite"


def test_clients_reject_malformed_responses_and_mode_mismatch() -> None:
    with pytest.raises(ClientError, match="no choices"):
        normalize_chat_response({})
    with pytest.raises(ClientError, match="function call"):
        normalize_responses_response(
            {"output": [{"type": "function_call", "name": "read_file", "arguments": "{}"}]}
        )
    client = OpenAICompatibleClient(
        "demo",
        provider="vllm",
        api_mode="responses",
        sdk_client=FakeSDK({"output_text": "done", "output": []}),
    )
    with pytest.raises(ClientError, match="configured for responses"):
        client.complete(_request("chat_completion"))
