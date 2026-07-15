from __future__ import annotations

from pathlib import Path

import pytest

from file_analysis_agent import Agent, AgentConfig
from file_analysis_agent.agent.loop import AgentLoop
from file_analysis_agent.agent.models import CompletionRequest, LLMResponse, ToolCall
from file_analysis_agent.clients.fake import SequenceClient
from file_analysis_agent.errors import AgentExecutionError, MaxStepsExceeded
from file_analysis_agent.sandbox.resolver import SandboxResolver
from file_analysis_agent.tools.filesystem import FileSystemTools
from file_analysis_agent.tools.registry import ToolRegistry


def _loop(workspace: Path, client: SequenceClient, max_steps: int = 100) -> AgentLoop:
    registry = ToolRegistry.filesystem(FileSystemTools(SandboxResolver(workspace)))
    return AgentLoop(
        client=client,
        registry=registry,
        system_prompt="immutable root",
        task="inspect",
        model="demo",
        max_steps=max_steps,
    )


def test_final_text_completes_in_one_decision_round(workspace: Path) -> None:
    client = SequenceClient([LLMResponse(content="final answer", finish_reason="stop")])

    assert _loop(workspace, client).run() == "final answer"
    assert len(client.requests) == 1
    assert client.requests[0].messages[0].content == "immutable root"


def test_tool_observation_is_returned_with_id_before_final_text(workspace: Path) -> None:
    client = SequenceClient(
        [
            LLMResponse(tool_calls=(ToolCall("c1", "read_file", '{"filepath":"a.txt"}'),)),
            LLMResponse(content="observed", finish_reason="stop"),
        ]
    )

    assert _loop(workspace, client).run() == "observed"
    second_messages = client.requests[1].messages
    assert second_messages[-1].role == "tool"
    assert second_messages[-1].tool_call_id == "c1"
    assert "1 | first" in (second_messages[-1].content or "")


def test_multiple_tool_calls_are_sequential_and_ordered(workspace: Path) -> None:
    client = SequenceClient(
        [
            LLMResponse(
                tool_calls=(
                    ToolCall("one", "list_dir", "{}"),
                    ToolCall("two", "read_file", '{"filepath":"a.txt"}'),
                )
            ),
            LLMResponse(content="done"),
        ]
    )

    assert _loop(workspace, client).run() == "done"
    tool_messages = [message for message in client.requests[1].messages if message.role == "tool"]
    assert [message.tool_call_id for message in tool_messages] == ["one", "two"]


def test_unknown_tool_and_invalid_arguments_become_observations(workspace: Path) -> None:
    client = SequenceClient(
        [
            LLMResponse(
                tool_calls=(
                    ToolCall("unknown", "remove_file", "{}"),
                    ToolCall("bad", "read_file", "not-json"),
                )
            ),
            LLMResponse(content="handled"),
        ]
    )

    assert _loop(workspace, client).run() == "handled"
    observations = [
        message.content or "" for message in client.requests[1].messages if message.role == "tool"
    ]
    assert "Unknown Tool" in observations[0]
    assert "Invalid JSON Arguments" in observations[1]


def test_max_steps_terminates_a_non_finishing_client(workspace: Path) -> None:
    call = ToolCall("loop", "list_dir", "{}")
    client = SequenceClient([LLMResponse(tool_calls=(call,)), LLMResponse(tool_calls=(call,))])

    with pytest.raises(MaxStepsExceeded, match="maximum Agent decision rounds"):
        _loop(workspace, client, max_steps=2).run()


def test_provider_failure_is_explicit(workspace: Path) -> None:
    class BrokenClient:
        def complete(self, _request: CompletionRequest) -> LLMResponse:
            raise RuntimeError("transport down")

    registry = ToolRegistry.filesystem(FileSystemTools(SandboxResolver(workspace)))
    loop = AgentLoop(
        client=BrokenClient(),
        registry=registry,
        system_prompt="root",
        task="inspect",
    )

    with pytest.raises(AgentExecutionError, match="transport down"):
        loop.run()


def test_subagent_has_independent_prompt_and_cannot_recurse(workspace: Path) -> None:
    client = SequenceClient(
        [
            LLMResponse(
                tool_calls=(ToolCall("parent-sub", "call_subagent", '{"task":"find files"}'),)
            ),
            LLMResponse(content="subagent result"),
            LLMResponse(content="parent result"),
        ]
    )
    config = AgentConfig(workspace_dir=workspace, provider="fake", model="demo")

    assert Agent(config, client=client).run("delegate") == "parent result"
    subagent_request = client.requests[1]
    assert subagent_request.messages[0].content != client.requests[0].messages[0].content
    assert all(schema["function"]["name"] != "call_subagent" for schema in subagent_request.tools)
    parent_observation = [
        message for message in client.requests[2].messages if message.role == "tool"
    ][0]
    assert parent_observation.tool_call_id == "parent-sub"
    assert parent_observation.content == "subagent result"


def test_subagent_registry_returns_unknown_tool_observation_for_recursion(workspace: Path) -> None:
    client = SequenceClient(
        [
            LLMResponse(
                tool_calls=(ToolCall("parent-sub", "call_subagent", '{"task":"recurse"}'),)
            ),
            LLMResponse(tool_calls=(ToolCall("child-sub", "call_subagent", '{"task":"again"}'),)),
            LLMResponse(content="child stopped"),
            LLMResponse(content="parent stopped"),
        ]
    )
    config = AgentConfig(workspace_dir=workspace, provider="fake", model="demo")

    assert Agent(config, client=client).run("delegate") == "parent stopped"
    child_observation = [
        message for message in client.requests[2].messages if message.role == "tool"
    ][0]
    assert "Unknown Tool: call_subagent" in (child_observation.content or "")
