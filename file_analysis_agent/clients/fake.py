"""Deterministic fake clients useful for tests and offline CLI smoke checks."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from file_analysis_agent.agent.models import CompletionRequest, LLMResponse
from file_analysis_agent.errors import ClientError


class SequenceClient:
    """Return a fixed sequence of responses and record every request."""

    def __init__(
        self, responses: Iterable[LLMResponse | Callable[[CompletionRequest], LLMResponse]]
    ) -> None:
        self._responses = list(responses)
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> LLMResponse:
        self.requests.append(request)
        if not self._responses:
            raise ClientError("fake client has no remaining responses")
        response = self._responses.pop(0)
        return response(request) if callable(response) else response


class StaticClient:
    """Return one final answer for every request."""

    def __init__(self, content: str = "Fake client completed the task.") -> None:
        self.content = content
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(content=self.content, finish_reason="stop")
