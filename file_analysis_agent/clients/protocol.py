"""The provider-independent synchronous client contract."""

from __future__ import annotations

from typing import Protocol

from file_analysis_agent.agent.models import CompletionRequest, LLMResponse


class LLMClient(Protocol):
    """Any synchronous model client usable by the AgentLoop."""

    def complete(self, request: CompletionRequest) -> LLMResponse:
        """Return one normalized model decision."""


__all__ = ["CompletionRequest", "LLMClient", "LLMResponse"]
