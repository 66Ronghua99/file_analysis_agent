"""Injectable token-counting boundary with a deterministic default."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol

from file_analysis_agent.agent.models import Message


class TokenCounter(Protocol):
    """Count the approximate provider tokens in normalized messages."""

    def count(self, messages: Sequence[Message]) -> int:
        """Return a non-negative token estimate."""


class ApproximateTokenCounter:
    """Conservative UTF-8-size estimate that is deterministic in tests."""

    def __init__(self, per_message_overhead: int = 4) -> None:
        if per_message_overhead < 0:
            raise ValueError("per_message_overhead cannot be negative")
        self.per_message_overhead = per_message_overhead

    def count(self, messages: Sequence[Message]) -> int:
        serialized = json.dumps(
            [message.to_dict() for message in messages],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        utf8_tokens = (len(serialized.encode("utf-8")) + 3) // 4
        return utf8_tokens + len(messages) * self.per_message_overhead
