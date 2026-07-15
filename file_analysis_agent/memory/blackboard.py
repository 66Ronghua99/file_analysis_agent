"""Structured Blackboard state and atomic conversation compression."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from file_analysis_agent.agent.models import CompletionRequest, Message
from file_analysis_agent.clients.protocol import LLMClient
from file_analysis_agent.errors import BlackboardUpdateError, ContextBudgetError
from file_analysis_agent.memory.token_counter import ApproximateTokenCounter, TokenCounter
from file_analysis_agent.resources_loader import load_prompt


@dataclass
class BlackboardEntry:
    """One finding with optional source provenance."""

    text: str
    provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Blackboard entries require non-empty text")
        if self.provenance is not None and not isinstance(self.provenance, dict):
            raise ValueError("Blackboard provenance must be an object")

    @classmethod
    def from_value(cls, value: Any) -> BlackboardEntry:
        if isinstance(value, cls):
            return cls(value.text, dict(value.provenance) if value.provenance else None)
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, Mapping) and isinstance(value.get("text"), str):
            provenance = value.get("provenance")
            if provenance is not None and not isinstance(provenance, Mapping):
                raise ValueError("Blackboard provenance must be an object")
            return cls(value["text"], dict(provenance) if provenance else None)
        raise ValueError("Blackboard list items must be strings or objects with text")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"text": self.text}
        if self.provenance:
            result["provenance"] = dict(self.provenance)
        return result


@dataclass
class BlackboardState:
    """Validated structured state owned by one Agent instance."""

    task: str
    facts: list[BlackboardEntry] = field(default_factory=list)
    findings: list[BlackboardEntry] = field(default_factory=list)
    decisions: list[BlackboardEntry] = field(default_factory=list)
    open_questions: list[BlackboardEntry] = field(default_factory=list)
    progress: str = ""
    next_action: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("Blackboard task must be non-empty")
        self.facts = self._entries(self.facts)
        self.findings = self._entries(self.findings)
        self.decisions = self._entries(self.decisions)
        self.open_questions = self._entries(self.open_questions)
        if not isinstance(self.progress, str) or not isinstance(self.next_action, str):
            raise ValueError("Blackboard progress and next_action must be strings")

    @staticmethod
    def _entries(values: Sequence[Any]) -> list[BlackboardEntry]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError("Blackboard sections must be arrays")
        return [BlackboardEntry.from_value(value) for value in values]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BlackboardState:
        if not isinstance(value, Mapping):
            raise ValueError("Blackboard payload must be an object")
        fields = {
            "task",
            "facts",
            "findings",
            "decisions",
            "open_questions",
            "progress",
            "next_action",
        }
        unknown = set(value) - fields
        if unknown:
            raise ValueError(f"unknown Blackboard fields: {', '.join(sorted(unknown))}")
        if not isinstance(value.get("task"), str):
            raise ValueError("Blackboard payload requires a string task")
        return cls(
            task=value["task"],
            facts=value.get("facts", []),
            findings=value.get("findings", []),
            decisions=value.get("decisions", []),
            open_questions=value.get("open_questions", []),
            progress=value.get("progress", ""),
            next_action=value.get("next_action", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "facts": [entry.to_dict() for entry in self.facts],
            "findings": [entry.to_dict() for entry in self.findings],
            "decisions": [entry.to_dict() for entry in self.decisions],
            "open_questions": [entry.to_dict() for entry in self.open_questions],
            "progress": self.progress,
            "next_action": self.next_action,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def clone(self) -> BlackboardState:
        return BlackboardState.from_dict(self.to_dict())


@dataclass
class ReActRound:
    """One model decision plus its ordered tool observations."""

    messages: list[Message] = field(default_factory=list)

    def to_dict(self) -> list[dict[str, Any]]:
        return [message.to_dict() for message in self.messages]


class BlackboardUpdater(Protocol):
    """Turn older rounds into a validated candidate Blackboard."""

    def update(
        self,
        current: BlackboardState,
        older_rounds: Sequence[ReActRound],
    ) -> BlackboardState | Mapping[str, Any]:
        """Return a candidate state without mutating the current state."""


class ModelBlackboardUpdater:
    """Use a no-tools model request to update structured Blackboard state."""

    def __init__(
        self,
        client: LLMClient,
        model: str,
        prompt_dir: Any = None,
        api_mode: str = "chat_completion",
    ) -> None:
        self.client = client
        self.model = model
        self.api_mode = api_mode
        self.prompt = load_prompt("blackboard_update.md", prompt_dir)

    def update(
        self,
        current: BlackboardState,
        older_rounds: Sequence[ReActRound],
    ) -> BlackboardState:
        payload = {
            "task": current.task,
            "blackboard": current.to_dict(),
            "older_rounds": [round_item.to_dict() for round_item in older_rounds],
        }
        request = CompletionRequest(
            messages=(
                Message(role="system", content=self.prompt),
                Message(
                    role="user",
                    content=json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            ),
            tools=(),
            model=self.model,
            api_mode=self.api_mode,
        )
        try:
            response = self.client.complete(request)
            if response.tool_calls:
                raise ValueError("Blackboard updater returned tool calls")
            if not response.content:
                raise ValueError("Blackboard updater returned no JSON content")
            value = json.loads(response.content)
            if not isinstance(value, Mapping):
                raise ValueError("Blackboard updater JSON must be an object")
            return BlackboardState.from_dict(value)
        except Exception as exc:  # Boundary: preserve failed update context for the Agent.
            if isinstance(exc, BlackboardUpdateError):
                raise
            raise BlackboardUpdateError(
                f"Blackboard update failed: {type(exc).__name__}: {exc}"
            ) from exc


class BlackboardMemory:
    """Assemble root prompt, Blackboard packet, and bounded raw conversation history."""

    def __init__(
        self,
        root_prompt: str,
        state: BlackboardState,
        *,
        context_window: int,
        updater: BlackboardUpdater,
        token_counter: TokenCounter | None = None,
        blackboard_budget_ratio: float = 0.2,
        max_complete_rounds: int = 5,
    ) -> None:
        if context_window <= 0:
            raise ValueError("context_window must be positive")
        if not 0 < blackboard_budget_ratio < 1:
            raise ValueError("blackboard_budget_ratio must be between 0 and 1")
        if max_complete_rounds <= 0:
            raise ValueError("max_complete_rounds must be positive")
        self.root_prompt = root_prompt
        self.state = state
        self.context_window = context_window
        self.updater = updater
        self.token_counter = token_counter or ApproximateTokenCounter()
        self.blackboard_budget_ratio = blackboard_budget_ratio
        self.max_complete_rounds = max_complete_rounds
        self.rounds: list[ReActRound] = []
        self.active_round: ReActRound | None = None
        self._history_revision = 0
        self._last_compacted_revision = -1

    def begin_round(self) -> None:
        if self.active_round is not None:
            raise RuntimeError("a ReAct round is already active")
        self.active_round = ReActRound()
        self._history_revision += 1

    def add_to_active(self, message: Message) -> None:
        if self.active_round is None:
            raise RuntimeError("no active ReAct round")
        self.active_round.messages.append(message)
        self._history_revision += 1

    def finish_round(self) -> None:
        if self.active_round is None:
            raise RuntimeError("no active ReAct round")
        self.rounds.append(self.active_round)
        self.active_round = None
        self._history_revision += 1

    def assemble_messages(self) -> list[Message]:
        messages = [
            Message(role="system", content=self.root_prompt),
            Message(role="system", content=self.state.to_json(), name="blackboard"),
            Message(role="user", content=self.state.task),
        ]
        for round_item in self.rounds:
            messages.extend(round_item.messages)
        if self.active_round is not None:
            messages.extend(self.active_round.messages)
        return messages

    def ensure_context(self) -> None:
        current_count = self._count(self.assemble_messages())
        threshold = self.context_window * 0.8
        if current_count < threshold and current_count <= self.context_window:
            return
        if self._last_compacted_revision == self._history_revision:
            if current_count > self.context_window:
                raise ContextBudgetError(
                    f"context budget exceeded: {current_count} > {self.context_window} tokens"
                )
            return
        self._compact()

    def _compact(self) -> None:
        retained = self.rounds[-self.max_complete_rounds :]
        older = self.rounds[: -self.max_complete_rounds]
        try:
            candidate_value = self.updater.update(self.state.clone(), older)
            candidate = (
                candidate_value.clone()
                if isinstance(candidate_value, BlackboardState)
                else BlackboardState.from_dict(candidate_value)
            )
            if candidate.task != self.state.task:
                raise BlackboardUpdateError("Blackboard update changed the original task")
            candidate_packet = Message(
                role="system", content=candidate.to_json(), name="blackboard"
            )
            if self._count([candidate_packet]) > self.context_window * self.blackboard_budget_ratio:
                raise ContextBudgetError("Blackboard packet exceeds its context budget")
            candidate_messages = [
                Message(role="system", content=self.root_prompt),
                candidate_packet,
                Message(role="user", content=candidate.task),
            ]
            for round_item in retained:
                candidate_messages.extend(round_item.messages)
            if self.active_round is not None:
                candidate_messages.extend(self.active_round.messages)
            candidate_count = self._count(candidate_messages)
            if candidate_count > self.context_window:
                raise ContextBudgetError(
                    f"context budget exceeded after compression: {candidate_count} > "
                    f"{self.context_window} tokens"
                )
        except ContextBudgetError:
            raise
        except Exception as exc:
            if isinstance(exc, BlackboardUpdateError):
                raise
            raise BlackboardUpdateError(
                f"Blackboard update failed: {type(exc).__name__}: {exc}"
            ) from exc
        self.state = candidate
        self.rounds = list(retained)
        self._last_compacted_revision = self._history_revision

    def _count(self, messages: Sequence[Message]) -> int:
        try:
            count = self.token_counter.count(messages)
        except Exception as exc:
            raise ContextBudgetError(f"token counting failed: {type(exc).__name__}: {exc}") from exc
        if not isinstance(count, int) or count < 0:
            raise ContextBudgetError("token counter returned an invalid count")
        return count
