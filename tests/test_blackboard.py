from __future__ import annotations

from collections.abc import Sequence

import pytest

from file_analysis_agent.agent.models import Message
from file_analysis_agent.errors import BlackboardUpdateError, ContextBudgetError
from file_analysis_agent.memory.blackboard import (
    BlackboardEntry,
    BlackboardMemory,
    BlackboardState,
    BlackboardUpdater,
    ReActRound,
)


class ThresholdCounter:
    def __init__(self, current: int = 90, packet: int = 1) -> None:
        self.current = current
        self.packet = packet
        self.calls = 0

    def count(self, messages: Sequence[Message]) -> int:
        self.calls += 1
        return (
            self.packet if len(messages) == 1 and messages[0].name == "blackboard" else self.current
        )


class RecordingUpdater:
    def __init__(self) -> None:
        self.calls: list[tuple[BlackboardState, Sequence[ReActRound]]] = []

    def update(
        self, current: BlackboardState, older_rounds: Sequence[ReActRound]
    ) -> BlackboardState:
        self.calls.append((current, older_rounds))
        current.findings.append(
            type(current.facts[0])("compacted") if current.facts else _entry("compacted")
        )
        return current


class FailingUpdater:
    def update(
        self, current: BlackboardState, older_rounds: Sequence[ReActRound]
    ) -> BlackboardState:
        raise ValueError("bad structured payload")


def _entry(text: str) -> BlackboardEntry:
    return BlackboardEntry(text)


def _memory(updater: BlackboardUpdater, counter: ThresholdCounter) -> BlackboardMemory:
    memory = BlackboardMemory(
        root_prompt="ROOT PROMPT",
        state=BlackboardState(task="inspect files", facts=[_entry("known")]),
        context_window=100,
        updater=updater,
        token_counter=counter,
    )
    for index in range(7):
        memory.begin_round()
        memory.add_to_active(Message(role="assistant", content=f"round {index}"))
        memory.finish_round()
    return memory


def test_threshold_compaction_keeps_latest_five_and_injects_packet() -> None:
    updater = RecordingUpdater()
    memory = _memory(updater, ThresholdCounter())

    memory.ensure_context()

    assert len(updater.calls) == 1
    assert len(updater.calls[0][1]) == 2
    assert len(memory.rounds) == 5
    assembled = memory.assemble_messages()
    assert assembled[0].content == "ROOT PROMPT"
    assert assembled[1].role == "system" and assembled[1].name == "blackboard"
    assert "compacted" in (assembled[1].content or "")
    assert [round_item.messages[0].content for round_item in memory.rounds] == [
        "round 2",
        "round 3",
        "round 4",
        "round 5",
        "round 6",
    ]


def test_active_round_survives_compaction() -> None:
    updater = RecordingUpdater()
    memory = _memory(updater, ThresholdCounter())
    memory.begin_round()
    memory.add_to_active(Message(role="assistant", content="unfinished"))

    memory.ensure_context()

    assert memory.active_round is not None
    assert memory.active_round.messages[0].content == "unfinished"


def test_failed_update_is_atomic() -> None:
    memory = _memory(FailingUpdater(), ThresholdCounter())
    before_state = memory.state.to_dict()
    before_rounds = [round_item.to_dict() for round_item in memory.rounds]

    with pytest.raises(BlackboardUpdateError, match="bad structured payload"):
        memory.ensure_context()

    assert memory.state.to_dict() == before_state
    assert [round_item.to_dict() for round_item in memory.rounds] == before_rounds


def test_blackboard_packet_budget_is_explicit() -> None:
    memory = _memory(RecordingUpdater(), ThresholdCounter(packet=30))

    with pytest.raises(ContextBudgetError, match="Blackboard packet"):
        memory.ensure_context()


def test_blackboard_state_validates_and_preserves_provenance() -> None:
    state = BlackboardState.from_dict(
        {
            "task": "inspect",
            "facts": [{"text": "config found", "provenance": {"file": "a.py", "line": 4}}],
            "findings": ["one finding"],
            "decisions": [],
            "open_questions": [],
            "progress": "started",
            "next_action": "read more",
        }
    )

    encoded = state.to_dict()

    assert encoded["facts"][0]["provenance"]["file"] == "a.py"
    assert encoded["findings"][0]["text"] == "one finding"
    with pytest.raises(ValueError, match="unknown Blackboard fields"):
        BlackboardState.from_dict({"task": "x", "unexpected": []})
