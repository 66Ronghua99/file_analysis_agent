"""Bounded independent subagent construction."""

from __future__ import annotations

from typing import Any

from file_analysis_agent.clients.protocol import LLMClient
from file_analysis_agent.config import AgentConfig
from file_analysis_agent.memory.token_counter import TokenCounter
from file_analysis_agent.resources_loader import load_prompt
from file_analysis_agent.tools.filesystem import FileSystemTools
from file_analysis_agent.tools.registry import ToolRegistry


class SubagentRunner:
    """Create a fresh loop and restricted registry for each delegated question."""

    def __init__(
        self,
        client: LLMClient,
        filesystem_tools: FileSystemTools,
        config: AgentConfig,
        *,
        token_counter: TokenCounter | None = None,
        updater: Any = None,
    ) -> None:
        self.client = client
        self.filesystem_tools = filesystem_tools
        self.config = config
        self.token_counter = token_counter
        self.updater = updater
        self.invocations = 0

    def reset(self) -> None:
        self.invocations = 0

    def run(self, task: str) -> str:
        if self.invocations >= self.config.max_subagent_invocations:
            return (
                f"[Error] Subagent invocation limit reached: {self.config.max_subagent_invocations}"
            )
        if not isinstance(task, str) or not task.strip():
            return "[Error] Invalid Argument: subagent task must not be empty"
        self.invocations += 1
        from file_analysis_agent.agent.loop import AgentLoop
        from file_analysis_agent.memory.blackboard import ModelBlackboardUpdater

        root_prompt = load_prompt("subagent_system.md", self.config.prompt_dir)
        updater = self.updater
        if updater is None:
            updater = ModelBlackboardUpdater(
                self.client,
                self.config.model,
                self.config.prompt_dir,
                self.config.api_mode,
            )
        registry = ToolRegistry.filesystem(
            self.filesystem_tools,
            prompt_dir=self.config.prompt_dir,
        )
        loop = AgentLoop(
            client=self.client,
            registry=registry,
            system_prompt=root_prompt,
            task=task,
            model=self.config.model,
            api_mode=self.config.api_mode,
            max_steps=self.config.subagent_max_steps,
            context_window=self.config.context_window,
            blackboard_budget_ratio=self.config.blackboard_budget_ratio,
            token_counter=self.token_counter,
            updater=updater,
            temperature=self.config.temperature,
        )
        try:
            return loop.run()
        except Exception as exc:  # Boundary: return a bounded parent observation.
            return f"[Error] Subagent Failed: {type(exc).__name__}: {exc}"
