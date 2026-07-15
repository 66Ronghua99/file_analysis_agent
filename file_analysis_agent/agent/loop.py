"""Provider-independent synchronous ReAct orchestration."""

from __future__ import annotations

from file_analysis_agent.agent.models import CompletionRequest, LLMResponse, Message
from file_analysis_agent.agent.subagent import SubagentRunner
from file_analysis_agent.clients.factory import create_client
from file_analysis_agent.clients.protocol import LLMClient
from file_analysis_agent.config import AgentConfig
from file_analysis_agent.errors import AgentExecutionError, ClientError, MaxStepsExceeded
from file_analysis_agent.memory.blackboard import (
    BlackboardMemory,
    BlackboardState,
    BlackboardUpdater,
    ModelBlackboardUpdater,
)
from file_analysis_agent.memory.token_counter import TokenCounter
from file_analysis_agent.resources_loader import load_prompt
from file_analysis_agent.sandbox.resolver import SandboxResolver
from file_analysis_agent.tools.filesystem import FileSystemTools
from file_analysis_agent.tools.registry import ToolRegistry


class AgentLoop:
    """Run normalized model decisions and sequentially dispatch tool calls."""

    def __init__(
        self,
        *,
        client: LLMClient,
        registry: ToolRegistry,
        system_prompt: str,
        task: str,
        model: str = "model",
        api_mode: str = "chat_completion",
        max_steps: int = 100,
        context_window: int = 8192,
        blackboard_budget_ratio: float = 0.2,
        token_counter: TokenCounter | None = None,
        updater: BlackboardUpdater | None = None,
        blackboard: BlackboardState | None = None,
        temperature: float | None = None,
        memory: BlackboardMemory | None = None,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.client = client
        self.registry = registry
        self.model = model
        self.api_mode = api_mode
        self.max_steps = max_steps
        self.temperature = temperature
        if memory is not None:
            self.memory = memory
        else:
            state = blackboard or BlackboardState(task=task)
            selected_updater = updater or ModelBlackboardUpdater(client, model, api_mode=api_mode)
            self.memory = BlackboardMemory(
                root_prompt=system_prompt,
                state=state,
                context_window=context_window,
                updater=selected_updater,
                token_counter=token_counter,
                blackboard_budget_ratio=blackboard_budget_ratio,
            )
        self._task = task

    def run(self) -> str:
        """Return final assistant text or raise an explicit termination error."""

        for _step in range(self.max_steps):
            self.memory.ensure_context()
            self.memory.begin_round()
            request = CompletionRequest(
                messages=tuple(self.memory.assemble_messages()),
                tools=tuple(self.registry.schemas),
                model=self.model,
                api_mode=self.api_mode,
                temperature=self.temperature,
            )
            try:
                response = self.client.complete(request)
            except ClientError as exc:
                self.memory.active_round = None
                raise AgentExecutionError(f"LLM client failed: {exc}") from exc
            except Exception as exc:
                self.memory.active_round = None
                raise AgentExecutionError(
                    f"LLM client failed: {type(exc).__name__}: {exc}"
                ) from exc
            self._validate_response(response)
            assistant_message = Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
            )
            self.memory.add_to_active(assistant_message)
            if not response.tool_calls:
                self.memory.finish_round()
                if response.content is None or not response.content.strip():
                    raise AgentExecutionError("model returned neither final text nor tool calls")
                return response.content
            for call in response.tool_calls:
                observation = self.registry.dispatch(call)
                self.memory.add_to_active(observation.as_message())
            self.memory.finish_round()
        raise MaxStepsExceeded(f"maximum Agent decision rounds reached: {self.max_steps}")

    @staticmethod
    def _validate_response(response: LLMResponse) -> None:
        if not isinstance(response, LLMResponse):
            raise AgentExecutionError("LLM client returned an invalid response object")
        for call in response.tool_calls:
            if not call.id or not call.name:
                raise AgentExecutionError("LLM client returned a tool call without id or name")


class Agent:
    """High-level package API that builds the sandbox, tools, loop, and subagents."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        client: LLMClient | None = None,
        token_counter: TokenCounter | None = None,
        updater: BlackboardUpdater | None = None,
    ) -> None:
        self.config = config
        self.client = client if client is not None else create_client(config)
        self.token_counter = token_counter
        self.updater = updater

    def run(self, task: str) -> str:
        if not isinstance(task, str) or not task.strip():
            raise AgentExecutionError("task must not be empty")
        resolver = SandboxResolver(self.config.workspace_dir)
        filesystem_tools = FileSystemTools(
            resolver,
            max_search_matches=self.config.max_search_matches,
            max_read_lines=self.config.max_read_lines,
        )
        subagent_runner = SubagentRunner(
            self.client,
            filesystem_tools,
            self.config,
            token_counter=self.token_counter,
            updater=self.updater,
        )
        registry = ToolRegistry.filesystem(
            filesystem_tools,
            call_subagent=lambda args: subagent_runner.run(args["task"]),
            prompt_dir=self.config.prompt_dir,
        )
        root_prompt = load_prompt("system.md", self.config.prompt_dir)
        updater = self.updater or ModelBlackboardUpdater(
            self.client, self.config.model, self.config.prompt_dir, self.config.api_mode
        )
        loop = AgentLoop(
            client=self.client,
            registry=registry,
            system_prompt=root_prompt,
            task=task,
            model=self.config.model,
            api_mode=self.config.api_mode,
            max_steps=self.config.max_steps,
            context_window=self.config.context_window,
            blackboard_budget_ratio=self.config.blackboard_budget_ratio,
            token_counter=self.token_counter,
            updater=updater,
            temperature=self.config.temperature,
        )
        return loop.run()
