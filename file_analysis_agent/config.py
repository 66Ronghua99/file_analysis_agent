"""Validated configuration shared by the package API and CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from file_analysis_agent.errors import ConfigurationError

SUPPORTED_PROVIDERS = frozenset({"openrouter", "vllm", "litellm", "fake"})
SUPPORTED_API_MODES = frozenset({"chat_completion", "responses"})


@dataclass
class AgentConfig:
    """Runtime configuration with validation at construction time."""

    workspace_dir: Path
    model: str
    provider: str = "openrouter"
    base_url: str | None = None
    api_mode: str = "chat_completion"
    max_steps: int = 100
    context_window: int = 8192
    prompt_dir: Path | None = None
    max_search_matches: int = 200
    max_read_lines: int = 300
    subagent_max_steps: int = 30
    max_subagent_invocations: int = 8
    blackboard_budget_ratio: float = 0.2
    api_key: str | None = None
    temperature: float | None = None

    def __post_init__(self) -> None:
        self.workspace_dir = self._validate_directory(self.workspace_dir, "workspace")
        if self.prompt_dir is not None:
            self.prompt_dir = self._validate_directory(self.prompt_dir, "prompt")
        if not isinstance(self.model, str):
            raise ConfigurationError("model is required")
        self.model = self.model.strip()
        if not self.model:
            raise ConfigurationError("model is required")
        if not isinstance(self.provider, str) or self.provider not in SUPPORTED_PROVIDERS:
            choices = ", ".join(sorted(SUPPORTED_PROVIDERS))
            raise ConfigurationError(f"unknown provider {self.provider!r}; choose from {choices}")
        if not isinstance(self.api_mode, str) or self.api_mode not in SUPPORTED_API_MODES:
            choices = ", ".join(sorted(SUPPORTED_API_MODES))
            raise ConfigurationError(f"invalid api mode {self.api_mode!r}; choose from {choices}")
        self._validate_positive(self.max_steps, "max_steps")
        self._validate_positive(self.context_window, "context_window")
        self._validate_positive(self.max_search_matches, "max_search_matches")
        self._validate_positive(self.max_read_lines, "max_read_lines")
        self._validate_positive(self.subagent_max_steps, "subagent_max_steps")
        self._validate_positive(self.max_subagent_invocations, "max_subagent_invocations")
        if (
            isinstance(self.blackboard_budget_ratio, bool)
            or not isinstance(self.blackboard_budget_ratio, (int, float))
            or not 0 < self.blackboard_budget_ratio < 1
        ):
            raise ConfigurationError("blackboard_budget_ratio must be between 0 and 1")
        if self.provider == "vllm" and not self.base_url:
            raise ConfigurationError("base_url is required for the vllm provider")

    @staticmethod
    def _validate_directory(value: Path, label: str) -> Path:
        try:
            path = Path(value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"{label}_dir must be a filesystem path") from exc
        if not path.is_absolute():
            raise ConfigurationError(f"{label}_dir must be an absolute path")
        if label == "workspace":
            try:
                if not path.is_dir():
                    if path.exists():
                        raise ConfigurationError(f"{label}_dir is not a directory: {path}")
                    raise ConfigurationError(f"{label}_dir does not exist: {path}")
            except OSError as exc:
                raise ConfigurationError(f"cannot inspect {label}_dir {path}: {exc}") from exc
            return path
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ConfigurationError(f"{label}_dir does not exist: {path}") from exc
        except OSError as exc:
            raise ConfigurationError(f"cannot resolve {label}_dir {path}: {exc}") from exc
        if not resolved.is_dir():
            raise ConfigurationError(f"{label}_dir is not a directory: {path}")
        return resolved

    @staticmethod
    def _validate_positive(value: int, label: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigurationError(f"{label} must be a positive integer")
