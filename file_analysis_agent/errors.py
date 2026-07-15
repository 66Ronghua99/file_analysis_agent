"""Exceptions used at package boundaries."""


class FileAnalysisAgentError(Exception):
    """Base class for expected package failures."""


class ConfigurationError(FileAnalysisAgentError, ValueError):
    """Raised when configuration cannot be used safely."""


class ResourceError(FileAnalysisAgentError):
    """Raised when a prompt or schema resource cannot be loaded."""


class ClientError(FileAnalysisAgentError):
    """Raised when a provider client cannot produce a normalized response."""


class OptionalDependencyError(ClientError):
    """Raised when a selected provider's optional SDK is unavailable."""


class AgentExecutionError(FileAnalysisAgentError):
    """Raised when an agent run must terminate without a final answer."""


class ContextBudgetError(AgentExecutionError):
    """Raised when a request cannot fit in the configured context window."""


class BlackboardUpdateError(AgentExecutionError):
    """Raised when context compression cannot produce valid state."""


class MaxStepsExceeded(AgentExecutionError):
    """Raised when a loop reaches its configured decision-round limit."""
