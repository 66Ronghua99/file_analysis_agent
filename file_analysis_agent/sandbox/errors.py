"""Stable, tool-facing sandbox errors."""


class SandboxError(Exception):
    """Base class for expected path-boundary failures."""


class OutOfBoundsError(SandboxError):
    """A path resolves outside the configured workspace."""

    def __init__(self) -> None:
        super().__init__("[Error] Permission Denied: out of bounds")


class InvalidSandboxPathError(SandboxError):
    """A user path cannot be resolved safely."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"[Error] Invalid Path: {detail}")
