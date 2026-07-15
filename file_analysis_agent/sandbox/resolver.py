"""Single path resolver shared by all read-only filesystem tools."""

from __future__ import annotations

from pathlib import Path

from file_analysis_agent.sandbox.errors import InvalidSandboxPathError, OutOfBoundsError


class SandboxResolver:
    """Resolve user paths against one strict, absolute workspace root."""

    def __init__(self, workspace_dir: str | Path) -> None:
        root = Path(workspace_dir)
        if not root.is_absolute():
            raise InvalidSandboxPathError("workspace root must be absolute")
        try:
            resolved = root.resolve(strict=True)
        except FileNotFoundError as exc:
            raise InvalidSandboxPathError("workspace root does not exist") from exc
        except OSError as exc:
            raise InvalidSandboxPathError(f"cannot resolve workspace root: {exc}") from exc
        if not resolved.is_dir():
            raise InvalidSandboxPathError("workspace root is not a directory")
        self._workspace_root = resolved

    @property
    def workspace_root(self) -> Path:
        return self._workspace_root

    @property
    def root(self) -> Path:
        """Compatibility alias for callers that refer to the workspace as root."""

        return self._workspace_root

    def contains(self, path: Path) -> bool:
        """Return whether an already-resolved path is within the workspace."""

        try:
            path.relative_to(self._workspace_root)
        except ValueError:
            return False
        return True

    def resolve(self, user_path: str | Path) -> Path:
        """Resolve a relative or absolute user path and enforce containment."""

        try:
            candidate = Path(user_path)
        except TypeError as exc:
            raise InvalidSandboxPathError("path must be text") from exc
        if not str(candidate):
            raise InvalidSandboxPathError("path must not be empty")
        if not candidate.is_absolute():
            candidate = self._workspace_root / candidate
        try:
            resolved = candidate.resolve(strict=False)
        except RuntimeError as exc:
            raise InvalidSandboxPathError("symlink resolution failed") from exc
        except OSError as exc:
            raise InvalidSandboxPathError(f"cannot resolve path: {exc}") from exc
        if not self.contains(resolved):
            raise OutOfBoundsError()
        return resolved

    def resolve_path(self, user_path: str | Path) -> Path:
        """Explicitly named alias for the shared path resolution operation."""

        return self.resolve(user_path)

    def relative_path(self, path: Path) -> str:
        """Return a stable workspace-relative display path."""

        try:
            return path.relative_to(self._workspace_root).as_posix() or "."
        except ValueError as exc:
            raise OutOfBoundsError() from exc


Sandbox = SandboxResolver
