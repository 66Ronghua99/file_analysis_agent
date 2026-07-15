"""Bounded, deterministic, read-only filesystem operations."""

from __future__ import annotations

import errno
from collections.abc import Iterator
from pathlib import Path

from file_analysis_agent.sandbox.errors import SandboxError
from file_analysis_agent.sandbox.resolver import SandboxResolver


class FileSystemTools:
    """Implement the three filesystem tools behind one resolver."""

    def __init__(
        self,
        resolver: SandboxResolver,
        max_search_matches: int = 200,
        max_read_lines: int = 300,
    ) -> None:
        if (
            isinstance(max_search_matches, bool)
            or not isinstance(max_search_matches, int)
            or isinstance(max_read_lines, bool)
            or not isinstance(max_read_lines, int)
            or max_search_matches <= 0
            or max_read_lines <= 0
        ):
            raise ValueError("filesystem limits must be positive")
        self.resolver = resolver
        self.max_search_matches = max_search_matches
        self.max_read_lines = min(max_read_lines, 300)

    def list_dir(self, path: str = ".") -> str:
        """List direct children in stable name order."""

        try:
            directory = self.resolver.resolve(path)
        except SandboxError as exc:
            return str(exc)
        try:
            if not directory.exists():
                return f"[Error] File Not Found: {path}"
            if not directory.is_dir():
                return f"[Error] Not A Directory: {path}"
            children = sorted(directory.iterdir(), key=lambda item: item.name)
            results: list[str] = []
            for child in children:
                result = self._list_entry(child)
                if result is not None:
                    results.append(result)
            return "\n".join(results) if results else "(empty directory)"
        except OSError as exc:
            return self._os_error(exc, path)

    def search_text(self, keyword: str, path: str = ".") -> str:
        """Search UTF-8 text files recursively without leaving the workspace."""

        if not isinstance(keyword, str) or not keyword:
            return "[Error] Invalid Argument: keyword must not be empty"
        try:
            root = self.resolver.resolve(path)
        except SandboxError as exc:
            return str(exc)
        try:
            if not root.exists():
                return f"[Error] File Not Found: {path}"
            if not root.is_dir():
                return f"[Error] Not A Directory: {path}"
            matches: list[str] = []
            warnings: list[str] = []
            seen_directories: set[Path] = set()
            seen_files: set[Path] = set()
            truncated = False
            candidates = list(self._walk(root, seen_directories, warnings))
            candidates.sort(key=lambda item: item[1])
            for candidate, display_path in candidates:
                if candidate in seen_files:
                    continue
                seen_files.add(candidate)
                try:
                    text = candidate.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    warnings.append(
                        f"[Error] Decode Failed: {self.resolver.relative_path(candidate)}"
                    )
                    continue
                except OSError as exc:
                    warnings.append(self._os_error(exc, self.resolver.relative_path(candidate)))
                    continue
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if keyword not in line:
                        continue
                    matches.append(f"{display_path}:{line_number}: {line}")
                    if len(matches) >= self.max_search_matches:
                        truncated = True
                        break
                if truncated:
                    break
            if truncated:
                warnings.append(
                    f"[Truncated] Search results limited to {self.max_search_matches} matches."
                )
            return "\n".join(matches + warnings) if matches or warnings else "No matches found."
        except OSError as exc:
            return self._os_error(exc, path)

    def read_file(
        self,
        filepath: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> str:
        """Read a bounded 1-based line range from a UTF-8 file."""

        if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 1:
            return "[Error] Invalid Argument: start_line must be at least 1"
        if end_line is not None and (
            isinstance(end_line, bool) or not isinstance(end_line, int) or end_line < start_line
        ):
            return "[Error] Invalid Argument: end_line must be at least start_line"
        try:
            filepath_resolved = self.resolver.resolve(filepath)
        except SandboxError as exc:
            return str(exc)
        try:
            if not filepath_resolved.exists():
                return f"[Error] File Not Found: {filepath}"
            if filepath_resolved.is_dir():
                return f"[Error] Is A Directory: {filepath}"
            lines = filepath_resolved.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            return f"[Error] Decode Failed: {filepath}"
        except OSError as exc:
            return self._os_error(exc, filepath)

        if lines and start_line > len(lines):
            return f"[Error] Invalid Range: start_line {start_line} is beyond end of file"

        requested_end = len(lines) if end_line is None else end_line
        limit_end = start_line + self.max_read_lines - 1
        was_clamped = requested_end > limit_end
        actual_end = min(requested_end, limit_end)
        selected = lines[start_line - 1 : actual_end]
        output = [f"{number} | {line}" for number, line in enumerate(selected, start=start_line)]
        if was_clamped:
            output.append(
                f"[Truncated] Read limited to {self.max_read_lines} lines starting at {start_line}."
            )
        return "\n".join(output) if output else "(empty range)"

    def _list_entry(self, child: Path) -> str | None:
        try:
            target = child.resolve(strict=False)
            if not self.resolver.contains(target):
                return f"{child.name} [Error] Permission Denied: out of bounds"
            marker = "/" if target.is_dir() else ""
            return f"{child.name}{marker}"
        except OSError as exc:
            return self._os_error(exc, child.name)

    def _walk(
        self,
        root: Path,
        seen_directories: set[Path],
        warnings: list[str],
    ) -> Iterator[tuple[Path, str]]:
        pending: list[tuple[Path, str]] = [(root, self.resolver.relative_path(root))]
        while pending:
            directory, display_directory = pending.pop(0)
            resolved_directory = directory.resolve(strict=False)
            if not self.resolver.contains(resolved_directory):
                warnings.append("[Error] Permission Denied: out of bounds")
                continue
            if resolved_directory in seen_directories:
                continue
            seen_directories.add(resolved_directory)
            try:
                children = sorted(directory.iterdir(), key=lambda item: item.name)
            except OSError as exc:
                warnings.append(self._os_error(exc, display_directory))
                continue
            for child in children:
                try:
                    target = child.resolve(strict=False)
                    if not self.resolver.contains(target):
                        warnings.append(f"[Error] Permission Denied: out of bounds: {child.name}")
                        continue
                    display_path = self.resolver.relative_path(child)
                    if target.is_dir():
                        pending.append((child, display_path))
                    elif target.is_file():
                        yield target, display_path
                except OSError as exc:
                    warnings.append(self._os_error(exc, child.name))

    @staticmethod
    def _os_error(error: OSError, path: str | Path) -> str:
        if isinstance(error, FileNotFoundError):
            return f"[Error] File Not Found: {path}"
        if isinstance(error, PermissionError) or error.errno == errno.EACCES:
            return f"[Error] Permission Denied: {path}"
        if isinstance(error, IsADirectoryError):
            return f"[Error] Is A Directory: {path}"
        if isinstance(error, NotADirectoryError):
            return f"[Error] Not A Directory: {path}"
        return f"[Error] Filesystem Failure: {path}: {error}"
