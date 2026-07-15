from __future__ import annotations

import json
from pathlib import Path

from file_analysis_agent.agent.models import ToolCall
from file_analysis_agent.sandbox.resolver import SandboxResolver
from file_analysis_agent.tools.filesystem import FileSystemTools
from file_analysis_agent.tools.registry import ToolRegistry


def test_resolver_accepts_in_workspace_paths_and_rejects_escape(workspace: Path) -> None:
    resolver = SandboxResolver(workspace)

    assert resolver.resolve("nested/../a.txt") == workspace / "a.txt"
    assert resolver.resolve(workspace / "a.txt") == workspace / "a.txt"
    assert resolver.resolve("nested/notes.txt") == workspace / "nested" / "notes.txt"
    assert "Permission Denied: out of bounds" in str(_resolve_error(resolver, "../outside"))
    assert "Permission Denied: out of bounds" in str(_resolve_error(resolver, "/tmp"))


def test_symlink_resolution_accepts_internal_and_rejects_external(
    workspace: Path, tmp_path: Path
) -> None:
    internal = workspace / "inside-link.txt"
    internal.symlink_to(workspace / "a.txt")
    external_target = tmp_path.parent / "outside-file.txt"
    external_target.write_text("secret", encoding="utf-8")
    external = workspace / "outside-link.txt"
    external.symlink_to(external_target)
    resolver = SandboxResolver(workspace)

    assert resolver.resolve(internal).read_text(encoding="utf-8") == "first\nsecond\n"
    assert "Permission Denied: out of bounds" in str(_resolve_error(resolver, external))


def test_list_dir_is_sorted_and_marks_directories(workspace: Path) -> None:
    tools = FileSystemTools(SandboxResolver(workspace))

    assert tools.list_dir(".").splitlines() == ["a.txt", "nested/", "z.txt"]
    assert tools.list_dir("missing") == "[Error] File Not Found: missing"
    assert tools.list_dir("a.txt") == "[Error] Not A Directory: a.txt"


def test_search_is_recursive_deterministic_and_skips_external_symlink(
    workspace: Path, tmp_path: Path
) -> None:
    external_dir = tmp_path.parent / "outside-dir"
    external_dir.mkdir()
    (external_dir / "secret.txt").write_text("needle secret", encoding="utf-8")
    (workspace / "external-dir").symlink_to(external_dir, target_is_directory=True)
    tools = FileSystemTools(SandboxResolver(workspace))

    result = tools.search_text("needle")
    lines = result.splitlines()
    assert lines[:2] == ["nested/notes.txt:2: needle nested", "z.txt:2: needle root"]
    assert "secret" not in result
    assert "Permission Denied: out of bounds" in result


def test_search_reports_decode_failure_and_caps_matches(workspace: Path) -> None:
    (workspace / "bad.bin").write_bytes(b"\xff\xfe")
    for index in range(5):
        (workspace / f"match-{index}.txt").write_text("hit\n", encoding="utf-8")
    tools = FileSystemTools(SandboxResolver(workspace), max_search_matches=3)

    result = tools.search_text("hit")
    assert result.count(":1: hit") == 3
    assert "[Truncated] Search results limited to 3 matches." in result
    assert "[Error] Decode Failed: bad.bin" in tools.search_text("anything")


def test_read_file_numbers_lines_and_clamps_range(workspace: Path) -> None:
    (workspace / "long.txt").write_text("\n".join(str(i) for i in range(1, 305)), encoding="utf-8")
    tools = FileSystemTools(SandboxResolver(workspace))

    assert tools.read_file("a.txt", 1, 2) == "1 | first\n2 | second"
    result = tools.read_file("long.txt", 1, 304)
    assert len([line for line in result.splitlines() if " | " in line]) == 300
    assert "[Truncated] Read limited to 300 lines" in result
    assert tools.read_file("a.txt", 0) == "[Error] Invalid Argument: start_line must be at least 1"
    assert "end_line must be at least start_line" in tools.read_file("a.txt", 3, 2)
    assert tools.read_file("nested") == "[Error] Is A Directory: nested"
    assert "Permission Denied: out of bounds" in tools.read_file("../outside")


def test_read_file_reports_utf8_failure(workspace: Path) -> None:
    (workspace / "invalid.txt").write_bytes(b"ok\n\xff")

    result = FileSystemTools(SandboxResolver(workspace)).read_file("invalid.txt")

    assert result == "[Error] Decode Failed: invalid.txt"


def test_registry_validates_arguments_and_preserves_call_id(workspace: Path) -> None:
    tools = FileSystemTools(SandboxResolver(workspace))
    registry = ToolRegistry.filesystem(tools)

    observation = registry.dispatch(
        ToolCall("tool-7", "read_file", json.dumps({"filepath": "a.txt"}))
    )
    assert observation.tool_call_id == "tool-7"
    assert observation.content.startswith("1 | first")
    assert "Unknown Tool" in registry.dispatch(ToolCall("x", "rm", "{}")).content
    assert "Invalid JSON Arguments" in registry.dispatch(ToolCall("x", "read_file", "{")).content
    assert "missing required field" in registry.dispatch(ToolCall("x", "read_file", "{}")).content
    assert not registry.has_tool("call_subagent")


def _resolve_error(resolver: SandboxResolver, path: str | Path) -> Exception:
    try:
        resolver.resolve(path)
    except Exception as exc:  # The helper only captures the expected boundary exception.
        return exc
    raise AssertionError("path unexpectedly resolved")
