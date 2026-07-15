"""Load editable prompts and function schemas from package resources."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from file_analysis_agent.errors import ResourceError

PROMPT_NAMES = frozenset({"system.md", "subagent_system.md", "blackboard_update.md"})


def load_prompt(name: str, prompt_dir: str | Path | None = None) -> str:
    """Load a default prompt, with an optional editable directory override."""

    if name not in PROMPT_NAMES:
        raise ResourceError(f"unknown prompt resource: {name}")
    override = _find_override(name, prompt_dir)
    try:
        if override is not None:
            return override.read_text(encoding="utf-8")
        return (
            files("file_analysis_agent.resources.prompts")
            .joinpath(name)
            .read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError) as exc:
        raise ResourceError(f"cannot load prompt {name}: {exc}") from exc


def load_tool_schemas(prompt_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the external Function Calling schema list."""

    override = _find_override("file_tools.json", prompt_dir)
    try:
        if override is not None:
            value = json.loads(override.read_text(encoding="utf-8"))
        else:
            value = json.loads(
                files("file_analysis_agent.resources.tools")
                .joinpath("file_tools.json")
                .read_text(encoding="utf-8")
            )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResourceError(f"cannot load tool schemas: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("tools"), list):
        raise ResourceError("tool schema resource must contain a tools list")
    return [item for item in value["tools"] if isinstance(item, dict)]


def _find_override(name: str, prompt_dir: str | Path | None) -> Path | None:
    if prompt_dir is None:
        return None
    root = Path(prompt_dir)
    candidates = [root / name]
    if name == "file_tools.json":
        candidates.extend([root / "tools" / name, root / "prompts" / name])
    else:
        candidates.append(root / "prompts" / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
