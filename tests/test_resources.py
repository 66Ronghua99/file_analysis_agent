from __future__ import annotations

import json
from pathlib import Path

from file_analysis_agent.resources_loader import load_prompt, load_tool_schemas


def test_prompt_override_can_be_partial_and_schema_override_is_external(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "system.md").write_text("custom root", encoding="utf-8")
    assert load_prompt("system.md", prompt_dir) == "custom root"
    assert "read-only" in load_prompt("subagent_system.md", prompt_dir)

    custom_schema = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "custom_tool",
                    "description": "custom",
                    "parameters": {"type": "object", "properties": []},
                },
            }
        ]
    }
    (prompt_dir / "file_tools.json").write_text(
        json.dumps(custom_schema), encoding="utf-8"
    )

    assert load_tool_schemas(prompt_dir)[0]["function"]["name"] == "custom_tool"
