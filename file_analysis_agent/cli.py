"""One-shot command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import TextIO

from file_analysis_agent.agent.loop import Agent
from file_analysis_agent.clients.factory import create_client
from file_analysis_agent.config import AgentConfig
from file_analysis_agent.errors import ConfigurationError, FileAnalysisAgentError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze local files with a read-only Agent.")
    parser.add_argument("--workspace", required=True, help="Absolute workspace directory.")
    parser.add_argument(
        "--provider",
        required=True,
        help="Provider: openrouter, vllm, litellm, or fake.",
    )
    parser.add_argument("--model", required=True, help="Provider model name.")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint URL.")
    parser.add_argument(
        "--api-mode",
        default="chat_completion",
        choices=("chat_completion", "responses"),
        help="Provider API shape (default: chat_completion).",
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--context-window", type=int, default=8192)
    parser.add_argument("--prompt-dir", help="Directory overriding prompts and file_tools.json.")
    parser.add_argument(
        "task", nargs="?", help="One task; otherwise read exactly one task from stdin."
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one task and return a process-style exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    input_stream = stdin or sys.stdin
    output_stream = stdout or sys.stdout
    error_stream = stderr or sys.stderr
    try:
        task = _task_from_argument_or_stdin(args.task, input_stream)
        config = AgentConfig(
            workspace_dir=args.workspace,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_mode=args.api_mode,
            max_steps=args.max_steps,
            context_window=args.context_window,
            prompt_dir=args.prompt_dir,
        )
        client = create_client(config)
        result = Agent(config, client=client).run(task)
    except (ConfigurationError, FileAnalysisAgentError) as exc:
        print(f"file-analysis-agent: error: {exc}", file=error_stream)
        return 2
    except Exception as exc:
        print(f"file-analysis-agent: error: {type(exc).__name__}: {exc}", file=error_stream)
        return 1
    print(result, file=output_stream)
    return 0


def _task_from_argument_or_stdin(task: str | None, stdin: TextIO) -> str:
    if task is not None:
        if not stdin.isatty():
            extra = stdin.read()
            if extra.strip():
                raise ConfigurationError("provide a task argument or stdin input, not both")
        if not task.strip():
            raise ConfigurationError("task must not be empty")
        return task.strip()
    value = stdin.read().strip()
    if not value:
        raise ConfigurationError("provide one task argument or one non-empty stdin task")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
