from __future__ import annotations

from io import StringIO
from pathlib import Path

from file_analysis_agent.cli import main


def _args(workspace: Path, *extra: str) -> list[str]:
    return [
        "--workspace",
        str(workspace),
        "--provider",
        "fake",
        "--model",
        "fake",
        *extra,
    ]


def test_cli_argument_task_writes_only_final_text_to_stdout(workspace: Path) -> None:
    stdout = StringIO()
    stderr = StringIO()

    code = main(_args(workspace, "inspect files"), stdin=StringIO(), stdout=stdout, stderr=stderr)

    assert code == 0
    assert stdout.getvalue() == "Fake client completed the task.\n"
    assert stderr.getvalue() == ""


def test_cli_reads_one_task_from_stdin(workspace: Path) -> None:
    stdout = StringIO()

    code = main(
        _args(workspace), stdin=StringIO("inspect from stdin\n"), stdout=stdout, stderr=StringIO()
    )

    assert code == 0
    assert stdout.getvalue().endswith("\n")


def test_cli_rejects_both_task_sources_and_invalid_configuration(workspace: Path) -> None:
    stderr = StringIO()
    code = main(
        _args(workspace, "argument task"),
        stdin=StringIO("stdin task"),
        stdout=StringIO(),
        stderr=stderr,
    )
    assert code == 2
    assert "argument or stdin" in stderr.getvalue()

    bad_stderr = StringIO()
    bad_code = main(
        ["--workspace", str(workspace), "--provider", "unknown", "--model", "demo", "task"],
        stdin=StringIO(),
        stdout=StringIO(),
        stderr=bad_stderr,
    )
    assert bad_code == 2
    assert "unknown provider" in bad_stderr.getvalue()


def test_cli_rejects_missing_or_relative_workspace(tmp_path: Path) -> None:
    stderr = StringIO()
    code = main(
        ["--workspace", str(tmp_path / "missing"), "--provider", "fake", "--model", "fake", "task"],
        stdin=StringIO(),
        stdout=StringIO(),
        stderr=stderr,
    )
    assert code == 2
    assert "does not exist" in stderr.getvalue()

    relative_stderr = StringIO()
    relative_code = main(
        ["--workspace", ".", "--provider", "fake", "--model", "fake", "task"],
        stdin=StringIO(),
        stdout=StringIO(),
        stderr=relative_stderr,
    )
    assert relative_code == 2
    assert "absolute" in relative_stderr.getvalue()
