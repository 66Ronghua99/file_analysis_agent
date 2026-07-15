from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "notes.txt").write_text("alpha\nneedle nested\n", encoding="utf-8")
    (tmp_path / "z.txt").write_text("zeta\nneedle root\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("first\nsecond\n", encoding="utf-8")
    return tmp_path
