from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    prompt: str
    verifier: Callable[[Path], tuple[bool, str]]
    setup: Callable[[Path], None] | None = None
    tags: tuple[str, ...] = ()


def _verify_file(path: Path, expected: str) -> tuple[bool, str]:
    try:
        value = (path / "result.txt").read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"missing result.txt: {exc}"

    ok = value.strip() == expected
    return ok, "result.txt matches expected content" if ok else "result.txt mismatch"


def _setup_empty(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


CASES = (
    BenchmarkCase(
        name="basic_file_creation",
        prompt=(
            "Create a file named result.txt in the workspace containing exactly "
            "the single line: baseline-ok. Verify it by reading the file."
        ),
        verifier=lambda path: _verify_file(path, "baseline-ok"),
        setup=_setup_empty,
        tags=("tool_use", "filesystem"),
    ),
    BenchmarkCase(
        name="basic_file_repair",
        prompt=(
            "Create result.txt with the wrong content, then repair it so it "
            "contains exactly: repaired. Verify the final file."
        ),
        verifier=lambda path: _verify_file(path, "repaired"),
        setup=lambda path: (
            path.mkdir(parents=True, exist_ok=True),
            (path / "result.txt").write_text("wrong", encoding="utf-8"),
        ),
        tags=("tool_use", "recovery"),
    ),
)
