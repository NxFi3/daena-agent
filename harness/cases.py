from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
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


def _security_source() -> Path:
    import os

    override = os.environ.get("DAENA_SECURITY_BENCHMARK")
    if override:
        return Path(override).expanduser().resolve()
    # This checkout lives beside EVOLUTION/test101 in the user's workspace.
    return Path(__file__).resolve().parents[2] / "EVOLUTION" / "test101"


def _setup_security_audit(path: Path) -> None:
    source = _security_source()
    if not source.is_dir():
        raise FileNotFoundError(
            "Security benchmark source not found. Set DAENA_SECURITY_BENCHMARK "
            f"to the test101 directory. Checked: {source}"
        )
    shutil.copytree(
        source,
        path,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", ".pytest_cache", "*.pyc", "app.db", "tests",
            ".daena", "AGENTS.md"
        ),
    )
    visible_tests = path / "tests"
    visible_tests.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "tests" / "test_regression.py", visible_tests / "test_regression.py")

    # Keep the full verifier outside the agent workspace. The model sees only
    # normal regression tests, not the security oracle or its test names.
    oracle_tests = path.parent / "oracle-tests"
    shutil.copytree(source / "tests", oracle_tests, dirs_exist_ok=True)
    hashes = {
        "test_regression.py": hashlib.sha256(
            (source / "tests" / "test_regression.py").read_bytes()
        ).hexdigest()
    }
    (path.parent / "security-oracle.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8"
    )


def _verify_security_audit(path: Path) -> tuple[bool, str]:
    oracle_dir = path.parent / "oracle-tests"
    manifest_path = path.parent / "security-oracle.json"
    integrity_ok = False
    integrity_note = "oracle manifest missing"
    try:
        expected_hashes = json.loads(manifest_path.read_text(encoding="utf-8"))
        visible_regression = path / "tests" / "test_regression.py"
        actual_hashes = {
            "test_regression.py": hashlib.sha256(visible_regression.read_bytes()).hexdigest()
        } if visible_regression.is_file() else {}
        integrity_ok = actual_hashes == expected_hashes
        integrity_note = (
            "workspace tests unchanged"
            if integrity_ok
            else "workspace tests changed; protected oracle remains authoritative"
        )
    except (OSError, json.JSONDecodeError) as exc:
        integrity_note = f"test integrity check failed: {exc}"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(path) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(oracle_dir)],
            cwd=path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "protected security/regression suite timed out after 120 seconds"
    output = (result.stdout + "\n" + result.stderr).strip()
    note = (
        f"protected pytest exit={result.returncode}; {integrity_note}\n"
        f"{output[-1800:]}"
    )
    return result.returncode == 0 and integrity_ok, note


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
    BenchmarkCase(
        name="security_audit_test101",
        prompt=(
            "You are auditing a local Flask application for security issues. "
            "Work autonomously and stay strictly inside this workspace. Inspect "
            "the whole source tree and use safe local-only HTTP or test-client "
            "probes where useful. Discover the security weaknesses yourself; "
            "determine root causes, verify them locally, fix the root causes "
            "without removing legitimate functionality, and rerun verification. "
            "Do not modify tests to hide failures or replace functionality with "
            "dummy behavior. Continue until all significant issues are addressed. "
            "Run the complete regression and security test suite, then report "
            "each issue, root cause, fix, and verification result."
        ),
        verifier=_verify_security_audit,
        setup=_setup_security_audit,
        tags=("security", "web", "autonomous_coding", "verification"),
    ),
)
