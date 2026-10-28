from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.agent.agentloop import Loop
from src.context.contextbuilder import ContextBuilder


def _loop_for_guards(tmp_path):
    loop = object.__new__(Loop)
    loop._workspace_root = Path(tmp_path).resolve()
    loop.workspace_revision = 0
    loop._evidence_revision = 0
    loop._read_receipts = []
    loop._failed_exact_calls = {}
    loop._failed_action_scopes = {}
    loop._guard_block_streak = 0
    loop._stagnation_stop_reason = None
    loop.metrics = {"iterations": 3}
    loop.tool = SimpleNamespace(get_tool=lambda _name: None)
    return loop


def _receipt(path, *, start_offset, end_offset, requested_end, start_line, end_line, truncated=False):
    return {
        "path": str(path.resolve()),
        "workspace_revision": 0,
        "evidence_revision": 0,
        "evidence": {
            "path": str(path.resolve()),
            "file_sha256": "f" * 64,
            "source_start_offset": start_offset,
            "source_end_offset": end_offset,
            "requested_end_offset": requested_end,
            "start_line": start_line,
            "end_line": end_line,
            "returned_end_line": end_line,
            "total_lines": 20,
            "truncated": truncated,
        },
    }


def test_read_guard_blocks_range_already_delivered(tmp_path):
    file_path = tmp_path / "target.h"
    file_path.write_text("\n".join(f"line {i}" for i in range(1, 21)))
    loop = _loop_for_guards(tmp_path)
    loop._read_receipts.append(
        _receipt(file_path, start_offset=0, end_offset=200, requested_end=200,
                 start_line=1, end_line=20)
    )
    call = SimpleNamespace(
        valid=True,
        name="read_file",
        action="inspect",
        args={"file_path": str(file_path), "start_line": 4, "end_line": 9},
    )

    blocked = Loop._covered_read_result(loop, call)

    assert blocked is not None
    assert blocked.metadata["covered_read"] is True
    assert blocked.content["prior_evidence"]["path"] == str(file_path.resolve())


def test_read_guard_allows_uncovered_range(tmp_path):
    file_path = tmp_path / "target.h"
    file_path.write_text("\n".join(f"line {i}" for i in range(1, 21)))
    loop = _loop_for_guards(tmp_path)
    loop._read_receipts.append(
        _receipt(file_path, start_offset=0, end_offset=60, requested_end=200,
                 start_line=1, end_line=5, truncated=True)
    )
    call = SimpleNamespace(
        valid=True,
        name="read_file",
        action="inspect",
        args={"file_path": str(file_path), "start_line": 6, "end_line": 10},
    )

    assert Loop._read_is_already_covered(loop, call) is None



def test_only_model_delivered_file_bytes_enter_the_read_guard(tmp_path):
    file_path = tmp_path / "target.h"
    file_path.write_text("abc", encoding="utf-8")
    loop = _loop_for_guards(tmp_path)
    receipt = {
        "success": True,
        "evidence": {
            "read_id": "read-unique-1",
            "path": str(file_path.resolve()),
            "file_sha256": "f" * 64,
            "source_start_offset": 0,
            "source_end_offset": 3,
            "requested_end_offset": 3,
            "content": "abc",
            "start_line": 1,
            "end_line": 1,
            "returned_end_line": 1,
            "total_lines": 1,
            "truncated": False,
        },
    }
    messages = [
        {
            "role": "tool",
            "tool_name": "read_file",
            "content": json.dumps(receipt),
        }
    ]

    Loop._record_delivered_read_receipts(loop, messages)
    Loop._record_delivered_read_receipts(loop, messages)

    assert len(loop._read_receipts) == 1
    call = SimpleNamespace(
        valid=True,
        name="read_file",
        action="inspect",
        args={"file_path": str(file_path), "start_line": 1, "end_line": 1},
    )
    assert Loop._read_is_already_covered(loop, call) is not None


def test_same_truncated_line_read_is_blocked_and_can_use_next_offset(tmp_path):
    file_path = tmp_path / "large.h"
    file_path.write_text("x" * 9000 + "\\nsecond line\\n", encoding="utf-8")
    loop = _loop_for_guards(tmp_path)
    loop._read_receipts.append(
        _receipt(
            file_path,
            start_offset=0,
            end_offset=8000,
            requested_end=9013,
            start_line=1,
            end_line=2,
            truncated=True,
        )
    )
    repeated = SimpleNamespace(
        valid=True,
        name="read_file",
        action="inspect",
        args={"file_path": str(file_path), "start_line": 1, "end_line": 2},
    )
    continuation = SimpleNamespace(
        valid=True,
        name="read_file",
        action="inspect",
        args={"file_path": str(file_path), "start_offset": 8000, "max_output_chars": 8000},
    )

    assert Loop._read_is_already_covered(loop, repeated) is not None
    assert Loop._read_is_already_covered(loop, continuation) is None

def test_repeated_failed_action_is_blocked_after_two_failures(tmp_path):
    loop = _loop_for_guards(tmp_path)
    call = SimpleNamespace(
        valid=True,
        name="apply_patch",
        action="modify",
        target="files:target.h",
        args={"patch": "*** Begin Patch\n*** Update File: target.h\n..."},
    )
    exact = Loop._failure_exact_key(loop, call)
    scope = Loop._failure_scope_key(loop, call)
    loop._failed_exact_calls[exact] = {"count": 2, "summary": "Patch context did not match."}
    loop._failed_action_scopes[scope] = {"count": 2, "summary": "Patch context did not match."}

    result = Loop._repeated_failed_action_result(loop, call)

    assert result is not None
    assert result.content["error"]["type"] == "repeated_failed_action"
    assert "Patch context did not match" in result.content["error"]["message"]
    assert result.metadata["loop_guard_blocked"] is True


def test_context_preserves_recovery_guidance_and_patch_diagnostics():
    builder = object.__new__(ContextBuilder)
    raw = {
        "success": False,
        "summary": "Patch context did not match.",
        "content": {
            "error": {"type": "patch_context_mismatch", "message": "No unique match."},
            "recovery": "Read the relevant section and provide a unique patch context.",
            "path": "include/target.h",
            "hunk": 2,
            "old_snippet": "struct Target {",
            "content_preview": "struct Target {\n  int old;\n};",
        },
        "evidence": {"error": {"type": "patch_context_mismatch"}},
        "metadata": {"recovery_required": True, "recovery_hint": "Change the patch."},
    }

    payload = json.loads(ContextBuilder._tool_payload(builder, raw))

    assert payload["recovery"] == "Read the relevant section and provide a unique patch context."
    assert payload["error"]["type"] == "patch_context_mismatch"
    assert payload["path"] == "include/target.h"
    assert payload["hunk"] == 2
    assert payload["recovery_hint"] == "Change the patch."
