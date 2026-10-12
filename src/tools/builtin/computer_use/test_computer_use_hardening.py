"""Regression tests for the Computer Use hardening pass.

Put this file in tests/. It uses only the repo's own modules plus pytest.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.models.ToolResult import ToolResult
from src.tools.builtin.computer_use import act as act_module
from src.tools.builtin.computer_use import computer_use as cu_module
from src.tools.builtin.computer_use import driver
from src.tools.builtin.computer_use.act import ComputerActTool
from src.tools.builtin.computer_use.computer_use import ComputerUseTool
from src.tools.builtin.computer_use.driver import CuaDriverError, pointer_state_path


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still answers kill(0); treat it as dead if /proc says so.
    try:
        return "Z" not in Path(f"/proc/{pid}/stat").read_text().split(") ")[-1][:2]
    except OSError:
        return False


def _fake_driver(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "cua-driver"
    script.write_text("#!/bin/sh\n" + body + "\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


# ------------------------------------------------------------------ THIRD ISSUE
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_run_bounded_kills_helper_and_its_children_on_timeout(tmp_path):
    pidfile = tmp_path / "child.pid"
    command = ["sh", "-c", f"sleep 60 & echo $! > {pidfile}; wait"]
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        driver._run_bounded(command, timeout=1.0)
    assert time.monotonic() - started < 8
    child = int(pidfile.read_text().strip())
    time.sleep(0.3)
    assert not _alive(child), "helper child process was left behind after timeout"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
def test_call_driver_times_out_instead_of_hanging(tmp_path, monkeypatch):
    fake = _fake_driver(tmp_path, "sleep 60")
    monkeypatch.setattr(driver, "driver_executable", lambda: str(fake))
    started = time.monotonic()
    with pytest.raises(CuaDriverError, match="timed out"):
        driver.call_driver("list_apps", {}, timeout=1.0)
    assert time.monotonic() - started < 8


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
def test_call_driver_never_inherits_stdin(tmp_path, monkeypatch):
    # `cat` blocks forever if stdin is inherited from an interactive terminal/pipe.
    fake = _fake_driver(tmp_path, "cat >/dev/null; echo '{}'")
    monkeypatch.setattr(driver, "driver_executable", lambda: str(fake))
    started = time.monotonic()
    assert driver.call_driver("list_apps", {}, timeout=5.0) == "{}"
    assert time.monotonic() - started < 4


def test_long_text_is_typed_in_bounded_chunks(tmp_path, monkeypatch):
    calls: list[dict] = []

    def fake_call(name, args=None, *, timeout=30.0):
        calls.append({"name": name, "args": dict(args or {}), "timeout": timeout})
        return "{}"

    monkeypatch.setattr(act_module, "call_driver", fake_call)
    monkeypatch.setattr(act_module.time, "sleep", lambda *_: None)
    tool = ComputerActTool()
    tool.set_workspace(tmp_path)
    text = "x" * 1000
    result = tool.execute(action="type_text", text=text)
    assert result.success
    sent = [c["args"]["text"] for c in calls if c["name"] == "type_text"]
    assert "".join(sent) == text
    assert max(len(part) for part in sent) <= act_module._TYPE_CHUNK_CHARS
    assert all(c["timeout"] <= 20 for c in calls)


def test_chunked_typing_failure_reports_partial_text_and_never_retries(tmp_path, monkeypatch):
    calls: list[str] = []

    def fake_call(name, args=None, *, timeout=30.0):
        calls.append(args["text"])
        if len(calls) == 2:
            raise CuaDriverError("Resource temporarily unavailable (os error 11)")
        return "{}"

    monkeypatch.setattr(act_module, "call_driver", fake_call)
    monkeypatch.setattr(act_module.time, "sleep", lambda *_: None)
    tool = ComputerActTool()
    tool.set_workspace(tmp_path)
    result = tool.execute(action="type_text", text="y" * 1000)
    assert not result.success
    assert len(calls) == 2, "a failed chunk must not be retried"
    message = result.content["error"]["message"]
    assert "partial" in message and "os error 11" in message
    assert "recovery_hint" in result.metadata


def test_short_text_still_uses_a_single_call(tmp_path, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        act_module, "call_driver",
        lambda name, args=None, *, timeout=30.0: calls.append(args) or "{}",
    )
    tool = ComputerActTool()
    tool.set_workspace(tmp_path)
    assert tool.execute(action="type_text", text="hello").success
    assert len(calls) == 1 and calls[0]["text"] == "hello"


# ------------------------------------------------------------------ FOURTH ISSUE
def test_daemon_that_exits_immediately_fails_fast_with_a_clear_error(tmp_path, monkeypatch):
    fake = _fake_driver(tmp_path, "exit 3")
    monkeypatch.setattr(driver, "_daemon_status", lambda exe: (False, "daemon is not running"))
    monkeypatch.setattr(driver, "_desktop_session_environment", lambda: dict(os.environ))
    monkeypatch.setattr(driver.Path, "home", classmethod(lambda cls: tmp_path))
    started = time.monotonic()
    with pytest.raises(CuaDriverError, match="exited immediately with code 3"):
        driver._ensure_daemon_running(str(fake))
    assert time.monotonic() - started < driver._DAEMON_START_TIMEOUT_SECONDS


# ------------------------------------------------------------------ PRIMARY BUG
def _prepare_click_state(tool: ComputerUseTool, role: str, selected: bool, checked=None):
    record = {"role": role, "label": "Rectangle", "selected": selected, "actions": ["click"]}
    if checked is not None:
        record["checked"] = checked
    state = {
        "latest_element_ids": {"1": "tok-1"},
        "latest_accessibility_elements": {"tok-1": record},
        "latest_computer_use_target": {},
    }
    path = pointer_state_path(tool.workspace_root, tool._observe._session_label())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")


def _run_click(tmp_path, monkeypatch, *, role, before, after_elements, checked=None):
    monkeypatch.setattr(cu_module, "ensure_driver_running", lambda *_a, **_k: None)
    tool = ComputerUseTool()
    tool.set_workspace(str(tmp_path))
    _prepare_click_state(tool, role, before, checked)
    monkeypatch.setattr(
        tool, "_act_once",
        lambda action, **kw: ToolResult(success=True, name="computer_act", content={}, metadata={}, summary="dispatched"),
    )
    monkeypatch.setattr(
        tool, "_capture",
        lambda **kw: ToolResult(
            success=True, name="computer_use",
            content={"elements": after_elements, "window": {}, "element_count": len(after_elements)},
            metadata={}, summary="captured",
        ),
    )
    return tool.execute(action="click", element=1)


def test_toolbar_button_that_only_reports_selected_is_not_treated_as_activated(tmp_path, monkeypatch):
    """The Paint failure: Rectangle shows selected=True, app status still says Pencil."""
    result = _run_click(
        tmp_path, monkeypatch, role="push button", before=False,
        after_elements=[{"id": 1, "role": "push button", "label": "Rectangle", "selected": True}],
    )
    assert result.success is False
    assert result.metadata["effect_verified"] is False
    assert "recovery_hint" in result.metadata
    assert "dependent" in result.metadata["recovery_hint"].lower()
    assert result.content["verification_status"] == "semantic_state_unavailable"


def test_selected_only_button_that_is_not_selected_afterwards_is_not_flagged(tmp_path, monkeypatch):
    result = _run_click(
        tmp_path, monkeypatch, role="push button", before=False,
        after_elements=[{"id": 1, "role": "push button", "label": "Rectangle", "selected": False}],
    )
    assert result.content["verification_status"] != "semantic_state_unavailable"


def test_selected_is_still_accepted_as_state_for_tabs(tmp_path, monkeypatch):
    result = _run_click(
        tmp_path, monkeypatch, role="page tab", before=False,
        after_elements=[{"id": 1, "role": "page tab", "label": "Rectangle", "selected": True}],
    )
    assert result.content["verification_status"] != "semantic_state_unavailable"


def test_real_checked_transition_is_still_verified(tmp_path, monkeypatch):
    result = _run_click(
        tmp_path, monkeypatch, role="toggle button", before=False, checked=False,
        after_elements=[{"id": 1, "role": "toggle button", "label": "Rectangle", "checked": True, "selected": True}],
    )
    assert result.content["verification_status"] == "semantic_verified"
    assert result.success is True


def test_checked_state_unchanged_after_click_fails_closed(tmp_path, monkeypatch):
    result = _run_click(
        tmp_path, monkeypatch, role="toggle button", before=False, checked=False,
        after_elements=[{"id": 1, "role": "toggle button", "label": "Rectangle", "checked": False, "selected": True}],
    )
    assert result.success is False
    assert result.content["verification_status"] == "semantic_state_unchanged"


def test_selected_control_state_requires_a_unique_match():
    elements = [
        {"role": "push button", "label": "A", "selected": True},
        {"role": "push button", "label": "A", "selected": False},
    ]
    assert ComputerUseTool._selected_control_state(elements, "push button", "A") is None
    assert ComputerUseTool._selected_control_state(elements[:1], "push button", "A") is True


# ------------------------------------------------------------------ ROUND 2 (live-log findings)
from src.tools.builtin.computer_use.inspect import ComputerInspectTool


def _click_with_hash_sequence(tmp_path, monkeypatch, hashes):
    monkeypatch.setattr(cu_module, "ensure_driver_running", lambda *_a, **_k: None)
    monkeypatch.setattr(cu_module.time, "sleep", lambda *_: None)
    tool = ComputerUseTool()
    tool.set_workspace(str(tmp_path))
    _prepare_click_state(tool, "push button", False)
    path = pointer_state_path(tool.workspace_root, tool._observe._session_label())
    state = json.loads(path.read_text())
    state["latest_screenshot_hash"] = "BEFORE"
    path.write_text(json.dumps(state))
    dispatched = []
    monkeypatch.setattr(
        tool, "_act_once",
        lambda action, **kw: dispatched.append(action) or ToolResult(
            success=True, name="computer_act", content={}, metadata={}, summary="dispatched"),
    )
    captures = []

    def fake_capture(**kw):
        value = hashes[min(len(captures), len(hashes) - 1)]
        captures.append(value)
        current_path, current = tool._current_state()
        current["latest_screenshot_hash"] = value
        tool._save_state(current_path, current)
        return ToolResult(success=True, name="computer_use",
                          content={"elements": [], "window": {}, "screenshot_hash": value},
                          metadata={}, summary="captured")

    monkeypatch.setattr(tool, "_capture", fake_capture)
    result = tool.execute(action="click", element=1)
    return result, captures, dispatched


def test_late_repaint_is_not_reported_as_no_change(tmp_path, monkeypatch):
    result, captures, dispatched = _click_with_hash_sequence(
        tmp_path, monkeypatch, ["BEFORE", "BEFORE", "AFTER"])
    assert captures == ["BEFORE", "BEFORE", "AFTER"]
    assert result.success is True
    assert result.content["capture_after"]["screen_hash_changed"] is True
    assert dispatched == ["click"]


def test_truly_unchanged_screen_still_fails_and_input_is_never_resent(tmp_path, monkeypatch):
    result, captures, dispatched = _click_with_hash_sequence(tmp_path, monkeypatch, ["BEFORE"])
    assert len(captures) == 1 + len(ComputerUseTool._SETTLE_DELAYS)
    assert result.success is False
    assert dispatched == ["click"], "waiting must never re-dispatch the input"


def test_control_at_point_picks_the_smallest_containing_control():
    elements = [
        {"label": "Toolbar", "bounds": {"x": 0, "y": 0, "w": 500, "h": 100}},
        {"label": "Rectangle", "bounds": {"x": 280, "y": 40, "w": 30, "h": 30}},
    ]
    assert ComputerUseTool._control_at_point(elements, 290, 50) == "Rectangle"
    assert ComputerUseTool._control_at_point(elements, 10, 10) == "Toolbar"
    assert ComputerUseTool._control_at_point(elements, 900, 900) is None


def _payload(states):
    return {
        "window_title": "Paint",
        "window_bounds": {"x": 0, "y": 0, "width": 800, "height": 600},
        "elements": [
            {"element_index": 0, "parent_index": None, "role": "frame", "label": "Paint",
             "element_token": "t0", "frame": {"x": 0, "y": 0, "w": 800, "h": 600}},
            {"element_index": 1, "parent_index": 0, "role": "check box", "label": "Rectangle",
             "element_token": "t1", "actions": ["click"], "states": states,
             "frame": {"x": 10, "y": 10, "w": 30, "h": 30}},
        ],
    }


def test_checked_is_read_from_a_state_list():
    def checked(states):
        rows = ComputerInspectTool._rank_elements(_payload(states), "")
        return next(r for r in rows if r["element_token"] == "t1").get("checked", "absent")

    assert checked(["checkable", "checked", "focused"]) is True
    assert checked(["checkable", "focused"]) is False
    assert checked(["focused"]) == "absent"  # unknown stays unknown -> still fails closed


def test_elements_text_does_not_present_selected_as_proof(tmp_path, monkeypatch):
    monkeypatch.setattr(cu_module, "ensure_driver_running", lambda *_a, **_k: None)
    tool = ComputerUseTool()
    tool.set_workspace(str(tmp_path))
    window = {"pid": 11, "window_id": 22, "title": "Paint", "app_name": "Paint",
              "x": 0, "y": 0, "width": 800, "height": 600, "is_on_screen": True}
    rows = [
        {"element_token": "a", "role": "check box", "label": "Pencil", "selected": True,
         "frame": {"x": 10, "y": 10, "w": 30, "h": 30}},
        {"element_token": "b", "role": "check box", "label": "Brush", "selected": False, "checked": True,
         "frame": {"x": 50, "y": 10, "w": 30, "h": 30}},
        {"element_token": "c", "role": "push button", "label": "Save",
         "frame": {"x": 90, "y": 10, "w": 30, "h": 30}},
    ]
    monkeypatch.setattr(tool._inspect, "_windows", lambda: [window])
    monkeypatch.setattr(
        tool._inspect, "execute",
        lambda **kw: ToolResult(success=True, name="computer_inspect",
                                content={"window": dict(window, app="Paint"), "elements": rows},
                                metadata={}, summary="ok"),
    )
    monkeypatch.setattr(
        tool._observe, "execute",
        lambda **kw: ToolResult(success=True, name="computer_observe",
                                content={"width": 800, "height": 600, "coordinate_frame": "window",
                                         "capture_id": "cap", "screenshot_hash": "h"},
                                metadata={}, summary="shot"),
    )
    result = tool._capture(mode="som", app="Paint")
    text = result.content["elements_text"]
    assert "NOT proof" in text           # selected-only must be labelled as non-proof
    assert "[CHECKED]" in text           # a real checked state is shown as such
    assert "[SELECTED]" not in text
