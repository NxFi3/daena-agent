import hashlib
import json
import threading
import time

from computer_use import ComputerUseTool
from computer_use.runtime import atomic_json, desktop_lease
from conftest import capture


def test_another_agent_invalidates_old_desktop_observation(engine):
    tool, backend = engine
    other = ComputerUseTool(
        backend, workspace=tool.workspace_root / "other", lock_path=tool.lock_path
    )
    first = capture(tool)
    second = capture(other)
    assert other.execute("click", snapshot_id=second, target="Save", seconds=0).success
    result = tool.execute("click", snapshot_id=first, target="Save")
    assert result.content["error"]["code"] == "stale_snapshot"
    assert len(backend.calls) == 1


def test_desktop_lease_prevents_capture_input_interleaving(engine):
    tool, backend = engine
    other = ComputerUseTool(
        backend, workspace=tool.workspace_root / "other", lock_path=tool.lock_path
    )
    sid = capture(tool)
    started = threading.Event()
    calls = []
    original = backend.dispatch

    def slow_dispatch(session, action, arguments):
        calls.append("input-start")
        started.set()
        time.sleep(0.1)
        value = original(session, action, arguments)
        calls.append("input-end")
        return value

    backend.dispatch = slow_dispatch
    first = threading.Thread(
        target=lambda: tool.execute("click", snapshot_id=sid, target="Save", seconds=0)
    )
    first.start()
    assert started.wait(2)
    capture(other)
    calls.append("other-capture")
    first.join(timeout=3)
    assert calls == ["input-start", "input-end", "other-capture"]


def test_cancellation_while_waiting_for_desktop_lock(engine):
    tool, backend = engine
    held = threading.Event()
    release = threading.Event()

    def holder():
        with desktop_lease(tool.lock_path):
            held.set()
            release.wait(2)

    thread = threading.Thread(target=holder)
    thread.start()
    assert held.wait(1)
    timer = threading.Timer(0.1, tool.cancel)
    timer.start()
    start = time.monotonic()
    result = tool.execute("capture")
    release.set()
    thread.join()
    timer.join()
    assert time.monotonic() - start < 1
    assert result.content["error"]["code"] == "cancelled"
    assert backend.captures == 0


def test_crash_receipt_does_not_replay_unknown_action(engine):
    tool, backend = engine
    sid = capture(tool)
    request = {"action": "click", "snapshot_id": sid, "target": "Save", "request_id": "crashed"}
    digest = hashlib.sha256(
        json.dumps(request, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    path = tool.root / "receipts" / (hashlib.sha256(b"crashed").hexdigest() + ".json")
    atomic_json(path, {"digest": digest, "status": "started"})
    result = tool.execute(**request)
    assert result.content["error"]["code"] == "outcome_unknown"
    assert not backend.calls


def test_daena_parent_stop_token_without_wait_method(engine):
    tool, _ = engine

    class Parent:
        def is_set(self):
            return True

    class Manager:
        stop_event = Parent()

    tool.set_process_manager(Manager())
    result = tool.execute("capture")
    assert result.content["error"]["code"] == "cancelled"
