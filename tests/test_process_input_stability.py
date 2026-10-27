"""Backpressured stdin must not block cancellation or process cleanup."""

import sys
import threading
import time

from src.tools.builtin.command_exec.process_manager import ProcessManager


def start_unread(manager, tmp_path):
    script = tmp_path / "unread.py"
    script.write_text('import time\nprint("ready", flush=True)\ntime.sleep(30)\n')
    result = manager.start(
        command=[sys.executable, str(script)],
        workdir=tmp_path,
        yield_time_ms=50,
        max_output_chars=500,
    )
    return result["process_id"]


def test_stop_can_interrupt_blocked_stdin(tmp_path):
    manager = ProcessManager()
    ident = start_unread(manager, tmp_path)
    results = []
    writer = threading.Thread(
        target=lambda: results.append(
            manager.write(process_id=ident, input_text="x" * 2_000_000)
        ),
        daemon=True,
    )
    try:
        writer.start()
        time.sleep(0.15)
        before = time.monotonic()
        manager.stop(process_id=ident)
        writer.join(2)
        assert time.monotonic() - before < 3
        assert not writer.is_alive()
        assert results[0]["status"] != "accepted"
        assert results[0]["bytes_written"] < 2_000_000
    finally:
        manager.close(discard_output=True)


def test_stdin_timeout_and_run_cancellation_stop_child(tmp_path):
    for cancel in (False, True):
        manager = ProcessManager()
        manager.STDIN_WRITE_TIMEOUT = 0.15
        manager.stop_event = threading.Event()
        ident = start_unread(manager, tmp_path)
        timer = threading.Timer(0.08, manager.stop_event.set) if cancel else None
        try:
            if timer:
                timer.start()
            result = manager.write(process_id=ident, input_text="x" * 2_000_000)
            assert result["status"] == ("cancelled" if cancel else "timed_out")
            assert result["bytes_written"] < result["bytes_requested"]
            assert (
                manager.poll(process_id=ident, wait_ms=0, max_output_chars=100)[
                    "status"
                ]
                != "running"
            )
        finally:
            if timer:
                timer.cancel()
            manager.close(discard_output=True)


def test_cancelled_run_never_dispatches_input(tmp_path):
    manager = ProcessManager()
    ident = start_unread(manager, tmp_path)
    manager.stop_event = threading.Event()
    manager.stop_event.set()
    try:
        result = manager.write(process_id=ident, input_text="do not send\n")
        assert result["delivery"] == "not_sent" and result["bytes_written"] == 0
    finally:
        manager.close(discard_output=True)


def test_start_does_not_reuse_a_child_that_exited_since_last_poll(tmp_path):
    manager = ProcessManager()
    script = tmp_path / "exit.py"
    script.write_text("import time\ntime.sleep(.1)\n")
    arguments = dict(
        command=[sys.executable, str(script)],
        workdir=tmp_path,
        yield_time_ms=1,
        max_output_chars=100,
    )
    try:
        first = manager.start(**arguments)
        manager._processes[first["process_id"]].process.wait(timeout=3)
        second = manager.start(**arguments)
        assert not second.get("reused_existing_process")
        assert second["pid"] != first["pid"]
    finally:
        manager.close(discard_output=True)
