import sys
import threading

from src.tools.builtin.command_exec.process_manager import ProcessManager


def test_cancel_during_initial_process_wait(tmp_path):
    script = tmp_path / "slow.py"
    script.write_text("import time\nprint('started', flush=True)\ntime.sleep(60)\n")
    manager = ProcessManager()
    stop = threading.Event()
    manager.stop_event = stop
    result = []
    thread = threading.Thread(
        target=lambda: result.append(
            manager.start(
                command=[sys.executable, str(script)],
                workdir=tmp_path,
                yield_time_ms=30000,
                max_output_chars=1000,
            )
        )
    )
    try:
        thread.start()
        stop.set()
        thread.join(2)
        assert not thread.is_alive()
        assert result[0]["status"] != "running"
    finally:
        stop.set()
        thread.join(2)
        manager.close()


def test_cancel_during_process_poll(tmp_path):
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(60)\n")
    manager = ProcessManager()
    stop = threading.Event()
    manager.stop_event = stop
    try:
        initial = manager.start(
            command=[sys.executable, str(script)],
            workdir=tmp_path,
            yield_time_ms=0,
            max_output_chars=1000,
        )
        results = []
        thread = threading.Thread(
            target=lambda: results.append(
                manager.poll(
                    process_id=initial["process_id"],
                    wait_ms=30000,
                    max_output_chars=1000,
                )
            )
        )
        thread.start()
        stop.set()
        thread.join(2)
        assert not thread.is_alive() and results
    finally:
        stop.set()
        manager.close()
