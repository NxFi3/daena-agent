from __future__ import annotations

import sys

from src.tools.builtin.command_exec.tool import CommandExec
from src.tools.builtin.command_exec.process_manager import PROCESS_MANAGER


def test_quick_command_completes():
    tool = CommandExec()

    result = tool.execute(
        command=[sys.executable, "-c", "print('done')"],
        yield_time_ms=1_000,
    )

    assert result.success is True
    assert result.content["status"] == "exited"
    assert result.content["exit_code"] == 0
    assert result.content["process_id"] is None
    assert "Command succeeded" in result.summary


def test_long_running_command_returns_managed_process():
    tool = CommandExec()

    result = tool.execute(
        command=[
            sys.executable,
            "-c",
            "import time; print('started', flush=True); time.sleep(0.8)",
        ],
        yield_time_ms=25,
    )

    assert result.success is False
    assert result.content["status"] == "running"
    assert result.content["operation_complete"] is False
    process_id = result.content["process_id"]
    assert isinstance(process_id, str)
    assert process_id.startswith("proc-")
    assert "still running" in result.summary

    finished = PROCESS_MANAGER.poll(
        process_id=process_id,
        wait_ms=2_000,
        max_output_chars=8_000,
    )
    assert finished["status"] == "exited"
    assert finished["exit_code"] == 0
    combined_output = result.content["stdout"] + finished["stdout"]
    assert "started" in combined_output


def test_incremental_output_and_process_write():
    tool = CommandExec()

    result = tool.execute(
        command=[
            sys.executable,
            "-c",
            "import sys; line=sys.stdin.readline(); print(line.strip(), flush=True)",
        ],
        yield_time_ms=25,
        pipe_stdin=True,
    )

    assert result.success is False
    assert result.content["status"] == "running"
    process_id = result.content["process_id"]
    assert isinstance(process_id, str)

    write = PROCESS_MANAGER.write(
        process_id=process_id,
        input_text="hello\n",
    )
    assert write["status"] == "accepted"

    finished = PROCESS_MANAGER.poll(
        process_id=process_id,
        wait_ms=2_000,
        max_output_chars=8_000,
    )
    assert finished["status"] == "exited"
    assert "hello" in finished["stdout"]

    second_poll = PROCESS_MANAGER.poll(
        process_id=process_id,
        wait_ms=0,
        max_output_chars=8_000,
    )
    assert second_poll["stdout"] == ""
    assert second_poll["stderr"] == ""


def test_process_stop_terminates_running_process():
    tool = CommandExec()

    result = tool.execute(
        command=[
            sys.executable,
            "-c",
            "import time; time.sleep(10)",
        ],
        yield_time_ms=25,
    )

    assert result.success is False
    assert result.content["status"] == "running"
    process_id = result.content["process_id"]
    assert isinstance(process_id, str)

    stopped = PROCESS_MANAGER.stop(process_id=process_id)

    assert stopped["status"] in {"exited", "terminated"}
    assert stopped["process_id"] == process_id

    
def test_duplicate_running_command_reuses_existing_process():
    tool = CommandExec()
    command = [
        sys.executable,
        "-c",
        "import time; print('started', flush=True); time.sleep(2)",
    ]

    first = tool.execute(command=command, yield_time_ms=25)
    assert first.success is False
    assert first.content["status"] == "running"

    second = tool.execute(command=command, yield_time_ms=25)
    assert second.success is False
    assert second.content["status"] == "running"
    assert second.content["process_id"] == first.content["process_id"]
    assert second.content.get("reused_existing_process") is True

    PROCESS_MANAGER.stop(process_id=first.content["process_id"])


def test_poll_result_preserves_command_metadata():
    tool = CommandExec()
    command = [sys.executable, "-c", "import time; time.sleep(0.2)"]
    result = tool.execute(command=command, yield_time_ms=20)
    assert result.content["status"] == "running"

    process_id = result.content["process_id"]
    finished = PROCESS_MANAGER.poll(
        process_id=process_id,
        wait_ms=1_000,
        max_output_chars=8_000,
    )

    assert finished["status"] == "exited"
    assert finished["command"] == command
    assert "workdir" in finished
    PROCESS_MANAGER.stop(process_id=process_id)


def test_subprocess_resource_failure_is_structured():
    from src.tools.builtin.command_exec.tool import CommandExec

    failure_type = CommandExec._classify_failure(
        stdout="",
        stderr="Error: listen EADDRINUSE: address already in use :::3000",
    )

    assert failure_type == "resource_in_use"



def test_known_execution_failure_keeps_concrete_diagnostic():
    tool = CommandExec()

    result = tool._execute_managed(
        command=["sh", "-c", "printf 'Error: listen EADDRINUSE: address already in use :::3000\\n' >&2; exit 1"],
        workdir=None,
        yield_time_ms=1_000,
        max_output_chars=8_000,
    )

    assert result.success is False
    assert result.content["error"]["type"] == "resource_in_use"
    assert "EADDRINUSE" in result.content["error"]["message"]
