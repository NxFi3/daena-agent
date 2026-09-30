from __future__ import annotations

import sys

from src.tools.ToolManager import ToolManager


def _config() -> dict:
    return {
        "security": {
            "workspace_only": False,
            "allow_background": True,
            "allow_network_tools": True,
            "force_approve": False,
        }
    }


def test_tool_manager_owns_a_session_scoped_process_manager():
    first = ToolManager(_config())
    second = ToolManager(_config())

    try:
        first_command = first.get_tool("command_exec")
        second_command = second.get_tool("command_exec")

        assert first_command.process_manager is first.process_manager
        assert second_command.process_manager is second.process_manager
        assert first.process_manager is not second.process_manager

        result = first_command.execute(
            command=[
                sys.executable,
                "-c",
                "import time; time.sleep(5)",
            ],
            yield_time_ms=25,
        )
        assert result.success is True
        assert result.content["status"] == "running"

        process_id = result.content["process_id"]
        first.close()

        state = first.process_manager.poll(
            process_id=process_id,
            wait_ms=0,
            max_output_chars=512,
        )
        assert state["status"] in {"exited", "terminated"}
    finally:
        first.close()
        second.close()
