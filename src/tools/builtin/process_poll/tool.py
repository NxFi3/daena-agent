from __future__ import annotations

from typing import Any

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool
from src.tools.builtin.command_exec.process_manager import PROCESS_MANAGER


class ProcessPoll(Tool):
    name = "process_poll"
    action = "inspect"

    # Polling is an observation of externally changing process state. The same
    # arguments may legitimately return different results without a workspace edit.
    allow_same_revision_repeat = True

    description = (
        "Poll a managed process by process_id. Use this after command_exec returns "
        "status=running. Returns only newly available stdout/stderr since the last "
        "poll, plus the current process status and exit code."
    )

    parameters = {
        "type": "object",
        "properties": {
            "process_id": {"type": "string", "description": "Managed process ID."},
            "wait_ms": {
                "type": "integer",
                "minimum": 0,
                "maximum": 30_000,
                "default": 1_000,
                "description": "How long to wait for new output or process exit.",
            },
            "max_output_chars": {
                "type": "integer",
                "minimum": 512,
                "maximum": 32_000,
                "default": 8_000,
                "description": "Maximum newly returned output characters.",
            },
        },
        "required": ["process_id"],
        "additionalProperties": False,
    }

    def execute(
        self,
        process_id: str,
        wait_ms: int = 1_000,
        max_output_chars: int = 8_000,
    ) -> ToolResult:
        if not isinstance(process_id, str) or not process_id.strip():
            return self._error("invalid_argument", "process_id is required.")

        if not isinstance(wait_ms, int) or isinstance(wait_ms, bool) or not 0 <= wait_ms <= 30_000:
            return self._error("invalid_argument", "wait_ms must be between 0 and 30000.")

        if not isinstance(max_output_chars, int) or isinstance(max_output_chars, bool) or not 512 <= max_output_chars <= 32_000:
            return self._error("invalid_argument", "max_output_chars must be between 512 and 32000.")

        result = PROCESS_MANAGER.poll(
            process_id=process_id.strip(),
            wait_ms=wait_ms,
            max_output_chars=max_output_chars,
        )

        success = result.get("status") != "unknown"

        return ToolResult(
            success=success,
            name=self.name,
            content={
                "success": success,
                **result,
            },
            metadata={},
        )

    def _error(self, error_type: str, message: str) -> ToolResult:
        return ToolResult(
            success=False,
            name=self.name,
            content={
                "success": False,
                "error": {"type": error_type, "message": message},
            },
            metadata={},
        )
