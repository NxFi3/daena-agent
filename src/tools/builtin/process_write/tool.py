from __future__ import annotations

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool
from src.tools.builtin.command_exec.process_manager import PROCESS_MANAGER


class ProcessWrite(Tool):
    name = "process_write"
    action = "modify"

    description = (
        "Write text to the stdin of a managed process. Use only for a process that "
        "was started with writable stdin and is still running."
    )

    parameters = {
        "type": "object",
        "properties": {
            "process_id": {"type": "string", "description": "Managed process ID."},
            "input": {"type": "string", "description": "Text to write to process stdin."},
        },
        "required": ["process_id", "input"],
        "additionalProperties": False,
    }

    def execute(self, process_id: str, input: str) -> ToolResult:
        if not isinstance(process_id, str) or not process_id.strip():
            return self._error("invalid_argument", "process_id is required.")
        if not isinstance(input, str):
            return self._error("invalid_argument", "input must be a string.")

        result = PROCESS_MANAGER.write(
            process_id=process_id.strip(),
            input_text=input,
        )
        success = result.get("status") == "accepted"

        return ToolResult(
            success=success,
            name=self.name,
            content={"success": success, **result},
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
