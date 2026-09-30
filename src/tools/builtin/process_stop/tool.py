from __future__ import annotations

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool
from src.tools.builtin.command_exec.process_manager import PROCESS_MANAGER


class ProcessStop(Tool):
    # ToolManager replaces this with its session-scoped ProcessManager.\n    process_manager = PROCESS_MANAGER
    name = "process_stop"
    action = "modify"

    description = (
        "Stop a managed process by process_id. Terminates its process group and "
        "returns the final state and remaining output."
    )

    parameters = {
        "type": "object",
        "properties": {
            "process_id": {"type": "string", "description": "Managed process ID."},
        },
        "required": ["process_id"],
        "additionalProperties": False,
    }

    def execute(self, process_id: str) -> ToolResult:
        if not isinstance(process_id, str) or not process_id.strip():
            return self._error("invalid_argument", "process_id is required.")

        result = self.process_manager.stop(process_id=process_id.strip())
        success = result.get("status") != "unknown"

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
