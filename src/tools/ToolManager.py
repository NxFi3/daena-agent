from __future__ import annotations

import re
from typing import Any

from src.models.ToolCall import ToolCall
from src.models.ToolResult import ToolResult
from src.security.securityService import SecurityService
from src.tools.ToolDispatcher import ToolDispatcher
from src.tools.ToolRegistry import ToolRegistry
from src.utils.logger import get_logger


class ToolManager:
    """Own tool discovery, normalization, policy checks and execution."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.logger = get_logger("[TOOLMANAGER]")

        self.config = config or {}
        self.toolregistry = ToolRegistry()
        self.dispatcher = ToolDispatcher(self.toolregistry)
        self.security = SecurityService(self.config.get("security"))
        self._definitions: list[dict] | None = None

    def set_workspace(self, directory: str) -> None:
        self.security.set_workspace(directory)

    def approve_background_command(
        self,
        command: list[str],
        workdir: str | None = None,
    ) -> None:
        """Grant an exact background command for the current session."""
        self.security.approve_background_command(
            command=command,
            workdir=workdir,
        )

    def revoke_background_command(
        self,
        command: list[str],
        workdir: str | None = None,
    ) -> None:
        """Revoke an exact background command approval."""
        self.security.revoke_background_command(
            command=command,
            workdir=workdir,
        )

    def clear_background_approvals(self) -> None:
        """Clear all session-scoped background command approvals."""
        self.security.clear_background_approvals()

    def get_tools(self) -> list[dict]:
        if self._definitions is None:
            self.toolregistry.discover()
            self._definitions = self.toolregistry.get_definitions()
        return self._definitions

    def _find_tool(self, name: str):
        return self.toolregistry.get(str(name).strip().lower())

    def get_tool(self, name: str):
        if not self.toolregistry.tools:
            self.toolregistry.discover()
        return self._find_tool(name)

    def _normalize_workspace_args(self, toolcall: ToolCall) -> ToolCall:
        """Turn workspace-relative tool arguments into absolute safe paths.

        Security validation alone is insufficient when a tool interprets a
        relative path against the process CWD. Normalize after approval so the
        actual execution target is the same target that policy evaluated.
        """
        if not self.security.policy.workspace_only:
            return toolcall

        args = dict(toolcall.args or {})
        name = toolcall.name

        if name == "read_file":
            args["file_path"] = str(self.security.sandbox.resolve(args["file_path"]))

        elif name == "command_exec":
            args["workdir"] = str(
                self.security.sandbox.resolve(args.get("workdir") or ".")
            )

        elif name == "apply_patch":
            patch = args["patch"]
            lines = patch.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            normalized: list[str] = []

            for line in lines:
                match = re.match(
                    r"^(\*\*\*\s+(?:Add|Update|Delete) File:\s*)(.+?)\s*$",
                    line,
                )
                if match:
                    path = match.group(2).strip()
                    line = match.group(1) + str(
                        self.security.sandbox.resolve(path)
                    )
                normalized.append(line)

            args["patch"] = "\n".join(normalized)

        toolcall.args = args
        return toolcall

    def execute(self, tool_calls: list[ToolCall]) -> dict:
        calls: list[ToolCall] = []
        results: list[ToolResult] = []

        if not tool_calls:
            return {"calls": [], "results": []}

        for incoming_call in tool_calls:
            if not isinstance(incoming_call, ToolCall):
                invalid_call = ToolCall(name="", valid=False)
                calls.append(invalid_call)
                results.append(
                    ToolResult(
                        success=False,
                        name="",
                        content={
                            "success": False,
                            "error": {
                                "type": "invalid_tool_call",
                                "message": "ToolManager received a non-ToolCall object.",
                            },
                        },
                        metadata={},
                    )
                )
                continue

            toolcall = incoming_call

            if not toolcall.valid:
                calls.append(toolcall)
                results.append(
                    ToolResult(
                        success=False,
                        name=toolcall.name,
                        content={
                            "success": False,
                            "error": {
                                "type": "invalid_tool_call",
                                "message": "Invalid tool call.",
                            },
                        },
                        metadata={"tool_call_id": toolcall.id},
                    )
                )
                continue

            tool = self._find_tool(toolcall.name)
            if tool is None:
                calls.append(toolcall)
                results.append(
                    ToolResult(
                        success=False,
                        name=toolcall.name,
                        content={
                            "success": False,
                            "error": {
                                "type": "tool_not_found",
                                "message": f"Tool '{toolcall.name}' was not found.",
                            },
                        },
                        metadata={"tool_call_id": toolcall.id},
                    )
                )
                continue

            try:
                toolcall = self.security.check(toolcall)
            except Exception as exc:
                self.logger.error("Security check failed for '%s': %s", toolcall.name, exc)
                toolcall.approved = False
                toolcall.security_reason = f"Security check failed: {exc}"
                toolcall.security_rule = "security_failure"

            calls.append(toolcall)

            if not toolcall.approved:
                results.append(
                    ToolResult(
                        success=False,
                        name=toolcall.name,
                        content={
                            "success": False,
                            "error": {
                                "type": "security_denied",
                                "message": toolcall.security_reason or "Tool execution denied by security policy.",
                            },
                        },
                        metadata={
                            "tool_call_id": toolcall.id,
                            "security_rule": toolcall.security_rule,
                        },
                    )
                )
                continue

            try:
                toolcall = self._normalize_workspace_args(toolcall)
                calls[-1] = toolcall
                result = tool.execute(**toolcall.args)
                if not isinstance(result, ToolResult):
                    result = ToolResult(
                        success=False,
                        name=toolcall.name,
                        content={
                            "success": False,
                            "error": {
                                "type": "invalid_tool_result",
                                "message": "Tool returned an invalid result.",
                            },
                        },
                        metadata={"tool_call_id": toolcall.id},
                    )

                if not isinstance(result.metadata, dict):
                    result.metadata = {}

                result.metadata.setdefault("tool_call_id", toolcall.id)
                results.append(result)

            except Exception as exc:
                self.logger.error("Tool '%s' failed: %s", toolcall.name, exc)
                results.append(
                    ToolResult(
                        success=False,
                        name=toolcall.name,
                        content={
                            "success": False,
                            "error": {
                                "type": "tool_execution_error",
                                "message": f"Unexpected error while using tool: {exc}",
                            },
                        },
                        metadata={"tool_call_id": toolcall.id},
                    )
                )

        return {"calls": calls, "results": results}
