from __future__ import annotations

import re
from typing import Any

from src.models.ToolCall import ToolCall
from src.models.ToolResult import ToolResult
from src.security.securityService import SecurityService
from src.tools.ToolDispatcher import ToolDispatcher
from src.tools.ToolRegistry import ToolRegistry
from src.tools.builtin.command_exec.process_manager import ProcessManager
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
        self.process_manager = ProcessManager()

    def set_workspace(self, directory: str) -> None:
        self.security.set_workspace(directory)

        # Plan state is task-scoped workspace state. Keep the plan tool on the
        # same filesystem coordinate system as the other workspace tools.
        plan_tool = self.get_tool("plan")
        set_workspace = getattr(plan_tool, "set_workspace", None)
        if callable(set_workspace):
            set_workspace(directory)

    def close(self) -> None:
        """Release runtime-owned resources, including managed processes."""
        self.process_manager.close()

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

    def _bind_runtime_services(self) -> None:
        """Inject runtime-owned services into discovered tool instances."""
        for tool in self.toolregistry.tools.values():
            setter = getattr(tool, "set_process_manager", None)
            if callable(setter):
                setter(self.process_manager)
            elif hasattr(tool, "process_manager"):
                tool.process_manager = self.process_manager

    def get_tools(self) -> list[dict]:
        if self._definitions is None:
            self.toolregistry.discover()
            self._bind_runtime_services()
            self._definitions = self.toolregistry.get_definitions()
        else:
            self._bind_runtime_services()

        allowed = set(self.security.policy.allowed_tools)
        network_allowed = bool(self.security.policy.allow_network_tools)

        visible: list[dict] = []
        for definition in self._definitions:
            function = definition.get("function", {}) if isinstance(definition, dict) else {}
            name = str(function.get("name", "")).strip().lower()
            if not name or name not in allowed:
                continue
            if name in {"web_search", "web_fetch"} and not network_allowed:
                continue
            visible.append(definition)

        return visible

    def _find_tool(self, name: str):
        return self.toolregistry.get(str(name).strip().lower())

    def get_tool(self, name: str):
        if not self.toolregistry.tools:
            self.toolregistry.discover()
        self._bind_runtime_services()
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

    @staticmethod
    def _recovery_hint(result: ToolResult) -> str:
        content = result.content if isinstance(result.content, dict) else {}
        error = content.get("error")
        error_type = ""
        if isinstance(error, dict):
            error_type = str(error.get("type", "")).strip().lower()
        elif error:
            error_type = str(error).strip().lower()

        hints = {
            "workspace_boundary": (
                "Keep every path/workdir inside the active workspace. "
                "Use relative paths from the workspace root."
            ),
            "command_not_found": (
                "The executable was not found. Check the project setup first; "
                "for Node projects prefer the package script (for example npm test) "
                "or npx when appropriate."
            ),
            "resource_in_use": (
                "The command reached an existing resource such as a listening address. "
                "Inspect the active process state or choose the intended existing process "
                "before starting another instance."
            ),
            "invalid_argument": (
                "Reissue the call with schema-valid arguments. Optional arguments "
                "can be omitted instead of guessing."
            ),
            "tool_argument_error": (
                "The tool rejected the argument contract at execution time. "
                "Re-read the documented parameters and provide all required "
                "arguments with the correct types."
            ),
            "invalid_tool_call": (
                "Reissue the call with the documented tool name and argument schema; "
                "do not repeat the malformed call unchanged."
            ),
            "tool_not_found": (
                "Use one of the currently available tools instead of retrying the "
                "missing tool."
            ),
            "security_denied": (
                "The runtime denied this action. Respect the reported security rule "
                "and choose a compliant alternative."
            ),
            "tool_execution_error": (
                "Inspect the concrete execution error and change the action before "
                "retrying; do not blindly repeat the same call."
            ),
            "unexpected_error": (
                "The tool failed unexpectedly. Inspect the error and try a different "
                "action or corrected arguments."
            ),
        }
        return hints.get(
            error_type,
            "Use the concrete tool result to decide the next action; avoid repeating "
            "an unchanged failed call.",
        )

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
                message = (
                    str(toolcall.validation_error).strip()
                    or "Invalid tool call."
                )
                calls.append(toolcall)
                results.append(
                    ToolResult(
                        success=False,
                        name=toolcall.name,
                        content={
                            "success": False,
                            "error": {
                                "type": "invalid_tool_call",
                                "message": message,
                            },
                        },
                        metadata={
                            "tool_call_id": toolcall.id,
                            "normalization_notes": list(
                                toolcall.normalization_notes or []
                            ),
                            "recovery_hint": (
                                "Reissue the call using the documented schema. "
                                "Do not repeat an invalid call unchanged."
                            ),
                        },
                        summary=message,
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
                            "recovery_hint": (
                                "Respect the reported security rule and choose a "
                                "different compliant action."
                            ),
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

                if not result.success:
                    result.metadata.setdefault(
                        "recovery_hint",
                        self._recovery_hint(result),
                    )

                results.append(result)

            except TypeError as exc:
                self.logger.error(
                    "Tool '%s' rejected its execution arguments: %s",
                    toolcall.name,
                    exc,
                )
                results.append(
                    ToolResult(
                        success=False,
                        name=toolcall.name,
                        content={
                            "success": False,
                            "error": {
                                "type": "tool_argument_error",
                                "message": (
                                    f"Tool '{toolcall.name}' rejected the provided "
                                    f"arguments: {exc}"
                                ),
                            },
                        },
                        metadata={
                            "tool_call_id": toolcall.id,
                            "recovery_hint": (
                                "Treat this as an argument/contract error. "
                                "Re-read the tool schema and provide the required "
                                "arguments instead of retrying unchanged."
                            ),
                        },
                    )
                )
            except Exception as exc:
                self.logger.error(
                    "Tool '%s' failed: %s",
                    toolcall.name,
                    exc,
                )
                results.append(
                    ToolResult(
                        success=False,
                        name=toolcall.name,
                        content={
                            "success": False,
                            "error": {
                                "type": "tool_execution_error",
                                "message": (
                                    f"Unexpected error while using tool "
                                    f"{type(exc).__name__}: {exc}"
                                ),
                            },
                        },
                        metadata={
                            "tool_call_id": toolcall.id,
                            "recovery_hint": (
                                "Inspect the concrete runtime error and choose "
                                "a corrected action or different tool. Do not "
                                "blindly repeat the same call."
                            ),
                        },
                    )
                )

        return {"calls": calls, "results": results}
