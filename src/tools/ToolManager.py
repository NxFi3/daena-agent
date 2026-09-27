from __future__ import annotations

from src.models.ToolCall import ToolCall
from src.models.ToolResult import ToolResult
from src.tools.ToolDispatcher import ToolDispatcher
from src.tools.ToolRegistry import ToolRegistry
from src.utils.logger import get_logger
from src.security.securityService import security  # NotImplemented


class ToolManager:

    def __init__(
        self,
    ) -> None:

        self.logger = get_logger("[TOOLMANAGER]")

        self.toolregistry = ToolRegistry()

        self.dispatcher = ToolDispatcher(self.toolregistry)

        self.security = security()

        self._definitions: list[dict] | None = None

    def get_tools(
        self,
    ):
        """
        Discover builtin tools once and
        return their definitions.
        """

        if self._definitions is None:

            self.toolregistry.discover()

            self._definitions = self.toolregistry.get_definitions()

        return self._definitions

    def _find_definition(
        self,
        name: str,
    ) -> dict | None:

        normalized_name = str(name).strip().lower()

        for definition in self.get_tools():

            if not isinstance(
                definition,
                dict,
            ):
                continue

            function = definition.get(
                "function",
                {},
            )

            if not isinstance(
                function,
                dict,
            ):
                continue

            definition_name = (
                str(
                    function.get(
                        "name",
                        "",
                    )
                )
                .strip()
                .lower()
            )

            if definition_name == normalized_name:
                return definition

        return None

    def _find_tool(
        self,
        name: str,
    ):

        return self.toolregistry.get(str(name).strip().lower())

    def execute(
        self,
        tool_calls: list[ToolCall],
    ) -> dict:

        calls: list[ToolCall] = []
        results: list[ToolResult] = []

        if not tool_calls:

            return {
                "calls": [],
                "results": [],
            }

        for incoming_call in tool_calls:

            if not isinstance(
                incoming_call,
                ToolCall,
            ):

                self.logger.error(
                    "ToolManager received "
                    "non-ToolCall object: "
                    f"{type(incoming_call).__name__}"
                )

                invalid_call = ToolCall(
                    name="",
                    valid=False,
                )

                calls.append(invalid_call)

                results.append(
                    ToolResult(
                        success=False,
                        name="",
                        content={
                            "success": False,
                            "error": {
                                "type": ("invalid_tool_call"),
                                "message": (
                                    "ToolManager received " "a non-ToolCall object."
                                ),
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
                                "type": ("invalid_tool_call"),
                                "message": ("Invalid tool call."),
                            },
                        },
                        metadata={
                            "tool_call_id": toolcall.id,
                        },
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
                                "type": ("tool_not_found"),
                                "message": (
                                    f"Tool '{toolcall.name}' " "was not found."
                                ),
                            },
                        },
                        metadata={
                            "tool_call_id": toolcall.id,
                        },
                    )
                )

                continue

            try:

                checked_call = self.security.check(toolcall)

                if isinstance(
                    checked_call,
                    ToolCall,
                ):

                    toolcall = checked_call

            except Exception as exc:

                self.logger.error(
                    "Security check failed for " f"'{toolcall.name}': {exc}"
                )

                calls.append(toolcall)

                results.append(
                    ToolResult(
                        success=False,
                        name=toolcall.name,
                        content={
                            "success": False,
                            "error": {
                                "type": ("security_check_failed"),
                                "message": (f"Security check failed: " f"{exc}"),
                            },
                        },
                        metadata={
                            "tool_call_id": toolcall.id,
                        },
                    )
                )

                continue

            calls.append(toolcall)

            if not toolcall.approved:

                results.append(
                    ToolResult(
                        success=False,
                        name=toolcall.name,
                        content={
                            "success": False,
                            "error": {
                                "type": ("tool_not_allowed"),
                                "message": ("You are not allowed " "to use this tool."),
                            },
                        },
                        metadata={
                            "tool_call_id": toolcall.id,
                        },
                    )
                )

                continue

            try:

                result = tool.execute(**toolcall.args)

                if not isinstance(
                    result,
                    ToolResult,
                ):

                    self.logger.error(
                        f"Tool '{toolcall.name}' " "returned an invalid " "result type."
                    )

                    result = ToolResult(
                        success=False,
                        name=toolcall.name,
                        content={
                            "success": False,
                            "error": {
                                "type": ("invalid_tool_result"),
                                "message": ("Tool returned an " "invalid result."),
                            },
                        },
                        metadata={
                            "tool_call_id": toolcall.id,
                        },
                    )

                results.append(result)

            except Exception as exc:

                self.logger.error(f"Tool '{toolcall.name}' " f"failed: {exc}")

                results.append(
                    ToolResult(
                        success=False,
                        name=toolcall.name,
                        content={
                            "success": False,
                            "error": {
                                "type": ("tool_execution_error"),
                                "message": (
                                    "Unexpected error while " f"using tool: {exc}"
                                ),
                            },
                        },
                        metadata={
                            "tool_call_id": toolcall.id,
                        },
                    )
                )

        return {
            "calls": calls,
            "results": results,
        }
