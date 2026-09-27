from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from src.models.ToolCall import ToolCall
from src.utils.logger import get_logger


class ToolDispatcher:

    ARGUMENT_ALIASES: dict[str, dict[str, str]] = {
        "read_file": {
            "line_start": "start_line",
            "line_end": "end_line",
            "path": "file_path",
            "filename": "file_path",
        },
        "apply_patch": {
            "content": "patch",
            "diff": "patch",
        },
        "command_exec": {
            "cmd": "command",
            "working_dir": "workdir",
            "cwd": "workdir",
            "timeout": "timeout_ms",
        },
    }

    def __init__(
        self,
        tool_registry,
    ) -> None:

        self.tool_registry = tool_registry
        self.logger = get_logger("[TOOLDISPATCHER]")

    def dispatch(
        self,
        raw_calls: Any,
    ) -> list[ToolCall]:

        if raw_calls is None:
            return []

        if isinstance(
            raw_calls,
            (dict, Mapping),
        ):

            raw_calls = [raw_calls]

        elif not isinstance(
            raw_calls,
            (list, tuple),
        ):

            raw_calls = [raw_calls]

        results: list[ToolCall] = []

        for raw_call in raw_calls:
            results.append(self._dispatch_call(raw_call))

        return results

    def _dispatch_call(
        self,
        raw_call: Any,
    ) -> ToolCall:

        try:

            call_id, name, arguments = self._extract_tool_call(raw_call)

            if not isinstance(name, str):
                return ToolCall(
                    name="",
                    id=call_id,
                    valid=False,
                )

            name = name.strip()

            if not name:
                return ToolCall(
                    name="",
                    id=call_id,
                    valid=False,
                )

            arguments = self._normalize_arguments(arguments)

            if arguments is None:
                return ToolCall(
                    name=name,
                    id=call_id,
                    valid=False,
                )

            arguments = self._apply_aliases(
                name,
                arguments,
            )

            arguments = self._filter_unknown_args(
                name,
                arguments,
            )

            if not self.tool_registry.is_available(name):
                return ToolCall(
                    name=name,
                    id=call_id,
                    args=arguments,
                    valid=False,
                )

            tool = self.tool_registry.get(name)

            if tool is None:
                return ToolCall(
                    name=name,
                    id=call_id,
                    args=arguments,
                    valid=False,
                )

            validate = getattr(
                tool,
                "validate",
                None,
            )

            if callable(validate):

                try:
                    validation_result = validate(arguments)

                except Exception:
                    return ToolCall(
                        name=name,
                        id=call_id,
                        args=arguments,
                        valid=False,
                    )

                if validation_result is False:
                    return ToolCall(
                        name=name,
                        id=call_id,
                        args=arguments,
                        valid=False,
                    )

            action = "execute"
            target = ""

            describe_call = getattr(
                tool,
                "describe_call",
                None,
            )

            if callable(describe_call):

                try:

                    description = describe_call(arguments)

                    if isinstance(
                        description,
                        dict,
                    ):

                        action = (
                            str(
                                description.get(
                                    "action",
                                    "execute",
                                )
                            ).strip()
                            or "execute"
                        )

                        target = str(
                            description.get(
                                "target",
                                "",
                            )
                        ).strip()

                except Exception:

                    action = (
                        str(
                            getattr(
                                tool,
                                "action",
                                "execute",
                            )
                        ).strip()
                        or "execute"
                    )

                    target = ""

            return ToolCall(
                name=name,
                id=call_id,
                args=arguments,
                valid=True,
                action=action,
                target=target,
                path=target,
            )

        except Exception:

            # The dispatcher guarantees that every normalized
            # ToolCall still has an ID, even on malformed input.
            return ToolCall(
                name="",
                id=str(uuid4()),
                valid=False,
            )

    @classmethod
    def _apply_aliases(
        cls,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        aliases = cls.ARGUMENT_ALIASES.get(tool_name.lower())

        if not aliases:
            return arguments

        normalized: dict[str, Any] = {}

        for key, value in arguments.items():

            canonical = aliases.get(
                key,
                key,
            )

            if canonical in normalized and key != canonical:
                continue

            normalized[canonical] = value

        return normalized

    def _filter_unknown_args(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        tool = self.tool_registry.get(tool_name)

        if tool is None:
            return arguments

        schema = (
            getattr(
                tool,
                "parameters",
                {},
            )
            or {}
        )

        allowed = set(
            schema.get(
                "properties",
                {},
            ).keys()
        )

        if not allowed:
            return arguments

        filtered = {key: value for key, value in arguments.items() if key in allowed}

        dropped = set(arguments.keys()) - allowed

        if dropped:

            self.logger.warning(
                f"Dropped unknown args for '{tool_name}': "
                f"{sorted(dropped)}. "
                f"Allowed: {sorted(allowed)}"
            )

        return filtered

    @staticmethod
    def _normalize_call_id(
        raw_id: Any,
    ) -> str:

        if raw_id is None:
            return str(uuid4())

        call_id = str(raw_id).strip()

        if not call_id:
            return str(uuid4())

        return call_id

    @classmethod
    def _extract_tool_call(
        cls,
        raw_call: Any,
    ) -> tuple[str, Any, Any]:

        call_id = str(uuid4())

        # OpenAI / OpenRouter / Ollama-style tool call object:
        #
        # {
        #     "id": "call_xxx",
        #     "type": "function",
        #     "function": {
        #         "name": "...",
        #         "arguments": "..."
        #     }
        # }
        if raw_call is not None:

            raw_id = getattr(
                raw_call,
                "id",
                None,
            )

            call_id = cls._normalize_call_id(raw_id)

            function = getattr(
                raw_call,
                "function",
                None,
            )

            if function is not None:

                if isinstance(
                    function,
                    Mapping,
                ):

                    name = function.get("name")

                    arguments = function.get(
                        "arguments",
                        {},
                    )

                else:

                    name = getattr(
                        function,
                        "name",
                        None,
                    )

                    arguments = getattr(
                        function,
                        "arguments",
                        None,
                    )

                return (
                    call_id,
                    name,
                    arguments,
                )

        if isinstance(
            raw_call,
            Mapping,
        ):

            raw_id = raw_call.get("id")

            call_id = cls._normalize_call_id(raw_id)

            nested_function = raw_call.get("function")

            if isinstance(
                nested_function,
                Mapping,
            ):

                return (
                    call_id,
                    nested_function.get("name"),
                    nested_function.get(
                        "arguments",
                        {},
                    ),
                )

            return (
                call_id,
                raw_call.get("name"),
                raw_call.get(
                    "arguments",
                    {},
                ),
            )

        return (
            call_id,
            None,
            None,
        )

    @staticmethod
    def _normalize_arguments(
        arguments: Any,
    ) -> dict[str, Any] | None:

        if arguments is None:
            return {}

        if isinstance(
            arguments,
            Mapping,
        ):
            return dict(arguments)

        if isinstance(
            arguments,
            str,
        ):

            arguments = arguments.strip()

            if not arguments:
                return {}

            try:
                parsed = json.loads(arguments)

            except (
                json.JSONDecodeError,
                TypeError,
            ):
                return None

            if not isinstance(
                parsed,
                Mapping,
            ):
                return None

            return dict(parsed)

        return None
