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
            "timeout": "yield_time_ms",
            "timeout_ms": "yield_time_ms",
            "yield_time": "yield_time_ms",
            "yield_time": "yield_time_ms",
        },
    }

    def __init__(
        self,
        tool_registry,
    ) -> None:
        self.tool_registry = tool_registry
        self.logger = get_logger("[TOOLDISPATCHER]")

    def dispatch(self, raw_calls: Any) -> list[ToolCall]:
        if raw_calls is None:
            return []

        if isinstance(raw_calls, (dict, Mapping)):
            raw_calls = [raw_calls]
        elif not isinstance(raw_calls, (list, tuple)):
            raw_calls = [raw_calls]

        return [self._dispatch_call(raw_call) for raw_call in raw_calls]

    def _dispatch_call(self, raw_call: Any) -> ToolCall:
        call_id = str(uuid4())

        try:
            call_id, name, arguments = self._extract_tool_call(raw_call)

            if not isinstance(name, str) or not name.strip():
                return ToolCall(
                    name="",
                    id=call_id,
                    valid=False,
                    validation_error="Tool name is missing or empty.",
                )

            name = name.strip()
            arguments = self._normalize_arguments(arguments)

            if arguments is None:
                return ToolCall(
                    name=name,
                    id=call_id,
                    valid=False,
                    validation_error=(
                        "Tool arguments are not valid JSON object data. "
                        "Reissue the call with an object matching the tool schema."
                    ),
                )

            arguments = self._apply_aliases(name, arguments)

            tool = self.tool_registry.get(name)
            if tool is None or not self.tool_registry.is_available(name):
                return ToolCall(
                    name=name,
                    id=call_id,
                    args=arguments,
                    valid=False,
                    validation_error=f"Tool '{name}' is not available.",
                )

            normalize = getattr(tool, "normalize_arguments", None)
            normalization_notes: list[str] = []
            if callable(normalize):
                try:
                    normalized = normalize(arguments)
                    if isinstance(normalized, tuple) and len(normalized) == 2:
                        arguments, notes = normalized
                        if isinstance(notes, list):
                            normalization_notes = [
                                str(note) for note in notes if str(note).strip()
                            ]
                    elif isinstance(normalized, dict):
                        arguments = normalized
                except Exception as exc:
                    return ToolCall(
                        name=name,
                        id=call_id,
                        args=arguments,
                        valid=False,
                        validation_error=f"Argument normalization failed: {exc}",
                    )

            schema = getattr(tool, "parameters", {}) or {}
            properties = schema.get("properties", {})
            if not isinstance(properties, dict):
                properties = {}

            unknown = sorted(
                key for key in arguments.keys()
                if key not in properties
            )
            if unknown and schema.get("additionalProperties", True) is False:
                return ToolCall(
                    name=name,
                    id=call_id,
                    args=arguments,
                    valid=False,
                    validation_error=(
                        "Unknown argument(s): "
                        + ", ".join(unknown)
                        + ". Remove them and use only the documented parameters."
                    ),
                    normalization_notes=normalization_notes,
                )

            if unknown:
                self.logger.warning(
                    f"Dropped unknown args for '{name}': {unknown}. "
                    f"Allowed: {sorted(properties.keys())}"
                )
                arguments = {
                    key: value
                    for key, value in arguments.items()
                    if key in properties
                }

            required = schema.get("required", [])
            if isinstance(required, list):
                missing = [
                    str(key)
                    for key in required
                    if str(key) not in arguments
                ]
                if missing:
                    return ToolCall(
                        name=name,
                        id=call_id,
                        args=arguments,
                        valid=False,
                        validation_error=(
                            "Missing required argument(s): "
                            + ", ".join(missing)
                            + ". Provide every required parameter."
                        ),
                        normalization_notes=normalization_notes,
                    )

            validate = getattr(tool, "validate", None)
            if callable(validate):
                try:
                    validation_result = validate(arguments)
                except Exception as exc:
                    return ToolCall(
                        name=name,
                        id=call_id,
                        args=arguments,
                        valid=False,
                        validation_error=f"Tool validation raised an error: {exc}",
                        normalization_notes=normalization_notes,
                    )

                if validation_result is False:
                    return ToolCall(
                        name=name,
                        id=call_id,
                        args=arguments,
                        valid=False,
                        validation_error=(
                            "Tool rejected the normalized arguments. "
                            "Reissue the call using values allowed by the schema."
                        ),
                        normalization_notes=normalization_notes,
                    )

            action = "execute"
            target = ""

            describe_call = getattr(tool, "describe_call", None)
            if callable(describe_call):
                try:
                    description = describe_call(arguments)
                    if isinstance(description, dict):
                        action = (
                            str(description.get("action", "execute")).strip()
                            or "execute"
                        )
                        target = str(description.get("target", "")).strip()
                except Exception:
                    action = (
                        str(getattr(tool, "action", "execute")).strip()
                        or "execute"
                    )

            return ToolCall(
                name=name,
                id=call_id,
                args=arguments,
                valid=True,
                action=action,
                target=target,
                path=target,
                normalization_notes=normalization_notes,
            )

        except Exception as exc:
            self.logger.error(f"Tool dispatch failed: {exc}")
            return ToolCall(
                name="",
                id=call_id,
                valid=False,
                validation_error=f"Unexpected tool-dispatch error: {exc}",
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
            canonical = aliases.get(key, key)
            if canonical in normalized and key != canonical:
                continue
            normalized[canonical] = value
        return normalized

    @staticmethod
    def _normalize_call_id(raw_id: Any) -> str:
        if raw_id is None:
            return str(uuid4())

        call_id = str(raw_id).strip()
        return call_id or str(uuid4())

    @classmethod
    def _extract_tool_call(
        cls,
        raw_call: Any,
    ) -> tuple[str, Any, Any]:
        call_id = str(uuid4())

        if raw_call is not None:
            raw_id = getattr(raw_call, "id", None)
            call_id = cls._normalize_call_id(raw_id)

            function = getattr(raw_call, "function", None)
            if function is not None:
                if isinstance(function, Mapping):
                    name = function.get("name")
                    arguments = function.get("arguments", {})
                else:
                    name = getattr(function, "name", None)
                    arguments = getattr(function, "arguments", None)

                return call_id, name, arguments

        if isinstance(raw_call, Mapping):
            raw_id = raw_call.get("id")
            call_id = cls._normalize_call_id(raw_id)

            nested_function = raw_call.get("function")
            if isinstance(nested_function, Mapping):
                return (
                    call_id,
                    nested_function.get("name"),
                    nested_function.get("arguments", {}),
                )

            return (
                call_id,
                raw_call.get("name"),
                raw_call.get("arguments", {}),
            )

        return call_id, None, None

    @staticmethod
    def _normalize_arguments(arguments: Any) -> dict[str, Any] | None:
        if arguments is None:
            return {}

        if isinstance(arguments, Mapping):
            return dict(arguments)

        if isinstance(arguments, str):
            arguments = arguments.strip()
            if not arguments:
                return {}

            try:
                parsed = json.loads(arguments)
            except (json.JSONDecodeError, TypeError, ValueError):
                return None

            if not isinstance(parsed, Mapping):
                return None

            return dict(parsed)

        return None
