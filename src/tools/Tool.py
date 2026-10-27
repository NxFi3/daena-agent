from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, ClassVar

from src.models.ToolResult import ToolResult


class Tool(ABC):
    name: ClassVar[str] = ""

    description: ClassVar[str] = ""

    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "additionalProperties": True,
    }

    action: ClassVar[str] = "execute"

    allow_same_revision_repeat: ClassVar[bool] = False
    # Only side-effect-free, thread-safe tools may opt into parallel execution.
    parallel_safe: ClassVar[bool] = False
    # Host adapters (including future MCP transports) must declare networking.
    requires_network: ClassVar[bool] = False

    def duplicate_key(
        self,
        arguments: dict[str, Any],
        *,
        workspace_root: str | None = None,
    ) -> dict[str, Any]:
        normalized = dict(arguments or {})
        properties = self.parameters.get("properties", {})
        if isinstance(properties, dict):
            for key, schema in properties.items():
                if (
                    key not in normalized
                    and isinstance(schema, dict)
                    and "default" in schema
                ):
                    normalized[key] = schema["default"]

        del workspace_root
        return normalized

    def normalize_arguments(
        self,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """
        Apply safe, schema-driven normalization before validation/execution.

        Models frequently serialize an integer as a string or choose a value
        as a numeric string. Range clamping is opt-in via the schema extension
        ``x-daena-clamp: true``; adapters otherwise receive validation errors
        for out-of-range values instead of silently changed requests.
        """
        normalized = dict(arguments or {})
        notes: list[str] = []

        properties = self.parameters.get("properties", {})
        if not isinstance(properties, dict):
            return normalized, notes

        for key, schema in properties.items():
            if key not in normalized or not isinstance(schema, dict):
                continue

            value = normalized[key]
            schema_type = schema.get("type")

            if schema_type == "integer":
                original = value
                if isinstance(value, str):
                    stripped = value.strip()
                    try:
                        if stripped and (
                            stripped.isdigit()
                            or (stripped.startswith("-") and stripped[1:].isdigit())
                        ):
                            value = int(stripped)
                    except ValueError:
                        pass

                if isinstance(value, int) and not isinstance(value, bool):
                    minimum = schema.get("minimum")
                    maximum = schema.get("maximum")
                    clamp = schema.get("x-daena-clamp") is True
                    if clamp and isinstance(minimum, int) and value < minimum:
                        value = minimum
                    if clamp and isinstance(maximum, int) and value > maximum:
                        value = maximum

                if value != original:
                    notes.append(f"{key} normalized from {original!r} to {value!r}.")
                    normalized[key] = value

            elif schema_type == "boolean" and isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "false"}:
                    normalized[key] = lowered == "true"
                    notes.append(
                        f"{key} normalized from {value!r} to {normalized[key]!r}."
                    )

        return normalized, notes

    @abstractmethod
    def execute(
        self,
        **kwargs: Any,
    ) -> ToolResult:
        raise NotImplementedError

    def validate(
        self,
        arguments: dict[str, Any],
    ) -> bool:
        return True

    def describe_call(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, str]:
        if not isinstance(arguments, dict):
            arguments = {}

        return {
            "action": self.action,
            "target": self._infer_target(arguments),
        }

    @staticmethod
    def _infer_target(
        arguments: dict[str, Any],
    ) -> str:

        for key in (
            "file_path",
            "path",
            "url",
            "query",
            "command",
            "workdir",
        ):
            value = arguments.get(key)

            if value is None:
                continue

            if isinstance(value, list):
                return " ".join(str(item) for item in value).strip()

            return str(value).strip()

        return ""

    def get_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": deepcopy(self.parameters),
            },
        }

    def __repr__(self) -> str:
        return f"<Tool name='{self.name}'>"
