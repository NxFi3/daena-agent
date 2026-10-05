# src.agent/agentstate.py
from __future__ import annotations

from typing import Any

from src.models.ToolCall import ToolCall
from src.models.ToolResult import ToolResult


class AgentState:
    """
    Runtime state of the agent.

    AgentState intentionally knows nothing about concrete tool names.
    Tool semantics are provided through ToolCall.action / ToolCall.target.
    """

    _ACTION_LABELS = {
        "create": "Created",
        "add": "Created",
        "modify": "Modified",
        "update": "Modified",
        "delete": "Deleted",
        "inspect": "Inspected",
        "read": "Read",
        "run": "Ran",
        "verify": "Verified",
        "execute": "Executed",
    }

    _OPERATION_TO_ACTION = {
        "add": "create",
        "create": "create",
        "update": "modify",
        "modify": "modify",
        "delete": "delete",
        "read": "inspect",
        "inspect": "inspect",
        "run": "run",
        "execute": "execute",
        "verify": "verify",
    }

    def __init__(
        self,
        max_progress_items: int = 15,
    ) -> None:

        self.max_progress_items = max(
            1,
            int(max_progress_items),
        )

        self.status: str = "idle"

        self.tool: str | None = None
        self.action: str | None = None
        self.target: str | None = None

        self.iteration: int = 0

        self.error: str | None = None

        self._progress: list[str] = []


    def reset(self) -> None:
        self.status = "idle"

        self.tool = None
        self.action = None
        self.target = None

        self.iteration = 0

        self.error = None

        self._progress.clear()

    def begin(
        self,
        tool_call: ToolCall,
        iteration: int,
    ) -> None:

        self.status = "executing"

        self.tool = (
            str(
                getattr(
                    tool_call,
                    "name",
                    "",
                )
            ).strip()
            or None
        )

        self.action = (
            str(
                getattr(
                    tool_call,
                    "action",
                    "execute",
                )
            ).strip()
            or "execute"
        )

        self.target = (
            str(
                getattr(
                    tool_call,
                    "target",
                    "",
                )
            ).strip()
            or None
        )

        self.iteration = max(
            0,
            int(iteration),
        )

        self.error = None

    def succeed(self) -> None:
        self.status = "succeeded"
        self.error = None

    def fail(
        self,
        error: str | None = None,
    ) -> None:

        self.status = "failed"

        if error is not None:
            error = str(error).strip()

        self.error = error or None

    def complete(self) -> None:
        self.status = "completed"

        self.tool = None
        self.action = None
        self.target = None

        self.error = None

    def stop(
        self,
        reason: str,
    ) -> None:

        self.status = "stopped"

        self.tool = None
        self.action = None
        self.target = None

        self.error = str(reason).strip() or None


    @property
    def progress(self) -> list[str]:
        return list(self._progress)

    def add_progress(
        self,
        message: str,
    ) -> None:

        message = str(message).strip()

        if not message:
            return

        if message in self._progress:
            self._progress.remove(message)

        self._progress.append(message)

        if len(self._progress) > self.max_progress_items:
            del self._progress[: len(self._progress) - self.max_progress_items]

    def clear_progress(self) -> None:
        self._progress.clear()


    def update_from_result(
        self,
        result: ToolResult,
    ) -> None:

        if not isinstance(
            result,
            ToolResult,
        ):
            self.fail("Invalid tool result.")
            return

        if result.success:
            effects = self._extract_effects(result)

            if effects:
                for effect in effects:
                    self._add_effect_progress(effect)

            else:
                self._add_current_action_progress()

            self.succeed()
            return

        error_message = self._extract_error_message(
            result,
        )

        self.fail(error_message)

        if error_message:
            self.add_progress(f"Failed: {error_message}")

    def _extract_effects(
        self,
        result: ToolResult,
    ) -> list[dict[str, str]]:

        effects: list[dict[str, str]] = []

        metadata = result.metadata

        if isinstance(
            metadata,
            dict,
        ):

            raw_effects = metadata.get(
                "effects",
                [],
            )

            if isinstance(
                raw_effects,
                list,
            ):

                for effect in raw_effects:

                    if not isinstance(
                        effect,
                        dict,
                    ):
                        continue

                    normalized = self._normalize_effect(
                        effect,
                    )

                    if normalized:
                        effects.append(normalized)

        if effects:
            return effects

        content = result.content

        if not isinstance(
            content,
            dict,
        ):
            return []

        # Standard multi-file result contract.
        files = content.get("files")

        if isinstance(
            files,
            list,
        ):

            for file_info in files:

                if not isinstance(
                    file_info,
                    dict,
                ):
                    continue

                operation = file_info.get(
                    "operation",
                    "",
                )

                target = file_info.get("path") or file_info.get("target") or ""

                effect = self._normalize_effect(
                    {
                        "action": operation,
                        "target": target,
                    }
                )

                if effect:
                    effects.append(effect)

            if effects:
                return effects

        # Standard single-target result contract.
        target = content.get("path") or content.get("target") or ""

        if target:
            return [
                {
                    "action": (self.action or "execute"),
                    "target": str(target),
                }
            ]

        return []

    def _normalize_effect(
        self,
        effect: dict[str, Any],
    ) -> dict[str, str] | None:

        raw_action = (
            str(
                effect.get(
                    "action",
                    "",
                )
            )
            .strip()
            .lower()
        )

        target = str(
            effect.get(
                "target",
                "",
            )
        ).strip()

        if not raw_action:
            return None

        action = self._OPERATION_TO_ACTION.get(
            raw_action,
            raw_action,
        )

        return {
            "action": action,
            "target": target,
        }

    def _add_effect_progress(
        self,
        effect: dict[str, str],
    ) -> None:

        action = effect.get(
            "action",
            "",
        ).strip()

        target = effect.get(
            "target",
            "",
        ).strip()

        if not action:
            return

        label = self._ACTION_LABELS.get(
            action,
            action.capitalize(),
        )

        message = label

        if target:
            message = f"{label} {target}"

        self.add_progress(message)

    def _add_current_action_progress(self) -> None:

        action = self.action or "execute"

        target = self.target or ""

        label = self._ACTION_LABELS.get(
            action,
            action.capitalize(),
        )

        if target:
            self.add_progress(f"{label} {target}")
        else:
            self.add_progress(label)

    @staticmethod
    def _extract_error_message(
        result: ToolResult,
    ) -> str | None:

        content = result.content

        if not isinstance(
            content,
            dict,
        ):
            return None

        error = content.get("error")

        if isinstance(
            error,
            dict,
        ):

            message = error.get("message")

            if message:
                return str(message)

        message = content.get("content")

        if (
            isinstance(
                message,
                str,
            )
            and message.strip()
        ):

            return message.strip()

        return None


    def state_context(self) -> dict[str, Any]:

        return {
            "status": self.status,
            "tool": self.tool,
            "action": self.action,
            "target": self.target,
            "iteration": self.iteration,
            "error": self.error,
        }

    def progress_context(self) -> dict[str, Any]:

        return {
            "items": self.progress,
        }
