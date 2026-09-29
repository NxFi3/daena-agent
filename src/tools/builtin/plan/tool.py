from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool


@dataclass
class PlanStep:
    number: int
    status: str
    description: str


class Plan(Tool):
    """
    Manage Daena's current execution plan in AgentInstruction/plan.md.

    The file is the source of truth. ContextBuilder reads it on the next
    model call, so this tool returns only a minimal operation summary.
    """

    name = "plan"
    action = "modify"

    description = (
        "Manage the current execution plan in AgentInstruction/plan.md. "
        "Use exactly one operation: create, update, or delete. "
        "Update can change the goal, edit a step, add a step, or remove a step. "
        "The updated plan is visible through context on the next model call; "
        "do not expect the tool to return the full plan."
    )

    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["create", "update", "delete"],
                "description": "Plan operation to perform.",
            },
            "goal": {
                "type": "string",
                "description": (
                    "Plan goal. Required for create and for update/set_goal."
                ),
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Initial plan steps. Required for create; "
                    "steps start as pending."
                ),
            },
            "action": {
                "type": "string",
                "enum": [
                    "set_goal",
                    "set_step",
                    "add_step",
                    "remove_step",
                ],
                "description": "Update action. Required when operation is update.",
            },
            "step": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "1-based step number for set_step/remove_step."
                ),
            },
            "status": {
                "type": "string",
                "enum": [
                    "pending",
                    "in_progress",
                    "completed",
                    "blocked",
                ],
                "description": "New step status for set_step.",
            },
            "description_text": {
                "type": "string",
                "description": (
                    "New step description for set_step, or step text for add_step."
                ),
            },
            "after": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Insert the new step after this 1-based step. "
                    "Omit to append."
                ),
            },
        },
        "required": ["operation"],
        "additionalProperties": False,
    }

    PLAN_PATH = Path("AgentInstruction") / "plan.md"

    MAX_GOAL_CHARS = 2_000
    MAX_STEP_CHARS = 1_000
    MAX_STEPS = 50
    MAX_PLAN_CHARS = 12_000

    _STATUSES = {
        "pending",
        "in_progress",
        "completed",
        "blocked",
    }

    _STEP_RE = re.compile(
        r"^\s*(\d+)\.\s+"
        r"\[(pending|in_progress|completed|blocked)\]\s+"
        r"(.+?)\s*$"
    )

    def validate(self, arguments: dict[str, Any]) -> bool:
        if not isinstance(arguments, dict):
            return False

        operation = arguments.get("operation")
        if operation not in {"create", "update", "delete"}:
            return False

        if operation == "create":
            return (
                self._valid_text(
                    arguments.get("goal"),
                    self.MAX_GOAL_CHARS,
                )
                and self._valid_steps(arguments.get("steps"))
            )

        if operation == "delete":
            return all(
                key == "operation" or value is None
                for key, value in arguments.items()
            )

        action = arguments.get("action")
        if action not in {
            "set_goal",
            "set_step",
            "add_step",
            "remove_step",
        }:
            return False

        if action == "set_goal":
            return (
                self._valid_text(
                    arguments.get("goal"),
                    self.MAX_GOAL_CHARS,
                )
                and set(arguments).issubset({"operation", "action", "goal"})
            )

        if action == "set_step":
            step = arguments.get("step")
            has_description = arguments.get("description_text") is not None
            has_status = arguments.get("status") is not None

            if (
                type(step) is not int
                or step < 1
                or not (has_description or has_status)
            ):
                return False

            if has_description and not self._valid_text(
                arguments.get("description_text"),
                self.MAX_STEP_CHARS,
            ):
                return False

            if has_status and arguments.get("status") not in self._STATUSES:
                return False

            return set(arguments).issubset(
                {
                    "operation",
                    "action",
                    "step",
                    "description_text",
                    "status",
                }
            )

        if action == "add_step":
            if not self._valid_text(
                arguments.get("description_text"),
                self.MAX_STEP_CHARS,
            ):
                return False

            after = arguments.get("after")
            return (
                after is None
                or (type(after) is int and after >= 1)
            ) and set(arguments).issubset(
                {
                    "operation",
                    "action",
                    "description_text",
                    "after",
                }
            )

        if action == "remove_step":
            step = arguments.get("step")
            return (
                type(step) is int
                and step >= 1
                and set(arguments).issubset(
                    {"operation", "action", "step"}
                )
            )

        return False

    def describe_call(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, str]:
        return {
            "action": "modify",
            "target": str(self.PLAN_PATH),
        }

    def execute(
        self,
        operation: str,
        goal: str | None = None,
        steps: list[str] | None = None,
        action: str | None = None,
        step: int | None = None,
        status: str | None = None,
        description_text: str | None = None,
        after: int | None = None,
    ) -> ToolResult:
        if operation == "create":
            return self._create(goal, steps)

        if operation == "update":
            return self._update(
                action=action,
                goal=goal,
                step=step,
                status=status,
                description_text=description_text,
                after=after,
            )

        if operation == "delete":
            return self._delete()

        return self._error(
            "invalid_operation",
            "operation must be create, update, or delete.",
        )

    def _create(
        self,
        goal: str | None,
        steps: list[str] | None,
    ) -> ToolResult:
        if not self._valid_text(goal, self.MAX_GOAL_CHARS):
            return self._error(
                "invalid_argument",
                "goal must be a non-empty string.",
            )

        if not self._valid_steps(steps):
            return self._error(
                "invalid_argument",
                f"steps must contain 1-{self.MAX_STEPS} non-empty strings.",
            )

        try:
            existing = self._read_raw()
        except OSError as exc:
            return self._error(
                "read_error",
                f"Could not read the current plan: {exc}",
            )

        if existing.strip():
            return self._error(
                "plan_exists",
                "A plan already exists. Use update or delete before create.",
            )

        plan_steps = [
            PlanStep(
                number=index,
                status="pending",
                description=str(value).strip(),
            )
            for index, value in enumerate(steps or [], 1)
        ]

        content = self._render(str(goal).strip(), plan_steps)

        write_error = self._write_atomic(content)
        if write_error is not None:
            return write_error

        return self._success(
            f"Plan created with {len(plan_steps)} step(s).",
            action="create",
        )

    def _update(
        self,
        *,
        action: str | None,
        goal: str | None,
        step: int | None,
        status: str | None,
        description_text: str | None,
        after: int | None,
    ) -> ToolResult:
        try:
            raw = self._read_raw()
        except OSError as exc:
            return self._error(
                "read_error",
                f"Could not read the plan: {exc}",
            )

        if not raw.strip():
            return self._error(
                "plan_missing",
                "No plan exists. Use create first.",
            )

        try:
            current_goal, current_steps = self._parse(raw)
        except ValueError as exc:
            return self._error(
                "invalid_plan",
                str(exc),
            )

        if action == "set_goal":
            if not self._valid_text(goal, self.MAX_GOAL_CHARS):
                return self._error(
                    "invalid_argument",
                    "goal must be a non-empty string.",
                )

            new_goal = str(goal).strip()
            if new_goal == current_goal:
                return self._success(
                    "Plan unchanged.",
                    action="update",
                )

            current_goal = new_goal
            summary = "Plan updated: goal changed."

        elif action == "set_step":
            if (
                type(step) is not int
                or not 1 <= step <= len(current_steps)
            ):
                return self._error(
                    "invalid_step",
                    f"step must be between 1 and {len(current_steps)}.",
                )

            if description_text is None and status is None:
                return self._error(
                    "invalid_argument",
                    "set_step requires description_text or status.",
                )

            if description_text is not None:
                if not self._valid_text(
                    description_text,
                    self.MAX_STEP_CHARS,
                ):
                    return self._error(
                        "invalid_argument",
                        "description_text must be a non-empty string.",
                    )

                current_steps[step - 1].description = str(
                    description_text
                ).strip()

            if status is not None:
                if status not in self._STATUSES:
                    return self._error(
                        "invalid_status",
                        "Invalid step status.",
                    )

                current_steps[step - 1].status = status

            parts: list[str] = []

            if status is not None:
                parts.append(status.replace("_", " "))

            if description_text is not None:
                parts.append("description changed")

            summary = (
                f"Plan updated: step {step} "
                + ", ".join(parts)
                + "."
            )

        elif action == "add_step":
            if not self._valid_text(
                description_text,
                self.MAX_STEP_CHARS,
            ):
                return self._error(
                    "invalid_argument",
                    "description_text must be a non-empty string.",
                )

            if len(current_steps) >= self.MAX_STEPS:
                return self._error(
                    "plan_limit",
                    f"Plan cannot contain more than {self.MAX_STEPS} steps.",
                )

            if after is not None and (
                type(after) is not int
                or not 1 <= after <= len(current_steps)
            ):
                return self._error(
                    "invalid_step",
                    f"after must be between 1 and {len(current_steps)}.",
                )

            insert_at = len(current_steps) if after is None else after
            current_steps.insert(
                insert_at,
                PlanStep(
                    number=0,
                    status="pending",
                    description=str(description_text).strip(),
                ),
            )
            self._renumber(current_steps)

            added_step = insert_at + 1
            summary = f"Plan updated: added step {added_step}."

        elif action == "remove_step":
            if (
                type(step) is not int
                or not 1 <= step <= len(current_steps)
            ):
                return self._error(
                    "invalid_step",
                    f"step must be between 1 and {len(current_steps)}.",
                )

            if len(current_steps) == 1:
                return self._error(
                    "invalid_plan",
                    "A plan must contain at least one step; delete the plan instead.",
                )

            current_steps.pop(step - 1)
            self._renumber(current_steps)
            summary = f"Plan updated: removed step {step}."

        else:
            return self._error(
                "invalid_action",
                (
                    "update requires set_goal, set_step, "
                    "add_step, or remove_step."
                ),
            )

        content = self._render(current_goal, current_steps)

        if len(content) > self.MAX_PLAN_CHARS:
            return self._error(
                "plan_limit",
                "Updated plan is too large.",
            )

        write_error = self._write_atomic(content)
        if write_error is not None:
            return write_error

        return self._success(summary, action="update")

    def _delete(self) -> ToolResult:
        try:
            self.PLAN_PATH.unlink(missing_ok=True)
        except OSError as exc:
            return self._error(
                "delete_error",
                f"Could not delete the plan: {exc}",
            )

        return self._success(
            "Plan deleted.",
            action="delete",
        )

    def _read_raw(self) -> str:
        if not self.PLAN_PATH.exists():
            return ""

        return self.PLAN_PATH.read_text(encoding="utf-8")

    def _write_atomic(self, content: str) -> ToolResult | None:
        if len(content) > self.MAX_PLAN_CHARS:
            return self._error(
                "plan_limit",
                "Plan is too large.",
            )

        temp_path: Path | None = None

        try:
            self.PLAN_PATH.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.PLAN_PATH.parent,
                prefix=".plan.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temp_path, self.PLAN_PATH)
            return None

        except OSError as exc:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

            return self._error(
                "write_error",
                f"Could not write the plan: {exc}",
            )

    def _parse(
        self,
        text: str,
    ) -> tuple[str, list[PlanStep]]:
        lines = (
            text.replace("\r\n", "\n")
            .replace("\r", "\n")
            .split("\n")
        )

        if not lines or lines[0].strip() != "# Plan":
            raise ValueError("Plan must start with '# Plan'.")

        try:
            goal_index = next(
                i
                for i, line in enumerate(lines)
                if line.strip() == "## Goal"
            )
            steps_index = next(
                i
                for i, line in enumerate(lines)
                if line.strip() == "## Steps"
            )
        except StopIteration as exc:
            raise ValueError(
                "Plan must contain '## Goal' and '## Steps' sections."
            ) from exc

        if goal_index >= steps_index:
            raise ValueError("Plan sections are out of order.")

        goal_lines = [
            line.strip()
            for line in lines[goal_index + 1 : steps_index]
            if line.strip()
        ]

        goal = " ".join(goal_lines).strip()

        if not self._valid_text(goal, self.MAX_GOAL_CHARS):
            raise ValueError("Plan goal is missing or invalid.")

        plan_steps: list[PlanStep] = []

        for line in lines[steps_index + 1 :]:
            if not line.strip():
                continue

            match = self._STEP_RE.match(line)
            if match is None:
                raise ValueError("Plan contains an invalid step line.")

            number = int(match.group(1))
            status = match.group(2)
            description = match.group(3).strip()

            if number != len(plan_steps) + 1:
                raise ValueError(
                    "Plan step numbers must be contiguous starting at 1."
                )

            if not self._valid_text(
                description,
                self.MAX_STEP_CHARS,
            ):
                raise ValueError(
                    f"Plan step {number} is missing or too long."
                )

            plan_steps.append(
                PlanStep(
                    number=number,
                    status=status,
                    description=description,
                )
            )

        if not plan_steps:
            raise ValueError("Plan must contain at least one step.")

        if len(plan_steps) > self.MAX_STEPS:
            raise ValueError(
                f"Plan cannot contain more than {self.MAX_STEPS} steps."
            )

        return goal, plan_steps

    @staticmethod
    def _render(
        goal: str,
        steps: list[PlanStep],
    ) -> str:
        lines = [
            "# Plan",
            "",
            "## Goal",
            goal,
            "",
            "## Steps",
            "",
        ]

        lines.extend(
            f"{item.number}. [{item.status}] {item.description}"
            for item in steps
        )

        return "\n".join(lines) + "\n"

    @staticmethod
    def _renumber(
        steps: list[PlanStep],
    ) -> None:
        for index, item in enumerate(steps, 1):
            item.number = index

    @staticmethod
    def _valid_text(
        value: Any,
        limit: int,
    ) -> bool:
        return (
            isinstance(value, str)
            and bool(value.strip())
            and len(value.strip()) <= limit
        )

    def _valid_steps(self, steps: Any) -> bool:
        if not isinstance(steps, list):
            return False

        if not 1 <= len(steps) <= self.MAX_STEPS:
            return False

        return all(
            self._valid_text(item, self.MAX_STEP_CHARS)
            for item in steps
        )

    def _success(
        self,
        summary: str,
        *,
        action: str,
    ) -> ToolResult:
        return ToolResult(
            success=True,
            name=self.name,
            content={
                "success": True,
                "summary": summary,
            },
            metadata={
                "effects": [
                    {
                        "action": action,
                        "target": str(self.PLAN_PATH),
                    }
                ]
            },
        )

    def _error(
        self,
        error_type: str,
        message: str,
    ) -> ToolResult:
        return ToolResult(
            success=False,
            name=self.name,
            content={
                "success": False,
                "error": {
                    "type": error_type,
                    "message": message,
                },
            },
            metadata={},
        )

    def __repr__(self) -> str:
        return "<Tool name='plan'>"
