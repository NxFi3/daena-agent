from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agent.planstate import PlanState, PlanStepState
from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool


@dataclass
class PlanStep:
    number: int
    status: str
    description: str


class Plan(Tool):
    """
    Manage Daena's current execution plan.

    The plan file is the source of truth.
    ContextBuilder reads it on the next model call.

    The generated plan.md also contains explicit execution instructions
    reminding the agent to keep the plan synchronized with real work.
    """

    name = "plan"
    action = "modify"

    description = (
        "Your execution checklist for multi-step tasks. The current plan is shown "
        "in your context under <plan>. It is only accurate if you keep it updated. "
        "A plan whose statuses do not match the real work is a bug.\n"
        "\n"
        "WHEN TO UPDATE (this is part of the work, not optional):\n"
        "- Before starting a step: set it to in_progress. Keep only one step in_progress.\n"
        "- Right after a step's work succeeds (confirmed by a tool result): set it to "
        "completed. Do this before starting the next step. Do not batch these up for later.\n"
        "- If a step fails and you cannot fix it: set it to blocked, then add a step "
        "for the fix.\n"
        "- If you discover new required work: add_step before doing it.\n"
        "- Before your final answer: every step must be completed or blocked.\n"
        "- Never mark a step completed unless the work actually succeeded.\n"
        "- Do not repeat an update that would leave the plan unchanged.\n"
        "- Do not create duplicate steps with the same description.\n"
        "\n"
        "OPERATIONS (each update call changes exactly ONE thing):\n"
        '- create (only when no plan exists): {"operation":"create","goal":"...","steps":["...","..."]}\n'
        '- start a step: {"operation":"update","step":2,"status":"in_progress"}\n'
        '- finish a step: {"operation":"update","step":2,"status":"completed"}\n'
        '- block a step: {"operation":"update","step":2,"status":"blocked"}\n'
        '- reword a step: {"operation":"update","step":2,"description":"..."}\n'
        '- append a step: {"operation":"update","add_step":"..."}\n'
        '- remove a step: {"operation":"update","remove_step":5}\n'
        '- change the goal: {"operation":"update","goal":"..."}\n'
        '- delete the plan: {"operation":"delete"}\n'
        "\n"
        "Do not combine goal, step, add_step and remove_step in one call. "
        "Use 1-based step numbers. Statuses: pending, in_progress, completed, blocked.\n"
        "\n"
        "WRITING GOOD STEPS: 5-12 concrete steps, ordered the way you will actually "
        "execute them, each with a clear finish condition. Include verification steps "
        "(build, tests, running the app), not only implementation."
    )

    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["create", "update", "delete"],
                "description": "Plan operation.",
            },
            "goal": {
                "type": "string",
                "description": (
                    "Plan goal. Required when creating a plan. "
                    "Use it in update to replace the current goal."
                ),
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Initial plan steps. Use with create. "
                    "Each item is a short concrete step."
                ),
            },
            "step": {
                "type": "integer",
                "minimum": 1,
                "description": "1-based step number to update.",
            },
            "status": {
                "type": "string",
                "enum": [
                    "pending",
                    "in_progress",
                    "completed",
                    "blocked",
                ],
                "description": "New status for the selected step.",
            },
            "description": {
                "type": "string",
                "description": "New description for the selected step.",
            },
            "add_step": {
                "type": "string",
                "description": "Add a new step to the end of the plan.",
            },
            "remove_step": {
                "type": "integer",
                "minimum": 1,
                "description": "Remove the given 1-based step number.",
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
        r"^\s*(\d+)\.\s+" r"\[(pending|in_progress|completed|blocked)\]\s+" r"(.+?)\s*$"
    )

    _PLAN_INSTRUCTIONS = [
        "This is the active execution plan.",
        "The agent MUST keep this plan synchronized with actual work.",
        "",
        "Plan update rules:",
        "- Before starting a step, mark it as [in_progress].",
        "- After a step's work succeeds, mark it as [completed] before starting the next step.",
        "- If a step cannot be completed, mark it as [blocked] and add a step describing the required fix.",
        "- If new required work is discovered, add a new step before performing that work.",
        "- Keep only one step [in_progress] at a time.",
        "- Never mark a step [completed] unless the corresponding work actually succeeded.",
        "- Do not repeat an update that would leave the plan unchanged.",
        "- Do not create duplicate steps with the same description.",
        "- Before finishing the task, all steps must be [completed] or [blocked].",
        "",
        "The plan is persistent and represents the current state of the task.",
        "Do not ignore or silently bypass it.",
    ]

    def validate(self, arguments: dict[str, Any]) -> bool:
        if not isinstance(arguments, dict):
            return False

        operation = arguments.get("operation")

        if operation not in {"create", "update", "delete"}:
            return False

        if operation == "create":
            return (
                set(arguments).issubset(
                    {
                        "operation",
                        "goal",
                        "steps",
                    }
                )
                and self._valid_text(
                    arguments.get("goal"),
                    self.MAX_GOAL_CHARS,
                )
                and self._valid_steps(arguments.get("steps"))
            )

        if operation == "delete":
            return set(arguments) == {"operation"}

        allowed = {
            "operation",
            "goal",
            "step",
            "status",
            "description",
            "add_step",
            "remove_step",
        }

        if not set(arguments).issubset(allowed):
            return False

        has_goal = arguments.get("goal") is not None

        has_step_edit = arguments.get("step") is not None and (
            arguments.get("status") is not None
            or arguments.get("description") is not None
        )

        has_add = arguments.get("add_step") is not None
        has_remove = arguments.get("remove_step") is not None

        modes = sum(
            (
                has_goal,
                has_step_edit,
                has_add,
                has_remove,
            )
        )

        if modes != 1:
            return False

        if has_goal:
            return self._valid_text(
                arguments.get("goal"),
                self.MAX_GOAL_CHARS,
            ) and set(arguments) == {
                "operation",
                "goal",
            }

        if has_step_edit:
            step = arguments.get("step")

            if type(step) is not int or step < 1:
                return False

            if (
                arguments.get("status") is not None
                and arguments.get("status") not in self._STATUSES
            ):
                return False

            if arguments.get("description") is not None and not self._valid_text(
                arguments.get("description"),
                self.MAX_STEP_CHARS,
            ):
                return False

            return set(arguments).issubset(
                {
                    "operation",
                    "step",
                    "status",
                    "description",
                }
            )

        if has_add:
            return set(arguments) == {
                "operation",
                "add_step",
            } and self._valid_text(
                arguments.get("add_step"),
                self.MAX_STEP_CHARS,
            )

        if has_remove:
            step = arguments.get("remove_step")

            return (
                set(arguments)
                == {
                    "operation",
                    "remove_step",
                }
                and type(step) is int
                and step >= 1
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

    def snapshot(
        self,
    ) -> PlanState:
        """
        Return the current plan as an immutable runtime snapshot.

        PlanState does not persist independently; plan.md remains the only
        source of truth.
        """
        try:
            raw = self._read_raw()
        except OSError as exc:
            return PlanState(
                exists=True,
                error=f"Could not read the plan: {exc}",
            )

        if not raw.strip():
            return PlanState.empty()

        try:
            goal, steps = self._parse(raw)
        except ValueError as exc:
            return PlanState(
                exists=True,
                error=str(exc),
            )

        return PlanState(
            exists=True,
            goal=goal,
            steps=tuple(
                PlanStepState(
                    number=item.number,
                    status=item.status,
                    description=item.description,
                )
                for item in steps
            ),
        )

    def execute(
        self,
        operation: str,
        goal: str | None = None,
        steps: list[str] | None = None,
        step: int | None = None,
        status: str | None = None,
        description: str | None = None,
        add_step: str | None = None,
        remove_step: int | None = None,
    ) -> ToolResult:

        if operation == "create":
            return self._create(
                goal=goal,
                steps=steps,
            )

        if operation == "update":
            return self._update(
                goal=goal,
                step=step,
                status=status,
                description=description,
                add_step=add_step,
                remove_step=remove_step,
            )

        if operation == "delete":
            return self._delete()

        return self._error(
            "invalid_operation",
            "operation must be create, update, or delete.",
        )

    def _create(
        self,
        *,
        goal: str | None,
        steps: list[str] | None,
    ) -> ToolResult:

        if not self._valid_text(
            goal,
            self.MAX_GOAL_CHARS,
        ):
            return self._error(
                "invalid_argument",
                "goal must be a non-empty string.",
            )

        if not self._valid_steps(steps):
            return self._error(
                "invalid_argument",
                f"steps must contain 1-{self.MAX_STEPS} non-empty strings.",
            )

        normalized_steps = self._normalize_steps(steps or [])

        if self._has_duplicate_values(normalized_steps):
            return self._error(
                "duplicate_step",
                "Plan cannot contain duplicate step descriptions.",
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
                "A plan already exists. Use update or delete.",
            )

        plan_steps = [
            PlanStep(
                number=index,
                status="pending",
                description=str(value).strip(),
            )
            for index, value in enumerate(
                steps or [],
                start=1,
            )
        ]

        content = self._render(
            str(goal).strip(),
            plan_steps,
        )

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
        goal: str | None,
        step: int | None,
        status: str | None,
        description: str | None,
        add_step: str | None,
        remove_step: int | None,
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

        if goal is not None:

            if not self._valid_text(
                goal,
                self.MAX_GOAL_CHARS,
            ):
                return self._error(
                    "invalid_argument",
                    "goal must be a non-empty string.",
                )

            new_goal = goal.strip()

            if self._same_text(
                new_goal,
                current_goal,
            ):
                return self._success(
                    "Plan unchanged: goal already has this value.",
                    action="update",
                    duplicate=True,
                )

            current_goal = new_goal

            summary = "Plan updated: goal changed."

        elif step is not None:

            if not 1 <= step <= len(current_steps):
                return self._error(
                    "invalid_step",
                    f"step must be between 1 and {len(current_steps)}.",
                )

            target = current_steps[step - 1]

            # Description update
            if description is not None:

                if not self._valid_text(
                    description,
                    self.MAX_STEP_CHARS,
                ):
                    return self._error(
                        "invalid_argument",
                        "description must be a non-empty string.",
                    )

                new_description = description.strip()

                if self._same_text(
                    new_description,
                    target.description,
                ):
                    return self._success(
                        f"Plan unchanged: step {step} already has this description.",
                        action="update",
                        duplicate=True,
                    )

                # Prevent duplicate descriptions.
                if self._description_exists(
                    current_steps,
                    new_description,
                    exclude_index=step - 1,
                ):
                    return self._error(
                        "duplicate_step",
                        f"Another step already has the description: '{new_description}'.",
                    )

                target.description = new_description

                summary = f"Plan updated: step {step} description."

            # Status update
            elif status is not None:

                if status not in self._STATUSES:
                    return self._error(
                        "invalid_status",
                        "Invalid step status.",
                    )

                # Exact duplicate update.
                if status == target.status:
                    return self._success(
                        (f"Plan unchanged: step {step} " f"is already {status}."),
                        action="update",
                        duplicate=True,
                    )

                # Prevent multiple active steps.
                if status == "in_progress":

                    active_step = self._find_in_progress(current_steps)

                    if active_step is not None and active_step.number != step:
                        return self._error(
                            "active_step_exists",
                            (
                                f"Step {active_step.number} is already "
                                "in_progress. Complete or block it before "
                                f"starting step {step}."
                            ),
                        )

                # Prevent reopening finalized steps accidentally.
                if (
                    target.status
                    in {
                        "completed",
                        "blocked",
                    }
                    and status == "in_progress"
                ):
                    return self._error(
                        "invalid_transition",
                        (
                            f"Step {step} is already {target.status} "
                            "and cannot be started again."
                        ),
                    )

                # Only allow sensible lifecycle transitions.
                if not self._valid_status_transition(
                    current=target.status,
                    new=status,
                ):
                    return self._error(
                        "invalid_transition",
                        (
                            f"Cannot change step {step} "
                            f"from {target.status} to {status}."
                        ),
                    )

                target.status = status

                summary = f"Plan updated: step {step} status."

            else:
                return self._error(
                    "invalid_argument",
                    "Step update requires status or description.",
                )

        elif add_step is not None:

            if not self._valid_text(
                add_step,
                self.MAX_STEP_CHARS,
            ):
                return self._error(
                    "invalid_argument",
                    "add_step must be a non-empty string.",
                )

            if len(current_steps) >= self.MAX_STEPS:
                return self._error(
                    "plan_limit",
                    f"Plan cannot contain more than {self.MAX_STEPS} steps.",
                )

            normalized_description = add_step.strip()

            if self._description_exists(
                current_steps,
                normalized_description,
            ):
                return self._error(
                    "duplicate_step",
                    ("A step with the same description " "already exists."),
                )

            new_number = len(current_steps) + 1

            current_steps.append(
                PlanStep(
                    number=new_number,
                    status="pending",
                    description=normalized_description,
                )
            )

            summary = f"Plan updated: added step {new_number}."

        elif remove_step is not None:

            if not 1 <= remove_step <= len(current_steps):
                return self._error(
                    "invalid_step",
                    f"step must be between 1 and {len(current_steps)}.",
                )

            if len(current_steps) == 1:
                return self._error(
                    "invalid_plan",
                    ("The last step cannot be removed. " "Delete the plan instead."),
                )

            removed = current_steps.pop(remove_step - 1)

            self._renumber(current_steps)

            summary = (
                f"Plan updated: removed step {remove_step} " f"({removed.description})."
            )

        else:
            return self._error(
                "invalid_argument",
                (
                    "Update must specify goal, "
                    "step with status/description, "
                    "add_step, or remove_step."
                ),
            )

        content = self._render(
            current_goal,
            current_steps,
        )

        if len(content) > self.MAX_PLAN_CHARS:
            return self._error(
                "plan_limit",
                "Updated plan is too large.",
            )

        write_error = self._write_atomic(content)

        if write_error is not None:
            return write_error

        return self._success(
            summary,
            action="update",
        )

    def _delete(self) -> ToolResult:
        try:
            self.PLAN_PATH.unlink(
                missing_ok=True,
            )
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

        return self.PLAN_PATH.read_text(
            encoding="utf-8",
        )

    def _write_atomic(
        self,
        content: str,
    ) -> ToolResult | None:

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

            os.replace(
                temp_path,
                self.PLAN_PATH,
            )

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

        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        if not lines or lines[0].strip() != "# Plan":
            raise ValueError("Plan must start with '# Plan'.")

        section_indices: dict[str, int] = {}

        for index, line in enumerate(lines):

            stripped = line.strip()

            if stripped.startswith("## "):

                section_name = stripped[3:].strip()

                if section_name in section_indices:
                    raise ValueError(f"Duplicate plan section: '{section_name}'.")

                section_indices[section_name] = index

        if "Goal" not in section_indices:
            raise ValueError("Plan must contain a '## Goal' section.")

        if "Steps" not in section_indices:
            raise ValueError("Plan must contain a '## Steps' section.")

        goal_index = section_indices["Goal"]
        steps_index = section_indices["Steps"]

        if goal_index >= steps_index:
            raise ValueError("Plan sections are out of order.")

        goal_lines = [
            line.strip() for line in lines[goal_index + 1 : steps_index] if line.strip()
        ]

        goal = " ".join(goal_lines).strip()

        if not self._valid_text(
            goal,
            self.MAX_GOAL_CHARS,
        ):
            raise ValueError("Plan goal is missing or invalid.")

        next_sections = [
            index for name, index in section_indices.items() if index > steps_index
        ]

        steps_end = min(next_sections) if next_sections else len(lines)

        plan_steps: list[PlanStep] = []

        for line in lines[steps_index + 1 : steps_end]:

            if not line.strip():
                continue

            match = self._STEP_RE.match(line)

            if match is None:
                raise ValueError("Plan contains an invalid step line.")

            number = int(match.group(1))

            status = match.group(2)

            description = match.group(3).strip()

            expected_number = len(plan_steps) + 1

            if number != expected_number:
                raise ValueError(
                    ("Plan step numbers must be " "contiguous starting at 1.")
                )

            if not self._valid_text(
                description,
                self.MAX_STEP_CHARS,
            ):
                raise ValueError((f"Plan step {number} " "is missing or too long."))

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
                (f"Plan cannot contain more than " f"{self.MAX_STEPS} steps.")
            )

        if self._has_duplicate_values([item.description for item in plan_steps]):
            raise ValueError("Plan contains duplicate step descriptions.")

        active_steps = [item for item in plan_steps if item.status == "in_progress"]

        if len(active_steps) > 1:
            raise ValueError("Plan cannot contain more than one in_progress step.")

        return goal, plan_steps

    @classmethod
    def _render(
        cls,
        goal: str,
        steps: list[PlanStep],
    ) -> str:

        lines = [
            "# Plan",
            "",
            "## Instructions",
            "",
        ]

        lines.extend(cls._PLAN_INSTRUCTIONS)

        lines.extend(
            [
                "",
                "## Goal",
                goal,
                "",
                "## Steps",
                "",
            ]
        )

        lines.extend(
            (f"{item.number}. " f"[{item.status}] " f"{item.description}")
            for item in steps
        )

        return "\n".join(lines) + "\n"

    @staticmethod
    def _renumber(
        steps: list[PlanStep],
    ) -> None:

        for index, item in enumerate(
            steps,
            start=1,
        ):
            item.number = index

    @staticmethod
    def _normalize_text(
        value: str,
    ) -> str:

        return " ".join(value.strip().split()).casefold()

    @classmethod
    def _same_text(
        cls,
        left: str,
        right: str,
    ) -> bool:

        return cls._normalize_text(left) == cls._normalize_text(right)

    @classmethod
    def _description_exists(
        cls,
        steps: list[PlanStep],
        description: str,
        *,
        exclude_index: int | None = None,
    ) -> bool:

        normalized = cls._normalize_text(description)

        for index, item in enumerate(steps):

            if exclude_index is not None and index == exclude_index:
                continue

            if cls._normalize_text(item.description) == normalized:
                return True

        return False

    @classmethod
    def _normalize_steps(
        cls,
        steps: list[str],
    ) -> list[str]:

        return [cls._normalize_text(item) for item in steps]

    @staticmethod
    def _has_duplicate_values(
        values: list[str],
    ) -> bool:

        return len(values) != len(set(values))

    @staticmethod
    def _find_in_progress(
        steps: list[PlanStep],
    ) -> PlanStep | None:

        for item in steps:
            if item.status == "in_progress":
                return item

        return None

    @staticmethod
    def _valid_status_transition(
        *,
        current: str,
        new: str,
    ) -> bool:

        allowed: dict[str, set[str]] = {
            "pending": {
                "in_progress",
                "blocked",
            },
            "in_progress": {
                "completed",
                "blocked",
            },
            "completed": set(),
            "blocked": set(),
        }

        return new in allowed.get(
            current,
            set(),
        )

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

    def _valid_steps(
        self,
        steps: Any,
    ) -> bool:

        if not isinstance(
            steps,
            list,
        ):
            return False

        if not 1 <= len(steps) <= self.MAX_STEPS:
            return False

        return all(
            self._valid_text(
                item,
                self.MAX_STEP_CHARS,
            )
            for item in steps
        )

    def _success(
        self,
        summary: str,
        *,
        action: str,
        duplicate: bool = False,
    ) -> ToolResult:

        return ToolResult(
            success=True,
            name=self.name,
            content={
                "success": True,
                "summary": summary,
                "duplicate": duplicate,
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
