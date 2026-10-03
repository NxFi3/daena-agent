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

    The plan file is the source of truth and lives at
    ".daena/plan.md" inside the active workspace.
    ContextBuilder reads it automatically on the next model call.

    The generated plan.md also contains explicit execution instructions
    reminding the agent to keep the plan synchronized with real work.
    """

    name = "plan"
    action = "modify"

    description = (
        "Manage the execution plan for the current task. Keep this tool simple: "
        "the runtime owns the current step and status. You only decide what to do next.\n\n"
        "A plan lives at .daena/plan.md in the active workspace and is already "
        "provided to you in <plan>. Do not read that file with read_file or search.\n\n"
        "ACTIONS:\n"
        '- create: start a new plan. {"action":"create","goal":"...","steps":["...","..."]}\n'
        '- complete: finish the current step after its work actually succeeded. {"action":"complete"}\n'
        '- block: mark the current step blocked when it cannot be completed. {"action":"block","reason":"..."}\n'
        '- add: append newly discovered required work. {"action":"add","step":"..."}\n'
        "Completing or blocking a step automatically advances the next pending step. "
        "Never try to choose a step number or manually set a status. "
        "Use exactly one action per plan call. Include verification work in the plan. "
        "Before the final answer, every step must be completed or blocked."
    )

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "complete", "block", "add"],
                "description": "What to do with the current execution plan.",
            },
            "goal": {
                "type": "string",
                "description": "Goal for create.",
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Initial ordered steps for create.",
            },
            "reason": {
                "type": "string",
                "description": "Why the current step is blocked.",
            },
            "step": {
                "type": "string",
                "description": "New step to append with add.",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    }

    # Plan state belongs to the active workspace, not Daena's installation.
    # Runtime callers set the active workspace through set_workspace().
    PLAN_PATH = Path(".daena") / "plan.md"

    def __init__(self) -> None:
        self._workspace_root: Path | None = None

    def set_workspace(self, directory: str | Path) -> None:
        self._workspace_root = Path(directory).expanduser().resolve()

    @property
    def _plan_path(self) -> Path:
        if self._workspace_root is None:
            return self.PLAN_PATH
        return self._workspace_root / self.PLAN_PATH

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
        "The runtime owns the current step and status.",
        "",
        "Plan rules:",
        "- The first step becomes [in_progress] when the plan is created.",
        "- Use [complete] only after the current step's work actually succeeded.",
        "- Use [block] when the current step cannot be completed; give a concise reason.",
        "- Completing or blocking a step automatically advances the next pending step.",
        "- Use [add] when new required work is discovered before doing that work.",
        "- Keep exactly one step [in_progress] at a time.",
        "- Before finishing the task, all steps must be [completed] or [blocked].",
        "The plan is persistent and represents the current task state.",
    ]

    def validate(self, arguments: dict[str, Any]) -> bool:
        if not isinstance(arguments, dict):
            return False

        action = arguments.get("action")

        if action == "create":
            return (
                set(arguments) == {"action", "goal", "steps"}
                and self._valid_text(arguments.get("goal"), self.MAX_GOAL_CHARS)
                and self._valid_steps(arguments.get("steps"))
            )

        if action == "complete":
            return set(arguments) == {"action"}

        if action == "block":
            return (
                set(arguments) == {"action", "reason"}
                and self._valid_text(arguments.get("reason"), self.MAX_STEP_CHARS)
            )

        if action == "add":
            return (
                set(arguments) == {"action", "step"}
                and self._valid_text(arguments.get("step"), self.MAX_STEP_CHARS)
            )

        return False

    def describe_call(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, str]:
        return {
            "action": "modify",
            "target": str(self._plan_path),
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
        action: str | None = None,
        goal: str | None = None,
        steps: list[str] | None = None,
        reason: str | None = None,
        step: str | None = None,
        # Legacy direct-call compatibility. These fields are not exposed in the
        # model schema, but keeping them here avoids breaking existing callers/tests.
        operation: str | None = None,
        status: str | None = None,
        description: str | None = None,
        add_step: str | None = None,
        remove_step: int | None = None,
    ) -> ToolResult:

        if action is None and operation is not None:
            if operation == "create":
                action = "create"
            elif operation == "delete":
                return self._error(
                    "unsupported_action",
                    "Plan deletion is no longer available to the model.",
                )
            elif operation == "update":
                if status == "completed":
                    action = "complete"
                elif status == "blocked":
                    action = "block"
                elif add_step is not None:
                    action = "add"
                    step = add_step
                else:
                    return self._error(
                        "unsupported_action",
                        "Use create, complete, block, or add.",
                    )

        if action == "create":
            return self._create(goal=goal, steps=steps)

        if action == "complete":
            return self._set_current_status("completed")

        if action == "block":
            return self._set_current_status("blocked", reason=reason)

        if action == "add":
            return self._add_step(step)

        return self._error(
            "invalid_action",
            "action must be create, complete, block, or add.",
        )

    def _set_current_status(
        self,
        status: str,
        *,
        reason: str | None = None,
    ) -> ToolResult:
        try:
            raw = self._read_raw()
        except OSError as exc:
            return self._error("read_error", f"Could not read the plan: {exc}")

        if not raw.strip():
            return self._error("plan_missing", "No plan exists. Use create first.")

        try:
            current_goal, current_steps = self._parse(raw)
        except ValueError as exc:
            return self._error("invalid_plan", str(exc))

        current = self._find_in_progress(current_steps)
        if current is None:
            return self._error(
                "no_active_step",
                "There is no in_progress step to update.",
            )

        current.status = status

        if status == "completed":
            next_active = None
            for item in current_steps:
                if item.status == "pending":
                    item.status = "in_progress"
                    next_active = item
                    break

            if next_active is not None:
                summary = (
                    f"Plan updated: step {current.number} completed; "
                    f"step {next_active.number} is now in_progress."
                )
            else:
                summary = (
                    f"Plan updated: step {current.number} completed; "
                    "the plan is now complete."
                )
        else:
            summary = f"Plan updated: step {current.number} blocked."
            if reason and reason.strip():
                summary += f" Reason: {reason.strip()}"
            for item in current_steps:
                if item.status == "pending":
                    item.status = "in_progress"
                    summary += f" Step {item.number} is now in_progress."
                    break

        content = self._render(current_goal, current_steps)
        write_error = self._write_atomic(content)
        if write_error is not None:
            return write_error

        return self._success(summary, action="update")

    def _add_step(self, step: str | None) -> ToolResult:
        if not self._valid_text(step, self.MAX_STEP_CHARS):
            return self._error(
                "invalid_argument",
                "step must be a non-empty string.",
            )

        try:
            raw = self._read_raw()
        except OSError as exc:
            return self._error("read_error", f"Could not read the plan: {exc}")

        if not raw.strip():
            return self._error("plan_missing", "No plan exists. Use create first.")

        try:
            current_goal, current_steps = self._parse(raw)
        except ValueError as exc:
            return self._error("invalid_plan", str(exc))

        normalized = self._normalize_text(step or "")
        if any(
            self._normalize_text(item.description) == normalized
            for item in current_steps
        ):
            return self._error(
                "duplicate_step",
                "A step with the same description already exists.",
            )

        if len(current_steps) >= self.MAX_STEPS:
            return self._error(
                "plan_limit",
                f"Plan cannot contain more than {self.MAX_STEPS} steps.",
            )

        number = len(current_steps) + 1
        current_steps.append(
            PlanStep(
                number=number,
                status="pending",
                description=str(step).strip(),
            )
        )
        content = self._render(current_goal, current_steps)
        write_error = self._write_atomic(content)
        if write_error is not None:
            return write_error

        return self._success(
            f"Plan updated: added step {number}.",
            action="update",
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
                "A plan already exists. Continue using the current plan.",
            )

        plan_steps = [
            PlanStep(
                number=index,
                status=("in_progress" if index == 1 else "pending"),
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

                if status == "completed":
                    next_active: PlanStep | None = None
                    for next_step in current_steps:
                        if next_step.status == "pending":
                            next_step.status = "in_progress"
                            next_active = next_step
                            break

                    if next_active is not None:
                        summary = (
                            f"Plan updated: step {step} completed; "
                            f"step {next_active.number} is now in_progress."
                        )
                    else:
                        summary = f"Plan updated: step {step} completed; the plan is now complete."
                else:
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
            self._plan_path.unlink(
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
        plan_path = self._plan_path
        if not plan_path.exists():
            return ""

        return plan_path.read_text(
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
            plan_path = self._plan_path
            plan_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=plan_path.parent,
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
                plan_path,
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
                        "target": str(self._plan_path),
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
