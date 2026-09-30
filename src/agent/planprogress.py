from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agent.planstate import PlanState
from src.models.ToolCall import ToolCall
from src.models.ToolResult import ToolResult


@dataclass
class StepProgress:
    """Runtime evidence collected while one plan step is active."""

    number: int
    started_iteration: int
    started_revision: int
    successful_actions: int = 0
    failed_actions: int = 0
    last_result_success: bool = False
    last_result_terminal: bool = False
    last_result_tool: str = ""
    last_result_iteration: int = 0


class PlanProgressTracker:
    """
    Track execution evidence for the active plan step without interpreting
    the step's natural-language description.

    Plan.md remains the source of truth for lifecycle state. This tracker only
    answers a narrower runtime question: has the active step produced any
    successful tool work since it became active?
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.current_step: StepProgress | None = None
        self._history: dict[int, StepProgress] = {}

    def sync(
        self,
        state: PlanState,
        *,
        iteration: int,
        workspace_revision: int,
    ) -> None:
        active = state.current_step

        if active is None:
            self.current_step = None
            return

        if self.current_step is not None and self.current_step.number == active.number:
            return

        progress = self._history.get(active.number)
        if progress is None:
            progress = StepProgress(
                number=active.number,
                started_iteration=iteration,
                started_revision=workspace_revision,
            )
            self._history[active.number] = progress

        self.current_step = progress

    def record(
        self,
        tool_call: ToolCall,
        result: ToolResult,
        *,
        iteration: int = 0,
    ) -> None:
        if self.current_step is None:
            return

        if str(getattr(tool_call, "name", "")).strip().lower() == "plan":
            return

        content = result.content if isinstance(result.content, dict) else {}
        status = str(content.get("status", "") or "").strip().lower()
        terminal = status not in {"running"}

        self.current_step.last_result_success = bool(result.success)
        self.current_step.last_result_terminal = terminal
        self.current_step.last_result_tool = str(
            getattr(tool_call, "name", result.name)
        )
        self.current_step.last_result_iteration = int(iteration or 0)

        if result.success:
            self.current_step.successful_actions += 1
        else:
            self.current_step.failed_actions += 1

    def can_complete(self, step_number: int) -> bool:
        progress = self._history.get(step_number)
        if progress is None:
            return False

        return (
            progress.successful_actions > 0
            and progress.last_result_success
            and progress.last_result_terminal
        )

    def has_work(self, step_number: int) -> bool:
        progress = self._history.get(step_number)
        return bool(progress and progress.successful_actions > 0)

    def context(self) -> dict[str, Any]:
        progress = self.current_step
        if progress is None:
            return {}

        return {
            "step": progress.number,
            "started_iteration": progress.started_iteration,
            "started_revision": progress.started_revision,
            "successful_actions": progress.successful_actions,
            "failed_actions": progress.failed_actions,
            "last_result_success": progress.last_result_success,
            "last_result_terminal": progress.last_result_terminal,
            "last_result_tool": progress.last_result_tool,
            "last_result_iteration": progress.last_result_iteration,
        }
