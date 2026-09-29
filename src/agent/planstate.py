"""Runtime snapshot of the active execution plan."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanStepState:
    number: int
    status: str
    description: str


@dataclass(frozen=True)
class PlanState:
    """
    Immutable runtime snapshot of plan.md.

    PlanState does not own persistence. Plan Tool remains the source of truth
    for the file; Loop uses this snapshot to enforce execution ordering.
    """

    exists: bool = False
    goal: str = ""
    steps: tuple[PlanStepState, ...] = ()
    error: str | None = None

    @classmethod
    def empty(cls) -> "PlanState":
        return cls()

    @property
    def current_step(self) -> PlanStepState | None:
        for step in self.steps:
            if step.status == "in_progress":
                return step
        return None

    @property
    def next_pending_step(self) -> PlanStepState | None:
        for step in self.steps:
            if step.status == "pending":
                return step
        return None

    @property
    def has_active_step(self) -> bool:
        return self.current_step is not None

    @property
    def is_complete(self) -> bool:
        if not self.exists or self.error:
            return False

        return bool(self.steps) and all(
            step.status in {"completed", "blocked"}
            for step in self.steps
        )

    @property
    def requires_step_start(self) -> bool:
        return (
            self.exists
            and not self.error
            and not self.is_complete
            and self.current_step is None
        )

    def step(self, number: int) -> PlanStepState | None:
        for item in self.steps:
            if item.number == number:
                return item
        return None
