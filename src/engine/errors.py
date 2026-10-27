"""Typed runtime/provider errors used for safe generation recovery."""


class RecoverableGenerationError(RuntimeError):
    """A generation failed before any tool call was safe to execute."""

    def __init__(self, message: str, *, recovery_hint: str = "") -> None:
        super().__init__(message)
        self.recovery_hint = str(recovery_hint or "").strip()


class OutputTokenLimitError(RecoverableGenerationError):
    """The provider stopped because the configured output budget was exhausted."""
