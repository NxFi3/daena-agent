from __future__ import annotations

from typing import Any

from src.models.ToolCall import ToolCall
from src.security.Policy import SecurityPolicy
from src.security.Sandbox import WorkspaceSandbox


class SecurityService:
    """Single security gate used by ToolManager before every execution."""

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self.policy = SecurityPolicy.from_config(config)
        self.sandbox = WorkspaceSandbox()

        # Background approvals are exact-command, session-scoped grants.
        # They disappear when this SecurityService instance is discarded.
        self._approved_background_commands: set[
            tuple[str, tuple[str, ...]]
        ] = set()

        # Compatibility knob retained, but disabled by default. Explicitly
        # enabling it is an intentional opt-out from policy enforcement.
        self.force_approve = bool(config.get("force_approve", False))

    def set_workspace(self, workspace: str) -> None:
        self.sandbox.set_root(workspace)

    def approve_background_command(
        self,
        command: list[str],
        workdir: str | None = None,
    ) -> None:
        """Approve one exact background command for the current session."""
        key = self._background_command_key(command, workdir)
        self._approved_background_commands.add(key)

    def revoke_background_command(
        self,
        command: list[str],
        workdir: str | None = None,
    ) -> None:
        """Revoke a previously approved background command."""
        key = self._background_command_key(command, workdir)
        self._approved_background_commands.discard(key)

    def clear_background_approvals(self) -> None:
        """Remove all session-scoped background command approvals."""
        self._approved_background_commands.clear()

    def check(self, toolcall: ToolCall) -> ToolCall:
        if not isinstance(toolcall, ToolCall):
            raise TypeError("security.check expects a ToolCall.")

        toolcall.approved = False
        toolcall.security_reason = ""
        toolcall.security_rule = ""

        if not toolcall.valid:
            toolcall.security_reason = "Tool call is invalid."
            toolcall.security_rule = "invalid_call"
            return toolcall

        # force_approve may bypass an interactive approval step, but it must
        # never bypass the actual security policy (workspace boundaries,
        # blocked executables, inline evaluation, network policy, etc.).
        decision = self.policy.evaluate(toolcall, self.sandbox)

        if (
            not decision.allowed
            and decision.rule == "background_disabled"
            and self._is_approved_background_command(toolcall)
        ):
            decision = type(decision)(
                allowed=True,
                reason="Background command approved for this session.",
                rule="background_session_approval",
            )

        toolcall.approved = decision.allowed
        toolcall.security_reason = decision.reason
        toolcall.security_rule = decision.rule

        if toolcall.approved and self.force_approve:
            toolcall.security_reason = (
                "Allowed by security policy; force_approve skipped interactive approval."
            )
            toolcall.security_rule = "force_approve"

        return toolcall

    def _is_approved_background_command(
        self,
        toolcall: ToolCall,
    ) -> bool:
        args: dict[str, Any] = getattr(toolcall, "args", {}) or {}
        command = args.get("command")
        workdir = args.get("workdir")

        if not isinstance(command, list):
            return False

        if workdir is not None and not isinstance(workdir, str):
            return False

        try:
            key = self._background_command_key(command, workdir)
        except (PermissionError, ValueError, TypeError):
            return False

        return key in self._approved_background_commands

    def _background_command_key(
        self,
        command: list[str],
        workdir: str | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        if not isinstance(command, list) or not command:
            raise ValueError("command must be a non-empty list.")

        if not all(isinstance(item, str) for item in command):
            raise TypeError("command arguments must all be strings.")

        resolved_workdir = self.sandbox.resolve(
            workdir or "."
        )

        return (
            str(resolved_workdir),
            tuple(command),
        )


# Backward-compatible name used by older imports.
security = SecurityService
