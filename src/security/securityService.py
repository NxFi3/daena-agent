from __future__ import annotations

from src.models.ToolCall import ToolCall
from src.security.Policy import SecurityPolicy
from src.security.Sandbox import WorkspaceSandbox


class SecurityService:
    """Single security gate used by ToolManager before every execution."""

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self.policy = SecurityPolicy.from_config(config)
        self.sandbox = WorkspaceSandbox()
        # Compatibility knob retained, but disabled by default. Explicitly
        # enabling it is an intentional opt-out from policy enforcement.
        self.force_approve = bool(config.get("force_approve", False))

    def set_workspace(self, workspace: str) -> None:
        self.sandbox.set_root(workspace)

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

        if self.force_approve:
            toolcall.approved = True
            toolcall.security_reason = "Explicit force_approve override."
            toolcall.security_rule = "force_approve"
            return toolcall

        decision = self.policy.evaluate(toolcall, self.sandbox)

        toolcall.approved = decision.allowed
        toolcall.security_reason = decision.reason
        toolcall.security_rule = decision.rule
        return toolcall


# Backward-compatible name used by older imports.
security = SecurityService
