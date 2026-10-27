"""Live, bounded agent-to-agent messaging over the in-process task manager.

Messages are data, not instructions from a higher-trust role. Delivery to an
active child uses its existing Agent.steer() queue at the next model turn.
"""
from __future__ import annotations

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool


class AgentMessage(Tool):
    name = "agent_message"
    action = "communicate"
    description = (
        "Send a live message to the parent or another running subagent, or "
        "read messages addressed to this child. Messages are delivered within "
        "the running agents and do not grant extra privileges. Use this only "
        "when coordination materially helps the task."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["send", "inbox"]},
            "recipient": {"type": "string", "description": "parent or a known agent_id (send only)"},
            "text": {"type": "string", "minLength": 1, "maxLength": 4000},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
        },
        "required": ["action"], "additionalProperties": False,
    }

    def __init__(self, manager, child_id: str):
        self.manager = manager
        self.child_id = child_id

    def execute(self, action: str, recipient: str = "parent", text: str = "",
                limit: int = 20) -> ToolResult:
        try:
            if action == "send":
                result = self.manager.send_message(
                    sender=self.child_id, recipient=recipient, text=text
                )
            elif action == "inbox":
                result = {"messages": self.manager.inbox(self.child_id, limit=limit)}
            else:
                raise ValueError("Unknown messaging action")
            return ToolResult(success=True, name=self.name, content=result)
        except (ValueError, RuntimeError) as exc:
            return ToolResult(success=False, name=self.name,
                              content={"error": {"type": "message_failed", "message": str(exc)}})
