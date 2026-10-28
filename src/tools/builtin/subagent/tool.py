from copy import deepcopy

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool


class Subagent(Tool):
    name = "subagent"
    action = "delegate"
    allow_same_revision_repeat = True
    subagent_manager = None
    description = (
        "Delegate complex work to independent subagents in parallel when tasks can be separated cleanly; keep simple work in the parent. "
        "Each child has its own model session/context; worker edits happen in an isolated copy and must be explicitly reviewed and merged. "
        "Use message/broadcast to coordinate live while children run. Recipient 'all' publishes to the shared agent bus; bus reads recent shared messages and inbox collects newly delivered messages. "
        "spawn returns agent_id immediately. read_only permits reading, skills and allowed web tools; worker can edit its isolated copy. "
        "Use status/wait/wait_all to collect results, list to inspect tasks, and cancel to request cooperative stopping. "
        "After completion use diff to review changes, then merge or discard each worker; conflicts leave parent files unchanged. Verify integrated changes and resolve pending workers before finishing. "
        "Children cannot spawn more children or gain permissions beyond the parent."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "spawn",
                    "status",
                    "wait",
                    "wait_all",
                    "list",
                    "cancel",
                    "diff",
                    "merge",
                    "discard",
                    "message",
                    "broadcast",
                    "inbox",
                    "bus",
                ],
            },
            "task": {
                "type": "string",
                "description": "Required for spawn; include all necessary task context.",
            },
            "name": {"type": "string", "maxLength": 80, "description": "Short task label shown in the agent graph."},
            "context": {"type": "string", "maxLength": 24000, "description": "Relevant reference material; children do not inherit the whole chat."},
            "allowed_tools": {"type": "array", "items": {"type": "string"}, "description": "Optional narrower tool scope; never expands parent permissions."},
            "mode": {
                "type": "string",
                "enum": ["read_only", "worker"],
                "default": "read_only",
            },
            "agent_id": {
                "type": "string",
                "description": "Required for status, wait, cancel, diff, merge and discard.",
            },
            "text": {"type": "string", "maxLength": 4000, "description": "Live message content."},
            "recipient": {"type": "string", "description": "parent, a running agent_id, or all for a shared bus broadcast."},
            "wait_ms": {
                "type": "integer",
                "minimum": 0,
                "maximum": 300000,
                "default": 30000,
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "default": 20,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    }

    def execute(
        self,
        action: str,
        task: str = "",
        mode: str = "read_only",
        agent_id: str = "",
        wait_ms: int = 30000,
        name: str = "",
        context: str = "",
        allowed_tools: list | None = None,
        text: str = "",
        recipient: str = "",
        limit: int = 20,
    ) -> ToolResult:
        manager = self.subagent_manager
        if action == "spawn":
            result = manager.spawn(task, mode, name=name, context=context, allowed_tools=allowed_tools)
        elif action == "list":
            result = {
                "agents": [
                    {
                        key: value
                        for key, value in child.items()
                        if key not in {"response", "metrics"}
                    }
                    for child in manager.list()
                ],
                "hint": "Use status or wait for each agent_id to collect the full result.",
            }
        elif action == "status":
            result = manager.status(agent_id)
        elif action == "wait":
            result = manager.wait(agent_id, wait_ms)
        elif action == "wait_all":
            result = manager.wait_all(wait_ms)
        elif action == "cancel":
            result = manager.cancel(agent_id)
        elif action == "message":
            result = manager.send_message(sender="parent", recipient=recipient or agent_id, text=text)
        elif action == "broadcast":
            result = manager.send_message(sender="parent", recipient="all", text=text)
        elif action == "inbox":
            result = {"messages": manager.inbox("parent")}
        elif action == "bus":
            result = {"messages": manager.bus_history(limit=limit)}
        elif action == "diff":
            result = manager.diff(agent_id)
        elif action in {"merge", "discard"}:
            result = manager.integrate(agent_id, discard=action == "discard")
        else:
            raise ValueError("Unknown subagent action")
        # A failed child is data from a successful status operation; do not
        # confuse it with a failed tool invocation or retry the delegation.
        effects = []
        if action == "merge" and result.get("merged"):
            effects = [
                {"action": item["operation"], "target": item["path"]}
                for item in result["files"]
            ]
        return ToolResult(
            success=True,
            name=self.name,
            content=result,
            evidence=result,
            effects=effects,
            summary=f"Subagent {action}: {result.get('status', 'listed')}. Review child status separately from tool success.",
        )

    def get_definition(self):
        definition = deepcopy(super().get_definition())
        capabilities = self.subagent_manager.capabilities()
        definition["function"]["parameters"]["properties"]["mode"]["enum"] = (
            capabilities["modes"]
        )
        definition["function"]["description"] += (
            " Enabled modes: " + ", ".join(capabilities["modes"]) + "."
        )
        return definition
