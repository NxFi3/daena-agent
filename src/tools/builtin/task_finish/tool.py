from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool


class TaskFinish(Tool):
    """Explicit model-selected termination; no task-intent classification."""

    name = "task_finish"
    action = "respond"
    description = (
        "End this turn with your final answer. Ordinary assistant text is a progress update. "
        "Call this tool alone when finished, genuinely blocked, or needing user input. "
        "Use completed only for work actually done (or a conversational answer), never for a promise to start. "
        "Report concrete validation in evidence; an empty list is appropriate when no checks apply. "
        "Collect children and merge/discard pending edits first."
    )
    parameters = {
        "type": "object",
        "properties": {
            "outcome": {
                "type": "string",
                "enum": ["completed", "blocked", "needs_input"],
            },
            "response": {
                "type": "string",
                "description": "Final answer or the concrete blocker/question.",
            },
            "evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Checks actually performed and their results; [] when inapplicable.",
            },
        },
        "required": ["outcome", "response", "evidence"],
        "additionalProperties": False,
    }

    def validate(self, arguments):
        return (
            set(arguments) == {"outcome", "response", "evidence"}
            and isinstance(arguments["outcome"], str)
            and arguments["outcome"] in {"completed", "blocked", "needs_input"}
            and isinstance(arguments["response"], str)
            and bool(arguments["response"].strip())
            and isinstance(arguments["evidence"], list)
            and all(isinstance(item, str) for item in arguments["evidence"])
        )

    def execute(self, outcome, response, evidence):
        return ToolResult(
            success=True,
            name=self.name,
            content={"outcome": outcome, "response": response, "evidence": evidence},
            evidence={"outcome": outcome, "response": response, "checks": evidence},
            summary=f"Model reported turn outcome: {outcome}.",
        )
