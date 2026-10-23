"""Daena adapter; the engine is also usable without the Daena source tree."""

try:
    from src.models.ToolResult import ToolResult
    from src.tools.Tool import Tool
except ModuleNotFoundError as exc:
    if exc.name not in {
        "src",
        "src.models",
        "src.models.ToolResult",
        "src.tools",
        "src.tools.Tool",
    }:
        raise
    from dataclasses import dataclass, field
    from typing import Any

    class Tool:
        def get_definition(self):
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.parameters,
                },
            }

    @dataclass
    class ToolResult:
        success: bool
        name: str
        content: dict[str, Any] = field(default_factory=dict)
        metadata: dict[str, Any] = field(default_factory=dict)
        summary: str = ""
        evidence: dict[str, Any] = field(default_factory=dict)
        effects: list[dict[str, str]] = field(default_factory=list)

        def to_observation(self):
            return {
                "tool": self.name,
                "success": self.success,
                "summary": self.summary,
                "evidence": self.evidence,
                "effects": self.effects,
            }
