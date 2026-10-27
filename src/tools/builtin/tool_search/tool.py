from __future__ import annotations

from typing import Any, Callable

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool


class ToolSearch(Tool):
    """Find relevant runtime tools and activate their schemas for the next turn."""

    name = "tool_search"
    model_exposure = "always"
    action = "search"
    description = (
        "Search the agent's full tool catalog for a capability whose schema is not currently exposed. "
        "Some tools are always exposed; specialized capabilities are loaded on demand. "
        "Use this when the needed tool is not currently exposed. "
        "Write a short natural-language query describing the intended outcome or action, "
        "not merely the name of a tool. Matching tools are made callable on the next turn, "
        "with their full argument schemas. Only the currently selected tools are exposed; "
        "search again when another capability is needed. Do not call this for ordinary "
        "conversation or questions that can be answered without tools."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 2,
                "description": "Natural-language capability or action to find tools for.",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 8,
                "default": 5,
                "description": "Maximum number of tools to activate for the next turn (default 5).",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self._searcher: Callable[[str, int], dict[str, Any]] | None = None

    def set_searcher(self, searcher: Callable[[str, int], dict[str, Any]]) -> None:
        self._searcher = searcher

    def execute(self, query: str, top_k: int = 5) -> ToolResult:
        if not isinstance(query, str) or len(query.strip()) < 2:
            return ToolResult(
                success=False,
                name=self.name,
                content={"success": False, "error": {
                    "type": "invalid_argument",
                    "message": "query must describe the needed capability in at least two characters.",
                }},
                metadata={},
            )
        if self._searcher is None:
            return ToolResult(
                success=False,
                name=self.name,
                content={"success": False, "error": {
                    "type": "tool_search_unavailable",
                    "message": "The runtime tool catalog is not connected to tool_search.",
                }},
                metadata={},
            )

        try:
            outcome = self._searcher(query.strip(), top_k)
        except Exception as exc:
            return ToolResult(
                success=False,
                name=self.name,
                content={"success": False, "error": {
                    "type": "tool_search_error",
                    "message": f"Tool catalog search failed: {type(exc).__name__}: {exc}",
                }},
                metadata={},
            )

        matches = outcome.get("matches", [])
        active_names = outcome.get("active_tools", [])
        if matches:
            summary = (
                f"Found {len(matches)} relevant tool(s) and activated them for the next turn: "
                + ", ".join(str(item.get("name", "")) for item in matches)
                + ". Their full schemas will be supplied on the next model call."
            )
        else:
            summary = (
                "No matching additional tools were found. Core shell/file/process tools remain available; "
                "retry with concrete action words or a different description of the extra capability."
            )

        return ToolResult(
            success=True,
            name=self.name,
            content={
                "query": query.strip(),
                "matches": matches,
                "active_tools_for_next_turn": active_names,
                "next_step": (
                    "Choose and call the newly exposed tool(s) that best advance the task."
                    if matches else
                    "Refine the query before attempting the unavailable action."
                ),
            },
            metadata={"active_tools": active_names},
            summary=summary,
        )
