from __future__ import annotations

import json
from typing import Any, Callable

from src.models.ContextEvent import ContextRole, ContextType
from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool


class ContextSearch(Tool):
    """Search earlier session messages and tool evidence without replaying history."""

    name = "context_search"
    action = "search"

    description = (
        "Search prior session context and tool results by specific keywords. Use when "
        "the current prompt no longer contains an earlier observation, fetch result, "
        "command output, file path, URL, or diagnostic. Returns a few bounded evidence "
        "snippets, not the full history. This is read-only; do not repeat actions just "
        "because an old result was recalled."
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Short, specific keywords or identifiers to find in prior context.",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "default": 3,
            },
            "include_tool_results": {
                "type": "boolean",
                "default": True,
                "description": "Include earlier web/command/file tool results.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self._stm = None
        self._session_id_provider: Callable[[], Any] | None = None

    def set_context_store(self, stm, session_id_provider: Callable[[], Any]) -> None:
        self._stm = stm
        self._session_id_provider = session_id_provider

    def validate(self, arguments: dict[str, Any]) -> bool:
        query = arguments.get("query")
        top_k = arguments.get("top_k", 3)
        include = arguments.get("include_tool_results", True)
        return (
            isinstance(query, str)
            and bool(query.strip())
            and len(query) <= 300
            and isinstance(top_k, int)
            and not isinstance(top_k, bool)
            and 1 <= top_k <= 5
            and isinstance(include, bool)
        )

    def execute(
        self,
        query: str,
        top_k: int = 3,
        include_tool_results: bool = True,
    ) -> ToolResult:
        if not isinstance(query, str) or not query.strip() or len(query) > 300:
            return self._error("invalid_query", "query must contain 1-300 characters.")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 5:
            return self._error("invalid_top_k", "top_k must be an integer from 1 to 5.")
        if not isinstance(include_tool_results, bool):
            return self._error("invalid_argument", "include_tool_results must be boolean.")
        if self._stm is None or self._session_id_provider is None:
            return self._error("context_store_unavailable", "Context search is not bound to a session.")
        session_id = self._session_id_provider()
        if session_id is None:
            return self._error("session_unavailable", "There is no active session to search.")

        try:
            events = self._stm.search(
                session_id=session_id,
                query=query.strip(),
                top_k=top_k,
                include_tool_results=include_tool_results,
            )
        except Exception as exc:
            return self._error("context_search_failed", f"Context search failed: {exc}")

        records = []
        for event in events:
            text = str(event.content or "").strip()
            tool_name = ""
            success = None
            summary = ""
            if event.role == ContextRole.TOOL and event.type == ContextType.TOOL_RESULT:
                try:
                    payload = json.loads(text)
                except (TypeError, ValueError):
                    payload = {}
                if isinstance(payload, dict):
                    tool_name = str(payload.get("name") or "")
                    if "success" in payload:
                        success = bool(payload.get("success"))
                    summary = str(payload.get("summary") or "")
                    result_content = payload.get("content")
                    if isinstance(result_content, dict):
                        excerpt = (
                            result_content.get("content")
                            or result_content.get("stdout")
                            or result_content.get("stderr")
                            or result_content.get("error")
                        )
                        if excerpt:
                            text = str(excerpt)
                        else:
                            text = json.dumps(result_content, ensure_ascii=False, default=str)
                    elif result_content is not None:
                        text = str(result_content)
                    if summary and not text:
                        text = summary

            text = text[:1400]
            record = {
                "step": int(getattr(event, "step", 0) or 0),
                "role": event.role.value,
                "type": event.type.value,
                "tool": tool_name or None,
                "success": success,
                "summary": summary[:300] or None,
                "excerpt": text,
            }
            records.append(record)

        return ToolResult(
            success=True,
            name=self.name,
            content={
                "query": query.strip(),
                "result_count": len(records),
                "results": records,
            },
            summary=f"Context search matched {len(records)} earlier record(s) for {query.strip()!r}.",
            evidence={"query": query.strip(), "result_count": len(records)},
        )

    def _error(self, error_type: str, message: str) -> ToolResult:
        return ToolResult(
            success=False,
            name=self.name,
            content={"success": False, "error": {"type": error_type, "message": message}},
            summary=message,
            evidence={"error_type": error_type},
        )
