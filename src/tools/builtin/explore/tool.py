from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.models.ContextEvent import (
    ContextEvent,
    ContextPriority,
    ContextRole,
    ContextType,
)
from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool
from src.tools.builtin.workspace_utils import workspace_path


class Explore(Tool):
    name = "explore"
    action = "inspect"

    description = (
        "Delegate repository exploration to a read-only subagent with its own "
        "context. Use for broad codebase questions that would otherwise require "
        "many grep/glob/read_file calls in the main context. The subagent returns "
        "a concise report and cannot modify files, run commands, or change plans."
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What you need to learn about the repository.",
            },
            "path": {
                "type": "string",
                "default": ".",
                "description": "Workspace-relative area to focus on.",
            },
            "max_iterations": {
                "type": "integer",
                "minimum": 2,
                "maximum": 24,
                "default": 12,
                "description": "Maximum read-only reasoning/tool iterations.",
            },
            "max_report_chars": {
                "type": "integer",
                "minimum": 500,
                "maximum": 12000,
                "default": 6000,
                "description": "Maximum report size returned to the main agent.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self._workspace_root: Path | None = None
        self._llm_provider = None
        self._runtime_config = {}
        self._process_manager = None

    def set_workspace(self, directory: str | Path) -> None:
        self._workspace_root = Path(directory).expanduser().resolve()

    def set_llm_provider(self, llm_provider) -> None:
        self._llm_provider = llm_provider

    def set_runtime_config(self, config):
        self._runtime_config = config

    def set_process_manager(self, manager):
        self._process_manager = manager

    def execute(
        self,
        query: str,
        path: str = ".",
        max_iterations: int = 12,
        max_report_chars: int = 6000,
    ) -> ToolResult:
        if self._workspace_root is None:
            return self._error("workspace_missing", "No workspace is configured.")
        if self._llm_provider is None:
            return self._error("llm_missing", "No LLM provider is available for exploration.")
        if not isinstance(query, str) or not query.strip():
            return self._error("invalid_argument", "query must be a non-empty string.")
        if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or not 2 <= max_iterations <= 24:
            return self._error("invalid_argument", "max_iterations must be between 2 and 24.")
        if not isinstance(max_report_chars, int) or isinstance(max_report_chars, bool) or not 500 <= max_report_chars <= 12000:
            return self._error("invalid_argument", "max_report_chars must be between 500 and 12000.")

        try:
            focus = str(workspace_path(self._workspace_root, path).relative_to(self._workspace_root))
        except (ValueError, TypeError, PermissionError, OSError) as exc:
            return self._error("invalid_path", str(exc))
        isolated_config = copy.deepcopy(self._build_config(max_iterations))

        try:
            from src.agent.agentloop import Loop

            with tempfile.TemporaryDirectory(prefix="daena-explore-") as temp_dir:
                isolated_config.setdefault("memory", {})["database"] = str(
                    Path(temp_dir) / "stm.sqlite3"
                )

                loop = Loop(isolated_config, self._llm_provider)
                try:
                    loop.session_id = uuid4()
                    loop.set_workspace(str(self._workspace_root))

                    task = ContextEvent(
                        role=ContextRole.USER,
                        type=ContextType.MESSAGE,
                        content=(
                            "Explore this repository in read-only mode and answer the following.\n\n"
                            f"Focus path: {focus}\n"
                            f"Question: {query.strip()}\n\n"
                            "Use grep/glob/list_dir/read_file to gather concrete evidence. "
                            "Do not modify files, run commands, use plan bookkeeping, or speculate. "
                            "Return a concise report naming relevant files and explaining the evidence."
                        ),
                        priority=ContextPriority.HIGH,
                    )

                    result = loop.run(
                        user_task=task,
                        workspace_directory=str(self._workspace_root),
                        stop_event=getattr(self._process_manager, "stop_event", None),
                    )
                    metrics = loop.get_metrics()
                finally:
                    loop.close()

        except Exception as exc:
            return self._error("explore_failed", f"Read-only exploration failed: {exc}")

        report = str(getattr(result, "response", "") or "").strip()
        if not report:
            return self._error(
                "empty_report",
                "The exploration subagent returned no report.",
            )

        report = report[:max_report_chars]
        return ToolResult(
            success=bool(metrics.get("completed", False)),
            name=self.name,
            content={
                "query": query.strip(),
                "path": focus,
                "report": report,
                "completed": bool(metrics.get("completed", False)),
                "stop_reason": metrics.get("stop_reason"),
                "iterations": metrics.get("iterations", 0),
                "tool_successes": metrics.get("tool_successes", 0),
                "tool_failures": metrics.get("tool_failures", 0),
            },
            metadata={
                "read_only": True,
                "subagent": True,
            },
            summary=(
                f"Read-only explorer returned a {len(report)}-character report "
                f"after {metrics.get('iterations', 0)} iteration(s)."
            ),
        )

    def _build_config(self, max_iterations: int) -> dict[str, Any]:
        base = {
            "max_agent_iterations": max_iterations,
            "guard_level": "strict",
            "memory": {},
            "context": {
                "safe_margin": 0,
                "max_prompt_tokens": 12000,
                "recent_event_limit": 40,
                "compaction_enabled": True,
                "compaction_target_tokens": 3000,
                "compaction_trigger_ratio": 0.78,
            },
            "retrieval": {"top_k": 2},
            "security": {
                "allowed_tools": [
                    "read_file",
                    "grep",
                    "glob",
                    "list_dir",
                ],
                "workspace_only": True,
                "allow_background": False,
                "allow_network_tools": False,
            },
            "experience": {"enabled": False},
        }

        parent = getattr(self._llm_provider, "llm_config", {}) or {}
        generation = getattr(self._llm_provider, "generation_config", {}) or {}
        base["llm"] = {
            "provider": getattr(self._llm_provider, "provider_name", "ollama"),
            "provider_config": {
                "model_name": parent.get("model_name", ""),
                "generation_config": dict(generation),
            },
        }
        allowed = self._runtime_config.get("security", {}).get("allowed_tools")
        if allowed is not None:
            base["security"]["allowed_tools"] = [
                name for name in base["security"]["allowed_tools"] if name in allowed
            ]
        base["subagents"] = {"enabled": False}
        return base

    def _error(self, error_type: str, message: str) -> ToolResult:
        return ToolResult(
            success=False,
            name=self.name,
            content={
                "success": False,
                "error": {"type": error_type, "message": message},
            },
            metadata={"read_only": True, "subagent": True},
        )
