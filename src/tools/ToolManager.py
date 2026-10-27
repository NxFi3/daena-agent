from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.models.ToolCall import ToolCall
from src.models.ToolResult import ToolResult
from src.agent.subagents import SubagentManager
from src.skills import SkillRegistry
from src.security.securityService import SecurityService
from src.tools.ToolDispatcher import ToolDispatcher
from src.tools.ToolRegistry import ToolRegistry
from src.tools.Tool import Tool
from src.tools.builtin.command_exec.process_manager import ProcessManager
from src.utils.logger import get_logger


class ToolManager:
    """Own tool discovery, normalization, policy checks and execution."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.logger = get_logger("[TOOLMANAGER]")

        self.config = {} if config is None else config
        if not isinstance(self.config, dict):
            raise ValueError("config must be an object")
        self.toolregistry = ToolRegistry()
        self.dispatcher = ToolDispatcher(self.toolregistry)
        self.security = SecurityService(self.config.get("security"))
        self._definitions: list[dict] | None = None
        self.process_manager = ProcessManager()
        self.skills = SkillRegistry(self.config)
        self.subagents = SubagentManager(self.config)
        self.stop_event = None
        self._workspace_root: Path | None = None
        self._llm_provider = None
        self._context_stm = None
        self._session_id_provider = None
        self._active_tool_names: list[str] = []
        self._last_tool_search_query = ""
        self.cleanup_errors: list[str] = []

    def set_llm_provider(self, llm_provider) -> None:
        self._llm_provider = llm_provider
        self._bind_runtime_services()

    def set_context_store(self, stm, session_id_provider) -> None:
        self._context_stm = stm
        self._session_id_provider = session_id_provider
        self._bind_runtime_services()

    def set_workspace(self, directory: str) -> None:
        self._workspace_root = Path(directory).expanduser().resolve()
        self.subagents.set_workspace(directory)
        self.skills.set_workspace(directory)
        self.security.set_workspace(directory)

        # Plan state is task-scoped workspace state. Keep the plan tool on the
        # same filesystem coordinate system as the other workspace tools.
        plan_tool = self.get_tool("plan")
        set_workspace = getattr(plan_tool, "set_workspace", None)
        if callable(set_workspace):
            set_workspace(directory)

    def finish_run(self) -> None:
        """Release task resources while preserving conversation-owned browser state."""
        try:
            self.process_manager.close()
        finally:
            self.subagents.close()

    def close(self) -> None:
        """Release all task and conversation resources."""
        self.cleanup_errors.clear()
        cleanup = [
            ("processes", lambda: self.process_manager.close(discard_output=True)),
            ("subagents", self.subagents.close),
        ]
        seen = set()
        for name, tool in self.toolregistry.tools.items():
            closer = getattr(tool, "close", None)
            if callable(closer) and id(tool) not in seen:
                cleanup.append((name, closer))
                seen.add(id(tool))
        for name, closer in cleanup:
            try:
                closer()
            except Exception as exc:
                message = f"Cleanup failed for {name}: {type(exc).__name__}: {exc}"
                self.cleanup_errors.append(message)
                self.logger.error(message)

    def register_tool(self, tool: Tool) -> str:
        """Install a trusted host adapter before runs; policy never auto-allows it."""
        if not self.toolregistry.tools:
            self.toolregistry.discover()
        name = self.toolregistry.register(tool)
        self._definitions = None
        self._bind_runtime_services()
        return name

    def approve_background_command(
        self,
        command: list[str],
        workdir: str | None = None,
    ) -> None:
        """Grant an exact background command for the current session."""
        self.security.approve_background_command(
            command=command,
            workdir=workdir,
        )

    def revoke_background_command(
        self,
        command: list[str],
        workdir: str | None = None,
    ) -> None:
        """Revoke an exact background command approval."""
        self.security.revoke_background_command(
            command=command,
            workdir=workdir,
        )

    def clear_background_approvals(self) -> None:
        """Clear all session-scoped background command approvals."""
        self.security.clear_background_approvals()

    def _bind_runtime_services(self) -> None:
        """Inject runtime-owned services into discovered tool instances."""
        self.process_manager.stop_event = self.stop_event
        for tool in self.toolregistry.tools.values():
            config_setter = getattr(tool, "set_runtime_config", None)
            if callable(config_setter):
                config_setter(self.config)
            if hasattr(tool, "skill_registry"):
                tool.skill_registry = self.skills
            if hasattr(tool, "subagent_manager"):
                tool.subagent_manager = self.subagents

            process_setter = getattr(tool, "set_process_manager", None)
            if callable(process_setter):
                process_setter(self.process_manager)
            elif hasattr(tool, "process_manager"):
                tool.process_manager = self.process_manager
            workspace_setter = getattr(tool, "set_workspace", None)
            if callable(workspace_setter) and self._workspace_root is not None:
                workspace_setter(self._workspace_root)

            llm_setter = getattr(tool, "set_llm_provider", None)
            if callable(llm_setter) and self._llm_provider is not None:
                llm_setter(self._llm_provider)

            context_setter = getattr(tool, "set_context_store", None)
            if callable(context_setter) and self._context_stm is not None:
                context_setter(self._context_stm, self._session_id_provider)

            stop_setter = getattr(tool, "set_stop_event", None)
            if callable(stop_setter):
                stop_setter(self.stop_event)

            search_setter = getattr(tool, "set_searcher", None)
            if callable(search_setter):
                search_setter(self._search_and_activate_tools)

    def get_tools(self) -> list[dict]:
        if self._definitions is None:
            if not self.toolregistry.tools:
                self.toolregistry.discover()
            self._bind_runtime_services()
            self._definitions = self.toolregistry.get_definitions()
        else:
            self._bind_runtime_services()

        self._definitions = self.toolregistry.get_definitions()
        allowed = set(self.security.policy.allowed_tools)
        network_allowed = bool(self.security.policy.allow_network_tools)

        visible: list[dict] = []
        for definition in self._definitions:
            function = (
                definition.get("function", {}) if isinstance(definition, dict) else {}
            )
            name = str(function.get("name", "")).strip().lower()
            if not name or name not in allowed:
                continue
            if (
                name in {"web_search", "web_fetch", "browser_use"}
                or getattr(self.toolregistry.get(name), "requires_network", False)
            ) and not network_allowed:
                continue
            if name in {"skill_list", "skill_read"} and not self.skills.enabled:
                continue
            if (
                name == "task_finish"
                and (self.config.get("runtime") or {}).get(
                    "completion_protocol", "implicit"
                )
                != "explicit"
            ):
                continue
            if name == "subagent" and not self.subagents.enabled:
                continue
            visible.append(definition)

        return visible

    CORE_MODEL_TOOLS = frozenset(
        {
            "tool_search",
            "read_file",
            "apply_patch",
            "command_exec",
            "process_poll",
            "process_write",
            "process_stop",
            "make_directory",
            "skill_list",
            "skill_read",
            "subagent",
            "task_finish",
        }
    )

    @staticmethod
    def _search_tokens(value: str) -> list[str]:
        expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(value or ""))
        tokens = re.findall(r"[a-z0-9]+", expanded.casefold())
        aliases = {
            "editing": "edit",
            "edited": "edit",
            "writing": "write",
            "written": "write",
            "files": "file",
            "directories": "directory",
            "searching": "search",
            "finding": "find",
            "browsing": "browse",
            "desktop": "computer",
            "gui": "computer",
            "screen": "computer",
            "history": "context",
            "memory": "context",
        }
        return [aliases.get(token, token) for token in tokens]

    def _search_and_activate_tools(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Find allowed capabilities and expose the best matches next turn."""
        try:
            limit = max(1, min(8, int(top_k)))
        except (TypeError, ValueError):
            limit = 5

        query_tokens = set(self._search_tokens(query))
        stop = {
            "a",
            "an",
            "and",
            "for",
            "from",
            "i",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "please",
            "the",
            "this",
            "to",
            "tool",
            "tools",
            "use",
            "want",
            "with",
            "you",
            "your",
            "need",
        }
        query_tokens.difference_update(stop)
        if not query_tokens:
            return {
                "matches": [],
                "active_tools": list(self._active_tool_names),
            }

        synonyms = {
            "fix": {"edit", "patch", "write"},
            "edit": {"fix", "patch", "replace", "file"},
            "write": {"create", "file", "save"},
            "create": {"write", "file", "directory"},
            "search": {"find", "grep", "glob", "context"},
            "find": {"search", "grep", "glob"},
            "inspect": {"read", "list", "explore"},
            "codebase": {"explore", "grep", "glob", "list"},
            "repository": {"explore", "grep", "glob", "list"},
            "computer": {"desktop", "gui", "screen", "click", "type"},
            "browser": {"computer", "web"},
            "context": {"history", "memory", "previous", "earlier"},
            "previous": {"context", "history", "memory"},
            "earlier": {"context", "history", "memory"},
        }
        expanded = set(query_tokens)
        for token in tuple(query_tokens):
            expanded.update(synonyms.get(token, set()))

        definitions = self.get_tools()
        scored: list[tuple[float, dict[str, Any]]] = []
        for definition in definitions:
            function = (
                definition.get("function", {}) if isinstance(definition, dict) else {}
            )
            name = str(function.get("name", "")).strip().lower()
            if not name or name == "tool_search":
                continue

            name_tokens = set(self._search_tokens(name.replace("_", " ")))
            description_tokens = set(
                self._search_tokens(function.get("description", ""))
            )
            parameters = function.get("parameters", {})
            schema_tokens = set(self._search_tokens(str(parameters)))

            score = 0.0
            score += 6.0 * len(expanded & name_tokens)
            score += 2.0 * len(expanded & description_tokens)
            score += 0.5 * len(expanded & schema_tokens)
            normalized_query = " ".join(self._search_tokens(query))
            normalized_name = " ".join(self._search_tokens(name.replace("_", " ")))
            if normalized_name and normalized_name in normalized_query:
                score += 10.0

            if score > 0:
                scored.append((score, definition))

        query_words = " " + " ".join(self._search_tokens(query)) + " "

        def explicit_name(definition):
            name = str(definition.get("function", {}).get("name", ""))
            return (
                " " + " ".join(self._search_tokens(name.replace("_", " "))) + " "
                in query_words
            )

        scored.sort(
            key=lambda item: (
                not explicit_name(item[1]),
                -item[0],
                str((item[1].get("function") or {}).get("name", "")),
            )
        )
        selected = scored[:limit]

        matched_names = [
            str((definition.get("function") or {}).get("name", "")).strip().lower()
            for _, definition in selected
        ]
        tool_config = self.config.get("tools") or {}
        try:
            max_active = int(tool_config.get("max_active_search_tools", 8))
        except (AttributeError, TypeError, ValueError):
            max_active = 8
        max_active = max(1, min(12, max_active))

        previous = [
            name for name in self._active_tool_names if name not in matched_names
        ]
        self._active_tool_names = (matched_names + previous)[:max_active]
        self._last_tool_search_query = str(query)

        matches = [
            {
                "name": str((definition.get("function") or {}).get("name", "")),
                "description": str(
                    (definition.get("function") or {}).get("description", "")
                )[:260],
                "relevance": round(score, 3),
            }
            for score, definition in selected
        ]
        return {
            "matches": matches,
            "active_tools": list(self._active_tool_names),
        }

    def reset_tool_selection(self) -> None:
        self._active_tool_names = []
        self._last_tool_search_query = ""

    def get_model_tools(
        self, *, schema_token_budget=None, chars_per_token=4.0, required_names=()
    ) -> list[dict]:
        """Expose core tools plus capabilities selected through tool_search."""
        visible = self.get_tools()
        by_name = {
            str((definition.get("function") or {}).get("name", ""))
            .strip()
            .lower(): definition
            for definition in visible
            if isinstance(definition, dict)
            and isinstance(definition.get("function"), dict)
        }

        search_definition = by_name.get("tool_search")
        if search_definition is None:
            return visible

        always_names = set(self.CORE_MODEL_TOOLS)
        for name in by_name:
            tool = self.toolregistry.get(name)
            if str(getattr(tool, "model_exposure", "")).strip().lower() == "always":
                always_names.add(name)

        ordered: list[dict] = []
        emitted: set[str] = set()

        def add(name: str) -> None:
            if name in emitted:
                return
            definition = by_name.get(name)
            if definition is None:
                return
            emitted.add(name)
            ordered.append(definition)

        add("tool_search")
        for definition in visible:
            name = (
                str((definition.get("function") or {}).get("name", "")).strip().lower()
            )
            if name in always_names and name != "tool_search":
                add(name)
        for name in self._active_tool_names:
            add(name)

        # Advertise every allowed capability even when its full schema is lazy.
        # This keeps small contexts viable without hiding browser/desktop tools.
        from copy import deepcopy

        catalog = [name for name in by_name if name not in emitted]
        if catalog:
            ordered = deepcopy(ordered)
            ordered[0]["function"]["description"] += (
                " Available additional tools: "
                + ", ".join(sorted(catalog))
                + ". Search a tool name to load its complete argument schema."
            )
        if schema_token_budget is not None:
            import json
            import math
            from copy import deepcopy

            # Keep complete schemas. Less urgent capabilities remain discoverable;
            # schema packing never rewrites a tool's argument contract.
            required = {"tool_search", *required_names}
            required.update(
                name
                for name in by_name
                if str(
                    getattr(self.toolregistry.get(name), "model_exposure", "")
                ).lower()
                == "always"
            )

            def render(names):
                result = [deepcopy(by_name[name]) for name in names]
                catalog = sorted(set(by_name) - set(names))
                search = next(
                    (d for d in result if d["function"]["name"] == "tool_search"), None
                )
                if search and catalog:
                    search["function"]["description"] += (
                        " Available additional tools: "
                        + ", ".join(catalog)
                        + ". Search a tool name to load its complete argument schema."
                    )
                return result

            def fits(names):
                return (
                    math.ceil(
                        len(json.dumps(render(names), ensure_ascii=False, default=str))
                        / chars_per_token
                    )
                    <= schema_token_budget
                )

            selected = [name for name in by_name if name in required]
            priority = list(self._active_tool_names) + [
                d["function"]["name"] for d in ordered
            ]
            for name in priority:
                if name in by_name and name not in selected and fits([*selected, name]):
                    selected.append(name)
            return render(selected)
        return ordered

    def _find_tool(self, name: str):
        return self.toolregistry.get(str(name).strip().lower())

    def get_tool(self, name: str):
        if not self.toolregistry.tools:
            self.toolregistry.discover()
        self._bind_runtime_services()
        return self._find_tool(name)

    def _resolve_execution_path(self, value: str) -> Path:
        """Resolve a path against the active workspace and enforce policy."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Path cannot be empty.")

        raw = Path(value).expanduser()

        if self.security.policy.workspace_only:
            return self.security.sandbox.resolve(raw)

        workspace_root = self.security.sandbox.root
        if raw.is_absolute():
            return raw.resolve(strict=False)
        if workspace_root is not None:
            return (workspace_root / raw).resolve(strict=False)
        return raw.resolve(strict=False)

    def _normalize_workspace_args(self, toolcall: ToolCall) -> ToolCall:
        """Resolve tool paths relative to the active workspace."""
        args = dict(toolcall.args or {})
        name = toolcall.name

        if name == "make_directory":
            args["paths"] = [
                str(self._resolve_execution_path(p)) for p in args["paths"]
            ]

        elif name in {"read_file", "write_file", "edit_file"}:
            args["file_path"] = str(self._resolve_execution_path(args["file_path"]))

        elif name == "command_exec":
            args["workdir"] = str(
                self._resolve_execution_path(args.get("workdir") or ".")
            )

        elif name == "apply_patch":
            if "edits" in args:
                args["edits"] = [
                    {**edit, "path": str(self._resolve_execution_path(edit["path"]))}
                    for edit in args["edits"]
                ]
                toolcall.args = args
                return toolcall
            patch = args["patch"]
            lines = patch.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            normalized: list[str] = []

            for line in lines:
                match = re.match(
                    r"^(\*\*\*\s+(?:Add|Update|Delete) File:\s*)(.+?)\s*$",
                    line,
                )
                if match:
                    path = match.group(2).strip()
                    line = match.group(1) + str(self._resolve_execution_path(path))
                normalized.append(line)

            args["patch"] = "\n".join(normalized)

        toolcall.args = args
        return toolcall

    @staticmethod
    def _recovery_hint(result: ToolResult) -> str:
        content = result.content if isinstance(result.content, dict) else {}
        error = content.get("error")
        error_type = ""
        if isinstance(error, dict):
            error_type = str(error.get("type", "")).strip().lower()
        elif error:
            error_type = str(error).strip().lower()

        hints = {
            "workspace_boundary": (
                "Keep every path/workdir inside the active workspace. "
                "Use relative paths from the workspace root."
            ),
            "command_not_found": (
                "The executable was not found. Check the project setup first; "
                "for Node projects prefer the package script (for example npm test) "
                "or npx when appropriate."
            ),
            "resource_in_use": (
                "The command reached an existing resource such as a listening address. "
                "Inspect the active process state or choose the intended existing process "
                "before starting another instance."
            ),
            "invalid_argument": (
                "Reissue the call with schema-valid arguments. Optional arguments "
                "can be omitted instead of guessing."
            ),
            "tool_argument_error": (
                "The tool rejected the argument contract at execution time. "
                "Re-read the documented parameters and provide all required "
                "arguments with the correct types."
            ),
            "invalid_tool_call": (
                "Reissue the call with the documented tool name and argument schema; "
                "do not repeat the malformed call unchanged."
            ),
            "tool_not_found": (
                "Use one of the currently available tools instead of retrying the "
                "missing tool."
            ),
            "security_denied": (
                "The runtime denied this action. Respect the reported security rule "
                "and choose a compliant alternative."
            ),
            "tool_execution_error": (
                "Inspect the concrete execution error and change the action before "
                "retrying; do not blindly repeat the same call."
            ),
            "unexpected_error": (
                "The tool failed unexpectedly. Inspect the error and try a different "
                "action or corrected arguments."
            ),
        }
        return hints.get(
            error_type,
            "Use the concrete tool result to decide the next action; avoid repeating "
            "an unchanged failed call.",
        )

    def execute(self, tool_calls: list[ToolCall]) -> dict:
        self.get_tools()
        calls: list[ToolCall] = []
        results: list[ToolResult] = []

        if not tool_calls:
            return {"calls": [], "results": []}

        for incoming_call in tool_calls:
            if not isinstance(incoming_call, ToolCall):
                invalid_call = ToolCall(name="", valid=False)
                calls.append(invalid_call)
                results.append(
                    ToolResult(
                        success=False,
                        name="",
                        content={
                            "success": False,
                            "error": {
                                "type": "invalid_tool_call",
                                "message": "ToolManager received a non-ToolCall object.",
                            },
                        },
                        metadata={},
                    )
                )
                continue

            toolcall = incoming_call

            if not toolcall.valid:
                message = str(toolcall.validation_error).strip() or "Invalid tool call."
                calls.append(toolcall)
                results.append(
                    ToolResult(
                        success=False,
                        name=toolcall.name,
                        content={
                            "success": False,
                            "error": {
                                "type": "invalid_tool_call",
                                "message": message,
                            },
                        },
                        metadata={
                            "tool_call_id": toolcall.id,
                            "normalization_notes": list(
                                toolcall.normalization_notes or []
                            ),
                            "recovery_hint": (
                                "Reissue the call using the documented schema. "
                                "Do not repeat an invalid call unchanged."
                            ),
                        },
                        summary=message,
                    )
                )
                continue

            tool = self._find_tool(toolcall.name)
            if tool is None:
                calls.append(toolcall)
                results.append(
                    ToolResult(
                        success=False,
                        name=toolcall.name,
                        content={
                            "success": False,
                            "error": {
                                "type": "tool_not_found",
                                "message": f"Tool '{toolcall.name}' was not found.",
                            },
                        },
                        metadata={"tool_call_id": toolcall.id},
                    )
                )
                continue

            # A host may call execute() with a canonical ToolCall directly.
            # The valid flag is not a substitute for the registered contract.
            schema_error = self.dispatcher._schema_validation_error(
                getattr(tool, "parameters", {}) or {}, toolcall.args
            )
            if schema_error:
                toolcall.valid = False
                toolcall.validation_error = schema_error
                calls.append(toolcall)
                results.append(
                    ToolResult(
                        success=False,
                        name=toolcall.name,
                        content={
                            "error": {
                                "type": "invalid_tool_call",
                                "message": schema_error,
                            }
                        },
                        metadata={"tool_call_id": toolcall.id},
                        summary=schema_error,
                    )
                )
                continue

            try:
                toolcall = self.security.check(toolcall)
                if (
                    getattr(tool, "requires_network", False)
                    and not self.security.policy.allow_network_tools
                ):
                    toolcall.approved = False
                    toolcall.security_reason = "This tool requires network access, which is disabled by host configuration."
                    toolcall.security_rule = "network_disabled"
                if (
                    toolcall.name in {"skill_list", "skill_read"}
                    and not self.skills.enabled
                ) or (toolcall.name == "subagent" and not self.subagents.enabled):
                    toolcall.approved = False
                    toolcall.security_reason = (
                        "This feature is disabled by host configuration."
                    )
                    toolcall.security_rule = "feature_disabled"
                if self.stop_event is not None and self.stop_event.is_set():
                    toolcall.approved = False
                    toolcall.security_reason = "The current task has been cancelled."
                    toolcall.security_rule = "task_cancelled"
            except Exception as exc:
                self.logger.error(
                    "Security check failed for '%s': %s", toolcall.name, exc
                )
                toolcall.approved = False
                toolcall.security_reason = f"Security check failed: {exc}"
                toolcall.security_rule = "security_failure"

            calls.append(toolcall)

            if not toolcall.approved:
                results.append(
                    ToolResult(
                        success=False,
                        name=toolcall.name,
                        content={
                            "success": False,
                            "error": {
                                "type": "security_denied",
                                "message": toolcall.security_reason
                                or "Tool execution denied by security policy.",
                            },
                        },
                        metadata={
                            "tool_call_id": toolcall.id,
                            "security_rule": toolcall.security_rule,
                            "recovery_hint": (
                                "Respect the reported security rule and choose a "
                                "different compliant action."
                            ),
                        },
                    )
                )
                continue

            try:
                toolcall = self._normalize_workspace_args(toolcall)
                calls[-1] = toolcall
                result = tool.execute(**toolcall.args)
                if not isinstance(result, ToolResult):
                    result = ToolResult(
                        success=False,
                        name=toolcall.name,
                        content={
                            "success": False,
                            "error": {
                                "type": "invalid_tool_result",
                                "message": "Tool returned an invalid result.",
                            },
                        },
                        metadata={"tool_call_id": toolcall.id},
                    )

                if not isinstance(result.metadata, dict):
                    result.metadata = {}

                result.metadata.setdefault("tool_call_id", toolcall.id)

                if not result.success:
                    result.metadata.setdefault(
                        "recovery_hint",
                        self._recovery_hint(result),
                    )

                results.append(result)

            except (TypeError, ValueError) as exc:
                self.logger.error(
                    "Tool '%s' rejected its execution arguments: %s",
                    toolcall.name,
                    exc,
                )
                results.append(
                    ToolResult(
                        success=False,
                        name=toolcall.name,
                        content={
                            "success": False,
                            "error": {
                                "type": "tool_argument_error",
                                "message": (
                                    f"Tool '{toolcall.name}' rejected the provided "
                                    f"arguments: {exc}"
                                ),
                            },
                        },
                        metadata={
                            "tool_call_id": toolcall.id,
                            "recovery_hint": (
                                "Treat this as an argument/contract error. "
                                "Re-read the tool schema and provide the required "
                                "arguments instead of retrying unchanged."
                            ),
                        },
                    )
                )
            except Exception as exc:
                self.logger.error(
                    "Tool '%s' failed: %s",
                    toolcall.name,
                    exc,
                )
                results.append(
                    ToolResult(
                        success=False,
                        name=toolcall.name,
                        content={
                            "success": False,
                            "error": {
                                "type": "tool_execution_error",
                                "message": (
                                    f"Unexpected error while using tool "
                                    f"{type(exc).__name__}: {exc}"
                                ),
                            },
                        },
                        metadata={
                            "tool_call_id": toolcall.id,
                            "recovery_hint": (
                                "Inspect the concrete runtime error and choose "
                                "a corrected action or different tool. Do not "
                                "blindly repeat the same call."
                            ),
                        },
                    )
                )

        return {"calls": calls, "results": results}
