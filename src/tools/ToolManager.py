from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.models.ToolCall import ToolCall
from src.models.ToolResult import ToolResult
from src.security.securityService import SecurityService
from src.tools.ToolDispatcher import ToolDispatcher
from src.tools.ToolRegistry import ToolRegistry
from src.tools.builtin.command_exec.process_manager import ProcessManager
from src.utils.logger import get_logger


class ToolManager:
    """Own tool discovery, normalization, policy checks and execution."""

    # Keep the small, high-frequency coding/runtime primitives available on every
    # model turn. Tool Search is for expanding capabilities, not unlocking basics.
    CORE_TOOL_NAMES = frozenset({
        "command_exec",
        "read_file",
        "write_file",
        "edit_file",
        "apply_patch",
        "process_poll",
        "process_write",
        "process_stop",
        "computer_use",
    })

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.logger = get_logger("[TOOLMANAGER]")

        self.config = config or {}
        self.toolregistry = ToolRegistry()
        self.dispatcher = ToolDispatcher(self.toolregistry)
        self.security = SecurityService(self.config.get("security"))
        self._definitions: list[dict] | None = None
        self.process_manager = ProcessManager()
        self._workspace_root: Path | None = None
        self._llm_provider = None
        self._context_stm = None
        self._session_id_provider = None
        self._active_tool_names: list[str] = []
        self._last_tool_search_query = ""

    def set_llm_provider(self, llm_provider) -> None:
        self._llm_provider = llm_provider
        self._bind_runtime_services()

    def set_context_store(self, stm, session_id_provider) -> None:
        self._context_stm = stm
        self._session_id_provider = session_id_provider
        self._bind_runtime_services()

    def set_workspace(self, directory: str) -> None:
        self._workspace_root = Path(directory).expanduser().resolve()
        self.security.set_workspace(directory)

        # Plan state is task-scoped workspace state. Keep the plan tool on the
        # same filesystem coordinate system as the other workspace tools.
        plan_tool = self.get_tool("plan")
        set_workspace = getattr(plan_tool, "set_workspace", None)
        if callable(set_workspace):
            set_workspace(directory)

    def close(self) -> None:
        """Release runtime-owned resources, including managed processes."""
        self.process_manager.close()

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
        for tool in self.toolregistry.tools.values():
            setter = getattr(tool, "set_process_manager", None)
            if callable(setter):
                setter(self.process_manager)
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

            search_setter = getattr(tool, "set_searcher", None)
            if callable(search_setter):
                search_setter(self._search_and_activate_tools)

    def get_tools(self) -> list[dict]:
        if self._definitions is None:
            self.toolregistry.discover()
            self._bind_runtime_services()
            self._definitions = self.toolregistry.get_definitions()
        else:
            self._bind_runtime_services()

        allowed = set(self.security.policy.allowed_tools)
        network_allowed = bool(self.security.policy.allow_network_tools)
        tool_config = self.config.get("tools") or {}
        disabled = {
            str(name).strip().lower()
            for name in tool_config.get("disabled", [])
            if str(name).strip()
        } if isinstance(tool_config, dict) else set()
        guard_level = str(self.config.get("guard_level", "strict")).strip().lower()
        if guard_level not in {"strict", "light"}:
            guard_level = "strict"

        visible: list[dict] = []
        for definition in self._definitions:
            function = definition.get("function", {}) if isinstance(definition, dict) else {}
            name = str(function.get("name", "")).strip().lower()
            if not name or name not in allowed or name in disabled:
                continue
            if guard_level == "strict" and name == "plan":
                continue
            if name in {"web_search", "web_fetch"} and not network_allowed:
                continue
            visible.append(definition)

        return visible

    @staticmethod
    def _search_tokens(value: str) -> list[str]:
        """Normalize names, descriptions and queries, including common verb forms."""
        expanded = str(value or "")
        # Preserve common programming-language/tool identifiers that punctuation
        # tokenization would otherwise destroy (e.g. C++ -> cpp, g++ -> gxx).
        expanded = re.sub(r"(?i)\bc\s*\+\s*\+", " cpp ", expanded)
        expanded = re.sub(r"(?i)\bg\s*\+\s*\+", " gxx ", expanded)
        expanded = re.sub(r"(?i)\bc\s*#", " csharp ", expanded)
        expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", expanded)
        raw_tokens = re.findall(r"[a-z0-9]+", expanded.casefold())
        irregular = {
            "wrote": "write", "written": "write", "writes": "write",
            "ran": "run", "running": "run", "built": "build",
            "compiled": "compile", "compiling": "compile",
            "found": "find", "sent": "send", "took": "take",
            "bought": "buy", "saw": "see", "gave": "give",
        }
        silent_e = {"writ": "write", "creat": "create", "sav": "save",
                    "mak": "make", "us": "use", "tak": "take",
                    "giv": "give", "verifi": "verify"}
        normalized: list[str] = []
        for raw in raw_tokens:
            token = irregular.get(raw, raw)
            if token == raw and len(token) > 5 and token.endswith("ing"):
                stem = token[:-3]
                if len(stem) > 2 and stem[-1] == stem[-2]:
                    stem = stem[:-1]
                token = silent_e.get(stem, stem)
            elif token == raw and len(token) > 4 and token.endswith("ed"):
                stem = token[:-2]
                if len(stem) > 2 and stem[-1] == stem[-2]:
                    stem = stem[:-1]
                token = silent_e.get(stem, stem)
            elif token == raw and len(token) > 4 and token.endswith("ies"):
                token = token[:-3] + "y"
            elif (
                token == raw and len(token) > 3 and token.endswith("s")
                and not token.endswith(("ss", "ous", "ics"))
            ):
                token = token[:-1]
            normalized.append(token)
        return normalized

    def _search_and_activate_tools(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Rank allowed tool metadata, then expose only the best matching schemas."""
        try:
            limit = max(1, min(8, int(top_k)))
        except (TypeError, ValueError):
            limit = 5

        query_tokens = [
            token for token in self._search_tokens(query)
            if token not in {
                "a", "an", "and", "are", "as", "at", "be", "by", "can",
                "do", "for", "from", "i", "in", "is", "it", "me", "of",
                "on", "or", "please", "the", "this", "to", "use", "want",
                "what", "when", "where", "which", "with", "you", "your",
                "tool", "tools", "need", "help", "task", "make",
            }
        ]
        if not query_tokens:
            return {"matches": [], "active_tools": list(self._active_tool_names)}

        # Small lexical synonym groups improve natural-language recall without
        # creating an intent router or hard-coding task-to-tool sequences.
        synonyms = {
            "create": ("write", "generate", "save"),
            "fix": ("edit", "patch", "repair", "debug", "correct"),
            "repair": ("fix", "edit", "patch"),
            "test": ("pytest", "check", "verify", "validate"),
            "run": ("execute", "command", "script"),
            "execute": ("run", "command", "script"),
            "compile": ("build", "compiler", "gcc", "gxx", "clang", "cmake",
                        "link", "command", "execute", "run", "executable", "binary"),
            "compiler": ("compile", "build", "gcc", "gxx", "clang", "cmake", "link"),
            "build": ("compile", "compiler", "cmake", "make", "link", "package"),
            "cpp": ("cxx", "code", "source"),
            "cxx": ("cpp", "code", "source"),
            "read": ("open", "view", "inspect"),
            "inspect": ("read", "view", "list", "explore"),
            "search": ("find", "locate", "grep"),
            "find": ("search", "locate", "grep"),
            "browse": ("web", "search", "fetch"),
            "website": ("web", "search", "fetch"),
            "url": ("web", "fetch", "retrieve"),
            "dataset": ("data", "csv", "collect", "parse"),
            "collect": ("fetch", "retrieve", "data"),
        }
        # Original words carry full weight; synonyms are soft recall hints only.
        query_weights: dict[str, float] = {}
        for token in query_tokens:
            query_weights[token] = 1.0
            for synonym in synonyms.get(token, ()):
                if synonym not in query_weights:
                    query_weights[synonym] = 0.35

        catalog = [
            definition for definition in self.get_tools()
            if isinstance(definition, dict)
            and isinstance(definition.get("function"), dict)
            and str(definition["function"].get("name", "")).strip().lower() != "tool_search"
        ]
        if not catalog:
            return {"matches": [], "active_tools": list(self._active_tool_names)}

        docs: list[dict[str, Any]] = []
        term_document_frequency: Counter[str] = Counter()
        for definition in catalog:
            function = definition["function"]
            name = str(function.get("name", "")).strip().lower()
            description = str(function.get("description", ""))
            parameters = function.get("parameters", {})
            try:
                schema_text = json.dumps(parameters, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                schema_text = str(parameters)
            name_tokens = self._search_tokens(name.replace("_", " "))
            description_tokens = self._search_tokens(description)
            schema_tokens = self._search_tokens(schema_text)
            weighted_tokens = (
                name_tokens * 3 + description_tokens + schema_tokens
            )
            counts = Counter(weighted_tokens)
            for token in set(counts):
                term_document_frequency[token] += 1
            docs.append({
                "name": name,
                "description": description,
                "name_tokens": Counter(name_tokens),
                "description_tokens": Counter(description_tokens),
                "counts": counts,
                "length": max(1, len(weighted_tokens)),
            })

        doc_count = len(docs)
        average_length = sum(doc["length"] for doc in docs) / max(1, doc_count)

        # An unrelated or unsupported request should not activate arbitrary tools
        # just because a long description shares one generic word such as "inspect".
        catalog_vocabulary = set().union(*(set(doc["counts"]) for doc in docs))
        original_unique_tokens = set(query_tokens)
        # A query term is supported when either it appears directly in metadata or
        # a curated synonym does. This lets "compile" retrieve shell/build tools
        # without weakening the no-unrelated-results guard for unknown requests.
        covered_terms = {
            token for token in original_unique_tokens
            if token in catalog_vocabulary
            or any(synonym in catalog_vocabulary for synonym in synonyms.get(token, ()))
        }
        coverage = len(covered_terms) / max(1, len(original_unique_tokens))
        if coverage < 0.40:
            self.logger.debug(
                "Tool search found too little catalog evidence for query %r (coverage=%.2f)",
                query, coverage,
            )
            return {
                "matches": [],
                "active_tools": list(self._active_tool_names),
                "coverage": round(coverage, 3),
            }

        ranked: list[tuple[float, dict[str, Any]]] = []
        normalized_query = " ".join(query_tokens)
        original_unique_tokens = set(query_tokens)
        for doc in docs:
            score = 0.0
            for token, query_weight in query_weights.items():
                frequency = doc["counts"].get(token, 0)
                if not frequency:
                    continue
                inverse_document_frequency = math.log(
                    1.0 + (doc_count - term_document_frequency[token] + 0.5)
                    / (term_document_frequency[token] + 0.5)
                )
                denominator = frequency + 1.2 * (
                    1.0 - 0.75 + 0.75 * doc["length"] / average_length
                )
                score += query_weight * inverse_document_frequency * (frequency * 2.2 / denominator)
                if token in original_unique_tokens and token in doc["name_tokens"]:
                    score += inverse_document_frequency * 2.0
                elif token in doc["description_tokens"]:
                    score += query_weight * inverse_document_frequency * 0.15

            normalized_name = " ".join(self._search_tokens(doc["name"].replace("_", " ")))
            if normalized_name and normalized_name in normalized_query:
                score += 4.0
            if score > 0:
                ranked.append((score, doc))

        ranked.sort(key=lambda item: (-item[0], item[1]["name"]))
        selected = ranked[:limit]
        if selected:
            self._active_tool_names = [doc["name"] for _, doc in selected]
            self._last_tool_search_query = query
            matches = [
                {
                    "name": doc["name"],
                    "description": doc["description"][:260],
                    "relevance": round(score, 3),
                }
                for score, doc in selected
            ]
        else:
            matches = []

        return {
            "matches": matches,
            "active_tools": list(self._active_tool_names),
            "coverage": round(coverage, 3),
        }

    def reset_tool_selection(self) -> None:
        """Start a fresh task with only the catalog search entrypoint exposed."""
        self._active_tool_names = []
        self._last_tool_search_query = ""

    def get_model_tools(self) -> list[dict]:
        """Return Tool Search, always-on core primitives, and selected extra schemas."""
        visible = self.get_tools()
        by_name = {
            str((definition.get("function") or {}).get("name", "")).strip().lower(): definition
            for definition in visible
            if isinstance(definition, dict) and isinstance(definition.get("function"), dict)
        }
        search_definition = by_name.get("tool_search")

        # If a custom policy/config disables discovery, preserve the old allowlisted
        # behavior instead of making Tool Search a mandatory dependency.
        if search_definition is None:
            return visible

        core = [
            definition for definition in visible
            if str((definition.get("function") or {}).get("name", "")).strip().lower()
            in self.CORE_TOOL_NAMES
        ]
        core_names = {
            str((definition.get("function") or {}).get("name", "")).strip().lower()
            for definition in core
        }
        selected = [
            definition for definition in visible
            if str((definition.get("function") or {}).get("name", "")).strip().lower()
            in set(self._active_tool_names) - core_names - {"tool_search"}
        ]
        return [search_definition, *core, *selected]

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

        if name in {"read_file", "write_file", "edit_file"}:
            args["file_path"] = str(
                self._resolve_execution_path(args["file_path"])
            )

        elif name == "command_exec":
            args["workdir"] = str(
                self._resolve_execution_path(args.get("workdir") or ".")
            )

        elif name == "apply_patch":
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
                    line = match.group(1) + str(
                        self._resolve_execution_path(path)
                    )
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

            disabled_tools = {
                str(name).strip().lower()
                for name in ((self.config.get("tools") or {}).get("disabled", []) or [])
                if str(name).strip()
            }
            guard_level = str(self.config.get("guard_level", "strict")).strip().lower()
            if toolcall.name in disabled_tools or (guard_level != "light" and toolcall.name == "plan"):
                reason = (
                    f"Tool '{toolcall.name}' is disabled by tools.disabled."
                    if toolcall.name in disabled_tools
                    else "The plan tool is hidden in strict guard_level. Use direct inspection and implementation instead."
                )
                calls.append(toolcall)
                results.append(ToolResult(
                    success=False,
                    name=toolcall.name,
                    content={"success": False, "error": {
                        "type": "tool_disabled",
                        "message": reason,
                    }},
                    metadata={"tool_call_id": toolcall.id, "recovery_hint": "Choose a currently available tool and continue the task."},
                    summary=reason,
                ))
                continue

            if not toolcall.valid:
                message = (
                    str(toolcall.validation_error).strip()
                    or "Invalid tool call."
                )
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

            try:
                toolcall = self.security.check(toolcall)
            except Exception as exc:
                self.logger.error("Security check failed for '%s': %s", toolcall.name, exc)
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
                                "message": toolcall.security_reason or "Tool execution denied by security policy.",
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

            except TypeError as exc:
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
