# src/agent/agentloop.py
from __future__ import annotations

import copy
import csv
import hashlib
import json
import queue
import re
import shlex
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from uuid import UUID, uuid4

from src.agent.agentstate import AgentState
from src.agent.completionreview import CompletionReviewer, CompletionVerdict
from src.agent.planstate import PlanState
from src.agent.planprogress import PlanProgressTracker
from src.context.contextservice import ContextService
from src.context.workingset import WorkingSet
from src.engine.LlmProviderManager import LlmProvider
from src.models.ContextEvent import (
    ContextEvent,
    ContextPriority,
    ContextRole,
    ContextType,
)
from src.models.LLMResult import LLMResult
from src.models.ToolResult import ToolResult
from src.memories.stm.STM import STM
from src.tools.ToolManager import ToolManager
from src.utils.logger import get_logger


class Loop:

    EMPTY_RESPONSE_THRESHOLD = 3

    # How many times a single iteration may be retried in place after a
    # generation failure (provider exception, e.g. Ollama's own tool-call
    # parser choking on malformed output) or a dispatch failure (the model
    # returned tool_calls our ToolDispatcher couldn't make sense of).
    # Without this, a single hiccup used to kill the entire run instantly,
    # regardless of how much progress had already been made.
    MAX_GENERATION_RETRIES = 3

    DEFAULT_MAX_ITERATIONS = 100

    DEFAULT_RECENT_CONTEXT_LIMIT = 60
    DEFAULT_SEARCH_CONTEXT_TOP_K = 3

    def __init__(
        self,
        config,
        llm: LlmProvider,
    ) -> None:

        self.config = config

        self.logger = get_logger("[LOOP]")

        self.llm = llm

        memory_config = self.config.get("memory") or {}
        self.stm = STM(
            db_path=memory_config.get("stm_db_path", "data/stm.db")
        )

        self.session_id: UUID | None = None

        self._context_step = 0
        self._workspace_root: Path | None = None

        self.tool = ToolManager(config=self.config)
        self.tool.set_llm_provider(self.llm)
        self.tool.set_context_store(self.stm, lambda: self.session_id)

        self._run_started_at: float | None = None
        self.metrics: dict[str, Any] = {}

        self.context = ContextService(
            config=self.config,
            llm=self.llm,
            stm=self.stm,
        )

        self.agent_state = AgentState()

        self.working_set = WorkingSet()

        self._plan_progress = PlanProgressTracker()

        self.workspace_revision = 0

        self._same_revision_read_count = 0
        self._observation_action_count = 0
        self._phase = "explore"
        self._active_process_ids: set[str] = set()

        # A successful observation is reusable until an intervening write/action
        # changes the workspace. Prevent the model from burning turns by issuing
        # the exact same read/search/fetch again with no new evidence.
        self._workspace_mutation_epoch = 0
        self._successful_observation_signatures: dict[str, int] = {}
        self._successful_mutation_signatures: dict[str, int] = {}
        observation_cache_config = self.config.get("observation_cache") or {}
        self.observation_cache_enabled = bool(observation_cache_config.get("enabled", True))
        self.observation_cache_max_entries = 20
        self.observation_cache_max_result_chars = 200_000
        self._observation_cache: dict[str, tuple[int, ToolResult]] = {}
        self._consecutive_observation_cache_hits = 0
        # A deterministic command that fails repeatedly with the same result
        # should not consume the remaining agent iterations forever.
        # signature -> (workspace epoch, failure fingerprint, consecutive failures)
        self._failed_command_signatures: dict[str, tuple[int, str, int]] = {}
        self._identical_column_advisories_seen: set[tuple[str, str]] = set()

        self._generation_retries = 0
        # Consecutive model turns that produce neither a tool call nor a final
        # response are a liveness problem, not a strategy decision. Keep the
        # model in control, but make the next prompt explicitly demand progress.
        self._no_action_turns = 0

        self._event_sink: Callable[[dict[str, Any]], None] | None = None
        self._stop_event: Any | None = None
        self._steering_queue: queue.Queue[str] = queue.Queue()

        # Plan enforcement is scoped to the current run. A stale plan left
        # behind by an earlier user task must not block an unrelated new turn.
        self._plan_active_this_run = False

        context_config = self.config.get("context") or {}
        retrieval_config = self.config.get("retrieval") or {}

        try:
            self.recent_context_limit = max(
                1,
                int(
                    context_config.get(
                        "recent_event_limit",
                        self.DEFAULT_RECENT_CONTEXT_LIMIT,
                    )
                ),
            )
        except (TypeError, ValueError):
            self.recent_context_limit = self.DEFAULT_RECENT_CONTEXT_LIMIT

        try:
            self.search_context_top_k = max(
                1,
                int(
                    retrieval_config.get(
                        "top_k",
                        self.DEFAULT_SEARCH_CONTEXT_TOP_K,
                    )
                ),
            )
        except (TypeError, ValueError):
            self.search_context_top_k = self.DEFAULT_SEARCH_CONTEXT_TOP_K

        self.max_iterations = self._read_max_iterations()

        # Optional model-directed finalization review. It evaluates outcomes,
        # not a fixed workflow, and can send an unfinished task back to the loop.
        review_config = self.config.get("completion_review") or {}
        self.completion_review_enabled = bool(review_config.get("enabled", False))
        try:
            self.completion_review_max_retries = max(
                0, min(5, int(review_config.get("max_retries", 3)))
            )
        except (TypeError, ValueError):
            self.completion_review_max_retries = 3
        self.completion_reviewer = (
            CompletionReviewer(self.llm)
            if self.completion_review_enabled
            else None
        )
        self._completion_review_retries = 0
        self._tool_turns_since_recovery = 0
        try:
            self.recovery_review_interval = max(
                2, min(8, int(review_config.get("recovery_interval", 3)))
            )
            self.recovery_review_max_checks = max(
                0, min(8, int(review_config.get("max_recovery_checks", 4)))
            )
        except (TypeError, ValueError):
            self.recovery_review_interval = 3
            self.recovery_review_max_checks = 4
        self._recovery_review_count = 0

        self.tool_definitions = self.tool.get_tools()

    def _read_max_iterations(
        self,
    ) -> int:

        try:

            value = int(
                self.config.get(
                    "max_agent_iterations",
                    self.DEFAULT_MAX_ITERATIONS,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            value = self.DEFAULT_MAX_ITERATIONS

        return max(
            1,
            value,
        )

    def _next_step(
        self,
    ) -> int:

        self._context_step += 1

        return self._context_step

    def steer(self, text: str) -> None:
        """Queue operator guidance to be consumed by the next safe loop boundary."""
        message = str(text or "").strip()
        if not message:
            return
        self._steering_queue.put(message)

    def _consume_steering(self) -> None:
        while True:
            try:
                message = self._steering_queue.get_nowait()
            except queue.Empty:
                return

            self._store_nudge(
                "Operator steering for the current run:\n"
                f"{message}\n"
                "This is the latest operator instruction and may revise or "
                "replace the previous objective. Follow it exactly. Do not "
                "continue actions that conflict with it. If it says not to "
                "modify files, do not modify files. Treat older task details "
                "as background unless the latest instruction keeps them active."
            )
            self._emit_event("steering", text=message)

    def _emit_event(
        self,
        event_type: str,
        **payload: Any,
    ) -> None:
        callback = self._event_sink
        if not callable(callback):
            return
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "iteration": int(self.metrics.get("iterations", 0) or 0),
            **payload,
        }
        try:
            callback(event)
        except Exception as exc:
            self.logger.debug(f"Event sink failed: {type(exc).__name__}: {exc}")

    def _forward_llm_stream_event(self, event: dict[str, Any]) -> None:
        """Bridge provider stream events into the Loop event protocol.

        Providers emit one event dictionary at a time, while _emit_event()
        accepts an event type plus keyword payload. Passing the dictionary
        directly as the first positional argument makes event["type"] itself
        a dict and breaks every downstream consumer with
        TypeError: unhashable type: 'dict'.
        """
        if not isinstance(event, dict):
            return

        event_type = event.get("type")
        if not isinstance(event_type, str) or not event_type:
            return

        payload = {
            key: value
            for key, value in event.items()
            if key != "type"
        }
        self._emit_event(event_type, **payload)

    @staticmethod
    def _is_observation_command(command: Any) -> bool:
        if not isinstance(command, list):
            return False

        tokens = [str(item).strip().lower() for item in command if str(item).strip()]
        if not tokens:
            return False

        executable = tokens[0].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if executable.endswith(".exe"):
            executable = executable[:-4]

        if executable in {
            "pwd", "ls", "find", "grep", "rg", "cat", "head", "tail",
            "sed", "awk", "stat", "file", "which", "whereis", "type",
        }:
            return True

        if executable == "git":
            subcommand = tokens[1] if len(tokens) > 1 else ""
            return subcommand in {
                "status", "diff", "show", "log", "ls-files",
                "branch", "rev-parse", "describe", "remote",
            }

        if executable in {"python", "python3", "pypy", "pypy3"}:
            return "-c" in tokens or (
                "-m" in tokens and "inspect" in tokens
            )

        return False

    @classmethod
    def _is_observation_call(cls, call) -> bool:
        name = str(getattr(call, "name", "")).strip().lower()
        arguments = getattr(call, "args", {}) or {}
        if name in {"web_fetch", "web_search"} and isinstance(arguments, dict) and arguments.get("save_to"):
            return False

        if name in {"read_file", "grep", "glob", "list_dir", "explore", "web_fetch", "web_search"}:
            return True

        if name == "command_exec":
            arguments = getattr(call, "args", {}) or {}
            return cls._is_observation_command(arguments.get("command"))

        return False

    def _resume_step_counter(
        self,
    ) -> None:
        """
        `_reset_run_state()` zeroes the in-memory step counter on
        every `run()` call, but `session_id` and STM rows normally
        survive across many `run()` calls in the same session.

        Resume from the last stored step so newer turns always get
        higher step numbers.
        """

        if self.session_id is None:
            return

        try:

            recent = self.stm.get_recent(
                session_id=self.session_id,
                limit=1,
            )

        except Exception as exc:

            self.logger.warning("Could not resume step counter: " f"{exc}")

            return

        if recent:

            self._context_step = max(
                self._context_step,
                recent[-1].step,
            )

    def _store_event(
        self,
        event: ContextEvent,
    ) -> None:

        if self.session_id is None:

            raise RuntimeError(
                "Cannot store ContextEvent " "without an active session."
            )

        self.stm.add(
            session_id=self.session_id,
            event=event,
        )

    def _store_nudge(
        self,
        text: str,
    ) -> None:
        """
        Store a short corrective message as a normal USER turn (not
        SYSTEM: a mid-conversation system message is a coin-flip across
        providers, a user turn is universally supported) so the NEXT
        generation call actually sees different input.

        This matters because, previously, a failed/empty iteration was
        simply retried with byte-identical context — same messages, same
        temperature setting, no new information — so the model had no
        reason to behave differently and would often fail the exact same
        way 2-3 times in a row before the loop gave up.
        """

        self._store_event(
            ContextEvent(
                role=ContextRole.USER,
                type=ContextType.MESSAGE,
                content=text,
                priority=ContextPriority.HIGH,
                step=self._next_step(),
                metadata={"runtime_nudge": True},
            )
        )

    def _assistant_event(
        self,
        llmresult: LLMResult,
        normalized_tool_calls: list | None = None,
    ) -> ContextEvent:

        message = (
            dict(llmresult.message)
            if isinstance(
                llmresult.message,
                dict,
            )
            else {}
        )

        message["role"] = "assistant"

        message_content = message.get("content")

        if message_content is None:

            message_content = llmresult.response or ""

        if not isinstance(
            message_content,
            str,
        ):

            message_content = str(message_content)

        message["content"] = message_content

        if normalized_tool_calls is not None:

            serialized_tool_calls: list[dict[str, Any]] = []

            for call in normalized_tool_calls:

                call_id = str(
                    getattr(
                        call,
                        "id",
                        "",
                    )
                ).strip()

                name = str(
                    getattr(
                        call,
                        "name",
                        "",
                    )
                ).strip()

                arguments = (
                    getattr(
                        call,
                        "args",
                        {},
                    )
                    or {}
                )

                try:

                    arguments_json = json.dumps(
                        arguments,
                        ensure_ascii=False,
                        default=str,
                    )

                except Exception:

                    arguments_json = "{}"

                serialized_tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments_json,
                        },
                    }
                )

            message["tool_calls"] = serialized_tool_calls

        # Preserve model reasoning/thinking when the provider exposes it.
        if "thinking" not in message and llmresult.thinking:

            message["thinking"] = str(llmresult.thinking)

        metadata: dict[str, Any] = {
            "has_tool_calls": bool(normalized_tool_calls),
            "llm_message": message,
        }

        if llmresult.thinking:

            metadata["thinking"] = str(llmresult.thinking)

        return ContextEvent(
            role=ContextRole.ASSISTANT,
            type=ContextType.MESSAGE,
            content=message_content,
            priority=ContextPriority.NORMAL,
            step=self._next_step(),
            metadata=metadata,
        )

    def _tool_call_event(
        self,
        call,
    ) -> ContextEvent:

        payload = {
            "name": getattr(
                call,
                "name",
                "",
            ),
            "tool_call_id": getattr(
                call,
                "id",
                "",
            ),
            "args": getattr(
                call,
                "args",
                {},
            ),
            "action": getattr(
                call,
                "action",
                "execute",
            ),
            "target": getattr(
                call,
                "target",
                "",
            ),
            "valid": getattr(
                call,
                "valid",
                False,
            ),
            "validation_error": getattr(
                call,
                "validation_error",
                "",
            ),
            "normalization_notes": getattr(
                call,
                "normalization_notes",
                [],
            ),
            "approved": getattr(
                call,
                "approved",
                False,
            ),
        }

        return ContextEvent(
            role=ContextRole.ASSISTANT,
            type=ContextType.TOOL_CALL,
            content=json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            ),
            priority=ContextPriority.NORMAL,
            step=self._next_step(),
        )

    def _tool_result_event(
        self,
        call,
        result: ToolResult,
    ) -> ContextEvent:

        payload = {
            "name": result.name,
            "tool_call_id": getattr(
                call,
                "id",
                "",
            ),
            "success": result.success,
            "summary": result.summary,
            "evidence": result.evidence,
            "effects": result.effects,
            "content": result.content,
            "metadata": result.metadata,
        }

        return ContextEvent(
            role=ContextRole.TOOL,
            type=ContextType.TOOL_RESULT,
            content=json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            ),
            priority=ContextPriority.NORMAL,
            step=self._next_step(),
        )

    @staticmethod
    def _invalid_result(
        call,
    ) -> ToolResult:

        name = str(
            getattr(
                call,
                "name",
                "unknown",
            )
        )

        validation_error = str(
            getattr(
                call,
                "validation_error",
                "",
            )
        ).strip()

        message = validation_error or "Invalid tool call."

        notes = getattr(
            call,
            "normalization_notes",
            [],
        )

        return ToolResult(
            success=False,
            name=name,
            content={
                "success": False,
                "error": {
                    "type": "invalid_tool_call",
                    "message": message,
                },
            },
            metadata={
                "recovery_hint": (
                    "Reissue the tool call with arguments matching the documented "
                    "schema exactly. Do not repeat an invalid call unchanged."
                ),
                "normalization_notes": notes,
            },
            summary=message,
        )


    @staticmethod
    def _missing_result(
        call,
    ) -> ToolResult:

        name = str(
            getattr(
                call,
                "name",
                "unknown",
            )
        )

        return ToolResult(
            success=False,
            name=name,
            content={
                "success": False,
                "error": {
                    "type": "missing_tool_result",
                    "message": ("ToolManager did not " "return a result."),
                },
            },
            metadata={},
            summary=("Tool execution produced " "no result."),
        )

    def _stopped_result(
        self,
        reason: str,
    ) -> LLMResult:

        result = LLMResult(
            response=("I stopped before " f"finishing the task. " f"Reason: {reason}"),
            message={},
            tool_calls=[],
            thinking=None,
            usage=0,
            raw=None,
        )

        self._store_event(self._assistant_event(result))
        self.metrics["completed"] = False
        self.metrics["stop_reason"] = reason
        self.metrics["duration_ms"] = self._duration_ms()
        self.tool.close()

        return result

    def _read_plan_state(
        self,
    ) -> PlanState:
        try:
            plan_tool = self.tool.get_tool("plan")
            snapshot = getattr(plan_tool, "snapshot", None)

            if callable(snapshot):
                state = snapshot()

                if isinstance(state, PlanState):
                    return state

        except Exception as exc:
            self.logger.warning("Could not read plan state: " f"{exc}")

        return PlanState.empty()

    @staticmethod
    def _is_plan_call(
        call,
    ) -> bool:
        return str(getattr(call, "name", "")).strip().lower() == "plan"


    def _observation_epoch(self, call) -> int:
        name = str(getattr(call, "name", "")).strip().lower()
        arguments = getattr(call, "args", {}) or {}
        if name in {"web_fetch", "web_search"} and isinstance(arguments, dict) and arguments.get("save_to"):
            return self._workspace_mutation_epoch
        # Local writes cannot make an identical remote search/fetch more useful.
        return -1 if name in {"web_fetch", "web_search"} else self._workspace_mutation_epoch

    def _observation_signature(self, call) -> str | None:
        """Return a stable fingerprint for an action, normalizing workspace paths."""
        arguments = getattr(call, "args", {}) or {}
        if isinstance(arguments, dict):
            arguments = dict(arguments)
            # ToolManager may anchor a relative path to an absolute workspace path
            # between classification and result handling. Normalize it before
            # hashing so the no-progress guard sees both spellings as the same call.
            path_keys = {
                "file_path", "path", "target", "directory", "save_to",
                "working_directory", "workspace_directory",
            }
            for key, value in list(arguments.items()):
                if not isinstance(value, str):
                    continue
                if str(key).strip().lower() not in path_keys and not str(key).strip().lower().endswith("_path"):
                    continue
                try:
                    normalized = Path(value).expanduser()
                    if not normalized.is_absolute() and self._workspace_root is not None:
                        normalized = self._workspace_root / normalized
                    arguments[key] = str(normalized.resolve(strict=False))
                except (OSError, RuntimeError, ValueError):
                    pass
        try:
            canonical = json.dumps(
                arguments, sort_keys=True, ensure_ascii=False,
                separators=(",", ":"), default=str,
            )
        except Exception:
            canonical = str(arguments)
        raw = f"{str(getattr(call, 'name', '')).strip().lower()}:{canonical}"
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _repeated_mutation_result(call) -> ToolResult:
        name = str(getattr(call, "name", "unknown")).strip() or "unknown"
        message = (
            "Repeated mutation blocked: this exact file-write/patch operation already "
            "succeeded and no intervening workspace change explains repeating it. "
            "Do not rewrite identical content or route around this guard. Use the "
            "observed validation/read results to make a concrete correction, then write "
            "the changed content and verify the resulting file."
        )
        return ToolResult(
            success=False,
            name=name,
            content={
                "success": False,
                "error": {
                    "type": "repeated_mutation_no_progress",
                    "message": message,
                },
            },
            metadata={
                "recovery_hint": (
                    "Change the file content based on the actual diagnostic or new evidence; "
                    "do not replay the same successful write/patch unchanged."
                ),
            },
            summary=message,
        )

    @staticmethod
    def _command_failure_fingerprint(result: ToolResult) -> str:
        content = result.content if isinstance(result.content, dict) else {}
        error = content.get("error")
        evidence = {
            "summary": str(result.summary or "")[:500],
            "error": error,
            "exit_code": content.get("exit_code"),
            "stderr": str(content.get("stderr") or "")[:1200],
        }
        canonical = json.dumps(
            evidence, sort_keys=True, ensure_ascii=False, default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _repeated_failed_command_result(call) -> ToolResult:
        name = str(getattr(call, "name", "command_exec")).strip() or "command_exec"
        message = (
            "Repeated failed command blocked: this exact command has failed twice "
            "with the same observed result and no intervening workspace change. "
            "Use the actual error/output to fix the cause or choose a materially "
            "different permitted approach. Do not change only the quoting/wrapper "
            "to bypass this guard. A retry is appropriate only after new evidence "
            "supports a transient failure or the relevant state has changed."
        )
        return ToolResult(
            success=False,
            name=name,
            content={
                "success": False,
                "error": {
                    "type": "repeated_command_failure_no_progress",
                    "message": message,
                },
            },
            metadata={
                "recovery_hint": (
                    "Inspect the preceding command failure; correct its cause or "
                    "select a different approach instead of rerunning it unchanged."
                ),
            },
            summary=message,
        )

    def _cached_observation_result(self, call, signature: str) -> ToolResult | None:
        if not self.observation_cache_enabled:
            return None
        cached = self._observation_cache.get(signature)
        if cached is None:
            return None
        epoch, original = cached
        if epoch != self._observation_epoch(call):
            return None
        result = copy.deepcopy(original)
        result.success = True
        result.content = dict(result.content)
        result.content["cached"] = True
        if "success" in result.content:
            result.content["success"] = True
        result.metadata = dict(result.metadata)
        result.metadata["cached_observation"] = True
        result.summary = "(cached: identical earlier call, workspace unchanged) " + result.summary
        return result

    def _remember_observation(self, call, signature: str, result: ToolResult) -> None:
        if not self.observation_cache_enabled or not result.success:
            return
        if result.metadata.get("cached_observation"):
            return
        try:
            content_size = len(json.dumps(
                {"content": result.content, "summary": result.summary, "evidence": result.evidence},
                ensure_ascii=False, default=str,
            ))
        except Exception:
            content_size = len(str(result.content)) + len(result.summary)
        if content_size > self.observation_cache_max_result_chars:
            return
        self._observation_cache.pop(signature, None)
        self._observation_cache[signature] = (
            self._observation_epoch(call),
            copy.deepcopy(result),
        )
        while len(self._observation_cache) > self.observation_cache_max_entries:
            self._observation_cache.pop(next(iter(self._observation_cache)))

    @staticmethod
    def _repeated_observation_result(call) -> ToolResult:
        name = str(getattr(call, "name", "unknown")).strip() or "unknown"
        message = (
            "Repeated observation blocked: this exact read/search/fetch already "
            "succeeded and the workspace has not changed since. Reuse the earlier "
            "result and make progress from it (for example, parse/process the returned "
            "data or create/update the requested artifact). Only retry this observation "
            "if you change a relevant argument based on new evidence."
        )
        return ToolResult(
            success=False,
            name=name,
            content={
                "success": False,
                "error": {
                    "type": "repeated_observation_no_progress",
                    "message": message,
                },
            },
            metadata={
                "recovery_hint": (
                    "Do not repeat this call unchanged. Use evidence already in context; "
                    "take a concrete different action toward the original task and verify its result."
                ),
            },
            summary=message,
        )

    def _classify_calls(
        self,
        parsed_calls: list,
    ) -> tuple[list[int], dict[int, ToolResult]]:
        """Reject malformed calls and exact, no-progress observation repeats.

        This is a liveness guard, not a task workflow: it only blocks a read-only
        call when that same call has already succeeded in the current workspace
        state, or appears twice in the same batch. Writes and changed queries remain
        available to the model as normal.
        """
        allowed_indices: list[int] = []
        blocked_results: dict[int, ToolResult] = {}
        batch_signatures: set[str] = set()
        batch_mutation_signatures: set[str] = set()

        for index, call in enumerate(parsed_calls):
            if not getattr(call, "valid", False):
                blocked_results[index] = self._invalid_result(call)
                continue

            tool_name = str(getattr(call, "name", "")).strip().lower()
            signature = self._observation_signature(call)

            if tool_name == "command_exec":
                previous = self._failed_command_signatures.get(signature)
                if (
                    previous is not None
                    and previous[0] == self._workspace_mutation_epoch
                    and previous[2] >= 2
                ):
                    blocked_results[index] = self._repeated_failed_command_result(call)
                    self.metrics["repeated_command_failure_blocks"] = (
                        self.metrics.get("repeated_command_failure_blocks", 0) + 1
                    )
                    continue

            if tool_name in {"write_file", "edit_file", "apply_patch", "applypatch"}:
                previous_epoch = self._successful_mutation_signatures.get(signature)
                repeated_mutation_in_batch = signature in batch_mutation_signatures
                repeated_without_progress = (
                    previous_epoch is not None
                    and previous_epoch == self._workspace_mutation_epoch
                )
                if repeated_mutation_in_batch or repeated_without_progress:
                    blocked_results[index] = self._repeated_mutation_result(call)
                    self.metrics["duplicate_mutation_blocks"] = (
                        self.metrics.get("duplicate_mutation_blocks", 0) + 1
                    )
                    continue
                batch_mutation_signatures.add(signature)

            if self._is_observation_call(call):
                repeated_in_batch = signature in batch_signatures
                cached_result = (
                    None if repeated_in_batch
                    else self._cached_observation_result(call, signature)
                )
                if cached_result is not None:
                    self._consecutive_observation_cache_hits += 1
                    self.metrics["duplicate_observation_cache_hits"] = (
                        self.metrics.get("duplicate_observation_cache_hits", 0) + 1
                    )
                    guard_level = str(self.config.get("guard_level", "strict")).strip().lower()
                    hit_limit = 6 if guard_level == "light" else 3
                    if self._consecutive_observation_cache_hits >= hit_limit:
                        blocked_results[index] = self._repeated_observation_result(call)
                        self.metrics["duplicate_observation_blocks"] = (
                            self.metrics.get("duplicate_observation_blocks", 0) + 1
                        )
                    else:
                        blocked_results[index] = cached_result
                    continue

                previous_epoch = self._successful_observation_signatures.get(signature)
                repeated_without_progress = (
                    previous_epoch is not None
                    and previous_epoch == self._observation_epoch(call)
                )
                if repeated_in_batch or repeated_without_progress:
                    blocked_results[index] = self._repeated_observation_result(call)
                    self.metrics["duplicate_observation_blocks"] = (
                        self.metrics.get("duplicate_observation_blocks", 0) + 1
                    )
                    continue
                batch_signatures.add(signature)

            allowed_indices.append(index)

        return allowed_indices, blocked_results

    def _execute_allowed_calls(
        self,
        normalized_tool_calls: list,
        allowed_indices: list[int],
    ) -> tuple[
        list,
        list[ToolResult],
    ]:

        if not allowed_indices:

            return [], []

        allowed_calls = [normalized_tool_calls[index] for index in allowed_indices]

        try:

            output = self.tool.execute(allowed_calls)

        except Exception as exc:

            self.logger.error(f"Tool execution failed: {exc}")

            return [], []

        if not isinstance(
            output,
            dict,
        ):

            return [], []

        calls = output.get(
            "calls",
            [],
        )

        results = output.get(
            "results",
            [],
        )

        if not isinstance(
            calls,
            list,
        ):

            calls = []

        if not isinstance(
            results,
            list,
        ):

            results = []

        return (
            calls,
            results,
        )

    @staticmethod
    def _tool_result_status(result: ToolResult) -> str:
        content = result.content if isinstance(result.content, dict) else {}
        status = str(content.get("status", "")).strip().lower()
        if status in {"running", "exited", "terminated", "unknown", "accepted"}:
            return status
        return "completed" if result.success else "failed"

    @staticmethod
    def _tool_result_process_id(result: ToolResult) -> str | None:
        content = result.content if isinstance(result.content, dict) else {}
        process_id = content.get("process_id")
        return str(process_id).strip() if process_id else None



    def _apply_result(
        self,
        call,
        result: ToolResult,
        iteration: int,
    ) -> None:

        plan_before = self._read_plan_state()
        if result.success and self._is_plan_call(call):
            self._plan_active_this_run = True

        self._plan_progress.sync(
            plan_before,
            iteration=iteration,
            workspace_revision=self.workspace_revision,
        )

        self.logger.info(
            f"Tool result ← "
            f"{result.name} "
            f"| success={result.success} "
            f"| summary={result.summary}"
        )

        self.agent_state.begin(
            tool_call=call,
            iteration=iteration,
        )

        self._store_event(self._tool_call_event(call))

        self.agent_state.update_from_result(result)

        status = self._tool_result_status(result)
        process_id = self._tool_result_process_id(result)

        if status == "running" and process_id:
            # Background commands are intentionally detached from the current
            # execution boundary (for example a web server). Foreground
            # commands must be observed to completion before unrelated work.
            content = result.content if isinstance(result.content, dict) else {}
            is_background = bool(content.get("background", False))
            if not is_background:
                self._active_process_ids.add(process_id)
        elif process_id and status in {"exited", "terminated", "unknown"}:
            self._active_process_ids.discard(process_id)

        changed = self.working_set.update(
            tool_call=call,
            result=result,
            iteration=iteration,
        )

        tool_name = str(getattr(call, "name", result.name)).strip().lower()
        if result.metadata.get("cached_observation"):
            pass
        else:
            self._consecutive_observation_cache_hits = 0
            if self._is_observation_call(call) and result.success:
                observation_signature = self._observation_signature(call)
                if observation_signature:
                    self._remember_observation(call, observation_signature, result)

        if result.success or (tool_name == "command_exec" and changed):
            # Treat known mutating tools as a new workspace state. A failed command
            # can still alter files before exiting non-zero, so observed workspace
            # changes also invalidate the no-progress fingerprint.
            is_web_save = (
                tool_name in {"web_fetch", "web_search"}
                and isinstance(getattr(call, "args", {}), dict)
                and bool(call.args.get("save_to"))
            )
            if tool_name in {"write_file", "edit_file", "apply_patch", "applypatch", "command_exec", "process_write"} or (is_web_save and result.success):
                self._workspace_mutation_epoch += 1
            if tool_name in {"write_file", "edit_file", "apply_patch", "applypatch"}:
                signature = self._observation_signature(call)
                if signature:
                    self._successful_mutation_signatures[signature] = self._workspace_mutation_epoch
            if self._is_observation_call(call):
                signature = self._observation_signature(call)
                if signature:
                    self._successful_observation_signatures[signature] = self._observation_epoch(call)

        self._plan_progress.record(
            tool_call=call,
            result=result,
            iteration=iteration,
        )

        workspace_mutated = changed and tool_name != "plan"

        if result.success and workspace_mutated:
            self.workspace_revision += 1
            self._same_revision_read_count = 0
            self._observation_action_count = 0
            self._phase = "implement"

        if tool_name == "command_exec":
            signature = self._observation_signature(call)
            if signature:
                if result.success:
                    self._failed_command_signatures.pop(signature, None)
                else:
                    failure_fingerprint = self._command_failure_fingerprint(result)
                    previous = self._failed_command_signatures.get(signature)
                    if (
                        previous is not None
                        and previous[0] == self._workspace_mutation_epoch
                        and previous[1] == failure_fingerprint
                    ):
                        failures = previous[2] + 1
                    else:
                        failures = 1
                    self._failed_command_signatures[signature] = (
                        self._workspace_mutation_epoch,
                        failure_fingerprint,
                        failures,
                    )

        if result.success:
            self.metrics["tool_successes"] = self.metrics.get("tool_successes", 0) + 1
        else:
            self.metrics["tool_failures"] = self.metrics.get("tool_failures", 0) + 1
            if status != "running":
                self._phase = "recover"

        self.metrics["active_processes"] = len(self._active_process_ids)

        if result.success and str(getattr(call, "name", result.name)).strip().lower() == "read_file" and not changed:
            self._same_revision_read_count += 1

        if self._is_observation_call(call) and str(getattr(call, "name", "")).strip().lower() != "process_poll":
            self._observation_action_count += 1

        # Always append the tool result before any corrective USER nudge.
        # Inserting a user message between an assistant tool-call and its tool
        # result breaks native tool-call grouping for providers.
        self._store_event(
            self._tool_result_event(
                call,
                result,
            )
        )

        if status == "running":
            self._store_nudge(
                f"'{result.name}' started a managed process "
                f"{process_id or '(unknown id)'}. The command is not complete. "
                "Poll it before treating the operation as finished."
            )

        error_type = ""
        error_message = ""
        if isinstance(result.content, dict):
            error = result.content.get("error")
            if isinstance(error, dict):
                error_type = str(error.get("type", "")).strip().lower()
                error_message = str(error.get("message", "")).strip()
            elif error:
                error_message = str(error).strip()
                error_type = error_message.lower()

        if not result.success:
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            lesson = str(metadata.get("recovery_hint") or error_message or result.summary).strip()
            if lesson:
                lesson_key = f"{tool_name}:{error_type or 'unknown_error'}"
                self.working_set.lessons.pop(lesson_key, None)
                self.working_set.lessons[lesson_key] = lesson[:200]
                while len(self.working_set.lessons) > 8:
                    self.working_set.lessons.pop(next(iter(self.working_set.lessons)))

        # Schema/dispatcher failures are concrete evidence. Keep the result
        # in context and let the model decide how to recover; do not inject a
        # fixed recovery sequence here.

        # Refresh real filesystem state before the next model turn so a file
        # that was created earlier cannot disappear from model-visible state
        # merely because its original event aged out of STM.
        self.working_set.refresh_workspace()

        self._sync_plan_runtime_state(
            call=call,
            result=result,
            plan_before=plan_before,
        )

    def _sync_plan_runtime_state(
        self,
        call,
        result: ToolResult,
        plan_before: PlanState | None = None,
    ) -> None:
        if not isinstance(result, ToolResult) or not result.success:
            return

        if self._is_plan_call(call):
            plan_after = self._read_plan_state()
            self._plan_progress.sync(
                plan_after,
                iteration=int(self.metrics.get("iterations", 0) or 0),
                workspace_revision=self.workspace_revision,
            )


    def _validate_written_artifact(self, call, result: ToolResult) -> list[str]:
        """Validate basic syntax/shape of structured artifacts created or edited by file tools.

        This checks the file format only. It does not fill data, infer task-specific
        requirements, or modify the artifact; any finding is returned to the model.
        """
        if not isinstance(result, ToolResult) or not result.success:
            return []
        tool_name = str(getattr(call, "name", result.name)).strip().lower()
        if tool_name not in {"write_file", "edit_file", "apply_patch", "applypatch"}:
            return []
        payload = result.content if isinstance(result.content, dict) else {}
        candidates: list[tuple[str, str]] = []
        if tool_name == "write_file":
            raw_path = payload.get("path") or payload.get("file_path")
            if raw_path:
                candidates.append((str(raw_path), "write"))
        files = payload.get("files")
        if isinstance(files, list):
            for item in files:
                if not isinstance(item, dict):
                    continue
                raw_path = item.get("path")
                operation = str(item.get("operation") or "edit").lower()
                if raw_path and operation not in {"delete", "remove"}:
                    candidates.append((str(raw_path), operation))

        root = self._workspace_root
        if root is None:
            return []
        root = root.expanduser().resolve()
        findings: list[str] = []
        seen_paths: set[str] = set()
        for raw_path, _operation in candidates:
            try:
                path = Path(raw_path).expanduser().resolve()
                path.relative_to(root)
            except (OSError, ValueError):
                continue
            path_key = str(path)
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            if not path.is_file() or path.stat().st_size > 5_000_000:
                continue
            suffix = path.suffix.lower()
            try:
                if suffix == ".csv":
                    with path.open("r", encoding="utf-8-sig", newline="") as stream:
                        rows = list(csv.reader(stream, strict=True))
                    if not rows:
                        findings.append(f"{path.name}: CSV file is empty.")
                        continue
                    header_width = len(rows[0])
                    if header_width == 0 or not any(cell.strip() for cell in rows[0]):
                        findings.append(f"{path.name}: CSV header is empty.")
                        continue
                    bad_rows = [
                        (line_no, len(row))
                        for line_no, row in enumerate(rows[1:], start=2)
                        if len(row) != header_width
                    ]
                    if bad_rows:
                        samples = ", ".join(
                            f"line {line_no} has {width} fields (expected {header_width})"
                            for line_no, width in bad_rows[:5]
                        )
                        extra = f"; and {len(bad_rows) - 5} more" if len(bad_rows) > 5 else ""
                        findings.append(f"{path.name}: malformed CSV structure: {samples}{extra}.")
                    else:
                        data_rows = rows[1:]
                        for column_index, raw_header in enumerate(rows[0]):
                            header = re.sub(r"[^a-z0-9]+", "_", raw_header.strip().lower()).strip("_")
                            values = [(line_no, row[column_index].strip())
                                      for line_no, row in enumerate(data_rows, start=2)]
                            if not values:
                                continue

                            non_empty_values = [value for _, value in values if value]
                            advisory_key = (path_key, str(raw_header))
                            if (
                                len(values) >= 5
                                and len(non_empty_values) == len(values)
                                and len(set(non_empty_values)) == 1
                                and advisory_key not in self._identical_column_advisories_seen
                            ):
                                findings.append(
                                    f"ADVISORY: {path.name}: column '{raw_header}' has the same value "
                                    f"in all {len(values)} rows; confirm it is present in each source "
                                    "record, otherwise leave it blank."
                                )
                                self._identical_column_advisories_seen.add(advisory_key)

                            # Common semantic checks for columns whose names declare a format.
                            if "url" in header or "link" in header:
                                for line_no, value in values:
                                    if not value:
                                        findings.append(
                                            f"{path.name}: URL/link column '{raw_header}' is empty on line {line_no}; "
                                            "use the source URL or omit that record, never fabricate a link."
                                        )
                                        break
                                    parsed = urlparse(value)
                                    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                                        findings.append(
                                            f"{path.name}: URL/link column '{raw_header}' is not an absolute HTTP(S) URL "
                                            f"on line {line_no}."
                                        )
                                        break

                            numeric_tokens = {
                                "price", "amount", "cost", "area", "size", "room", "rooms",
                                "bedroom", "bedrooms", "m2", "sqm", "count", "quantity",
                                "latitude", "longitude", "lat", "lon", "year", "age",
                            }
                            header_tokens = set(header.split("_"))
                            if header_tokens & numeric_tokens:
                                for line_no, value in values:
                                    if not value:
                                        continue
                                    numeric = (value.replace(",", "").replace("٬", "")
                                               .replace("٫", ".").replace("−", "-"))
                                    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", numeric):
                                        findings.append(
                                            f"{path.name}: numeric-looking column '{raw_header}' has a nonnumeric "
                                            f"value on line {line_no}."
                                        )
                                        break

                            time_like = any(token in header for token in (
                                "timestamp", "retrieved_at", "created_at", "updated_at",
                                "collected_at", "scraped_at", "datetime", "date_time",
                            ))
                            if time_like:
                                for line_no, value in values:
                                    if not value:
                                        findings.append(
                                            f"{path.name}: timestamp column '{raw_header}' is empty on line {line_no}."
                                        )
                                        break
                                    try:
                                        datetime.fromisoformat(value.replace("Z", "+00:00"))
                                    except ValueError:
                                        findings.append(
                                            f"{path.name}: timestamp column '{raw_header}' is not ISO-8601 on line {line_no}."
                                        )
                                        break
                                    if ("T" not in value and not re.search(r"\s\d{2}:\d{2}", value)):
                                        findings.append(
                                            f"{path.name}: timestamp column '{raw_header}' contains a date without a time "
                                            f"on line {line_no}; use a full ISO-8601 date-time."
                                        )
                                        break
                elif suffix == ".json":
                    with path.open("r", encoding="utf-8-sig") as stream:
                        json.load(stream)
                elif suffix == ".jsonl":
                    with path.open("r", encoding="utf-8-sig") as stream:
                        for line_no, line in enumerate(stream, start=1):
                            if line.strip():
                                try:
                                    json.loads(line)
                                except json.JSONDecodeError as exc:
                                    findings.append(
                                        f"{path.name}: invalid JSONL at line {line_no}: {exc.msg}."
                                    )
                                    break
            except (OSError, UnicodeError, csv.Error, json.JSONDecodeError) as exc:
                findings.append(f"{path.name}: format validation failed ({type(exc).__name__}: {exc}).")
        return findings[:8]

    def _execute_tool_calls(
        self,
        parsed_calls: list,
        iteration: int,
    ) -> bool:

        if not parsed_calls:

            return False

        for call in parsed_calls:
            self._emit_event(
                "tool_call",
                name=str(getattr(call, "name", "")),
                arguments=getattr(call, "args", {}) or {},
            )

        self.metrics["tool_call_attempts"] = (
            self.metrics.get("tool_call_attempts", 0) + len(parsed_calls)
        )

        (
            allowed_indices,
            blocked_results,
        ) = self._classify_calls(parsed_calls)

        (
            executed_calls,
            executed_results,
        ) = self._execute_allowed_calls(
            normalized_tool_calls=(parsed_calls),
            allowed_indices=(allowed_indices),
        )

        executed: dict[
            int,
            tuple[
                Any,
                ToolResult,
            ],
        ] = {}

        for (
            position,
            original_index,
        ) in enumerate(allowed_indices):

            if position >= len(executed_results):

                break

            result = executed_results[position]

            if not isinstance(
                result,
                ToolResult,
            ):

                result = self._missing_result(parsed_calls[original_index])

            if position < len(executed_calls):

                call = executed_calls[position]

            else:

                call = parsed_calls[original_index]

            executed[original_index] = (
                call,
                result,
            )

        artifact_findings: list[str] = []
        artifact_advisories: list[str] = []
        for index, call in enumerate(parsed_calls):

            if index in blocked_results:

                result = blocked_results[index]

            elif index in executed:

                call, result = executed[index]

            else:

                result = self._missing_result(call)

            # Validate the actual written artifact before storing the tool result.
            # The model must see the findings IN that result, not only in a later
            # nudge it can overlook while the write itself appears fully successful.
            findings = self._validate_written_artifact(call, result)
            if findings:
                advisories = [item.removeprefix("ADVISORY: ").strip()
                              for item in findings if item.startswith("ADVISORY: ")]
                blocking_findings = [item for item in findings
                                     if not item.startswith("ADVISORY: ")]
                artifact_advisories.extend(advisories)
                artifact_findings.extend(blocking_findings)
                result_content = (
                    dict(result.content)
                    if isinstance(result.content, dict)
                    else {"value": result.content}
                )
                result_content["artifact_validation"] = {
                    "passed": not blocking_findings,
                    "findings": blocking_findings,
                    "advisories": advisories,
                }
                result.content = result_content
                if blocking_findings:
                    result.summary = (
                        str(result.summary or "")
                        + " | ARTIFACT VALIDATION FAILED: "
                        + " ".join(blocking_findings)
                        + " The file was written, but it is not a valid deliverable yet."
                    )
                elif advisories:
                    result.summary = (
                        str(result.summary or "")
                        + " | ARTIFACT VALIDATION ADVISORY: "
                        + " ".join(advisories)
                    )
                result_metadata = dict(result.metadata or {})
                if blocking_findings:
                    result_metadata["validation_findings"] = blocking_findings
                if advisories:
                    result_metadata["validation_advisories"] = advisories
                result.metadata = result_metadata
                result_evidence = dict(result.evidence or {})
                result_evidence["artifact_validation"] = {
                    "passed": not blocking_findings,
                    "findings": blocking_findings,
                    "advisories": advisories,
                }
                result.evidence = result_evidence

            self._apply_result(
                call=call,
                result=result,
                iteration=iteration,
            )

            self._emit_event(
                "tool_result",
                name=str(getattr(call, "name", result.name)),
                success=bool(result.success),
                summary=str(result.summary or ""),
                content=result.content if isinstance(result.content, (str, dict, list)) else str(result.content),
            )

        # Keep validation feedback after every tool call/result pair in a batch;
        # inserting a user event between a model tool call and its result is invalid.
        if artifact_findings:
            unique_findings = list(dict.fromkeys(artifact_findings))
            self._store_nudge(
                "Runtime artifact format validation found concrete issue(s): "
                + " ".join(unique_findings)
                + " This is a format finding, not a successful task outcome. Correct the affected file, "
                  "then inspect or validate it again. The runtime will not create or repair task data for you."
            )

        if artifact_advisories:
            unique_advisories = list(dict.fromkeys(artifact_advisories))
            self._store_nudge(
                "Artifact validation advisory (not a failure): "
                + " ".join(unique_advisories)
                + " Verify these values against the source; do not replace missing values with placeholders."
            )

        # Invalid tool calls and artifact format findings are returned as evidence;
        # the model remains responsible for choosing the correction.
        return False

    def _completion_review_evidence(self) -> list[dict[str, Any]]:
        """Build compact, factual evidence for the model's final-outcome review."""
        if self.session_id is None:
            return []

        evidence: list[dict[str, Any]] = []
        for event in self.stm.get_recent(self.session_id, limit=28):
            event_type = str(getattr(event.type, "value", event.type))
            role = str(getattr(event.role, "value", event.role))
            metadata = event.metadata if isinstance(event.metadata, dict) else {}

            # Canonical tool calls are stored inside the provider-neutral assistant
            # message so native tool-call/result grouping stays valid. Extract them
            # here for recovery review, including their actual arguments/URLs.
            if role == "assistant" and event_type == "message":
                message = metadata.get("llm_message")
                calls = message.get("tool_calls") if isinstance(message, dict) else None
                if isinstance(calls, list) and calls:
                    extracted_calls = []
                    for call in calls[:4]:
                        function = call.get("function", {}) if isinstance(call, dict) else {}
                        if not isinstance(function, dict):
                            continue
                        raw_args = function.get("arguments", {})
                        if isinstance(raw_args, str):
                            try:
                                raw_args = json.loads(raw_args)
                            except (TypeError, ValueError):
                                raw_args = {"raw": raw_args[:250]}
                        if not isinstance(raw_args, dict):
                            raw_args = {}
                        # Keep useful parameters for diagnosis while bounding prompt size.
                        args = {}
                        for key, value in list(raw_args.items())[:8]:
                            if isinstance(value, (str, int, float, bool)) or value is None:
                                args[str(key)] = str(value)[:220]
                            else:
                                args[str(key)] = str(value)[:220]
                        extracted_calls.append({
                            "tool": str(function.get("name") or "")[:80],
                            "arguments": args,
                        })
                    if extracted_calls:
                        evidence.append({
                            "step": int(getattr(event, "step", 0) or 0),
                            "kind": "tool_calls",
                            "calls": extracted_calls,
                        })
                continue

            if event_type != "tool_result":
                continue

            try:
                payload = json.loads(str(event.content or ""))
            except (TypeError, ValueError):
                payload = {}
            if not isinstance(payload, dict):
                payload = {}

            item: dict[str, Any] = {
                "step": int(getattr(event, "step", 0) or 0),
                "kind": event_type,
                "tool": str(payload.get("name") or "")[:80],
            }

            if "success" in payload:
                item["success"] = bool(payload.get("success", False))
                summary = str(payload.get("summary") or "").strip()
                if summary:
                    item["summary"] = summary[:400]

                effects = payload.get("effects")
                if effects:
                    item["effects"] = str(effects)[:350]

                result_content = payload.get("content")
                if isinstance(result_content, dict):
                    for key in (
                        "status", "exit_code", "url", "requested_url",
                        "title", "total_chars", "row_count", "count",
                        "result_count", "retrieved_at", "backend", "extractor",
                        "start_char", "end_char", "truncated", "path",
                        "bytes_written", "characters_written", "lines_written",
                        "sha256", "preview_truncated", "overwrote_existing",
                        "total_lines",
                    ):
                        value = result_content.get(key)
                        if value is not None and value != "":
                            item[key] = str(value)[:240]

                    # Preserve the useful payload needed to judge task completion.
                    # A 450-character excerpt hid nearly all listing records from the
                    # reviewer, which then made a false "only two records" conclusion.
                    excerpt = (
                        result_content.get("content")
                        or result_content.get("stdout")
                        or result_content.get("stderr")
                    )
                    if isinstance(excerpt, str) and excerpt.strip():
                        if str(payload.get("name") or "").lower() == "web_fetch":
                            excerpt_limit = 6500
                        elif str(payload.get("name") or "").lower() in {"read_file", "readfile"}:
                            excerpt_limit = 5200
                        else:
                            excerpt_limit = 1800
                        item["untrusted_output_excerpt"] = excerpt.strip()[:excerpt_limit]

                    preview = result_content.get("preview")
                    if isinstance(preview, str) and preview.strip():
                        item["artifact_preview"] = preview.strip()[:5000]
                        item["artifact_preview_truncated"] = bool(
                            result_content.get("preview_truncated", False)
                        )

                    error = result_content.get("error")
                    if isinstance(error, dict):
                        item["error"] = str(error.get("message") or error)[:450]
                    elif error:
                        item["error"] = str(error)[:450]

                    files = result_content.get("files")
                    if files:
                        item["files"] = str(files)[:1800]
                elif isinstance(result_content, str) and result_content.strip():
                    item["untrusted_output_excerpt"] = result_content.strip()[:450]

                result_evidence = payload.get("evidence")
                if result_evidence:
                    item["evidence"] = str(result_evidence)[:350]

            evidence.append(item)

        return evidence[-14:]

    def _recovery_review_workspace(self, workspace_directory: str) -> dict[str, Any]:
        return {
            "root": workspace_directory,
            "file_count": self.working_set.workspace_file_count,
            "directory_count": self.working_set.workspace_directory_count,
            "inventory_truncated": self.working_set.workspace_inventory_truncated,
            "files": [
                {
                    "path": item.get("path"),
                    "type": item.get("type"),
                    "size": item.get("size"),
                }
                for item in self.working_set.workspace_inventory[:60]
            ],
            "recent_actions": self.working_set.recent_actions_context(),
            "observations": self.working_set.observation_context(),
            "tool_call_attempts": self.metrics.get("tool_call_attempts", 0),
            "tool_successes": self.metrics.get("tool_successes", 0),
            "tool_failures": self.metrics.get("tool_failures", 0),
        }

    def _review_stalled_execution(
        self,
        user_task: ContextEvent,
        workspace_directory: str,
    ) -> CompletionVerdict:
        reviewer = self.completion_reviewer
        if reviewer is None:
            return CompletionVerdict(decision="unknown", error="Recovery review is disabled.")

        self.metrics["llm_calls"] = self.metrics.get("llm_calls", 0) + 1
        self.metrics["recovery_review_calls"] = (
            self.metrics.get("recovery_review_calls", 0) + 1
        )
        verdict = reviewer.review_recovery(
            task=str(user_task.content or ""),
            evidence=self._completion_review_evidence(),
            workspace=self._recovery_review_workspace(workspace_directory),
            stop_event=self._stop_event,
        )
        self.metrics["tokens"] = (
            self.metrics.get("tokens", 0) + max(0, int(verdict.usage or 0))
        )
        if verdict.error:
            self.metrics["recovery_review_failures"] = (
                self.metrics.get("recovery_review_failures", 0) + 1
            )
            self.logger.warning("Execution recovery review unavailable: %s", verdict.error)

        self._recovery_review_count += 1
        self.metrics["recovery_review_count"] = self._recovery_review_count
        self._emit_event(
            "recovery_review",
            decision=verdict.decision,
            reason=verdict.reason[:500],
            next_action=verdict.next_action[:500],
            error=verdict.error[:300],
            usage=verdict.usage,
        )
        return verdict

    def _review_final_candidate(
        self,
        user_task: ContextEvent,
        candidate: LLMResult,
        workspace_directory: str,
    ) -> CompletionVerdict:
        reviewer = self.completion_reviewer
        if reviewer is None:
            return CompletionVerdict(decision="complete")

        self.metrics["llm_calls"] = self.metrics.get("llm_calls", 0) + 1
        self.metrics["completion_review_calls"] = (
            self.metrics.get("completion_review_calls", 0) + 1
        )

        workspace_evidence = {
            "root": workspace_directory,
            "file_count": self.working_set.workspace_file_count,
            "directory_count": self.working_set.workspace_directory_count,
            "inventory_truncated": self.working_set.workspace_inventory_truncated,
            "files": [
                {
                    "path": item.get("path"),
                    "type": item.get("type"),
                    "size": item.get("size"),
                }
                for item in self.working_set.workspace_inventory[:60]
            ],
            "recent_actions": self.working_set.recent_actions_context(),
            "observations": self.working_set.observation_context(),
            "tool_call_attempts": self.metrics.get("tool_call_attempts", 0),
            "tool_successes": self.metrics.get("tool_successes", 0),
            "tool_failures": self.metrics.get("tool_failures", 0),
        }

        verdict = reviewer.review(
            task=str(user_task.content or ""),
            draft=str(candidate.response or ""),
            evidence=self._completion_review_evidence(),
            workspace=workspace_evidence,
            stop_event=self._stop_event,
        )
        self.metrics["tokens"] = (
            self.metrics.get("tokens", 0) + max(0, int(verdict.usage or 0))
        )
        if verdict.error:
            self.metrics["completion_review_failures"] = (
                self.metrics.get("completion_review_failures", 0) + 1
            )
            self.logger.warning("Task completion review unavailable: %s", verdict.error)

        self._emit_event(
            "completion_review",
            decision=verdict.decision,
            reason=verdict.reason[:500],
            next_action=verdict.next_action[:500],
            error=verdict.error[:300],
            usage=verdict.usage,
        )
        return verdict

    def run(
        self,
        user_task: ContextEvent,
        workspace_directory: str | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        stop_event: Any | None = None,
    ) -> LLMResult | None:

        if not isinstance(
            user_task,
            ContextEvent,
        ):

            raise TypeError("user_task must be " "a ContextEvent.")

        # A new run starts a fresh runtime ownership boundary. Any process
        # left from a previous run is no longer needed by this task.
        self.tool.close()

        if workspace_directory is None:
            workspace_directory = str(Path.cwd().resolve())

        self._reset_run_state()
        self._event_sink = event_sink
        self._stop_event = stop_event
        self._run_started_at = time.perf_counter()
        self.set_workspace(workspace_directory)

        # Must run AFTER _reset_run_state()
        # and BEFORE the first _next_step().
        self._resume_step_counter()

        self._emit_event(
            "run_start",
            session_id=str(self.session_id),
            workspace=workspace_directory,
            max_iterations=self.max_iterations,
        )

        self.logger.info(
            "Starting agent loop | "
            f"session={self.session_id} | "
            f"max_iterations={self.max_iterations} | "
            f"workspace={workspace_directory}"
        )

        user_task.step = self._next_step()
        # Planning is an optional capability selected by the model.
        self.metrics["plan_required"] = False

        # Persist the task immediately.
        self._store_event(user_task)

        empty_streak = 0

        for iteration in range(self.max_iterations):

            iteration_number = iteration + 1

            self.agent_state.iteration = iteration_number
            self.metrics["iterations"] = iteration_number

            if self._stop_event is not None and self._stop_event.is_set():
                self.agent_state.stop("Interrupted by user.")
                self._emit_event("run_stopped", reason="Interrupted by user.")
                return self._stopped_result("Interrupted by user.")

            self._consume_steering()

            self._emit_event(
                "iteration_start",
                iteration=iteration_number,
                max_iterations=self.max_iterations,
            )

            self.logger.info(
                f"Iteration " f"{iteration_number}/" f"{self.max_iterations}"
            )

            llmresult = self._generate_next_action(
                user_task=user_task,
                workspace_directory=(workspace_directory),
            )

            if self._stop_event is not None and self._stop_event.is_set():
                reason = "Interrupted by user."
                self.agent_state.stop(reason)
                self._emit_event("run_stopped", reason=reason)
                return self._stopped_result(reason)

            if llmresult is None:

                # A provider-level failure (e.g. Ollama's own tool-call
                # parser choking on malformed model output). Retry a
                # bounded number of times with a corrective nudge instead
                # of ending the whole task on the first hiccup.
                self._generation_retries += 1

                error_message = self.agent_state.error or "LLM generation failed."

                if self._generation_retries > self.MAX_GENERATION_RETRIES:

                    return self._stopped_result(error_message)

                self.logger.warning(
                    "Generation failed "
                    f"({self._generation_retries}/"
                    f"{self.MAX_GENERATION_RETRIES}): "
                    f"{error_message}"
                )

                self._store_nudge(
                    "Your last request failed: "
                    f"{error_message} "
                    "Try again — call a tool with valid, well-formed "
                    "arguments matching its schema exactly, or answer "
                    "directly if no tool is needed."
                )

                continue

            self._generation_retries = 0

            self.context.calibrate(llmresult)

            if llmresult.tool_calls:

                empty_streak = 0
                self._no_action_turns = 0

                try:

                    # The ONLY place where raw provider tool calls are
                    # converted into canonical ToolCall objects.
                    parsed_calls = self.tool.dispatcher.dispatch(llmresult.tool_calls)

                except Exception as exc:

                    self.logger.error("Tool dispatch failed: " f"{exc}")

                    # Persist the assistant message for diagnostics.
                    # _assistant_event() deliberately stores textual content
                    # only in ContextEvent.content and keeps the full message
                    # under metadata["llm_message"].
                    self._store_event(self._assistant_event(llmresult))

                    self._generation_retries += 1

                    if self._generation_retries > self.MAX_GENERATION_RETRIES:

                        self.agent_state.fail("Tool dispatch failed: " f"{exc}")

                        return self._stopped_result(
                            self.agent_state.error or "Tool dispatch failed."
                        )

                    self._store_nudge(
                        "Your last tool call could not be parsed "
                        f"({exc}). Reissue it with strictly valid JSON "
                        "arguments matching the tool's parameter schema "
                        "exactly, one tool call at a time."
                    )

                    continue

                if not parsed_calls:

                    self.logger.error(
                        "Tool dispatcher returned " "no normalized calls."
                    )

                    self._store_event(self._assistant_event(llmresult))

                    self._generation_retries += 1

                    if self._generation_retries > self.MAX_GENERATION_RETRIES:

                        self.agent_state.fail("Tool dispatcher returned " "no calls.")

                        return self._stopped_result(
                            self.agent_state.error or "Tool dispatch failed."
                        )

                    self._store_nudge(
                        "Your last tool call was not recognized. Reissue "
                        "it using exactly one of the available tools and "
                        "its documented arguments."
                    )

                    continue

                # Store the canonical calls with stable IDs.
                self._store_event(
                    self._assistant_event(
                        llmresult,
                        normalized_tool_calls=(parsed_calls),
                    )
                )

                should_stop = self._execute_tool_calls(
                    parsed_calls,
                    iteration_number,
                )

                if should_stop:
                    return self._stopped_result(
                        self.agent_state.error or "Agent stopped."
                    )

                self._tool_turns_since_recovery += 1
                if (
                    self.completion_reviewer is not None
                    and self.recovery_review_interval > 0
                    and self.recovery_review_max_checks > self._recovery_review_count
                    and self._tool_turns_since_recovery >= self.recovery_review_interval
                ):
                    # Periodic model-led review is a progress check, not a prescribed
                    # tool sequence. The reviewer sees the actual recent call arguments,
                    # outputs, and workspace state and recommends whether to continue.
                    self._tool_turns_since_recovery = 0
                    verdict = self._review_stalled_execution(
                        user_task=user_task,
                        workspace_directory=workspace_directory,
                    )
                    guidance = (
                        "A separate model reviewed the recent execution evidence. "
                        f"Decision: {verdict.decision}. "
                        f"Reason: {verdict.reason or 'No decisive reason was returned.'} "
                    )
                    if verdict.next_action:
                        guidance += f"Suggested direction: {verdict.next_action} "
                    if verdict.decision == "complete":
                        guidance += (
                            "Check the actual requested outcome and artifacts. If verified, "
                            "summarize the result to the user instead of continuing aimlessly."
                        )
                    elif verdict.decision == "blocked":
                        guidance += (
                            "Confirm that the blocker is supported by actual observations. "
                            "If a permitted alternative remains, pursue it; otherwise report "
                            "the precise blocker and what the evidence establishes."
                        )
                    else:
                        guidance += (
                            "Choose the next action yourself using the original task and "
                            "observed evidence. Prefer a materially different approach when "
                            "the previous one was uninformative; do not repeat it without a reason."
                        )
                    self._store_nudge(guidance)

                # Tool results become the next model-visible evidence. Do not
                # auto-finalize plans, force verification, or reinterpret a tool
                # turn as completion; the model owns that decision.
                continue

            if isinstance(llmresult.response, str) and llmresult.response.strip():

                if self.completion_reviewer is not None:
                    verdict = self._review_final_candidate(
                        user_task=user_task,
                        candidate=llmresult,
                        workspace_directory=workspace_directory,
                    )
                    if verdict.decision == "continue":
                        self._store_event(self._assistant_event(llmresult))
                        self._completion_review_retries += 1
                        self.metrics["completion_review_retries"] = (
                            self._completion_review_retries
                        )
                        if self._completion_review_retries > self.completion_review_max_retries:
                            reason = (
                                "Task outcome remains incomplete after "
                                f"{self.completion_review_max_retries} recovery attempts. "
                                f"{verdict.reason}".strip()
                            )
                            self.agent_state.stop(reason)
                            return self._stopped_result(reason)

                        guidance = (
                            "The task-outcome review found that this draft does not yet "
                            "satisfy the original request based on observed evidence. "
                            f"Finding: {verdict.reason or 'The outcome is not verified.'} "
                        )
                        if verdict.next_action:
                            guidance += f"Possible next direction: {verdict.next_action} "
                        guidance += (
                            "Continue the original task. Treat this review as guidance, "
                            "not a fixed workflow: choose the most useful next action, "
                            "use tools when they can resolve uncertainty, inspect results, "
                            "and do not repeat the same unsupported final answer."
                        )
                        self._store_nudge(guidance)
                        continue

                    if verdict.decision == "blocked":
                        reason = (
                            verdict.reason
                            or "The outcome review identified a concrete blocker."
                        )
                        self.agent_state.fail(reason)
                        self.metrics["completed"] = False
                        self.metrics["stop_reason"] = f"blocked: {reason}"
                        # Never surface a rejected completion draft as if it succeeded.
                        return self._stopped_result(
                            f"Completion review rejected the result: {reason}"
                        )

                    if verdict.decision == "unknown":
                        self.logger.warning(
                            "Outcome review inconclusive; preserving the model response."
                        )

                self._store_event(self._assistant_event(llmresult))

                self.agent_state.complete()
                self.metrics["completed"] = True
                self.metrics["stop_reason"] = ""
                self.metrics["duration_ms"] = self._duration_ms()
                self._emit_event(
                    "final_response",
                    text=str(llmresult.response or ""),
                    usage=int(getattr(llmresult, "usage", 0) or 0),
                )
                self._emit_event(
                    "run_end",
                    completed=True,
                    stop_reason="",
                    metrics=self.get_metrics(),
                )
                self.tool.close()

                return llmresult

            empty_streak += 1
            self._no_action_turns += 1

            self.logger.warning("LLM produced neither " "response nor tool calls.")

            if empty_streak >= self.EMPTY_RESPONSE_THRESHOLD:

                reason = (
                    "The model returned "
                    "an empty response "
                    f"{self.EMPTY_RESPONSE_THRESHOLD} "
                    "times in a row."
                )

                self.agent_state.stop(reason)

                return self._stopped_result(reason)

            # Without this, the next iteration would retry with
            # byte-identical context (same messages, same temperature),
            # so the model has no reason to behave differently and often
            # fails the exact same way 2-3 times before the loop gives up.
            # Nudging with what actually happened gives the retry a real
            # chance to recover.
            nudge = "Your last turn produced no reply and no tool call."

            if llmresult.thinking:

                nudge += " You only reasoned silently without acting."

            if self._no_action_turns >= 2:
                nudge += (
                    " This is now a repeated no-progress turn. Stop analyzing "
                    "the same facts: either call the concrete tool needed for "
                    "the next action, or give the final answer if the task is done."
                )
            else:
                nudge += " Call a tool now to make progress, or write your final answer."

            self._store_nudge(nudge)

        reason = "Maximum iterations reached."

        self.agent_state.stop(reason)
        self.metrics["completed"] = False
        self.metrics["stop_reason"] = reason

        return self._stopped_result(reason)

    def _generate_next_action(
        self,
        user_task: ContextEvent,
        workspace_directory: str,
    ) -> LLMResult | None:

        if self.session_id is None:

            raise RuntimeError("No active session.")

        # Deterministic workspace truth is refreshed on every generation
        # boundary. This is independent of STM retrieval and therefore remains
        # correct even when old file-creation events are no longer in context.
        self.working_set.refresh_workspace()

        # A persisted plan belongs to an explicitly activated plan lifecycle
        # in the current run. Do not synchronize stale workspace plan state into
        # the model context before this run has created/activated a plan.
        if self._plan_active_this_run:
            plan_state = self._read_plan_state()
            self._plan_progress.sync(
                plan_state,
                iteration=int(self.metrics.get("iterations", 0) or 0),
                workspace_revision=self.workspace_revision,
            )
        else:
            plan_state = PlanState.empty()

        # Tool availability describes the agent's capabilities, not an
        # enforced execution sequence. The LLM may inspect, edit, execute, or
        # observe processes whenever the task requires it. Runtime checks in
        # _classify_calls() remain the safety boundary for destructive,
        # duplicate, invalid, or otherwise unsafe actions.
        effective_tool_definitions = self.tool_definitions

        available_tool_names = {
            str(definition.get("function", {}).get("name", "")).strip().lower()
            for definition in effective_tool_definitions
            if isinstance(definition, dict)
            and isinstance(definition.get("function"), dict)
            and str(definition.get("function", {}).get("name", "")).strip()
        }

        working_context = self.working_set.context()
        # Expose plan progress only after this run has explicitly activated
        # planning. This prevents an old .daena/plan.md from steering a fresh
        # task into an unrelated plan completion/update.
        working_context["plan_progress"] = (
            self._plan_progress.context()
            if self._plan_active_this_run
            else {}
        )
        working_context["execution_phase"] = self._phase
        working_context["available_tools"] = sorted(available_tool_names)
        working_context["workspace_guidance"] = (
            "Workspace inventory is authoritative. Do not guess filenames. "
            "Use the tool whose documented capability best fits the next unknown: "
            "web_search discovers current information and likely URLs; web_fetch reads a "
            "known public URL; command_exec runs local commands/scripts and tests. "
            "A 404 or empty extraction only describes that URL/response, not the whole site. "
            "After a failed or uninformative attempt, use its evidence to choose a materially "
            "different approach instead of cosmetic variants of the same command."
        )
        working_context["tool_choice"] = {
            "read_file": "read a known file or a narrow line range with line numbers",
            "grep": "locate symbols/usages and return file:line:snippet matches",
            "glob": "discover files and paths by pattern",
            "list_dir": "inspect one workspace directory",
            "explore": "delegate broad read-only repository exploration to a separate context",
            "write_file": "write complete text files and generated artifacts; set overwrite=true only for intentional replacement",
            "edit_file": "replace an exact string in an existing file when a small targeted edit is needed",
            "context_search": "search prior session history and bounded tool-result evidence when relevant context is missing",
            "apply_patch": "make targeted edits to existing source files using exact context",
            "command_exec": "run local commands/scripts/tests; inspect output, exit status, and HTTP status",
            "web_search": "discover current information, find a correct URL, and identify relevant sources",
            "web_fetch": "retrieve a known public URL; inspect title, status, text, links, and empty-content notes",
        }

        capability_names = {
            "web_search", "web_fetch", "command_exec", "explore", "grep", "edit_file",
            "glob", "read_file", "list_dir", "process_poll", "process_write",
            "context_search",
        }
        working_context["tool_capabilities"] = [
            {
                "name": name,
                "description": str(definition.get("function", {}).get("description", ""))[:260],
            }
            for definition in effective_tool_definitions
            if isinstance(definition, dict)
            and isinstance(definition.get("function"), dict)
            and (name := str(definition["function"].get("name", "")).strip().lower())
            in capability_names
        ][:12]

        context = self.context.get_context(
            session_id=self.session_id,
            user_task=user_task,
            agent_state=self.agent_state,
            working_set=working_context,
            observation=(self.working_set.observation_context()),
            recent_actions=(self.working_set.recent_actions_context()),
            workspace_directory=(workspace_directory),
            recent_limit=self.recent_context_limit,
            search_top_k=self.search_context_top_k,
            available_tool_names=available_tool_names,
            include_plan=self._plan_active_this_run,
        )

        context_builder = self.context.contextbuilder
        estimated_context_tokens = context_builder.tokenbudget.estimate_messages_tokens(
            context
        )
        tool_message_count = sum(
            1 for message in context if message.get("role") == "tool"
        )
        read_message_count = sum(
            1
            for message in context
            if (
                message.get("role") == "tool"
                and message.get("tool_name") == "read_file"
            )
        )
        self.metrics["context_estimated_tokens"] = estimated_context_tokens
        self.metrics["context_messages"] = len(context)
        self.metrics["context_tool_results"] = tool_message_count
        self.metrics["context_read_results"] = read_message_count
        self.metrics["same_revision_read_count"] = self._same_revision_read_count
        self.metrics["observation_action_count"] = self._observation_action_count
        self.metrics["execution_phase"] = self._phase

        self._emit_event(
            "context",
            estimated_tokens=estimated_context_tokens,
            budget=int(getattr(context_builder.tokenbudget, "budget", 0) or 0),
            max_prompt_tokens=int(
                (self.config.get("context") or {}).get("max_prompt_tokens", 0) or 0
            ),
            messages=len(context),
            tool_results=tool_message_count,
            read_results=read_message_count,
            same_revision_reads=self._same_revision_read_count,
        )

        self.logger.info(
            "Context | "
            f"estimated_tokens={estimated_context_tokens} | "
            f"messages={len(context)} | "
            f"tool_results={tool_message_count} | "
            f"read_results={read_message_count} | "
            f"budget={context_builder.tokenbudget.budget} | "
            f"same_revision_reads={self._same_revision_read_count}"
        )

        self.metrics["llm_calls"] = self.metrics.get("llm_calls", 0) + 1

        try:

            # Keep the event stream optional so small/test LLM adapters that
            # implement the older generate(messages, tools=...) contract keep
            # working. The UI-enabled LlmProvider accepts on_event.
            try:
                result = self.llm.generate(
                    context,
                    tools=effective_tool_definitions,
                    on_event=self._forward_llm_stream_event,
                    stop_event=self._stop_event,
                )
            except TypeError as type_error:
                if "unexpected keyword argument 'on_event'" not in str(type_error):
                    raise
                result = self.llm.generate(
                    context,
                    tools=effective_tool_definitions,
                )

        except InterruptedError as exc:
            self.agent_state.stop(str(exc) or "Interrupted by user.")
            self._emit_event(
                "run_stopped",
                reason=str(exc) or "Interrupted by user.",
            )
            return self._stopped_result(
                str(exc) or "Interrupted by user."
            )

        except Exception as exc:

            self.metrics["generation_failures"] = (
                self.metrics.get("generation_failures", 0) + 1
            )
            self.logger.error(f"LLM generation failed: {exc}")

            self.agent_state.fail(str(exc))

            return None

        if result is None:

            self.metrics["generation_failures"] = (
                self.metrics.get("generation_failures", 0) + 1
            )
            self.agent_state.fail("LLM returned None.")

            return None

        self.metrics["tokens"] = (
            self.metrics.get("tokens", 0)
            + max(0, int(getattr(result, "usage", 0) or 0))
        )

        return result

    def set_workspace(
        self,
        workspace_directory: str,
    ) -> None:

        self._workspace_root = Path(workspace_directory).expanduser().resolve()
        self.tool.set_workspace(workspace_directory)
        self.working_set.set_workspace(str(self._workspace_root))
        self.working_set.refresh_workspace()

    def get_metrics(self) -> dict[str, Any]:
        return dict(self.metrics)

    def _duration_ms(self) -> float:
        if self._run_started_at is None:
            return 0.0
        return round((time.perf_counter() - self._run_started_at) * 1000.0, 2)

    def _reset_run_state(
        self,
    ) -> None:

        self.agent_state.reset()

        self.working_set.reset()

        self.workspace_revision = 0
        self._workspace_mutation_epoch = 0
        self._successful_observation_signatures.clear()
        self._successful_mutation_signatures.clear()
        self._failed_command_signatures.clear()
        self._identical_column_advisories_seen.clear()

        self._context_step = 0

        self._same_revision_read_count = 0
        self._observation_action_count = 0
        self._observation_cache.clear()
        self._consecutive_observation_cache_hits = 0
        self._phase = "explore"

        self._plan_progress.reset()
        self._plan_active_this_run = False
        self._active_process_ids.clear()

        self._generation_retries = 0
        self._no_action_turns = 0
        self._completion_review_retries = 0
        self._tool_turns_since_recovery = 0
        self._recovery_review_count = 0
        self._run_started_at = None
        self.metrics = {
            "iterations": 0,
            "llm_calls": 0,
            "tokens": 0,
            "completion_review_calls": 0,
            "completion_review_retries": 0,
            "completion_review_failures": 0,
            "recovery_review_calls": 0,
            "recovery_review_count": 0,
            "recovery_review_failures": 0,
            "context_estimated_tokens": 0,
            "context_messages": 0,
            "context_tool_results": 0,
            "context_read_results": 0,
            "generation_failures": 0,
            "same_revision_read_count": 0,
            "observation_action_count": 0,
            "execution_phase": self._phase,
            "tool_call_attempts": 0,
            "tool_successes": 0,
            "tool_failures": 0,
            "duplicate_observation_blocks": 0,
            "duplicate_mutation_blocks": 0,
            "repeated_command_failure_blocks": 0,
            "active_processes": 0,
            "plan_required": False,
            "completed": False,
            "stop_reason": "",
            "duration_ms": 0.0,
        }

    def close(
        self,
    ) -> None:

        self.tool.close()
        self.stm.close()
