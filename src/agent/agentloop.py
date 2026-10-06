# src/agent/agentloop.py
from __future__ import annotations

import json
import queue
import re
import time
from pathlib import Path
from typing import Any, Callable
from uuid import UUID, uuid4

from src.agent.agentstate import AgentState
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
from src.agent.toolguard import ToolLoopGuard
from src.tools.ToolManager import ToolManager
from src.utils.logger import get_logger


class Loop:

    FAILURE_STUCK_THRESHOLD = 4
    DUPLICATE_BLOCK_THRESHOLD = 3
    SAME_FAILURE_REPEAT_LIMIT = 1
    EMPTY_RESPONSE_THRESHOLD = 3

    # File reads are observations rather than mutations. They may legitimately
    # be repeated while the workspace revision is unchanged, but an identical
    # observation should not become an infinite loop.
    OBSERVATION_REPEAT_LIMIT = 3

    # After repeated failures with the same tool and normalized error, block
    # the next matching failure pattern so the model must change strategy.
    SEMANTIC_FAILURE_REPEAT_LIMIT = 2

    # Once a foreground managed process is running, the runtime owns the
    # execution boundary until that process is observed, fed, or stopped.
    # Read/edit/other work must not run around a still-live command.
    PROCESS_CONTROL_TOOLS = frozenset({"process_poll", "process_write", "process_stop"})

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

        self.stm = STM(db_path="data/stm.db")

        self.session_id: UUID | None = None

        self._context_step = 0
        self._workspace_root: Path | None = None

        self.tool = ToolManager(config=self.config)

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

        self._successful_tool_calls: dict[
            str,
            int,
        ] = {}

        self._last_duplicate_key: str | None = None

        self._duplicate_block_streak = 0

        # key -> (workspace_revision, successful_repeat_count)
        self._same_revision_call_counts: dict[str, tuple[int, int]] = {}
        self._same_revision_read_count = 0
        self._observation_action_count = 0
        self._phase = "explore"
        self._verification_required = False

        self._tool_loop_guard = ToolLoopGuard()

        # Per-tool semantic failure evidence. Unrelated successful tools must
        # not erase a different tool's recovery history.
        self._semantic_failure_counts: dict[str, int] = {}
        # Runtime-enforced recovery state. The model is not allowed to repeat
        # the exact failed action at the same workspace revision without
        # producing new evidence first.
        self._failed_call_keys: dict[str, tuple[int, int]] = {}
        self._active_process_ids: set[str] = set()
        self._recovery_mode = False

        self._generation_retries = 0

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

        if name == "read_file":
            return True

        if name == "command_exec":
            arguments = getattr(call, "args", {}) or {}
            return cls._is_observation_command(arguments.get("command"))

        return False

    @staticmethod
    def _is_verification_call(call, result: ToolResult) -> bool:
        if str(getattr(call, "name", "")).strip().lower() != "command_exec":
            return False

        content = result.content if isinstance(result.content, dict) else {}
        command = content.get("command")
        if not isinstance(command, list):
            return False

        lowered = [
            str(item).strip().lower()
            for item in command
            if str(item).strip()
        ]
        markers = {
            "test", "tests", "pytest", "jest", "vitest", "mocha",
            "check", "lint", "build", "typecheck", "verify",
        }
        return any(
            marker in token
            for token in lowered
            for marker in markers
        )

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

    def _tool_call_key(
        self,
        call,
    ) -> str:

        name = str(
            getattr(
                call,
                "name",
                "",
            )
        ).strip().lower()

        arguments = getattr(call, "args", {}) or {}

        tool = self.tool.get_tool(name)

        duplicate_key = getattr(tool, "duplicate_key", None) if tool else None

        if callable(duplicate_key):
            try:
                arguments = duplicate_key(
                    arguments,
                    workspace_root=(
                        str(self._workspace_root)
                        if self._workspace_root is not None
                        else None
                    ),
                )
            except Exception as exc:
                self.logger.debug(
                    f"Duplicate-key normalization failed for '{name}': {exc}"
                )

        payload = {
            "name": name,
            "args": arguments,
        }

        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    @staticmethod
    def _canonical_failure_message(message: str) -> str:
        """Normalize equivalent validation errors to one semantic cause."""
        text = re.sub(r"\s+", " ", str(message or "")).strip().lower()
        text = re.sub(r"0x[0-9a-f]+", "0xaddr", text)
        text = re.sub(r"\d+", "N", text)

        match = re.search(
            r"missing required argument(?:\(s\))?:\s*([^.;]+)",
            text,
        )
        if match:
            names = [item.strip() for item in match.group(1).split(",") if item.strip()]
            if names:
                return "missing_required:" + ",".join(sorted(set(names)))

        match = re.search(r"\b([a-z_][a-z0-9_]*)\s+is required\b", text)
        if match:
            return "missing_required:" + match.group(1)

        match = re.search(
            r"unknown argument(?:\(s\))?:\s*([^.;]+)",
            text,
        )
        if match:
            names = [item.strip() for item in match.group(1).split(",") if item.strip()]
            if names:
                return "unknown_argument:" + ",".join(sorted(set(names)))

        return text[:180]

    def _predict_failure_signature(self, call) -> str | None:
        """Predict deterministic argument/schema failures before execution."""
        name = str(getattr(call, "name", "")).strip().lower()
        if not name:
            return None

        validation_error = str(getattr(call, "validation_error", "") or "").strip()
        if validation_error:
            return f"{name}::{self._canonical_failure_message(validation_error)}"

        arguments = getattr(call, "args", {}) or {}
        if not isinstance(arguments, dict):
            return None

        try:
            tool = self.tool.get_tool(name)
            schema = getattr(tool, "parameters", {}) or {}
        except Exception:
            return None

        required = schema.get("required", []) if isinstance(schema, dict) else []
        if not isinstance(required, list):
            return None

        missing = [str(key) for key in required if str(key) not in arguments]
        empty = [
            str(key)
            for key in required
            if str(key) in arguments
            and isinstance(arguments.get(key), str)
            and not arguments.get(key).strip()
        ]
        fields = sorted(set(missing + empty))
        if fields:
            return f"{name}::missing_required:{','.join(fields)}"

        return None

    @classmethod
    def _failure_signature(
        cls,
        call,
        result: ToolResult,
    ) -> str:
        error_msg = ""
        content = result.content

        if isinstance(content, dict):
            error = content.get("error")
            if isinstance(error, dict):
                error_msg = str(error.get("message", "") or "")
            elif isinstance(error, str):
                error_msg = error

        if not error_msg:
            error_msg = result.summary or "unknown"

        name = str(getattr(call, "name", "unknown")).strip().lower() or "unknown"
        return f"{name}::{cls._canonical_failure_message(error_msg)}"

    @staticmethod
    def _duplicate_result(
        call,
        reason: str,
    ) -> ToolResult:

        name = str(
            getattr(
                call,
                "name",
                "unknown",
            )
        )

        summary = f"DUPLICATE BLOCKED: " f"'{name}'. " f"{reason}"

        return ToolResult(
            success=False,
            name=name,
            content={
                "success": False,
                "error": {
                    "type": "duplicate_action",
                    "message": summary,
                },
            },
            metadata={
                "duplicate_action": True,
                "recovery_hint": (
                    "Do not repeat the blocked call unchanged. "
                    "Choose a different tool or change the arguments based on the "
                    "last tool result."
                ),
            },
            summary=summary,
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
    def _loop_guard_result(
        call,
        code: str,
        message: str,
        count: int = 0,
    ) -> ToolResult:
        name = str(getattr(call, "name", "unknown")).strip() or "unknown"
        return ToolResult(
            success=False,
            name=name,
            content={
                "success": False,
                "error": {
                    "type": code,
                    "message": message,
                },
            },
            metadata={
                "loop_guard_block": True,
                "count": count,
                "recovery_hint": (
                    "Do not repeat the blocked action. Inspect the latest "
                    "verification result and choose a different corrective action."
                ),
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

    @staticmethod
    def _plan_transition(
        call,
    ) -> str | None:
        if str(getattr(call, "name", "")).strip().lower() != "plan":
            return None

        arguments = getattr(call, "args", {}) or {}
        if not isinstance(arguments, dict):
            return None

        action = arguments.get("action")
        if action == "create":
            return "create"
        if action == "complete":
            return "completed"
        if action == "block":
            return "blocked"
        if action == "add":
            return "add"

        return None

    @staticmethod
    def _plan_gate_result(
        call,
        error_type: str,
        message: str,
    ) -> ToolResult:
        name = str(getattr(call, "name", "unknown")).strip() or "unknown"

        return ToolResult(
            success=False,
            name=name,
            content={
                "success": False,
                "error": {
                    "type": error_type,
                    "message": message,
                },
            },
            metadata={
                "plan_gate": True,
            },
            summary=message,
        )

    def _is_plan_file_observation(self, call) -> bool:
        name = str(getattr(call, "name", "")).strip().lower()
        if name != "read_file":
            return False

        arguments = getattr(call, "args", {}) or {}
        if not isinstance(arguments, dict):
            return False

        candidates = (
            arguments.get("file_path"),
            arguments.get("path"),
            arguments.get("query"),
        )
        return any(
            ".daena/plan.md" in str(value).replace("\\", "/").lower()
            for value in candidates
            if value is not None
        )

    def _plan_gate_message(
        self,
        call,
        state: PlanState,
    ) -> tuple[str, str] | None:
        """
        Return an execution-order violation for a non-plan call, or None.

        Planning remains model-directed, but the persisted plan is internal
        runtime state and should be consumed through <plan>, not rediscovered
        with filesystem tools.
        """
        if self._is_plan_call(call):
            return None

        if self._is_plan_file_observation(call):
            return (
                "plan_internal_state",
                "Do not read .daena/plan.md with filesystem tools. The current plan is already provided in <plan>; use the plan tool to change it.",
            )

        if state.error:
            return (
                "plan_invalid",
                "The current plan is invalid. Repair the plan before continuing.",
            )

        if not state.exists:
            return None

        return None

    def _final_response_gate(
        self,
    ) -> tuple[str, str] | None:
        """Prevent a natural-language final answer while this run's plan is unfinished."""
        # A plan from an older user task is stale execution state. It must not
        # force an unrelated task into the old plan lifecycle.
        if not self._plan_active_this_run:
            return None

        state = self._read_plan_state()

        if state.error:
            return (
                "plan_invalid",
                "The execution plan is invalid. Repair the plan before giving the final answer.",
            )

        if not state.exists or state.is_complete:
            return None

        current = state.current_step
        if current is not None:
            return (
                "plan_incomplete",
                (
                    "The execution plan is not complete. Finish the active plan "
                    f"step {current.number} ({current.description!r}) before giving "
                    "the final answer. Update the plan when the step is actually "
                    "complete."
                ),
            )

        return (
            "plan_incomplete",
            "The execution plan is not complete. Continue the plan before giving the final answer.",
        )

    def _validate_plan_transition(
        self,
        call,
        state: PlanState,
    ) -> tuple[str, str] | None:
        transition = self._plan_transition(call)
        if transition is None:
            return None

        if state.error and transition != "create":
            return (
                "plan_invalid",
                "The current plan is invalid. Repair the plan before continuing.",
            )

        if transition == "create":
            if state.exists:
                return (
                    "plan_exists",
                    "A plan already exists. Continue using the current plan.",
                )
            return None

        if transition == "add":
            if not state.exists:
                return (
                    "plan_missing",
                    "No plan exists. Create the plan before adding steps.",
                )
            return None

        current = state.current_step
        if current is None:
            return (
                "no_active_step",
                "There is no in_progress step. Use the plan action that matches the current <plan> state.",
            )

        if transition == "completed":
            step_number = current.number
            if not self._plan_progress.can_complete(step_number):
                progress = self._plan_progress.context()
                if progress.get("last_result_success") is False:
                    return (
                        "completion_requires_success",
                        (
                            f"Step {step_number} cannot be completed yet. "
                            "The latest work action failed; recover from that result "
                            "and obtain a successful action before completing the step."
                        ),
                    )

                if progress.get("last_result_terminal") is False:
                    return (
                        "completion_requires_terminal_result",
                        (
                            f"Step {step_number} cannot be completed yet. "
                            "The latest action is still running; observe or finish "
                            "the managed process before completing the step."
                        ),
                    )

                return (
                    "completion_requires_work",
                    (
                        f"Step {step_number} cannot be completed yet. "
                        "Perform and successfully finish work for the active step first."
                    ),
                )

        return None

    def _classify_calls(
        self,
        parsed_calls: list,
    ) -> tuple[
        list[int],
        dict[int, ToolResult],
    ]:

        allowed_indices: list[int] = []
        blocked_results: dict[int, ToolResult] = {}

        current_response_keys: set[str] = set()

        plan_state = (
            self._read_plan_state()
            if self._plan_active_this_run
            else PlanState.empty()
        )

        valid_plan_calls = [
            call
            for call in parsed_calls
            if getattr(call, "valid", False) and self._is_plan_call(call)
        ]

        plan_transition_present = any(
            self._plan_transition(call) is not None
            for call in valid_plan_calls
        )

        plan_create_present = any(
            self._plan_transition(call) == "create"
            for call in valid_plan_calls
        )

        plan_calls_allowed = 0

        for index, call in enumerate(parsed_calls):

            if not getattr(
                call,
                "valid",
                False,
            ):
                blocked_results[index] = self._invalid_result(call)
                continue

            key = self._tool_call_key(call)

            if key in current_response_keys:
                blocked_results[index] = self._duplicate_result(
                    call,
                    (
                        "The same tool "
                        "call appeared "
                        "multiple times "
                        "in this response."
                    ),
                )
                continue

            current_response_keys.add(key)

            guard_decision = self._tool_loop_guard.before_call(call)
            if guard_decision.should_block:
                blocked_results[index] = self._loop_guard_result(
                    call=call,
                    code=guard_decision.code,
                    message=guard_decision.message,
                    count=guard_decision.count,
                )
                continue

            # Plan transitions are stateful even when the filesystem has not
            # changed: completing step 1 and completing step 2 are intentionally
            # the same tool/arguments at the same workspace revision. The plan
            # state itself is the changing evidence, so generic same-revision
            # duplicate detection must not block later plan transitions.
            if not self._is_plan_call(call):
                previous_revision = self._successful_tool_calls.get(key)
                tool = self.tool.get_tool(call.name)
                allow_same_revision_repeat = bool(
                    getattr(tool, "allow_same_revision_repeat", False)
                ) if tool is not None else False

                if (
                    previous_revision is not None
                    and previous_revision == self.workspace_revision
                ):
                    if allow_same_revision_repeat:
                        _, repeat_count = self._same_revision_call_counts.get(
                            key,
                            (self.workspace_revision, 0),
                        )
                        if repeat_count >= self.OBSERVATION_REPEAT_LIMIT:
                            blocked_results[index] = self._duplicate_result(
                                call,
                                (
                                    "This observation has already been performed "
                                    f"{self.OBSERVATION_REPEAT_LIMIT} times at the "
                                    "same workspace revision. Inspect the returned "
                                    "evidence and choose a different action."
                                ),
                            )
                            continue
                    else:
                        blocked_results[index] = self._duplicate_result(
                            call,
                            (
                                "An identical successful call already ran at the "
                                "current workspace revision."
                            ),
                        )
                        continue

            if self._is_plan_call(call):

                if plan_calls_allowed >= 1:
                    blocked_results[index] = self._plan_gate_result(
                        call,
                        "duplicate_plan_call",
                        "Only one plan update is allowed per model response. Continue from the updated plan on the next turn.",
                    )
                    continue

                transition_error = self._validate_plan_transition(
                    call,
                    plan_state,
                )

                if transition_error is not None:
                    error_type, message = transition_error
                    blocked_results[index] = self._plan_gate_result(
                        call,
                        error_type,
                        message,
                    )
                    continue

                allowed_indices.append(index)
                plan_calls_allowed += 1
                continue

            if plan_create_present:
                blocked_results[index] = self._plan_gate_result(
                    call,
                    "plan_required_first",
                    "A plan operation is present in this response. Execute the plan update first; continue other work on the next turn.",
                )
                continue

            gate = self._plan_gate_message(
                call,
                plan_state,
            )

            if gate is not None:
                error_type, message = gate
                blocked_results[index] = self._plan_gate_result(
                    call,
                    error_type,
                    message,
                )
                continue

            if plan_transition_present:
                blocked_results[index] = self._plan_gate_result(
                    call,
                    "plan_update_required_first",
                    "Complete the plan state transition first. Continue other work on the next turn.",
                )
                continue

            allowed_indices.append(index)

        return (
            allowed_indices,
            blocked_results,
        )

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

    def _runtime_recovery_gate(
        self,
        call,
    ) -> ToolResult | None:
        name = str(getattr(call, "name", "")).strip().lower()

        # A foreground managed process owns the execution boundary until it
        # reaches a terminal state. Allow only the tools that can observe or
        # control that process; unrelated work cannot run around it.
        if self._active_process_ids and name not in self.PROCESS_CONTROL_TOOLS:
            if name == "plan":
                message = (
                    "A foreground managed process is still running. "
                    "Poll or stop it before updating/completing the plan."
                )
                summary = "PLAN BLOCKED: managed process still running."
                error_type = "active_process"
            else:
                message = (
                    "A foreground managed process is still running. "
                    "Use process_poll to observe it, process_write to provide "
                    "input when needed, or process_stop to terminate it before "
                    "performing unrelated work."
                )
                summary = "RUNTIME BLOCKED: foreground process requires observation."
                error_type = "active_process_requires_observation"

            return ToolResult(
                success=False,
                name=name,
                content={
                    "success": False,
                    "error": {
                        "type": error_type,
                        "message": message,
                    },
                },
                metadata={
                    "runtime_gate": True,
                    "active_process_ids": sorted(self._active_process_ids),
                    "process_control_tools": sorted(self.PROCESS_CONTROL_TOOLS),
                },
                summary=summary,
            )

        predicted_signature = self._predict_failure_signature(call)
        if predicted_signature:
            repeat_count = self._semantic_failure_counts.get(predicted_signature, 0)
            if repeat_count >= self.SEMANTIC_FAILURE_REPEAT_LIMIT:
                message = (
                    f"'{name}' has already produced the same failure "
                    f"{repeat_count} times in this run. Do not retry the same failing "
                    "strategy. Inspect the concrete error, correct the arguments, "
                    "or choose a different tool."
                )
                return ToolResult(
                    success=False,
                    name=name,
                    content={
                        "success": False,
                        "error": {
                            "type": "semantic_failure_repeat",
                            "message": message,
                        },
                    },
                    metadata={
                        "runtime_gate": True,
                        "recovery_required": True,
                        "failure_signature": predicted_signature,
                        "repeat_count": repeat_count,
                    },
                    summary=f"RECOVERY BLOCKED: {message}",
                )

        # Never repeat the exact failed action unchanged at the same revision.
        key = self._tool_call_key(call)
        failed = self._failed_call_keys.get(key)
        if failed is not None:
            failed_revision, count = failed
            if failed_revision == self.workspace_revision and count >= self.SAME_FAILURE_REPEAT_LIMIT:
                return ToolResult(
                    success=False,
                    name=name,
                    content={
                        "success": False,
                        "error": {
                            "type": "recovery_gate",
                            "message": (
                                "The exact action already failed at this workspace "
                                "revision. Inspect the failure and choose a different "
                                "diagnostic or corrective action before retrying it."
                            ),
                        },
                    },
                    metadata={"runtime_gate": True, "recovery_required": True},
                    summary="RECOVERY BLOCKED: exact failed action repeated.",
                )

        return None

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
                self._recovery_mode = True
        elif process_id and status in {"exited", "terminated", "unknown"}:
            self._active_process_ids.discard(process_id)

        changed = self.working_set.update(
            tool_call=call,
            result=result,
            iteration=iteration,
        )

        self._plan_progress.record(
            tool_call=call,
            result=result,
            iteration=iteration,
        )

        if result.success and changed:

            self.workspace_revision += 1
            self._same_revision_read_count = 0
            self._observation_action_count = 0
            self._phase = "implement"
            self._verification_required = True
            # Workspace progress invalidates the exact-failure recovery gate.
            self._failed_call_keys.clear()
            self._recovery_mode = False

        if result.success:
            self.metrics["tool_successes"] = self.metrics.get("tool_successes", 0) + 1
        else:
            self.metrics["tool_failures"] = self.metrics.get("tool_failures", 0) + 1

        if isinstance(result.metadata, dict) and result.metadata.get("duplicate_action"):
            self.metrics["tool_blocks"] = self.metrics.get("tool_blocks", 0) + 1

        if isinstance(result.metadata, dict) and result.metadata.get("plan_gate"):
            self.metrics["plan_blocks"] = self.metrics.get("plan_blocks", 0) + 1

        if isinstance(result.metadata, dict) and result.metadata.get("runtime_gate"):
            self.metrics["recovery_blocks"] = self.metrics.get("recovery_blocks", 0) + 1

        self.metrics["active_processes"] = len(self._active_process_ids)

        if isinstance(result.metadata, dict) and result.metadata.get("loop_guard_block"):
            self.metrics["loop_guard_blocks"] = (
                self.metrics.get("loop_guard_blocks", 0) + 1
            )

        guard_decision = self._tool_loop_guard.after_call(
            call=call,
            result=result,
            workspace_changed=changed,
        )
        if guard_decision.action == "warn":
            self.metrics["loop_guard_warnings"] = (
                self.metrics.get("loop_guard_warnings", 0) + 1
            )
            self._store_nudge(guard_decision.message)

        if result.success:
            tool_name = str(getattr(call, "name", result.name)).strip().lower()
            if tool_name == "read_file" and not changed:
                self._same_revision_read_count += 1
                if self._verification_required:
                    self._verification_required = False
                    self._phase = "verify"
            if tool_name:
                prefix = f"{tool_name}::"
                self._semantic_failure_counts = {
                    failure_key: count
                    for failure_key, count in self._semantic_failure_counts.items()
                    if not failure_key.startswith(prefix)
                }

            key = self._tool_call_key(call)

            self._successful_tool_calls[key] = self.workspace_revision

            tool = self.tool.get_tool(call.name)
            if bool(
                getattr(
                    tool,
                    "allow_same_revision_repeat",
                    False,
                )
            ):
                previous_revision, previous_count = self._same_revision_call_counts.get(
                    key,
                    (self.workspace_revision, 0),
                )
                if previous_revision == self.workspace_revision:
                    count = previous_count + 1
                else:
                    count = 1
                self._same_revision_call_counts[key] = (
                    self.workspace_revision,
                    count,
                )

        if not result.success and status not in {"running"}:
            self._phase = "recover"
            failure_signature = self._failure_signature(call, result)
            self._semantic_failure_counts[failure_signature] = (
                self._semantic_failure_counts.get(failure_signature, 0) + 1
            )

            key = self._tool_call_key(call)
            previous_revision, previous_count = self._failed_call_keys.get(
                key,
                (self.workspace_revision, 0),
            )
            count = previous_count + 1 if previous_revision == self.workspace_revision else 1
            self._failed_call_keys[key] = (self.workspace_revision, count)
            self._recovery_mode = True

        if self._is_observation_call(call) and str(getattr(call, "name", "")).strip().lower() != "process_poll":
            self._observation_action_count += 1

        if self._is_verification_call(call, result) and result.success:
            self._phase = "verify"
            self._observation_action_count = 0
            self._verification_required = False

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
        if isinstance(result.content, dict):
            error = result.content.get("error")
            if isinstance(error, dict):
                error_type = str(error.get("type", "")).strip().lower()
            elif error:
                error_type = str(error).strip().lower()

        if (
            isinstance(result.metadata, dict)
            and result.metadata.get("runtime_gate")
            and error_type == "active_process_requires_observation"
        ):
            self._store_nudge(
                "A foreground managed process is still running. "
                "Do not perform unrelated work yet. Poll the process, provide "
                "required stdin with process_write, or stop it with process_stop."
            )

        if (
            not result.success
            and error_type in {
                "invalid_tool_call",
                "tool_argument_error",
                "invalid_argument",
            }
        ):
            hint = ""
            if isinstance(result.metadata, dict):
                hint = str(result.metadata.get("recovery_hint") or "").strip()

            if (
                error_type == "invalid_tool_call"
                and self.tool.get_tool(str(result.name)) is None
            ):
                self._store_nudge(
                    "The requested tool is not available in this runtime. "
                    "Use one of the available tools."
                )
            else:
                self._store_nudge(
                    f"Tool '{result.name}' rejected the last call. "
                    f"{result.summary or 'Use valid arguments.'} "
                    f"{hint}".strip()
                )

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

        if isinstance(result.metadata, dict) and result.metadata.get("plan_gate"):
            return

        if self._is_plan_call(call):
            plan_after = self._read_plan_state()
            self._plan_progress.sync(
                plan_after,
                iteration=int(self.metrics.get("iterations", 0) or 0),
                workspace_revision=self.workspace_revision,
            )

    def _check_failure_stuck(
        self,
        call,
        result: ToolResult,
    ) -> str | None:
        is_duplicate = bool(
            isinstance(result.metadata, dict)
            and result.metadata.get("duplicate_action", False)
        )

        status = self._tool_result_status(result)
        if status == "running":
            return None

        if isinstance(result.metadata, dict) and result.metadata.get("runtime_gate"):
            return None

        if result.success or is_duplicate:
            return None

        if isinstance(result.metadata, dict) and result.metadata.get("loop_guard_block"):
            return None

        signature = self._failure_signature(call, result)
        count = self._semantic_failure_counts.get(signature, 0)
        if count >= self.FAILURE_STUCK_THRESHOLD:
            return f"STUCK: '{call.name}' produced the same semantic failure {count} times in this run."

        return None

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

        runtime_blocked: dict[int, ToolResult] = {}

        for index, call in enumerate(parsed_calls):
            gated = self._runtime_recovery_gate(call)
            if gated is not None:
                runtime_blocked[index] = gated

        (
            allowed_indices,
            blocked_results,
        ) = self._classify_calls(parsed_calls)

        blocked_results.update(runtime_blocked)

        # Remove runtime-blocked calls from the dispatcher input by filtering
        # the resulting allowed indices. This preserves the original indices
        # used by the existing result assembly.
        allowed_indices = [
            index for index in allowed_indices
            if index not in runtime_blocked
        ]

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

        for index, call in enumerate(parsed_calls):

            if index in blocked_results:

                result = blocked_results[index]

            elif index in executed:

                call, result = executed[index]

            else:

                result = self._missing_result(call)

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

            stop_reason = self._check_failure_stuck(
                call,
                result,
            )

            if stop_reason:

                self.agent_state.stop(stop_reason)

                return True

        if blocked_results and not allowed_indices:

            index = min(blocked_results)

            key = self._tool_call_key(parsed_calls[index])

            if key == self._last_duplicate_key:

                self._duplicate_block_streak += 1

            else:

                self._duplicate_block_streak = 1

            self._last_duplicate_key = key

        else:

            self._duplicate_block_streak = 0
            self._last_duplicate_key = None

        if self._duplicate_block_streak >= 2:
            self._store_nudge(
                "The last tool action was blocked because it repeated an earlier "
                "action without progress. Do not repeat it unchanged. Diagnose the "
                "last result and choose a different tool or arguments."
            )

        if self._duplicate_block_streak >= self.DUPLICATE_BLOCK_THRESHOLD:

            reason = (
                "Repeated identical tool "
                "action was blocked "
                f"{self.DUPLICATE_BLOCK_THRESHOLD} "
                "times."
            )

            self.agent_state.stop(reason)

            return True

        return False

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

                continue

            if isinstance(llmresult.response, str) and llmresult.response.strip():

                if self._verification_required:
                    self.metrics["verification_gate_blocks"] = (
                        self.metrics.get("verification_gate_blocks", 0) + 1
                    )
                    self._store_nudge(
                        "A successful workspace change has not been verified yet. "
                        "Perform a relevant verification step (read the changed "
                        "artifact or run the appropriate test/check) before reporting "
                        "the task as complete."
                    )
                    continue

                plan_gate = self._final_response_gate()
                if plan_gate is not None:
                    self.metrics["plan_final_blocks"] = (
                        self.metrics.get("plan_final_blocks", 0) + 1
                    )
                    self._store_nudge(plan_gate[1])
                    continue

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

        available_tool_names = {
            str(definition.get("function", {}).get("name", "")).strip().lower()
            for definition in self.tool_definitions
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
        working_context["tool_choice"] = {
            "read_file": "read a known file or a narrow line range",
            "apply_patch": "create or modify files",
            "command_exec": "run tests, builds, or necessary commands",
        }

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
        self.metrics["verification_required"] = self._verification_required

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
                    tools=self.tool_definitions,
                    on_event=self._forward_llm_stream_event,
                    stop_event=self._stop_event,
                )
            except TypeError as type_error:
                if "unexpected keyword argument 'on_event'" not in str(type_error):
                    raise
                result = self.llm.generate(
                    context,
                    tools=self.tool_definitions,
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

        self._context_step = 0

        self._successful_tool_calls.clear()
        self._same_revision_call_counts.clear()
        self._same_revision_read_count = 0
        self._observation_action_count = 0
        self._phase = "explore"
        self._verification_required = False

        self._tool_loop_guard.reset()

        self._plan_progress.reset()
        self._plan_active_this_run = False

        self._last_duplicate_key = None

        self._duplicate_block_streak = 0

        self._semantic_failure_counts.clear()
        self._failed_call_keys.clear()
        self._active_process_ids.clear()
        self._recovery_mode = False

        self._generation_retries = 0
        self._run_started_at = None
        self.metrics = {
            "iterations": 0,
            "llm_calls": 0,
            "tokens": 0,
            "context_estimated_tokens": 0,
            "context_messages": 0,
            "context_tool_results": 0,
            "context_read_results": 0,
            "generation_failures": 0,
            "same_revision_read_count": 0,
            "observation_action_count": 0,
            "execution_phase": self._phase,
            "verification_required": False,
            "verification_gate_blocks": 0,
            "tool_call_attempts": 0,
            "tool_successes": 0,
            "tool_failures": 0,
            "tool_blocks": 0,
            "plan_blocks": 0,
            "plan_final_blocks": 0,
            "loop_guard_warnings": 0,
            "loop_guard_blocks": 0,
            "active_processes": 0,
            "recovery_blocks": 0,
            "completed": False,
            "stop_reason": "",
            "duration_ms": 0.0,
        }

    def close(
        self,
    ) -> None:

        self.tool.close()
        self.stm.close()
