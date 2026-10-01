# src/agent/agentloop.py
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any
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

    # Read/search are observations rather than mutations. They may legitimately
    # be repeated while the workspace revision is unchanged, but an identical
    # observation should not become an infinite loop.
    OBSERVATION_REPEAT_LIMIT = 3

    # How many times a single iteration may be retried in place after a
    # generation failure (provider exception, e.g. Ollama's own tool-call
    # parser choking on malformed output) or a dispatch failure (the model
    # returned tool_calls our ToolDispatcher couldn't make sense of).
    # Without this, a single hiccup used to kill the entire run instantly,
    # regardless of how much progress had already been made.
    MAX_GENERATION_RETRIES = 3

    DEFAULT_MAX_ITERATIONS = 100

    RECENT_CONTEXT_LIMIT = 100
    SEARCH_CONTEXT_TOP_K = 5

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

        self._tool_loop_guard = ToolLoopGuard()

        self._recent_failure_signatures: list[str] = []
        # Runtime-enforced recovery state. The model is not allowed to repeat
        # the exact failed action at the same workspace revision without
        # producing new evidence first.
        self._failed_call_keys: dict[str, tuple[int, int]] = {}
        self._active_process_ids: set[str] = set()
        self._recovery_mode = False

        self._generation_retries = 0

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
    def _failure_signature(
        call,
        result: ToolResult,
    ) -> str:

        error_msg = ""

        content = result.content

        if isinstance(
            content,
            dict,
        ):

            error = content.get("error")

            if isinstance(
                error,
                dict,
            ):

                error_msg = str(
                    error.get(
                        "message",
                        "",
                    )
                )

            elif isinstance(
                error,
                str,
            ):

                error_msg = error

        if not error_msg:

            error_msg = result.summary or "unknown"

        error_msg = re.sub(
            r"0x[0-9a-fA-F]+",
            "0xADDR",
            error_msg,
        )

        error_msg = re.sub(
            r"\d+",
            "N",
            error_msg,
        )

        return f"{getattr(call, 'name', 'unknown')}" f"::{error_msg[:150]}"

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

        if arguments.get("operation") == "create":
            return "create"

        if arguments.get("operation") != "update":
            return None

        status = arguments.get("status")

        if status in {"in_progress", "completed", "blocked"}:
            return str(status)

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

    def _plan_gate_message(
        self,
        call,
        state: PlanState,
    ) -> tuple[str, str] | None:
        """
        Return an execution-order violation for a non-plan call, or None.

        Planning itself remains model-directed. Once a plan exists, however,
        the runtime enforces the step lifecycle and prevents work from jumping
        across plan boundaries.
        """
        if self._is_plan_call(call):
            return None

        if state.error:
            return (
                "plan_invalid",
                "The current plan is invalid. Repair the plan before continuing.",
            )

        if not state.exists:
            return None

        # A completed plan is a planning milestone, not a runtime shutdown signal.
        # The agent must still be able to run verification, inspect results, perform
        # cleanup, or make other final workspace changes after the last plan step.
        # A missing active step is a recoverable plan-state inconsistency.
        # Non-plan work remains available so the model can inspect/repair state
        # instead of being trapped behind a hard gate.
        return None

    def _final_response_gate(
        self,
    ) -> tuple[str, str] | None:
        """Prevent a natural-language final answer while the plan is unfinished."""
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

        arguments = getattr(call, "args", {}) or {}
        if not isinstance(arguments, dict):
            return (
                "invalid_plan_update",
                "Plan update arguments must be an object.",
            )

        if state.error and transition != "create":
            return (
                "plan_invalid",
                "The current plan is invalid. Repair or replace it before continuing.",
            )

        if transition == "create":
            if state.exists:
                return (
                    "plan_exists",
                    "A plan already exists. Use update instead.",
                )
            return None

        step_number = arguments.get("step")
        if type(step_number) is not int or step_number < 1:
            return (
                "invalid_step",
                "Plan status updates require a valid 1-based step number.",
            )

        current = state.current_step

        if transition == "in_progress":
            # Starting the already-active step is idempotent. Some models still
            # emit this call even though plan creation auto-started step 1.
            # Treat it as a harmless no-op instead of spending an iteration on
            # a predictable runtime error.
            if current is not None:
                if step_number == current.number:
                    return None
                return (
                    "active_step_exists",
                    f"Step {current.number} is already in_progress. Complete or block it before starting another step.",
                )

            expected = state.next_pending_step
            if expected is not None and step_number != expected.number:
                return (
                    "wrong_step_order",
                    f"Step {step_number} cannot start yet. Start step {expected.number} next.",
                )

            return None

        if current is None:
            return (
                "no_active_step",
                "There is no in_progress step to finalize.",
            )

        if step_number != current.number:
            return (
                "not_current_step",
                f"Step {step_number} is not the current in_progress step. Current step is {current.number}.",
            )

        if transition == "completed":
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

        plan_state = self._read_plan_state()

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

    def _runtime_recovery_gate(self, call) -> ToolResult | None:
        name = str(getattr(call, "name", "")).strip().lower()

        # A plan transition cannot outrun an active managed process.
        if name == "plan" and self._active_process_ids:
            return ToolResult(
                success=False,
                name=name,
                content={
                    "success": False,
                    "error": {
                        "type": "active_process",
                        "message": (
                            "A managed process is still running. "
                            "Poll or stop it before updating/completing the plan."
                        ),
                    },
                },
                metadata={"runtime_gate": True, "active_process_ids": sorted(self._active_process_ids)},
                summary="PLAN BLOCKED: managed process still running.",
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
            key = self._tool_call_key(call)
            previous_revision, previous_count = self._failed_call_keys.get(
                key,
                (self.workspace_revision, 0),
            )
            count = previous_count + 1 if previous_revision == self.workspace_revision else 1
            self._failed_call_keys[key] = (self.workspace_revision, count)
            self._recovery_mode = True

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
            isinstance(
                result.metadata,
                dict,
            )
            and result.metadata.get(
                "duplicate_action",
                False,
            )
        )

        status = self._tool_result_status(result)

        if status == "running":
            # A managed process is a state transition, not a command failure.
            return None

        if isinstance(result.metadata, dict) and result.metadata.get("runtime_gate"):
            return None

        if result.success:

            self._recent_failure_signatures.clear()

            return None

        if is_duplicate:

            return None

        if isinstance(result.metadata, dict) and result.metadata.get("loop_guard_block"):
            return None

        signature = self._failure_signature(
            call,
            result,
        )

        self._recent_failure_signatures.append(signature)

        if len(self._recent_failure_signatures) > self.FAILURE_STUCK_THRESHOLD:

            self._recent_failure_signatures.pop(0)

        if (
            len(self._recent_failure_signatures) >= self.FAILURE_STUCK_THRESHOLD
            and len(set(self._recent_failure_signatures)) == 1
        ):

            return (
                f"STUCK: "
                f"'{call.name}' failed "
                f"{self.FAILURE_STUCK_THRESHOLD} "
                "times in a row."
            )

        return None

    def _execute_tool_calls(
        self,
        parsed_calls: list,
        iteration: int,
    ) -> bool:

        if not parsed_calls:

            return False

        self.metrics["tool_call_attempts"] = (
            self.metrics.get("tool_call_attempts", 0) + len(parsed_calls)
        )

        runtime_blocked: dict[int, ToolResult] = {}
        runtime_allowed: list = []

        for index, call in enumerate(parsed_calls):
            gated = self._runtime_recovery_gate(call)
            if gated is not None:
                runtime_blocked[index] = gated
            else:
                runtime_allowed.append(call)

        runtime_index_map = {
            new_index: original_index
            for new_index, original_index in enumerate(
                index for index in range(len(parsed_calls))
                if index not in runtime_blocked
            )
        }

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
        workspace_directory: str = "EvanaEval",
    ) -> LLMResult | None:

        if not isinstance(
            user_task,
            ContextEvent,
        ):

            raise TypeError("user_task must be " "a ContextEvent.")

        # A new run starts a fresh runtime ownership boundary. Any process
        # left from a previous run is no longer needed by this task.
        self.tool.close()

        self._reset_run_state()
        self._run_started_at = time.perf_counter()
        self.set_workspace(workspace_directory)

        # Must run AFTER _reset_run_state()
        # and BEFORE the first _next_step().
        self._resume_step_counter()

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

            self.logger.info(
                f"Iteration " f"{iteration_number}/" f"{self.max_iterations}"
            )

            llmresult = self._generate_next_action(
                user_task=user_task,
                workspace_directory=(workspace_directory),
            )

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

        plan_state = self._read_plan_state()
        self._plan_progress.sync(
            plan_state,
            iteration=int(self.metrics.get("iterations", 0) or 0),
            workspace_revision=self.workspace_revision,
        )

        working_context = self.working_set.context()
        working_context["plan_progress"] = self._plan_progress.context()

        context = self.context.get_context(
            session_id=self.session_id,
            user_task=user_task,
            agent_state=self.agent_state,
            working_set=working_context,
            observation=(self.working_set.observation_context()),
            recent_actions=(self.working_set.recent_actions_context()),
            workspace_directory=(workspace_directory),
            recent_limit=(self.RECENT_CONTEXT_LIMIT),
            search_top_k=(self.SEARCH_CONTEXT_TOP_K),
        )

        self.metrics["llm_calls"] = self.metrics.get("llm_calls", 0) + 1

        try:

            result = self.llm.generate(
                context,
                tools=self.tool_definitions,
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

        self._tool_loop_guard.reset()

        self._plan_progress.reset()

        self._last_duplicate_key = None

        self._duplicate_block_streak = 0

        self._recent_failure_signatures.clear()
        self._failed_call_keys.clear()
        self._active_process_ids.clear()
        self._recovery_mode = False

        self._generation_retries = 0
        self._run_started_at = None
        self.metrics = {
            "iterations": 0,
            "llm_calls": 0,
            "tokens": 0,
            "generation_failures": 0,
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
