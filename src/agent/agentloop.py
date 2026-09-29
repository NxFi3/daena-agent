# src/agent/agentloop.py
from __future__ import annotations

import json
import re
import time
from typing import Any
from uuid import UUID, uuid4

from src.agent.agentstate import AgentState
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

    FAILURE_STUCK_THRESHOLD = 3
    DUPLICATE_BLOCK_THRESHOLD = 5
    EMPTY_RESPONSE_THRESHOLD = 3

    # How many times a single iteration may be retried in place after a
    # generation failure (provider exception, e.g. Ollama's own tool-call
    # parser choking on malformed output) or a dispatch failure (the model
    # returned tool_calls our ToolDispatcher couldn't make sense of).
    # Without this, a single hiccup used to kill the entire run instantly,
    # regardless of how much progress had already been made.
    MAX_GENERATION_RETRIES = 2

    DEFAULT_MAX_ITERATIONS = 100

    RECENT_CONTEXT_LIMIT = 40
    SEARCH_CONTEXT_TOP_K = 3

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

        self.workspace_revision = 0

        self._successful_tool_calls: dict[
            str,
            int,
        ] = {}

        self._last_duplicate_key: str | None = None

        self._duplicate_block_streak = 0

        self._recent_failure_signatures: list[str] = []

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

    @staticmethod
    def _tool_call_key(
        call,
    ) -> str:

        payload = {
            "name": str(
                getattr(
                    call,
                    "name",
                    "",
                )
            )
            .strip()
            .lower(),
            "args": getattr(
                call,
                "args",
                {},
            )
            or {},
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

        return ToolResult(
            success=False,
            name=name,
            content={
                "success": False,
                "error": {
                    "type": "invalid_tool_call",
                    "message": "Invalid tool call.",
                },
            },
            metadata={},
            summary="Invalid tool call.",
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

        return result

    def _classify_calls(
        self,
        parsed_calls: list,
    ) -> tuple[
        list[int],
        dict[int, ToolResult],
    ]:

        allowed_indices: list[int] = []

        blocked_results: dict[
            int,
            ToolResult,
        ] = {}

        current_response_keys: set[str] = set()

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

            previous_revision = self._successful_tool_calls.get(key)

            if (
                previous_revision is not None
                and previous_revision == self.workspace_revision
            ):

                blocked_results[index] = self._duplicate_result(
                    call,
                    (
                        "An identical "
                        "successful call "
                        "already ran at "
                        "the current "
                        "workspace revision."
                    ),
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

    def _apply_result(
        self,
        call,
        result: ToolResult,
        iteration: int,
    ) -> None:

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

        changed = self.working_set.update(
            tool_call=call,
            result=result,
            iteration=iteration,
        )

        if result.success and changed:

            self.workspace_revision += 1

        if result.success:
            self.metrics["tool_successes"] = self.metrics.get("tool_successes", 0) + 1
        else:
            self.metrics["tool_failures"] = self.metrics.get("tool_failures", 0) + 1

        if isinstance(result.metadata, dict) and result.metadata.get("duplicate_action"):
            self.metrics["tool_blocks"] = self.metrics.get("tool_blocks", 0) + 1

        if result.success:

            key = self._tool_call_key(call)

            self._successful_tool_calls[key] = self.workspace_revision

        self._store_event(
            self._tool_result_event(
                call,
                result,
            )
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

        if result.success:

            self._recent_failure_signatures.clear()

            return None

        if is_duplicate:

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

                self._store_event(self._assistant_event(llmresult))

                self.agent_state.complete()
                self.metrics["completed"] = True
                self.metrics["stop_reason"] = ""
                self.metrics["duration_ms"] = self._duration_ms()

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

        context = self.context.get_context(
            session_id=self.session_id,
            user_task=user_task,
            agent_state=self.agent_state,
            working_set=(self.working_set.context()),
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

        self.tool.set_workspace(workspace_directory)

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

        self._last_duplicate_key = None

        self._duplicate_block_streak = 0

        self._recent_failure_signatures.clear()

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
            "completed": False,
            "stop_reason": "",
            "duration_ms": 0.0,
        }

    def close(
        self,
    ) -> None:

        self.stm.close()
