"""Model-directed execution with ordered tool receipts and bounded resources.

The runtime validates protocols and owns cancellation. It does not classify task
intent, infer verification from command names, or veto actions based on repetition.
"""

from __future__ import annotations

import json
import queue
import time
from pathlib import Path
from threading import Event
from uuid import uuid4

from src.agent.agentstate import AgentState
from src.context.contextservice import ContextService
from src.context.workingset import WorkingSet
from src.engine.invoke import invoke\nfrom src.engine.errors import RecoverableGenerationError
from src.memories.stm.STM import STM
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult
from src.models.ToolResult import ToolResult
from src.tools.ToolManager import ToolManager
from src.utils.logger import get_logger


class Loop:
    DEFAULT_MAX_ITERATIONS = 100
    DEFAULT_RECENT_CONTEXT_LIMIT = 160
    DEFAULT_SEARCH_CONTEXT_TOP_K = 3

    def __init__(self, config, llm):
        self.config, self.llm = config, llm
        self.logger = get_logger("[LOOP]")
        self.completion_protocol = (config.get("runtime") or {}).get(
            "completion_protocol", "implicit"
        )
        if self.completion_protocol not in {"implicit", "explicit"}:
            raise ValueError("runtime.completion_protocol must be implicit or explicit")
        self.tool = ToolManager(config)
        if (
            self.completion_protocol == "explicit"
            and "task_finish" not in self.tool.security.policy.allowed_tools
        ):
            raise ValueError(
                "Explicit completion requires task_finish in security.allowed_tools"
            )
        self.stm = STM(
            db_path=(config.get("memory") or {}).get("database", "data/stm.db")
        )
        self.context = ContextService(config, llm, self.stm)
        self.context.contextbuilder.skill_registry = self.tool.skills
        self.session_id = uuid4()
        self._workspace_root = None
        self._context_step = 0
        self._event_sink = None
        self._stop_event = None
        self._run_started_at = None
        self._steering_queue = queue.Queue()
        self.agent_state = AgentState()
        self.working_set = WorkingSet()
        self.recent_context_limit = self.context.default_recent_limit
        self.search_context_top_k = self.context.default_search_top_k
        self.max_iterations = max(1, int(config.get("max_agent_iterations", 100)))
        runtime_config = config.get("runtime") or {}
        self.max_generation_retries = max(
            0, min(5, int(runtime_config.get("generation_retries", 2)))
        )
        self.tool_definitions = self.tool.get_tools()
        self.tool.subagents.event_sink = lambda event: self._emit_event(
            "subagent_event", child_event=event
        )
        self._reset_run_state()

    def _reset_run_state(self):
        self.agent_state.reset()
        self.working_set.reset()
        self._plan_active_this_run = False
        self._finish_request = None
        self._active_process_ids = set()
        self.workspace_revision = 0
        self.metrics = {
            "iterations": 0,
            "llm_calls": 0,
            "tokens": 0,
            "context_estimated_tokens": 0,
            "context_messages": 0,
            "context_tool_results": 0,
            "context_read_results": 0,
            "generation_failures": 0,\n            "generation_retries": 0,
            "tool_call_attempts": 0,
            "tool_successes": 0,
            "tool_failures": 0,
            "resource_blocks": 0,
            "completed": False,
            "outcome": "running",
            "verification_evidence": [],
            "stop_reason": "",
            "duration_ms": 0.0,
        }

    def _next_step(self):
        self._context_step += 1
        return self._context_step

    def _store_event(self, event):
        if self.session_id is None:
            raise RuntimeError("No active session")
        self.stm.add(session_id=self.session_id, event=event)

    def _store_nudge(self, text):
        self._store_event(
            ContextEvent(
                role=ContextRole.USER,
                type=ContextType.MESSAGE,
                content=text,
                step=self._next_step(),
                metadata={"runtime_nudge": True},
            )
        )

    def steer(self, text):
        if str(text).strip():
            self._steering_queue.put(str(text).strip())

    def _consume_steering(self):
        while True:
            try:
                text = self._steering_queue.get_nowait()
            except queue.Empty:
                return
            self._store_event(
                ContextEvent(
                    role=ContextRole.USER,
                    type=ContextType.MESSAGE,
                    content=text,
                    step=self._next_step(),
                    metadata={"operator_steering": True},
                )
            )
            self._emit_event("steering", text=text)

    def _emit_event(self, event_type, **payload):
        if callable(self._event_sink):
            try:
                self._event_sink(
                    {
                        "type": event_type,
                        "timestamp": time.time(),
                        "iteration": self.metrics.get("iterations", 0),
                        **payload,
                    }
                )
            except Exception as exc:
                self.logger.debug("Event sink failed: %s", exc)

    def _forward_llm_stream_event(self, event):
        if isinstance(event, dict) and isinstance(event.get("type"), str):
            self._emit_event(
                event["type"], **{k: v for k, v in event.items() if k != "type"}
            )

    def _assistant_event(self, result, normalized_tool_calls=None):
        message = dict(result.message or {})
        message.update(role="assistant", content=str(result.response or ""))
        message.pop("tool_calls", None)
        if normalized_tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.args, ensure_ascii=False),
                    },
                }
                for call in normalized_tool_calls
            ]
        return ContextEvent(
            role=ContextRole.ASSISTANT,
            type=ContextType.MESSAGE,
            content=message["content"],
            step=self._next_step(),
            metadata={"llm_message": message},
        )

    def _tool_result_event(self, call, result):
        return ContextEvent(
            role=ContextRole.TOOL,
            type=ContextType.TOOL_RESULT,
            step=self._next_step(),
            content=json.dumps(
                {
                    "name": result.name,
                    "tool_call_id": call.id,
                    "success": result.success,
                    "content": result.content,
                    "summary": result.summary,
                    "evidence": result.evidence,
                    "effects": result.effects,
                    "metadata": result.metadata,
                },
                ensure_ascii=False,
                default=str,
            ),
        )

    def _apply_result(self, call, result, iteration):
        self.agent_state.update_from_result(result)
        content = result.content
        process_id = content.get("process_id")
        if process_id:
            if content.get("status") == "running" and not content.get(
                "background", False
            ):
                self._active_process_ids.add(str(process_id))
            else:
                self._active_process_ids.discard(str(process_id))
        if result.success and call.name == "plan":
            self._plan_active_this_run = True
        self.working_set.update(call, result, iteration)
        # File state is observed from disk, not guessed from command text.
        if self.working_set.refresh_workspace():
            self.workspace_revision += 1
        key = "tool_successes" if result.success else "tool_failures"
        self.metrics[key] += 1
        self._store_event(self._tool_result_event(call, result))
        self._emit_event(
            "tool_result",
            name=call.name,
            success=result.success,
            summary=result.summary,
            content=result.content,
        )

    def _execute_tool_calls(self, calls, iteration):
        # One receipt per call, in order, including malformed and cancelled calls.
        # Repeated calls are permitted: observations and retries can be legitimate.
        for call in calls:
            self.metrics["tool_call_attempts"] += 1
            self._emit_event("tool_call", name=call.name, arguments=call.args)
            pending = None
            if call.name == "task_finish":
                pending = (
                    "Call task_finish alone, after observing the results of other tools."
                    if len(calls) != 1
                    else self._final_response_gate()
                )
            if pending:
                result = ToolResult(
                    success=False,
                    name=call.name,
                    content={
                        "error": {"type": "unfinished_resources", "message": pending}
                    },
                )
                self.metrics["resource_blocks"] += 1
            else:
                result = self.tool.execute([call])["results"][0]
            self._apply_result(call, result, iteration)
            if call.name == "task_finish" and result.success:
                self._finish_request = result.content
        return self._stop_event is not None and self._stop_event.is_set()

    def _final_response_gate(self):
        if self.tool.subagents.has_pending():
            return "Collect subagent results and merge or discard worker changes before finishing."
        if self._active_process_ids:
            return (
                "Observe or stop the outstanding processes before finishing: "
                + ", ".join(sorted(self._active_process_ids))
            )
        return None

    def _generate_next_action(self, user_task, workspace_directory):
        self.working_set.refresh_workspace()
        self.tool_definitions = self.tool.get_tools()
        self.context.contextbuilder.tokenbudget.reserve_tools(self.tool_definitions)
        names = {item["function"]["name"] for item in self.tool_definitions}
        state = self.working_set.context()
        # These fields were heuristic task assessments. Only explicit tool
        # receipts and current filesystem inventory are authoritative.
        for key in ("verification", "last_failed_verification", "facts"):
            state.pop(key, None)
        state["available_tools"] = sorted(names)
        state["subagents"] = self.tool.subagents.capabilities()
        state["active_process_ids"] = sorted(self._active_process_ids)
        builder = self.context.contextbuilder
        builder.stop_event = self._stop_event
        messages = self.context.get_context(
            session_id=self.session_id,
            user_task=user_task,
            agent_state=self.agent_state,
            working_set=state,
            observation=self.working_set.observation_context(),
            recent_actions=self.working_set.recent_actions_context(),
            workspace_directory=workspace_directory,
            recent_limit=self.recent_context_limit,
            search_top_k=self.search_context_top_k,
            available_tool_names=names,
            include_plan=True,
        )
        estimated = builder.tokenbudget.estimate_request_tokens(messages)
        self.metrics.update(
            context_estimated_tokens=estimated,
            context_messages=len(messages),
            context_tool_results=sum(m.get("role") == "tool" for m in messages),
            context_read_results=sum(
                m.get("tool_name") == "read_file" for m in messages
            ),
        )
        self._emit_event(
            "context",
            estimated_tokens=estimated,
            budget=builder.tokenbudget.budget,
            messages=len(messages),
            tool_results=self.metrics["context_tool_results"],
            tool_schema_tokens=builder.tokenbudget.tool_tokens,
        )
        self.metrics["llm_calls"] += 1
        result = invoke(
            self.llm,
            messages,
            tools=self.tool_definitions,
            on_event=self._forward_llm_stream_event,
            stop_event=self._stop_event,
        )
        if not isinstance(result, LLMResult):
            raise TypeError("Provider must return LLMResult")
        self.metrics["tokens"] += max(0, int(result.usage or 0))
        self.context.calibrate(result)
        return result

    def _finish(self, result, completed, reason="", outcome=None, evidence=None):
        self.metrics.update(
            completed=completed,
            stop_reason=reason,
            duration_ms=self._duration_ms(),
            outcome=outcome or ("completed" if completed else "stopped"),
            verification_evidence=evidence or [],
        )
        if completed:
            self.agent_state.complete()
            self._emit_event(
                "final_response",
                text=result.response,
                usage=result.usage,
                authoritative=outcome is not None,
                outcome=self.metrics["outcome"],
            )
        else:
            self.agent_state.stop(reason)
            self._emit_event("run_stopped", reason=reason)
            if outcome in {"blocked", "needs_input"}:
                self._emit_event(
                    "final_response",
                    text=result.response,
                    usage=result.usage,
                    authoritative=True,
                    outcome=outcome,
                )
        self._store_event(self._assistant_event(result))
        self._emit_event(
            "run_end",
            completed=completed,
            stop_reason=reason,
            metrics=self.get_metrics(),
        )
        return result

    def _stopped_result(self, reason):
        return self._finish(
            LLMResult(f"Task stopped: {reason}", {}, [], None, 0), False, reason
        )

    def run(
        self, user_task, workspace_directory=None, event_sink=None, stop_event=None
    ):
        if not isinstance(user_task, ContextEvent):
            raise TypeError("user_task must be a ContextEvent")
        self.tool.close()
        self._reset_run_state()
        self._stop_event = stop_event if stop_event is not None else Event()
        self._event_sink = event_sink
        self._run_started_at = time.perf_counter()
        self.tool.stop_event = self._stop_event
        self.tool.subagents.begin_run(self._stop_event)
        self.tool.skills.active.clear()
        directory = workspace_directory or str(Path.cwd())
        self.set_workspace(directory)
        recent = self.stm.get_recent(self.session_id, limit=1)
        self._context_step = recent[-1].step if recent else 0
        user_task.step = self._next_step()
        self._store_event(user_task)
        self._emit_event(
            "run_start",
            session_id=str(self.session_id),
            workspace=directory,
            max_iterations=self.max_iterations,
        )
        try:
            for iteration in range(1, self.max_iterations + 1):
                self.metrics["iterations"] = iteration
                self.agent_state.iteration = iteration
                if self._stop_event.is_set():
                    raise InterruptedError("Interrupted by user")
                self._consume_steering()
                self._emit_event("iteration_start", max_iterations=self.max_iterations)
                result = self._generate_next_action(user_task, directory)
                if self._stop_event.is_set():
                    raise InterruptedError("Interrupted by user")
                if result.tool_calls:
                    calls = self.tool.dispatcher.dispatch(result.tool_calls)
                    self._store_event(self._assistant_event(result, calls))
                    self._execute_tool_calls(calls, iteration)
                    if self._stop_event.is_set():
                        raise InterruptedError("Interrupted by user")
                    if self._finish_request is not None:
                        report = self._finish_request
                        outcome = report["outcome"]
                        return self._finish(
                            LLMResult(report["response"], {}, [], None, result.usage),
                            outcome == "completed",
                            "" if outcome == "completed" else outcome,
                            outcome=outcome,
                            evidence=report["evidence"],
                        )
                elif str(result.response or "").strip():
                    if self.completion_protocol == "explicit":
                        self._store_event(self._assistant_event(result))
                        self._emit_event("assistant_message", text=result.response)
                        self._store_nudge(
                            "Progress recorded. Continue with a tool, or call task_finish for a final answer, blocker, or question."
                        )
                        continue
                    pending = self._final_response_gate()
                    if pending:
                        self.metrics["resource_blocks"] += 1
                        self._store_nudge(pending)
                        continue
                    return self._finish(result, True)
                else:
                    return self._stopped_result(
                        "Provider returned no text or tool calls"
                    )
            return self._stopped_result("Maximum iterations reached")
        except InterruptedError as exc:
            return self._stopped_result(str(exc))
        except Exception as exc:
            self.metrics["generation_failures"] += 1
            return self._stopped_result(f"{type(exc).__name__}: {exc}")
        finally:
            self.tool.close()

    def set_workspace(self, workspace_directory):
        self.tool.set_workspace(workspace_directory)
        self._workspace_root = Path(workspace_directory).expanduser().resolve()
        self.working_set.set_workspace(str(self._workspace_root))
        self.working_set.refresh_workspace()

    def get_metrics(self):
        return dict(self.metrics)

    def _duration_ms(self):
        return (
            round((time.perf_counter() - self._run_started_at) * 1000, 2)
            if self._run_started_at
            else 0
        )

    def close(self):
        self.tool.close()
        self.stm.close()
