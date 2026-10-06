from __future__ import annotations

from typing import Any
from uuid import UUID

from src.agent.agentstate import AgentState
from src.context.contextbuilder import ContextBuilder
from src.engine.LlmProviderManager import LlmProvider
from src.models.ContextEvent import ContextEvent
from src.models.LLMResult import LLMResult
from src.memories.stm import STM


class ContextService:
    """Retrieve STM context, merge it, then build provider-visible messages."""

    def __init__(self, config, llm: LlmProvider, stm: STM) -> None:
        self.config = config
        self.llm = llm
        self.stm = stm
        self.contextbuilder = ContextBuilder(config, llm)
        self.context: list[dict[str, Any]] = []

        context_config = config.get("context") or {}
        retrieval_config = config.get("retrieval") or {}
        self.default_recent_limit = max(
            1, int(context_config.get("recent_event_limit", 40))
        )
        self.default_search_top_k = max(
            1, int(retrieval_config.get("top_k", 3))
        )

    @staticmethod
    def _task_to_dict(event: ContextEvent | None) -> dict[str, Any]:
        if event is None:
            return {}
        return {
            "id": str(event.id),
            "content": event.content,
            "role": event.role.value,
            "type": event.type.value,
            "priority": event.priority.value,
            "step": event.step,
            "timestamp": event.timestamp.isoformat(),
            "metadata": event.metadata,
        }

    @staticmethod
    def _merge_events(
        recent_events: list[ContextEvent],
        relevant_events: list[ContextEvent],
    ) -> list[ContextEvent]:
        merged: list[ContextEvent] = []
        seen: set[str] = set()

        for event in (*recent_events, *relevant_events):
            if not isinstance(event, ContextEvent):
                continue
            event_id = str(event.id)
            if event_id in seen:
                continue
            seen.add(event_id)
            merged.append(event)

        merged.sort(key=lambda event: (event.step, event.timestamp))
        return merged

    def get_context(
        self,
        session_id: UUID | str,
        user_task: ContextEvent,
        agent_state: AgentState,
        working_set: dict[str, Any] | None = None,
        observation: dict[str, Any] | None = None,
        recent_actions: dict[str, Any] | None = None,
        workspace_directory: str | None = None,
        recent_limit: int | None = None,
        search_top_k: int | None = None,
        available_tool_names: set[str] | None = None,
        include_plan: bool = True,
    ) -> list[dict[str, Any]]:
        recent_events = self.stm.get_recent(
            session_id=session_id,
            limit=recent_limit or self.default_recent_limit,
        )

        query = str(user_task.content or "").strip()
        relevant_events = (
            self.stm.search(
                session_id=session_id,
                query=query,
                top_k=search_top_k or self.default_search_top_k,
            )
            if query
            else []
        )

        events = self._merge_events(recent_events, relevant_events)

        self.context = self.contextbuilder.build_context(
            events=events,
            task=self._task_to_dict(user_task),
            agent_state=agent_state.state_context(),
            progress=agent_state.progress_context(),
            working_set=working_set or {},
            observation=observation or {},
            recent_actions=recent_actions or {},
            workspace=workspace_directory,
            available_tool_names=available_tool_names,
            include_plan=include_plan,
        )
        return self.context

    def calibrate(self, llmresult: LLMResult) -> None:
        self.contextbuilder.tokenbudget.calibrate_from_response(
            self.context,
            llmresult,
        )
