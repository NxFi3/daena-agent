# src.context/contextservice.py
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
    """
    Runtime context orchestration.

    Responsibilities:

        1. Retrieve recent STM events.
        2. Retrieve relevant STM events.
        3. Merge and deduplicate them.
        4. Pass the resulting events to ContextBuilder.

    ContextService does NOT:

        - use embeddings
        - use rerankers
        - call an LLM
        - create long-term memory
        - append the current task to conversation
    """

    def __init__(
        self,
        config,
        llm: LlmProvider,
        stm: STM,
    ) -> None:

        self.config = config
        self.llm = llm
        self.stm = stm

        self.contextbuilder = ContextBuilder(
            self.config,
            self.llm,
        )

        self.context: list[dict[str, Any]] = []

    @staticmethod
    def _task_to_dict(
        event: ContextEvent | None,
    ) -> dict[str, Any]:

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

        for event in recent_events:

            if not isinstance(
                event,
                ContextEvent,
            ):
                continue

            event_id = str(event.id)

            if event_id in seen:
                continue

            seen.add(event_id)
            merged.append(event)

        for event in relevant_events:

            if not isinstance(
                event,
                ContextEvent,
            ):
                continue

            event_id = str(event.id)

            if event_id in seen:
                continue

            seen.add(event_id)
            merged.append(event)

        merged.sort(
            key=lambda event: (
                event.step,
                event.timestamp,
            )
        )

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
        recent_limit: int = 40,
        search_top_k: int = 5,
    ) -> list[dict[str, Any]]:

        recent_events = self.stm.get_recent(
            session_id=session_id,
            limit=recent_limit,
        )

        query = str(user_task.content or "").strip()

        if query:

            relevant_events = self.stm.search(
                session_id=session_id,
                query=query,
                top_k=search_top_k,
            )

        else:

            relevant_events = []

        events = self._merge_events(
            recent_events=recent_events,
            relevant_events=relevant_events,
        )

        self.context = self.contextbuilder.build_context(
            events=events,
            task=self._task_to_dict(user_task),
            agent_state=agent_state.state_context(),
            progress=agent_state.progress_context(),
            working_set=working_set or {},
            observation=observation or {},
            recent_actions=recent_actions or {},
            workspace=workspace_directory,
        )

        return self.context

    def calibrate(
        self,
        llmresult: LLMResult,
    ) -> None:

        self.contextbuilder.tokenbudget.calibrate_from_response(
            self.context,
            llmresult,
        )
