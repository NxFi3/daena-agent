from __future__ import annotations

from typing import Any
from uuid import UUID

from src.agent.agentstate import AgentState
from src.context.contextbuilder import ContextBuilder
from src.engine.LlmProviderManager import LlmProvider
from src.models.ContextEvent import (
    ContextEvent,
    ContextPriority,
    ContextRole,
    ContextType,
)
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
        self._retrieval_cache_key: tuple[str, str, int] | None = None
        self._retrieval_cache: list[ContextEvent] = []

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

    def _store_checkpoint(
        self,
        session_id: UUID | str,
        checkpoint: dict[str, Any] | None,
    ) -> None:
        if not isinstance(checkpoint, dict):
            return

        summary = str(checkpoint.get("summary") or "").strip()
        covered_step = checkpoint.get("covered_through_step")
        try:
            covered_step = int(covered_step)
        except (TypeError, ValueError):
            return

        if not summary or covered_step <= 0:
            return

        existing = self.stm.get_latest_checkpoint(session_id)
        if existing is not None:
            existing_covered = 0
            if isinstance(existing.metadata, dict):
                try:
                    existing_covered = int(existing.metadata.get("covered_through_step", 0) or 0)
                except (TypeError, ValueError):
                    existing_covered = 0
            if existing_covered >= covered_step:
                return

        event = ContextEvent(
            role=ContextRole.SYSTEM,
            type=ContextType.EVENT,
            content=summary,
            priority=ContextPriority.HIGH,
            step=covered_step,
            metadata={
                "checkpoint": True,
                "covered_through_step": covered_step,
            },
        )
        self.stm.add(session_id=session_id, event=event)

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
        checkpoint = self.stm.get_latest_checkpoint(session_id)
        checkpoint_step = 0
        checkpoint_events: list[ContextEvent] = []
        if checkpoint is not None:
            checkpoint_events = [checkpoint]
            if isinstance(checkpoint.metadata, dict):
                try:
                    checkpoint_step = int(
                        checkpoint.metadata.get("covered_through_step", checkpoint.step) or checkpoint.step
                    )
                except (TypeError, ValueError):
                    checkpoint_step = int(checkpoint.step)

        if checkpoint_step > 0:
            recent_events = self.stm.get_recent_after_step(
                session_id=session_id,
                after_step=checkpoint_step,
                limit=recent_limit or self.default_recent_limit,
            )
        else:
            recent_events = self.stm.get_recent(
                session_id=session_id,
                limit=recent_limit or self.default_recent_limit,
            )

        query = str(user_task.content or "").strip()
        top_k = search_top_k or self.default_search_top_k
        cache_key = (
            str(session_id),
            str(user_task.id),
            int(top_k),
            int(checkpoint_step),
        )

        if query and cache_key == self._retrieval_cache_key:
            relevant_events = list(self._retrieval_cache)
        elif query:
            relevant_events = self.stm.search(
                session_id=session_id,
                query=query,
                top_k=top_k,
                min_step=checkpoint_step if checkpoint_step > 0 else None,
            )
            self._retrieval_cache_key = cache_key
            self._retrieval_cache = list(relevant_events or [])
        else:
            relevant_events = []

        events = self._merge_events(
            recent_events,
            (*checkpoint_events, *relevant_events),
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
            available_tool_names=available_tool_names,
            include_plan=include_plan,
        )

        self._store_checkpoint(
            session_id,
            self.contextbuilder.consume_pending_checkpoint(),
        )
        return self.context

    def calibrate(self, llmresult: LLMResult) -> None:
        self.contextbuilder.tokenbudget.calibrate_from_response(
            self.context,
            llmresult,
        )
