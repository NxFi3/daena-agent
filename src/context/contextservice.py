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
        self.default_search_top_k = max(1, int(retrieval_config.get("top_k", 3)))
        self._retrieval_cache_key: tuple[str, str, int] | None = None
        self._retrieval_cache: list[ContextEvent] = []
        self._session_key = None
        self._session_revision = None

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
        attachments: list[dict[str, Any]] | None = None,
        contributions=None,
    ) -> list[dict[str, Any]]:
        revision_reader = getattr(self.stm, "get_session_revision", None)
        revision = revision_reader(session_id) if callable(revision_reader) else 0
        if self._session_key != str(session_id) or self._session_revision != revision:
            self._session_key = str(session_id)
            self._session_revision = revision
            for attr in (
                "_history_task",
                "_history_summary",
                "_covered_groups",
                "_pending_history",
            ):
                if hasattr(self.contextbuilder, attr):
                    delattr(self.contextbuilder, attr)
            self._retrieval_cache_key = None
            self._saved_checkpoint = None
            loader = getattr(self.stm, "get_checkpoint", None)
            checkpoint = loader(session_id) if callable(loader) else None
            if checkpoint:
                self.contextbuilder._history_summary = checkpoint["summary"]
                self.contextbuilder._covered_groups = set(checkpoint["covered_groups"])
        recent_reader = getattr(self.stm, "get_recent_complete", self.stm.get_recent)
        recent_events = recent_reader(
            session_id=session_id,
            limit=recent_limit or self.default_recent_limit,
        )

        query = str(user_task.content or "").strip()
        top_k = search_top_k or self.default_search_top_k
        cache_key = (str(session_id), str(user_task.id), int(top_k))

        if query and cache_key == self._retrieval_cache_key:
            relevant_events = list(self._retrieval_cache)
        elif query:
            relevant_events = self.stm.search(
                session_id=session_id,
                query=query,
                top_k=top_k,
            )
            self._retrieval_cache_key = cache_key
            self._retrieval_cache = list(relevant_events or [])
        else:
            relevant_events = []

        events = self._merge_events(recent_events, relevant_events)

        # This ledger contains only text from earlier confirmed model requests.
        # It survives recent-event eviction and process restarts independently of
        # an LLM-generated historical summary. It never claims text is retained.
        attachments = list(attachments or [])
        reader = getattr(self.stm, "get_file_reads", None)
        ledger = reader(session_id) if callable(reader) else None
        if ledger and ledger.get("items"):
            from src.context.payloads import fit_json

            budget = self.contextbuilder.tokenbudget
            limit = max(
                256, min(2400, int(budget.budget * budget.chars_per_token * 0.08))
            )
            import json

            working_set = dict(working_set or {})
            working_set["file_read_history"] = json.loads(fit_json(ledger, limit))
            history = working_set["file_read_history"]
            history["shown_versions"] = len(history.get("items", []))
            history["total_versions"] = ledger["total_versions"]
            history["truncated"] = history["shown_versions"] < history["total_versions"]
            working_set["file_read_history"]["meaning"] = (
                "Historical delivered character ranges only; source text may no longer be in this request. Hashes are not revalidated against disk. Recover specific text with read_file and follow next_read."
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
            attachments=attachments,
            contributions=contributions,
        )
        builder = self.contextbuilder
        checkpoint_key = (builder._history_summary, frozenset(builder._covered_groups))
        saver = getattr(self.stm, "save_checkpoint", None)
        if (
            builder._history_summary
            and callable(saver)
            and checkpoint_key != self._saved_checkpoint
        ):
            saver(session_id, builder._history_summary, builder._covered_groups)
            self._saved_checkpoint = checkpoint_key
        return self.context

    def record_delivery(self, session_id):
        from src.context.file_reads import delivered_reads

        saver = getattr(self.stm, "record_file_reads", None)
        if callable(saver):
            saver(session_id, list(delivered_reads(self.context)))

    def calibrate(self, llmresult: LLMResult) -> None:
        self.contextbuilder.tokenbudget.calibrate_from_response(
            self.context,
            llmresult,
        )
