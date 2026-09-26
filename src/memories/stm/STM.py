# src.memories/stm.py
from __future__ import annotations

from pathlib import Path
from uuid import UUID

from src.models.ContextEvent import ContextEvent
from src.memories.stm.stmdatabase import STMDatabase


class STM:
    """
    Short-Term Memory.

    STM provides two independent retrieval paths:

    1. Recent context
       - deterministic
       - ordered chronologically
       - used for normal working context

    2. Search
       - lexical FTS5 search
       - BM25 ranking
       - returns the most relevant older events

    STM does not build prompts.
    STM does not perform embeddings.
    STM does not use an LLM.
    """

    def __init__(
        self,
        db_path: str | Path = "data/stm.db",
    ) -> None:

        self.db = STMDatabase(db_path)

    def add(
        self,
        session_id: UUID | str,
        event: ContextEvent,
    ) -> bool:

        return self.db.add(
            session_id=session_id,
            event=event,
        )

    def get_recent(
        self,
        session_id: UUID | str,
        limit: int = 10,
    ) -> list[ContextEvent]:

        return self.db.get_recent(
            session_id=session_id,
            limit=limit,
        )

    def search(
        self,
        session_id: UUID | str,
        query: str,
        top_k: int = 3,
    ) -> list[ContextEvent]:

        return self.db.search(
            session_id=session_id,
            query=query,
            top_k=top_k,
        )

    def delete(
        self,
        event_id: UUID | str,
    ) -> bool:

        return self.db.delete(event_id)

    def clear_session(
        self,
        session_id: UUID | str,
    ) -> int:

        return self.db.clear_session(session_id)

    def count(
        self,
        session_id: UUID | str,
    ) -> int:

        return self.db.count(session_id)

    def close(self) -> None:

        self.db.close()

    def __enter__(self) -> "STM":
        return self

    def __exit__(
        self,
        *_args,
    ) -> None:

        self.close()
