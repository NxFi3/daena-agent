# src.memories/stmdatabase.py
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from src.models.ContextEvent import (
    ContextEvent,
    ContextPriority,
    ContextRole,
    ContextType,
)


class STMDatabase:
    """
    Persistent SQLite storage for Short-Term Memory.

    Responsibilities:
        - Persist ContextEvent objects.
        - Retrieve recent events for a session.
        - Search events using SQLite FTS5/BM25.
        - Delete individual events.
        - Clear complete sessions.

    STMDatabase deliberately does NOT:
        - use embeddings
        - use rerankers
        - call an LLM
        - build prompts
        - perform semantic memory consolidation
    """

    def __init__(
        self,
        db_path: str | Path = "data/stm.db",
    ) -> None:
        self.db_path = Path(db_path)

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
        )

        self.connection.row_factory = sqlite3.Row

        self._configure()
        self._create_tables()

    def _configure(self) -> None:
        self.connection.execute("PRAGMA journal_mode=WAL;")

        self.connection.execute("PRAGMA synchronous=NORMAL;")

        self.connection.execute("PRAGMA foreign_keys=ON;")

    def _create_tables(self) -> None:
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS context_events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,

                role TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                priority TEXT NOT NULL,

                step INTEGER NOT NULL,

                timestamp TEXT NOT NULL,

                metadata TEXT NOT NULL
            );


            CREATE INDEX IF NOT EXISTS
            idx_context_events_session_step

            ON context_events (
                session_id,
                step
            );


            CREATE INDEX IF NOT EXISTS
            idx_context_events_session_timestamp

            ON context_events (
                session_id,
                timestamp
            );


            CREATE VIRTUAL TABLE IF NOT EXISTS
            context_events_fts

            USING fts5 (
                event_id UNINDEXED,
                session_id UNINDEXED,
                content
            );
            """)

        self.connection.commit()

    @staticmethod
    def _serialize_content(
        content: Any,
    ) -> str:
        """
        ContextEvent.content is normally already a string.

        This method still safely handles structured values so
        database persistence never receives an unsupported SQLite
        object.
        """

        if isinstance(content, str):
            return content

        try:
            return json.dumps(
                content,
                ensure_ascii=False,
                default=str,
            )

        except Exception:
            return str(content)

    @staticmethod
    def _serialize_metadata(
        metadata: Any,
    ) -> str:
        """
        Metadata is persisted as JSON.
        """

        if not isinstance(metadata, dict):
            metadata = {}

        try:
            return json.dumps(
                metadata,
                ensure_ascii=False,
                default=str,
            )

        except Exception:
            return "{}"

    @staticmethod
    def _deserialize_metadata(
        metadata: str | None,
    ) -> dict[str, Any]:
        """
        Convert JSON metadata from SQLite back into a dictionary.
        """

        if not metadata:
            return {}

        try:
            value = json.loads(metadata)

            if isinstance(value, dict):
                return value

        except Exception:
            pass

        return {}

    def add(
        self,
        session_id: UUID | str,
        event: ContextEvent,
    ) -> bool:
        """
        Persist a ContextEvent.

        Returns:
            True  -> event inserted
            False -> event already existed or insertion failed
        """

        event_id = str(event.id)
        session_id = str(session_id)

        content = self._serialize_content(event.content)

        metadata = self._serialize_metadata(event.metadata)

        if isinstance(event.timestamp, datetime):
            timestamp = event.timestamp.isoformat()
        else:
            timestamp = str(event.timestamp)

        try:
            cursor = self.connection.execute(
                """
                INSERT OR IGNORE INTO context_events (
                    id,
                    session_id,
                    role,
                    type,
                    content,
                    priority,
                    step,
                    timestamp,
                    metadata
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    event_id,
                    session_id,
                    # Always persist enum values.
                    event.role.value,
                    event.type.value,
                    content,
                    event.priority.value,
                    int(event.step),
                    timestamp,
                    metadata,
                ),
            )

            inserted = cursor.rowcount > 0

            if inserted:
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO context_events_fts (
                        event_id,
                        session_id,
                        content
                    )
                    VALUES (?, ?, ?)
                    """,
                    (
                        event_id,
                        session_id,
                        content,
                    ),
                )

            self.connection.commit()

            return inserted

        except Exception:
            self.connection.rollback()
            raise

    def get_recent(
        self,
        session_id: UUID | str,
        limit: int = 10,
    ) -> list[ContextEvent]:
        """
        Return the newest `limit` events.

        SQL retrieves newest -> oldest.
        The returned list is reversed to chronological order
        because ContextBuilder should receive:

            oldest -> newest
        """

        limit = max(
            1,
            int(limit),
        )

        rows = self.connection.execute(
            """
            SELECT
                id,
                session_id,
                role,
                type,
                content,
                priority,
                step,
                timestamp,
                metadata

            FROM context_events

            WHERE session_id = ?

            ORDER BY
                step DESC,
                timestamp DESC

            LIMIT ?
            """,
            (
                str(session_id),
                limit,
            ),
        ).fetchall()

        rows = list(reversed(rows))

        return [self._row_to_event(row) for row in rows]

    def search(
        self,
        session_id: UUID | str,
        query: str,
        top_k: int = 3,
    ) -> list[ContextEvent]:
        """
        Lexical retrieval using SQLite FTS5 + BM25.

        Search is restricted to one session.

        Important:
            session_id is UNINDEXED in the FTS table, so it is
            filtered through SQL rather than MATCH.
        """

        query = str(query or "").strip()

        if not query:
            return []

        top_k = max(
            1,
            int(top_k),
        )

        terms = [term.strip() for term in query.split() if term.strip()]

        if not terms:
            return []

        safe_terms: list[str] = []

        for term in terms:
            cleaned = term.replace('"', "").replace("'", "")

            if cleaned:
                safe_terms.append(f'"{cleaned}"')

        if not safe_terms:
            return []

        select_sql = """
            SELECT
                e.id,
                e.session_id,
                e.role,
                e.type,
                e.content,
                e.priority,
                e.step,
                e.timestamp,
                e.metadata

            FROM context_events_fts AS f

            JOIN context_events AS e
                ON e.id = f.event_id

            WHERE
                f.session_id = ?
                AND e.type = 'message'
                AND e.role IN ('user', 'assistant')
                AND f.context_events_fts MATCH ?

            ORDER BY
                bm25(context_events_fts)

            LIMIT ?
        """

        # Prefer a precise AND match. Long natural-language tasks often contain
        # terms that never co-occur in one historical event, though, which made
        # STM return nothing and forced the agent to rediscover the same evidence.
        fts_query = " AND ".join(safe_terms)

        rows = self.connection.execute(
            select_sql,
            (
                str(session_id),
                fts_query,
                top_k,
            ),
        ).fetchall()

        if not rows and len(safe_terms) > 1:
            fts_query = " OR ".join(safe_terms)

            rows = self.connection.execute(
                select_sql,
                (
                    str(session_id),
                    fts_query,
                    top_k,
                ),
            ).fetchall()

        return [self._row_to_event(row) for row in rows]

    def delete(
        self,
        event_id: UUID | str,
    ) -> bool:
        """
        Delete one event from both the main table and FTS index.
        """

        event_id = str(event_id)

        try:
            self.connection.execute(
                """
                DELETE FROM context_events_fts
                WHERE event_id = ?
                """,
                (event_id,),
            )

            cursor = self.connection.execute(
                """
                DELETE FROM context_events
                WHERE id = ?
                """,
                (event_id,),
            )

            self.connection.commit()

            return cursor.rowcount > 0

        except Exception:
            self.connection.rollback()
            raise

    def clear_session(
        self,
        session_id: UUID | str,
    ) -> int:
        """
        Delete every event belonging to a session.

        Returns:
            Number of deleted events from the main table.
        """

        session_id = str(session_id)

        try:
            self.connection.execute(
                """
                DELETE FROM context_events_fts
                WHERE session_id = ?
                """,
                (session_id,),
            )

            cursor = self.connection.execute(
                """
                DELETE FROM context_events
                WHERE session_id = ?
                """,
                (session_id,),
            )

            self.connection.commit()

            return cursor.rowcount

        except Exception:
            self.connection.rollback()
            raise

    def count(
        self,
        session_id: UUID | str,
    ) -> int:
        """
        Return the number of events in a session.
        """

        row = self.connection.execute(
            """
            SELECT COUNT(*) AS count

            FROM context_events

            WHERE session_id = ?
            """,
            (str(session_id),),
        ).fetchone()

        return int(row["count"])

    @staticmethod
    def _row_to_event(
        row: sqlite3.Row,
    ) -> ContextEvent:
        """
        Reconstruct a proper ContextEvent from SQLite.

        SQLite stores primitive strings.

        The application layer receives proper Enum instances.
        """

        try:
            event_id = UUID(str(row["id"]))
        except (ValueError, TypeError):
            event_id = row["id"]

        try:
            timestamp = datetime.fromisoformat(str(row["timestamp"]))
        except (ValueError, TypeError):
            timestamp = datetime.now()

        try:
            role = ContextRole(row["role"])
        except (ValueError, TypeError):
            role = ContextRole.SYSTEM

        try:
            event_type = ContextType(row["type"])
        except (ValueError, TypeError):
            event_type = ContextType.EVENT

        try:
            priority = ContextPriority(row["priority"])
        except (ValueError, TypeError):
            priority = ContextPriority.NORMAL

        return ContextEvent(
            id=event_id,
            role=role,
            type=event_type,
            content=str(row["content"]),
            priority=priority,
            step=int(row["step"]),
            timestamp=timestamp,
            metadata=STMDatabase._deserialize_metadata(row["metadata"]),
        )

    def close(self) -> None:
        """
        Close SQLite connection.
        """

        if self.connection is not None:
            self.connection.close()

    def __enter__(self) -> "STMDatabase":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()
