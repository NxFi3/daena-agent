from uuid import uuid4

from src.memories.stm.stmdatabase import STMDatabase
from src.models.ContextEvent import (
    ContextEvent,
    ContextPriority,
    ContextRole,
    ContextType,
)


def event(content: str, step: int) -> ContextEvent:
    return ContextEvent(
        id=uuid4(),
        role=ContextRole.USER,
        type=ContextType.MESSAGE,
        content=content,
        step=step,
    )


def test_stm_search_falls_back_to_or_when_and_has_no_match(tmp_path):
    db = STMDatabase(tmp_path / "stm.db")
    session_id = uuid4()

    try:
        db.add(session_id, event("alpha", 1))
        db.add(session_id, event("gamma", 2))

        results = db.search(
            session_id=session_id,
            query="alpha beta",
            top_k=3,
        )

        assert len(results) == 1
        assert results[0].content == "alpha"
    finally:
        db.close()


def test_stm_search_prefers_exact_and_match(tmp_path):
    db = STMDatabase(tmp_path / "stm.db")
    session_id = uuid4()

    try:
        db.add(session_id, event("alpha beta", 1))
        db.add(session_id, event("alpha", 2))

        results = db.search(
            session_id=session_id,
            query="alpha beta",
            top_k=3,
        )

        assert results
        assert results[0].content == "alpha beta"
    finally:
        db.close()


def test_stm_search_excludes_raw_tool_events(tmp_path):
    db = STMDatabase(tmp_path / "stm.db")
    session_id = uuid4()

    try:
        user_event = event("process_poll runtime issue", 1)
        db.add(session_id, user_event)

        tool_event = ContextEvent(
            id=uuid4(),
            role=ContextRole.TOOL,
            type=ContextType.TOOL_RESULT,
            content='{"name":"search","success":false,"content":"process_poll runtime issue"}',
            step=2,
        )
        db.add(session_id, tool_event)

        results = db.search(
            session_id=session_id,
            query="process_poll runtime issue",
            top_k=5,
        )

        assert results
        assert all(
            item.role in {ContextRole.USER, ContextRole.ASSISTANT}
            and item.type == ContextType.MESSAGE
            for item in results
        )
        assert not any(item.id == tool_event.id for item in results)
    finally:
        db.close()


def test_stm_search_can_retrieve_tool_results_when_explicitly_requested(tmp_path):
    db = STMDatabase(tmp_path / "stm.db")
    session_id = uuid4()
    tool_event = ContextEvent(
        id=uuid4(),
        role=ContextRole.TOOL,
        type=ContextType.TOOL_RESULT,
        content=(
            '{"name":"web_fetch","success":true,'
            '"content":{"content":"public listing token abc123 neighborhood Fin"}}'
        ),
        step=4,
    )
    db.add(session_id, tool_event)

    try:
        # Existing callers retain message-only behavior by default.
        assert db.search(session_id, "abc123 neighborhood", top_k=3) == []

        # Context retrieval can explicitly recall the earlier tool evidence.
        results = db.search(
            session_id,
            "abc123 neighborhood",
            top_k=3,
            include_tool_results=True,
        )
        assert [item.id for item in results] == [tool_event.id]
        assert results[0].type == ContextType.TOOL_RESULT
    finally:
        db.close()


def test_stm_recent_after_step_returns_newest_chronological_slice(tmp_path):
    db = STMDatabase(tmp_path / "stm.db")
    session_id = uuid4()

    try:
        for step in range(1, 8):
            db.add(session_id, event(f"event-{step}", step))

        results = db.get_recent_after_step(session_id, after_step=3, limit=2)

        assert [item.step for item in results] == [6, 7]
        assert [item.content for item in results] == ["event-6", "event-7"]
    finally:
        db.close()


def test_stm_search_respects_min_step(tmp_path):
    db = STMDatabase(tmp_path / "stm.db")
    session_id = uuid4()

    try:
        db.add(session_id, event("important old fact", 2))
        db.add(session_id, event("important new fact", 8))

        results = db.search(
            session_id=session_id,
            query="important fact",
            top_k=5,
            min_step=4,
        )

        assert [item.content for item in results] == ["important new fact"]
    finally:
        db.close()


def test_latest_checkpoint_is_persisted_and_retrievable(tmp_path):
    db = STMDatabase(tmp_path / "stm.db")
    session_id = uuid4()
    checkpoint = ContextEvent(
        id=uuid4(),
        role=ContextRole.SYSTEM,
        type=ContextType.EVENT,
        content="checkpoint summary",
        priority=ContextPriority.HIGH,
        step=10,
        metadata={
            "checkpoint": True,
            "covered_through_step": 10,
        },
    )

    try:
        assert db.add(session_id, checkpoint) is True
        loaded = db.get_latest_checkpoint(session_id)
        assert loaded is not None
        assert loaded.content == "checkpoint summary"
        assert loaded.metadata["covered_through_step"] == 10
    finally:
        db.close()
