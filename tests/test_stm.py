from uuid import uuid4

from src.memories.stm.stmdatabase import STMDatabase
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType


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
