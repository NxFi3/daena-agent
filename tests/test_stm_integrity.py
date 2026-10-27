"""SQLite operations are serialized across Studio and Agent worker threads."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Event

from src.memories.stm.stmdatabase import STMDatabase
from src.models.ContextEvent import ContextRole
from tests.test_stm import event


def test_clear_cannot_interleave_between_event_and_fts_insert(tmp_path):
    db = STMDatabase(tmp_path / "memory.db")
    entered, release, clearing = Event(), Event(), Event()
    native = db.connection

    class PausedInsert:
        def execute(self, sql, *args):
            result = native.execute(sql, *args)
            if "INSERT OR IGNORE INTO context_events (" in sql:
                entered.set()
                assert release.wait(3)
            return result

        def __getattr__(self, name):
            return getattr(native, name)

    db.connection = PausedInsert()
    with ThreadPoolExecutor(max_workers=2) as pool:
        inserted = pool.submit(db.add, "s", event("evidence", 1))
        assert entered.wait(2)

        def clear():
            clearing.set()
            return db.clear_session("s")

        removed = pool.submit(clear)
        assert clearing.wait(2)
        try:
            assert not removed.done()
        finally:
            release.set()
        assert inserted.result(timeout=2)
        assert removed.result(timeout=2) == 1
    assert db.count("s") == 0
    assert native.execute("SELECT COUNT(*) FROM context_events_fts").fetchone()[0] == 0
    db.close()
    db.close()


def test_session_list_uses_activity_time_not_unrelated_step_counters(tmp_path):
    with STMDatabase(tmp_path / "memory.db") as db:
        older, newer = event("old", 900), event("new", 1)
        older.timestamp, newer.timestamp = datetime(2025, 1, 1), datetime(2026, 1, 1)
        db.add("older", older)
        db.add("newer", newer)
        assert [row["session_id"] for row in db.list_sessions()] == ["newer", "older"]


def test_invalid_stored_role_never_becomes_system_authority(tmp_path):
    with STMDatabase(tmp_path / "memory.db") as db:
        db.add("s", event("Untrusted imported content", 1))
        db.connection.execute("UPDATE context_events SET role='invalid'")
        db.connection.commit()
        assert db.get_recent("s")[0].role == ContextRole.USER
