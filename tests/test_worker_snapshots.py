import threading
from pathlib import Path

import pytest

from src.agent.subagents import SubagentManager
from src.agent.workspace_snapshot import WorkspaceSnapshot
from src.models.LLMResult import LLMResult
from src.tools.ToolManager import ToolManager


def test_parallel_workers_have_isolated_files_and_nonoverlapping_merges(tmp_path):
    arrived = threading.Barrier(3)
    release = threading.Event()
    roots = []

    class Writer:
        last_run_metrics = {"completed": True}

        def __init__(self, config):
            self.root = Path(config["workspace"]["directory"])
            roots.append(self.root)

        def act(self, event, stop_event=None):
            (self.root / (event.content + ".txt")).write_text(event.content)
            arrived.wait(timeout=3)
            assert release.wait(3)
            return LLMResult("written", {}, [], None, 1)

        def close(self):
            pass

    manager = SubagentManager({"subagents": {"allow_writes": True}}, Writer)
    manager.set_workspace(str(tmp_path))
    ids = [manager.spawn(name, "worker")["agent_id"] for name in ("a", "b")]
    try:
        arrived.wait(timeout=3)
        assert len(set(roots)) == 2
        assert all(root != tmp_path for root in roots)
        assert not list(tmp_path.glob("*.txt"))
        (tmp_path / "parent.txt").write_text("independent parent edit")
        assert manager.has_active()
        release.set()
        for ident in ids:
            result = manager.wait(ident, 3000)
            assert result["status"] == "completed", result
            assert result["integration"] == "pending"
            assert manager.has_pending()
            merged = manager.integrate(ident)
            assert merged["merged"], merged
        assert not manager.has_pending()
        assert (tmp_path / "a.txt").read_text() == "a"
        assert (tmp_path / "b.txt").read_text() == "b"
        assert (tmp_path / "parent.txt").read_text() == "independent parent edit"
        assert all(not root.exists() for root in roots)
    finally:
        release.set()
        manager.close()


def test_conflict_applies_none_of_the_changes(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "a").write_text("base")
    storage = tmp_path / "worker"
    storage.mkdir()
    snapshot = WorkspaceSnapshot(str(root), str(storage))
    (snapshot.root / "a").write_text("worker")
    (snapshot.root / "b").write_text("new file")
    (root / "a").write_text("parent")
    result = snapshot.merge()
    assert not result["merged"] and result["conflicts"] == ["a"]
    assert (root / "a").read_text() == "parent"
    assert not (root / "b").exists()


def test_merge_add_update_delete_binary_modes_and_ignored_runtime(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "old").write_text("remove")
    (root / "update").write_text("before")
    (root / "memory.db").write_bytes(b"parent db")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("private")
    storage = tmp_path / "worker"
    storage.mkdir()
    snapshot = WorkspaceSnapshot(str(root), str(storage), excluded=[root / "memory.db"])
    assert not (snapshot.root / "memory.db").exists()
    assert not (snapshot.root / ".git").exists()
    (snapshot.root / "old").unlink()
    (snapshot.root / "update").write_text("after")
    (snapshot.root / "binary").write_bytes(b"\xff\x00\xfe")
    (snapshot.root / "binary").chmod(0o755)
    (root / "memory.db").write_bytes(b"new parent db")
    assert snapshot.merge()["merged"]
    assert not (root / "old").exists()
    assert (root / "update").read_text() == "after"
    assert (root / "binary").read_bytes() == b"\xff\x00\xfe"
    assert (root / "binary").stat().st_mode & 0o777 == 0o755
    assert (root / "memory.db").read_bytes() == b"new parent db"


def test_snapshot_rejects_symlinks_and_file_parent_conflicts(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    storage = tmp_path / "worker"
    storage.mkdir()
    snapshot = WorkspaceSnapshot(str(root), str(storage))
    (snapshot.root / "dir").mkdir()
    (snapshot.root / "dir" / "file").write_text("worker")
    (root / "dir").write_text("parent file")
    assert snapshot.merge()["conflicts"] == ["dir/file"]
    (snapshot.root / "link").symlink_to(root / "dir")
    with pytest.raises(ValueError, match="symlink"):
        snapshot.changes()


def test_worker_mode_schema_matches_host_configuration(tmp_path):
    for allowed in (True, False):
        manager = ToolManager({"subagents": {"allow_writes": allowed}})
        manager.set_workspace(str(tmp_path))
        try:
            schema = next(
                t["function"]
                for t in manager.get_tools()
                if t["function"]["name"] == "subagent"
            )
            modes = schema["parameters"]["properties"]["mode"]["enum"]
            assert ("worker" in modes) == allowed
            assert manager.subagents.capabilities()["modes"] == modes
        finally:
            manager.close()


def test_diff_reviews_conflicts_without_mutation(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "a").write_text("base\n")
    storage = tmp_path / "worker"
    storage.mkdir()
    snapshot = WorkspaceSnapshot(str(root), str(storage))
    (snapshot.root / "a").write_text("worker\n")
    (root / "a").write_text("parent\n")
    review = snapshot.diff()
    assert "-parent" in review["diff"] and "+worker" in review["diff"]
    assert not review["truncated"]
    assert (root / "a").read_text() == "parent\n"


def test_failed_merge_rolls_back_already_written_files(tmp_path, monkeypatch):
    import os

    root = tmp_path / "project"
    root.mkdir()
    for name in ("a", "b"):
        (root / name).write_text("base")
    storage = tmp_path / "worker"
    storage.mkdir()
    snapshot = WorkspaceSnapshot(str(root), str(storage))
    for name in ("a", "b"):
        (snapshot.root / name).write_text("changed")
    original = os.replace

    def fail_second(source, target):
        if Path(target).name == "b":
            raise OSError("simulated write failure")
        original(source, target)

    monkeypatch.setattr(os, "replace", fail_second)
    with pytest.raises(OSError, match="simulated"):
        snapshot.merge()
    assert (root / "a").read_text() == "base"
    assert (root / "b").read_text() == "base"
