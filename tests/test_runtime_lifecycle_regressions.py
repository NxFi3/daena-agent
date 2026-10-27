"""Interrupted turns and failed native cleanup cannot corrupt later runs."""

import shutil
from threading import Event
from types import SimpleNamespace

import pytest

from src.agent.agent import Agent
from src.agent.agentloop import Loop
from src.agent.subagents import SubagentManager
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult
from tests.test_subagents import FakeChild


def test_subagent_cleanup_failure_still_finishes_and_can_be_retried(
    tmp_path, monkeypatch
):
    locked = Event()
    locked.set()
    storage_path = tmp_path / "owned-storage"
    storage_path.mkdir()

    class Storage:
        name = str(storage_path)

        def cleanup(self):
            if locked.is_set():
                raise PermissionError("temporary file is still open")
            if storage_path.exists():
                shutil.rmtree(storage_path)

    monkeypatch.setattr(
        "src.agent.subagents.TemporaryDirectory", lambda **kwargs: Storage()
    )
    manager = SubagentManager({}, FakeChild)
    manager.set_workspace(str(tmp_path))
    child = manager.spawn("Inspect")
    assert manager._get(child["agent_id"]).done.wait(2)
    result = manager.status(child["agent_id"])
    assert result["status"] == "completed"
    assert "PermissionError" in result["cleanup_error"]
    assert not manager.has_active()
    manager.close()  # Repeated cleanup failures cannot prevent other shutdown.
    with pytest.raises(RuntimeError, match="release previous subagent resources"):
        manager.begin_run()
    locked.clear()
    manager.begin_run()
    assert not manager.tasks and not storage_path.exists()


def test_delegation_preserves_host_instruction_and_restricts_tools(tmp_path):
    seen = []

    def factory(config):
        seen.append(config)
        return FakeChild(config)

    manager = SubagentManager(
        {"runtime_instruction": "Project behavior supplied by host"}, factory
    )
    manager.set_workspace(str(tmp_path))
    child = manager.spawn("Inspect")
    assert manager.wait(child["agent_id"], 2000)["done"]
    assert "Project behavior supplied by host" in seen[0]["runtime_instruction"]
    assert "independent context" in seen[0]["runtime_instruction"]
    assert seen[0]["subagents"] == {"enabled": False}


def test_interrupted_steering_does_not_leak_into_next_turn(tmp_path):
    stop = Event()
    requests = []

    class Model:
        model = None

        def generate(self, messages, **kwargs):
            requests.append(messages)
            if len(requests) == 1:
                agent.steer("Obsolete instruction for the canceled task")
                stop.set()
            return LLMResult("Done", {}, [], None, 1)

    agent = Agent(
        {
            "memory": {"database": str(tmp_path / "memory.db")},
            "workspace": {"directory": str(tmp_path)},
        },
        llm=Model(),
    )

    def task(text):
        return ContextEvent(
            role=ContextRole.USER, type=ContextType.MESSAGE, content=text
        )

    try:
        agent.act(task("Old task"), stop_event=stop)
        agent.act(task("New task"))
        assert len(requests) == 2
        assert "Obsolete instruction" not in str(requests[-1])
    finally:
        agent.close()


def test_closed_agent_rejects_workspace_and_steering_mutations(tmp_path):
    agent = Agent(
        {
            "memory": {"database": str(tmp_path / "memory.db")},
            "workspace": {"directory": str(tmp_path)},
        },
        llm=SimpleNamespace(model=None),
    )
    agent.close()
    with pytest.raises(RuntimeError, match="closed"):
        agent.set_workingdirectory(str(tmp_path / "must-not-create"))
    with pytest.raises(RuntimeError, match="closed"):
        agent.steer("No more work")
    assert not (tmp_path / "must-not-create").exists()


def test_loop_closes_memory_even_when_tool_cleanup_raises():
    closed = []
    loop = Loop.__new__(Loop)

    def fail():
        raise OSError("A tool could not close")

    loop.tool = SimpleNamespace(close=fail)
    loop.stm = SimpleNamespace(close=lambda: closed.append("memory"))
    with pytest.raises(OSError):
        loop.close()
    assert closed == ["memory"]
