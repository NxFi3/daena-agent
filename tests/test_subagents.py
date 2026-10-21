import threading
from pathlib import Path

import pytest

from src.agent.agent import Agent
from src.agent.subagents import SubagentManager
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult
from src.models.ToolCall import ToolCall
from src.security.Policy import SecurityPolicy
from src.tools.ToolManager import ToolManager


class FakeChild:
    def __init__(self, config, release=None):
        self.config = config
        self.release = release
        self.closed = False
        self.last_run_metrics = {"completed": True, "iterations": 1}

    def act(self, event, stop_event=None):
        if self.release is not None:
            assert self.release.wait(5), "test did not release child"
        return LLMResult("child evidence", {}, [], None, 1)

    def close(self):
        self.closed = True


def make_manager(tmp_path, options=None, release=None):
    children = []

    def factory(config):
        child = FakeChild(config, release)
        children.append(child)
        return child

    manager = SubagentManager({"subagents": options or {}}, factory)
    manager.set_workspace(str(tmp_path))
    return manager, children


def test_results_memory_isolation_and_cleanup(tmp_path):
    manager, children = make_manager(tmp_path)
    first = manager.spawn("Inspect A")
    second = manager.spawn("Inspect B")
    assert first["agent_id"] != second["agent_id"]
    for task in (first, second):
        result = manager.wait(task["agent_id"], 2000)
        assert result["status"] == "completed"
        assert result["response"] == "child evidence"
    assert not manager.has_pending()
    assert all(child.closed for child in children)
    databases = [child.config["memory"]["database"] for child in children]
    assert len(set(databases)) == 2
    assert all(not Path(path).parent.exists() for path in databases)
    for child in children:
        policy = SecurityPolicy.from_config(child.config["security"])
        assert policy.workspace_only
        assert not policy.allow_background
        assert "subagent" not in policy.allowed_tools
        assert "command_exec" not in policy.allowed_tools
        assert "apply_patch" not in policy.allowed_tools
        assert child.config["subagents"]["enabled"] is False


def test_concurrency_total_budget_and_uncollected_results(tmp_path):
    release = threading.Event()
    manager, _ = make_manager(tmp_path, {"max_concurrent": 1, "max_tasks": 1}, release)
    task = manager.spawn("Inspect")
    try:
        with pytest.raises(ValueError, match="budget"):
            manager.spawn("Inspect again")
        assert manager.has_active()
    finally:
        release.set()
    assert manager._get(task["agent_id"]).done.wait(2)
    assert manager.has_pending()
    manager.list()  # UI inspection must not consume model-visible results.
    assert manager.has_pending()
    manager.status(task["agent_id"])
    assert not manager.has_pending()
    manager.begin_run()
    assert not manager.list()


def test_cancel_keeps_capacity_until_thread_has_stopped(tmp_path):
    release = threading.Event()
    manager, _ = make_manager(tmp_path, {"max_concurrent": 1}, release)
    task = manager.spawn("Inspect")
    # Wait until the fake child is actually inside its blocking call.
    # Factory completion is signaled explicitly for deterministic cancellation.
    child = manager._get(task["agent_id"])
    try:
        result = manager.cancel(task["agent_id"])
        if not result["done"]:
            assert result["status"] == "cancelling"
        if manager.has_active():
            with pytest.raises(RuntimeError):
                manager.begin_run()
            with pytest.raises(ValueError, match="concurrency"):
                manager.spawn("Another")
    finally:
        release.set()
    assert manager.wait(child.id, 2000)["status"] == "cancelled"


def test_parent_stop_and_deadline_propagate(tmp_path):
    for timeout in (False, True):
        release = threading.Event()
        entered = threading.Event()
        parent_stop = threading.Event()
        captured = []

        class BlockingChild(FakeChild):
            def act(self, event, stop_event=None):
                captured.append(stop_event)
                entered.set()
                assert release.wait(2)
                assert stop_event.is_set()
                return super().act(event, stop_event)

        manager = SubagentManager({}, lambda config: BlockingChild(config))
        manager.set_workspace(str(tmp_path))
        manager.begin_run(parent_stop)
        task = manager.spawn("Inspect")
        assert entered.wait(2)
        if timeout:
            captured[0].deadline = 0
        else:
            parent_stop.set()
        release.set()
        assert manager._get(task["agent_id"]).done.wait(2)
        assert manager.status(task["agent_id"])["status"] == (
            "timed_out" if timeout else "cancelled"
        )


def test_child_failure_closes_resources(tmp_path):
    children = []

    class Broken(FakeChild):
        def act(self, event, stop_event=None):
            raise RuntimeError("provider unavailable")

    def factory(config):
        children.append(Broken(config))
        return children[-1]

    manager = SubagentManager({}, factory)
    manager.set_workspace(str(tmp_path))
    task = manager.spawn("Inspect")
    result = manager.wait(task["agent_id"], 2000)
    assert result["status"] == "failed"
    assert "provider unavailable" in result["error"]
    assert children[0].closed


def test_worker_disabled_and_permissions_never_expand(tmp_path):
    manager, _ = make_manager(tmp_path)
    with pytest.raises(ValueError, match="allow_writes"):
        manager.spawn("Edit", "worker")
    seen = []

    def factory(config):
        seen.append(config)
        return FakeChild(config)

    manager = SubagentManager(
        {"security": {"allowed_tools": [], "allow_network_tools": False}}, factory
    )
    manager.set_workspace(str(tmp_path))
    task = manager.spawn("Inspect")
    manager.wait(task["agent_id"], 2000)
    assert seen[0]["security"]["allowed_tools"] == []
    assert seen[0]["security"]["allow_network_tools"] is False
    assert SecurityPolicy.from_config(seen[0]["security"]).allowed_tools == set()


@pytest.mark.parametrize(
    "options", [{"max_concurrent": 0}, {"max_tasks": True}, {"timeout_seconds": "10"}]
)
def test_invalid_budgets_fail_early(options):
    with pytest.raises(ValueError):
        SubagentManager({"subagents": options})


@pytest.mark.parametrize(
    "options", [False, [], {"enabled": "false"}, {"allow_writes": "true"}]
)
def test_invalid_feature_config_is_rejected(options):
    with pytest.raises(ValueError):
        SubagentManager({"subagents": options})


def test_disabled_subagents_reject_direct_tool_calls(tmp_path):
    manager = ToolManager({"subagents": {"enabled": False}})
    manager.set_workspace(str(tmp_path))
    try:
        call = ToolCall(
            name="subagent", valid=True, args={"action": "spawn", "task": "Inspect"}
        )
        assert "subagent" not in {
            item["function"]["name"] for item in manager.get_tools()
        }
        result = manager.execute([call])["results"][0]
        assert result.metadata["security_rule"] == "feature_disabled"
        assert not manager.subagents.list()
    finally:
        manager.close()


def test_worker_changes_require_explicit_integration(tmp_path):
    def output(text="", name=None, args=None):
        calls = (
            [{"id": "call-worker", "function": {"name": name, "arguments": args}}]
            if name
            else []
        )
        return LLMResult(text, {"role": "assistant", "content": text}, calls, None, 10)

    class Worker:
        model = None
        count = 0

        def generate(self, messages, tools=None, **kwargs):
            self.count += 1
            if self.count == 1:
                return output(
                    name="apply_patch",
                    args={
                        "patch": "*** Begin Patch\n*** Add File: result.txt\n+verified child artifact\n*** End Patch"
                    },
                )
            if self.count == 2:
                return output(name="read_file", args={"file_path": "result.txt"})
            return output("Created and read result.txt")

    class Parent:
        model = None
        count = 0

        def generate(self, messages, tools=None, **kwargs):
            self.count += 1
            if self.count == 1:
                return output(
                    name="subagent",
                    args={
                        "action": "spawn",
                        "task": "Create result.txt and verify it",
                        "mode": "worker",
                    },
                )
            if self.count == 2:
                agent_id = agent.loop.tool.subagents.list()[0]["agent_id"]
                return output(
                    name="subagent",
                    args={"action": "wait", "agent_id": agent_id, "wait_ms": 2000},
                )
            if self.count == 3:
                return output("Premature success before parent verification")
            if self.count == 4:
                assert not (tmp_path / "result.txt").exists()
                agent_id = agent.loop.tool.subagents.list()[0]["agent_id"]
                return output(
                    name="subagent", args={"action": "merge", "agent_id": agent_id}
                )
            if self.count == 5:
                return output(name="read_file", args={"file_path": "result.txt"})
            return output("Parent verified the actual artifact")

    config = {
        "workspace": {"directory": str(tmp_path)},
        "memory": {"database": str(tmp_path / "parent.db")},
        "subagents": {"allow_writes": True},
        "max_agent_iterations": 8,
        "context": {"compaction_enabled": False},
    }
    with Agent(config, llm=Parent()) as agent:
        agent.loop.tool.subagents.agent_factory = lambda child_config: Agent(
            child_config, llm=Worker()
        )
        result = agent.act(
            ContextEvent(
                role=ContextRole.USER,
                type=ContextType.MESSAGE,
                content="Delegate creation",
            )
        )
        assert result.response == "Parent verified the actual artifact"
        assert agent.last_run_metrics["completed"]
        assert agent.last_run_metrics["resource_blocks"] == 1
        assert (
            tmp_path / "result.txt"
        ).read_text().strip() == "verified child artifact"


def test_real_parent_and_child_loops_collect_results(tmp_path):
    (tmp_path / "hello.txt").write_text("hello from workspace")

    def output(text="", name=None, args=None):
        calls = (
            [{"id": "call-1", "function": {"name": name, "arguments": args}}]
            if name
            else []
        )
        return LLMResult(text, {"role": "assistant", "content": text}, calls, None, 10)

    class Reader:
        model = None
        count = 0

        def generate(self, messages, tools=None, **kwargs):
            self.count += 1
            names = {item["function"]["name"] for item in tools}
            assert "subagent" not in names and "apply_patch" not in names
            if self.count == 1:
                return output(name="read_file", args={"file_path": "hello.txt"})
            assert any("hello from workspace" in str(item) for item in messages)
            return output("Child inspected hello.txt")

    class Parent:
        model = None
        count = 0

        def generate(self, messages, tools=None, **kwargs):
            self.count += 1
            if self.count == 1:
                return output(
                    name="subagent",
                    args={"action": "spawn", "task": "Read hello.txt and report"},
                )
            if self.count == 2:
                return output("Premature final answer")
            if self.count == 3:
                child_id = agent.loop.tool.subagents.list()[0]["agent_id"]
                return output(
                    name="subagent",
                    args={"action": "wait", "agent_id": child_id, "wait_ms": 2000},
                )
            assert any("Child inspected hello.txt" in str(item) for item in messages)
            return output("Reviewed child evidence")

    config = {
        "workspace": {"directory": str(tmp_path)},
        "memory": {"database": str(tmp_path / "parent.db")},
        "max_agent_iterations": 8,
        "context": {"compaction_enabled": False},
    }
    with Agent(config, llm=Parent()) as agent:
        agent.loop.tool.subagents.agent_factory = lambda child_config: Agent(
            child_config, llm=Reader()
        )
        result = agent.act(
            ContextEvent(
                role=ContextRole.USER,
                type=ContextType.MESSAGE,
                content="Delegate inspection",
            )
        )
        assert result.response == "Reviewed child evidence"
        assert agent.last_run_metrics["completed"]
        assert agent.last_run_metrics["resource_blocks"] == 1
        assert agent.loop.tool.subagents.list()[0]["status"] == "completed"
