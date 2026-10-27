"""Live multi-agent messages: ordered, scoped and safe during execution."""
import threading
import pytest

from src.agent.subagents import SubagentManager
from src.agent.messaging import AgentMessage


class LiveChild:
    def __init__(self, config, ready, release):
        self.config = config
        self.ready = ready
        self.release = release
        self.received = []
        self.last_run_metrics = {"completed": True}

    def steer(self, text):
        self.received.append(text)

    def act(self, event, stop_event=None):
        self.ready.set()
        assert self.release.wait(4)
        from src.models.LLMResult import LLMResult
        return LLMResult("ready", {}, [], None, 1)

    def close(self):
        pass


def test_parent_child_live_message_without_restart(tmp_path):
    ready, release = threading.Event(), threading.Event()
    instances = []
    def factory(config):
        child = LiveChild(config, ready, release)
        instances.append(child)
        return child
    manager = SubagentManager({}, factory)
    manager.set_workspace(str(tmp_path))
    spawned = manager.spawn("Inspect files")
    agent_id = spawned["agent_id"]
    try:
        assert ready.wait(3)
        receipt = manager.send_message(sender="parent", recipient=agent_id,
                                       text="Please also check the tests")
        assert receipt["delivery"] == "queued"
        assert len(instances) == 1
        assert "Please also check the tests" in instances[0].received[-1]
        child_tool = AgentMessage(manager, agent_id)
        assert child_tool.execute(action="send", text="I found two problems").success
        inbox = manager.inbox()
        assert len(inbox) == 1
        assert inbox[0]["sender"] == agent_id
        assert manager.inbox() == []
        with pytest.raises(ValueError, match="yourself"):
            manager.send_message(sender=agent_id, recipient=agent_id, text="no")
    finally:
        release.set()
        manager.wait(agent_id, 3000)


def test_peer_message_and_rejection_of_unknown_agents(tmp_path):
    releases = threading.Event()
    class PassiveChild:
        last_run_metrics = {"completed": True}
        def __init__(self, config): self.messages = []
        def steer(self, message): self.messages.append(message)
        def act(self, event, stop_event=None):
            releases.wait(4)
            from src.models.LLMResult import LLMResult
            return LLMResult("ok", {}, [], None, 1)
        def close(self): pass
    objects = []
    def factory(config):
        o = PassiveChild(config)
        objects.append(o)
        return o
    manager = SubagentManager({}, factory)
    manager.set_workspace(str(tmp_path))
    a = manager.spawn("A")["agent_id"]
    b = manager.spawn("B")["agent_id"]
    try:
        manager.send_message(sender=a, recipient=b, text="Shared observation")
        assert len(manager.inbox(b)) == 1
        assert manager.inbox(b) == []
        with pytest.raises(ValueError, match="Unknown"):
            manager.send_message(sender="random", recipient=b, text="No")
    finally:
        releases.set()
        manager.wait_all(3000)


def test_live_messaging_disabled_when_delegation_forbidden(tmp_path):
    manager = SubagentManager({"security": {"allowed_tools": []}},
                              lambda cfg: None)
    manager.set_workspace(str(tmp_path))
    from src.agent.subagents import ChildTask, Cancellation
    child = ChildTask("c", "Task", "read_only", Cancellation(None, 30))
    config = manager._child_config(child, str(tmp_path))
    assert "agent_message" not in config["security"]["allowed_tools"]
