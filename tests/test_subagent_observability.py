import threading
import time

from src.agent.subagents import SubagentManager
from src.models.LLMResult import LLMResult


class Child:
    def __init__(self, config):
        self.config = config
        self.last_run_metrics = {"completed": True}

    def act(self, event, stop_event=None, on_event=None):
        on_event({"type": "context", "estimated_tokens": 500, "budget": 4000})
        on_event(
            {
                "type": "tool_call",
                "name": "read_file",
                "arguments": {"file_path": "file.txt"},
            }
        )
        return LLMResult("Inspected", {}, [], None, 0)

    def close(self):
        pass


def test_named_children_publish_events_and_wait_all_collects(tmp_path):
    configs = []
    events = []

    def factory(config):
        configs.append(config)
        return Child(config)

    manager = SubagentManager(
        {"security": {"allowed_tools": ["read_file", "web_search"]}}, factory
    )
    manager.event_sink = events.append
    manager.set_workspace(str(tmp_path))
    manager.begin_run()
    child = manager.spawn(
        "Inspect",
        name="Reviewer",
        context="Relevant file only",
        allowed_tools=["read_file", "command_exec"],
    )
    result = manager.wait_all(2000)
    assert result["all_done"] and result["agents"][0]["status"] == "completed"
    assert configs[0]["security"]["allowed_tools"] == ["read_file"]
    assert not manager.has_pending()
    snapshot = manager.inspect(child["agent_id"])
    assert (
        snapshot["name"] == "Reviewer"
        and snapshot["finished_at"] >= snapshot["created_at"]
    )
    assert snapshot["progress"]["name"] == "read_file"
    assert any(e["event"]["type"] == "finished" for e in events)
    manager.close()


def test_long_wait_all_observes_parent_cancellation(tmp_path):
    entered = threading.Event()
    parent = threading.Event()

    class Slow(Child):
        def act(self, event, stop_event=None, on_event=None):
            entered.set()
            while not stop_event.is_set():
                time.sleep(0.01)
            return LLMResult("Stopped", {}, [], None, 0)

    manager = SubagentManager({}, Slow)
    manager.set_workspace(str(tmp_path))
    manager.begin_run(parent)
    child = manager.spawn("Wait")
    assert entered.wait(1)
    result = []
    waiter = threading.Thread(target=lambda: result.append(manager.wait_all(300000)))
    waiter.start()
    parent.set()
    waiter.join(0.5)
    assert not waiter.is_alive()
    manager.tasks[child["agent_id"]].done.wait(1)
    assert manager.status(child["agent_id"])["status"] == "cancelled"
    manager.close()
