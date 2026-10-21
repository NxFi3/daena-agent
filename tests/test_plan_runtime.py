"""Public loop regressions: the model owns planning and completion decisions."""

import json
import sys
import threading

from src.agent.agent import Agent
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult


def output(name=None, args=None, text=""):
    calls = [{"function": {"name": name, "arguments": args}}] if name else []
    return LLMResult(text, {}, calls, None, 1)


class Script:
    model = None

    def __init__(self, results):
        self.results = iter(results)
        self.history = []

    def generate(self, messages, **kwargs):
        self.history.append(messages)
        result = next(self.results)
        return result(messages) if callable(result) else result


def run(tmp_path, results, **overrides):
    model = Script(results)
    config = {
        "workspace": {"directory": str(tmp_path)},
        "memory": {"database": str(tmp_path / ".daena" / "memory.db")},
        "context": {"compaction_enabled": False},
        "max_agent_iterations": 12,
        **overrides,
    }
    agent = Agent(config, llm=model)
    result = agent.act(
        ContextEvent(
            role=ContextRole.USER,
            type=ContextType.MESSAGE,
            content="Implement and check the requested change",
        )
    )
    return agent, model, result


def test_repeated_observations_and_failed_retries_reach_tools(tmp_path):
    (tmp_path / "file.txt").write_text("real content")
    script = [output("read_file", {"file_path": "file.txt"})] * 3
    script += [output("read_file", {"file_path": "missing.txt"})] * 2
    script += [output(text="Reported the missing file")]
    agent, model, result = run(tmp_path, script)
    try:
        assert agent.last_run_metrics["tool_successes"] == 3
        assert agent.last_run_metrics["tool_failures"] == 2
        receipts = [
            json.loads(m["content"].split("\n\n")[0])
            for m in model.history[-1]
            if m["role"] == "tool"
        ]
        assert len(receipts) == 5
        assert all("BLOCKED" not in str(r) for r in receipts)
        assert result.response == "Reported the missing file"
    finally:
        agent.close()


def test_plan_can_complete_without_guessed_terminal_work_and_followup_continues(
    tmp_path,
):
    agent, model, result = run(
        tmp_path,
        [
            output(
                "plan",
                {
                    "action": "create",
                    "goal": "Answer",
                    "steps": ["Reason about the question"],
                },
            ),
            output("plan", {"action": "complete"}),
            output(
                "apply_patch",
                {
                    "patch": "*** Begin Patch\n*** Add File: answer.txt\n+42\n*** End Patch"
                },
            ),
            output("read_file", {"file_path": "answer.txt"}),
            output(text="Answer checked"),
        ],
    )
    try:
        assert result.response == "Answer checked"
        assert agent.last_run_metrics["tool_successes"] == 4
        assert (tmp_path / "answer.txt").read_text().strip() == "42"
    finally:
        agent.close()


def test_successful_command_does_not_force_finalization(tmp_path):
    (tmp_path / "check.py").write_text("print('passed')")
    agent, model, result = run(
        tmp_path,
        [
            output("command_exec", {"command": [sys.executable, "check.py"]}),
            output(
                "apply_patch",
                {
                    "patch": "*** Begin Patch\n*** Add File: next.txt\n+next step\n*** End Patch"
                },
            ),
            output(text="Done"),
        ],
    )
    try:
        assert result.response == "Done"
        assert agent.last_run_metrics["tool_successes"] == 2
        assert (tmp_path / "next.txt").exists()
    finally:
        agent.close()


def test_max_iterations_bounds_repetition_without_rewriting_intent(tmp_path):
    agent, _, result = run(
        tmp_path,
        [output("read_file", {"file_path": "missing"})] * 3,
        max_agent_iterations=3,
    )
    try:
        assert not agent.last_run_metrics["completed"]
        assert agent.last_run_metrics["tool_failures"] == 3
        assert "Maximum iterations" in result.response
    finally:
        agent.close()


def test_cancellation_prevents_following_tool_in_same_batch(tmp_path):
    stop = threading.Event()
    model = Script([])
    config = {
        "workspace": {"directory": str(tmp_path)},
        "memory": {"database": str(tmp_path / "memory.db")},
        "context": {"compaction_enabled": False},
    }
    with Agent(config, llm=model) as agent:
        calls = [
            output("read_file", {"file_path": "missing"}).tool_calls[0],
            output(
                "apply_patch",
                {
                    "patch": "*** Begin Patch\n*** Add File: forbidden.txt\n+bad\n*** End Patch"
                },
            ).tool_calls[0],
        ]
        model.results = iter([LLMResult("", {}, calls, None, 1)])

        def sink(event):
            if event["type"] == "tool_result":
                stop.set()

        agent.act(
            ContextEvent(
                role=ContextRole.USER, type=ContextType.MESSAGE, content="work"
            ),
            on_event=sink,
            stop_event=stop,
        )
        assert not (tmp_path / "forbidden.txt").exists()
        assert not agent.last_run_metrics["completed"]
        events = agent.loop.stm.get_recent(agent.session_id, limit=20)
        assert sum(e.role == ContextRole.TOOL for e in events) == 2


def test_steering_is_after_receipt_and_before_next_generation(tmp_path):
    (tmp_path / "file.txt").write_text("hello")
    model = Script(
        [output("read_file", {"file_path": "file.txt"}), output(text="Stopped editing")]
    )
    config = {
        "workspace": {"directory": str(tmp_path)},
        "memory": {"database": str(tmp_path / "memory.db")},
        "context": {"compaction_enabled": False},
    }
    with Agent(config, llm=model) as agent:

        def sink(event):
            if event["type"] == "tool_result":
                agent.steer("Only explain; do not edit")

        agent.act(
            ContextEvent(
                role=ContextRole.USER, type=ContextType.MESSAGE, content="Read first"
            ),
            on_event=sink,
        )
        assert model.history[-1][-1] == {
            "role": "user",
            "content": "Only explain; do not edit",
        }
        assert model.history[-1][-2]["role"] == "tool"
