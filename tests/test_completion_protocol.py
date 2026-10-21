"""General turn lifecycle; no language, command-name, or task-specific gates."""

import pytest

from src.agent.agent import Agent
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult


def call(name, **args):
    return LLMResult("", {}, [{"function": {"name": name, "arguments": args}}], None, 1)


def finish(response="Finished", outcome="completed", evidence=None):
    return call(
        "task_finish", outcome=outcome, response=response, evidence=evidence or []
    )


def task(text):
    return ContextEvent(role=ContextRole.USER, type=ContextType.MESSAGE, content=text)


class Script:
    model = None

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []

    def generate(self, messages, **kwargs):
        self.messages.append(messages)
        assert "task_finish" in {item["function"]["name"] for item in kwargs["tools"]}
        result = next(self.outputs)
        return result(messages) if callable(result) else result


def config(tmp_path):
    return {
        "runtime": {"completion_protocol": "explicit"},
        "workspace": {"directory": str(tmp_path)},
        "memory": {"database": str(tmp_path / ".daena" / "stm.db")},
        "context": {"compaction_enabled": False},
        "max_agent_iterations": 10,
    }


@pytest.mark.parametrize(
    "progress", ["I will start now", "بذار شروع کنم", "準備します", "Almost ready"]
)
def test_progress_text_is_not_a_terminal_event(tmp_path, progress):
    model = Script(
        [
            LLMResult(progress, {}, [], None, 1),
            call(
                "apply_patch",
                patch="*** Begin Patch\n*** Add File: result.txt\n+actual work\n*** End Patch",
            ),
            call("read_file", file_path="result.txt"),
            finish(
                "Created and read result.txt", evidence=["Read result.txt: actual work"]
            ),
        ]
    )
    events = []
    with Agent(config(tmp_path), llm=model) as agent:
        result = agent.act(task("Create a file"), on_event=events.append)
        assert result.response == "Created and read result.txt"
        assert (tmp_path / "result.txt").read_text().strip() == "actual work"
        assert agent.last_run_metrics["iterations"] == 4
        assert agent.last_run_metrics["verification_evidence"] == [
            "Read result.txt: actual work"
        ]
        assert len([e for e in events if e["type"] == "run_end"]) == 1
        assert events[-1]["completed"] is True
        assert any(progress == m.get("content") for m in model.messages[1])


@pytest.mark.parametrize("outcome", ["blocked", "needs_input"])
def test_blocker_or_question_is_not_reported_as_completed(tmp_path, outcome):
    model = Script([finish("Provide the missing connection details", outcome)])
    with Agent(config(tmp_path), llm=model) as agent:
        result = agent.act(task("Connect to the service"))
        assert "connection" in result.response
        assert agent.last_run_metrics["outcome"] == outcome
        assert not agent.last_run_metrics["completed"]


def test_conversation_can_finish_without_using_workspace_tools(tmp_path):
    with Agent(config(tmp_path), llm=Script([finish("Hello!")])) as agent:
        assert agent.act(task("hello")).response == "Hello!"
        assert agent.last_run_metrics["tool_successes"] == 1


def test_finish_must_follow_tool_receipts_not_share_a_batch(tmp_path):
    patch = call(
        "apply_patch",
        patch="*** Begin Patch\n*** Add File: result.txt\n+work\n*** End Patch",
    )
    batch = LLMResult(
        "", {}, patch.tool_calls + finish("Unreviewed").tool_calls, None, 1
    )
    model = Script(
        [batch, call("read_file", file_path="result.txt"), finish("Reviewed")]
    )
    with Agent(config(tmp_path), llm=model) as agent:
        assert agent.act(task("Work")).response == "Reviewed"
        assert agent.last_run_metrics["resource_blocks"] == 1
        receipts = [m for m in model.messages[1] if m["role"] == "tool"]
        assert len(receipts) == 2
        assert "unfinished_resources" in receipts[-1]["content"]


def test_plan_and_tool_receipts_survive_followup_turn(tmp_path):
    model = Script(
        [
            call(
                "plan",
                action="create",
                goal="Original objective",
                steps=["Inspect", "Implement"],
            ),
            call("make_directory", paths=["src", "tests"]),
            finish("Need a detail", "needs_input"),
            call("plan", action="complete"),
            finish("Continued existing plan"),
        ]
    )
    with Agent(config(tmp_path), llm=model) as agent:
        agent.act(task("Build the project"))
        assert (
            agent.act(task("Continue with the default")).response
            == "Continued existing plan"
        )
        first_followup = model.messages[3]
        assert "Original objective" in first_followup[0]["content"]
        assert any(m.get("tool_name") == "make_directory" for m in first_followup)
        assert first_followup[-1]["content"] == "Continue with the default"


def test_iteration_limit_reports_noncompletion_for_progress_only_model(tmp_path):
    settings = config(tmp_path)
    settings["max_agent_iterations"] = 3
    with Agent(
        settings, llm=Script([LLMResult("Starting", {}, [], None, 1)] * 3)
    ) as agent:
        agent.act(task("Work"))
        assert not agent.last_run_metrics["completed"]
        assert agent.last_run_metrics["stop_reason"] == "Maximum iterations reached"


def test_explicit_protocol_rejects_incompatible_tool_allowlist(tmp_path):
    settings = config(tmp_path)
    settings["security"] = {"allowed_tools": ["read_file"]}
    with pytest.raises(ValueError, match="requires task_finish"):
        Agent(settings, llm=Script([]))


def test_explicit_parent_and_worker_complete_and_integrate(tmp_path):
    settings = config(tmp_path)
    settings["subagents"] = {"allow_writes": True}
    worker = Script(
        [
            call("make_directory", paths=["src"]),
            call(
                "apply_patch",
                patch="*** Begin Patch\n*** Add File: src/result.txt\n+worker output\n*** End Patch",
            ),
            finish("Worker created src/result.txt", evidence=["apply_patch succeeded"]),
        ]
    )

    def collect(messages):
        ident = agent.loop.tool.subagents.list()[0]["agent_id"]
        return call("subagent", action="wait", agent_id=ident, wait_ms=3000)

    def merge(messages):
        ident = agent.loop.tool.subagents.list()[0]["agent_id"]
        return call("subagent", action="merge", agent_id=ident)

    parent = Script(
        [
            call(
                "subagent", action="spawn", task="Create src/result.txt", mode="worker"
            ),
            finish("Premature"),
            collect,
            finish("Still unmerged"),
            merge,
            call("read_file", file_path="src/result.txt"),
            finish(
                "Integrated and read worker output",
                evidence=["Read src/result.txt after merge"],
            ),
        ]
    )
    with Agent(settings, llm=parent) as agent:
        agent.loop.tool.subagents.agent_factory = lambda child_config: Agent(
            child_config, llm=worker
        )
        result = agent.act(task("Delegate a file creation and inspect it"))
        assert result.response == "Integrated and read worker output"
        assert agent.last_run_metrics["completed"]
        assert agent.last_run_metrics["resource_blocks"] == 2
        assert (tmp_path / "src/result.txt").read_text().strip() == "worker output"
