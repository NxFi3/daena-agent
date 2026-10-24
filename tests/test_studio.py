"""Offscreen end-to-end desktop tests using real runtime tools, no model server."""

import json
import os
import time
from threading import Event

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtTest import QTest

from src.agent.agent import Agent
from src.models.LLMResult import LLMResult
from src.studio.window import StudioWindow
from src.studio.state import StudioStore, default_config, validate_config
from src.studio.blueprint import Blueprint, Node, execute


class Script:
    model = None

    def __init__(self, results):
        self.results = iter(results)
        self.messages = []

    def generate(self, messages, **kwargs):
        self.messages.append(messages)
        result = next(self.results)
        return result(messages) if callable(result) else result


def call(tool_name, **arguments):
    return LLMResult(
        "",
        {},
        [
            {
                "id": str(time.time_ns()),
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(arguments)},
            }
        ],
        None,
        1,
    )


@pytest.fixture(scope="module")
def app(qt_app):
    return qt_app


@pytest.fixture
def window(tmp_path, app, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user-data"))
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: errors.append(a[-1]))
    w = StudioWindow(tmp_path)
    w._errors = errors
    yield w
    if w.busy:
        w.stop()
        wait_until(lambda: not w.busy)
    w.close()
    app.processEvents()
    assert not errors


def wait_until(predicate, timeout=5):
    end = time.monotonic() + timeout
    while not predicate() and time.monotonic() < end:
        QApplication.processEvents()
        QTest.qWait(10)
    assert predicate(), "UI operation timed out"


def test_desktop_real_file_tools_history_and_followup(window, tmp_path):
    model = Script(
        [
            call(
                "apply_patch",
                patch="*** Begin Patch\n*** Add File: result.txt\n+studio output\n*** End Patch",
            ),
            call("read_file", file_path="result.txt"),
            LLMResult("Created and checked result.txt", {}, [], None, 1),
            LLMResult("Salam!", {}, [], None, 1),
        ]
    )
    window.agent_factory = lambda config: Agent(config, llm=model)
    window.input.setPlainText("Create result.txt and check it")
    window.send()
    wait_until(lambda: not window.busy)
    assert (tmp_path / "result.txt").read_text().strip() == "studio output"
    assert "Created and checked result.txt" in window.transcript.toPlainText()
    assert window.history.count() == 1 and window._result["metrics"]["completed"]
    ident = window.session_id
    window.open_history(window.history.item(0))
    assert window.session_id == ident
    window.input.setPlainText("salam")
    window.send()
    wait_until(lambda: not window.busy)
    assert "Salam!" in window.transcript.toPlainText()
    assert window.history.count() == 1
    assert len(window.store.runs()) == 2


def test_desktop_stop_and_close_are_cooperative(window):
    entered = Event()

    class Waiting:
        model = None

        def generate(self, messages, stop_event=None, **kwargs):
            entered.set()
            while not stop_event.is_set():
                time.sleep(0.01)
            raise InterruptedError("Stopped by operator")

    window.agent_factory = lambda config: Agent(config, llm=Waiting())
    window.input.setPlainText("Wait")
    window.send()
    wait_until(entered.is_set)
    window.close()
    wait_until(lambda: not window.busy)
    assert not window._result["metrics"]["completed"]


def test_graph_reports_real_worker_and_integrated_files(window, tmp_path):
    worker = Script(
        [
            call(
                "apply_patch",
                patch="*** Begin Patch\n*** Add File: child.txt\n+child output\n*** End Patch",
            ),
            LLMResult("Created child.txt", {}, [], None, 1),
        ]
    )

    runtime = {}

    def wait(messages):
        ident = runtime["agent"].loop.tool.subagents.list()[0]["agent_id"]
        return call("subagent", action="wait", agent_id=ident, wait_ms=3000)

    def merge(messages):
        ident = runtime["agent"].loop.tool.subagents.list()[0]["agent_id"]
        return call("subagent", action="merge", agent_id=ident)

    parent = Script(
        [
            call(
                "subagent",
                action="spawn",
                mode="worker",
                name="Implementation",
                task="Create child.txt",
            ),
            wait,
            merge,
            LLMResult("Integrated", {}, [], None, 1),
        ]
    )

    def factory(config):
        agent = Agent(config, llm=parent)
        runtime["agent"] = agent
        agent.loop.tool.subagents.agent_factory = lambda c: Agent(c, llm=worker)
        return agent

    window.agent_factory = factory
    window.input.setPlainText("Delegate and integrate")
    window.send()
    wait_until(lambda: not window.busy)
    assert window._result["metrics"]["completed"], window._result
    assert (tmp_path / "child.txt").read_text().strip() == "child output"
    assert len(window.graph.nodes) == 2 and len(window.graph.edges) == 1
    child = next(iter(window.children))
    window.inspect_child(child)
    assert "merged" in window.agent_detail.toPlainText()
    assert window.graph.nodes[child].status == "completed"
    window.nav.setCurrentRow(1)
    window.show()
    QApplication.processEvents()
    window.grab().save(str(tmp_path / "graph.png"))


def test_settings_persist_tool_scope_and_generation_options(window):
    settings = window.settings
    settings.image_reserve.setValue(3072)
    settings.browser_headless.setChecked(True)
    settings.browser_executable.setText("/usr/bin/chromium")
    settings.select_tools(False)
    settings.apply_form()
    assert settings.value()["browser"]["headless"] is True
    assert settings.value()["browser"]["executable_path"] == "/usr/bin/chromium"
    saved = json.loads(window.store.profile.read_text())
    assert saved["security"]["allowed_tools"] == []
    assert saved["context"]["image_token_reserve"] == 3072
    assert window.config["security"]["allowed_tools"] == []
    settings.provider.setCurrentText("openrouter")
    assert "num_ctx" not in json.loads(settings.generation.toPlainText())
    assert settings.model.text() == "openrouter/free"


def test_blueprint_editor_wires_ports_and_rejects_cycles(window):
    panel = window.automation
    panel.add_node()
    panel.add_node()
    a, b = panel.graph.nodes
    panel.wire(a.id, b.id)
    assert panel.graph.nodes[1].depends_on == [a.id]
    failures = []
    panel.error.disconnect()
    panel.error.connect(failures.append)
    panel.wire(b.id, a.id)
    assert failures and "cycle" in failures[0]
    assert panel.graph.nodes[0].depends_on == []
    restored = Blueprint.load(panel.graph.dump())
    assert [n.id for n in restored.order()] == [a.id, b.id]


def test_blueprint_executes_independent_real_agents_with_upstream_reports(tmp_path):
    config = StudioStore(tmp_path).runtime_config(default_config())
    models = []
    events = []

    def factory(config):
        if not models:
            model = Script(
                [
                    call(
                        "apply_patch",
                        patch="*** Begin Patch\n*** Add File: flow.txt\n+flow output\n*** End Patch",
                    ),
                    LLMResult("Created flow.txt", {}, [], None, 1),
                ]
            )
        else:

            def check(messages):
                assert "Report from build" in str(messages)
                return call("read_file", file_path="flow.txt")

            model = Script([check, LLMResult("Verified flow.txt", {}, [], None, 1)])
        models.append(model)
        return Agent(config, llm=model)

    graph = Blueprint(
        "Build and check",
        [
            Node("build", "Build", "Create a file", tools=["apply_patch"]),
            Node(
                "check",
                "Check",
                "Read and verify",
                depends_on=["build"],
                tools=["read_file"],
            ),
        ],
    )
    result = execute(graph, config, Event(), events.append, factory)
    assert result["completed"] and (tmp_path / "flow.txt").exists()
    assert [e["status"] for e in events if e["type"] == "automation_node"] == [
        "running",
        "completed",
        "running",
        "completed",
    ]
    assert len(models) == 2


def test_blueprint_failure_stops_downstream_nodes(tmp_path):
    config = StudioStore(tmp_path).runtime_config(default_config())
    graph = Blueprint("flow", [Node("a", "A", "One"), Node("b", "B", "Two", ["a"])])
    calls = []

    def factory(c):
        calls.append(1)
        return Agent(
            {**c, "max_agent_iterations": 1},
            llm=Script([call("read_file", file_path="missing.txt")]),
        )

    result = execute(graph, config, Event(), lambda e: None, factory)
    assert not result["completed"] and result["failed_node"] == "a" and len(calls) == 1


def test_profile_validation_rejects_bad_limits_and_preserves_empty_tool_scopes():
    assert (
        validate_config({"security": {"allowed_tools": []}})["security"][
            "allowed_tools"
        ]
        == []
    )
    with pytest.raises(ValueError):
        validate_config({"context": {"recent_event_limit": "all"}})
    with pytest.raises(ValueError):
        validate_config(
            {
                "runtime": {"completion_protocol": "explicit"},
                "security": {"allowed_tools": []},
            }
        )


def test_open_history_restores_original_workspace_profile(window, tmp_path):
    from uuid import uuid4

    other = tmp_path / "other"
    other.mkdir()
    store = StudioStore(other)
    config = default_config()
    config["llm"]["provider"] = "gemini"
    config["llm"]["provider_config"]["model_name"] = "workspace-model"
    store.save_config(config)
    session = uuid4()
    store.save_run(session, "Other workspace", [], {"completed": True})
    window.refresh_history()
    window.open_history(window.history.item(0))
    assert window.workspace == other
    assert window.config == config
    assert window.session_id == session
    assert window.workspace_edit.text() == str(other)


def test_open_history_missing_workspace_preserves_current_conversation(
    window, tmp_path
):
    from uuid import uuid4
    import shutil

    other = tmp_path / "removed"
    other.mkdir()
    store = StudioStore(other)
    store.save_run(uuid4(), "Missing workspace", [], {"completed": True})
    shutil.rmtree(other)
    before = window.session_id
    window.refresh_history()
    window.open_history(window.history.item(0))
    assert window.session_id == before and window.workspace == tmp_path
    assert "unavailable" in window._errors.pop()
