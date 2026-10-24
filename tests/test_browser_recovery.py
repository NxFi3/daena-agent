
import pytest

from src.tools.builtin.browser_use.executable import resolve_executable
from src.tools.builtin.browser_use.contracts import BrowserError


def executable(tmp_path, name):
    path = tmp_path / name
    path.write_text("browser placeholder")
    path.chmod(0o755)
    return str(path)


def test_browser_resolution_precedence_and_missing_explicit_path(tmp_path, monkeypatch):
    configured = executable(tmp_path, "configured")
    bundled = executable(tmp_path, "bundled")
    system = executable(tmp_path, "chrome")
    monkeypatch.setattr("shutil.which", lambda name: system)
    assert resolve_executable({"executable_path": configured}, bundled) == (
        configured,
        "configured",
    )
    assert resolve_executable({}, bundled) == (bundled, "playwright")
    assert resolve_executable({}, str(tmp_path / "missing")) == (system, "system")
    with pytest.raises(BrowserError, match="Configured browser executable"):
        resolve_executable({"executable_path": str(tmp_path / "bad")}, bundled)


def test_empty_generation_retries_and_discovered_tools_survive_followup(tmp_path):
    from src.agent.agent import Agent
    from src.models.LLMResult import LLMResult

    class Model:
        model = None

        def __init__(self):
            self.calls = 0

        def generate(self, messages, **kwargs):
            self.calls += 1
            return LLMResult("" if self.calls == 1 else "Ready", {}, [], None, 1)

    config = {
        "memory": {"database": str(tmp_path / "history.db")},
        "workspace": {"directory": str(tmp_path)},
        "max_agent_iterations": 4,
    }
    model = Model()
    agent = Agent(config, llm=model)
    try:
        # Use the runtime's public loop entry point for both conversation turns.
        from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
        from src.agent.agentloop import Loop

        loop = Loop(config, model)
        try:
            loop.tool._search_and_activate_tools("browser_use", 1)
            task = ContextEvent(
                role=ContextRole.USER, type=ContextType.MESSAGE, content="Hello"
            )
            assert loop.run(task, str(tmp_path)).response == "Ready"
            assert loop.get_metrics()["generation_retries"] == 1
            assert "browser_use" in loop.tool._active_tool_names
            assert loop.run(task, str(tmp_path)).response == "Ready"
            assert "browser_use" in loop.tool._active_tool_names
        finally:
            loop.close()
    finally:
        agent.close()


def test_named_tool_search_outranks_description_overlap(tmp_path):
    from src.tools.ToolManager import ToolManager

    manager = ToolManager({})
    try:
        manager.set_workspace(str(tmp_path))
        result = manager._search_and_activate_tools("browser use open url", 3)
        assert result["matches"][0]["name"] == "browser_use"
    finally:
        manager.close()


def test_empty_generation_exhausts_configured_retries(tmp_path):
    from src.agent.agentloop import Loop
    from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
    from src.models.LLMResult import LLMResult

    class Empty:
        model = None

        def generate(self, *args, **kwargs):
            return LLMResult("", {}, [], None, 1)

    loop = Loop(
        {
            "memory": {"database": str(tmp_path / "memory.db")},
            "runtime": {"generation_retries": 1},
            "max_agent_iterations": 5,
        },
        Empty(),
    )
    try:
        loop.run(
            ContextEvent(
                role=ContextRole.USER, type=ContextType.MESSAGE, content="Hello"
            ),
            str(tmp_path),
        )
        assert not loop.get_metrics()["completed"]
        assert loop.get_metrics()["llm_calls"] == 2
        assert "Recovery limit" in loop.get_metrics()["stop_reason"]
    finally:
        loop.close()
