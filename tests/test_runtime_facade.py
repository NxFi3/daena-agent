import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.agent import Agent
from src.engine.LlmProviderManager import LlmProvider
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult
from src.models.ToolCall import ToolCall
from src.security.Policy import SecurityPolicy
from src.security.securityService import SecurityService
from src.tools.ToolManager import ToolManager


def test_provider_wrapper_returns_canonical_result(monkeypatch):
    expected = LLMResult("response", {}, [], None, 42)
    inputs = []

    class Model:
        defaultModel = "test-model"

        def generate(self, value):
            inputs.append(value)
            return expected

    class Registry:
        def discover(self):
            return ["test"]

        def is_available(self, name):
            return name == "test"

        def get(self, name):
            return Model()

    monkeypatch.setattr("src.engine.LlmProviderManager.ProviderRegistry", Registry)
    provider = LlmProvider(
        {
            "llm": {
                "provider": "test",
                "provider_config": {"generation_config": {"temperature": 0.1}},
            }
        }
    )
    assert provider.generate([], options={"temperature": 0.5}) is expected
    assert inputs[0].model_name == "test-model"
    assert inputs[0].options["temperature"] == 0.5
    with pytest.raises(ValueError, match="unavailable"):
        LlmProvider({"llm": {"provider": "missing"}})


def test_empty_allowlists_deny_everything(tmp_path):
    service = SecurityService({"allowed_tools": []})
    service.set_workspace(str(tmp_path))
    assert not service.check(
        ToolCall(name="read_file", args={"file_path": "x"}, valid=True)
    ).approved
    service = SecurityService({"allowed_commands": []})
    service.set_workspace(str(tmp_path))
    assert not service.check(
        ToolCall(name="command_exec", args={"command": ["ls"]}, valid=True)
    ).approved


@pytest.mark.parametrize(
    "config",
    [
        [],
        False,
        {"workspace_only": "false"},
        {"allowed_tools": "read_file"},
        {"allow_background": 1},
    ],
)
def test_invalid_policy_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        SecurityPolicy.from_config(config)


def test_cancellation_blocks_tools_at_execution_boundary(tmp_path):
    manager = ToolManager()
    manager.set_workspace(str(tmp_path))
    manager.stop_event = threading.Event()
    manager.stop_event.set()
    try:
        call = ToolCall(
            name="apply_patch",
            valid=True,
            args={
                "patch": "*** Begin Patch\n*** Add File: forbidden.txt\n+no\n*** End Patch"
            },
        )
        result = manager.execute([call])["results"][0]
        assert not result.success
        assert result.metadata["security_rule"] == "task_cancelled"
        assert not (tmp_path / "forbidden.txt").exists()
    finally:
        manager.close()


def test_agent_reentrancy_cleanup_and_closed_state(tmp_path):
    config = {
        "memory": {"database": str(tmp_path / "memory.db")},
        "workspace": {"directory": str(tmp_path)},
    }
    agent = Agent(config, llm=SimpleNamespace(model=None))
    task = ContextEvent(role=ContextRole.USER, type=ContextType.MESSAGE, content="test")
    agent._run_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            agent.act(task)
        with pytest.raises(RuntimeError, match="workspace"):
            agent.set_workingdirectory(str(tmp_path / "other"))
        with pytest.raises(RuntimeError, match="Stop"):
            agent.close()
    finally:
        agent._run_lock.release()
    agent.close()
    agent.close()
    with pytest.raises(RuntimeError, match="closed"):
        agent.act(task)


def test_default_config_and_bundled_config_match():
    root = Path(__file__).resolve().parents[1]
    assert json.loads((root / "config.json").read_text()) == json.loads(
        (root / "src/default_config.json").read_text()
    )
