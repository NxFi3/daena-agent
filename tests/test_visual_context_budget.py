"""Budget pressure must not erase current visual evidence or tool contracts."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.agent.agentloop import Loop
from src.context.contextbuilder import ContextBudgetError, ContextBuilder
from src.context.tokenbudget import TokenBudget
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult
from src.tools.ToolManager import ToolManager
from tests.computer_use.conftest import FakeBackend
from tests.test_context_continuity import SummaryModel, build


def defaults(tmp_path):
    config = json.loads(Path("src/default_config.json").read_text())
    config["memory"] = {"database": str(tmp_path / "memory.db")}
    config["experience"] = {"enabled": False}
    return config


def test_first_capture_recovers_schema_pressure_at_default_context(tmp_path):
    config = defaults(tmp_path)
    events = []
    delivered = []

    class Model:
        model = None

        def generate(self, messages, tools=None, **kwargs):
            delivered.append(deepcopy(messages))
            names = {t["function"]["name"] for t in tools}
            assert {"computer_use", "tool_search"} <= names
            if len(delivered) == 1:
                call = {
                    "id": "capture-1",
                    "type": "function",
                    "function": {
                        "name": "computer_use",
                        "arguments": json.dumps({"action": "capture", "window_id": 99}),
                    },
                }
                return LLMResult(
                    "", {"role": "assistant", "tool_calls": [call]}, [call], None, 1
                )
            assert any(m.get("images") for m in messages), [
                (m.get("role"), str(m.get("content"))[:250]) for m in messages
            ]
            assert any(m.get("tool_call_id") == "capture-1" for m in messages), [
                (m.get("role"), m.get("tool_call_id"), str(m.get("content"))[:250])
                for m in messages
            ]
            return LLMResult("Observed the window.", {}, [], None, 1)

    loop = Loop(config, Model())
    loop.tool._active_tool_names = ["computer_use", "browser_use", "subagent"]
    desktop = loop.tool.get_tool("computer_use")
    desktop.backend = FakeBackend()
    desktop.lock_path = tmp_path / "desktop.lock"
    # A valid learned token density from earlier text turns. The first image
    # must still fit by paging schemas rather than deleting visual evidence.
    loop.context.contextbuilder.tokenbudget.chars_per_token = 1.5
    task = ContextEvent(
        role=ContextRole.USER,
        type=ContextType.MESSAGE,
        content="Use the existing desktop window and report what you observe.",
    )
    try:
        result = loop.run(
            task, workspace_directory=str(tmp_path), event_sink=events.append
        )
        assert loop.metrics["completed"], loop.metrics.get("stop_reason")
        assert result.response == "Observed the window."
        assert len(delivered) == 2
        assert any(
            e["type"] == "context_tools_reduced"
            and e["budget_details"]["image_tokens"] == 4096
            for e in events
        )
        attached = next(m["images"][0] for m in delivered[-1] if m.get("images"))
        import base64

        observation = desktop.current_observation()
        assert base64.b64decode(attached) == Path(observation["image_ref"]).read_bytes()
        assert all(
            e["estimated_tokens"] <= e["budget"]
            for e in events
            if e["type"] == "context"
        )
    finally:
        loop.close()


def test_schema_packing_keeps_complete_required_contracts_and_discovery(tmp_path):
    config = defaults(tmp_path)
    manager = ToolManager(config)
    manager._active_tool_names = ["computer_use", "browser_use"]
    full = {d["function"]["name"]: d for d in manager.get_tools()}
    reduced = manager.get_model_tools(
        schema_token_budget=4000, required_names={"computer_use"}
    )
    names = {d["function"]["name"] for d in reduced}
    assert {"computer_use", "tool_search"} <= names
    assert (
        next(d for d in reduced if d["function"]["name"] == "computer_use")
        == full["computer_use"]
    )
    budget = TokenBudget(config, SummaryModel())
    budget.reserve_tools(reduced)
    assert budget.tool_tokens <= 4000
    catalog = next(d for d in reduced if d["function"]["name"] == "tool_search")[
        "function"
    ]["description"]
    assert all(name in catalog for name in set(full) - names)


def test_impossible_image_budget_reports_components_without_discarding_image():
    config = {
        "llm": {
            "provider_config": {
                "generation_config": {"num_ctx": 8192, "num_predict": 8192}
            }
        }
    }
    builder = ContextBuilder(config, SummaryModel())
    task = ContextEvent(
        role=ContextRole.USER, type=ContextType.MESSAGE, content="Keep this instruction"
    )
    image = {
        "role": "user",
        "content": "Current observation",
        "images": ["private-image-bytes"],
    }
    original = deepcopy(image)
    with pytest.raises(ContextBudgetError) as failure:
        build(builder, [task], task, attachments=[image])
    details = failure.value.details
    assert details["image_tokens"] == 4096
    assert details["estimated_tokens"] > details["budget"]
    assert details["output_reserve"] == 8192
    assert image == original
    assert "private-image-bytes" not in str(failure.value)
