import base64
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from src.agent.agent import Agent
from src.context.contextbuilder import ContextBuilder
from src.context.tokenbudget import TokenBudget
from src.engine.providers.builtin.ollama.ollama import OllamaProvider
from src.engine.transport import canonical_messages
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult
from src.tools.builtin.computer_use.integration import (
    desktop_messages,
    remember_result,
    observation_message,
)
from tests.computer_use.conftest import FakeBackend, capture
from tests.test_context_continuity import SummaryModel, exchange, build, task


def loop_for(tool, provider="ollama"):
    return SimpleNamespace(
        config={"llm": {"provider": provider}},
        tool=SimpleNamespace(get_tool=lambda name: tool),
    )


def test_current_frame_persists_after_errors_inspect_and_tool_instance_recreation(
    engine,
):
    tool, backend = engine
    backend.rows.append(dict(backend.rows[1], element_token="duplicate"))
    sid = capture(tool)
    result = tool.execute("click", snapshot_id=sid, target="Save")
    loop = loop_for(tool)
    remember_result(loop, SimpleNamespace(name="computer_use"), result)
    first = desktop_messages(loop, {"computer_use"})
    assert first[0]["images"]
    assert sid in first[0]["content"]
    assert tool.execute("inspect", snapshot_id=sid, query="Save").success
    assert desktop_messages(loop, {"computer_use"})[0]["images"] == first[0]["images"]
    clone = type(tool)(backend, workspace=tool.workspace_root, lock_path=tool.lock_path)
    assert (
        desktop_messages(loop_for(clone), {"computer_use"})[0]["images"]
        == first[0]["images"]
    )
    assert desktop_messages(loop, set()) == []
    tool.set_context_store(None, lambda: "different-session")
    assert desktop_messages(loop, {"computer_use"}) == []


def test_zoom_retains_mapping_and_next_capture_replaces_it(engine):
    tool, _ = engine
    sid = capture(tool)
    zoom = tool.execute("zoom", snapshot_id=sid, region=[10, 20, 100, 80], scale=2)
    loop = loop_for(tool)
    remember_result(loop, SimpleNamespace(name="computer_use"), zoom)
    message = desktop_messages(loop, {"computer_use"})[0]
    state = json.loads(message["content"].split("\n", 1)[1])
    assert state["origin"] == [10, 20] and state["scale_x"] == 2
    assert state["width"] == 640 and state["attached_image_dimensions"] == [200, 160]
    capture(tool)
    state = json.loads(
        desktop_messages(loop, {"computer_use"})[0]["content"].split("\n", 1)[1]
    )
    assert "mapping" not in state and state["attached_image_dimensions"] == [640, 480]


def test_consumed_or_expired_frame_does_not_claim_current_actionability(engine):
    tool, _ = engine
    sid = capture(tool)
    assert tool.execute(
        "click", snapshot_id=sid, target="Save", capture_after=False
    ).success
    state = tool.current_observation()
    assert not state["input_allowed"] and state["snapshot_consumed"]
    assert state["last_input"]["dispatch_status"] == "dispatched"
    capture(tool)
    tool.max_snapshot_age = -1
    assert not tool.current_observation()["input_allowed"]


def test_missing_image_is_explicit_and_workspace_changes_do_not_leak_frames(
    engine, tmp_path
):
    tool, _ = engine
    capture(tool)
    state = tool.current_observation()
    state["image_ref"] = str(tmp_path.parent / "secret.png")
    msg = observation_message(state, "ollama", tool.workspace_root)
    assert "images" not in msg and "image_unavailable" in msg["content"]
    tool.set_workspace(tmp_path / "other")
    assert desktop_messages(loop_for(tool), {"computer_use"}) == []


@pytest.mark.parametrize("provider", ["ollama", "openrouter", "gemini"])
def test_provider_image_bytes_and_budget_are_consistent(engine, provider):
    tool, _ = engine
    capture(tool)
    messages = desktop_messages(loop_for(tool, provider), {"computer_use"})
    budget = TokenBudget({"context": {"image_token_reserve": 3072}}, SummaryModel())
    assert budget.image_tokens(messages) == 3072
    assert 3072 < budget.estimate_messages_tokens(messages) < 6000
    if provider == "ollama":
        encoded = OllamaProvider._prepare_messages(messages)[0]["images"][0]
    elif provider == "openrouter":
        encoded = canonical_messages(messages)[0]["content"][1]["image_url"][
            "url"
        ].split(",", 1)[1]
    else:
        pytest.importorskip("google.genai")
        from src.engine.providers.builtin.gemini.geminiprovider import GeminiProvider

        native, _ = GeminiProvider._convert_messages(messages)
        assert native[0].parts[1].inline_data.data.startswith(b"\x89PNG")
        encoded = messages[0]["computer_image"]
    assert base64.b64decode(encoded).startswith(b"\x89PNG")
    before = budget.chars_per_token
    budget.calibrate_from_response(
        messages, LLMResult("", {}, [], None, 0, raw={"prompt_eval_count": 6000})
    )
    assert budget.chars_per_token == before


def test_images_count_as_reserved_tokens_not_base64_characters():
    budget = TokenBudget({"context": {"image_token_reserve": 4096}}, SummaryModel())

    def image(size):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "screen"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + "a" * size},
                    },
                ],
            }
        ]

    assert budget.estimate_messages_tokens(
        image(100)
    ) == budget.estimate_messages_tokens(image(1_000_000))


def test_context_budgets_pin_current_frame_and_keep_receipts_valid_json(engine):
    tool, _ = engine
    capture(tool)
    attachment = desktop_messages(loop_for(tool), {"computer_use"})
    builder = ContextBuilder(
        {"context": {"max_prompt_tokens": 8000, "compaction_enabled": False}},
        SummaryModel(),
    )
    user = task()
    events = [user]
    for i in range(1, 15):
        events.extend(exchange(i))
    messages = build(builder, events, user, attachments=attachment)
    assert messages[-1] == attachment[0]
    assert builder.tokenbudget.fits(messages)
    for message in messages:
        if message.get("role") == "tool":
            assert isinstance(json.loads(message["content"]), dict)
    tiny = ContextBuilder({"context": {"max_prompt_tokens": 3000}}, SummaryModel())
    with pytest.raises(ValueError, match="attached observations"):
        build(tiny, [user], user, attachments=attachment)


def test_real_agent_loop_sees_current_state_on_each_action_without_image_history(
    tmp_path,
):
    class Model:
        model = None

        def __init__(self):
            self.calls = []

        def generate(self, messages, **kwargs):
            self.calls.append(deepcopy(messages))
            index = len(self.calls)
            state = None
            if index > 2:
                assert len([m for m in messages if m.get("images")]) == 1
                state = json.loads(messages[-1]["content"].split("\n", 1)[1])
            if index == 1:
                name, args = (
                    "tool_search",
                    {"query": "computer desktop control", "top_k": 1},
                )
            elif index == 2:
                name, args = "computer_use", {"action": "capture", "window_id": 99}
            elif index == 3:
                name, args = (
                    "computer_use",
                    {
                        "action": "click",
                        "snapshot_id": state["snapshot_id"],
                        "target": "Name",
                        "delivery_mode": "foreground",
                        "seconds": 0,
                    },
                )
            elif index == 4:
                assert state["delivery_mode"] == "foreground"
                assert state["last_input"]["action"] == "click"
                name, args = (
                    "computer_use",
                    {
                        "action": "type",
                        "snapshot_id": state["snapshot_id"],
                        "text": "hello",
                        "seconds": 0,
                    },
                )
            elif index == 5:
                assert state["last_input"]["action"] == "type"
                name, args = (
                    "computer_use",
                    {
                        "action": "inspect",
                        "snapshot_id": state["snapshot_id"],
                        "query": "Name",
                    },
                )
            else:
                assert state["elements"][0]["value"] == "hello"
                return LLMResult("Typed and inspected the result.", {}, [], None, 1)
            call = {
                "id": f"c-{index}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            return LLMResult(
                "",
                {"role": "assistant", "content": "", "tool_calls": [call]},
                [call],
                None,
                1,
            )

    model = Model()
    config = {
        "llm": {
            "provider": "ollama",
            "provider_config": {"generation_config": {"num_ctx": 32768}},
        },
        "workspace": {"directory": str(tmp_path)},
        "memory": {"database": str(tmp_path / "history.db")},
        "context": {"compaction_enabled": False},
        "security": {"allowed_tools": ["tool_search", "computer_use"]},
        "max_agent_iterations": 8,
    }
    with Agent(config, llm=model) as agent:
        tool = agent.loop.tool.get_tool("computer_use")
        backend = FakeBackend()
        tool.backend = backend
        tool.lock_path = tmp_path / "desktop.lock"
        agent.act(
            ContextEvent(
                role=ContextRole.USER,
                type=ContextType.MESSAGE,
                content="Type hello into the Name field and inspect it.",
            )
        )
        assert agent.last_run_metrics["completed"]
        assert len(model.calls) == 6
        assert agent.last_run_metrics["context_image_tokens"] == 4096
        assert [a for a, _ in backend.calls] == ["click", "type_text"]
        assert all(args["delivery_mode"] == "foreground" for _, args in backend.calls)
        events = agent.loop.stm.get_recent(agent.loop.session_id, limit=100)
        assert not any(
            "data:image" in str(e.content) or '"images"' in str(e.content)
            for e in events
        )
