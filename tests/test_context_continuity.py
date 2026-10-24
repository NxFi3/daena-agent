import json
import threading

import pytest

from src.context.compactor import Compactor
from src.context.contextbuilder import ContextBuilder
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult


class SummaryModel:
    model = None

    def __init__(self):
        self.inputs = []

    def generate(self, messages, **kwargs):
        self.inputs.append(messages)
        return LLMResult(
            "Historical checkpoint: no files existed at the start.", {}, [], None, 1
        )


def task():
    return ContextEvent(
        role=ContextRole.USER,
        type=ContextType.MESSAGE,
        content="Build the application",
        step=1,
    )


def exchange(index, filename="file.txt", body=None):
    call_id = f"call-{index}"
    assistant = ContextEvent(
        role=ContextRole.ASSISTANT,
        type=ContextType.MESSAGE,
        content="",
        step=index * 2,
        metadata={
            "llm_message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "apply_patch",
                            "arguments": json.dumps({"patch": body or "x" * 12000}),
                        },
                    }
                ],
            }
        },
    )
    receipt = ContextEvent(
        role=ContextRole.TOOL,
        type=ContextType.TOOL_RESULT,
        step=index * 2 + 1,
        content=json.dumps(
            {
                "name": "apply_patch",
                "tool_call_id": call_id,
                "success": True,
                "summary": f"Created {filename}",
                "effects": [{"action": "add", "target": filename}],
                "evidence": {
                    "path": filename,
                    "preview": "real artifact " + "y" * 8000,
                },
            }
        ),
    )
    return [assistant, receipt]


def build(builder, events, user, **kwargs):
    return builder.build_context(
        events,
        task={"id": str(user.id), "content": user.content, "step": user.step},
        **kwargs,
    )


def test_large_patch_preserves_latest_receipt_and_valid_arguments():
    model = SummaryModel()
    builder = ContextBuilder(
        {"context": {"max_prompt_tokens": 4000, "compaction_enabled": False}}, model
    )
    user = task()
    pair = exchange(1, "index.html", "very long patch\n" * 20000)
    original = pair[0].metadata["llm_message"]["tool_calls"][0]["function"]["arguments"]
    messages = build(builder, [user, *pair], user)
    assert messages[1] == {"role": "user", "content": user.content}
    assert messages[-1]["role"] == "tool"
    assert "Created index.html" in messages[-1]["content"]
    assert messages[-2]["tool_calls"][0]["id"] == messages[-1]["tool_call_id"]
    assert isinstance(
        json.loads(messages[-2]["tool_calls"][0]["function"]["arguments"]), dict
    )
    assert builder.tokenbudget.fits(messages)
    assert (
        pair[0].metadata["llm_message"]["tool_calls"][0]["function"]["arguments"]
        == original
    )


def test_incremental_summary_keeps_newer_artifacts_and_never_resummarizes_consumed_events():
    model = SummaryModel()
    builder = ContextBuilder(
        {
            "context": {
                "max_prompt_tokens": 6000,
                "compaction_trigger_ratio": 0.5,
                "compaction_target_tokens": 256,
            }
        },
        model,
    )
    user = task()
    events = [user]
    for index in range(1, 13):
        events.extend(exchange(index, f"artifact-{index}.txt"))
    messages = build(
        builder,
        events,
        user,
        working_set={
            "workspace_inventory": [
                {"path": "index.html", "type": "file", "size": 32606}
            ],
            "workspace_file_count": 1,
        },
    )
    assert model.inputs
    assert "index.html" in messages[0]["content"]
    assert messages[-1]["tool_call_id"] == "call-12"
    assert "Created artifact-12.txt" in messages[-1]["content"]
    first_input = str(model.inputs[0])
    assert "call-1" in first_input and "call-12" not in first_input
    # Calibration changes transport bounds; stable event IDs still prevent replay.
    builder.tokenbudget.chars_per_token = 2.8
    events.extend(exchange(13, "new.css"))
    build(builder, events, user)
    for prompt in model.inputs[1:]:
        assert '"id": "call-1"' not in str(prompt)
    assert all("_history_id" not in message for message in messages)


def test_historical_tool_availability_does_not_erase_current_receipt():
    builder = ContextBuilder({"context": {"compaction_enabled": False}}, SummaryModel())
    user = task()
    messages = build(
        builder, [user, *exchange(1)], user, available_tool_names={"read_file"}
    )
    assert messages[-1]["role"] == "tool"
    assert messages[-2]["tool_calls"][0]["function"]["name"] == "apply_patch"


def test_compaction_interrupt_is_propagated_without_fallback():
    class Cancelled:
        def generate(self, messages, **kwargs):
            assert kwargs["stop_event"] is stop
            raise InterruptedError("cancelled during compaction")

    stop = threading.Event()
    with pytest.raises(InterruptedError, match="during compaction"):
        Compactor(Cancelled()).compact("history", 100, stop_event=stop)


def test_compaction_preserves_latest_steering_position():
    builder = ContextBuilder({"context": {"compaction_enabled": False}}, SummaryModel())
    user = task()
    steering = ContextEvent(
        role=ContextRole.USER,
        type=ContextType.MESSAGE,
        content="Only inspect now",
        step=4,
        metadata={"operator_steering": True},
    )
    messages = build(builder, [user, *exchange(1), steering], user)
    assert messages[-1] == {"role": "user", "content": "Only inspect now"}
    assert messages[-2]["role"] == "tool"


def test_sliding_retrieval_window_checkpoints_before_forgetting():
    model = SummaryModel()
    builder = ContextBuilder(
        {
            "context": {
                "max_prompt_tokens": 24000,
                "compaction_trigger_ratio": 0.99,
                "compaction_target_tokens": 512,
            }
        },
        model,
    )
    user = task()
    pairs = {i: exchange(i, f"file-{i}.txt", body="small") for i in range(1, 6)}
    build(builder, [user, *pairs[1], *pairs[2], *pairs[3]], user)
    assert not model.inputs
    result = build(builder, [user, *pairs[3], *pairs[4]], user)
    assert model.inputs
    assert "call-1" in str(model.inputs[0]) and "call-2" in str(model.inputs[0])
    assert result[-1]["tool_call_id"] == "call-4"
    assert "event:" + str(pairs[1][0].id) in builder._covered_groups
    # Recall of an older, covered exchange must not replay it as current evidence.
    result = build(builder, [user, *pairs[1], *pairs[4], *pairs[5]], user)
    assert not any(m.get("tool_call_id") == "call-1" for m in result)
    assert all(not any(k.startswith("_history") for k in m) for m in result)


def test_empty_compactor_uses_explicit_receipt_checkpoint():
    class Empty(SummaryModel):
        def generate(self, messages, **kwargs):
            self.inputs.append(messages)
            return LLMResult("", {}, [], None, 1)

    model = Empty()
    builder = ContextBuilder(
        {"context": {"max_prompt_tokens": 24000, "compaction_trigger_ratio": 0.99}},
        model,
    )
    user = task()
    one, two, three = (
        exchange(1, body="small"),
        exchange(2, body="small"),
        exchange(3, body="small"),
    )
    build(builder, [user, *one, *two], user)
    result = build(builder, [user, *two, *three], user)
    assert "receipt_only_checkpoint" in builder._history_summary
    assert result[-1]["tool_call_id"] == "call-3"
