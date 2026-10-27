"""Action contracts and feedback are independent of application workflows."""

import json

import pytest

from src.tools.ToolDispatcher import ToolDispatcher
from src.tools.ToolRegistry import ToolRegistry
from tests.computer_use.conftest import FakeBackend, capture


def test_invalid_action_explains_primitives_without_guessing(engine):
    tool, backend = engine
    registry = ToolRegistry()
    registry.tools[tool.name] = tool
    call = ToolDispatcher(registry).dispatch(
        [
            {
                "id": "bad",
                "function": {
                    "name": "computer_use",
                    "arguments": json.dumps(
                        {
                            "action": "interact",
                            "coordinate": [10, 20],
                            "snapshot_id": "none",
                        }
                    ),
                },
            }
        ]
    )[0]
    assert not call.valid
    assert "never interact" in call.validation_error
    assert "type requires text" in call.validation_error
    assert not backend.calls


@pytest.mark.parametrize(
    "action,args",
    [
        ("key", {"key": "Enter", "coordinate": [4, 5]}),
        ("click", {"coordinate": [4, 5], "text": "ignored before"}),
        (
            "drag",
            {
                "path": [[1, 1], [2, 2]],
                "from_coordinate": [1, 1],
                "to_coordinate": [2, 2],
            },
        ),
        ("key", {"key": "Enter", "target": "Save"}),
    ],
)
def test_irrelevant_or_conflicting_arguments_do_not_consume_frame(engine, action, args):
    tool, backend = engine
    sid = capture(tool)
    result = tool.execute(action, snapshot_id=sid, **args)
    assert not result.success
    assert result.content["dispatch_status"] == "not_dispatched"
    assert not backend.calls
    assert tool.current_observation()["input_allowed"]


def test_foreground_preference_is_selected_before_input_but_explicit_mode_wins(engine):
    tool, backend = engine
    backend.capabilities = lambda session, window: {
        "preferred_delivery_mode": "foreground",
        "delivery_modes": {"foreground": True, "background": False},
    }
    first = tool.execute("capture")
    assert first.content["delivery_mode"] == "foreground"
    result = tool.execute(
        "click",
        snapshot_id=first.content["snapshot_id"],
        coordinate=[10, 20],
        seconds=0,
    )
    assert backend.calls[-1][1]["delivery_mode"] == "foreground"
    assert result.content["observation"]["delivery_mode"] == "foreground"
    explicit = tool.execute("capture", delivery_mode="background")
    assert explicit.content["delivery_mode"] == "background"
    assert tool.execute("capture").content["delivery_mode"] == "background"


def test_receipt_keeps_target_and_accessible_state_without_duplicating_typed_text(
    engine,
):
    tool, backend = engine
    sid = capture(tool)
    result = tool.execute("click", snapshot_id=sid, target="Enabled", seconds=0)
    assert result.success
    assert result.content["accessibility_delta"]["changed_controls"] == 1
    assert "checked" in result.summary
    receipt = tool.current_observation()["last_input"]
    assert receipt["target"] == {"target": "Enabled"}
    assert receipt["next_snapshot_id"] == result.content["observation"]["snapshot_id"]
    result = tool.execute(
        "type",
        snapshot_id=receipt["next_snapshot_id"],
        text="private text",
        target="Name",
        seconds=0,
    )
    receipt = tool.current_observation()["last_input"]
    assert receipt["target"]["typed_characters"] == 12
    assert "text" not in receipt["target"]
    assert result.content["accessibility_delta"]["status"] == "changed"


def test_settling_observes_delayed_change_without_replaying_input(engine):
    tool, backend = engine
    sid = capture(tool)
    original = backend.capture

    def delayed(*args, **kwargs):
        if backend.captures >= 2:
            backend.rows[0]["value"] = "asynchronously updated"
        return original(*args, **kwargs)

    backend.capture = delayed
    result = tool.execute("click", snapshot_id=sid, target="Save")
    assert result.success
    assert len(backend.calls) == 1
    assert result.content["observation"]["settling"]["status"] == "quiet"
    assert result.content["observation"]["settling"]["samples"] >= 3
    assert result.content["accessibility_delta"]["changed_controls"] == 1
    assert result.content["verification"]["status"] == "not_requested"


def test_continuous_animation_returns_observation_at_settling_deadline(engine):
    tool, backend = engine
    sid = capture(tool)
    original = backend.capture

    def changing(*args, **kwargs):
        backend.rows[0]["value"] = str(backend.captures)
        return original(*args, **kwargs)

    backend.capture = changing
    result = tool.execute("click", snapshot_id=sid, target="Save", settle_seconds=0.4)
    assert result.success
    assert len(backend.calls) == 1
    assert result.content["observation"]["settling"]["status"] == "deadline"
    assert "still changing" in result.summary


def test_polyline_validates_every_vertex_before_delivery(engine):
    tool, backend = engine
    sid = capture(tool)
    result = tool.execute(
        "drag", snapshot_id=sid, path=[[20, 20], [640, 90], [50, 50]], seconds=0
    )
    assert not result.success
    assert not backend.calls
    assert tool.current_observation()["input_allowed"]


def test_navigation_honors_explicit_observation_controls(engine):
    tool, backend = engine
    sid = capture(tool)
    result = tool.execute(
        "navigate", snapshot_id=sid, url="https://example.org", seconds=0, mode="vision"
    )
    assert result.success, result.content
    assert len(backend.calls) == 3
    assert result.content["observation"]["mode"] == "vision"
    assert "settling" not in result.content["observation"]


def test_wait_rejects_a_different_snapshot_without_observing_wrong_target(engine):
    tool, backend = engine
    capture(tool)
    count = backend.captures
    result = tool.execute("wait", snapshot_id="s_not_observed_here", seconds=0)
    assert not result.success
    assert backend.captures == count
    assert not backend.calls


def test_refused_first_long_text_chunk_keeps_explicit_zero_delivery(engine):
    from src.tools.builtin.computer_use.runtime import ComputerError

    tool, backend = engine
    sid = capture(tool)
    backend.failure = ComputerError(
        "background_unavailable", "No background route", not_dispatched=True
    )
    result = tool.execute("type", snapshot_id=sid, text="x" * 900, seconds=0)
    assert not result.success
    assert result.content["dispatch_status"] == "not_dispatched"
    assert not result.content["snapshot_consumed"]
    assert len(backend.calls) == 1
    assert "partial" not in result.summary.lower()


def test_capture_capabilities_use_public_actions_not_driver_verbs(engine):
    tool, backend = engine
    backend.capabilities = lambda session, window: {
        "actions": ["click", "type_text", "press_key", "hotkey"],
        "known_delivery_routes": [{"action": "type_text", "supported": True}],
    }
    result = tool.execute("capture")
    assert result.content["capabilities"]["actions"] == ["click", "key", "type"]
    assert (
        result.content["capabilities"]["known_delivery_routes"][0]["action"] == "type"
    )
