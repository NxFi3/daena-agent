import json
import threading
import time

import pytest

from computer_use.contracts import Rect
from computer_use.runtime import ComputerError
from conftest import capture


def test_exact_labels_and_native_tokens(engine):
    tool, backend = engine
    sid = capture(tool)
    r = tool.execute("click", snapshot_id=sid, target="Save", seconds=0)
    assert r.success
    assert backend.calls == [
        (
            "click",
            {
                "delivery_mode": "background",
                "pid": 123,
                "window_id": 99,
                "element_token": "save",
                "button": "left",
            },
        )
    ]
    assert r.content["observation"]["snapshot_id"] != sid


def test_unchanged_pixels_are_not_a_failed_click_or_proof(engine):
    tool, backend = engine
    sid = capture(tool)
    r = tool.execute("click", snapshot_id=sid, coordinate=[45, 40], seconds=0)
    assert r.success
    assert r.content["visual_delta"]["status"] == "unchanged"
    assert r.content["verification"]["status"] == "not_requested"
    # Clicking the same point after a fresh observation is legitimate.
    r2 = tool.execute(
        "click", snapshot_id=r.content["observation"]["snapshot_id"], coordinate=[45, 40], seconds=0
    )
    assert r2.success and len(backend.calls) == 2


def test_stale_and_consumed_snapshot_prevent_input(engine):
    tool, backend = engine
    old = capture(tool)
    new = capture(tool)
    assert not tool.execute("click", snapshot_id=old, element=1).success
    assert tool.execute("click", snapshot_id=new, element=1, capture_after=False).success
    r = tool.execute("click", snapshot_id=new, element=1)
    assert r.content["error"]["code"] == "snapshot_consumed"
    assert len(backend.calls) == 1


def test_snapshot_survives_new_instance_but_ids_do_not_cross_sessions(engine):
    tool, backend = engine
    sid = capture(tool)
    clone = type(tool)(backend, workspace=tool.workspace_root, lock_path=tool.lock_path)
    assert clone.execute("click", snapshot_id=sid, target="Save", seconds=0).success
    clone.set_context_store(None, lambda: "another session")
    assert not clone.execute("click", snapshot_id=sid, target="Save").success


def test_ambiguous_target_does_not_choose_first(engine):
    tool, backend = engine
    backend.rows.append(dict(backend.rows[1], element_token="save2"))
    sid = capture(tool)
    r = tool.execute("click", snapshot_id=sid, target="save")
    assert r.content["error"]["code"] == "target_ambiguous"
    assert not backend.calls


def test_explicit_role_disambiguates(engine):
    tool, backend = engine
    backend.rows.append(dict(backend.rows[1], role="link", element_token="save2"))
    sid = capture(tool)
    assert tool.execute(
        "click", snapshot_id=sid, target={"label": "Save", "role": "button"}, seconds=0
    ).success
    assert backend.calls[0][1]["element_token"] == "save"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coordinate": [640, 20]},
        {"coordinate": [-1, 20]},
        {"coordinate": [True, 20]},
        {"coordinate": [float("nan"), 20]},
        {"coordinate": [float("inf"), 20]},
        {"coordinate": ["20", 20]},
        {"coordinate": [20]},
        {"bogus": True},
        {"amount": 21, "coordinate": [1, 2]},
        {"coordinate": [1, 2], "element": 2},
    ],
)
def test_invalid_input_never_reaches_driver(engine, kwargs):
    tool, backend = engine
    sid = capture(tool)
    assert not tool.execute("click", snapshot_id=sid, **kwargs).success
    assert not backend.calls


def test_resize_rejected_but_window_movement_preserves_local_coordinates(engine):
    tool, backend = engine
    sid = capture(tool)
    backend.window.bounds = Rect(250, 300, 640, 480)
    assert tool.execute("click", snapshot_id=sid, coordinate=[20, 20], seconds=0).success
    sid = capture(tool)
    backend.window.bounds = Rect(250, 300, 641.5, 480)
    r = tool.execute("click", snapshot_id=sid, coordinate=[20, 20])
    assert r.content["error"]["code"] == "geometry_changed"
    assert len(backend.calls) == 1


def test_checked_postcondition_verified_even_if_pixels_unchanged(engine):
    tool, _ = engine
    sid = capture(tool)
    r = tool.execute(
        "click",
        snapshot_id=sid,
        target="Enabled",
        seconds=0,
        expect={"kind": "checked", "target": "Enabled", "value": True},
    )
    assert r.success
    assert r.content["verification"]["status"] == "verified"
    assert r.content["visual_delta"]["status"] == "unchanged"


def test_selected_never_substitutes_for_checked(engine):
    tool, backend = engine
    del backend.rows[1]["actions"]
    backend.rows[1]["selected"] = True
    sid = capture(tool)
    r = tool.execute(
        "click",
        snapshot_id=sid,
        target="Save",
        seconds=0,
        expect={"kind": "checked", "target": "Save", "value": True},
    )
    assert not r.success and r.content["dispatch_status"] == "dispatched"
    assert r.content["verification"]["status"] == "unavailable"
    assert len(backend.calls) == 1


def test_wait_polls_only_observations_and_catches_delayed_change(engine):
    tool, backend = engine
    sid = capture(tool)
    thread = threading.Thread(
        target=lambda: (time.sleep(0.15), backend.rows[0].update(value="ready"))
    )
    thread.start()
    r = tool.execute(
        "click",
        snapshot_id=sid,
        target="Save",
        seconds=1,
        expect={"kind": "value", "target": "Name", "value": "ready"},
    )
    thread.join()
    assert r.success and len(backend.calls) == 1 and backend.captures >= 3


def test_absent_cannot_be_verified_from_truncated_tree(engine):
    tool, backend = engine
    backend.truncated = True
    capture(tool)
    r = tool.execute("wait", seconds=0, expect={"kind": "absent", "target": "Missing"})
    assert not r.success and r.content["verification"]["status"] == "unavailable"


def test_window_title_postcondition_uses_fresh_title(engine):
    tool, backend = engine
    capture(tool)
    backend.window.title = "Saved"
    r = tool.execute("wait", seconds=0, expect={"kind": "window_title", "value": "Saved"})
    assert r.success


def test_timeout_never_replays_or_escalates_delivery(engine):
    tool, backend = engine
    sid = capture(tool)
    backend.failure = ComputerError("timeout", "background_unavailable foreground timeout")
    r = tool.execute("click", snapshot_id=sid, target="Save")
    assert not r.success and r.content["dispatch_status"] == "unknown"
    assert len(backend.calls) == 1 and backend.calls[0][1]["delivery_mode"] == "background"


def test_idempotency_receipt_prevents_reexecution_and_conflicts(engine):
    tool, backend = engine
    sid = capture(tool)
    request = dict(snapshot_id=sid, target="Save", seconds=0, request_id="job-1")
    first = tool.execute("click", **request)
    replay = tool.execute("click", **request)
    assert first.success and replay.success and replay.metadata["replayed_receipt"]
    assert len(backend.calls) == 1
    conflict = tool.execute("click", **dict(request, target="Enabled"))
    assert conflict.content["error"]["code"] == "request_id_conflict"


def test_sequence_stops_when_explicit_expectation_is_not_met(engine):
    tool, backend = engine
    sid = capture(tool)
    r = tool.execute(
        "sequence",
        snapshot_id=sid,
        steps=[
            {
                "action": "click",
                "target": "Save",
                "seconds": 0,
                "expect": {"kind": "value", "target": "Name", "value": "saved"},
            },
            {"action": "type", "target": "Name", "text": "must not type"},
        ],
    )
    assert not r.success and r.content["stopped_at"] == 0
    assert len(backend.calls) == 1


def test_sequence_validates_every_step_before_dispatch(engine):
    tool, backend = engine
    sid = capture(tool)
    r = tool.execute(
        "sequence",
        snapshot_id=sid,
        steps=[
            {"action": "click", "target": "Save"},
            {"action": "click", "coordinate": [20, 20]},
        ],
    )
    assert not r.success and not backend.calls


def test_partial_typing_reports_acknowledged_characters_no_retry(engine):
    tool, backend = engine
    sid = capture(tool)

    def fail_second(session, action, args):
        backend.calls.append((action, args))
        if len(backend.calls) == 2:
            raise ComputerError("timeout", "transport timeout")
        return {}

    backend.dispatch = fail_second
    r = tool.execute("type", snapshot_id=sid, text="a" * 1000)
    assert not r.success and len(backend.calls) == 2
    assert "400/1000" in r.content["error"]["message"]


def test_native_element_text_is_not_split_into_replacement_chunks(engine):
    tool, backend = engine
    sid = capture(tool)
    r = tool.execute("type", snapshot_id=sid, target="Name", text="سلام" * 1000, seconds=0)
    assert r.success and len(backend.calls) == 1
    assert backend.rows[0]["value"] == "سلام" * 1000


def test_cancel_interrupts_wait_without_input(engine):
    tool, backend = engine
    capture(tool)
    thread = threading.Thread(target=lambda: (time.sleep(0.1), tool.cancel()))
    thread.start()
    start = time.monotonic()
    r = tool.execute("wait", seconds=10)
    thread.join()
    assert time.monotonic() - start < 1
    assert not r.success and r.content["error"]["code"] == "cancelled"
    assert not backend.calls


def test_post_capture_failure_records_unknown_after_input(engine):
    tool, backend = engine
    sid = capture(tool)
    backend.capture_failure = True
    r = tool.execute("click", snapshot_id=sid, target="Save", seconds=0)
    assert not r.success and len(backend.calls) == 1
    assert r.content["dispatch_status"] == "dispatched"


def test_zoom_keeps_native_frame_and_does_not_consume_snapshot(engine):
    tool, backend = engine
    sid = capture(tool)
    r = tool.execute("zoom", snapshot_id=sid, region=[10, 20, 100, 50], scale=3)
    assert r.success and r.content["origin"] == [10, 20] and r.content["scale_x"] == 3
    assert tool.execute("click", snapshot_id=sid, coordinate=[10, 20], seconds=0).success
    assert backend.captures == 2


def test_password_values_and_typed_text_not_in_event_log(engine):
    tool, backend = engine
    backend.rows.append(
        {"element_token": "p", "role": "password text", "label": "Password", "value": "supersecret"}
    )
    sid = capture(tool)
    state = tool._state()
    assert "supersecret" not in json.dumps(state)
    r = tool.execute("type", snapshot_id=sid, text="PRIVATE INPUT", seconds=0)
    assert r.success
    assert "PRIVATE INPUT" not in (tool.root / "events.jsonl").read_text()


def test_no_literal_site_preference_or_guessed_titlebar_inset(engine):
    tool, backend = engine
    backend.rows = [
        dict(backend.rows[0], label="Other"),
        dict(backend.rows[1], label="جست‌وجو در آپارات"),
    ]
    r = tool.execute("capture")
    assert [e["label"] for e in r.content["elements"]] == ["Other", "جست‌وجو در آپارات"]
    assert r.content["elements"][0]["bounds"]["y"] == 30
    backend.frame = None
    r = tool.execute("capture")
    assert "bounds" not in r.content["elements"][0]


def test_explicit_desktop_scope_never_falls_back_from_failed_window_selection(engine):
    tool, backend = engine
    r = tool.execute("capture", app="Nonexistent")
    assert not r.success and backend.captures == 0
    r = tool.execute("capture", scope="desktop")
    assert r.success and r.content["coordinate_frame"] == "desktop" and r.content["window"] is None


def test_corrupted_state_has_a_structured_error_and_no_input(engine):
    tool, backend = engine
    (tool.root / "state.json").write_text("[]")
    r = tool.execute("click", snapshot_id="invalid", target="Save")
    assert not r.success and r.content["error"]["code"] == "state_unreadable"
    assert not backend.calls


def test_malformed_later_sequence_key_is_validated_before_first_click(engine):
    tool, backend = engine
    sid = capture(tool)
    r = tool.execute(
        "sequence",
        snapshot_id=sid,
        steps=[{"action": "click", "target": "Save"}, {"action": "key"}],
    )
    assert not r.success and not backend.calls


def test_model_observation_is_bounded_but_exact_labels_are_preserved(engine):
    tool, backend = engine
    backend.rows = [
        dict(
            backend.rows[1],
            label=("label-" + str(i)) * 200,
            element_token=f"token-{i}",
            value="x" * 2000,
        )
        for i in range(100)
    ]
    result = tool.execute("capture", max_results=100)
    assert result.success and result.content["truncated"]
    assert len(json.dumps(result.content["elements"], ensure_ascii=False)) < 14000
    assert result.content["elements"][0]["label_truncated"]
    sid = result.content["snapshot_id"]
    exact = backend.rows[0]["label"]
    assert tool.execute("click", snapshot_id=sid, target=exact, seconds=0).success


def test_log_write_failure_does_not_claim_input_was_unsent(engine, monkeypatch):
    from pathlib import Path

    tool, backend = engine
    sid = capture(tool)
    original = Path.open

    def fail_log(path, *args, **kwargs):
        if path.name == "events.jsonl":
            raise OSError("disk error")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_log)
    result = tool.execute("click", snapshot_id=sid, target="Save", capture_after=False)
    assert not result.success and result.content["dispatch_status"] == "dispatched"
    assert len(backend.calls) == 1


def test_missing_backend_diagnostic_is_json_not_an_exception(engine):
    tool, backend = engine
    backend.doctor = lambda: {"installed": False, "error": "Driver not installed"}
    result = tool.execute("doctor")
    assert not result.success
    assert result.content["error"] == {"code": "backend_error", "message": "Driver not installed"}
    assert not backend.calls
