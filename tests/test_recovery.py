import json

import pytest

from computer_use.backend import CuaBackend
from computer_use import driver
from computer_use.runtime import ComputerError
from conftest import capture
from test_driver import executable


def test_foreground_choice_survives_click_type_capture_and_reinstantiation(engine):
    tool, backend = engine
    sid = capture(tool)
    result = tool.execute(
        "click", snapshot_id=sid, target="Name", delivery_mode="foreground", seconds=0
    )
    assert result.success
    clone = type(tool)(backend, workspace=tool.workspace_root, lock_path=tool.lock_path)
    sid = capture(clone)
    typed = clone.execute("type", snapshot_id=sid, text="hello", seconds=0)
    assert typed.success
    assert [args["delivery_mode"] for _, args in backend.calls] == ["foreground", "foreground"]
    assert typed.content["observation"]["delivery_mode"] == "foreground"


def test_delivery_choice_is_scoped_to_session_and_window(engine):
    tool, backend = engine
    capture(tool, delivery_mode="foreground")
    backend.window.window_id += 1
    assert tool.execute("capture").content["delivery_mode"] == "background"
    capture(tool, delivery_mode="foreground")
    tool.set_context_store(None, lambda: "another-user-session")
    assert tool.execute("capture").content["delivery_mode"] == "background"


def test_explicit_mode_override_is_retained(engine):
    tool, backend = engine
    sid = capture(tool, delivery_mode="foreground")
    r = tool.execute("key", snapshot_id=sid, key="Tab", delivery_mode="background", seconds=0)
    assert r.success and r.content["observation"]["delivery_mode"] == "background"


def test_navigation_uses_three_inputs_without_address_bar_clicks(engine):
    tool, backend = engine
    sid = capture(tool, delivery_mode="foreground")
    r = tool.execute("navigate", snapshot_id=sid, url="https://example.org/demo")
    assert r.success
    assert [a for a, _ in backend.calls] == ["hotkey", "type_text", "press_key"]
    assert backend.calls[1][1]["text"] == "https://example.org/demo"
    assert all(args["delivery_mode"] == "foreground" for _, args in backend.calls)
    assert r.content["observation"]["snapshot_id"] != sid


def test_ambiguity_returns_usable_ids_outside_first_page(engine):
    tool, backend = engine
    backend.rows = [
        {"element_token": str(i), "label": f"Container {i}", "role": "group"} for i in range(1000)
    ] + [
        {"element_token": f"url-{i}", "label": "Address", "role": "entry", "actions": ["set_value"]}
        for i in range(8)
    ]
    sid = capture(tool)
    r = tool.execute("click", snapshot_id=sid, target="Address")
    assert not r.success and r.content["snapshot_consumed"] is False
    assert r.content["candidate_count"] == 8
    candidate = r.evidence["candidates"][0]["id"]
    clicked = tool.execute("click", snapshot_id=sid, element=candidate, seconds=0)
    assert clicked.success and backend.calls[0][1]["element_token"] == "url-0"


def test_inspect_pages_without_capturing_or_changing_ids(engine):
    tool, backend = engine
    backend.rows = [
        {"element_token": str(i), "label": f"Entry {i}", "role": "entry"} for i in range(1013)
    ]
    sid = capture(tool)
    found = []
    offset = 0
    while offset is not None:
        r = tool.execute("inspect", snapshot_id=sid, query="Entry 10", offset=offset)
        assert r.success
        found.extend(e["id"] for e in r.evidence["elements"])
        offset = r.evidence["next_offset"]
    assert len(found) == len(set(found)) == 24
    assert backend.captures == 1 and not backend.calls


def test_focused_state_and_value_are_exposed_from_driver_states(engine):
    tool, backend = engine
    backend.rows[0].update(states=["focused"], value="https://example.org")
    r = tool.execute("capture")
    assert r.content["focused_elements"][0]["value"] == "https://example.org"


def test_tool_evidence_fits_host_json_budget_with_long_accessibility_labels(engine):
    tool, backend = engine
    backend.rows = [
        dict(backend.rows[0], element_token=str(i), label="word " * 300, value="v" * 1000)
        for i in range(1013)
    ]
    r = tool.execute("capture")
    payload = json.dumps(
        {"success": r.success, "summary": r.summary, "evidence": r.evidence, "effects": r.effects},
        ensure_ascii=False,
    )
    assert len(payload) < 8000
    assert json.loads(payload)["evidence"]["snapshot_id"] == r.content["snapshot_id"]
    assert r.evidence["next_offset"] == len(r.evidence["elements"])


@pytest.mark.parametrize("exit_code", [0, 1])
def test_structured_background_refusal_allows_corrected_call_without_recapture(
    engine, tmp_path, monkeypatch, exit_code
):
    tool, backend = engine
    binary = executable(
        tmp_path,
        """import json,sys
args=json.loads(sys.argv[3])
if args['delivery_mode']=='background':
 print(json.dumps({'code':'background_unavailable','detail':'no focus-free backend'}))
 sys.exit(EXIT_CODE)
print(json.dumps({'ok':True}))
""".replace("EXIT_CODE", str(exit_code)),
    )
    monkeypatch.setattr(driver, "driver_executable", lambda: binary)
    monkeypatch.setattr(backend, "dispatch", CuaBackend().dispatch)
    sid = capture(tool)
    refused = tool.execute("type", snapshot_id=sid, text="hello")
    assert refused.content["error"]["code"] == "background_unavailable"
    assert refused.content["dispatch_status"] == "not_dispatched"
    assert refused.content["snapshot_consumed"] is False
    corrected = tool.execute(
        "type", snapshot_id=sid, text="hello", delivery_mode="foreground", seconds=0
    )
    assert corrected.success


def test_partial_typing_does_not_restore_snapshot_on_later_refusal(engine, monkeypatch):
    tool, backend = engine

    def dispatch(session, action, args):
        backend.calls.append((action, args))
        if len(backend.calls) == 2:
            raise ComputerError("background_unavailable", "no route", not_dispatched=True)
        return {}

    monkeypatch.setattr(backend, "dispatch", dispatch)
    sid = capture(tool)
    result = tool.execute("type", snapshot_id=sid, text="a" * 500)
    assert result.content["dispatch_status"] == "unknown"
    assert result.content["snapshot_consumed"] is True
    assert "400/500" in result.content["error"]["message"]
    assert not tool.execute("type", snapshot_id=sid, text="a" * 500).success
    assert len(backend.calls) == 2
