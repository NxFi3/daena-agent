"""Observation feedback must describe evidence, not infer application workflows."""

from copy import deepcopy
import json

import pytest
from PIL import Image

from src.tools.builtin.computer_use.contracts import Element, Rect
from src.tools.builtin.computer_use.integration import observation_message
from src.tools.builtin.computer_use.observe import (
    accessibility_delta,
    observation_signature,
)


def element(identity, *, snapshot="old", focused=False, value="", token=None):
    return Element(
        f"{snapshot}:{identity}",
        token or str(identity),
        "entry",
        "Text field",
        Rect(10, 20, 100, 30),
        value=value,
        focused=focused,
    )


def test_signature_ignores_capture_identity_but_tracks_real_observed_changes():
    from dataclasses import asdict

    before = {
        "snapshot_id": "old",
        "captured_at": 1,
        "image_ref": "old.png",
        "pixel_hash": "pixels",
        "width": 640,
        "height": 480,
        "window": {"pid": 1, "window_id": 2, "title": "Editor"},
        "elements": [asdict(element(1)), asdict(element(2))],
        "complete": True,
    }
    after = deepcopy(before)
    after.update(snapshot_id="new", captured_at=2, image_ref="new.png")
    after["elements"].reverse()
    for row in after["elements"]:
        row["id"] = row["id"].replace("old", "new")
    assert observation_signature(before) == observation_signature(after)
    for mutate in [
        lambda state: state.update(pixel_hash="different pixels"),
        lambda state: state["elements"][0].update(focused=True),
        lambda state: state["elements"][0].update(value="typed text"),
        lambda state: state["window"].update(title="New title"),
    ]:
        changed = deepcopy(after)
        mutate(changed)
        assert observation_signature(before) != observation_signature(changed)


def test_accessibility_reports_focus_and_value_without_pixel_or_success_claims():
    old = [element(1)]
    new = [element(1, snapshot="new", focused=True, value="hello")]
    delta = accessibility_delta(old, new)
    assert delta["status"] == "changed"
    assert delta["compared_controls"] == 1
    assert delta["changes"][0]["element"] == "new:1"
    assert delta["changes"][0]["fields"] == {
        "focused": {"before": False, "after": True},
        "value": {"before": "", "after": "hello"},
    }
    assert "completion" in delta["meaning"]
    assert "success" not in delta
    assert accessibility_delta(old, deepcopy(old))["status"] == "unchanged"
    assert accessibility_delta([], [])["status"] == "unavailable"


def test_accessibility_never_guesses_ambiguous_geometry_identity():
    first, second = element(1), element(2)
    first.token = second.token = ""
    before = [first, second]
    after = deepcopy(before)
    after[0].value = "could be either control"
    delta = accessibility_delta(before, after)
    assert delta["compared_controls"] == 0
    assert delta["ambiguous_identities"] == 1
    assert delta["changes"] == []
    assert delta["status"] == "unavailable"


def test_accessibility_change_details_are_bounded():
    before = [element(index) for index in range(20)]
    after = [element(index, snapshot="new", value="x" * 2000) for index in range(20)]
    delta = accessibility_delta(before, after, limit=3)
    assert delta["changed_controls"] == 20
    assert len(delta["changes"]) == 3 and delta["truncated"]
    assert delta["changes"][0]["fields"]["value"]["truncated"]
    assert len(delta["changes"][0]["fields"]["value"]["after"]) == 400
    json.dumps(delta)


def text_payload(message):
    content = message["content"]
    if isinstance(content, list):
        content = content[0]["text"]
    return json.loads(content.split("\n", 1)[1])


@pytest.mark.parametrize("provider", ["ollama", "openrouter", "gemini"])
def test_zoom_mapping_matches_attached_pixels_for_all_provider_encodings(
    tmp_path, provider
):
    image_dir = tmp_path / ".daena" / "computer_use"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "zoom.png"
    Image.new("RGB", (200, 160), "white").save(image_path)
    observation = {
        "snapshot_id": "snapshot",
        "image_ref": str(image_path),
        "width": 640,
        "height": 480,
        "origin": [10, 20],
        "scale_x": 2,
        "scale_y": 2,
        "pointer_coordinates": {"origin": "old misleading origin"},
    }
    original = deepcopy(observation)
    message = observation_message(observation, provider, tmp_path)
    payload = text_payload(message)
    mapping = payload["image_coordinate_mapping"]
    assert mapping["status"] == "verified"
    assert mapping["attached_size"] == [200, 160]
    assert mapping["native_input_size"] == [640, 480]
    assert mapping["native_origin"] == [10, 20] and mapping["image_scale"] == [2, 2]
    assert "full native capture" in payload["pointer_coordinates"]["origin"]
    assert observation == original


def test_unmapped_image_size_mismatch_is_explicit(tmp_path):
    image_dir = tmp_path / ".daena" / "computer_use"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "different-size.png"
    Image.new("RGB", (320, 240), "white").save(image_path)
    message = observation_message(
        {
            "snapshot_id": "snapshot",
            "image_ref": str(image_path),
            "width": 640,
            "height": 480,
        },
        "ollama",
        tmp_path,
    )
    mapping = text_payload(message)["image_coordinate_mapping"]
    assert mapping["status"] == "unverified"
    assert "do not guess" in mapping["guidance"]
    assert message["images"]


def test_oversized_diagnostics_do_not_erase_the_current_frame_or_target():
    from src.tools.builtin.computer_use.evidence import compact_evidence, encoded_size

    observation = {
        "snapshot_id": "s_current",
        "width": 640,
        "height": 480,
        "image_ref": "/workspace/.daena/computer_use/"
        + "nested/" * 150
        + "s_current.raw.png",
        "origin": [10, 20],
        "scale_x": 2,
        "scale_y": 2,
        "coordinate_frame": "window",
        "input_allowed": True,
        "window": {"pid": 123, "window_id": 99, "title": "Current window"},
        "last_input": {
            "action": "click",
            "dispatch_status": "dispatched",
            "snapshot_id_used": "s_previous",
            "next_snapshot_id": "s_current",
            "target": {"coordinate": [45, 40]},
            "visual_delta": {"status": "unchanged"},
            "diagnostics": {str(index): "detail" * 500 for index in range(20)},
        },
        "diagnostics": {str(index): "detail" * 500 for index in range(20)},
    }
    evidence = compact_evidence(observation, budget=4200)
    assert encoded_size(evidence) <= 4200
    assert evidence["snapshot_id"] == observation["snapshot_id"]
    assert evidence["image_ref"] == observation["image_ref"]
    assert evidence["origin"] == [10, 20] and evidence["scale_x"] == 2
    assert evidence["width"] == 640 and evidence["height"] == 480
    assert evidence["coordinate_frame"] == "window" and evidence["input_allowed"]
    assert evidence["window"]["window_id"] == 99
    assert evidence["last_input"]["target"] == {"coordinate": [45, 40]}
    assert evidence["last_input"]["next_snapshot_id"] == "s_current"
    assert evidence["last_input"]["visual_delta"]["status"] == "unchanged"
    assert evidence["details_omitted"]
