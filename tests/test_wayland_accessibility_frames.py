from src.tools.builtin.computer_use.inspect import ComputerInspectTool


def test_wayland_native_control_frame_is_offset_into_full_window_screenshot(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.delenv("DAENA_CUA_WAYLAND_TOP_INSET", raising=False)
    payload = {
        "window_title": "Paint",
        "coordinate_frame": "window",
        "window_bounds": {"x": 770, "y": 452, "width": 1282, "height": 890},
        "elements": [{
            "element_index": 4,
            "parent_index": None,
            "role": "check box",
            "label": "Rectangle",
            "frame": {"x": 241, "y": 34, "w": 87, "h": 26},
            "element_token": "snapshot:4",
            "actions": ["Toggle", "Press"],
            "selected": False,
            "enabled": True,
        }],
    }

    rows = ComputerInspectTool._rank_elements(payload, "Rectangle", 8)

    assert len(rows) == 1
    assert rows[0]["frame_coordinate_space"] == "window"
    assert rows[0]["frame"] == {"x": 241, "y": 72.0, "w": 87, "h": 26}


def test_wayland_inset_can_be_configured(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DAENA_CUA_WAYLAND_TOP_INSET", "42")

    assert ComputerInspectTool._wayland_window_top_inset() == 42


def test_wayland_is_detected_from_runtime_socket_when_gui_env_is_missing(monkeypatch, tmp_path):
    socket = tmp_path / "wayland-0"
    socket.touch()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("DAENA_CUA_WAYLAND_TOP_INSET", raising=False)

    assert ComputerInspectTool._wayland_window_top_inset() == 38


def test_x11_does_not_apply_wayland_inset(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("DAENA_CUA_WAYLAND_TOP_INSET", raising=False)

    assert ComputerInspectTool._wayland_window_top_inset() == 0
