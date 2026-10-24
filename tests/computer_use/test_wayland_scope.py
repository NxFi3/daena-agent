import pytest

from src.tools.builtin.computer_use import ComputerUseTool
from src.tools.builtin.computer_use.x11 import X11Backend
from src.tools.builtin.computer_use.runtime import ComputerError


def test_wayland_blocker_is_not_reported_as_missing_application(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    backend = X11Backend()
    monkeypatch.setattr(backend, "ready", lambda session: None)
    monkeypatch.setattr(backend, "windows", lambda session: [])
    tool = ComputerUseTool(backend, workspace=tmp_path, lock_path=tmp_path / "lock")
    for action in ("list_windows", "list_apps", "capture"):
        result = tool.execute(action)
        assert not result.success
        assert result.content["error"]["code"] == "native_wayland_unavailable"
        assert result.content["retry_requires_setup_change"]
        assert result.content["dispatch_status"] == "not_dispatched"
    report = tool.execute("doctor").content
    assert not report["native_desktop_supported"]
    assert not report["whole_desktop_capture_supported"]
    assert report["window_scope"] == "Xwayland clients only"
    import sys
    monkeypatch.setitem(sys.modules, "Xlib", None)
    with pytest.raises(ComputerError, match="not a capture of the Wayland desktop"):
        backend.capture("test", None, tmp_path / "must-not-exist.png")
    assert not (tmp_path / "must-not-exist.png").exists()


def test_empty_x11_inventory_keeps_normal_semantics(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    backend = X11Backend()
    monkeypatch.setattr(backend, "ready", lambda session: None)
    monkeypatch.setattr(backend, "windows", lambda session: [])
    tool = ComputerUseTool(backend, workspace=tmp_path, lock_path=tmp_path / "lock")
    assert tool.execute("list_windows").success
    result = tool.execute("capture")
    assert result.content["error"]["code"] == "window_missing"
    assert backend.doctor()["native_desktop_supported"]
