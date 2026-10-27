"""Backend declarations and transport evidence, without task or UI inference."""

import json
from types import SimpleNamespace

import pytest

from src.tools.builtin.computer_use import driver
from src.tools.builtin.computer_use.backend import CuaBackend, decode
from src.tools.builtin.computer_use.contracts import Window
from src.tools.builtin.computer_use.runtime import ComputerError
from src.tools.builtin.computer_use.x11 import X11Backend


def test_capability_declarations_do_not_probe_or_send_input(monkeypatch):
    monkeypatch.setattr(
        driver, "call_driver", lambda *a, **kw: pytest.fail("probe sent input")
    )
    cua = CuaBackend().capabilities("s")
    x11 = X11Backend().capabilities("s")
    assert (
        cua["preferred_delivery_mode"] == x11["preferred_delivery_mode"] == "foreground"
    )
    assert cua["delivery_modes"]["background"] == "driver-dependent"
    assert x11["delivery_modes"] == {"foreground": True, "background": False}
    assert not cua["drag_path"] and x11["drag_path"]
    assert not cua["pointer_move"] and x11["pointer_move"]


def test_refusal_records_only_observed_route_and_never_retries(monkeypatch):
    backend = CuaBackend()
    calls = []
    response = json.dumps(
        {
            "code": "background_unavailable",
            "detail": "No focus-free route",
            "escalation": {
                "delivery_mode": "foreground",
                "reason": "Global focus required",
            },
        }
    )

    def send(action, arguments, **kwargs):
        calls.append((action, arguments))
        if arguments["delivery_mode"] == "background":
            raise driver.response_error("refused", response)
        return '{"ok":true}'

    monkeypatch.setattr(driver, "call_driver", send)
    args = {"delivery_mode": "background", "pid": 10, "window_id": 20, "x": 5, "y": 6}
    with pytest.raises(ComputerError) as failure:
        backend.dispatch("s", "click", args)
    assert len(calls) == 1
    assert failure.value.not_dispatched and not failure.value.dispatched
    assert failure.value.driver_details["escalation"]["delivery_mode"] == "foreground"
    known = backend.capabilities("s", Window(10, 20))["known_delivery_routes"]
    assert known == [
        {
            "action": "click",
            "target_kind": "coordinate",
            "delivery_mode": "background",
            "supported": False,
            "code": "background_unavailable",
        }
    ]
    assert (
        backend.capabilities("different", Window(10, 20))["known_delivery_routes"] == []
    )
    assert backend.capabilities("s", Window(10, 99))["known_delivery_routes"] == []
    backend.dispatch("s", "click", dict(args, delivery_mode="foreground"))
    assert len(calls) == 2
    assert (
        backend.capabilities("s", Window(10, 20))["known_delivery_routes"][-1][
            "supported"
        ]
        is True
    )


def test_uncertain_error_never_becomes_known_unsupported_route(monkeypatch):
    backend = CuaBackend()

    def uncertain(*args, **kwargs):
        raise driver.CuaDriverError(
            "timed out; background_unavailable might have appeared"
        )

    monkeypatch.setattr(driver, "call_driver", uncertain)
    with pytest.raises(ComputerError) as failure:
        backend.dispatch("s", "click", {"delivery_mode": "background", "x": 1, "y": 2})
    assert not failure.value.not_dispatched
    assert backend.capabilities("s")["known_delivery_routes"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {"error": {"code": "permission_denied"}},
        {"status": "failed"},
        {"delivered": False},
        {"code": "background_unavailable"},
    ],
)
def test_direct_transport_negative_responses_are_not_acknowledgements(payload):
    with pytest.raises(ComputerError):
        decode(json.dumps(payload))


def test_cua_multi_point_path_is_rejected_before_dispatch(monkeypatch):
    monkeypatch.setattr(
        driver, "call_driver", lambda *a, **kw: pytest.fail("input was sent")
    )
    with pytest.raises(ComputerError) as failure:
        CuaBackend().dispatch("s", "drag", {"path": [[0, 0], [20, 20], [20, 0]]})
    assert failure.value.not_dispatched
    assert "endpoint drag only" in str(failure.value)


def test_x11_polyline_visits_each_point_with_one_held_button(monkeypatch):
    import src.tools.builtin.computer_use.x11 as x11

    events = []
    backend = X11Backend()
    monkeypatch.setattr(backend, "_run", lambda command: events.append(command))
    monkeypatch.setattr(x11.shutil, "which", lambda command: "/usr/bin/xdotool")
    monkeypatch.setattr(
        x11.subprocess, "run", lambda command, **kw: events.append(command)
    )
    path = [[2, 3], [20, 3], [20, 20], [2, 3]]
    backend._drag(
        {"path": path, "duration_ms": 0}, lambda x, y: events.append(["move", x, y]), 1
    )
    assert events == [
        ["move", 2, 3],
        ["mousedown", 1],
        ["move", 20, 3],
        ["move", 20, 20],
        ["move", 2, 3],
        ["/usr/bin/xdotool", "mouseup", "1"],
    ]


def test_x11_polyline_releases_button_on_cancellation(monkeypatch):
    import src.tools.builtin.computer_use.x11 as x11

    events = []
    backend = X11Backend()
    monkeypatch.setattr(backend, "_run", lambda command: events.append(command))
    monkeypatch.setattr(x11.shutil, "which", lambda command: "/usr/bin/xdotool")
    monkeypatch.setattr(
        x11.subprocess, "run", lambda command, **kw: events.append(command)
    )

    def cancel():
        raise ComputerError("cancelled", "test cancellation")

    monkeypatch.setattr(x11, "checkpoint", cancel)
    with pytest.raises(ComputerError, match="cancellation"):
        backend._drag(
            {"path": [[2, 3], [20, 3]], "duration_ms": 100}, lambda *point: None, 1
        )
    assert events == [["mousedown", 1], ["/usr/bin/xdotool", "mouseup", "1"]]


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_native_platform_environment_does_not_require_linux_display(
    monkeypatch, platform
):
    monkeypatch.setattr(driver.sys, "platform", platform)
    monkeypatch.setattr(
        driver,
        "_read_process_environment",
        lambda: pytest.fail("Linux session discovery used"),
    )
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("DAENA_NATIVE_ENV_TEST", "present")
    env = driver._desktop_session_environment()
    assert env["DAENA_NATIVE_ENV_TEST"] == "present"
    assert "DISPLAY" not in env and "XDG_RUNTIME_DIR" not in env


def test_windows_cancellation_targets_only_owned_helper_process_tree(monkeypatch):
    calls = []
    monkeypatch.setattr(driver.sys, "platform", "win32")
    monkeypatch.setattr(
        driver.subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    owned = SimpleNamespace(
        pid=4321, poll=lambda: None, kill=lambda: calls.append("kill-owned")
    )
    driver._kill_process_group(owned)
    assert calls == [["taskkill", "/PID", "4321", "/T", "/F"], "kill-owned"]


def test_x11_drag_releases_even_when_initial_press_acknowledgement_fails(monkeypatch):
    import src.tools.builtin.computer_use.x11 as x11

    events = []
    backend = X11Backend()

    def uncertain(command):
        events.append(command)
        raise ComputerError(
            "input_timeout", "press may have been delivered", dispatched=True
        )

    monkeypatch.setattr(backend, "_run", uncertain)
    monkeypatch.setattr(x11.shutil, "which", lambda command: "/usr/bin/xdotool")
    monkeypatch.setattr(
        x11.subprocess, "run", lambda command, **kw: events.append(command)
    )
    with pytest.raises(ComputerError, match="press may"):
        backend._drag(
            {"path": [[2, 3], [20, 3]], "duration_ms": 100}, lambda *point: None, 1
        )
    assert events == [["mousedown", 1], ["/usr/bin/xdotool", "mouseup", "1"]]


def test_x11_drag_reports_unconfirmed_button_cleanup(monkeypatch):
    import src.tools.builtin.computer_use.x11 as x11

    backend = X11Backend()
    monkeypatch.setattr(backend, "_run", lambda command: None)
    monkeypatch.setattr(x11.shutil, "which", lambda command: "/usr/bin/xdotool")
    monkeypatch.setattr(
        x11.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1)
    )
    with pytest.raises(ComputerError) as failure:
        backend._drag(
            {"path": [[2, 3], [20, 3]], "duration_ms": 0}, lambda *point: None, 1
        )
    assert failure.value.code == "button_release_failed" and failure.value.dispatched
