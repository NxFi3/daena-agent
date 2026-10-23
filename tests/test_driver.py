import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from computer_use import driver
from computer_use.backend import CuaBackend, decode
from computer_use.contracts import Rect, Window
from computer_use.runtime import ComputerError, operation


def executable(tmp_path, body):
    path = tmp_path / "fake-driver"
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o700)
    return str(path)


def test_large_json_response_is_not_truncated(tmp_path, monkeypatch):
    path = executable(tmp_path, "import json\nprint(json.dumps({'windows':['x'*20000]}))\n")
    monkeypatch.setattr(driver, "driver_executable", lambda: path)
    output = driver.call_driver("list_windows")
    assert len(json.loads(output)["windows"][0]) == 20000


def test_bounded_timeout_kills_helper_group(tmp_path):
    path = tmp_path / "child"
    command = [
        sys.executable,
        "-c",
        "import subprocess,time,pathlib; p=subprocess.Popen(['sleep','60']); pathlib.Path("
        + repr(str(path))
        + ").write_text(str(p.pid)); time.sleep(60)",
    ]
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        driver._run_bounded(command, 0.3)
    assert time.monotonic() - start < 2
    child = int(path.read_text())
    time.sleep(0.1)
    proc = Path(f"/proc/{child}/stat")
    assert not proc.exists() or proc.read_text().split(") ")[-1].startswith("Z")


def test_cancellation_interrupts_blocked_driver(tmp_path):
    event = threading.Event()
    thread = threading.Thread(target=lambda: (time.sleep(0.1), event.set()))
    thread.start()
    start = time.monotonic()
    with operation(30, event), pytest.raises(ComputerError, match="cancelled"):
        driver._run_bounded([sys.executable, "-c", "import time; time.sleep(60)"], 30)
    thread.join()
    assert time.monotonic() - start < 1


def test_driver_stdin_is_closed(tmp_path):
    result = driver._run_bounded(
        [sys.executable, "-c", "import sys;print(len(sys.stdin.read()))"], 1
    )
    assert result.stdout.strip() == "0"


@pytest.mark.parametrize(
    "value",
    [
        '{"success":false}',
        '{"ok":false}',
        '{"isError":true}',
        '{"error":{"code":"denied"}}',
        "{bad",
    ],
)
def test_structured_errors_are_not_success(value):
    with pytest.raises(ComputerError):
        decode(value)


def test_capture_must_attest_exact_window(tmp_path, monkeypatch):
    monkeypatch.setattr(
        driver, "call_driver", lambda *a, **kw: json.dumps({"pid": 1, "window_id": 3})
    )
    with pytest.raises(ComputerError, match="PID/window_id"):
        CuaBackend().capture(
            "s",
            Window(1, 2, bounds=Rect(0, 0, 20, 20)),
            tmp_path / "shot.png",
            pixels=True,
            accessibility=True,
        )


def test_cua_capture_requests_one_atomic_window_observation(tmp_path, monkeypatch):
    from PIL import Image

    calls = []

    def call(name, args, **kwargs):
        calls.append((name, args))
        Image.new("RGB", (100, 80)).save(args["screenshot_out_file"])
        return json.dumps(
            {
                "pid": 1,
                "window_id": 2,
                "capture_id": "cap",
                "coordinate_frame": "window",
                "elements": [],
            }
        )

    monkeypatch.setattr(driver, "call_driver", call)
    cap = CuaBackend().capture(
        "s",
        Window(1, 2, bounds=Rect(0, 0, 100, 80)),
        tmp_path / "shot.png",
        pixels=True,
        accessibility=True,
    )
    assert cap.capture_id == "cap" and len(calls) == 1
    assert calls[0][1]["include_screenshot"] and calls[0][1]["include_accessibility_tree"]


def test_input_does_not_replay_after_daemon_error(tmp_path, monkeypatch):
    path = executable(
        tmp_path, "import sys\nprint('daemon not running',file=sys.stderr)\nsys.exit(1)\n"
    )
    monkeypatch.setattr(driver, "driver_executable", lambda: path)
    monkeypatch.setattr(
        driver, "_ensure_daemon_running", lambda *_: pytest.fail("input replay was attempted")
    )
    with pytest.raises(ComputerError):
        driver.call_driver("click", {"x": 5, "y": 5})


def test_cua_drag_primes_pixels_only_and_rejects_changed_screen(tmp_path, monkeypatch):
    from PIL import Image
    from computer_use.contracts import Capture
    from computer_use.observe import image_info

    path = tmp_path / "screen.png"
    Image.new("RGB", (100, 80), "white").save(path)
    snapshot = {"window": None, "raw_image_ref": str(path), **image_info(path)}
    backend = CuaBackend()
    calls = []

    def capture(session, window, output, **kwargs):
        calls.append(kwargs)
        Image.new("RGB", (100, 80), "black").save(output)
        return Capture(str(output), "new", None, "desktop")

    monkeypatch.setattr(backend, "capture", capture)
    with pytest.raises(ComputerError, match="changed before drag"):
        backend.prepare_action("s", "drag", {}, snapshot)
    assert calls == [{"pixels": True, "accessibility": False}]
    assert not list(tmp_path.glob(".drag-prime*"))
