"""Real XTest input into a Qt app on Xvfb; assertions come from the app's events."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from src.tools.builtin.computer_use import ComputerUseTool
from src.tools.builtin.computer_use.x11 import X11Backend


@pytest.mark.desktop
@pytest.mark.skipif(
    not os.environ.get("DAENA_TEST_DESKTOP"),
    reason="Set DAENA_TEST_DESKTOP=1 with an isolated X11 display",
)
def test_real_pointer_keyboard_unicode_scroll_and_drag(tmp_path):
    pytest.importorskip("PySide6")
    assert shutil.which("xdotool")
    state_path = tmp_path / "app.json"
    app = Path(__file__).parents[2] / "examples" / "computer_use_workbench.py"
    process = subprocess.Popen(
        [sys.executable, str(app), str(state_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "QT_QPA_PLATFORM": "xcb"},
    )
    try:
        deadline = time.monotonic() + 15
        while (
            not state_path.exists()
            and time.monotonic() < deadline
            and process.poll() is None
        ):
            time.sleep(0.05)
        if not state_path.exists():
            stdout, stderr = process.communicate(timeout=2)
            pytest.fail(
                "Qt desktop fixture failed to start:\n"
                + stderr.decode(errors="replace")
                + stdout.decode(errors="replace"),
                pytrace=False,
            )

        def state():
            return json.loads(state_path.read_text())

        tool = ComputerUseTool(
            X11Backend(), workspace=tmp_path, lock_path=tmp_path / "desktop.lock"
        )
        r = tool.execute("capture", window_title="Test Workbench", mode="vision")
        assert r.success, r.content
        sid = r.content["snapshot_id"]
        focused = tool.execute(
            "click",
            snapshot_id=sid,
            delivery_mode="foreground",
            position=dict(zip(("x", "y"), state()["points"]["name"])),
        )
        assert focused.success, focused.content
        sid = focused.content["observation"]["snapshot_id"]
        for action, args in [
            ("type", {"text": "Daena سلام"}),
            ("click", {"coordinate": state()["points"]["check"]}),
            ("click", {"coordinate": state()["points"]["save"]}),
        ]:
            r = tool.execute(action, snapshot_id=sid, **args)
            assert r.success, r.content
            sid = r.content["observation"]["snapshot_id"]
        time.sleep(0.15)
        assert state()["text"] == "Daena سلام"
        assert state()["checked"] and state()["saved"] == 1
        x, y, w, h = state()["canvas"]
        r = tool.execute(
            "drag",
            snapshot_id=sid,
            delivery_mode="foreground",
            from_coordinate=[x + 40, y + 40],
            to_coordinate=[x + w - 50, y + h - 50],
            duration_ms=400,
        )
        assert r.success, r.content
        assert r.content["visual_delta"]["status"] == "changed"
        sid = r.content["observation"]["snapshot_id"]
        r = tool.execute(
            "scroll",
            snapshot_id=sid,
            delivery_mode="foreground",
            coordinate=[x + 100, y + 100],
            direction="down",
            amount=3,
        )
        assert r.success, r.content
        time.sleep(0.1)
        assert state()["stroke_count"] == 1 and state()["point_count"] >= 3
        assert not state()["mouse_down"] and state()["scroll_total"] < 0
        # Draw a house outline on the unlabelled real Qt canvas using only
        # observed image coordinates; no accessibility token represents it.
        points = [(x+80,y+100),(x+80,y+190),(x+260,y+190),
                  (x+260,y+100),(x+170,y+35),(x+80,y+100),(x+260,y+100)]
        drawing_started = time.monotonic()
        for start, end in zip(points, points[1:]):
            r = tool.execute(
                "drag", snapshot_id=r.content["observation"]["snapshot_id"],
                from_coordinate=list(start), to_coordinate=list(end),
                delivery_mode="foreground", duration_ms=160,
            )
            assert r.success, r.content
            assert r.content["visual_delta"]["status"] == "changed"
        assert time.monotonic() - drawing_started < 20, "Consecutive strokes stalled at an unchanged pointer position"
        deadline = time.monotonic() + 2
        while state()["stroke_count"] < 7 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert state()["stroke_count"] == 7 and not state()["mouse_down"]
        artifact = os.environ.get("DAENA_TEST_ARTIFACTS")
        if artifact:
            Path(artifact).mkdir(parents=True, exist_ok=True)
            shutil.copy2(r.content["observation"]["image_ref"], Path(artifact)/"house.png")
        # A single path gesture holds the button across every corner, with no
        # application-specific shape command or extra focus clicks.
        r = tool.execute(
            "drag", snapshot_id=r.content["observation"]["snapshot_id"],
            path=[[x+70,y+80], [x+110,y+40], [x+150,y+80], [x+150,y+150], [x+70,y+150], [x+70,y+80]],
            duration_ms=500,
        )
        assert r.success, r.content
        deadline = time.monotonic() + 2
        while state()["stroke_count"] < 8 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert state()["stroke_count"] == 8 and not state()["mouse_down"]
        assert r.content["visual_delta"]["status"] == "changed"
        assert tool.current_observation()["capabilities"]["drag_path"]
        # A browser-style address field uses real Ctrl+L, typing, Return events.
        r = tool.execute(
            "navigate",
            snapshot_id=r.content["observation"]["snapshot_id"],
            url="https://example.org/demo",
        )
        assert r.success, r.content
        time.sleep(0.1)
        assert state()["submitted"] == "https://example.org/demo"
        assert state()["text"] == "https://example.org/demo"
        # Stress distinct Unicode keysyms across the adapter's free-keycode batches.
        # A real client processing barrier must preserve text and restore XKB state.
        from Xlib.display import Display
        from contextlib import closing

        def keymap():
            with closing(Display()) as display:
                first, last = (
                    display.display.info.min_keycode,
                    display.display.info.max_keycode,
                )
                return [
                    tuple(row)
                    for row in display.get_keyboard_mapping(first, last - first + 1)
                ]

        before = keymap()
        for value in ["سلام فارسی Ελληνικά 中文 日本語 😀 café —", "سلام سلام سلام"]:
            r = tool.execute(
                "key",
                snapshot_id=r.content["observation"]["snapshot_id"],
                keys=["CTRL", "l"],
            )
            assert r.success, r.content
            r = tool.execute(
                "type", snapshot_id=r.content["observation"]["snapshot_id"], text=value
            )
            assert r.success, r.content
            deadline = time.monotonic() + 2
            while state()["text"] != value and time.monotonic() < deadline:
                time.sleep(0.02)
            assert state()["text"] == value
            assert keymap() == before
        # Save a real fixture screenshot/trace for the release evidence.
        artifact = os.environ.get("DAENA_TEST_ARTIFACTS")
        if artifact:
            Path(artifact).mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                r.content["observation"]["image_ref"], Path(artifact) / "workbench.png"
            )
            shutil.copytree(
                tmp_path / ".daena", Path(artifact) / ".daena", dirs_exist_ok=True
            )
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@pytest.mark.desktop
@pytest.mark.skipif(not os.environ.get('DAENA_TEST_DESKTOP'), reason='Requires isolated X11 display')
def test_reparented_client_without_ewmh_pid_is_discovered(tmp_path, monkeypatch):
    from contextlib import closing
    from Xlib import X
    from Xlib.display import Display
    with closing(Display()) as display:
        assert display.has_extension('X-Resource'), 'Native test server must expose XRes'
        root=display.screen().root
        outer=root.create_window(40,50,400,300,0,X.CopyFromParent,X.InputOutput,X.CopyFromParent)
        child=outer.create_window(10,20,200,150,0,X.CopyFromParent,X.InputOutput,X.CopyFromParent)
        child.set_wm_name('Unadvertised native client')
        child.set_wm_class('native-test','NativeTest')
        outer.map();child.map();display.sync()
        try:
            backend=X11Backend()
            found=[w for w in backend.windows('test') if w.window_id==child.id]
            assert len(found)==1 and found[0].pid==os.getpid()
            assert found[0].bounds.x==50 and found[0].bounds.y==70
            tool=ComputerUseTool(backend,workspace=tmp_path,lock_path=tmp_path/'lock')
            result=tool.execute('capture',window_id=child.id,mode='vision')
            assert result.success, result.content
            assert result.content['width']==200 and result.content['height']==150
            with monkeypatch.context() as patch:
                patch.setattr('shutil.which', lambda name: None)
                observation=tool.execute('capture',window_id=child.id,mode='vision')
                assert observation.success, observation.content
                blocked=tool.execute('click', snapshot_id=observation.content['snapshot_id'], coordinate=[10,10], delivery_mode='foreground')
                assert not blocked.success and blocked.content['dispatch_status']=='not_dispatched'
                assert 'xdotool' in str(blocked.content)
                assert tool.current_observation()['input_allowed']
            monkeypatch.setattr(backend,'_local_client_pid',lambda *args:None)
            assert child.id not in [w.window_id for w in backend.windows('test')]
            assert backend.window_diagnostics['missing_identity']>=1
        finally:
            outer.destroy();display.sync()
