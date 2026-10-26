import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from src.tools.builtin.browser_use.integration import browser_messages
from .conftest import ok, element

pytestmark = pytest.mark.browser


def test_canvas_position_and_pointer_drag(browser, website):
    ok(browser, "open", url=website)
    data = ok(
        browser, "click", target={"css": "#drawing"}, position={"x": 30, "y": 20}
    )["observation"]
    assert "Canvas 30,20" in data["text"]
    data = ok(
        browser,
        "drag",
        target={"role": "button", "name": "Drag source"},
        destination={"role": "button", "name": "Drop target"},
    )["observation"]
    assert "Dropped" in data["text"]


def test_real_login_across_turns_unicode_and_compact_context(browser, website):
    data = ok(browser, "open", url=website)["observation"]
    assert data["url"].startswith(website)
    assert "never-include-me" not in json.dumps(data)
    assert "image_ref" not in data
    assert browser.model_exposure == "always"
    phone = element(data, "Phone")
    data = ok(browser, "fill", target={"ref": phone}, text="سلام 09120000000")[
        "observation"
    ]
    assert any(row.get("value") == "سلام 09120000000" for row in data["elements"])
    assert not browser.execute("fill", target={"ref": phone}, text="stale").success
    ok(browser, "click", target={"role": "button", "name": "Send code"})
    # Separate tool calls simulate a user providing an OTP in a later message.
    ok(browser, "fill", target={"label": "One-time code"}, text="123456")
    data = ok(browser, "click", target={"role": "button", "name": "Sign in"})[
        "observation"
    ]
    assert "Signed in" in data["text"]
    loop = SimpleNamespace(tool=SimpleNamespace(get_tool=lambda _: browser), config={})
    message = browser_messages(loop, {"browser_use"})[0]
    assert "Signed in" in message["content"] and "images" not in message


def test_ambiguous_controls_refs_replacement_frames_shadow_and_form(browser, website):
    ok(browser, "open", url=website)
    result = browser.execute("click", target={"role": "button", "name": "Duplicate"})
    assert not result.success and result.content["error"]["code"] == "target_ambiguous"
    assert result.content["delivery"] == "not_sent"
    data = ok(browser, "observe", query="Duplicate")["observation"]
    assert len(data["elements"]) == 2
    ok(browser, "click", target={"ref": data["elements"][1]["ref"]})
    assert "Right chosen" in browser.current_observation()["text"]
    data = ok(browser, "observe")["observation"]
    frame = next(
        f["frame_id"] for f in data["frames"] if f["url"].endswith("frame.html")
    )
    data = ok(
        browser,
        "fill",
        target={"label": "Frame field", "frame_id": frame},
        text="Inside iframe",
    )["observation"]
    assert any(e.get("value") == "Inside iframe" for e in data["elements"])
    data = ok(browser, "fill", target={"label": "Shadow input"}, text="Shadow text")[
        "observation"
    ]
    assert any(e.get("value") == "Shadow text" for e in data["elements"])
    data = ok(browser, "check", target={"label": "Accept terms"}, checked=True)[
        "observation"
    ]
    assert next(e for e in data["elements"] if e["name"] == "Accept terms")["checked"]
    data = ok(browser, "select", target={"label": "Plan"}, values=["pro"])[
        "observation"
    ]
    assert next(e for e in data["elements"] if e["name"] == "Plan")["value"] == "pro"
    old = element(data, "Replace me")
    ok(browser, "click", target={"ref": old})
    assert not browser.execute("click", target={"ref": old}).success


def test_popup_download_upload_scroll_and_wait(browser, website, tmp_path):
    data = ok(browser, "open", url=website)["observation"]
    main = data["tab_id"]
    ok(browser, "click", target={"role": "link", "name": "Open popup"})
    tabs = ok(browser, "tabs")["tabs"]
    popup = next(t["tab_id"] for t in tabs if t["tab_id"] != main)
    data = ok(browser, "select_tab", tab_id=popup)["observation"]
    assert "Popup loaded" in data["text"]
    ok(browser, "close_tab", tab_id=popup)
    ok(browser, "select_tab", tab_id=main)
    data = ok(browser, "observe")["observation"]
    old_frame = next(
        f["frame_id"] for f in data["frames"] if f["url"].endswith("frame.html")
    )
    other = ok(browser, "open", url=website + "/popup.html", new_tab=True)[
        "observation"
    ]["tab_id"]
    ok(browser, "select_tab", tab_id=main)
    wrong = browser.execute(
        "fill",
        tab_id=other,
        target={"label": "Frame field", "frame_id": old_frame},
        text="Wrong tab",
    )
    assert not wrong.success and wrong.content["error"]["code"] == "frame_not_found"
    ok(browser, "select_tab", tab_id=main)
    (tmp_path / "upload.txt").write_text("uploaded payload")
    data = ok(browser, "upload", target={"label": "Upload"}, paths=["upload.txt"])[
        "observation"
    ]
    assert "upload.txt" in data["text"]
    result = browser.execute(
        "upload", target={"label": "Upload"}, paths=["../outside.txt"]
    )
    assert (
        not result.success and result.content["error"]["code"] == "workspace_boundary"
    )
    ok(browser, "click", target={"role": "link", "name": "Download report"})
    downloads = ok(browser, "downloads")["downloads"]
    assert downloads
    ok(
        browser,
        "save_download",
        download_id=downloads[0]["download_id"],
        path="downloads/report.txt",
    )
    assert (
        tmp_path / "downloads/report.txt"
    ).read_text() == "A real downloaded report\n"
    assert not browser.execute(
        "save_download",
        download_id=downloads[0]["download_id"],
        path="downloads/report.txt",
    ).success
    ok(browser, "click", target={"role": "button", "name": "Delayed"})
    data = ok(browser, "wait", text="Delayed result")["observation"]
    assert "Delayed result" in data["text"]
    ok(browser, "scroll", delta_y=500)
    ok(browser, "press", text="Home")
    data = ok(browser, "read", offset=0, limit=1)
    assert len(data["text"]) == 100 and data["next_offset"] == 100


def test_dialog_screenshot_and_session_isolation(browser, website, tmp_path):
    ok(browser, "open", url=website)
    # Some Chromium revisions wait for the modal to resolve before click returns.
    result = browser.execute(
        "click", target={"role": "button", "name": "Confirm"}, timeout_ms=500
    )
    observation = result.content.get("observation") or browser.current_observation()
    assert observation["dialog"]["message"] == "Continue operation?"
    data = ok(browser, "dialog", accept=False)["observation"]
    assert "Dismissed" in data["text"]
    data = ok(browser, "screenshot")["observation"]
    assert Path(data["image_ref"]).read_bytes().startswith(b"\x89PNG")
    loop = SimpleNamespace(tool=SimpleNamespace(get_tool=lambda _: browser), config={})
    assert browser_messages(loop, {"browser_use"})[0]["images"]
    ok(browser, "observe")
    assert "images" not in browser_messages(loop, {"browser_use"})[0]
    browser.set_context_store(None, lambda: "another conversation")
    assert browser.current_observation() is None
    assert ok(browser, "tabs")["tabs"] == []
    browser.set_workspace(tmp_path / "other")
    assert browser.current_observation() is None


def test_cancel_in_flight_wait_does_not_replay_and_browser_remains_usable(
    browser, website
):
    ok(browser, "open", url=website)
    stop = threading.Event()
    browser.set_process_manager(SimpleNamespace(stop_event=stop))
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(
            browser.execute, "wait", text="Never appears", timeout_ms=30000
        )
        time.sleep(0.2)
        start = time.monotonic()
        stop.set()
        result = future.result(timeout=2)
        assert (
            time.monotonic() - start < 2
            and result.content["error"]["code"] == "cancelled"
        )
    stop.clear()
    assert "Browser workbench" in ok(browser, "observe")["observation"]["text"]


def test_detached_node_is_not_retargeted_and_missing_locator_times_out(
    browser, website
):
    data = ok(browser, "open", url=website)["observation"]
    ref = element(data, "Replace me")

    async def replace():
        await browser._session.page.locator("#replace").evaluate(
            "e=>e.outerHTML='<button id=replace>New target</button>'"
        )

    asyncio.run_coroutine_threadsafe(replace(), browser._loop).result(2)
    result = browser.execute("click", target={"ref": ref})
    assert (
        result.content["error"]["code"] == "detached_target"
        and result.content["delivery"] == "not_sent"
    )
    result = browser.execute(
        "click", target={"role": "button", "name": "Missing"}, timeout_ms=100
    )
    assert not result.success and result.content["error"]["code"] == "timeout"


def test_cancel_pending_click_aborts_driver_work_before_next_action(browser, website):
    ok(browser, "open", url=website)
    stop = threading.Event()
    browser.set_process_manager(SimpleNamespace(stop_event=stop))
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(
            browser.execute,
            "click",
            target={"role": "button", "name": "Not yet present"},
            timeout_ms=30000,
        )
        time.sleep(0.2)
        stop.set()
        result = future.result(timeout=2)
        assert result.content["error"]["code"] == "cancelled"
        assert result.content["delivery"] == "unknown"
    stop.clear()
    assert not ok(browser, "doctor")["running"]
    assert (
        "Browser workbench" in ok(browser, "open", url=website)["observation"]["text"]
    )


def test_system_chrome_fallback_without_playwright_browser_download(browser, website, monkeypatch):
    import os
    from src.tools.builtin.browser_use import executable
    installed = os.environ.get('DAENA_TEST_CHROMIUM')
    if not installed:
        pytest.skip('Requires a separately installed Chromium executable')
    browser.set_runtime_config({'browser': {'headless': True}})
    original = executable.resolve_executable
    # Simulate an absent Playwright download with an installed system browser.
    monkeypatch.setattr('shutil.which', lambda name: installed if name == 'google-chrome-stable' else None)
    monkeypatch.setattr('src.tools.builtin.browser_use.engine.resolve_executable', lambda config, bundled=None: original(config, '/nonexistent/playwright/chrome'))
    data = ok(browser, 'open', url=website)
    assert 'Ready' in data['observation']['text']
    diagnostic = ok(browser, 'doctor')
    assert diagnostic['executable_source'] == 'system'
    ok(browser, 'fill', target={'label':'Phone'}, text='09120000000')
    data = ok(browser, 'click', target={'role':'button','name':'Send code'})
    assert 'Code sent' in data['observation']['text']


def test_embedded_modal_controls_survive_busy_parent_and_context_budget(browser, website):
    from src.context.payloads import fit_json
    ok(browser, 'open', url=website)
    async def modal():
        page = browser._session.page
        await page.evaluate('''() => {
            document.body.innerHTML = '';
            for (let i=0;i<1300;i++) {
                const b=document.createElement('button'); b.textContent='Background '+i;
                document.body.appendChild(b);
            }
            const overlay=document.createElement('div');
            overlay.style.cssText='position:fixed;inset:0;background:white;z-index:100';
            const frame=document.createElement('iframe');
            frame.title='Account dialog';
            frame.srcdoc='<label>Phone<input type="tel"></label><button>Continue</button>';
            overlay.appendChild(frame); document.body.appendChild(overlay);
        }''')
        await page.frame_locator('iframe').get_by_label('Phone').wait_for()
    asyncio.run_coroutine_threadsafe(modal(), browser._loop).result(5)
    observation = ok(browser, 'observe', limit=40)['observation']
    phone = next(e for e in observation['elements'] if e['name']=='Phone')
    assert phone['frame_id'] != 'frame_0'
    assert any(e.get('occluded') for e in observation['elements'] if e['frame_id']=='frame_0')
    compact = json.loads(fit_json(observation,12000))
    assert any(e['name']=='Phone' for e in compact['elements'])
    assert any('Phone' in f.get('text','') for f in observation['frames'])
    after = ok(browser,'fill',target={'ref':phone['ref']},text='09120000000')['observation']
    assert any(e.get('value')=='09120000000' for e in after['elements'])


def test_browser_lazy_element_handles_preserve_paginated_refs(browser, website):
    """Large DOMs only expose refs for returned controls, including later pages."""
    ok(browser, "open", url=website)
    async def create_many():
        await browser._session.page.evaluate("""() => {
            const host = document.createElement('section');
            for (let i = 0; i < 180; i++) {
                const btn = document.createElement('button');
                btn.textContent = 'Paged entry ' + i;
                btn.addEventListener('click', () => {
                    document.body.dataset.selectedEntry = String(i);
                });
                host.appendChild(btn);
            }
            document.body.appendChild(host);
        }""")
    asyncio.run_coroutine_threadsafe(create_many(), browser._loop).result(5)
    first = ok(browser, "observe", query="Paged entry", limit=8)["observation"]
    assert len(first["elements"]) == 8
    assert len(browser._session.refs) == 8
    last = ok(browser, "observe", query="Paged entry", offset=160, limit=8)["observation"]
    assert len(last["elements"]) == 8
    assert len(browser._session.refs) == 8
    assert not browser.execute("click", target={"ref": first["elements"][0]["ref"]}).success
    selected = last["elements"][0]
    assert selected["name"] == "Paged entry 160"
    ok(browser, "click", target={"ref": selected["ref"]})
    async def read_selected():
        return await browser._session.page.evaluate("document.body.dataset.selectedEntry")
    assert asyncio.run_coroutine_threadsafe(read_selected(), browser._loop).result(5) == "160"


def test_screenshot_grounded_pixel_actions_and_stale_rejection(browser, website):
    """Vision input operates on browser screenshot coordinates, never desktop."""
    ok(browser, "open", url=website)
    observation = ok(browser, "screenshot")["observation"]
    snapshot = observation["snapshot_id"]
    assert observation["screenshot_size"]["width"] > 0
    assert observation["screenshot_size"]["height"] > 0
    invalid = browser.execute(
        "click_at", snapshot_id=snapshot, coordinate=[100000, 100000]
    )
    assert not invalid.success
    assert invalid.content["error"]["code"] == "outside_capture"
    # The rejection does not invalidate the original screenshot.
    x = min(10, observation["screenshot_size"]["width"] - 1)
    y = min(10, observation["screenshot_size"]["height"] - 1)
    ok(browser, "move_at", snapshot_id=snapshot, coordinate=[x, y])
    # A follow-up observation has invalidated the visual capture.
    stale = browser.execute("click_at", snapshot_id=snapshot, coordinate=[x, y])
    assert not stale.success
    assert stale.content["error"]["code"] == "stale_snapshot"


def test_visual_canvas_click_and_drag(browser, website):
    ok(browser, "open", url=website)
    async def bounds():
        return await browser._session.page.locator("#drawing").bounding_box()
    rect = asyncio.run_coroutine_threadsafe(bounds(), browser._loop).result(3)
    shot = ok(browser, "screenshot")["observation"]
    x, y = rect["x"] + 30, rect["y"] + 20
    updated = ok(browser, "click_at", snapshot_id=shot["snapshot_id"],
                 coordinate=[x, y])["observation"]
    # Canvas event offsets truncate fractional layout coordinates. A one-pixel
    # difference is expected when the element bounds have fractional origins.
    import re
    match = re.search(r"Canvas (\d+),(\d+)", updated["text"])
    assert match, updated["text"]
    actual_x, actual_y = map(int, match.groups())
    assert abs(actual_x - 30) <= 1
    assert abs(actual_y - 20) <= 1
    newer = ok(browser, "screenshot")["observation"]
    ok(browser, "drag_at", snapshot_id=newer["snapshot_id"],
       from_coordinate=[x, y], to_coordinate=[x + 5, y + 5])


def test_browser_profile_persists_across_agent_conversations(tmp_path):
    """New Agent instances in one workspace reuse a Chrome profile directory."""
    from src.tools.builtin.browser_use.engine import BrowserSession
    first = BrowserSession(tmp_path, "conversation-a", {})
    second = BrowserSession(tmp_path, "conversation-b", {})
    assert first.root == second.root
    assert first.root.parent == (tmp_path / ".daena" / "browser_use")
    isolated = BrowserSession(tmp_path, "conversation-a", {"profile_scope": "conversation"})
    other = BrowserSession(tmp_path, "conversation-b", {"profile_scope": "conversation"})
    assert isolated.root != other.root
    assert isolated.root != first.root
    different_workspace = BrowserSession(tmp_path / "other", "conversation-a", {})
    assert different_workspace.root != first.root


def test_browser_profile_scope_rejects_unknown_value():
    import pytest
    from src.tools.builtin.browser_use.contracts import validate_config
    assert validate_config({}) == {}
    assert validate_config({"profile_scope": "conversation"})["profile_scope"] == "conversation"
    with pytest.raises(ValueError, match="profile_scope"):
        validate_config({"profile_scope": "temporary"})
