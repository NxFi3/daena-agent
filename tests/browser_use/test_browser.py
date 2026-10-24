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
