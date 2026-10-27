import json
import os
import socket
import subprocess
import time
from urllib.request import urlopen

import pytest

from .conftest import ok

pytestmark = pytest.mark.browser


def test_cdp_disconnect_keeps_external_browser_alive(browser, website, tmp_path):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        executable = (
            os.environ.get("DAENA_TEST_CHROMIUM") or pw.chromium.executable_path
        )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen(
        [
            executable,
            "--headless",
            "--no-sandbox",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={tmp_path / 'external-profile'}",
            "about:blank",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    endpoint = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                with urlopen(endpoint + "/json/version", timeout=0.3) as response:
                    assert json.load(response)["webSocketDebuggerUrl"]
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)
        browser.set_runtime_config({"browser": {"cdp_url": endpoint}})
        existing = ok(browser, "tabs")["tabs"]
        assert existing and existing[0]["url"] == "about:blank"
        assert (
            "Browser workbench"
            in ok(browser, "open", url=website)["observation"]["text"]
        )
        data = ok(browser, "close")
        assert data["detached"]
        assert process.poll() is None
        with urlopen(endpoint + "/json", timeout=1) as response:
            pages = json.load(response)
            assert any(page["url"].startswith(website) for page in pages)
            assert any(page["url"] == "about:blank" for page in pages)
    finally:
        browser.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
