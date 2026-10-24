import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
import threading

import pytest

from src.tools.builtin.browser_use.tool import BrowserUseTool


SITE = """<!doctype html><html><head><meta charset="utf-8"><title>Browser workbench</title></head>
<body><h1>Browser workbench</h1>
<label>Phone<input id="phone" type="tel"></label><button id="send" onclick="document.querySelector('#code-step').hidden=false; document.querySelector('#status').textContent='Code sent'">Send code</button>
<section id="code-step" hidden><label>One-time code<input id="code"></label><button onclick="document.querySelector('#status').textContent=document.querySelector('#code').value==='123456'?'Signed in':'Invalid code'">Sign in</button></section>
<p id="status">Ready</p><label>Password<input type="password" value="never-include-me"></label>
<canvas id="drawing" width="200" height="80" style="border:1px solid" onclick="document.querySelector('#status').textContent='Canvas '+event.offsetX+','+event.offsetY"></canvas>
<div role="button" aria-label="Drag source" style="display:inline-block;width:80px;height:40px;background:lightblue" onpointerdown="window.dragStarted=true">Source</div>
<div role="button" aria-label="Drop target" style="display:inline-block;width:80px;height:40px;background:pink" onpointerup="if(window.dragStarted) document.querySelector('#status').textContent='Dropped'">Target</div>
<label>Accept terms<input type="checkbox" id="terms"></label>
<label for="plan">Plan</label><select id="plan"><option value="basic">Basic</option><option value="pro">Pro</option></select>
<button onclick="document.querySelector('#status').textContent='Left chosen'">Duplicate</button>
<button onclick="document.querySelector('#status').textContent='Right chosen'">Duplicate</button>
<button id="replace" onclick="this.outerHTML='<button id=replace>Replacement</button>'">Replace me</button>
<button onclick="setTimeout(()=>document.querySelector('#status').textContent='Delayed result',150)">Delayed</button>
<a href="/popup.html" target="_blank">Open popup</a>
<button onclick="document.querySelector('#status').textContent=confirm('Continue operation?')?'Accepted':'Dismissed'">Confirm</button>
<label>Upload<input type="file" id="upload" onchange="document.querySelector('#status').textContent=this.files[0].name"></label>
<a href="/report.txt" download>Download report</a>
<iframe title="Payment" src="/frame.html"></iframe><div id="shadow"></div>
<script>const root=document.querySelector('#shadow').attachShadow({mode:'open'}); root.innerHTML='<label>Shadow input<input></label><button>Shadow button</button>';</script>
<div style="height:1800px">Scroll region</div><p>End of page</p></body></html>"""


@pytest.fixture(scope="module")
def website(tmp_path_factory):
    root = tmp_path_factory.mktemp("browser-site")
    (root / "index.html").write_text(SITE)
    (root / "frame.html").write_text(
        "<label>Frame field<input></label><button>Frame button</button>"
    )
    (root / "popup.html").write_text(
        "<h1>Popup loaded</h1><button>Popup button</button>"
    )
    (root / "report.txt").write_text("A real downloaded report\n")

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Handler, directory=str(root))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.fixture
def browser(tmp_path):
    if os.environ.get("DAENA_TEST_BROWSER") != "1":
        pytest.skip("Set DAENA_TEST_BROWSER=1 for real Chromium tests")
    pytest.importorskip("playwright")
    tool = BrowserUseTool()
    tool.set_workspace(tmp_path)
    executable = os.environ.get("DAENA_TEST_CHROMIUM", "")
    tool.set_runtime_config(
        {"browser": {"headless": True, "executable_path": executable}}
    )
    yield tool
    tool.close()


def ok(tool, action, **kwargs):
    result = tool.execute(action, **kwargs)
    assert result.success, result.content
    return result.content


def element(observation, name):
    return next(row["ref"] for row in observation["elements"] if row["name"] == name)
