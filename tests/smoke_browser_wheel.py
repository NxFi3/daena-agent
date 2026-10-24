"""Run outside the checkout using the installed wheel's browser integration."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Thread

import src
from src.tools.ToolRegistry import ToolRegistry


def main():
    source = Path(sys.argv[1]).resolve()
    assert not Path(src.__file__).resolve().is_relative_to(source)
    registry = ToolRegistry()
    registry.discover()
    tool = registry.get("browser_use")
    assert tool is not None
    with TemporaryDirectory(prefix="daena-wheel-browser-") as directory:
        root = Path(directory)
        (root / "index.html").write_text("""<!doctype html><meta charset="utf-8">
        <label>Message<input id="message"></label><button onclick="document.querySelector('#result').textContent=document.querySelector('#message').value">Save</button><p id="result"></p>""")

        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(Handler, directory=directory)
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        tool.set_workspace(directory)
        tool.set_runtime_config(
            {
                "browser": {
                    "headless": True,
                    "executable_path": os.environ.get("DAENA_TEST_CHROMIUM", ""),
                }
            }
        )
        try:
            for arguments in [
                {"action": "open", "url": f"http://127.0.0.1:{server.server_port}"},
                {
                    "action": "fill",
                    "target": {"label": "Message"},
                    "text": "سلام wheel",
                },
                {"action": "click", "target": {"role": "button", "name": "Save"}},
            ]:
                result = tool.execute(**arguments)
                assert result.success, result.content
            assert "سلام wheel" in result.content["observation"]["text"]
        finally:
            tool.close()
            server.shutdown()
            server.server_close()
            thread.join()
    print("INSTALLED_BROWSER_WHEEL_OK")


if __name__ == "__main__":
    main()
