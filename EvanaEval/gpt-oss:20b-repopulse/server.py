"""Simple HTTP server for RepoPulse.

Run with ``python server.py``.  It starts a server on ``localhost:8000``.
"""

import socketserver
from http.server import HTTPServer

# Import using absolute path to avoid relative import issues when running
# the module directly.  The package name is ``repopulse``.
from gpt-oss:20b-repopulse.handler import RepoPulseHandler


def run_server(host: str = "localhost", port: int = 8000):
    httpd = HTTPServer((host, port), RepoPulseHandler)
    print(f"Serving on http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()

