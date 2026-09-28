"""RepoPulse web application.

This module starts a simple HTTP server that serves a single page where a
user can paste a public GitHub repository URL and receive a quick analysis of
that repository.  The application uses only the Python standard library – no
external dependencies are required.

The server listens on :pydata:`HOST` and :pydata:`PORT`.

Running the server:
    python -m repopulse.app

The server will start and listen on ``http://localhost:8000``.
"""

from __future__ import annotations

import json
import re
import sys
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from .utils import analyze_repository

HOST = "localhost"
PORT = 8000


class RepoPulseHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the RepoPulse application."""

    def do_GET(self) -> None:  # pragma: no cover - trivial
        if self.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND, "Page not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(self._render_form().encode("utf-8"))

    def do_POST(self) -> None:  # pragma: no cover - trivial
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        data = parse_qs(body)
        repo_url = data.get("repo_url", [""])[0]
        result_html = self._handle_analysis(repo_url)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(result_html.encode("utf-8"))

    # ---------------------------------------------------------------------
    def _render_form(self, *, error: str | None = None, result: dict | None = None) -> str:
        """Return the full HTML page.

        Parameters
        ----------
        error:
            Optional error message to display.
        result:
            Optional dictionary with analysis results.
        """

        error_html = f"<p style='color:red;'>{escape(error)}</p>" if error else ""
        result_html = ""
        if result:
            languages = ", ".join(f"{k} ({v} bytes)" for k, v in result["languages"].items())
            result_html = f"""
            <h2>Analysis Results</h2>
            <ul>
                <li><strong>Name:</strong> {escape(result['name'])}</li>
                <li><strong>Description:</strong> {escape(result['description'])}</li>
                <li><strong>Stars:</strong> {result['stargazers_count']}</li>
                <li><strong>Forks:</strong> {result['forks_count']}</li>
                <li><strong>Primary Language:</strong> {escape(result['language'])}</li>
                <li><strong>Languages:</strong> {escape(languages)}</li>
                <li><strong>Contributors:</strong> {result['contributors_count']}</li>
            </ul>
            """

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>RepoPulse</title>
        </head>
        <body>
            <h1>RepoPulse</h1>
            <form method="post">
                <label for="repo_url">GitHub Repository URL:</label>
                <input type="url" id="repo_url" name="repo_url" required placeholder="https://github.com/owner/repo" style="width: 400px;" />
                <button type="submit">Analyze</button>
            </form>
            {error_html}
            {result_html}
        </body>
        </html>
        """

    # ---------------------------------------------------------------------
    def _handle_analysis(self, repo_url: str) -> str:
        """Validate the URL and perform the analysis.

        Returns the rendered HTML page.
        """
        # Basic validation of GitHub URL pattern
        pattern = re.compile(r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)$")
        match = pattern.match(repo_url.strip())
        if not match:
            return self._render_form(error="Invalid GitHub repository URL.")
        owner, repo = match.group("owner"), match.group("repo")
        try:
            analysis = analyze_repository(owner, repo)
        except Exception as exc:  # pragma: no cover - exercised via tests
            return self._render_form(error=str(exc))
        return self._render_form(result=analysis)


def run_server() -> None:
    """Start the HTTP server."""
    server = HTTPServer((HOST, PORT), RepoPulseHandler)
    print(f"RepoPulse server running at http://{HOST}:{PORT}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":  # pragma: no cover - manual execution
    run_server()

