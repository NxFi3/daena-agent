"""HTTP request handler for RepoPulse.

The handler serves static files from the ``static`` directory and
provides a single JSON API endpoint at ``/api/analyze``.
"""

import json
import os
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from .utils import (
    get_repo_contributors,
    get_repo_info,
    get_repo_languages,
    parse_github_url,
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")


class RepoPulseHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/analyze":
            self.handle_api_analyze(parsed.query)
        else:
            self.handle_static(parsed.path)

    def handle_api_analyze(self, query: str):
        params = parse_qs(query)
        url_list = params.get("url")
        if not url_list:
            self.respond_error(400, "Missing 'url' query parameter")
            return
        repo_url = url_list[0]
        try:
            owner, repo = parse_github_url(repo_url)
        except ValueError as e:
            self.respond_error(400, str(e))
            return
        try:
            info = get_repo_info(owner, repo)
            langs = get_repo_languages(owner, repo)
            contributors = get_repo_contributors(owner, repo)
            primary_lang = max(langs.items(), key=lambda kv: kv[1])[0] if langs else None
            result = {
                "name": info["name"],
                "description": info["description"],
                "stars": info["stargazers_count"],
                "forks": info["forks_count"],
                "primary_language": primary_lang,
                "language_stats": langs,
                "contributor_count": len(contributors),
            }
            self.respond_json(200, result)
        except RuntimeError as e:
            self.respond_error(500, str(e))

    def handle_static(self, path: str):
        # Default to index.html
        if path == "/" or path == "":
            path = "/index.html"
        file_path = os.path.normpath(os.path.join(STATIC_DIR, path.lstrip("/")))
        # Prevent directory traversal
        if not file_path.startswith(STATIC_DIR):
            self.respond_error(403, "Forbidden")
            return
        if not os.path.exists(file_path):
            self.respond_error(404, "File not found")
            return
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            mime = self.guess_mime(file_path)
            self.send_header("Content-Type", mime)
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.respond_error(500, f"Server error: {e}")

    def guess_mime(self, path: str) -> str:
        if path.endswith(".html"):
            return "text/html"
        if path.endswith(".js"):
            return "application/javascript"
        if path.endswith(".css"):
            return "text/css"
        return "application/octet-stream"

    def respond_json(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def respond_error(self, status: int, message: str):
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

