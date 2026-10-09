from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from src.tools.builtin.web.httpclient import HttpResponse
from src.tools.builtin.web.webfetch import WebFetch
from src.tools.builtin.web.websearch import WebSearch
from src.tools.builtin.web.backends import SearchHit


def test_web_fetch_saves_full_text_and_returns_only_receipt(tmp_path):
    body = ("<html><head><title>Example</title></head><body>" + "A" * 12000 + "</body></html>").encode()
    response = HttpResponse(
        url="https://example.com/data", status=200, content_type="text/html",
        charset="utf-8", body=body, truncated=False,
    )
    tool = WebFetch()
    tool.set_workspace(tmp_path)
    with patch("src.tools.builtin.web.webfetch.http_get", return_value=response):
        result = tool.execute(url=response.url, max_chars=500, save_to="raw/page.txt")

    saved = (tmp_path / "raw" / "page.txt").read_text(encoding="utf-8")
    assert result.success
    assert len(saved) > 500
    assert result.content["path"] == "raw/page.txt"
    assert result.content["bytes"] == len(saved.encode("utf-8"))
    assert result.content["sha256"]
    assert len(result.content["preview"]) <= 1500
    assert "content" not in result.content
    assert result.metadata["effects"][0]["action"] == "create"


def test_web_fetch_json_extractor_saves_raw_preloaded_state(tmp_path):
    state = {"nb": {"listWidgets": [{"id": 1, "all_fields": {"nested": True}}], "extra": "preserve me"}}
    html = "<html><script>window.__PRELOADED_STATE__ = " + json.dumps(state) + ";</script></html>"
    response = HttpResponse(
        url="https://example.com/state", status=200, content_type="text/html",
        charset="utf-8", body=html.encode(), truncated=False,
    )
    tool = WebFetch()
    tool.set_workspace(tmp_path)
    with patch("src.tools.builtin.web.webfetch.http_get", return_value=response):
        result = tool.execute(url=response.url, extractor="json", save_to="state.json", save_format="json")
    saved = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert result.success
    assert saved == state
    assert result.content["record_count"] == 1


def test_web_fetch_json_extractor_errors_when_state_is_unavailable():
    response = HttpResponse(
        url="https://example.com", status=200, content_type="text/html",
        charset="utf-8", body=b"<html><p>no embedded state</p></html>", truncated=False,
    )
    with patch("src.tools.builtin.web.webfetch.http_get", return_value=response):
        result = WebFetch().execute(url=response.url, extractor="json")
    assert not result.success
    assert result.content["error"]["type"] == "json_extraction_unavailable"


def test_web_search_saves_json_results(tmp_path):
    from src.tools.builtin.web import websearch as module

    class Backend:
        name = "test"
        def is_available(self): return True
        def unavailable_reason(self): return ""
        def search(self, *args, **kwargs):
            return [SearchHit(title="One", url="https://example.com/1", snippet="first", source="test", published="")]

    tool = WebSearch()
    tool.set_workspace(tmp_path)
    with patch.object(module, "select_backends", return_value=[Backend()]):
        result = tool.execute(query="example", save_to="search.json", save_format="json")
    saved = json.loads((tmp_path / "search.json").read_text(encoding="utf-8"))
    assert result.success
    assert result.content["record_count"] == 1
    assert saved["results"][0]["title"] == "One"
    assert "content" not in result.content


def test_web_save_rejects_path_escape(tmp_path):
    tool = WebFetch()
    tool.set_workspace(tmp_path)
    response = HttpResponse(
        url="https://example.com", status=200, content_type="text/plain",
        charset="utf-8", body=b"safe", truncated=False,
    )
    with patch("src.tools.builtin.web.webfetch.http_get", return_value=response):
        result = tool.execute(url=response.url, save_to="../outside.txt")
    assert not result.success
    assert result.content["error"]["type"] == "save_failed"
