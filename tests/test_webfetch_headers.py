from unittest.mock import patch

from src.tools.builtin.web.httpclient import FetchError, HttpResponse
from src.tools.builtin.web.webfetch import WebFetch


def test_web_fetch_passes_optional_headers_to_http_client():
    response = HttpResponse(
        url="https://example.com/",
        status=200,
        content_type="text/html",
        charset="utf-8",
        body=b"<html><head><title>Example</title></head><body><h1>Hello</h1></body></html>",
        truncated=False,
    )
    headers = {"Accept-Language": "fa-IR,fa;q=0.9"}
    with patch(
        "src.tools.builtin.web.webfetch.http_get", return_value=response
    ) as mocked:
        result = WebFetch().execute(url=response.url, headers=headers)

    assert result.success is True
    assert mocked.call_args.kwargs["headers"] == headers


def test_invalid_header_arguments_are_returned_as_tool_errors():
    with patch(
        "src.tools.builtin.web.webfetch.http_get",
        side_effect=FetchError("invalid_headers", "Cookie is not allowed."),
    ):
        result = WebFetch().execute(
            url="https://example.com/",
            headers={"Cookie": "session=secret"},
        )

    assert result.success is False
    assert result.content["error"]["type"] == "invalid_headers"


def test_empty_javascript_app_html_has_actionable_diagnostic():
    body = (
        b'<html><head><title>App</title><script id="__NEXT_DATA__">{}</script>'
        b'</head><body><div id="root"></div>'
        + (b" " * 14000)
        + b"</body></html>"
    )
    response = HttpResponse(
        url="https://example.com/",
        status=200,
        content_type="text/html",
        charset="utf-8",
        body=body,
        truncated=False,
    )
    with patch("src.tools.builtin.web.webfetch.http_get", return_value=response):
        result = WebFetch().execute(url=response.url)

    assert result.success is True
    assert "JavaScript" in result.content.get("note", "")
    assert "do not keep retrying" in result.content.get("note", "").lower()
