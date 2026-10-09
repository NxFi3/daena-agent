import json
from unittest.mock import patch

from src.tools.builtin.web.httpclient import HttpResponse
from src.tools.builtin.web.webfetch import WebFetch


def _html_with_preloaded_rows():
    row = {
        "data": {
            "widgetType": "POST_ROW",
            "dto": {
                "data": {
                    "title": "۹۵ متر / ۲ خواب",
                    "token": "abc123XY",
                    "middle_description_text": "۵,۶۰۰,۰۰۰,۰۰۰ تومان",
                    "bottom_description_text": "ملک کاشان در صنعت",
                    "action": {
                        "payload": {
                            "token": "abc123XY",
                            "web_info": {
                                "district_persian": "صنعت",
                                "city_persian": "کاشان",
                            },
                        }
                    },
                }
            },
        }
    }
    state = {
        "nb": {
            "listTopWidgets": [
                {"dto": {"data": {"text": "Kashan apartments for sale"}}}
            ],
            "listWidgets": [
                row,
                {"data": {"widgetType": "OTHER", "dto": {"data": {"text": "ignore"}}}},
            ],
        }
    }
    return (
        "<html><head><title>Divar</title></head><body>"
        "<p>Add to home screen</p>"
        "<script>window.__PRELOADED_STATE__ = "
        + json.dumps(state, ensure_ascii=False)
        + ";</script></body></html>"
    ).encode("utf-8")


def test_extracts_public_preloaded_post_rows_when_readability_is_sparse():
    html = _html_with_preloaded_rows().decode("utf-8")
    text, title, links = WebFetch._extract_preloaded_state(
        html, "https://divar.ir/s/kashan/buy-apartment"
    )

    assert "Extracted 1 public listing records" in text
    assert "۹۵ متر / ۲ خواب" in text
    assert "۵,۶۰۰,۰۰۰,۰۰۰ تومان" in text
    assert "Neighborhood: صنعت" in text
    assert "City: کاشان" in text
    assert "https://divar.ir/v/abc123XY" in text
    assert title == "Kashan apartments for sale"
    assert links == [
        {"text": "۹۵ متر / ۲ خواب", "url": "https://divar.ir/v/abc123XY"}
    ]


def test_web_fetch_uses_embedded_state_as_fallback():
    response = HttpResponse(
        url="https://divar.ir/s/kashan/buy-apartment",
        status=200,
        content_type="text/html",
        charset="utf-8",
        body=_html_with_preloaded_rows(),
        truncated=False,
    )
    with patch(
        "src.tools.builtin.web.webfetch.http_get",
        return_value=response,
    ):
        result = WebFetch().execute(url=response.url, include_links=True)

    assert result.success is True
    assert result.content["extractor"] == "embedded_json"
    assert result.content["title"] in ("Divar", "Kashan apartments for sale")
    assert "۵,۶۰۰,۰۰۰,۰۰۰ تومان" in result.content["content"]
    assert result.content["links"][0]["url"] == "https://divar.ir/v/abc123XY"


def test_invalid_or_absent_preloaded_state_is_ignored():
    text, title, links = WebFetch._extract_preloaded_state(
        "<html><script>window.__PRELOADED_STATE__ = {broken};</script></html>",
        "https://example.com/",
    )
    assert (text, title, links) == ("", "", [])
