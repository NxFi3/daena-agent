import pytest

from src.tools.builtin.web.httpclient import FetchError, _normalize_request_headers


def test_normalizes_supported_non_sensitive_headers():
    assert _normalize_request_headers({
        "accept-language": "fa-IR,fa;q=0.9",
        "User-Agent": "DaenaTest/1.0",
    }) == {
        "Accept-Language": "fa-IR,fa;q=0.9",
        "User-Agent": "DaenaTest/1.0",
    }


@pytest.mark.parametrize("name", ["Cookie", "Authorization", "Host", "Connection", "X-Internal"])
def test_rejects_unsupported_or_sensitive_headers(name):
    with pytest.raises(FetchError) as caught:
        _normalize_request_headers({name: "secret"})
    assert caught.value.error_type == "invalid_headers"


def test_rejects_newline_in_header_value():
    with pytest.raises(FetchError) as caught:
        _normalize_request_headers({"Referer": "https://example.com/\r\nCookie: secret"})
    assert caught.value.error_type == "invalid_headers"


def test_rejects_non_string_header_values():
    with pytest.raises(FetchError) as caught:
        _normalize_request_headers({"Accept": ["text/html"]})
    assert caught.value.error_type == "invalid_headers"
