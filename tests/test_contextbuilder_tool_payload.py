import json

from src.context.contextbuilder import ContextBuilder


def test_tool_payload_preserves_result_content_when_summary_and_evidence_exist():
    listing_text = (
        "Extracted 24 public listing records from embedded JSON.\n"
        "Title: ۹۵ متر / ۲ خواب | Price: ۵,۶۰۰,۰۰۰,۰۰۰ تومان | "
        "Neighborhood: صنعت | Post token: gaIHL6aX"
    )
    event_content = {
        "name": "web_fetch",
        "success": True,
        "summary": "Fetched https://divar.ir/s/kashan/buy-apartment (5891 chars).",
        "evidence": {"data": json.dumps({"status": 200, "extractor": "embedded_json"})},
        "effects": [],
        "content": {
            "success": True,
            "url": "https://divar.ir/s/kashan/buy-apartment",
            "status": 200,
            "extractor": "embedded_json",
            "total_chars": len(listing_text),
            "content": listing_text,
        },
    }

    builder = object.__new__(ContextBuilder)
    rendered = builder._tool_payload(event_content)
    decoded = json.loads(rendered)

    assert decoded["summary"].startswith("Fetched")
    assert decoded["result"]["extractor"] == "embedded_json"
    assert "gaIHL6aX" in decoded["result"]["content"]
    assert "۵,۶۰۰,۰۰۰,۰۰۰ تومان" in decoded["result"]["content"]


def test_tool_payload_keeps_large_but_bounded_listing_output_readable():
    listing_text = "\n".join(
        f"{i}. Title: listing {i} | Post token: tok{i:04d}" for i in range(24)
    )
    event_content = {
        "name": "web_fetch",
        "success": True,
        "summary": "Fetched public listing data.",
        "evidence": {"url": "https://example.com"},
        "content": {
            "success": True,
            "extractor": "embedded_json",
            "content": listing_text,
        },
    }

    builder = object.__new__(ContextBuilder)
    rendered = builder._tool_payload(event_content)
    assert len(rendered) <= ContextBuilder.MAX_TOOL_CHARS
    decoded = json.loads(rendered)
    assert "tok0000" in decoded["result"]["content"]
    assert "tok0023" in decoded["result"]["content"]
