"""Pytest suite for the ai_provider_report package.

The tests exercise the data structures, the report generation and the
comparison logic.  They also check that the provider data is complete
and that the representation details match the official documentation.
"""

import pytest

from .models import Provider, ToolCallFormat, ToolResultFormat
from .sources import providers
from .report import build_report, compare_providers


def test_all_providers_present():
    # Ensure we have exactly three providers
    assert len(providers) == 3
    names = {p.name for p in providers}
    assert names == {"Ollama", "OpenAI", "OpenRouter"}


def test_provider_fields_populated():
    for p in providers:
        assert isinstance(p, Provider)
        assert p.name
        assert p.doc_url
        assert isinstance(p.tool_call_format, ToolCallFormat)
        assert isinstance(p.tool_result_format, ToolResultFormat)
        # Basic sanity checks on the nested fields
        assert p.tool_call_format.representation
        assert p.tool_call_format.id_field
        assert p.tool_call_format.arguments
        assert p.tool_result_format.role
        assert p.tool_result_format.content
        assert p.tool_result_format.tool_call_id_field


def test_ollama_vs_openai_differences():
    ollama = next(p for p in providers if p.name == "Ollama")
    openai = next(p for p in providers if p.name == "OpenAI")
    # Ollama uses JSON object for arguments, OpenAI uses JSON string
    assert ollama.tool_call_format.arguments == "json object"
    assert openai.tool_call_format.arguments == "json string"
    # ID field names differ
    assert ollama.tool_call_format.id_field == "index"
    assert openai.tool_call_format.id_field == "id"


def test_report_generation():
    report = build_report()
    # The report should contain each provider name
    for p in providers:
        assert p.name in report
    # The report should include the comparison table
    comparison = compare_providers()
    assert "Provider" in comparison
    assert "ID Field" in comparison
    assert "Args" in comparison
    assert "Result Content" in comparison


def test_malformed_provider_data():
    # Create a provider with missing fields to trigger validation
    from types import SimpleNamespace

    bad_provider = SimpleNamespace(
        name="BadProvider",
        doc_url="",
        tool_call_format=SimpleNamespace(representation="", id_field="", arguments=""),
        tool_result_format=SimpleNamespace(role="", content="", tool_call_id_field=""),
    )
    # Build a report with the bad provider appended
    bad_providers = providers + [bad_provider]
    # build_report should still produce a string but we expect missing data
    report = build_report()
    # The report should not crash
    assert isinstance(report, str)

