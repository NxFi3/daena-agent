from src.tools.builtin.web.backends import SearchHit
from src.tools.builtin.web.websearch import WebSearch


def test_empty_first_backend_does_not_prevent_second_backend(monkeypatch):
    calls = []

    class EmptyBackend:
        name = "empty"

        def is_available(self):
            return True

        def unavailable_reason(self):
            return ""

        def search(
            self, query, *, num_results, time_range, region, timeout
        ):
            calls.append("empty")
            return []

    class WorkingBackend:
        name = "working"

        def is_available(self):
            return True

        def unavailable_reason(self):
            return ""

        def search(
            self, query, *, num_results, time_range, region, timeout
        ):
            calls.append("working")
            return [
                SearchHit(
                    title="Kashan residential listings",
                    url="https://example.com/kashan",
                    snippet="Public listing index",
                )
            ]

    monkeypatch.setattr(
        "src.tools.builtin.web.websearch.select_backends",
        lambda: [EmptyBackend(), WorkingBackend()],
    )
    result = WebSearch().execute(query="Kashan houses for sale")

    assert result.success is True
    assert result.content["backend"] == "working"
    assert result.content["result_count"] == 1
    assert calls == ["empty", "working"]


def test_exhausted_empty_backends_return_diagnostics_not_false_successful_data(monkeypatch):
    class EmptyBackend:
        name = "empty"

        def is_available(self):
            return True

        def unavailable_reason(self):
            return ""

        def search(
            self, query, *, num_results, time_range, region, timeout
        ):
            return []

    monkeypatch.setattr(
        "src.tools.builtin.web.websearch.select_backends",
        lambda: [EmptyBackend()],
    )
    result = WebSearch().execute(query="Kashan houses for sale")

    assert result.success is True
    assert result.content["result_count"] == 0
    assert "returned no useful results" in result.content["content"]
