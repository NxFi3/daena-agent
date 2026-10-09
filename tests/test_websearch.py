from src.tools.builtin.web.backends import SearchHit
from src.tools.builtin.web.websearch import WebSearch


def test_empty_time_range_is_treated_as_no_filter(monkeypatch):
    observed = {}

    class FakeBackend:
        name = "fake"

        def is_available(self):
            return True

        def unavailable_reason(self):
            return ""

        def search(
            self,
            query,
            *,
            num_results,
            time_range,
            region,
            timeout,
        ):
            observed["time_range"] = time_range
            observed["region"] = region
            return [
                SearchHit(
                    title="Kashan apartment listings",
                    url="https://example.com/listings",
                    snippet="Apartment listings in Kashan",
                )
            ]

    monkeypatch.setattr(
        "src.tools.builtin.web.websearch.select_backends",
        lambda: [FakeBackend()],
    )

    result = WebSearch().execute(
        query="Kashan apartment listings",
        time_range="",
        region="",
    )

    assert result.success is True
    assert observed["time_range"] is None
    assert observed["region"] is None


def test_invalid_nonempty_time_range_still_returns_argument_error(monkeypatch):
    class UnusedBackend:
        name = "unused"

        def is_available(self):
            raise AssertionError("backend must not be called")

        def unavailable_reason(self):
            return ""

    monkeypatch.setattr(
        "src.tools.builtin.web.websearch.select_backends",
        lambda: [UnusedBackend()],
    )

    result = WebSearch().execute(
        query="Kashan apartment listings",
        time_range="forever",
    )

    assert result.success is False
    assert result.content["error"]["type"] == "invalid_argument"
