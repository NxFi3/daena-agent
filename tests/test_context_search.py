import json
from uuid import uuid4

from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.tools.builtin.context_search.tool import ContextSearch


class FakeSTM:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.events


def test_context_search_returns_bounded_tool_evidence():
    event = ContextEvent(
        role=ContextRole.TOOL,
        type=ContextType.TOOL_RESULT,
        content=json.dumps({
            "name": "web_fetch",
            "success": True,
            "summary": "Fetched earlier public records.",
            "content": {
                "url": "https://example.invalid/listings",
                "content": "UniqueListingToken price 5000000; " + ("x" * 2000),
            },
        }),
        step=7,
    )
    memory = FakeSTM([event])
    tool = ContextSearch()
    tool.set_context_store(memory, lambda: "session-1")

    result = tool.execute(query="UniqueListingToken", top_k=2)

    assert result.success
    assert memory.calls[0]["session_id"] == "session-1"
    assert memory.calls[0]["include_tool_results"] is True
    assert result.content["result_count"] == 1
    record = result.content["results"][0]
    assert record["tool"] == "web_fetch"
    assert "UniqueListingToken" in record["excerpt"]
    assert len(record["excerpt"]) <= 1400


def test_context_search_fails_closed_when_not_bound_to_a_session():
    result = ContextSearch().execute(query="some previous result")
    assert not result.success
    assert result.content["error"]["type"] == "context_store_unavailable"
