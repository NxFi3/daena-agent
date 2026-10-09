import json
from pathlib import Path
from uuid import uuid4

from src.agent.agentloop import Loop
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult


class FakeModel:
    defaultConfig = {"num_ctx": 4096}


def text_result(text, usage=1):
    return LLMResult(
        response=text,
        message={"role": "assistant", "content": text},
        tool_calls=[],
        thinking=None,
        usage=usage,
    )


def read_call_result():
    call = {
        "id": "read-1",
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": json.dumps({"file_path": "hello.txt"}),
        },
    }
    return LLMResult(
        response="",
        message={"role": "assistant", "content": "", "tool_calls": [call]},
        tool_calls=[call],
        thinking=None,
        usage=2,
    )


class ReviewRecoveryLLM:
    def __init__(self):
        self.model = FakeModel()
        self.calls = 0

    def generate(self, messages, tools=None, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return text_result("I'm sorry, I can't help with that.")
        if self.calls == 2:
            return text_result(
                json.dumps({
                    "decision": "continue",
                    "reason": "No action was attempted and the requested outcome is unmet.",
                    "next_action": "Use the available file-reading tool.",
                })
            )
        if self.calls == 3:
            return read_call_result()
        if self.calls == 4:
            return text_result("The file contains: hello baseline.")
        if self.calls == 5:
            return text_result(
                json.dumps({
                    "decision": "complete",
                    "reason": "The file was read successfully and the result is reported.",
                    "next_action": "",
                })
            )
        raise AssertionError(f"Unexpected model call {self.calls}")


def test_completion_review_recovers_from_premature_refusal(tmp_path):
    (tmp_path / "hello.txt").write_text("hello baseline", encoding="utf-8")
    llm = ReviewRecoveryLLM()
    config = {
        "llm": {"provider_config": {"generation_config": {"num_ctx": 4096}}},
        "context": {
            "safe_margin": 0,
            "recent_event_limit": 20,
            "compaction_enabled": False,
        },
        "retrieval": {"top_k": 1},
        "security": {
            "workspace_only": True,
            "allow_background": False,
            "allow_network_tools": True,
            "force_approve": False,
        },
        "memory": {"stm_db_path": str(tmp_path / "stm.db")},
        "max_agent_iterations": 6,
        "completion_review": {"enabled": True, "max_retries": 2},
        "experience": {"enabled": False},
    }
    loop = Loop(config, llm)
    loop.session_id = uuid4()

    try:
        task = ContextEvent(
            role=ContextRole.USER,
            type=ContextType.MESSAGE,
            content="Read hello.txt and tell me what it contains.",
        )
        result = loop.run(task, workspace_directory=str(tmp_path))

        assert result is not None
        assert result.response == "The file contains: hello baseline."
        assert llm.calls == 5
        metrics = loop.get_metrics()
        assert metrics["completed"] is True
        assert metrics["completion_review_calls"] == 2
        assert metrics["completion_review_retries"] == 1
        assert metrics["tool_successes"] == 1
    finally:
        loop.close()


def test_completion_review_is_disabled_unless_enabled_in_config(tmp_path):
    from src.agent.completionreview import CompletionReviewer

    config = {
        "llm": {"provider_config": {"generation_config": {"num_ctx": 4096}}},
        "context": {"safe_margin": 0, "recent_event_limit": 10, "compaction_enabled": False},
        "retrieval": {"top_k": 1},
        "security": {
            "workspace_only": True,
            "allow_background": False,
            "allow_network_tools": True,
            "force_approve": False,
        },
        "memory": {"stm_db_path": str(tmp_path / "stm-disabled.db")},
        "max_agent_iterations": 2,
        "experience": {"enabled": False},
    }
    loop = Loop(config, ReviewRecoveryLLM())
    try:
        assert loop.completion_reviewer is None
    finally:
        loop.close()


def test_completion_review_evidence_preserves_full_fetch_and_artifact_preview(tmp_path):
    """Reviewer evidence must not shrink a real multi-record page to 450 chars."""
    from src.models.ContextEvent import ContextPriority, ContextRole, ContextType

    loop = Loop(
        {
            "llm": {"provider_config": {"generation_config": {"num_ctx": 4096}}},
            "context": {"safe_margin": 0, "recent_event_limit": 30, "compaction_enabled": False},
            "retrieval": {"top_k": 1},
            "security": {
                "workspace_only": True,
                "allow_background": False,
                "allow_network_tools": True,
                "force_approve": False,
            },
            "memory": {"stm_db_path": str(tmp_path / "evidence.db")},
            "max_agent_iterations": 4,
            "experience": {"enabled": False},
        },
        ReviewRecoveryLLM(),
    )
    loop.session_id = uuid4()

    source_text = "\n".join(
        f"{n}. Title: real listing {n} | Displayed price: {n * 1000} | Post token: token{n}"
        for n in range(1, 26)
    )
    csv_preview = (
        "price_toman,area_m2,rooms,parking,elevator,neighborhood,listing_url,retrieved_at\n"
        + "\n".join(
            f"{n * 1000},,{n % 4},,,,https://divar.ir/v/token{n},2026-10-09T12:30:00+04:00"
            for n in range(1, 11)
        )
    )
    fetch_event = ContextEvent(
        role=ContextRole.TOOL,
        type=ContextType.TOOL_RESULT,
        content=json.dumps({
            "name": "web_fetch",
            "success": True,
            "summary": "Fetched a public search page.",
            "content": {
                "url": "https://divar.ir/s/kashan/buy-apartment",
                "extractor": "embedded_json",
                "total_chars": len(source_text),
                "content": source_text,
            },
        }, ensure_ascii=False),
        step=1,
    )
    write_event = ContextEvent(
        role=ContextRole.TOOL,
        type=ContextType.TOOL_RESULT,
        content=json.dumps({
            "name": "write_file",
            "success": True,
            "summary": "Wrote dataset.csv.",
            "content": {
                "path": str(tmp_path / "dataset.csv"),
                "bytes_written": len(csv_preview.encode("utf-8")),
                "lines_written": 11,
                "preview": csv_preview,
                "preview_truncated": False,
            },
        }, ensure_ascii=False),
        step=3,
    )

    try:
        loop.stm.get_recent = lambda _session_id, limit=28: [fetch_event, write_event]
        evidence = loop._completion_review_evidence()
        fetch = next(item for item in evidence if item.get("tool") == "web_fetch")
        written = next(item for item in evidence if item.get("tool") == "write_file")

        assert len(fetch["untrusted_output_excerpt"]) > 1000
        assert "Title: real listing 25" in fetch["untrusted_output_excerpt"]
        assert "extractor" not in fetch or fetch.get("extractor") == "embedded_json"
        assert "artifact_preview" in written
        assert "https://divar.ir/v/token10" in written["artifact_preview"]
        assert written["artifact_preview_truncated"] is False
    finally:
        loop.close()


class BlockedCompletionLLM:
    def __init__(self):
        self.model = FakeModel()

    def generate(self, messages, tools=None, **_kwargs):
        if tools == []:
            return text_result(json.dumps({
                "decision": "blocked",
                "reason": "The requested artifact was not verified.",
                "next_action": "Inspect the file.",
            }))
        return text_result("The artifact is complete and validated.")


def test_blocked_completion_does_not_return_false_success_claim(tmp_path):
    config = {
        "llm": {"provider_config": {"generation_config": {"num_ctx": 4096}}},
        "context": {"safe_margin": 0, "recent_event_limit": 20, "compaction_enabled": False},
        "retrieval": {"top_k": 1},
        "security": {
            "workspace_only": True,
            "allow_background": False,
            "allow_network_tools": True,
            "force_approve": False,
        },
        "memory": {"stm_db_path": str(tmp_path / "blocked.db")},
        "max_agent_iterations": 3,
        "completion_review": {"enabled": True, "max_retries": 1},
        "experience": {"enabled": False},
    }
    loop = Loop(config, BlockedCompletionLLM())
    loop.session_id = uuid4()
    try:
        task = ContextEvent(
            role=ContextRole.USER,
            type=ContextType.MESSAGE,
            content="Create and verify an artifact.",
        )
        result = loop.run(task, workspace_directory=str(tmp_path))
        assert result is not None
        assert "I stopped before finishing the task." in result.response
        assert "artifact is complete and validated" not in result.response
        metrics = loop.get_metrics()
        assert metrics["completed"] is False
        assert metrics["stop_reason"].startswith("Completion review rejected the result")
    finally:
        loop.close()
