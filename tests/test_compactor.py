from src.context.compactor import Compactor
from src.models.LLMResult import LLMResult


class FakeLLM:
    def __init__(self):
        self.messages = None

    def generate(self, messages, tools=None, options=None):
        self.messages = messages
        return LLMResult(
            response="Task: fix\nCompleted: parser updated",
            message={"role": "assistant", "content": "Task: fix\nCompleted: parser updated"},
            tool_calls=[],
            thinking=None,
            usage=12,
        )


def test_compactor_returns_model_summary():
    llm = FakeLLM()
    compacted = Compactor(llm).compact("long history", 128)

    assert "parser updated" in compacted
    assert llm.messages
    assert "long history" in llm.messages[0]["content"]


def _builder_for_chunk_tests():
    from src.context.contextbuilder import ContextBuilder
    config = {
        "llm": {"provider": "ollama", "provider_config": {"generation_config": {"num_ctx": 4096}}},
        "context": {
            "safe_margin": 0,
            "compaction_enabled": True,
            "compaction_target_tokens": 400,
            "max_prompt_tokens": 4096,
        },
        "experience": {"enabled": False},
    }
    return ContextBuilder(config, FakeLLM())


def test_compaction_input_summarizes_chunks_before_final_aggregation():
    builder = _builder_for_chunk_tests()
    calls = []

    def summarize(text, target):
        calls.append((text, target))
        return f"summary-{len(calls)}-of-{len(text)}"

    builder.compactor.compact = summarize
    builder.tokenbudget.budget = 2048
    builder.tokenbudget.chars_per_token = 2.0
    builder.compaction_target_tokens = 400
    raw = "\\n\\n".join(
        f"SECTION {index}: " + (chr(65 + index % 26) * 1100)
        for index in range(6)
    )
    summarized_input = builder._compaction_input(raw)
    assert len(calls) >= 3
    assert all(target == 200 for _, target in calls)
    assert summarized_input.count("summary-") == len(calls)
    assert "middle of history omitted" not in summarized_input


def test_compaction_input_uses_head_tail_only_when_chunk_summary_fails():
    builder = _builder_for_chunk_tests()
    builder.tokenbudget.budget = 1024
    builder.tokenbudget.chars_per_token = 2.0
    calls = 0

    def summarize(text, target):
        nonlocal calls
        calls += 1
        return "" if calls == 2 else "chunk summary"

    builder.compactor.compact = summarize
    raw = "HEAD_MARKER\\n\\n" + ("middle data " * 1000) + "\\n\\nTAIL_MARKER"
    result = builder._compaction_input(raw)
    assert "middle of history omitted before compaction" in result
    assert result.startswith("HEAD_MARKER")
    assert result.endswith("TAIL_MARKER")
