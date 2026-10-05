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
