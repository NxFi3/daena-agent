"""Optional memory data stays bounded and cannot evict trusted instructions."""

import pytest

from src.context.contributions import ContextContribution, contribution_messages
from src.context.contextbuilder import ContextBuilder, ContextBudgetError
from tests.test_context import FakeLLM, base_config, event
from src.models.ContextEvent import ContextRole, ContextType


def test_retrieved_data_retains_provenance_and_is_bounded_not_system_instruction():
    record = ContextContribution("memory-42", "memory://session/42", "hello" * 10000)
    messages = contribution_messages([record, record], max_chars=700)
    assert len(messages) == 1 and messages[0]["role"] == "user"
    assert len(messages[0]["content"]) <= 700
    assert '"source": "memory://session/42"' in messages[0]["content"]
    assert '"truncated": true' in messages[0]["content"]
    assert messages[0]["content"].count('"id": "memory-42"') == 1
    assert contribution_messages([record], max_chars=10) == []
    with pytest.raises(TypeError):
        contribution_messages([{"role": "system", "content": "bad"}])


def test_memory_is_per_request_and_never_enters_pending_history():
    builder = ContextBuilder(base_config(), FakeLLM())
    request = event(ContextRole.USER, ContextType.MESSAGE, "Current task", 1)
    args = dict(
        events=[request], task={"id": str(request.id), "content": request.content}
    )
    messages = builder.build_context(
        **args,
        contributions=[ContextContribution("1", "archive:test", "Remembered evidence")],
    )
    assert "Remembered evidence" in str(messages)
    assert "Remembered evidence" not in messages[0]["content"]
    assert "Remembered evidence" not in str(builder._pending_history)
    assert "Remembered evidence" not in str(builder.build_context(**args))


def test_budget_fitting_preserves_entire_system_and_active_mode():
    config = base_config()
    config["context"].update(max_prompt_tokens=1600, compaction_enabled=False)
    builder = ContextBuilder(config, FakeLLM())
    builder.runtime_instruction_source = lambda: "MANDATORY_ACTIVE_MODE"
    request = event(ContextRole.USER, ContextType.MESSAGE, "Current task", 1)
    messages = builder.build_context(
        events=[request],
        task={"id": str(request.id), "content": request.content},
        learned_experience="optional learned data " * 1000,
        contributions=[ContextContribution("1", "memory:x", "retrieved " * 1000)],
    )
    assert builder.window.system_instruction in messages[0]["content"]
    assert "MANDATORY_ACTIVE_MODE" in messages[0]["content"]
    assert builder.tokenbudget.fits(messages)


def test_impossible_budget_raises_instead_of_silently_deleting_policy():
    config = base_config()
    config["context"]["max_prompt_tokens"] = 512
    builder = ContextBuilder(config, FakeLLM())
    with pytest.raises(ContextBudgetError):
        builder.build_context(events=[], task={"id": "x", "content": "Hello"})
