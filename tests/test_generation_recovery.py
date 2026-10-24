"""Regression coverage for safe provider-generation recovery."""

from copy import deepcopy

from src.agent.agent import Agent
from src.context.tokenbudget import TokenBudget
from src.engine.errors import OutputTokenLimitError
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult
from src.studio.state import default_config


class FlakyGeneration:
    model = None

    def __init__(self, failures=1):
        self.failures = failures
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise OutputTokenLimitError(
                "output truncated",
                recovery_hint="Retry with smaller tool calls.",
            )
        return LLMResult("Recovered successfully.", {}, [], None, 1)


def _config(tmp_path, retries=2):
    config = deepcopy(default_config())
    config["workspace"] = {"directory": str(tmp_path)}
    config["memory"] = {"database": str(tmp_path / "history.sqlite3")}
    config["runtime"]["generation_retries"] = retries
    return config


def test_output_limit_retries_without_executing_partial_generation(tmp_path):
    llm = FlakyGeneration(failures=1)
    events = []
    agent = Agent(_config(tmp_path), llm=llm)
    try:
        result = agent.act(
            ContextEvent(ContextRole.USER, ContextType.MESSAGE, "Finish the task"),
            on_event=events.append,
        )
    finally:
        agent.close()

    assert result.response == "Recovered successfully."
    assert llm.calls == 2
    assert agent.last_run_metrics["completed"] is True
    assert agent.last_run_metrics["generation_failures"] == 1
    assert agent.last_run_metrics["generation_retries"] == 1
    retry = next(event for event in events if event["type"] == "generation_retry")
    assert retry["attempt"] == 1
    assert "smaller tool calls" in retry["recovery_hint"]
    assert not any(event["type"] == "tool_call" for event in events)


def test_output_limit_stops_after_bounded_retries(tmp_path):
    llm = FlakyGeneration(failures=99)
    agent = Agent(_config(tmp_path, retries=2), llm=llm)
    try:
        result = agent.act(
            ContextEvent(ContextRole.USER, ContextType.MESSAGE, "Finish the task")
        )
    finally:
        agent.close()

    assert llm.calls == 3
    assert result.response.startswith("Task stopped:")
    assert agent.last_run_metrics["completed"] is False
    assert agent.last_run_metrics["generation_failures"] == 3
    assert agent.last_run_metrics["generation_retries"] == 2
    assert "Recovery limit (2) exhausted" in agent.last_run_metrics["stop_reason"]


def test_token_budget_reserves_provider_output_cap():
    class Model:
        defaultConfig = {"num_ctx": 32768, "num_predict": 8192}

    class Provider:
        model = Model()

    config = {
        "llm": {
            "provider_config": {
                "generation_config": {"num_ctx": 32768, "num_predict": 8192}
            }
        },
        "context": {
            "safe_margin": 0,
            "output_reserve_tokens": 4096,
            "max_prompt_tokens": 30000,
        },
    }
    budget = TokenBudget(config, Provider())
    assert budget.output_reserve_tokens == 8192
    assert budget.budget == 24576
