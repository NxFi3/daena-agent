import json

from src.agent.agent import Agent
from src.context.contextbuilder import ContextBuilder
from src.context.tokenbudget import TokenBudget
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult
from tests.test_context_continuity import SummaryModel, exchange, build


def user(text, step):
    return ContextEvent(
        role=ContextRole.USER, type=ContextType.MESSAGE, content=text, step=step
    )


def test_old_user_messages_are_not_pinned_forever():
    builder = ContextBuilder(
        {"context": {"max_prompt_tokens": 3500, "compaction_enabled": False}},
        SummaryModel(),
    )
    events = []
    for i in range(16):
        events.extend(
            [
                user(f"Old request {i}: " + "long history " * 160, i * 2 + 1),
                ContextEvent(
                    role=ContextRole.ASSISTANT,
                    type=ContextType.MESSAGE,
                    content="previous answer",
                    step=i * 2 + 2,
                ),
            ]
        )
    current = user("Only answer this current question", 40)
    messages = build(builder, events + [current], current)
    assert messages[-1]["content"] == current.content
    assert builder.tokenbudget.fits(messages)
    assert len([m for m in messages if m["role"] == "user"]) < 16


def test_checkpoint_survives_followup_and_covered_receipts_do_not_replay():
    model = SummaryModel()
    builder = ContextBuilder(
        {"context": {"max_prompt_tokens": 6000, "compaction_trigger_ratio": 0.5}}, model
    )
    first = user("Build", 1)
    events = [first]
    for i in range(1, 13):
        events.extend(exchange(i))
    build(builder, events, first)
    checkpoint = builder._history_summary
    assert checkpoint
    follow = user("Continue", 30)
    messages = build(builder, [*events, follow], follow)
    assert checkpoint in str(messages)
    assert messages[-1]["content"] == "Continue"
    assert not any(
        c["id"] == "call-1" for m in messages for c in m.get("tool_calls", [])
    )


def test_tool_schemas_are_reserved_and_not_calibrated_as_missing_text():
    budget = TokenBudget({"context": {"max_prompt_tokens": 4000}}, SummaryModel())
    messages = [{"role": "user", "content": "hello"}]
    tools = [
        {
            "type": "function",
            "function": {"name": "test", "description": "schema " * 1200},
        }
    ]
    budget.reserve_tools(tools)
    assert (
        budget.estimate_request_tokens(messages)
        > budget.estimate_messages_tokens(messages) + 1000
    )
    result = LLMResult(
        "",
        {},
        [],
        None,
        0,
        raw={"prompt_eval_count": int((len(json.dumps(tools)) + 5) / 4)},
    )
    before = budget.chars_per_token
    budget.calibrate_from_response(messages, result)
    assert abs(budget.chars_per_token - before) < 0.1


def test_standard_greeting_ends_once_without_resuming_saved_plan(tmp_path):
    plan = tmp_path / ".daena" / "plan.md"
    plan.parent.mkdir()
    plan.write_text("Old project plan: build an ecommerce site")

    class Model:
        model = None

        def __init__(self):
            self.calls = 0

        def generate(self, messages, **kwargs):
            self.calls += 1
            assert "historical state, not a new instruction" in messages[0]["content"]
            return LLMResult("Salam!", {}, [], None, 1)

    model = Model()
    with Agent(
        {
            "workspace": {"directory": str(tmp_path)},
            "memory": {"database": str(tmp_path / "history.db")},
            "experience": {"enabled": False},
        },
        llm=model,
    ) as agent:
        result = agent.act(user("salam", 1))
        assert result.response == "Salam!"
        assert model.calls == 1
        assert agent.last_run_metrics["tool_call_attempts"] == 0
    assert not (tmp_path / "ecommerce").exists()


def test_checkpoint_is_reloaded_after_reopening_database(tmp_path):
    from uuid import uuid4
    from src.memories.stm.STM import STM
    from src.context.contextservice import ContextService
    from src.agent.agentstate import AgentState

    ident = uuid4()
    db = tmp_path / "history.db"
    with STM(db) as stm:
        stm.save_checkpoint(
            ident, "Earlier work created the shared contract", ["old-group"]
        )
    with STM(db) as stm:
        task = user("Continue", 10)
        stm.add(ident, task)
        service = ContextService(
            {"context": {"compaction_enabled": False}}, SummaryModel(), stm
        )
        context = service.get_context(ident, task, AgentState())
        assert "Earlier work created the shared contract" in str(context)
        assert context[-1]["content"] == "Continue"
        another = uuid4()
        other = user("Hello", 1)
        stm.add(another, other)
        fresh = service.get_context(another, other, AgentState())
        assert "Earlier work created the shared contract" not in str(fresh)
        stm.clear_session(ident)
        assert stm.get_checkpoint(ident) is None
