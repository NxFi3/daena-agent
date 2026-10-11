from cli.render import StreamRenderer


def test_toolbar_escapes_dynamic_status_and_model_text():
    renderer = StreamRenderer(
        model="gpt-oss:20b & <dev>",
        workspace="/tmp/workspace",
        session_id="session",
    )
    renderer.status = "tool:<bad & status>"
    renderer.context_tokens = 2300
    renderer.context_budget = 65536
    renderer.max_iterations = 10

    # Constructing the Prompt Toolkit HTML must not interpret dynamic text as XML.
    renderer.toolbar()

    renderer.status = "result:a > b & c < d"
    renderer.toolbar()


def test_completion_and_recovery_review_events_are_rendered():
    renderer = StreamRenderer(
        model="reviewer", workspace="/tmp", session_id="session"
    )
    output = []
    renderer._print = output.append
    renderer.handle({
        "type": "completion_review", "decision": "continue",
        "reason": "the requested result has not been verified",
        "next_action": "inspect the result file",
    })
    renderer.handle({
        "type": "recovery_review", "decision": "blocked",
        "reason": "all permitted options are exhausted",
        "next_action": "",
    })
    rendered = "\\n".join(output)
    assert "completion review: continue" in rendered
    assert "recovery review: blocked" in rendered
    assert "inspect the result file" in rendered


def test_finish_summary_reports_total_llm_calls_and_review_breakdown():
    renderer = StreamRenderer(
        model="gpt-oss:20b", workspace="/tmp", session_id="session"
    )
    output = []
    renderer._print = output.append
    renderer.total_iterations = 19
    renderer.handle({
        "type": "run_end",
        "completed": True,
        "metrics": {
            "llm_calls": 24,
            "tokens": 17300,
            "completion_review_calls": 1,
            "recovery_review_calls": 4,
        },
    })
    renderer.finish()

    rendered = "\\n".join(output)
    assert "LLM 24" in rendered
    assert "reviews R4/F1" in rendered
    assert "17.3k tok" in rendered
