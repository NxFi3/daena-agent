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
