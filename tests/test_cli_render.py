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
