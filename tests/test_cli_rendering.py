from cli.render import StreamRenderer


def test_stopped_run_does_not_render_done(monkeypatch):
    lines = []
    renderer = StreamRenderer(model="test", workspace=".", session_id="test")
    monkeypatch.setattr(renderer, "_print", lines.append)
    renderer.handle({"type": "run_stopped", "reason": "Interrupted by user"})
    renderer.handle(
        {
            "type": "run_end",
            "completed": False,
            "stop_reason": "Interrupted by user",
            "metrics": {"tokens": 42},
        }
    )
    renderer.finish()
    assert renderer.status == "stopped"
    assert renderer.usage == 42
    assert "✓ done" not in "\n".join(lines)


def test_final_answer_not_hidden_by_previous_iterations_stream(monkeypatch):
    lines = []
    renderer = StreamRenderer(model="test", workspace=".", session_id="test")
    monkeypatch.setattr(renderer, "_print", lines.append)
    renderer.handle({"type": "iteration_start", "iteration": 1})
    renderer.handle({"type": "content_delta", "text": "Inspecting first"})
    renderer.handle({"type": "iteration_start", "iteration": 2})
    renderer.handle({"type": "final_response", "text": "Actual final result"})
    assert "Actual final result" in "\n".join(lines)
