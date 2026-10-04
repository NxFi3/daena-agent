import json

from src.context.contextbuilder import ContextBuilder


def _read_message(call_id, path, start, end, text):
    header = {
        "success": True,
        "type": "file",
        "path": path,
        "start_line": start,
        "end_line": end,
    }
    return {
        "role": "tool",
        "tool_name": "read_file",
        "tool_call_id": call_id,
        "content": json.dumps(header) + "\n\n[content]\n" + text,
    }


def _command_message(call_id, text):
    return {
        "role": "tool",
        "tool_name": "command_exec",
        "tool_call_id": call_id,
        "content": json.dumps({"success": True}) + "\n\n[stdout]\n" + text,
    }


def test_old_file_read_is_pinned_while_other_old_results_are_shrunk():
    builder = ContextBuilder.__new__(ContextBuilder)

    body = "x" * 2000
    messages = [_read_message("r1", "/ws/app.py", 1, 100, body)]
    messages += [_command_message(f"c{i}", body) for i in range(9)]

    original_read = messages[0]["content"]

    builder._shrink_old_tool_results(messages)

    # The read is older than the 6-result window but is pinned, so it stays.
    assert messages[0]["content"] == original_read
    # Old non-read results are still cut down.
    assert len(messages[1]["content"]) < 500
    assert "old result truncated" in messages[1]["content"]
    # The most recent results stay in full.
    assert len(messages[-1]["content"]) > 1500


def test_only_the_latest_identical_read_is_pinned():
    builder = ContextBuilder.__new__(ContextBuilder)

    body = "y" * 2000
    messages = [
        _read_message("r1", "/ws/app.py", 1, 100, body),
        _read_message("r2", "/ws/app.py", 1, 100, body),
    ]
    messages += [_command_message(f"c{i}", body) for i in range(8)]

    builder._shrink_old_tool_results(messages)

    # r1 and r2 are both outside the recent window and have the same range;
    # only the newer one (r2) is kept in full.
    assert len(messages[0]["content"]) < 500
    assert len(messages[1]["content"]) > 1500


def test_head_tail_truncation_keeps_both_ends():
    text = "START" + ("m" * 5000) + "END"

    result = ContextBuilder._head_tail(text, 400)

    assert len(result) <= 400
    assert result.startswith("START")
    assert result.endswith("END")
    assert "characters omitted" in result


def test_execution_state_stays_valid_json_when_over_budget():
    builder = ContextBuilder.__new__(ContextBuilder)

    working = {
        "last_failed_verification": {
            "command": ["pytest", "-q"],
            "success": False,
            "exit_code": 1,
            "output_excerpt": "Expected: 200",
        },
        "workspace_inventory": [
            {"path": f"dir/file_{i}.py", "type": "file", "size": 100}
            for i in range(80)
        ],
        "artifacts": {
            f"file-{i}.js": {
                "status": "known",
                "known": True,
                "preview": "x" * 600,
            }
            for i in range(6)
        },
        "facts": ["f" * 900 for _ in range(6)],
    }

    text = builder._compact_execution_state(
        agent_state=None,
        progress=None,
        working_set=working,
        observation=None,
        recent_actions=None,
    )

    parsed = json.loads(text)
    assert len(text) <= builder.MAX_EXECUTION_STATE_CHARS
    assert "last_failed_verification" in parsed
    assert "omitted_for_space" in parsed
