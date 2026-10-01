from types import SimpleNamespace

from src.engine.providers.builtin.gemini.geminiprovider import GeminiProvider


def test_gemini_tool_results_use_tool_name_and_single_user_content():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{}",
                    },
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{}",
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_name": "read_file",
            "tool_call_id": "call-1",
            "content": '{"ok": true}',
        },
        {
            "role": "tool",
            "tool_name": "read_file",
            "tool_call_id": "call-2",
            "content": '{"ok": true}',
        },
    ]

    contents, _ = GeminiProvider._convert_messages(messages)

    assert len(contents) == 2
    assert contents[0].role == "model"
    assert contents[1].role == "user"
    assert len(contents[1].parts) == 2
    assert contents[1].parts[0].function_response.name == "read_file"
    assert contents[1].parts[1].function_response.name == "read_file"


def test_gemini_response_preserves_function_call_thought_signature():
    signature = b"signature-a"
    function_call = SimpleNamespace(
        id="call-1",
        name="read_file",
        args={"path": "x.txt"},
    )
    part = SimpleNamespace(
        text=None,
        thought=False,
        function_call=function_call,
        thought_signature=signature,
    )
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[part]),
        finish_reason="STOP",
        finish_message=None,
    )
    response = SimpleNamespace(
        candidates=[candidate],
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=4,
            total_token_count=14,
        ),
    )

    _, message, calls, _ = GeminiProvider._normalize_response(response)

    assert message["finish_reason"] == "STOP"
    assert message["gemini_parts"][0]["type"] == "function_call"
    assert message["gemini_parts"][0]["thought_signature"]
    assert calls[0]["thought_signature"] == message["gemini_parts"][0]["thought_signature"]


def test_gemini_serialized_signature_round_trips():
    signature = "c2lnbmF0dXJlLWE="
    messages = [
        {
            "role": "assistant",
            "content": "",
            "gemini_parts": [
                {
                    "type": "function_call",
                    "id": "call-1",
                    "name": "read_file",
                    "args": {"path": "x.txt"},
                    "thought_signature": signature,
                }
            ],
        }
    ]

    contents, _ = GeminiProvider._convert_messages(messages)

    assert len(contents) == 1
    part = contents[0].parts[0]
    assert part.function_call.name == "read_file"
    assert part.thought_signature == b"signature-a"
