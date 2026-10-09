import base64
import json
from types import SimpleNamespace

from google.genai import types

from src.agent.agentloop import Loop
from src.engine.providers.builtin.gemini.geminiprovider import GeminiProvider
from src.models.LLMResult import LLMResult
from src.models.ToolCall import ToolCall


def _tool_message(call_id: str, name: str, args: dict, signature: bytes) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
        "thought_signature_b64": base64.b64encode(signature).decode("ascii"),
    }


def test_gemini_function_response_turn_uses_user_role_and_signature_round_trips():
    signature = b"opaque-gemini-thought-signature"
    messages = [
        {"role": "user", "content": "Inspect a file."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_message("call-1", "read_file", {"path": "x.py"}, signature)],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "read_file",
            "content": json.dumps({"success": True, "content": "ok"}),
        },
    ]

    contents, system_instruction = GeminiProvider._convert_messages(messages)

    assert system_instruction is None
    assert [content.role for content in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].function_call.name == "read_file"
    assert contents[1].parts[0].thought_signature == signature
    response_part = contents[2].parts[0]
    assert response_part.function_response.name == "read_file"
    assert response_part.function_response.id == "call-1"
    assert response_part.function_response.response == {"success": True, "content": "ok"}


def test_gemini_groups_parallel_function_responses_into_one_user_turn():
    messages = [
        {"role": "user", "content": "Inspect two files."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_message("call-a", "read_file", {"path": "a.py"}, b"sig-a"),
                _tool_message("call-b", "read_file", {"path": "b.py"}, b"sig-b"),
            ],
        },
        {"role": "tool", "tool_call_id": "call-a", "name": "read_file", "content": '{"success": true}'},
        {"role": "tool", "tool_call_id": "call-b", "name": "read_file", "content": '{"success": true}'},
    ]

    contents, _ = GeminiProvider._convert_messages(messages)

    assert [content.role for content in contents] == ["user", "model", "user"]
    assert len(contents[-1].parts) == 2
    assert [part.function_response.id for part in contents[-1].parts] == ["call-a", "call-b"]


def test_gemini_normalizes_thought_signature_from_model_function_call():
    signature = b"sig-from-response"
    part = types.Part(
        function_call=types.FunctionCall(
            id="call-7",
            name="list_dir",
            args={"path": "."},
        ),
        thought_signature=signature,
    )
    response = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))]
    )

    _, message, calls, _ = GeminiProvider._normalize_response(response)

    expected = base64.b64encode(signature).decode("ascii")
    assert calls[0]["thought_signature_b64"] == expected
    assert message["tool_calls"][0]["thought_signature_b64"] == expected


def test_agent_loop_preserves_gemini_signature_when_normalizing_tool_call():
    signature = base64.b64encode(b"opaque-signature").decode("ascii")
    llmresult = LLMResult(
        response="",
        message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_message("call-1", "read_file", {"path": "x.py"}, b"opaque-signature")
            ],
        },
        tool_calls=[],
        thinking=None,
        usage=20,
    )
    loop = Loop.__new__(Loop)
    loop._context_step = 0
    canonical_call = ToolCall(
        id="call-1",
        name="read_file",
        args={"path": "x.py"},
    )

    event = loop._assistant_event(llmresult, normalized_tool_calls=[canonical_call])

    stored_calls = event.metadata["llm_message"]["tool_calls"]
    assert stored_calls[0]["thought_signature_b64"] == signature


def test_gemini_repairs_daena_history_with_orphaned_function_call():
    signature = b"stored-signature"
    messages = [
        {"role": "system", "content": "System instruction"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_message("call-9", "list_dir", {"path": "."}, signature)
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-9",
            "tool_name": "list_dir",
            "content": '{"success": true, "summary": "Listed entries"}',
        },
        {"role": "user", "content": "Your last request failed; recover carefully."},
        {"role": "user", "content": "<runtime_state>Latest status</runtime_state>"},
        {"role": "user", "content": "Continue the original task."},
    ]

    contents, system_instruction = GeminiProvider._convert_messages(messages)

    assert system_instruction == "System instruction"
    assert [item.role for item in contents] == ["user", "model", "user"]
    assert [part.text for part in contents[0].parts] == [
        "Your last request failed; recover carefully.",
        "<runtime_state>Latest status</runtime_state>",
        "Continue the original task.",
    ]
    assert contents[1].parts[0].function_call.name == "list_dir"
    assert contents[1].parts[0].thought_signature == signature
    assert len(contents[2].parts) == 1
    assert contents[2].parts[0].function_response.name == "list_dir"
    assert contents[2].parts[0].function_response.id == "call-9"
