import json

from src.engine.providers.builtin.smollm3.smollm3 import SmolLM3Provider


def test_tool_schema_accepts_openai_format():
    tool = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
    assert SmolLM3Provider._tool_schema(tool) == {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    }


def test_prompt_includes_smollm3_xml_tool_instructions():
    prompt = SmolLM3Provider._render_prompt(
        [{"role": "user", "content": "Read hello.txt"}],
        [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }],
    )
    assert "<tools>" in prompt
    assert '"name":"read_file"' in prompt
    assert "<tool_call>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n")


def test_parse_xml_tool_call_to_daena_canonical_shape():
    content, calls, thinking = SmolLM3Provider._parse_response(
        'Let me inspect it. <tool_call>{"name":"read_file","arguments":{"path":"hello.txt"}}</tool_call>'
    )
    assert content == "Let me inspect it."
    assert thinking is None
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "hello.txt"}


def test_parse_reasoning_and_tool_call():
    content, calls, thinking = SmolLM3Provider._parse_response(
        '<think>Need inspect the file.</think><tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>'
    )
    assert content == ""
    assert thinking == "Need inspect the file."
    assert len(calls) == 1
