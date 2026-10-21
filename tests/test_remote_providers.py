import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from src.engine.providers.builtin.openrouter.openrouter import OpenRouterProvider
from src.engine.providers.builtin.gemini.geminiprovider import GeminiProvider, types
from src.engine.LlmProviderManager import LlmProvider
from src.models.LLMInput import LLMInput


@pytest.fixture
def server():
    state = {"events": [], "requests": [], "delay": 0, "entered": threading.Event()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            state["requests"].append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            state["entered"].set()
            time.sleep(state["delay"])
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for event in state["events"]:
                    self.wfile.write(
                        (
                            "data: "
                            + (event if isinstance(event, str) else json.dumps(event))
                            + "\n\n"
                        ).encode()
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield state, f"http://127.0.0.1:{http.server_port}"
    http.shutdown()
    http.server_close()
    thread.join()


def choice(delta=None, finish=None):
    return {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}


def test_openrouter_real_sse_tools_usage_and_wire_options(server, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    state, url = server
    state["events"] = [
        choice(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "abc",
                        "function": {"name": "read_file", "arguments": '{"file_'},
                    }
                ]
            }
        ),
        choice(
            {"tool_calls": [{"index": 0, "function": {"arguments": 'path":"a.txt"}'}}]}
        ),
        choice(finish="tool_calls"),
        {"choices": [], "usage": {"total_tokens": 101, "prompt_tokens": 80}},
        "[DONE]",
    ]
    events = []
    result = OpenRouterProvider().generate(
        LLMInput(
            "test-model",
            [
                {
                    "role": "tool",
                    "tool_name": "read_file",
                    "tool_call_id": "old",
                    "content": "ok",
                }
            ],
            options={
                "base_url": url,
                "num_ctx": 32768,
                "num_thread": 16,
                "request_timeout_seconds": None,
            },
            stream_callback=events.append,
        )
    )
    assert result.tool_calls[0]["function"]["arguments"] == '{"file_path":"a.txt"}'
    assert result.usage == 101 and result.raw["prompt_eval_count"] == 80
    payload = state["requests"][0]
    assert "num_ctx" not in payload and "request_timeout_seconds" not in payload
    assert (
        payload["messages"][0]["name"] == "read_file"
        and "tool_name" not in payload["messages"][0]
    )


@pytest.mark.parametrize(
    "tail", [[choice(finish="length"), "[DONE]"], [choice(finish="tool_calls")], []]
)
def test_openrouter_never_returns_partial_tools(server, monkeypatch, tail):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    state, url = server
    state["events"] = [
        choice(
            {
                "tool_calls": [
                    {"index": 0, "function": {"name": "apply_patch", "arguments": "{"}}
                ]
            }
        ),
        *tail,
    ]
    with pytest.raises(RuntimeError, match="no partial tools"):
        OpenRouterProvider().generate(LLMInput("test", [], options={"base_url": url}))


def test_openrouter_cancel_during_response_headers(server, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    state, url = server
    state["delay"] = 1
    stop = threading.Event()
    errors = []

    def run():
        try:
            OpenRouterProvider().generate(
                LLMInput("test", [], options={"base_url": url}, stop_event=stop)
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert state["entered"].wait(2)
    started = time.monotonic()
    stop.set()
    thread.join(0.8)
    assert not thread.is_alive() and time.monotonic() - started < 0.8
    assert isinstance(errors[0], InterruptedError)


def test_provider_switch_restores_independent_settings():
    provider = LlmProvider(
        {
            "llm": {
                "provider": "ollama",
                "provider_config": {
                    "model_name": "local-model",
                    "base_url": "http://localhost:1234",
                    "generation_config": {"num_ctx": 32768},
                },
            }
        }
    )
    provider.switch_provider("openrouter")
    assert (
        "base_url" not in provider.llm_config
        and "num_ctx" not in provider.generation_config
    )
    provider.switch_provider("ollama")
    assert provider.llm_config["model_name"] == "local-model" and provider.llm_config[
        "base_url"
    ].endswith("1234")


def test_gemini_signed_parts_and_tool_receipt_names_roundtrip():
    if types is None:
        pytest.skip("google-genai not installed")
    signed = types.Part(
        function_call=types.FunctionCall(name="read_file", args={"file_path": "a.txt"}),
        thought_signature=b"opaque-signature",
    )
    raw = signed.model_dump(mode="json", exclude_none=True)
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "canonical-1",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"file_path":"a.txt"}',
                    },
                }
            ],
            "gemini_parts": [raw],
        },
        {
            "role": "tool",
            "tool_name": "read_file",
            "tool_call_id": "canonical-1",
            "content": '{"success":true}',
        },
    ]
    content, _ = GeminiProvider._convert_messages(messages)
    assert content[0].parts[0].model_dump(mode="json", exclude_none=True) == raw
    response = content[1].parts[0].function_response
    assert (
        content[1].role == "user"
        and response.name == "read_file"
        and response.id is None
    )


def test_gemini_async_generation_closes_and_preserves_signatures(monkeypatch):
    if types is None:
        pytest.skip("google-genai not installed")
    from src.engine.providers.builtin.gemini import geminiprovider as module

    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    state = {"async_closed": False, "sync_closed": False}
    signed = types.Part(
        function_call=types.FunctionCall(name="read_file", args={"file_path": "a.txt"}),
        thought_signature=b"signed",
    )

    class Client:
        def __init__(self, **kwargs):
            self.aio = SimpleNamespace(models=self, aclose=self.aclose)

        async def generate_content_stream(self, **kwargs):
            state["config"] = kwargs["config"]

            async def stream():
                yield types.GenerateContentResponse(
                    candidates=[
                        types.Candidate(
                            content=types.Content(role="model", parts=[signed]),
                            finish_reason="STOP",
                        )
                    ]
                )

            return stream()

        async def aclose(self):
            state["async_closed"] = True

        def close(self):
            state["sync_closed"] = True

    monkeypatch.setattr(module.genai, "Client", Client)
    result = GeminiProvider().generate(
        LLMInput(
            "test",
            [{"role": "user", "content": "inspect"}],
            options={
                "num_ctx": 32768,
                "num_thread": 16,
                "request_timeout_seconds": None,
            },
        )
    )
    assert result.tool_calls[0]["function"]["name"] == "read_file"
    assert result.message["gemini_parts"][0] == signed.model_dump(
        mode="json", exclude_none=True
    )
    assert state["async_closed"] and state["sync_closed"]


def test_gemini_real_sdk_http_stream_and_option_mapping(server, monkeypatch):
    if types is None:
        pytest.skip("google-genai not installed")
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    state, url = server
    state["events"] = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "SDK transport works"}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"totalTokenCount": 12, "promptTokenCount": 8},
        }
    ]
    result = GeminiProvider().generate(
        LLMInput(
            "test",
            [{"role": "user", "content": "hello"}],
            options={
                "base_url": url,
                "num_ctx": 32768,
                "num_thread": 16,
                "num_predict": 100,
                "request_timeout_seconds": 10,
            },
        )
    )
    assert result.response == "SDK transport works" and result.usage == 12
    assert state["requests"][0]["generationConfig"]["maxOutputTokens"] == 100
    assert "num_ctx" not in json.dumps(state["requests"][0])


def test_gemini_real_sdk_cancel_before_headers(server, monkeypatch):
    if types is None:
        pytest.skip("google-genai not installed")
    monkeypatch.setenv("GEMINI_API_KEY", "test-only")
    state, url = server
    state["delay"] = 1
    stop = threading.Event()
    errors = []

    def run():
        try:
            GeminiProvider().generate(
                LLMInput(
                    "test",
                    [{"role": "user", "content": "hello"}],
                    options={"base_url": url},
                    stop_event=stop,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert state["entered"].wait(2)
    stop.set()
    thread.join(0.8)
    assert not thread.is_alive() and isinstance(errors[0], InterruptedError)
