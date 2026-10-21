"""Exercise the real Ollama SDK against a local HTTP server, without a model."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("ollama")
from src.engine.providers.builtin.ollama.ollama import OllamaProvider
from src.models.LLMInput import LLMInput


@pytest.fixture
def server(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            mode = request["model"]
            if mode == "stall-headers":
                entered.set()
                release.wait(5)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            chunks = [
                {
                    "message": {
                        "role": "assistant",
                        "thinking": "reasoning",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "read_file",
                                    "arguments": {"file_path": "a"},
                                }
                            }
                        ],
                    },
                    "done": False,
                },
                {
                    "message": {
                        "role": "assistant",
                        "content": "done",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "read_file",
                                    "arguments": {"file_path": "b"},
                                }
                            }
                        ],
                    },
                    "done": True,
                    "prompt_eval_count": 11,
                    "eval_count": 7,
                },
            ]
            if mode == "agent-cycle":
                exchanges = [
                    {"content": "I will begin now"},
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "plan",
                                    "arguments": {
                                        "action": "create",
                                        "goal": "Build artifact",
                                        "steps": ["Create", "Read"],
                                    },
                                }
                            }
                        ]
                    },
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "make_directory",
                                    "arguments": {"paths": ["src", "tests"]},
                                }
                            }
                        ]
                    },
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "apply_patch",
                                    "arguments": {
                                        "patch": "*** Begin Patch\n*** Add File: src/result.txt\n+HTTP protocol verified\n*** End Patch"
                                    },
                                }
                            }
                        ]
                    },
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "read_file",
                                    "arguments": {"file_path": "src/result.txt"},
                                }
                            }
                        ]
                    },
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "task_finish",
                                    "arguments": {
                                        "outcome": "completed",
                                        "response": "Created and checked artifact",
                                        "evidence": ["Read src/result.txt"],
                                    },
                                }
                            }
                        ]
                    },
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "plan",
                                    "arguments": {"action": "read"},
                                }
                            }
                        ]
                    },
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "task_finish",
                                    "arguments": {
                                        "outcome": "completed",
                                        "response": "Plan retained",
                                        "evidence": ["Read existing plan"],
                                    },
                                }
                            }
                        ]
                    },
                ]
                index = sum(r["model"] == "agent-cycle" for r in requests) - 1
                chunks = [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            **exchanges[index],
                        },
                        "done": True,
                        "prompt_eval_count": 20,
                        "eval_count": 5,
                    }
                ]
            if mode == "incomplete":
                chunks = chunks[:1]
            if mode == "limited":
                chunks[-1]["done_reason"] = "length"
            try:
                for index, chunk in enumerate(chunks):
                    self.wfile.write((json.dumps(chunk) + "\n").encode())
                    self.wfile.flush()
                    if mode == "stall-chunks" and index == 0:
                        entered.set()
                        release.wait(5)
            except (BrokenPipeError, ConnectionResetError):
                pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{http.server_port}")
    try:
        yield entered, release, requests
    finally:
        release.set()
        http.shutdown()
        http.server_close()
        thread.join(2)


@pytest.mark.parametrize("mode", ["stall-headers", "stall-chunks"])
@pytest.mark.parametrize("callback", [None, lambda event: None])
def test_cancel_during_network_wait_with_or_without_ui(server, mode, callback):
    entered, release, requests = server
    stop, finished = threading.Event(), threading.Event()
    errors = []

    def work():
        try:
            OllamaProvider().generate(
                LLMInput(
                    mode,
                    [{"role": "user", "content": "test"}],
                    stop_event=stop,
                    stream_callback=callback,
                )
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    assert entered.wait(3)
    stop.set()
    assert finished.wait(1.5), "Cancellation waited for server response"
    thread.join()
    assert len(errors) == 1 and isinstance(errors[0], InterruptedError)
    assert requests[0]["stream"] is True
    # The server is still blocked: cancellation actually closed local I/O.
    assert not release.is_set()


def test_tools_across_stream_chunks_and_protocol_conversion(server):
    _, _, requests = server
    inputs = LLMInput(
        "normal",
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "abc",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"file_path":"old"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "abc",
                "tool_name": "read_file",
                "content": "old file",
            },
        ],
        options={"think": False},
    )
    events = []
    inputs.stream_callback = events.append
    result = OllamaProvider().generate(inputs)
    assert result.response == "done" and result.thinking == "reasoning"
    assert result.usage == 18
    assert [
        call["function"]["arguments"]["file_path"] for call in result.tool_calls
    ] == ["a", "b"]
    assert requests[0]["messages"][0]["tool_calls"][0]["function"]["arguments"] == {
        "file_path": "old"
    }
    assert requests[0]["think"] is False
    assert "think" not in requests[0]["options"]
    assert events[-1]["type"] == "generation_done"


def test_cancelling_one_request_does_not_cancel_another(server):
    entered, release, requests = server
    stop = threading.Event()
    errors = []

    def blocked():
        try:
            OllamaProvider().generate(LLMInput("stall-headers", [], stop_event=stop))
        except InterruptedError as exc:
            errors.append(exc)

    thread = threading.Thread(target=blocked, daemon=True)
    thread.start()
    assert entered.wait(3)
    assert OllamaProvider().generate(LLMInput("normal", [])).response == "done"
    stop.set()
    thread.join(1.5)
    assert not thread.is_alive() and len(errors) == 1


@pytest.mark.parametrize("mode", ["stall-headers", "stall-chunks"])
def test_explicit_read_timeout_reports_useful_diagnostics(server, mode):
    with pytest.raises(
        TimeoutError, match="Ollama ReadTimeout.*Partial tool calls were not executed"
    ):
        OllamaProvider().generate(
            LLMInput(mode, [], options={"request_timeout_seconds": 0.1})
        )


def test_null_read_timeout_allows_delayed_tool_arguments(server):
    entered, release, requests = server
    results, errors = [], []

    def work():
        try:
            results.append(
                OllamaProvider().generate(
                    LLMInput(
                        "stall-chunks", [], options={"request_timeout_seconds": None}
                    )
                )
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    assert entered.wait(3)
    # Beyond the short timeout used in the regression above, the request is still pending.
    thread.join(0.2)
    assert thread.is_alive()
    release.set()
    thread.join(3)
    assert not thread.is_alive() and not errors
    assert len(results[0].tool_calls) == 2
    assert "request_timeout_seconds" not in requests[0]["options"]


def test_overall_deadline_is_independent_of_read_timeout(server):
    with pytest.raises(TimeoutError, match="total_timeout_seconds"):
        OllamaProvider().generate(
            LLMInput(
                "stall-chunks",
                [],
                options={
                    "request_timeout_seconds": None,
                    "total_timeout_seconds": 0.15,
                },
            )
        )


@pytest.mark.parametrize(
    "mode,match",
    [("incomplete", "completion marker"), ("limited", "output-token limit")],
)
def test_partial_or_truncated_tool_stream_is_not_executable(server, mode, match):
    with pytest.raises(RuntimeError, match=match):
        OllamaProvider().generate(LLMInput(mode, []))


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_invalid_timeout_configuration_is_rejected(value):
    with pytest.raises(ValueError, match="positive seconds"):
        OllamaProvider().generate(
            LLMInput("test", [], options={"request_timeout_seconds": value})
        )


def test_wait_telemetry_remains_cancellable(server):
    entered, release, requests = server
    stop, heartbeat = threading.Event(), threading.Event()
    events, errors = [], []

    def callback(event):
        events.append(event)
        if event["type"] == "generation_wait":
            heartbeat.set()
            stop.set()

    def work():
        try:
            OllamaProvider().generate(
                LLMInput("stall-headers", [], stop_event=stop, stream_callback=callback)
            )
        except InterruptedError as exc:
            errors.append(exc)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    assert entered.wait(3)
    assert heartbeat.wait(3)
    thread.join(1.5)
    assert not thread.is_alive() and len(errors) == 1
    waiting = next(e for e in events if e["type"] == "generation_wait")
    assert waiting["elapsed_seconds"] >= 2 and waiting["stream_chunks"] == 0


def test_real_sdk_provider_loop_files_and_followup(server, tmp_path):
    from src.agent.agent import Agent
    from src.models.ContextEvent import ContextEvent, ContextRole, ContextType

    config = {
        "llm": {"provider": "ollama", "provider_config": {"model_name": "agent-cycle"}},
        "runtime": {"completion_protocol": "explicit"},
        "context": {"compaction_enabled": False},
        "workspace": {"directory": str(tmp_path)},
        "memory": {"database": str(tmp_path / ".daena" / "memory.db")},
    }
    with Agent(config) as agent:
        first = agent.act(
            ContextEvent(
                role=ContextRole.USER,
                type=ContextType.MESSAGE,
                content="Create and inspect an artifact",
            )
        )
        assert first.response == "Created and checked artifact"
        assert (
            tmp_path / "src/result.txt"
        ).read_text().strip() == "HTTP protocol verified"
        assert (tmp_path / "tests").is_dir()
        assert agent.last_run_metrics["iterations"] == 6
        second = agent.act(
            ContextEvent(
                role=ContextRole.USER,
                type=ContextType.MESSAGE,
                content="Show the existing plan",
            )
        )
        assert second.response == "Plan retained"
    followup = server[2][6]["messages"]
    assert "Build artifact" in followup[0]["content"]
    assert any(
        m["role"] == "tool" and "HTTP protocol verified" in m["content"]
        for m in followup
    )
    assert followup[-1]["content"] == "Show the existing plan"
