from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any, Callable, ClassVar

import httpx
import ollama

from src.engine.providers.ProviderBase import ProviderBase
from src.models.LLMInput import LLMInput
from src.models.LLMResult import LLMResult
from src.utils.logger import get_logger

logger = get_logger("[OLLAMA]")


class OllamaProvider(ProviderBase):
    """Synchronous facade over cancellable, per-request asynchronous HTTP I/O."""

    name = "ollama"
    defaultModel = "qwen3.6:35b-a3b-q4_K_M"
    defaultConfig: ClassVar[dict] = {"temperature": 0.3, "num_ctx": 32768}
    DEFAULT_THINK: ClassVar[Any] = None

    @staticmethod
    def _prepare_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Convert canonical/OpenAI-style messages into the format expected
        by the Ollama Python SDK.

        Internal canonical format:
            function.arguments -> JSON string

        Ollama SDK format:
            function.arguments -> dict

        This conversion belongs here at the provider boundary so the rest
        of the runtime remains provider-neutral.
        """

        prepared: list[dict[str, Any]] = []

        for message in messages:
            if not isinstance(
                message,
                dict,
            ):
                continue

            item = dict(message)

            tool_calls = item.get("tool_calls")

            if isinstance(
                tool_calls,
                list,
            ):
                prepared_tool_calls: list[dict[str, Any]] = []

                for tool_call in tool_calls:
                    if not isinstance(
                        tool_call,
                        dict,
                    ):
                        continue

                    call = dict(tool_call)

                    function = call.get("function")

                    if isinstance(
                        function,
                        dict,
                    ):
                        function = dict(function)

                        arguments = function.get(
                            "arguments",
                            {},
                        )

                        if isinstance(
                            arguments,
                            str,
                        ):
                            try:
                                parsed_arguments = json.loads(arguments)

                            except (
                                json.JSONDecodeError,
                                TypeError,
                                ValueError,
                            ):
                                logger.warning(
                                    "Could not parse Ollama "
                                    "tool-call arguments as JSON. "
                                    f"Using empty arguments. "
                                    f"tool={function.get('name', '')}"
                                )

                                parsed_arguments = {}

                            arguments = parsed_arguments

                        if not isinstance(
                            arguments,
                            dict,
                        ):
                            logger.warning(
                                "Ollama tool-call arguments were "
                                "not a dictionary. "
                                f"tool={function.get('name', '')}"
                            )

                            arguments = {}

                        function["arguments"] = arguments

                        call["function"] = function

                    prepared_tool_calls.append(call)

                item["tool_calls"] = prepared_tool_calls

            prepared.append(item)

        return prepared

    @classmethod
    def _resolve_think(
        cls,
        options: dict[str, Any],
    ) -> Any:
        """
        `think` is not a real Ollama chat "option" — it is a top-level
        ollama.chat() parameter — so it must be popped out of the merged
        options dict rather than left inside it. Falls back to
        DEFAULT_THINK when nothing set one explicitly.
        """

        if "think" in options:
            return options.pop("think")

        return cls.DEFAULT_THINK

    @staticmethod
    def _emit_stream_event(
        callback: Callable[[dict[str, Any]], None] | None,
        event_type: str,
        **payload: Any,
    ) -> None:
        if not callable(callback):
            return
        try:
            callback({"type": event_type, **payload})
        except Exception as exc:
            # UI/telemetry callbacks must never break agent execution.
            logger.debug(f"Stream callback failed: {type(exc).__name__}: {exc}")

    def generate(self, inputs: LLMInput) -> LLMResult:
        if inputs.stop_event is not None and inputs.stop_event.is_set():
            raise InterruptedError("Generation cancelled")
        return asyncio.run(self._request(inputs))

    async def _request(self, inputs: LLMInput) -> LLMResult:
        options = {**self.defaultConfig, **(inputs.options or {})}
        think = self._resolve_think(options)

        def seconds(key, default, optional=False):
            value = options.pop(key, default)
            if optional and value is None:
                return None
            if isinstance(value, bool):
                raise ValueError(
                    f"{key} must be positive seconds" + (" or null" if optional else "")
                )
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{key} must be finite positive seconds"
                    + (" or null" if optional else "")
                )
            return value

        # Local inference can buffer a large tool call for minutes. Null read
        # timeout waits without treating model latency as a transport failure;
        # cancellation and an optional overall deadline remain independent.
        read_timeout = seconds("request_timeout_seconds", None, optional=True)
        connect_timeout = seconds("connect_timeout_seconds", 10)
        total_timeout = seconds("total_timeout_seconds", None, optional=True)
        host = options.pop("base_url", None)
        client = ollama.AsyncClient(
            host=host,
            timeout=httpx.Timeout(
                connect=connect_timeout, read=read_timeout, write=30, pool=10
            )
        )
        started = time.monotonic()
        progress = {"last_chunk": started, "chunks": 0, "phase": "waiting for response"}
        callback = inputs.stream_callback
        model = inputs.model_name or self.defaultModel

        async def consume():
            self._emit_stream_event(
                callback, "generation_start", model=model, think=think, streaming=True
            )
            stream = await client.chat(
                model=model,
                messages=self._prepare_messages(inputs.messages),
                tools=inputs.tools or [],
                think=think,
                options=options,
                stream=True,
            )
            content, thinking, calls = [], [], []
            final = None
            chunks = 0
            try:
                async for chunk in stream:
                    final = chunk
                    chunks += 1
                    progress.update(
                        last_chunk=time.monotonic(), chunks=chunks, phase="streaming"
                    )
                    message = chunk.message
                    if message is None:
                        continue
                    if message.thinking:
                        thinking.append(message.thinking)
                        self._emit_stream_event(
                            callback, "thinking_delta", text=message.thinking
                        )
                    if message.content:
                        content.append(message.content)
                        self._emit_stream_event(
                            callback, "content_delta", text=message.content
                        )
                    calls.extend(message.tool_calls or [])
            finally:
                await stream.aclose()
            if final is None or not final.done:
                raise RuntimeError(
                    "Ollama stream ended without a completion marker; partial tool calls were not executed."
                )
            if getattr(final, "done_reason", None) == "length":
                raise RuntimeError(
                    "Ollama reached its output-token limit; partial tool calls were not executed. Increase num_predict or request smaller edits."
                )
            serialized = [
                call.model_dump() if hasattr(call, "model_dump") else dict(call)
                for call in calls
            ]
            message = {"role": "assistant", "content": "".join(content)}
            if serialized:
                message["tool_calls"] = serialized
            reasoning = "".join(thinking) or None
            usage = int(final.prompt_eval_count or 0) + int(final.eval_count or 0)
            self._emit_stream_event(
                callback, "generation_done", usage=usage, stream_chunks=chunks
            )
            return LLMResult(
                message["content"], message, serialized, reasoning, usage, raw=final
            )

        request = asyncio.create_task(consume())
        next_heartbeat = started + 2
        try:
            # This check runs even while connect(), response headers or the next
            # stream chunk are pending. Cancelling the task closes the HTTP stream.
            while not request.done():
                if inputs.stop_event is not None and inputs.stop_event.is_set():
                    self._emit_stream_event(callback, "generation_stopped")
                    raise InterruptedError("Generation cancelled")
                now = time.monotonic()
                if total_timeout is not None and now - started >= total_timeout:
                    raise TimeoutError(
                        f"Ollama total_timeout_seconds={total_timeout:g} elapsed for model {model}; partial tool calls were not executed."
                    )
                if now >= next_heartbeat:
                    self._emit_stream_event(
                        callback,
                        "generation_wait",
                        elapsed_seconds=round(now - started, 1),
                        idle_seconds=round(now - progress["last_chunk"], 1),
                        stream_chunks=progress["chunks"],
                        phase=progress["phase"],
                    )
                    next_heartbeat = now + 2
                await asyncio.wait({request}, timeout=0.05)
            if inputs.stop_event is not None and inputs.stop_event.is_set():
                raise InterruptedError("Generation cancelled")
            return await request
        except httpx.TimeoutException as exc:
            raise TimeoutError(
                f"Ollama {type(exc).__name__} for model {model} during {progress['phase']} "
                f"after {time.monotonic() - started:.1f}s ({progress['chunks']} chunks). "
                "Partial tool calls were not executed. For slow local inference, set "
                "request_timeout_seconds to null or a larger value; interruption remains available."
            ) from exc
        except (httpx.ConnectError, ConnectionError) as exc:
            raise ConnectionError(
                f"Cannot reach Ollama for model {model}. Check the server and OLLAMA_HOST."
            ) from exc
        finally:
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
            await client.close()
