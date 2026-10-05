from __future__ import annotations

import json
from typing import Any, Callable, ClassVar

import ollama

from src.engine.providers.ProviderBase import ProviderBase
from src.models.LLMInput import LLMInput
from src.models.LLMResult import LLMResult
from src.utils.logger import get_logger

logger = get_logger("[OLLAMA]")


class OllamaProvider(ProviderBase):

    name = "ollama"

    defaultModel = "gpt-oss:20b"

    defaultConfig: ClassVar[dict] = {
        "temperature": 0.3,
        "num_ctx": 120000,
    }

    # `think` was never being sent to ollama.chat(), so every "thinking"
    # model (gpt-oss, qwen3, gemma3, deepseek-r1...) fell back to its own
    # default reasoning behavior, and that default differs per model AND
    # per Ollama version. Observed effects of leaving it unset:
    #   - gpt-oss: a turn can end having only reasoned, with BOTH
    #     message.content and message.tool_calls empty (the harmony
    #     "final"/tool-call channel is never reached).
    #   - qwen3: a tool-call attempt can drift into the reasoning text
    #     instead of structured tool_calls, which then fails to parse.
    # Explicitly requesting `think` is the documented fix: Ollama then
    # separates reasoning into message.thinking and leaves
    # message.content / message.tool_calls as the model's actual answer.
    # Override per model via generation_config: {"think": false} (or
    # "low"/"medium"/"high" for models with graded effort) in config.json.
    DEFAULT_THINK: ClassVar[bool] = True

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

    def _generate_streaming(
        self,
        *,
        model_name: str,
        messages: list[dict[str, Any]],
        tools: list,
        think: Any,
        options: dict[str, Any],
        callback: Callable[[dict[str, Any]], None],
    ) -> LLMResult:
        """Stream Ollama reasoning/content while preserving one final LLMResult."""
        thinking_parts: list[str] = []
        response_parts: list[str] = []
        tool_calls: list[Any] = []
        final_chunk = None

        def consume(stream):
            nonlocal final_chunk, tool_calls
            for chunk in stream:
                final_chunk = chunk
                message = getattr(chunk, "message", None)
                if message is None:
                    continue

                thinking_delta = str(getattr(message, "thinking", None) or "")
                if thinking_delta:
                    thinking_parts.append(thinking_delta)
                    self._emit_stream_event(
                        callback,
                        "thinking_delta",
                        text=thinking_delta,
                    )

                content_delta = str(getattr(message, "content", None) or "")
                if content_delta:
                    response_parts.append(content_delta)
                    self._emit_stream_event(
                        callback,
                        "content_delta",
                        text=content_delta,
                    )

                raw_tool_calls = getattr(message, "tool_calls", None) or []
                if raw_tool_calls:
                    # Ollama normally emits tool calls as complete structures;
                    # retain the latest non-empty list rather than duplicating
                    # calls across streaming chunks.
                    tool_calls = list(raw_tool_calls)

        try:
            consume(
                ollama.chat(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    think=think,
                    options=options,
                    stream=True,
                )
            )
        except Exception as exc:
            if tools and think is not False and "error parsing tool call" in str(exc).lower():
                logger.warning(
                    "Ollama rejected a streamed tool call; retrying once with think=False."
                )
                thinking_parts.clear()
                response_parts.clear()
                tool_calls.clear()
                final_chunk = None
                try:
                    consume(
                        ollama.chat(
                            model=model_name,
                            messages=messages,
                            tools=tools,
                            think=False,
                            options=options,
                            stream=True,
                        )
                    )
                except Exception as retry_exc:
                    raise RuntimeError(
                        "Ollama streaming failed: "
                        f"{type(retry_exc).__name__}: {retry_exc}"
                    ) from retry_exc
            else:
                raise RuntimeError(
                    "Ollama streaming failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        if final_chunk is None:
            raise RuntimeError("Ollama returned an empty streaming response.")

        message = getattr(final_chunk, "message", None)
        message_data = message.model_dump() if message and hasattr(message, "model_dump") else {}
        response_text = "".join(response_parts)
        thinking_text = "".join(thinking_parts) or None
        message_data["content"] = response_text
        if thinking_text:
            message_data["thinking"] = thinking_text

        serialized_tool_calls: list[dict[str, Any]] = []
        for call in tool_calls:
            if hasattr(call, "model_dump"):
                serialized_tool_calls.append(call.model_dump())
            elif isinstance(call, dict):
                serialized_tool_calls.append(dict(call))
        if serialized_tool_calls:
            message_data["tool_calls"] = serialized_tool_calls

        prompt_tokens = int(getattr(final_chunk, "prompt_eval_count", 0) or 0)
        completion_tokens = int(getattr(final_chunk, "eval_count", 0) or 0)
        usage = prompt_tokens + completion_tokens

        self._emit_stream_event(
            callback,
            "generation_done",
            thinking_tokens=len(thinking_text.split()) if thinking_text else 0,
            response_chars=len(response_text),
            tool_calls=len(tool_calls),
            usage=usage,
        )

        return LLMResult(
            response=response_text,
            message=message_data,
            tool_calls=tool_calls,
            thinking=thinking_text,
            usage=usage,
            raw=final_chunk,
        )

    def generate(
        self,
        inputs: LLMInput,
    ) -> LLMResult:

        model_name = inputs.model_name or self.defaultModel

        messages = self._prepare_messages(inputs.messages or [])

        stream_callback = getattr(inputs, "stream_callback", None)

        tools = inputs.tools or []

        options = dict(self.defaultConfig)

        if inputs.options:
            options.update(inputs.options)

        think = self._resolve_think(options)

        if callable(stream_callback):
            return self._generate_streaming(
                model_name=model_name,
                messages=messages,
                tools=tools,
                think=think,
                options=options,
                callback=stream_callback,
            )

        try:

            chat_response = ollama.chat(
                model=model_name,
                messages=messages,
                tools=tools,
                think=think,
                options=options,
            )

        except Exception as exc:
            error_text = str(exc)

            # Some gpt-oss/Ollama combinations can emit a malformed tool-call
            # payload containing prose before the JSON tool call. Ollama then
            # rejects its own output with HTTP 500. Retry once without
            # reasoning before surfacing the provider failure.
            if (
                tools
                and think is not False
                and "error parsing tool call" in error_text.lower()
            ):
                logger.warning(
                    "Ollama rejected a malformed tool call; retrying once with think=False."
                )
                try:
                    chat_response = ollama.chat(
                        model=model_name,
                        messages=messages,
                        tools=tools,
                        think=False,
                        options=options,
                    )
                except Exception as retry_exc:
                    logger.error(
                        "Tool-call recovery failed: "
                        f"{type(retry_exc).__name__}: {retry_exc}"
                    )
                    raise RuntimeError(
                        "Ollama generation failed: "
                        f"{type(retry_exc).__name__}: {retry_exc}"
                    ) from retry_exc
            else:
                logger.error(
                    "Chat generation failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                raise RuntimeError(
                    "Ollama generation failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        message = chat_response.message if chat_response else None

        prompt_tokens = (
            getattr(
                chat_response,
                "prompt_eval_count",
                0,
            )
            or 0
        )

        completion_tokens = (
            getattr(
                chat_response,
                "eval_count",
                0,
            )
            or 0
        )

        total_tokens = prompt_tokens + completion_tokens

        tool_calls = []

        if message:

            raw_tool_calls = (
                getattr(
                    message,
                    "tool_calls",
                    None,
                )
                or []
            )

            tool_calls = list(raw_tool_calls)

        normalized_message = message.model_dump() if message else {}

        thinking = (
            getattr(
                message,
                "thinking",
                None,
            )
            if message
            else None
        )

        response_text = message.content if message else ""

        if not response_text and not tool_calls and thinking:

            # Not fabricating a response here — Loop is responsible for
            # deciding what to do about an empty turn (see agentloop.py's
            # nudge-and-retry handling). This log line exists so a
            # thinking-only turn is distinguishable from a truly broken
            # one when reading logs.
            logger.warning(
                f"Model '{model_name}' produced only reasoning this turn "
                "(content and tool_calls are both empty, thinking is not)."
            )

        return LLMResult(
            response=response_text,
            message=normalized_message,
            tool_calls=tool_calls,
            thinking=thinking,
            usage=total_tokens,
            raw=chat_response,
        )
