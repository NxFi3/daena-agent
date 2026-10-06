import os
from typing import Any, Callable, ClassVar

from openrouter import OpenRouter

from src.engine.providers.ProviderBase import ProviderBase
from src.models.LLMInput import LLMInput
from src.models.LLMResult import LLMResult
from src.utils.logger import get_logger

logger = get_logger("[OPENROUTER]")


class OpenRouterProvider(ProviderBase):

    name = "openrouter"

    defaultModel = "openrouter/free"

    defaultConfig: ClassVar[dict] = {
        "temperature": 0.3,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }

    def __init__(self) -> None:
        self.client: OpenRouter | None = None

    def _create_client(self) -> None:
        if self.client is not None:
            return

        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")

        self.client = OpenRouter(
            api_key=api_key,
        )

    @staticmethod
    def _serialize_tool_call(
        tool_call: Any,
    ) -> dict:
        """
        Convert an OpenRouter SDK tool-call object into
        a plain Python dictionary.
        """

        if hasattr(tool_call, "model_dump"):
            return tool_call.model_dump(exclude_none=True)

        if isinstance(tool_call, dict):
            return dict(tool_call)

        raise TypeError("Unsupported tool call type: " f"{type(tool_call).__name__}")

    @staticmethod
    def _serialize_message(
        message: Any,
    ) -> dict:
        """
        Convert an OpenRouter SDK message object into
        a plain Python dictionary.

        The normalized message is used consistently for:
            - content
            - tool_calls
            - reasoning
            - persistence
        """

        if hasattr(message, "model_dump"):
            return message.model_dump(exclude_none=True)

        if isinstance(message, dict):
            return dict(message)

        raise TypeError("Unsupported message type: " f"{type(message).__name__}")

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
            logger.debug(
                f"Stream callback failed: {type(exc).__name__}: {exc}"
            )

    @staticmethod
    def _reasoning_delta(delta: Any) -> str:
        value = getattr(delta, "reasoning", None)
        if value:
            return str(value)

        details = getattr(delta, "reasoning_details", None)
        if not details:
            return ""

        parts: list[str] = []
        if isinstance(details, list):
            for item in details:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
                elif isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
        elif isinstance(details, dict) and details.get("text"):
            parts.append(str(details["text"]))

        return "".join(parts)

    @classmethod
    def _generate_streaming(
        cls,
        client: OpenRouter,
        *,
        model_name: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
        think: Any,
        callback: Callable[[dict[str, Any]], None],
    ) -> LLMResult:
        thinking_parts: list[str] = []
        content_parts: list[str] = []
        tool_buffers: dict[int, dict[str, Any]] = {}
        usage_total = 0
        last_event: Any = None

        cls._emit_stream_event(
            callback,
            "generation_start",
            model=model_name,
            think=think,
            streaming=True,
        )

        request_kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            **options,
        }
        if tools:
            request_kwargs["tools"] = tools

        try:
            stream = client.chat.send(**request_kwargs)

            for event in stream:
                last_event = event

                usage = getattr(event, "usage", None)
                if usage is not None:
                    total = getattr(usage, "total_tokens", None)
                    if total is not None:
                        try:
                            usage_total = int(total)
                        except (TypeError, ValueError):
                            pass

                choices = getattr(event, "choices", None) or []
                if not choices:
                    continue

                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue

                reasoning_delta = cls._reasoning_delta(delta)
                if reasoning_delta:
                    thinking_parts.append(reasoning_delta)
                    cls._emit_stream_event(
                        callback,
                        "thinking_delta",
                        text=reasoning_delta,
                    )

                content_delta = getattr(delta, "content", None)
                if content_delta:
                    content_delta = str(content_delta)
                    content_parts.append(content_delta)
                    cls._emit_stream_event(
                        callback,
                        "content_delta",
                        text=content_delta,
                    )

                raw_tool_calls = getattr(delta, "tool_calls", None) or []
                for tool_call in raw_tool_calls:
                    index = getattr(tool_call, "index", None)
                    if index is None and isinstance(tool_call, dict):
                        index = tool_call.get("index", 0)
                    try:
                        index = int(index if index is not None else 0)
                    except (TypeError, ValueError):
                        index = 0

                    target = tool_buffers.setdefault(
                        index,
                        {
                            "id": None,
                            "type": "function",
                            "function": {
                                "name": "",
                                "arguments": "",
                            },
                        },
                    )

                    call_id = getattr(tool_call, "id", None)
                    if call_id:
                        target["id"] = str(call_id)

                    function = getattr(tool_call, "function", None)
                    if function is None and isinstance(tool_call, dict):
                        function = tool_call.get("function")

                    if function is not None:
                        name = getattr(function, "name", None)
                        arguments = getattr(function, "arguments", None)
                        if isinstance(function, dict):
                            name = function.get("name")
                            arguments = function.get("arguments")

                        if name:
                            target["function"]["name"] += str(name)
                        if arguments:
                            target["function"]["arguments"] += str(arguments)

        except Exception as exc:
            logger.error(
                "Streaming generation failed: "
                f"{type(exc).__name__}: {exc}"
            )
            raise RuntimeError(
                "OpenRouter streaming failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        serialized_tool_calls = [
            value
            for _, value in sorted(tool_buffers.items(), key=lambda item: item[0])
            if value.get("function", {}).get("name")
        ]

        thinking = "".join(thinking_parts) or None
        content = "".join(content_parts)

        message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if thinking:
            message["reasoning"] = thinking
            message["thinking"] = thinking
        if serialized_tool_calls:
            message["tool_calls"] = serialized_tool_calls

        cls._emit_stream_event(
            callback,
            "generation_done",
            thinking_tokens=len(thinking.split()) if thinking else 0,
            response_chars=len(content),
            tool_calls=len(serialized_tool_calls),
            usage=usage_total,
        )

        return LLMResult(
            response=content,
            message=message,
            tool_calls=serialized_tool_calls,
            thinking=thinking,
            usage=usage_total,
            raw=last_event,
        )

    def generate(
        self,
        llminput: LLMInput,
    ) -> LLMResult:

        self._create_client()

        model_name = llminput.model_name or self.defaultModel

        messages = llminput.messages or []

        tools = llminput.tools or []

        options = dict(self.defaultConfig)

        if llminput.options:
            options.update(llminput.options)

        stream_callback = getattr(llminput, "stream_callback", None)
        think = options.pop("think", None)

        # Map Daena's shared reasoning setting to OpenRouter's reasoning option.
        if think not in (None, False, "false", "off"):
            effort = "medium" if think is True else str(think).lower()
            if effort in {"low", "medium", "high"}:
                options["reasoning"] = {"effort": effort}

        # Runtime settings are shared at the CLI level, but some are
        # provider-specific. Ollama's local CPU-thread setting is ignored
        # by OpenRouter instead of leaking into the SDK request.
        options.pop("num_thread", None)
        options.pop("num_threads", None)

        # Do not send tool-specific options when
        # the request has no tools.
        if not tools:
            options.pop("tool_choice", None)

            options.pop("parallel_tool_calls", None)

        if callable(stream_callback):
            return self._generate_streaming(
                self.client,
                model_name=model_name,
                messages=messages,
                tools=tools,
                options=options,
                think=think,
                callback=stream_callback,
            )

        request_kwargs = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            **options,
        }

        if tools:
            request_kwargs["tools"] = tools

        try:
            response = self.client.chat.send(**request_kwargs)

        except Exception as exc:
            logger.error("Chat generation failed: " f"{type(exc).__name__}: {exc}")

            raise RuntimeError(
                "OpenRouter generation failed: " f"{type(exc).__name__}: {exc}"
            ) from exc

        if not response.choices:
            raise RuntimeError("OpenRouter returned no choices")

        # ------------------------------------------------------------
        # Normalize message FIRST.
        # ------------------------------------------------------------

        raw_message = response.choices[0].message

        normalized_message = self._serialize_message(raw_message)

        # ------------------------------------------------------------
        # Extract tool calls from normalized message.
        # ------------------------------------------------------------

        raw_tool_calls = normalized_message.get("tool_calls") or []

        normalized_tool_calls = []

        for tool_call in raw_tool_calls:
            normalized_tool_calls.append(self._serialize_tool_call(tool_call))

        # ------------------------------------------------------------
        # Extract content.
        # ------------------------------------------------------------

        content = normalized_message.get("content") or ""

        if not isinstance(
            content,
            str,
        ):
            content = str(content)

        # ------------------------------------------------------------
        # Extract reasoning.
        # ------------------------------------------------------------

        thinking = normalized_message.get("reasoning") or None

        if thinking is not None:
            thinking = str(thinking)

        # ------------------------------------------------------------
        # Usage.
        # ------------------------------------------------------------

        usage = getattr(response, "usage", None)

        total_tokens = 0

        if usage is not None:
            total_tokens = getattr(usage, "total_tokens", 0)

            if total_tokens is None:
                total_tokens = 0

        try:
            total_tokens = int(total_tokens)
        except (
            TypeError,
            ValueError,
        ):
            total_tokens = 0

        # ------------------------------------------------------------
        # Diagnostics.
        # ------------------------------------------------------------

        tool_names = []

        for call in normalized_tool_calls:
            if not isinstance(
                call,
                dict,
            ):
                continue

            function = call.get("function")

            if not isinstance(
                function,
                dict,
            ):
                continue

            name = function.get("name")

            if name:
                tool_names.append(str(name))

        logger.debug(
            f"Model={model_name} "
            f"tool_calls={len(normalized_tool_calls)} "
            f"usage={total_tokens}"
        )

        logger.debug("OpenRouter message keys=" f"{list(normalized_message.keys())}")

        if tool_names:
            logger.debug(f"OpenRouter tool call names={tool_names}")

        # ------------------------------------------------------------
        # Return runtime-independent result.
        # ------------------------------------------------------------

        return LLMResult(
            response=content,
            message=normalized_message,
            tool_calls=normalized_tool_calls,
            thinking=thinking,
            usage=total_tokens,
            raw=response,
        )
