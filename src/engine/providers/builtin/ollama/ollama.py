from __future__ import annotations

import json
from typing import Any, ClassVar

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

    def generate(
        self,
        inputs: LLMInput,
        on_event=None,
    ) -> LLMResult:
        model_name = inputs.model_name or self.defaultModel

        messages = self._prepare_messages(inputs.messages or [])
        tools = inputs.tools or []

        options = dict(self.defaultConfig)
        if inputs.options:
            options.update(inputs.options)

        think = self._resolve_think(options)

        try:
            chat_response = ollama.chat(
                model=model_name,
                messages=messages,
                tools=tools,
                think=think,
                options=options,
                stream=bool(on_event),
            )

            if not on_event:
                return self._result_from_response(chat_response, model_name)

            thinking_parts: list[str] = []
            content_parts: list[str] = []
            last_message = None
            last_response = None
            tool_calls = []

            for chunk in chat_response:
                last_response = chunk
                message = getattr(chunk, "message", None)
                last_message = message or last_message

                thinking = getattr(message, "thinking", None) if message else None
                content = getattr(message, "content", None) if message else None

                if thinking:
                    thinking = str(thinking)
                    thinking_parts.append(thinking)
                    on_event({"type": "thinking_delta", "text": thinking})

                if content:
                    content = str(content)
                    content_parts.append(content)
                    on_event({"type": "content_delta", "text": content})

                raw_calls = getattr(message, "tool_calls", None) if message else None
                if raw_calls:
                    tool_calls = list(raw_calls)

            thinking_text = "".join(thinking_parts)
            response_text = "".join(content_parts)

            if last_message is not None:
                normalized_message = (
                    last_message.model_dump()
                    if hasattr(last_message, "model_dump")
                    else {}
                )
            else:
                normalized_message = {}

            prompt_tokens = getattr(last_response, "prompt_eval_count", 0) or 0
            completion_tokens = getattr(last_response, "eval_count", 0) or 0

            if last_response is not None:
                raw_calls = getattr(last_message, "tool_calls", None) if last_message else None
                if raw_calls:
                    tool_calls = list(raw_calls)

            result = LLMResult(
                response=response_text,
                message=normalized_message,
                tool_calls=tool_calls,
                thinking=thinking_text or None,
                usage=int(prompt_tokens) + int(completion_tokens),
                raw=last_response,
            )

            on_event({
                "type": "generation_done",
                "usage": result.usage,
            })
            return result

        except Exception as exc:
            logger.error("Chat generation failed: " f"{type(exc).__name__}: {exc}")
            raise RuntimeError(
                "Ollama generation failed: " f"{type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _result_from_response(chat_response, model_name: str) -> LLMResult:
        message = chat_response.message if chat_response else None

        prompt_tokens = getattr(chat_response, "prompt_eval_count", 0) or 0
        completion_tokens = getattr(chat_response, "eval_count", 0) or 0
        total_tokens = prompt_tokens + completion_tokens

        tool_calls = list(getattr(message, "tool_calls", None) or []) if message else []
        normalized_message = (
            message.model_dump()
            if message and hasattr(message, "model_dump")
            else {}
        )
        thinking = getattr(message, "thinking", None) if message else None
        response_text = message.content if message else ""

        if not response_text and not tool_calls and thinking:
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

