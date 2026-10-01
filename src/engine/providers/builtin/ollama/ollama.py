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
        "num_ctx": 32768,
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
            )

        except Exception as exc:

            logger.error("Chat generation failed: " f"{type(exc).__name__}: {exc}")

            raise RuntimeError(
                "Ollama generation failed: " f"{type(exc).__name__}: {exc}"
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
