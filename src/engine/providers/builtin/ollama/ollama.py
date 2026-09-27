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

        try:

            chat_response = ollama.chat(
                model=model_name,
                messages=messages,
                tools=tools,
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

        return LLMResult(
            response=(message.content if message else ""),
            message=normalized_message,
            tool_calls=tool_calls,
            thinking=(
                getattr(
                    message,
                    "thinking",
                    None,
                )
                if message
                else None
            ),
            usage=total_tokens,
            raw=chat_response,
        )
