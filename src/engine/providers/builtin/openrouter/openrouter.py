import os
from typing import Any, ClassVar

from cerebras.cloud.sdk import Cerebras as CerebrasSDK

from src.engine.providers.ProviderBase import ProviderBase
from src.models.LLMInput import LLMInput
from src.models.LLMResult import LLMResult
from src.utils.logger import get_logger

logger = get_logger("[CEREBRAS]")


class CerebrasProvider(ProviderBase):

    name = "cerebras"

    defaultModel = "qwen-3.8-27b"

    defaultConfig: ClassVar[dict] = {
        "temperature": 0.2,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }

    def __init__(self) -> None:
        super().__init__()

        self.client: CerebrasSDK | None = None

    def _create_client(self) -> None:
        if self.client is not None:
            return

        api_key = os.getenv("CEREBRAS_API_KEY")

        if not api_key:
            raise RuntimeError(
                "CEREBRAS_API_KEY environment variable is not set"
            )

        self.client = CerebrasSDK(
            api_key=api_key,
        )

    @staticmethod
    def _serialize_tool_call(
        tool_call: Any,
    ) -> dict:
        if hasattr(tool_call, "model_dump"):
            return tool_call.model_dump(exclude_none=True)

        if isinstance(tool_call, dict):
            return dict(tool_call)

        raise TypeError(
            "Unsupported tool call type: "
            f"{type(tool_call).__name__}"
        )

    @staticmethod
    def _serialize_message(
        message: Any,
    ) -> dict:
        if hasattr(message, "model_dump"):
            return message.model_dump(exclude_none=True)

        if isinstance(message, dict):
            return dict(message)

        raise TypeError(
            "Unsupported message type: "
            f"{type(message).__name__}"
        )

    def generate(
        self,
        llminput: LLMInput,
    ) -> LLMResult:

        self._create_client()

        if self.client is None:
            raise RuntimeError("Cerebras client was not initialized")

        model_name = (
            llminput.model_name
            or self.defaultModel
        )

        messages = llminput.messages or []

        tools = llminput.tools or []

        options = dict(self.defaultConfig)

        if llminput.options:
            options.update(llminput.options)

        # Tool-specific options only make sense
        # when tools are actually supplied.
        if not tools:
            options.pop("tool_choice", None)
            options.pop("parallel_tool_calls", None)

        request_kwargs = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            **options,
        }

        if tools:
            request_kwargs["tools"] = tools

        try:
            response = self.client.chat.completions.create(
                **request_kwargs
            )

        except Exception as exc:
            logger.error(
                "Chat generation failed: "
                f"{type(exc).__name__}: {exc}"
            )

            raise RuntimeError(
                "Cerebras generation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if not response.choices:
            raise RuntimeError(
                "Cerebras returned no choices"
            )

        raw_message = response.choices[0].message

        normalized_message = self._serialize_message(
            raw_message
        )

        raw_tool_calls = (
            normalized_message.get("tool_calls")
            or []
        )

        normalized_tool_calls = []

        for tool_call in raw_tool_calls:
            normalized_tool_calls.append(
                self._serialize_tool_call(tool_call)
            )

        content = (
            normalized_message.get("content")
            or ""
        )

        if not isinstance(content, str):
            content = str(content)

        thinking = (
            normalized_message.get("reasoning")
            or normalized_message.get("reasoning_content")
            or None
        )

        if thinking is not None:
            thinking = str(thinking)

        usage = getattr(
            response,
            "usage",
            None,
        )

        total_tokens = 0

        if usage is not None:
            total_tokens = getattr(
                usage,
                "total_tokens",
                0,
            )

            if total_tokens is None:
                total_tokens = 0

        try:
            total_tokens = int(total_tokens)

        except (
            TypeError,
            ValueError,
        ):
            total_tokens = 0

        tool_names = []

        for call in normalized_tool_calls:

            if not isinstance(call, dict):
                continue

            function = call.get("function")

            if not isinstance(function, dict):
                continue

            name = function.get("name")

            if name:
                tool_names.append(str(name))

        logger.debug(
            f"Model={model_name} "
            f"tool_calls={len(normalized_tool_calls)} "
            f"usage={total_tokens}"
        )

        logger.debug(
            "Cerebras message keys="
            f"{list(normalized_message.keys())}"
        )

        if tool_names:
            logger.debug(
                f"Cerebras tool call names={tool_names}"
            )

        return LLMResult(
            response=content,
            message=normalized_message,
            tool_calls=normalized_tool_calls,
            thinking=thinking,
            usage=total_tokens,
            raw=response,
        )