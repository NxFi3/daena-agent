from __future__ import annotations

import json
import os
from typing import Any, ClassVar

from google import genai
from google.genai import types

from src.engine.providers.ProviderBase import ProviderBase
from src.models.LLMInput import LLMInput
from src.models.LLMResult import LLMResult
from src.utils.logger import get_logger

logger = get_logger("[GEMINI]")


class GeminiProvider(ProviderBase):
    name = "gemini"
    defaultModel = "gemini-2.5-flash-lite"
    defaultConfig: ClassVar[dict] = {"temperature": 0.3}

    def __init__(self) -> None:
        self.client: genai.Client | None = None

    def _create_client(self) -> None:
        if self.client is not None:
            return
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        self.client = genai.Client(api_key=api_key)

    @staticmethod
    def _sanitize_schema(value: Any) -> Any:
        if isinstance(value, list):
            return [GeminiProvider._sanitize_schema(item) for item in value]
        if not isinstance(value, dict):
            return value

        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key == "additional_properties":
                key = "additionalProperties"
            sanitized[key] = GeminiProvider._sanitize_schema(item)
        return sanitized

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[types.Tool]:
        if not tools:
            return []

        declarations: list[types.FunctionDeclaration] = []

        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                continue

            function = tool.get("function")
            if not isinstance(function, dict):
                continue

            name = function.get("name")
            if not name:
                continue

            parameters = function.get(
                "parameters",
                {"type": "object", "properties": {}},
            )

            declarations.append(
                types.FunctionDeclaration(
                    name=str(name),
                    description=str(function.get("description") or ""),
                    parameters_json_schema=GeminiProvider._sanitize_schema(parameters),
                )
            )

        if not declarations:
            return []

        return [types.Tool(function_declarations=declarations)]

    @staticmethod
    def _parse_tool_response(content: Any) -> Any:
        if not isinstance(content, str):
            return content
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {"result": content}

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[list[types.Content], str | None]:
        contents: list[types.Content] = []
        system_instruction: str | None = None

        for message in messages:
            if not isinstance(message, dict):
                continue

            role = message.get("role", "user")
            content = message.get("content")

            if role == "system":
                text = content if isinstance(content, str) else str(content or "")
                system_instruction = (
                    f"{system_instruction}\n\n{text}"
                    if system_instruction
                    else text
                )
                continue

            if role == "tool":
                name = message.get("name")
                tool_call_id = message.get("tool_call_id")

                if not name:
                    continue

                function_response = types.FunctionResponse(
                    name=str(name),
                    response=GeminiProvider._parse_tool_response(content),
                )

                if tool_call_id:
                    function_response.id = str(tool_call_id)

                contents.append(
                    types.Content(
                        role="tool",
                        parts=[types.Part(function_response=function_response)],
                    )
                )
                continue

            if role == "assistant":
                parts: list[types.Part] = []

                if isinstance(content, str) and content:
                    parts.append(types.Part.from_text(text=content))

                for tool_call in message.get("tool_calls") or []:
                    if not isinstance(tool_call, dict):
                        continue

                    function = tool_call.get("function")
                    if not isinstance(function, dict):
                        continue

                    name = function.get("name")
                    if not name:
                        continue

                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except (json.JSONDecodeError, TypeError, ValueError):
                            arguments = {}

                    if not isinstance(arguments, dict):
                        arguments = {}

                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                id=tool_call.get("id"),
                                name=str(name),
                                args=arguments,
                            )
                        )
                    )

                if parts:
                    contents.append(types.Content(role="model", parts=parts))
                continue

            text = content if isinstance(content, str) else str(content or "")
            if text:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=text)],
                    )
                )

        return contents, system_instruction

    @staticmethod
    def _normalize_response(
        response: Any,
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]], str | None]:
        content_text = ""
        thinking_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return "", {}, [], None

        candidate_content = getattr(candidates[0], "content", None)
        if candidate_content is None:
            return "", {}, [], None

        for part in getattr(candidate_content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                if getattr(part, "thought", False):
                    thinking_parts.append(str(text))
                else:
                    content_text += str(text)

            function_call = getattr(part, "function_call", None)
            if function_call is None:
                continue

            name = getattr(function_call, "name", None)
            if not name:
                continue

            args = getattr(function_call, "args", None) or {}
            tool_calls.append(
                {
                    "id": getattr(function_call, "id", None),
                    "type": "function",
                    "function": {
                        "name": str(name),
                        "arguments": json.dumps(
                            dict(args),
                            ensure_ascii=False,
                        ),
                    },
                }
            )

        message: dict[str, Any] = {
            "role": "assistant",
            "content": content_text,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        return (
            content_text,
            message,
            tool_calls,
            "\n".join(thinking_parts) or None,
        )

    @staticmethod
    def _usage_tokens(response: Any) -> int:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return 0

        total = getattr(usage, "total_token_count", None)
        if total is not None:
            try:
                return int(total)
            except (TypeError, ValueError):
                pass

        prompt = getattr(usage, "prompt_token_count", 0) or 0
        output = getattr(usage, "candidates_token_count", 0) or 0

        try:
            return int(prompt) + int(output)
        except (TypeError, ValueError):
            return 0

    def generate(self, llminput: LLMInput) -> LLMResult:
        self._create_client()

        model_name = llminput.model_name or self.defaultModel
        messages, system_instruction = self._convert_messages(
            llminput.messages or []
        )
        tools = self._convert_tools(llminput.tools or [])

        options = dict(self.defaultConfig)
        if llminput.options:
            options.update(llminput.options)

        tool_choice = options.pop("tool_choice", None)
        options.pop("parallel_tool_calls", None)

        config_kwargs = dict(options)

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        if tools:
            config_kwargs["tools"] = tools
            config_kwargs["automatic_function_calling"] = (
                types.AutomaticFunctionCallingConfig(disable=True)
            )

            if tool_choice == "none":
                mode = "NONE"
            elif tool_choice in ("required", "any"):
                mode = "ANY"
            else:
                mode = "AUTO"

            config_kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=mode)
            )

        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=messages,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:
            logger.error(
                "Chat generation failed: "
                f"{type(exc).__name__}: {exc}"
            )
            raise RuntimeError(
                "Gemini generation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        content, message, tool_calls, thinking = self._normalize_response(response)
        usage = self._usage_tokens(response)

        logger.debug(
            f"Model={model_name} "
            f"tool_calls={len(tool_calls)} "
            f"usage={usage}"
        )

        return LLMResult(
            response=content,
            message=message,
            tool_calls=tool_calls,
            thinking=thinking,
            usage=usage,
            raw=response,
        )
