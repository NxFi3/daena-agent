from __future__ import annotations

import json
import os
from typing import Any, Callable, ClassVar

try:
    from google import genai
    from google.genai import types
except ImportError:  # Optional provider dependency.
    genai = None
    types = None

from src.engine.providers.ProviderBase import ProviderBase
from src.models.LLMInput import LLMInput
from src.models.LLMResult import LLMResult
from src.utils.logger import get_logger

logger = get_logger("[GEMINI]")


class GeminiProvider(ProviderBase):
    name = "gemini"
    defaultModel = "gemini-3.5-flash-lite"
    defaultConfig: ClassVar[dict] = {"temperature": 0.3}

    def __init__(self) -> None:
        self.client: genai.Client | None = None

    def _create_client(self) -> None:
        if self.client is not None:
            return
        if genai is None or types is None:
            raise RuntimeError(
                "Gemini provider requires the google-genai package. "
                "Install dependencies with: pip install -r requirements-dev.txt"
            )
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

    @classmethod
    def _generate_streaming(
        cls,
        client: genai.Client,
        *,
        model_name: str,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
        callback: Callable[[dict[str, Any]], None],
        think: Any,
    ) -> LLMResult:
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        usage = 0
        last_chunk: Any = None

        cls._emit_stream_event(
            callback,
            "generation_start",
            model=model_name,
            think=think,
            streaming=True,
        )

        try:
            stream = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config,
            )

            for chunk in stream:
                last_chunk = chunk

                chunk_usage = cls._usage_tokens(chunk)
                if chunk_usage:
                    usage = chunk_usage

                candidates = getattr(chunk, "candidates", None) or []
                if not candidates:
                    continue

                candidate_content = getattr(candidates[0], "content", None)
                if candidate_content is None:
                    continue

                for part in getattr(candidate_content, "parts", None) or []:
                    text = getattr(part, "text", None)
                    if text:
                        text = str(text)
                        if getattr(part, "thought", False):
                            thinking_parts.append(text)
                            cls._emit_stream_event(
                                callback,
                                "thinking_delta",
                                text=text,
                            )
                        else:
                            content_parts.append(text)
                            cls._emit_stream_event(
                                callback,
                                "content_delta",
                                text=text,
                            )

                    function_call = getattr(part, "function_call", None)
                    if function_call is None:
                        continue

                    name = getattr(function_call, "name", None)
                    if not name:
                        continue

                    args = getattr(function_call, "args", None) or {}
                    call_id = getattr(function_call, "id", None)

                    normalized = {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": str(name),
                            "arguments": json.dumps(
                                dict(args),
                                ensure_ascii=False,
                            ),
                        },
                    }

                    identity = (
                        str(call_id)
                        if call_id
                        else f"{name}:{normalized['function']['arguments']}"
                    )
                    existing = next(
                        (
                            item
                            for item in tool_calls
                            if str(item.get("id") or "")
                            == identity
                            or (
                                not item.get("id")
                                and str(item["function"].get("name", "")) == str(name)
                            )
                        ),
                        None,
                    )
                    if existing is None:
                        tool_calls.append(normalized)

        except Exception as exc:
            logger.error(
                "Streaming generation failed: "
                f"{type(exc).__name__}: {exc}"
            )
            raise RuntimeError(
                "Gemini streaming failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        content = "".join(content_parts)
        thinking = "".join(thinking_parts) or None

        message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if thinking:
            message["thinking"] = thinking
        if tool_calls:
            message["tool_calls"] = tool_calls

        cls._emit_stream_event(
            callback,
            "generation_done",
            thinking_tokens=len(thinking.split()) if thinking else 0,
            response_chars=len(content),
            tool_calls=len(tool_calls),
            usage=usage,
        )

        return LLMResult(
            response=content,
            message=message,
            tool_calls=tool_calls,
            thinking=thinking,
            usage=usage,
            raw=last_chunk,
        )

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

        stream_callback = getattr(llminput, "stream_callback", None)
        think = options.pop("think", None)

        # Runtime settings are shared at the CLI level, but these options
        # belong to other providers and are not accepted by Gemini's SDK.
        options.pop("num_thread", None)
        options.pop("num_threads", None)
        options.pop("think", None)

        tool_choice = options.pop("tool_choice", None)
        options.pop("parallel_tool_calls", None)

        config_kwargs = dict(options)

        if think not in (None, False, "false", "off"):
            # Gemini streams thought summaries when include_thoughts is enabled.
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                include_thoughts=True
            )

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

        generate_config = types.GenerateContentConfig(**config_kwargs)

        if callable(stream_callback):
            return self._generate_streaming(
                self.client,
                model_name=model_name,
                contents=messages,
                config=generate_config,
                callback=stream_callback,
                think=think,
            )

        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=messages,
                config=generate_config,
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
