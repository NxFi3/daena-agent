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

    # Use a model available through Gemini API.
    defaultModel = "gemini-2.5-flash-lite"

    defaultConfig: ClassVar[dict] = {
        "temperature": 0.3,
    }

    def __init__(self) -> None:
        super().__init__()
        self.client: genai.Client | None = None

    def _create_client(self) -> None:

        if self.client is not None:
            return

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set"
            )

        self.client = genai.Client(
            api_key=api_key
        )

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, Any]],
    ) -> list[types.Content]:

        contents: list[types.Content] = []

        for message in messages:

            if not isinstance(message, dict):
                continue

            role = message.get("role")

    
            if role == "system":
                continue

          
            if role == "user":

                content = message.get("content")

                if content is None:
                    continue

                if not isinstance(content, str):
                    content = str(content)

                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(
                                text=content
                            )
                        ],
                    )
                )

                continue

         
            if role == "assistant":

                parts: list[types.Part] = []

                content = message.get("content")

                if content:
                    if not isinstance(content, str):
                        content = str(content)

                    parts.append(
                        types.Part.from_text(
                            text=content
                        )
                    )

                tool_calls = message.get("tool_calls") or []

                for tool_call in tool_calls:

                    if hasattr(tool_call, "model_dump"):
                        tool_call = tool_call.model_dump(
                            exclude_none=True
                        )

                    if not isinstance(tool_call, dict):
                        continue

                    function = tool_call.get("function")

                    if not isinstance(function, dict):
                        continue

                    name = function.get("name")

                    if not name:
                        continue

                    arguments = function.get("arguments") or {}

                    # OpenAI-compatible providers may return arguments
                    # as a JSON string.
                    if isinstance(arguments, str):

                        import json

                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}

                    if not isinstance(arguments, dict):
                        arguments = {}

                    call_id = tool_call.get("id")

                    parts.append(
                        types.Part.from_function_call(
                            name=name,
                            args=arguments,
                            id=call_id,
                        )
                    )

                if parts:

                    contents.append(
                        types.Content(
                            role="model",
                            parts=parts,
                        )
                    )

                continue

            if role == "tool":

                tool_name = message.get("name")

                if not tool_name:
                    continue

                result = message.get("content")

                if result is None:
                    result = ""

                if not isinstance(result, str):
                    result = str(result)

                call_id = message.get("tool_call_id")

                import json

                response_data: dict[str, Any]

                try:
                    parsed = json.loads(result)

                    if isinstance(parsed, dict):
                        response_data = parsed
                    else:
                        response_data = {
                            "result": parsed
                        }

                except (json.JSONDecodeError, TypeError):
                    response_data = {
                        "result": result
                    }

                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=tool_name,
                                response=response_data,
                                id=call_id,
                            )
                        ],
                    )
                )

                continue

        return contents


    @staticmethod
    def _extract_system_instruction(
        messages: list[dict[str, Any]],
    ) -> str | None:

        system_messages: list[str] = []

        for message in messages:

            if not isinstance(message, dict):
                continue

            if message.get("role") != "system":
                continue

            content = message.get("content")

            if content is None:
                continue

            if not isinstance(content, str):
                content = str(content)

            system_messages.append(content)

        if not system_messages:
            return None

        return "\n\n".join(system_messages)


    @staticmethod
    def _convert_tools(
        tools: list[dict[str, Any]],
    ) -> list[types.Tool]:

        if not tools:
            return []

        declarations: list[dict[str, Any]] = []

        for tool in tools:

            if not isinstance(tool, dict):
                continue

            # OpenAI format:
            #
            # {
            #   "type": "function",
            #   "function": {
            #       "name": "...",
            #       "description": "...",
            #       "parameters": {...}
            #   }
            # }

            if tool.get("type") == "function":

                function = tool.get("function")

                if not isinstance(function, dict):
                    continue

                declaration = {
                    "name": function.get("name"),
                    "description": function.get(
                        "description",
                        "",
                    ),
                    "parameters": function.get(
                        "parameters",
                        {
                            "type": "object",
                            "properties": {},
                        },
                    ),
                }

                if declaration["name"]:
                    declarations.append(declaration)

                continue

            # Already Gemini-like declaration.

            if tool.get("name"):
                declarations.append(tool)

        if not declarations:
            return []

        return [
            types.Tool(
                function_declarations=declarations
            )
        ]

 
    @staticmethod
    def _serialize_tool_call(
        function_call: Any,
    ) -> dict[str, Any]:

        name = getattr(
            function_call,
            "name",
            None,
        )

        args = getattr(
            function_call,
            "args",
            None,
        )

        call_id = getattr(
            function_call,
            "id",
            None,
        )

        if args is None:
            args = {}

        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": args,
            },
        }

    def generate(
        self,
        llminput: LLMInput,
    ) -> LLMResult:

        self._create_client()

        if self.client is None:
            raise RuntimeError(
                "Gemini client was not initialized"
            )

        model_name = (
            llminput.model_name
            or self.defaultModel
        )

        messages = llminput.messages or []
        tools = llminput.tools or []

        options = dict(
            self.defaultConfig
        )

        if llminput.options:
            options.update(
                llminput.options
            )

        contents = self._convert_messages(
            messages
        )

        system_instruction = (
            self._extract_system_instruction(
                messages
            )
        )

        gemini_tools = self._convert_tools(
            tools
        )

        config_kwargs: dict[str, Any] = {}

      
        if "temperature" in options:
            config_kwargs["temperature"] = (
                options["temperature"]
            )

        if "top_p" in options:
            config_kwargs["top_p"] = (
                options["top_p"]
            )

        if "top_k" in options:
            config_kwargs["top_k"] = (
                options["top_k"]
            )

        if "max_output_tokens" in options:
            config_kwargs["max_output_tokens"] = (
                options["max_output_tokens"]
            )

        if system_instruction:
            config_kwargs["system_instruction"] = (
                system_instruction
            )

        if gemini_tools:
            config_kwargs["tools"] = gemini_tools

            # We want DAENA to execute tools itself.
            config_kwargs[
                "automatic_function_calling"
            ] = types.AutomaticFunctionCallingConfig(
                disable=True
            )

        config = types.GenerateContentConfig(
            **config_kwargs
        )

  
        try:

            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
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

    
        if not response.candidates:
            raise RuntimeError(
                "Gemini returned no candidates"
            )

        candidate = response.candidates[0]

        if candidate.content is None:
            raise RuntimeError(
                "Gemini returned an empty candidate"
            )

     
        text_parts: list[str] = []
        normalized_tool_calls: list[dict[str, Any]] = []

        for part in candidate.content.parts:

            if getattr(part, "text", None):

                text_parts.append(
                    str(part.text)
                )

            function_call = getattr(
                part,
                "function_call",
                None,
            )

            if function_call is not None:

                normalized_tool_calls.append(
                    self._serialize_tool_call(
                        function_call
                    )
                )

        content = "\n".join(
            text_parts
        )

      
        thinking_parts: list[str] = []

        for part in candidate.content.parts:

            thought = getattr(
                part,
                "thought",
                False,
            )

            text = getattr(
                part,
                "text",
                None,
            )

            if thought and text:
                thinking_parts.append(
                    str(text)
                )

        thinking = (
            "\n".join(thinking_parts)
            if thinking_parts
            else None
        )

     
        usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        total_tokens = 0

        if usage is not None:

            total_tokens = getattr(
                usage,
                "total_token_count",
                0,
            )

            if total_tokens is None:
                total_tokens = 0

        try:
            total_tokens = int(
                total_tokens
            )

        except (TypeError, ValueError):
            total_tokens = 0

        normalized_message: dict[str, Any] = {
            "role": "assistant",
            "content": content or None,
        }

        if normalized_tool_calls:

            normalized_message[
                "tool_calls"
            ] = normalized_tool_calls


        tool_names = []

        for call in normalized_tool_calls:

            function = call.get(
                "function"
            )

            if not isinstance(
                function,
                dict,
            ):
                continue

            name = function.get(
                "name"
            )

            if name:
                tool_names.append(
                    str(name)
                )

        logger.debug(
            f"Model={model_name} "
            f"tool_calls="
            f"{len(normalized_tool_calls)} "
            f"usage={total_tokens}"
        )

        logger.debug(
            "Gemini message keys="
            f"{list(normalized_message.keys())}"
        )

        if tool_names:

            logger.debug(
                "Gemini tool call names="
                f"{tool_names}"
            )

  
        return LLMResult(
            response=content,
            message=normalized_message,
            tool_calls=normalized_tool_calls,
            thinking=thinking,
            usage=total_tokens,
            raw=response,
        )