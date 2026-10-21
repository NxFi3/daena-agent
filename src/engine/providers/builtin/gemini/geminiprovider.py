"""Gemini adapter preserving signed model parts across tool exchanges."""

from __future__ import annotations

import asyncio
import json
import os
from typing import ClassVar
from uuid import uuid4

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = types = None

from src.engine.providers.ProviderBase import ProviderBase
from src.engine.transport import emit, remote_options, seconds, supervise
from src.models.LLMResult import LLMResult


class GeminiProvider(ProviderBase):
    name = "gemini"
    defaultModel = "gemini-2.5-flash"
    defaultConfig: ClassVar[dict] = {"temperature": 0.3}

    @staticmethod
    def _convert_tools(tools):
        declarations = [
            types.FunctionDeclaration(
                name=t["function"]["name"],
                description=t["function"].get("description", ""),
                parameters_json_schema=t["function"].get("parameters", {}),
            )
            for t in tools
            if t.get("type") == "function"
        ]
        return [types.Tool(function_declarations=declarations)] if declarations else []

    @staticmethod
    def _convert_messages(messages):
        contents, systems, names, response_ids = [], [], {}, {}
        for m in messages:
            role, text = m.get("role"), m.get("content") or ""
            if role == "system":
                systems.append(str(text))
                continue
            if role == "assistant":
                calls = m.get("tool_calls") or []
                names.update({c.get("id"): c["function"]["name"] for c in calls})
                signed = m.get("gemini_parts")
                if signed:
                    # Keep opaque thought signatures exactly as supplied by Gemini.
                    parts = [types.Part.model_validate(p) for p in signed]
                    provider_calls = [p.function_call for p in parts if p.function_call]
                    for canonical, provider_call in zip(calls, provider_calls):
                        response_ids[canonical.get("id")] = provider_call.id
                else:
                    parts = [types.Part.from_text(text=str(text))] if text else []
                    for call in calls:
                        function = call["function"]
                        args = function.get("arguments") or {}
                        if isinstance(args, str):
                            args = json.loads(args)
                        parts.append(
                            types.Part(
                                function_call=types.FunctionCall(
                                    id=call.get("id"), name=function["name"], args=args
                                )
                            )
                        )
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
                continue
            if role == "tool":
                name = (
                    m.get("tool_name")
                    or m.get("name")
                    or names.get(m.get("tool_call_id"))
                )
                if not name:
                    raise ValueError(
                        "Gemini tool receipt has no matching function name"
                    )
                try:
                    result = json.loads(text) if isinstance(text, str) else text
                except ValueError:
                    result = {"result": text}
                if not isinstance(result, dict):
                    result = {"result": result}
                part = types.Part(
                    function_response=types.FunctionResponse(
                        id=response_ids.get(
                            m.get("tool_call_id"), m.get("tool_call_id")
                        ),
                        name=name,
                        response=result,
                    )
                )
                if (
                    contents
                    and contents[-1].role == "user"
                    and any(p.function_response for p in contents[-1].parts)
                ):
                    contents[-1].parts.append(part)
                else:
                    contents.append(types.Content(role="user", parts=[part]))
                continue
            if text:
                contents.append(
                    types.Content(
                        role="user", parts=[types.Part.from_text(text=str(text))]
                    )
                )
        return contents, "\n\n".join(systems) or None

    def generate(self, inputs):
        if genai is None:
            raise RuntimeError("Install daena-agent[gemini] to use Gemini")
        return asyncio.run(self._request(inputs))

    async def _request(self, inputs):
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set"
            )
        options = remote_options({**self.defaultConfig, **inputs.options})
        total = seconds(options, "total_timeout_seconds")
        read = seconds(options, "request_timeout_seconds", 600)
        seconds(options, "connect_timeout_seconds", 10)
        base = options.pop("base_url", None)
        think = options.pop("think", None)
        choice = options.pop("tool_choice", "auto")
        options.pop("parallel_tool_calls", None)
        if "max_tokens" in options:
            options["max_output_tokens"] = options.pop("max_tokens")
        if "max_completion_tokens" in options:
            options["max_output_tokens"] = options.pop("max_completion_tokens")
        messages, system = self._convert_messages(inputs.messages)
        tools = self._convert_tools(inputs.tools)
        if "thinking_config" not in options:
            if think is False or think == "off":
                options["thinking_config"] = {"include_thoughts": False, "thinking_budget": 0}
            elif think in {"low", "medium", "high"}:
                options["thinking_config"] = {
                    "include_thoughts": True,
                    "thinking_budget": {"low": 1024, "medium": 4096, "high": 8192}[think],
                }
            elif think is not None:
                options["thinking_config"] = {"include_thoughts": True}
        if system:
            options["system_instruction"] = system
        if tools:
            options["tools"] = tools
            options["automatic_function_calling"] = {"disable": True}
            options["tool_config"] = {
                "function_calling_config": {
                    "mode": "NONE"
                    if choice == "none"
                    else "ANY"
                    if choice in {"required", "any"}
                    else "AUTO"
                }
            }
        config = types.GenerateContentConfig(**options)
        http = {}
        if read is not None:
            http["timeout"] = int(read * 1000)
        if base:
            http["base_url"] = base
        callback = inputs.stream_callback

        async def consume():
            client = genai.Client(api_key=key, http_options=types.HttpOptions(**http))
            parts, text, thoughts, calls = [], [], [], []
            usage, last, reason = 0, None, None
            emit(callback, "generation_start", model=inputs.model_name, streaming=True)
            try:
                stream = await client.aio.models.generate_content_stream(
                    model=inputs.model_name or self.defaultModel,
                    contents=messages,
                    config=config,
                )
                try:
                    async for chunk in stream:
                        last = chunk
                        metadata = chunk.usage_metadata
                        if metadata:
                            usage = metadata.total_token_count or usage
                        if not chunk.candidates:
                            continue
                        candidate = chunk.candidates[0]
                        if candidate.finish_reason:
                            reason = (
                                candidate.finish_reason.value
                                if hasattr(candidate.finish_reason, "value")
                                else str(candidate.finish_reason)
                            )
                        for part in (
                            candidate.content.parts if candidate.content else []
                        ) or []:
                            if part.function_call:
                                call = part.function_call
                                calls.append(
                                    {
                                        "id": call.id or uuid4().hex,
                                        "type": "function",
                                        "function": {
                                            "name": call.name,
                                            "arguments": json.dumps(
                                                dict(call.args or {})
                                            ),
                                        },
                                    }
                                )
                            parts.append(
                                part.model_dump(mode="json", exclude_none=True)
                            )
                            if part.text:
                                (thoughts if part.thought else text).append(part.text)
                                emit(
                                    callback,
                                    "thinking_delta"
                                    if part.thought
                                    else "content_delta",
                                    text=part.text,
                                )
                finally:
                    await stream.aclose()
                if reason != "STOP":
                    raise RuntimeError(
                        f"Gemini stream incomplete or stopped with {reason!r}; no partial tools executed"
                    )
                message = {
                    "role": "assistant",
                    "content": "".join(text),
                    "gemini_parts": parts,
                }
                if calls:
                    message["tool_calls"] = calls
                emit(callback, "generation_done", usage=usage)
                return LLMResult(
                    message["content"],
                    message,
                    calls,
                    "".join(thoughts) or None,
                    usage,
                    raw=last,
                )
            finally:
                await client.aio.aclose()
                client.close()

        return await supervise(consume(), inputs.stop_event, total, callback)
