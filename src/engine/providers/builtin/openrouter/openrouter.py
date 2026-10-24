"""OpenRouter's OpenAI-compatible streaming API with per-request resources."""

from __future__ import annotations

import asyncio
import json
import os
from typing import ClassVar
from uuid import uuid4

import httpx

from src.engine.providers.ProviderBase import ProviderBase
from src.engine.errors import OutputTokenLimitError
from src.engine.transport import (
    canonical_messages,
    emit,
    remote_options,
    seconds,
    supervise,
)
from src.models.LLMResult import LLMResult


class OpenRouterProvider(ProviderBase):
    name = "openrouter"
    defaultModel = "openrouter/free"
    defaultConfig: ClassVar[dict] = {"temperature": 0.3}

    def generate(self, inputs):
        return asyncio.run(self._request(inputs))

    async def _request(self, inputs):
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")
        options = remote_options({**self.defaultConfig, **inputs.options})
        total = seconds(options, "total_timeout_seconds")
        read = seconds(options, "request_timeout_seconds")
        connect = seconds(options, "connect_timeout_seconds", 10)
        base = options.pop("base_url", "https://openrouter.ai/api/v1").rstrip("/")
        think = options.pop("think", None)
        if think is False or think == "off":
            options["reasoning"] = {"enabled": False}
        elif think is not None:
            options["reasoning"] = {"effort": "medium" if think is True else think}
        if not inputs.tools:
            options.pop("tool_choice", None)
            options.pop("parallel_tool_calls", None)
        payload = {
            "model": inputs.model_name or self.defaultModel,
            "messages": canonical_messages(inputs.messages),
            "stream": True,
            "stream_options": {"include_usage": True},
            **options,
        }
        if inputs.tools:
            payload["tools"] = inputs.tools
        callback = inputs.stream_callback

        async def consume():
            content, reasoning, details, calls = [], [], [], {}
            usage, prompt_tokens, finished, marker = 0, 0, None, False
            emit(callback, "generation_start", model=payload["model"], streaming=True)
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(read, connect=connect, write=30, pool=10)
            ) as client:
                async with client.stream(
                    "POST",
                    base + "/chat/completions",
                    json=payload,
                    headers={"Authorization": "Bearer " + key},
                ) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        raise RuntimeError(
                            f"OpenRouter HTTP {response.status_code}; check credentials, model access and provider quota"
                        )
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            marker = True
                            break
                        if not data:
                            continue
                        event = json.loads(data)
                        if event.get("error"):
                            raise RuntimeError(
                                "OpenRouter returned a stream error; no partial tools executed"
                            )
                        tokens = event.get("usage") or {}
                        usage = tokens.get("total_tokens", usage)
                        prompt_tokens = tokens.get("prompt_tokens", prompt_tokens)
                        for choice in event.get("choices", []):
                            if choice.get("index", 0) != 0:
                                continue
                            finished = choice.get("finish_reason") or finished
                            delta = choice.get("delta") or {}
                            if delta.get("content"):
                                content.append(delta["content"])
                                emit(callback, "content_delta", text=delta["content"])
                            if delta.get("reasoning"):
                                reasoning.append(delta["reasoning"])
                                emit(
                                    callback, "thinking_delta", text=delta["reasoning"]
                                )
                            details.extend(delta.get("reasoning_details") or [])
                            for call in delta.get("tool_calls") or []:
                                target = calls.setdefault(
                                    call.get("index", 0),
                                    {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    },
                                )
                                if call.get("id"):
                                    target["id"] = call["id"]
                                for field in ("name", "arguments"):
                                    target["function"][field] += (
                                        call.get("function") or {}
                                    ).get(field) or ""
            if finished in {"length", "max_tokens"}:
                raise OutputTokenLimitError(
                    "OpenRouter reached its output-token limit; no partial tools were executed.",
                    recovery_hint=(
                        "The previous generation was discarded before tool execution. "
                        "Continue the same task with smaller edits/tool calls."
                    ),
                )
            if not marker or finished not in {"stop", "tool_calls"}:
                raise RuntimeError(
                    f"OpenRouter stream incomplete or stopped with {finished!r}; no partial tools executed"
                )
            serialized = [calls[i] for i in sorted(calls)]
            for call in serialized:
                call["id"] = call["id"] or uuid4().hex
                if not call["function"]["name"] or not isinstance(
                    json.loads(call["function"]["arguments"]), dict
                ):
                    raise ValueError("OpenRouter returned invalid tool-call arguments")
            message = {"role": "assistant", "content": "".join(content)}
            if serialized:
                message["tool_calls"] = serialized
            if details:
                message["reasoning_details"] = details
            emit(callback, "generation_done", usage=usage)
            return LLMResult(
                message["content"],
                message,
                serialized,
                "".join(reasoning) or None,
                usage,
                raw={"prompt_eval_count": prompt_tokens},
            )

        return await supervise(consume(), inputs.stop_event, total, callback)
