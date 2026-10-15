from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, ClassVar

from src.engine.providers.ProviderBase import ProviderBase
from src.models.LLMInput import LLMInput
from src.models.LLMResult import LLMResult
from src.utils.logger import get_logger

logger = get_logger("[SMOLLM3]")


class SmolLM3Provider(ProviderBase):
    """Local text-only GGUF provider for SmolLM3 via llama-cpp-python.

    SmolLM3 documents XML-wrapped JSON tool calls. Daena's dispatcher still
    owns validation and execution; this class only normalizes model output.
    """

    name = "smollm3"
    defaultModel = os.getenv("DAENA_SMOLLM3_MODEL", "SmolLM3-3B-q4.gguf")
    defaultConfig: ClassVar[dict[str, Any]] = {
        "temperature": 0.3,
        "top_p": 0.95,
        "max_tokens": 512,
        "n_ctx": 4096,
        "n_threads": 4,
        "n_batch": 128,
        "n_gpu_layers": 0,
        "repeat_penalty": 1.05,
        "think": False,
    }

    def __init__(self) -> None:
        self._llm: Any | None = None
        self._load_signature: tuple[Any, ...] | None = None

    @staticmethod
    def _tool_schema(tool: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(tool, dict):
            return None
        if isinstance(tool.get("function"), dict):
            function = tool["function"]
        elif tool.get("name"):
            function = tool
        else:
            return None
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            return None
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}
        return {
            "name": name,
            "description": str(function.get("description") or ""),
            "parameters": parameters,
        }

    @classmethod
    def _render_prompt(
        cls,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        think: bool = False,
    ) -> str:
        system_parts: list[str] = []
        conversation: list[str] = []

        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user")
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = str(content)

            if role == "system":
                if content.strip():
                    system_parts.append(content.strip())
                continue

            if role == "assistant":
                body = content
                for call in message.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function") or {}
                    if not isinstance(function, dict) or not function.get("name"):
                        continue
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except (json.JSONDecodeError, TypeError):
                            arguments = {}
                    if not isinstance(arguments, dict):
                        arguments = {}
                    payload = json.dumps(
                        {"name": str(function["name"]), "arguments": arguments},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    body = (body + "\n" if body else "") + f"<tool_call>{payload}</tool_call>"
                conversation.append(f"<|im_start|>assistant\n{body}<|im_end|>\n")
                continue

            if role == "tool":
                tool_name = str(message.get("name") or "tool")
                content = f"Tool result ({tool_name}):\n{content}"
            conversation.append(f"<|im_start|>user\n{content}<|im_end|>\n")

        system_text = "\n\n".join(system_parts)
        if not system_text:
            system_text = "You are SmolLM, a helpful AI assistant. Follow instructions and use tools when useful."
        system_text += "\n\n" + ("/think" if think else "/no_think")

        normalized_tools = [
            item for tool in tools
            if (item := cls._tool_schema(tool)) is not None
        ]
        if normalized_tools:
            tool_lines = "\n".join(
                json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                for item in normalized_tools
            )
            system_text += (
                "\n\n## Tools\n\n### XML Tools\n\n"
                "You may call functions to assist with the user query. Function signatures are in <tools> tags:\n"
                "<tools>\n" + tool_lines + "\n</tools>\n\n"
                "For each function call, return JSON with function name and arguments inside "
                "<tool_call></tool_call> tags. Do not invent tool names or arguments.\n\n"
                "### Python Tools\n\nNone"
            )
        else:
            system_text += "\n\n## Tools\n\n### XML Tools\n\nNone\n\n### Python Tools\n\nNone"

        prefix = f"<|im_start|>system\n{system_text}<|im_end|>\n"
        assistant_prefix = "<|im_start|>assistant\n" + (
            "<think>\n" if think else "<think>\n\n</think>\n"
        )
        return prefix + "".join(conversation) + assistant_prefix

    @staticmethod
    def _parse_response(text: str) -> tuple[str, list[dict[str, Any]], str | None]:
        thinking_parts = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        thinking = "\n".join(part.strip() for part in thinking_parts if part.strip()) or None
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        calls: list[dict[str, Any]] = []

        for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", cleaned, flags=re.DOTALL):
            try:
                payload = json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                logger.warning("SmolLM3 emitted malformed XML tool-call JSON.")
                continue
            if not isinstance(payload, dict):
                continue
            name = payload.get("name")
            arguments = payload.get("arguments", {})
            if not isinstance(name, str) or not name.strip():
                continue
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
            if not isinstance(arguments, dict):
                continue
            calls.append({
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                },
            })

        cleaned = re.sub(r"<tool_call>\s*.*?\s*</tool_call>", "", cleaned, flags=re.DOTALL).strip()
        return cleaned, calls, thinking

    def _load_model(self, model_path: str, options: dict[str, Any]) -> Any:
        signature = (
            model_path,
            int(options.get("n_ctx", 4096)),
            int(options.get("n_threads", 4)),
            int(options.get("n_batch", 128)),
            int(options.get("n_gpu_layers", 0)),
        )
        if self._llm is not None and signature == self._load_signature:
            return self._llm
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "SmolLM3 requires llama-cpp-python. Install it with "
                "python -m pip install llama-cpp-python."
            ) from exc
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"SmolLM3 GGUF not found: {model_path}. Set provider_config.model_name "
                "to the full GGUF path or set DAENA_SMOLLM3_MODEL."
            )
        logger.info(f"Loading SmolLM3 GGUF: {model_path}")
        self._llm = Llama(
            model_path=model_path,
            n_ctx=signature[1],
            n_threads=signature[2],
            n_batch=signature[3],
            n_gpu_layers=signature[4],
            verbose=False,
        )
        self._load_signature = signature
        return self._llm

    def generate(self, inputs: LLMInput) -> LLMResult:
        if getattr(inputs, "images", None):
            raise NotImplementedError("SmolLM3-3B GGUF provider is text-only; images are unsupported.")
        stop_event = getattr(inputs, "stop_event", None)
        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("LLM generation interrupted before start.")

        options = dict(self.defaultConfig)
        options.update(inputs.options or {})
        model_path = os.path.expanduser(os.path.expandvars(str(inputs.model_name or self.defaultModel)))
        think_value = options.pop("think", False)
        think = bool(think_value) and str(think_value).lower() not in {"false", "off", "0", "none"}
        prompt = self._render_prompt(inputs.messages or [], inputs.tools or [], think=think)
        llm = self._load_model(model_path, options)

        max_tokens = int(options.pop("max_tokens", options.pop("num_predict", 512)))
        temperature = float(options.pop("temperature", 0.3))
        top_p = float(options.pop("top_p", 0.95))
        repeat_penalty = float(options.pop("repeat_penalty", 1.05))
        for key in ("n_ctx", "n_threads", "n_batch", "n_gpu_layers", "num_thread", "num_threads", "think"):
            options.pop(key, None)

        if stop_event is not None and stop_event.is_set():
            raise InterruptedError("LLM generation interrupted before inference.")
        raw = llm.create_completion(
            prompt=prompt,
            max_tokens=max(1, max_tokens),
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=["<|im_end|>", "<|end_of_text|>"],
            echo=False,
        )
        choices = raw.get("choices") or []
        generated = str(choices[0].get("text") or "") if choices else ""
        response, tool_calls, thinking = self._parse_response(generated)
        usage = int((raw.get("usage") or {}).get("total_tokens") or 0)
        message: dict[str, Any] = {"role": "assistant", "content": response}
        if tool_calls:
            message["tool_calls"] = tool_calls
        if thinking:
            message["thinking"] = thinking

        callback = getattr(inputs, "stream_callback", None)
        if callable(callback):
            events = [{"type": "generation_start", "model": model_path, "streaming": False}]
            if thinking:
                events.append({"type": "thinking_delta", "text": thinking})
            if response:
                events.append({"type": "content_delta", "text": response})
            events.append({
                "type": "generation_done",
                "response_chars": len(response),
                "tool_calls": len(tool_calls),
                "usage": usage,
                "streaming": False,
            })
            for event in events:
                try:
                    callback(event)
                except Exception as exc:
                    logger.debug(f"Stream callback failed: {type(exc).__name__}: {exc}")

        return LLMResult(
            response=response,
            message=message,
            tool_calls=tool_calls,
            thinking=thinking,
            usage=usage,
            raw=raw,
        )
