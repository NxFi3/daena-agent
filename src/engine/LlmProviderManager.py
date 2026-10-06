# src/Engine/llmManagment/LlmProvider.py

from typing import Any, Callable, Dict, Optional
from src.engine.providers.Registry import ProviderRegistry
from src.models.LLMInput import LLMInput
from src.models.LLMResult import LLMResult
from src.utils.logger import get_logger
import numpy as np

logger = get_logger("[LLM]")


class LlmProvider:
    def __init__(self, config) -> None:
        self.model = None
        llm_config = config.get("llm") or {}
        self.registry = ProviderRegistry()
        self.provider_name = llm_config.get("provider", "ollama")
        self.llm_config = llm_config.get("provider_config") or {"model_name": ""}
        self.generation_config = self.llm_config.get("generation_config", {})
        self.discovered_providers = self.registry.discover()
        self._loadModel(self.provider_name)

    def _loadModel(self, provider_name: str):

        if self.registry.is_available(provider_name):
            logger.info(f"Loading {provider_name} provider")
            self.model = self.registry.get(provider_name)
        else:
            logger.error(
                f"{provider_name} not found try one of {self.discovered_providers}"
            )

    def generate(
        self,
        messages: list[Dict[str, Any]],
        tools: Optional[list] = None,
        images: Optional[list[np.ndarray]] = None,
        options: Optional[Dict[str, Any]] = None,
        on_event: Optional[Callable[[dict[str, Any]], None]] = None,
    ):
        if self.model is not None:
            model_name = self.llm_config.get("model_name") or self.model.defaultModel

            generation_options = dict(self.generation_config or {})
            if options:
                generation_options.update(options)

            if callable(on_event):
                try:
                    on_event({
                        "type": "generation_requested",
                        "model": model_name,
                        "streaming": True,
                        "think": generation_options.get("think", "medium"),
                    })
                except Exception as exc:
                    logger.debug(
                        f"Generation event callback failed: {type(exc).__name__}: {exc}"
                    )

            inputs = LLMInput(
                model_name=model_name,
                tools=tools or [],
                images=images or [],
                messages=messages,
                options=generation_options,
                stream_callback=on_event,
            )
            results = self.model.generate(inputs)
            return results if isinstance(results, LLMResult) else None
