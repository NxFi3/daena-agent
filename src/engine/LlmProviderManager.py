# src/Engine/llmManagment/LlmProvider.py

from typing import Any, Dict, Optional
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
        on_event=None,
    ):
        if self.model is not None:
            model_name = self.llm_config.get("model_name") or self.model.defaultModel
            inputs = LLMInput(
                model_name=model_name,
                tools=tools or [],
                images=images or [],
                messages=messages,
                options=self.generation_config,
            )
            try:
                results = self.model.generate(inputs, on_event=on_event)
            except TypeError:
                # Providers that do not implement streaming yet remain compatible.
                results = self.model.generate(inputs)
            return results if isinstance(results, LLMResult) else None
