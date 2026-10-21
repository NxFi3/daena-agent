# src/Engine/llmManagment/LlmProvider.py

from typing import Any, Callable, Dict, Optional
from copy import deepcopy
from src.engine.providers.Registry import ProviderRegistry
from src.models.LLMInput import LLMInput
from src.models.LLMResult import LLMResult
from src.utils.logger import get_logger
import numpy as np

logger = get_logger("[LLM]")


class LlmProvider:
    def __init__(self, config) -> None:
        if not isinstance(config, dict) or not isinstance(config.get("llm", {}), dict):
            raise ValueError("config and llm must be objects")
        self.model = None
        llm_config = config.get("llm") or {}
        self.registry = ProviderRegistry()
        self.provider_name = llm_config.get("provider", "ollama")
        self.llm_config = llm_config.get("provider_config") or {"model_name": ""}
        if not isinstance(self.llm_config, dict):
            raise ValueError("llm.provider_config must be an object")
        self.generation_config = self.llm_config.get("generation_config", {})
        if not isinstance(self.generation_config, dict):
            raise ValueError("generation_config must be an object")
        self.discovered_providers = self.registry.discover()
        self._loadModel(self.provider_name)
        self._profiles = {}

    def switch_provider(self, name):
        if name == self.provider_name:
            return
        if not self.registry.is_available(name):
            raise ValueError(f"Provider '{name}' is unavailable")
        self._profiles[self.provider_name] = {
            **deepcopy(self.llm_config),
            "generation_config": deepcopy(self.generation_config),
        }
        self._loadModel(name)
        self.provider_name = name
        self.llm_config = deepcopy(
            self._profiles.get(
                name,
                {
                    "model_name": self.model.defaultModel,
                    "generation_config": dict(self.model.defaultConfig),
                },
            )
        )
        self.generation_config = self.llm_config.get("generation_config", {})

    def _loadModel(self, provider_name: str):

        if self.registry.is_available(provider_name):
            logger.info(f"Loading {provider_name} provider")
            self.model = self.registry.get(provider_name)
        else:
            raise ValueError(
                f"Provider '{provider_name}' is unavailable. Install its optional dependencies "
                f"and configure it. Available providers: {self.discovered_providers}"
            )

    def generate(
        self,
        messages: list[Dict[str, Any]],
        tools: Optional[list] = None,
        images: Optional[list[np.ndarray]] = None,
        options: Optional[Dict[str, Any]] = None,
        on_event: Optional[Callable[[dict[str, Any]], None]] = None,
        stop_event: Any | None = None,
    ):
        if self.model is not None:
            if stop_event is not None and stop_event.is_set():
                raise InterruptedError("LLM generation interrupted before start.")

            model_name = self.llm_config.get("model_name") or self.model.defaultModel

            generation_options = dict(self.generation_config or {})
            if self.llm_config.get("base_url"):
                generation_options["base_url"] = self.llm_config["base_url"]
            if options:
                generation_options.update(options)

            if callable(on_event):
                try:
                    on_event(
                        {
                            "type": "generation_requested",
                            "model": model_name,
                            "streaming": True,
                            "think": generation_options.get(
                                "think", getattr(self.model, "DEFAULT_THINK", None)
                            ),
                        }
                    )
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
                stop_event=stop_event,
            )
            results = self.model.generate(inputs)
            if not isinstance(results, LLMResult):
                raise TypeError(
                    f"Provider {self.provider_name} did not return LLMResult"
                )
            return results
