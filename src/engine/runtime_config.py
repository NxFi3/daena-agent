"""Resolve live provider settings after CLI model/provider/config changes."""

from copy import deepcopy


def provider_name(llm, config):
    return getattr(llm, "provider_name", None) or config.get("llm", {}).get(
        "provider", "ollama"
    )


def effective_config(config, llm):
    result = deepcopy(config)
    active = result.setdefault("llm", {})
    active["provider"] = provider_name(llm, config)
    live = getattr(llm, "llm_config", None)
    if isinstance(live, dict):
        active["provider_config"] = deepcopy(live)
    generation = getattr(llm, "generation_config", None)
    if isinstance(generation, dict):
        active.setdefault("provider_config", {})["generation_config"] = deepcopy(
            generation
        )
    return result
