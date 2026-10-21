"""Shared model/reasoning profile transformations for both desktop editors."""
from copy import deepcopy

DEFAULTS = {
    "ollama": {"model_name":"qwen3.6:35b-a3b-q4_K_M","generation_config":{"temperature":0.3,"num_ctx":32768,"request_timeout_seconds":None}},
    "openrouter": {"model_name":"openrouter/free","generation_config":{"temperature":0.3,"request_timeout_seconds":None}},
    "gemini": {"model_name":"gemini-2.5-flash","generation_config":{"temperature":0.3,"request_timeout_seconds":600}},
}
REASONING = [("Automatic","auto"),("Off","off"),("On","on"),("Low effort","low"),("Medium effort","medium"),("High effort","high"),("Custom options","custom")]


def reasoning_value(options):
    value = options.get("think")
    if value is None:
        return "custom" if options.get("reasoning") or options.get("thinking_config") else "auto"
    if value is True:
        return "on"
    if value is False:
        return "off"
    return str(value) if str(value) in {"low","medium","high"} else "custom"


def set_reasoning(options, mode):
    options = deepcopy(options)
    if mode == "custom":
        return options
    for key in ("think","reasoning","thinking_config"):
        options.pop(key,None)
    if mode != "auto":
        options["think"] = {"on":True,"off":False}.get(mode,mode)
    return options


def switch_profile(config, provider):
    result = deepcopy(config)
    current = result.get("llm",{}).get("provider","ollama")
    profiles = result.setdefault("studio",{}).setdefault("provider_profiles",{})
    profiles[current] = deepcopy(result.get("llm",{}).get("provider_config",DEFAULTS[current]))
    result["llm"] = {"provider":provider,"provider_config":deepcopy(profiles.get(provider,DEFAULTS[provider]))}
    return result
