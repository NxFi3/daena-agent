"""Provider invocation with explicit support for older embedding adapters."""

from inspect import Parameter, signature


def invoke(llm, messages, **kwargs):
    stop = kwargs.get("stop_event")
    if stop is not None and stop.is_set():
        raise InterruptedError("Generation cancelled")
    parameters = signature(llm.generate).parameters
    if not any(item.kind == Parameter.VAR_KEYWORD for item in parameters.values()):
        kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    result = llm.generate(messages, **kwargs)
    if stop is not None and stop.is_set():
        raise InterruptedError("Generation cancelled")
    return result
