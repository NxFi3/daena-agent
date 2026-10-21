"""Cancellation and deadlines shared by asynchronous provider transports."""

from __future__ import annotations

import asyncio
import math
import time


def emit(callback, kind, **payload):
    if callable(callback):
        try:
            callback({"type": kind, **payload})
        except Exception:
            pass


def seconds(options, name, default=None):
    value = options.pop(name, default)
    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite positive seconds or null")
    return float(value)


async def supervise(coroutine, stop=None, timeout=None, callback=None):
    request = asyncio.create_task(coroutine)
    started = time.monotonic()
    heartbeat = started + 2
    try:
        while not request.done():
            if stop is not None and stop.is_set():
                raise InterruptedError("Generation cancelled")
            now = time.monotonic()
            if timeout is not None and now - started >= timeout:
                raise TimeoutError(
                    f"Generation exceeded total_timeout_seconds={timeout:g}; no partial tools executed"
                )
            if now >= heartbeat:
                emit(
                    callback, "generation_wait", elapsed_seconds=round(now - started, 1)
                )
                heartbeat = now + 2
            await asyncio.wait({request}, timeout=0.05)
        if stop is not None and stop.is_set():
            raise InterruptedError("Generation cancelled")
        return await request
    finally:
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)


def canonical_messages(messages):
    """Convert internal annotations to the Chat Completions wire contract."""
    output = []
    fields = {
        "role",
        "content",
        "tool_calls",
        "tool_call_id",
        "name",
        "reasoning",
        "reasoning_details",
    }
    for message in messages:
        item = {k: v for k, v in message.items() if k in fields}
        if message.get("role") == "tool" and message.get("tool_name"):
            item["name"] = message["tool_name"]
        output.append(item)
    return output


def remote_options(options):
    """Shared local settings never leak into remote provider requests."""
    options = dict(options)
    if "num_predict" in options and "max_tokens" not in options:
        options["max_tokens"] = options["num_predict"]
    for key in (
        "num_ctx",
        "num_thread",
        "num_threads",
        "num_predict",
        "keep_alive",
        "seed_local",
    ):
        options.pop(key, None)
    return options
