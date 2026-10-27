# Context budgeting

The request limit is the smaller of `context.max_prompt_tokens` and the model
context window minus safety margin and output reserve. The output reserve is the
larger of `context.output_reserve_tokens` and the provider's output cap (for
Ollama, `num_predict`). Increasing only the prompt cap cannot exceed that limit.

With a 32768 context, an 8192 output reserve and the default 15% safety margin,
input has an estimated 19661-token budget. This includes message text, complete
tool schemas and a configurable allowance per image (4096 by default). Text and
schema estimates adapt to reported provider usage, so their estimated cost can
increase during a conversation even when settings stay unchanged. Image usage
is excluded from this calibration because providers do not consistently separate
visual and text costs.

When a request cannot fit, the runtime can retry context construction once with
fewer complete tool schemas. The active UI tool, tool discovery, explicit
completion tool and tools needed for outstanding processes/subagents are retained.
Other permitted capabilities remain advertised and discoverable with `tool_search`.
This changes the definitions supplied for that model request, not tool permissions.
It does not execute an action or discard the current screenshot or user task.

If mandatory content still cannot fit, `ContextBudgetError` reports the estimated
text, image and schema costs, input budget, context window and output reserve.
Use these values to identify the constraint. Increase the supported model context
if resources permit, or reduce both the provider output cap and configured reserve
when shorter responses are acceptable. Do not lower image estimates merely to
hide an overflow; actual visual-token costs depend on the provider and model.
