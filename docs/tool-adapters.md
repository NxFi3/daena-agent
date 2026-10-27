# Tool adapters

Native tools and host-owned adapters share one execution contract. This keeps
future MCP transports behind the same schema validation, allowlist, network
policy, cancellation and cleanup boundaries as built-in tools.

## Registering an adapter

Implement `src.tools.Tool.Tool`, give it a stable lowercase name, a valid JSON
Schema `parameters` object and an `execute(**kwargs)` method returning
`ToolResult`. Register it through `ToolManager.register_tool(adapter)` during
host setup, before a run starts. Registration rejects invalid schemas and name
collisions. It does not approve the tool: add its name to the host's ordinary
security allowlist if it should be callable.

Set `requires_network = True` for adapters that can make network requests.
Host network policy remains authoritative, including for calls submitted
directly as `ToolCall` objects. Implement `set_stop_event(event)` when the
operation can be cancelled. Implement `close()` to release adapter-owned
resources; cleanup attempts every registered resource even when one close
operation fails.

Tool schemas are validated locally before execution, including nested objects,
unions, bounds, local `$defs` and `patternProperties`. Remote schema references
are rejected, so validating an adapter schema never fetches an external URL.
Unknown fields are preserved when the schema allows them. Integer range
clamping is disabled by default; explicitly add `x-daena-clamp: true` to a
bounded integer property only when changing an out-of-range request to the
nearest bound is safe for that operation. Numeric strings may be converted to
integers as a transport normalization, with the normalization reported on the
tool call.

## MCP integration point

An MCP host can expose a remote server as a `Tool` adapter and register it with
the same method. The adapter should translate the validated Daena arguments to
the MCP call, map the result to `ToolResult`, propagate the stop event to the
transport when supported, and close its client/session in `close()`. Tool
discovery and execution must remain host-controlled; remote tools do not gain
trust, filesystem access or network access merely by registering. MCP package
selection, server configuration, credential storage and transport lifecycle
belong in the host adapter layer, not in the generic tool dispatcher.

Do not expose the raw MCP client or server prompts as trusted system context.
If an adapter retrieves reference material for future memory/context use, pass
it through the bounded `ContextContribution` interface documented in
[`context-extensions.md`](context-extensions.md), where it remains untrusted and
request-scoped.
