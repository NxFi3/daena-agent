# MCP stdio integration (experimental)

Daena's built-in tools and MCP tools use the same JSON Schema tool contract,
ToolDispatcher validation, ToolResult envelope and agent tool-search catalog.
An MCP server is a separate executable that can read data or perform actions
with the permissions of the current OS user. Install and trust servers
individually. **Do not install untrusted MCP packages.**

## Configure a server

Install the MCP server executable separately, then add to your local Daena
configuration file (not to a committed file containing credentials):

```json
{
  "mcp": {
    "servers": {
      "example": {
        "enabled": true,
        "command": ["python", "-m", "your_installed_mcp_server"],
        "timeout_seconds": 15
      }
    }
  }
}
```

Merge this object with the existing configuration rather than replacing it.
Optional `cwd` sets the server's working directory and optional `env` is a
map of environment variables (avoid committing secrets).

Restart Daena to load new server definitions. An enabled server launches when
the tool catalog is first requested; Daena initializes MCP, requests tools/list
and registers each tool under `mcp_<server>_<tool>`. The tool becomes
discoverable through `tool_search`; it does not automatically run.
The existing `security.allow_network_tools` control must be true. Explicit
`security.allowed_tools` lists must include `mcp_<server>_*` to authorize that server namespace;
Daena Studio adds this opt-in when an enabled server is saved. The wildcard
is resolved to known tool names at discovery, not used for arbitrary calls.

This implementation supports **stdio transport only**. Daena Studio offers a
JSON configuration editor, not a package installer. Streamable HTTP,
package install/uninstall UI, live reconfiguration, OAuth, roots, sampling,
elicitation, server-initiated tools/list changes, and full protocol lifecycle
coverage are **not implemented**. Stdio processes are terminated on agent
shutdown. A server crash is reported as a tool error, not masked as success.
Remote tools can perform actions outside Daena's workspace sandbox; run
trusted servers with least privilege.

Run `python -m pytest -q tests/test_tool_contract.py tests/test_mcp_stdio.py`
for contract and local transport tests.
