"""MCP stdio adapter. Native tools and MCP share the same Tool contract.

Only explicitly configured server commands are launched. Each server is
conversation-owned; unexpected transport failures are returned to the agent.
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
from typing import Any

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool
from src.tools.contract import normalize_tool_definition

_NAME = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")


class MCPTransportError(RuntimeError):
    pass


class StdioMCPServer:
    def __init__(self, name: str, command: list[str], *, cwd: str | None = None,
                 env: dict[str, str] | None = None, timeout: float = 15):
        if not command or not all(isinstance(x, str) and x for x in command):
            raise ValueError("MCP command must be a nonempty string array")
        self.name, self.command, self.cwd = name, list(command), cwd
        self.env = dict(env or {})
        self.timeout = max(1.0, min(120.0, float(timeout)))
        self.process = None
        self.events: queue.Queue = queue.Queue()
        self.lock = threading.RLock()
        self.counter = 0

    def start(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                return
            self.close()
            self.events = queue.Queue()
            self.process = subprocess.Popen(
                self.command, cwd=self.cwd, env={**os.environ, **self.env},
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, bufsize=0,
            )
            proc = self.process

            def read_lines():
                try:
                    for line in proc.stdout:
                        try:
                            message = json.loads(line)
                        except (UnicodeDecodeError, ValueError):
                            continue
                        if isinstance(message, dict):
                            self.events.put(message)
                finally:
                    self.events.put({"_transport_closed": True})

            threading.Thread(target=read_lines, daemon=True,
                             name=f"mcp-{self.name}-reader").start()
            self.request("initialize", {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "daena", "version": "0.6"},
            })
            self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _send(self, message):
        if self.process is None or self.process.poll() is not None:
            raise MCPTransportError(f"MCP server {self.name} is not running")
        try:
            self.process.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode())
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MCPTransportError("MCP server disconnected") from exc

    def request(self, method: str, params: dict[str, Any] | None = None):
        with self.lock:
            self.counter += 1
            call_id = self.counter
            self._send({"jsonrpc": "2.0", "id": call_id,
                        "method": method, "params": params or {}})
            # Notifications can arrive while a request is pending.
            import time
            deadline = time.monotonic() + self.timeout
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    self.close()
                    raise MCPTransportError(f"MCP {method} timed out")
                try:
                    data = self.events.get(timeout=left)
                except queue.Empty as exc:
                    self.close()
                    raise MCPTransportError(f"MCP {method} timed out") from exc
                if data.get("_transport_closed"):
                    raise MCPTransportError("MCP server closed the transport")
                if data.get("id") != call_id:
                    # Notifications and unrelated responses are not request results.
                    continue
                if "error" in data:
                    raise MCPTransportError(f"MCP {method} failed: {data['error']}")
                return data.get("result") or {}

    def list_tools(self) -> list[dict]:
        self.start()
        items, cursor = [], None
        seen_cursors = set()
        for _page in range(32):
            result = self.request("tools/list", {"cursor": cursor} if cursor else {})
            items.extend(result.get("tools", []))
            cursor = result.get("nextCursor")
            if not cursor:
                return items
            if cursor in seen_cursors or len(items) > 512:
                raise MCPTransportError("MCP tool catalog pagination exceeded bounds")
            seen_cursors.add(cursor)
        raise MCPTransportError("MCP tool catalog exceeded 32 pages")

    def call(self, name: str, args: dict):
        self.start()
        return self.request("tools/call", {"name": name, "arguments": args})

    def close(self):
        proc, self.process = self.process, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
        finally:
            for handle in (proc.stdin, proc.stdout):
                if handle is not None:
                    try:
                        handle.close()
                    except OSError:
                        pass


class MCPTool(Tool):
    action = "execute"
    model_exposure = "search"

    def __init__(self, server: StdioMCPServer, remote: dict):
        original_name = remote["name"]
        self.remote_name = original_name
        self.server = server
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", original_name)
        self.name = f"mcp_{server.name}_{safe_name}"
        self.description = str(remote.get("description", ""))[:2000]
        definition = normalize_tool_definition({
            "name": self.name, "description": self.description,
            "inputSchema": remote.get("inputSchema", {"type": "object"}),
        })
        self.parameters = definition["function"]["parameters"]

    def execute(self, **kwargs) -> ToolResult:
        try:
            result = self.server.call(self.remote_name, kwargs)
            error = bool(result.get("isError", False))
            return ToolResult(success=not error, name=self.name,
                              content={"result": result},
                              summary=f"MCP {self.remote_name} {'failed' if error else 'completed'}.")
        except Exception as exc:
            return ToolResult(success=False, name=self.name,
                              content={"error": {"type": "mcp_transport",
                                                 "message": str(exc)}})


def configured_servers(config: dict) -> list[StdioMCPServer]:
    servers = (config.get("mcp") or {}).get("servers") or {}
    if not isinstance(servers, dict):
        raise ValueError("mcp.servers must be an object")
    result = []
    for name, settings in servers.items():
        if not _NAME.fullmatch(name):
            raise ValueError(f"Invalid MCP server name: {name!r}")
        if not isinstance(settings, dict):
            raise ValueError(f"Invalid MCP server settings: {name}")
        if not settings.get("enabled", False):
            continue
        result.append(StdioMCPServer(
            name, settings["command"], cwd=settings.get("cwd"),
            env=settings.get("env"), timeout=settings.get("timeout_seconds", 15)))
    return result
