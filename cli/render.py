from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _stringify(value: Any, limit: int = 900) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, dict):
        parts = []
        for key, item in value.items():
            parts.append(f"{key}={item!r}")
        text = " ".join(parts)
    else:
        text = repr(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


@dataclass
class StreamRenderer:
    model: str
    workspace: str
    session_id: str
    think_enabled: Any = True
    iteration: int = 0
    max_iterations: int = 0
    phase: str = "idle"
    status: str = "ready"
    thinking: str = ""
    response: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)
    usage: int = 0
    final_text: str = ""
    error: str = ""

    def handle(self, event: dict[str, Any]) -> None:
        event_type = event.get("type", "")

        if event_type == "run_start":
            self.status = "running"
            self.workspace = str(event.get("workspace") or self.workspace)
            self.max_iterations = int(event.get("max_iterations") or 0)
            self.session_id = str(event.get("session_id") or self.session_id)
        elif event_type == "iteration_start":
            self.iteration = int(event.get("iteration") or 0)
            self.max_iterations = int(event.get("max_iterations") or self.max_iterations)
            self.phase = "thinking"
            self.thinking = ""
            self.response = ""
            self.status = "thinking"
        elif event_type == "context":
            self.context = dict(event)
        elif event_type == "thinking_delta":
            self.phase = "thinking"
            self.status = "thinking"
            self.thinking += str(event.get("text") or "")
        elif event_type == "content_delta":
            self.phase = "responding"
            self.status = "responding"
            self.response += str(event.get("text") or "")
        elif event_type == "generation_done":
            self.usage = int(event.get("usage") or self.usage or 0)
        elif event_type == "tool_call":
            item = {
                "name": str(event.get("name") or "unknown"),
                "arguments": event.get("arguments") or {},
                "success": None,
                "summary": "",
            }
            self.tools.append(item)
            self.tools = self.tools[-8:]
            self.phase = "tool"
            self.status = f"tool: {item['name']}"
        elif event_type == "tool_result":
            name = str(event.get("name") or "unknown")
            for item in reversed(self.tools):
                if item["name"] == name and item["success"] is None:
                    item["success"] = bool(event.get("success"))
                    item["summary"] = str(event.get("summary") or "")
                    break
            self.phase = "tool"
            self.status = f"tool result: {name}"
        elif event_type == "final_response":
            self.final_text = str(event.get("text") or "")
            self.usage = int(event.get("usage") or self.usage or 0)
            self.phase = "complete"
            self.status = "completed"
        elif event_type == "run_stopped":
            self.phase = "stopped"
            self.status = str(event.get("reason") or "stopped")
        elif event_type == "run_end":
            self.status = (
                "completed"
                if bool(event.get("completed"))
                else str(event.get("stop_reason") or "stopped")
            )
        elif event_type == "error":
            self.error = str(event.get("message") or "unknown error")
            self.status = self.error

    def _header(self) -> Panel:
        table = Table.grid(expand=True)
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_column(style="bold magenta", no_wrap=True)
        table.add_column(style="white")
        table.add_row("DAENA", self.model, "STATUS", self.status)
        table.add_row("WORKSPACE", self.workspace, "ITERATION", f"{self.iteration}/{self.max_iterations or '?'}")
        table.add_row("THINK", str(self.think_enabled).lower(), "SESSION", self.session_id[:8])
        return Panel(table, border_style="cyan", padding=(0, 1))

    def _thinking_panel(self) -> Panel:
        text = self.thinking[-12000:] if self.thinking else "Waiting for model reasoning..."
        return Panel(
            Text(text, style="bright_black" if self.thinking else "dim"),
            title="[bold yellow]thinking[/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
        )

    def _tools_panel(self) -> Panel:
        lines = []
        for item in self.tools:
            icon = "✓" if item["success"] is True else "…" if item["success"] is None else "✗"
            args = _stringify(item["arguments"], 650)
            summary = item["summary"]
            line = f"{icon} [bold cyan]{item['name']}[/bold cyan] {args}"
            if summary:
                line += f"  [dim]→ {_stringify(summary, 420)}[/dim]"
            lines.append(line)
        body = "\n".join(lines) if lines else "No tool calls yet."
        return Panel(body, title="[bold blue]tool activity[/bold blue]", border_style="blue", padding=(0, 1))

    def _response_panel(self) -> Panel:
        body = self.response or "No assistant output yet."
        return Panel(body[-9000:], title="[bold green]assistant[/bold green]", border_style="green", padding=(0, 1))

    def render(self) -> Group:
        return Group(
            self._header(),
            self._thinking_panel(),
            self._tools_panel(),
            self._response_panel(),
        )
