from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import ANSI


def _short(value: Any, limit: int = 500) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = " ".join(str(text).split())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


@dataclass
class StreamRenderer:
    """Compact terminal renderer for one live Agent run.

    Raw model reasoning is intentionally not printed. We expose reasoning as
    a live state in the bottom toolbar and keep tool activity visible inline.
    """

    model: str
    workspace: str
    session_id: str
    think_enabled: Any = "medium"

    iteration: int = 0
    max_iterations: int = 0
    status: str = "ready"
    tool_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    usage: int = 0
    response_streamed: bool = False

    _content_buffer: str = ""
    _thinking_seen: bool = False

    def handle(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")

        if event_type == "run_start":
            self.status = "starting"
            self.max_iterations = int(
                event.get("max_iterations") or self.max_iterations
            )
            self.session_id = str(event.get("session_id") or self.session_id)

        elif event_type == "iteration_start":
            self.iteration = int(event.get("iteration") or 0)
            self.max_iterations = int(
                event.get("max_iterations") or self.max_iterations
            )
            self.status = "thinking"
            self._thinking_seen = False

        elif event_type == "thinking_delta":
            self.status = "thinking"
            if not self._thinking_seen:
                self._thinking_seen = True

        elif event_type == "content_delta":
            self.status = "responding"
            delta = str(event.get("text") or "")
            if delta:
                self.response_streamed = True
                self._content_buffer += delta
                self._flush_complete_lines()

        elif event_type == "generation_done":
            self.usage = int(event.get("usage") or self.usage or 0)

        elif event_type == "tool_call":
            self._flush_content(force=True)
            self.tool_count += 1
            self.status = f"tool:{_short(event.get('name') or 'unknown', 70)}"
            self._print(
                f"\x1b[34m↳ {_short(event.get('name') or 'unknown', 70)}\x1b[0m"
                f"  \x1b[90m{_short(event.get('arguments') or {}, 560)}\x1b[0m"
            )

        elif event_type == "tool_result":
            self._flush_content(force=True)
            name = _short(event.get("name") or "unknown", 70)
            success = bool(event.get("success"))
            if success:
                self.success_count += 1
                icon = "\x1b[32m✓\x1b[0m"
            else:
                self.failure_count += 1
                icon = "\x1b[31m✗\x1b[0m"
            self.status = f"result:{name}"
            summary = _short(
                event.get("summary") or event.get("content") or "",
                600,
            )
            line = f"{icon} {name}"
            if summary:
                line += f"  \x1b[90m— {summary}\x1b[0m"
            self._print(line)

        elif event_type == "steering":
            self._flush_content(force=True)
            self.status = "redirected"
            self._print(
                f"\x1b[35m↪ redirect\x1b[0m  "
                f"\x1b[90m{_short(event.get('text') or '', 700)}\x1b[0m"
            )

        elif event_type == "final_response":
            self._flush_content(force=True)
            self.usage = int(event.get("usage") or self.usage or 0)
            self.status = "completed"

        elif event_type == "run_stopped":
            self._flush_content(force=True)
            self.status = "stopped"
            self._print(
                f"\x1b[33m■ stopped\x1b[0m  "
                f"\x1b[90m{_short(event.get('reason') or '', 800)}\x1b[0m"
            )

        elif event_type == "run_end":
            self._flush_content(force=True)
            self.status = (
                "completed"
                if bool(event.get("completed"))
                else str(event.get("stop_reason") or "stopped")
            )

        elif event_type == "error":
            self._flush_content(force=True)
            self.status = "error"
            self._print(
                f"\x1b[31m✗ error\x1b[0m  "
                f"\x1b[90m{_short(event.get('message') or '', 900)}\x1b[0m"
            )

    def _flush_complete_lines(self) -> None:
        while "\n" in self._content_buffer:
            line, self._content_buffer = self._content_buffer.split("\n", 1)
            self._print(f"\x1b[32m{line}\x1b[0m")

        # Prevent a very long line from sitting in the buffer indefinitely.
        if len(self._content_buffer) >= 180:
            self._flush_content(force=True)

    def _flush_content(self, force: bool = False) -> None:
        if not self._content_buffer:
            return
        if not force and "\n" not in self._content_buffer:
            return

        text = self._content_buffer
        self._content_buffer = ""
        self._print(f"\x1b[32m{text}\x1b[0m")

    def toolbar(self) -> str:
        iteration = (
            f"{self.iteration}/{self.max_iterations}"
            if self.max_iterations
            else str(self.iteration)
        )
        return (
            f" {self.status} · iter {iteration} · "
            f"tools {self.tool_count} "
            f"(✓{self.success_count} ✗{self.failure_count}) · "
            f"think {str(self.think_enabled).lower()} · {self.model} "
        )

    @staticmethod
    def _print(text: str) -> None:
        print_formatted_text(ANSI(text))
