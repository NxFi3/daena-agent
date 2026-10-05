from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import ANSI, HTML


def _short(value: Any, limit: int = 500) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = " ".join(str(text).split())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _tool_args(value: Any, limit: int = 260) -> str:
    if not isinstance(value, dict) or not value:
        return ""
    parts = []
    for key, item in value.items():
        if isinstance(item, str):
            shown = repr(item)
        elif isinstance(item, (int, float, bool)) or item is None:
            shown = str(item)
        else:
            shown = repr(item)
        parts.append(f"{key}={shown}")
    return _short(" · ".join(parts), limit)


@dataclass
class StreamRenderer:
    """Compact terminal renderer for one live Agent run.

    Raw model reasoning is intentionally not printed. We expose reasoning as
    a live state in the terminal and keep tool activity visible inline.
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
    context_tokens: int = 0
    context_budget: int = 0

    _content_buffer: str = ""
    _thinking_seen: bool = False
    _thinking_chars: int = 0
    _last_reasoning_notice: float = 0.0
    _reasoning_phase: str = "analyzing context"
    _finished_rendered: bool = False

    def show_user_context(self, workspace: str) -> None:
        """Kept for compatibility; workspace is shown once in the session header."""
        return

    def _print_status(self, state: str) -> None:
        if self.context_budget:
            percent = (self.context_tokens / self.context_budget) * 100.0
            context = (
                f"{self._fmt_tokens(self.context_tokens)}/"
                f"{self._fmt_tokens(self.context_budget)} ({percent:.1f}%)"
            )
        else:
            context = self._fmt_tokens(self.context_tokens)
        self._print(
            f"\x1b[90m   · {state} · ctx {context}\x1b[0m"
        )

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
            self._thinking_chars = 0
            self._last_reasoning_notice = 0.0
            self._reasoning_phase = "analyzing context"

        elif event_type == "context":
            self.context_tokens = int(event.get("estimated_tokens") or 0)
            self.context_budget = int(event.get("budget") or 0)
            self._print_status("thinking…")

        elif event_type == "thinking_delta":
            self.status = "thinking"
            delta = str(event.get("text") or "")
            self._thinking_chars += len(delta)

            now = time.monotonic()
            if not self._thinking_seen:
                self._thinking_seen = True
                self._last_reasoning_notice = now
                self._print(
                    f"\x1b[90mDAENA · reasoning · {self._reasoning_phase}…\x1b[0m"
                )
            elif now - self._last_reasoning_notice >= 0.75:
                self._last_reasoning_notice = now
                self._print(
                    f"\x1b[90mDAENA · reasoning · {self._reasoning_phase}… "
                    f"{self._thinking_chars:,} chars processed\x1b[0m"
                )

        elif event_type == "content_delta":
            self._reasoning_phase = "forming response"
            if self.status != "responding":
                self._print("\x1b[90m   · responding…\x1b[0m")
                self._print(f"\x1b[36mDAENA {self.model} ›\x1b[0m")
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
            self._reasoning_phase = f"choosing {str(event.get('name') or 'tool')}"
            self.tool_count += 1
            self.status = f"tool:{_short(event.get('name') or 'unknown', 70)}"
            name = _short(event.get("name") or "unknown", 70)
            args = _tool_args(event.get("arguments") or {})
            line = f"\x1b[34mDAENA ↳ {name}\x1b[0m"
            if args:
                line += f"  \x1b[90m{args}\x1b[0m"
            self._print(line)

        elif event_type == "tool_result":
            self._flush_content(force=True)
            self._reasoning_phase = "evaluating tool result"
            name = _short(event.get("name") or "unknown", 70)
            success = bool(event.get("success"))
            if success:
                self.success_count += 1
                icon = "\x1b[36m✓\x1b[0m"
            else:
                self.failure_count += 1
                icon = "\x1b[31m✗\x1b[0m"
            self.status = f"result:{name}"
            summary = _short(
                event.get("summary") or event.get("content") or "",
                600,
            )
            line = f"DAENA {icon} {name}"
            if summary:
                line += f"  \x1b[90m— {summary}\x1b[0m"
            self._print(line)

        elif event_type == "steering":
            self._flush_content(force=True)
            self.status = "redirected"
            self._print(
                f"\x1b[35mYOU ↪ redirect\x1b[0m  "
                f"\x1b[90m{_short(event.get('text') or '', 700)}\x1b[0m"
            )

        elif event_type == "final_response":
            self._flush_content(force=True)
            self.usage = int(event.get("usage") or self.usage or 0)
            response = str(event.get("text") or "")
            if response and not self.response_streamed:
                self._print(f"\x1b[36mDAENA {self.model} ›\x1b[0m")
                self._print(f"\x1b[37m{response}\x1b[0m")
                self.response_streamed = True
            self.status = "completed"

        elif event_type == "run_stopped":
            self._flush_content(force=True)
            self.status = "stopped"
            self._print(
                f"\x1b[33mDAENA ■ stopped\x1b[0m  "
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
                f"\x1b[31mDAENA ✗ error\x1b[0m  "
                f"\x1b[90m{_short(event.get('message') or '', 900)}\x1b[0m"
            )

    def _flush_complete_lines(self) -> None:
        while "\n" in self._content_buffer:
            line, self._content_buffer = self._content_buffer.split("\n", 1)
            self._print(f"\x1b[37m{line}\x1b[0m")

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
        self._print(f"\x1b[37m{text}\x1b[0m")

    def finish(self, result: Any = None, error: BaseException | None = None) -> None:
        """Render the turn tail immediately from the worker thread.

        This avoids waiting for the next user Enter just to display a completed
        response when the main thread is blocked inside PromptSession.prompt().
        """
        if self._finished_rendered:
            return

        response = getattr(result, "response", None) if result is not None else None
        if response and not self.response_streamed:
            self._print(f"\x1b[36mDAENA {self.model} ›\x1b[0m")
            self._print(f"\x1b[37m{str(response)}\x1b[0m")
            self.response_streamed = True

        if error is not None:
            state = "failed"
        elif self.status == "stopped":
            state = "stopped"
        else:
            state = "done"

        self._print(
            f"\x1b[90mDAENA · {state} · iter {self.iteration} · "
            f"tools {self.tool_count} · usage {self.usage}\x1b[0m"
        )
        self._finished_rendered = True

    @staticmethod
    def _fmt_tokens(value: int) -> str:
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if value >= 1_000:
            return f"{value / 1_000:.1f}k"
        return str(value)

    @staticmethod
    def _progress(current: int, maximum: int, width: int = 10) -> str:
        if maximum <= 0:
            return "──────────"
        ratio = max(0.0, min(1.0, current / maximum))
        filled = int(round(ratio * width))
        return "━" * filled + "─" * (width - filled)

    def toolbar(self) -> str:
        if self.status == "thinking":
            state = "thinking…"
        elif self.status == "responding":
            state = "responding…"
        elif self.status.startswith("tool:"):
            state = self.status[5:]
        elif self.status.startswith("result:"):
            state = self.status[7:]
        elif self.status == "redirected":
            state = "redirected"
        elif self.status == "completed":
            state = "ready"
        else:
            state = self.status

        iteration = (
            f"{self.iteration}/{self.max_iterations}"
            if self.max_iterations
            else str(self.iteration)
        )
        ctx = (
            f"{self._fmt_tokens(self.context_tokens)}/"
            f"{self._fmt_tokens(self.context_budget)}"
            if self.context_budget
            else self._fmt_tokens(self.context_tokens)
        )
        progress = self._progress(self.context_tokens, self.context_budget)

        return HTML(
            f"<ansimagenta>  ◆</ansimagenta> "
            f"<ansiwhite><b>{state}</b></ansiwhite>  "
            f"<ansiblue>{progress}</ansiblue>  "
            f"<ansiwhite>ctx {ctx}</ansiwhite> · "
            f"<ansiwhite>iter {iteration}</ansiwhite> · "
            f"<ansiwhite>tools {self.tool_count}</ansiwhite> "
            f"<ansicyan>✓{self.success_count}</ansicyan> "
            f"<ansired>✗{self.failure_count}</ansired> · "
            f"<ansigray>{self.model} · think={str(self.think_enabled).lower()}</ansigray>  "
        )

    @staticmethod
    def _print(text: str) -> None:
        print_formatted_text(ANSI(text))