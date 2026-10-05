from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import ANSI


def _short(value: Any, limit: int = 480) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = " ".join(str(text).split())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


@dataclass
class StreamRenderer:
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
    response_started: bool = False

    def handle(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")

        if event_type == "run_start":
            self.status = "working"
            self.max_iterations = int(event.get("max_iterations") or self.max_iterations)
            self.session_id = str(event.get("session_id") or self.session_id)
            self._print(
                f"\x1b[36m◆ Daena\x1b[0m  "
                f"\x1b[90m{self.model} · {self.workspace}\x1b[0m"
            )
        elif event_type == "iteration_start":
            self.iteration = int(event.get("iteration") or 0)
            self.max_iterations = int(event.get("max_iterations") or self.max_iterations)
            self.status = "thinking"
        elif event_type == "thinking_delta":
            # Do not render private/internal chain-of-thought. The CLI exposes
            # a live reasoning state via the prompt toolbar instead.
            self.status = "thinking"
        elif event_type == "content_delta":
            self.status = "responding"
            text = str(event.get("text") or "")
            if text:
                if not self.response_started:
                    self.response_started = True
                    self._print("\n\x1b[32mDaena ›\x1b[0m ")
                self._print(text)
        elif event_type == "generation_done":
            self.usage = int(event.get("usage") or self.usage or 0)
        elif event_type == "tool_call":
            self.tool_count += 1
            name = _short(event.get("name") or "unknown", 80)
            args = _short(event.get("arguments") or {}, 520)
            self.status = f"tool:{name}"
            self._print(f"\x1b[34m↳ {name}\x1b[0m  \x1b[90m{args}\x1b[0m")
        elif event_type == "tool_result":
            name = _short(event.get("name") or "unknown", 80)
            success = bool(event.get("success"))
            if success:
                self.success_count += 1
                icon = "\x1b[32m✓\x1b[0m"
            else:
                self.failure_count += 1
                icon = "\x1b[31m✗\x1b[0m"
            summary = _short(event.get("summary") or event.get("content") or "", 520)
            self.status = f"tool result:{name}"
            self._print(f"{icon} {name}" + (f"  \x1b[90m— {summary}\x1b[0m" if summary else ""))
        elif event_type == "steering":
            self._print(f"\x1b[35m↪ redirect queued\x1b[0m  {_short(event.get('text') or '', 700)}")
        elif event_type == "final_response":
            self.usage = int(event.get("usage") or self.usage or 0)
            self.status = "completed"
            if self.response_started:
                self._print("\n")
        elif event_type == "run_stopped":
            self.status = "stopped"
            self._print(f"\x1b[33m■ stopped\x1b[0m  {_short(event.get('reason') or '', 700)}")
        elif event_type == "error":
            self.status = "error"
            self._print(f"\x1b[31m✗ error\x1b[0m  {_short(event.get('message') or '', 900)}")

    def toolbar(self) -> str:
        phase = self.status
        if self.max_iterations:
            iteration = f"{self.iteration}/{self.max_iterations}"
        else:
            iteration = str(self.iteration)
        return (
            f"  {phase} · iter {iteration} · tools {self.tool_count} "
            f"(✓{self.success_count} ✗{self.failure_count}) · think {str(self.think_enabled).lower()} "
            f"· {self.model}  "
        )

    @staticmethod
    def _print(text: str) -> None:
        print_formatted_text(ANSI(text))
