"""Daena – streaming terminal renderer.

Reasoning chunks stream live inside a readable boxed ╭─ thinking ─╮ block.
Response chunks appear under a slim  ◆  daena  ──  model  header.
Tool events, errors, and the finish summary use the same blue-purple palette
with rose-red (#FB7185) reserved for failures and the run-end footer accent.
"""

from __future__ import annotations

import html
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import ANSI, HTML

_P    = "\x1b[38;2;167;139;250m"
_P2   = "\x1b[38;2;139;92;246m"
_B    = "\x1b[38;2;96;165;250m"
_TH   = "\x1b[38;2;109;101;156m"
_THD  = "\x1b[38;2;130;110;200m"
_RESP = "\x1b[38;2;219;234;254m"
_TOOL = "\x1b[38;2;99;179;237m"
_OK   = "\x1b[38;2;74;222;128m"
_FAIL = "\x1b[38;2;251;113;133m"
_ERR  = "\x1b[38;2;252;165;165m"
_SEP  = "\x1b[38;2;79;64;124m"
_DIM  = "\x1b[38;2;107;114;128m"
_MUT  = "\x1b[38;2;71;85;105m"
_R    = "\x1b[0m"
_BD   = "\x1b[1m"
_W = 68

def _sep(char: str = "─", w: int = _W, col: str = _SEP) -> str:
    return f"{col}{char * w}{_R}"

def _short(v: Any, lim: int = 500) -> str:
    s = v if isinstance(v, str) else repr(v)
    s = " ".join(str(s).split())
    return (s[: lim - 1] + "…") if len(s) > lim else s

def _tool_args(v: Any, lim: int = 240) -> str:
    if not isinstance(v, dict) or not v:
        return ""
    parts = [
        f"{k}={repr(i) if isinstance(i, str) else str(i)}"
        for k, i in v.items()
    ]
    return _short("  ".join(parts), lim)

def _fmt_tok(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)

def _bar(cur: int, tot: int, w: int = 12) -> str:
    if tot <= 0:
        return "─" * w
    r = max(0.0, min(1.0, cur / tot))
    f = int(round(r * w))
    return "━" * f + "─" * (w - f)

@dataclass
class StreamRenderer:
    """Polished blue-purple streaming renderer for a single Daena agent run."""
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
    context_tokens: int = 0
    context_budget: int = 0
    response_streamed: bool = False

    _content_buf: str = field(default="", repr=False)
    _think_buf: str = field(default="", repr=False)
    _think_open: bool = field(default=False, repr=False)
    _resp_open: bool = field(default=False, repr=False)
    _finished: bool = field(default=False, repr=False)
    _start_ts: float = field(default_factory=time.monotonic, repr=False)
    _finished_rendered: bool = field(default=False, repr=False)

    def begin_user_message(self, text: str = "") -> None:
        pass

    def show_user_context(self, workspace: str) -> None:
        return

    def handle(self, event: dict[str, Any]) -> None:
        t = str(event.get("type") or "")

        if t == "run_start":
            self.status = "starting"
            self.max_iterations = int(event.get("max_iterations") or self.max_iterations)
            self.session_id = str(event.get("session_id") or self.session_id)
            self._start_ts = time.monotonic()
            self._print(_sep())

        elif t == "iteration_start":
            self._close_thinking()
            self._close_response()
            self.iteration = int(event.get("iteration") or 0)
            self.max_iterations = int(event.get("max_iterations") or self.max_iterations)
            self.status = "thinking"

        elif t == "context":
            self.context_tokens = int(event.get("estimated_tokens") or 0)
            self.context_budget = int(event.get("budget") or 0)
            self._print(f"  {_MUT}ctx {self._ctx_str()}{_R}")

        elif t == "thinking_delta":
            self.status = "thinking"
            d = str(event.get("text") or "")
            if d:
                self._feed_think(d)

        elif t == "content_delta":
            self._close_thinking()
            if not self._resp_open:
                self._open_response()
            self.status = "responding"
            d = str(event.get("text") or "")
            if d:
                self.response_streamed = True
                self._content_buf += d
                self._flush_resp_lines()

        elif t == "generation_done":
            self.usage = int(event.get("usage") or self.usage or 0)

        elif t == "tool_call":
            self._close_thinking()
            self._close_response()
            self.tool_count += 1
            name = _short(str(event.get("name") or "unknown"), 32)
            args = _tool_args(event.get("arguments") or {})
            self.status = f"tool:{name}"
            row = f"\n  {_B}{_BD}↳{_R}  {_TOOL}{name:<26}{_R}"
            if args:
                row += f"  {_DIM}{args}{_R}"
            self._print(row)

        elif t == "tool_result":
            self._close_response()
            name = _short(str(event.get("name") or "unknown"), 32)
            ok = bool(event.get("success"))
            if ok:
                self.success_count += 1
                icon, col = f"{_OK}✓{_R}", _OK
            else:
                self.failure_count += 1
                icon, col = f"{_FAIL}✗{_R}", _FAIL
            summary = _short(event.get("summary") or event.get("content") or "", 380)
            self.status = f"result:{name}"
            row = f"  {icon}  {col}{name:<26}{_R}"
            if summary:
                row += f"  {_DIM}─  {summary}{_R}"
            self._print(row)

        elif t == "steering":
            self._close_thinking()
            self._close_response()
            self.status = "redirected"
            msg = _short(str(event.get("text") or ""), 600)
            self._print(f"\n  {_P}↪ redirect{_R}  {_DIM}{msg}{_R}")

        elif t == "final_response":
            self._close_thinking()
            self._close_response()
            self.usage = int(event.get("usage") or self.usage or 0)
            text = str(event.get("text") or "")
            if text and not self.response_streamed:
                self._open_response()
                for ln in text.splitlines():
                    self._print(f"  {_RESP}{ln}{_R}")
                self.response_streamed = True
            self.status = "completed"

        elif t == "run_stopped":
            self._close_thinking()
            self._close_response()
            self.status = "stopped"
            reason = _short(str(event.get("reason") or ""), 600)
            self._print(f"\n  {_FAIL}{_BD}■ stopped{_R}  {_DIM}{reason}{_R}")

        elif t == "run_end":
            self._close_thinking()
            self._close_response()
            self.status = (
                "completed"
                if bool(event.get("completed"))
                else str(event.get("stop_reason") or "stopped")
            )

        elif t == "error":
            self._close_thinking()
            self._close_response()
            self.status = "error"
            msg = _short(str(event.get("message") or ""), 700)
            self._print(f"\n  {_ERR}{_BD}✗ error{_R}  {_DIM}{msg}{_R}")

    def _feed_think(self, delta: str) -> None:
        if not self._think_open:
            mode = str(self.think_enabled or "").lower()
            suffix = f" · {mode}" if mode not in {"", "false", "none"} else ""
            label = f"─ thinking{suffix} "
            dashes = max(1, _W - len(label) - 2)
            self._print(f"\n  {_THD}╭{label}{'─' * dashes}╮{_R}")
            self._think_open = True

        self._think_buf += delta

        # Ollama's thinking stream is token-sized. Never print each token as
        # its own line: accumulate a readable window and flush at natural
        # whitespace boundaries while the model is still generating.
        while "\n" in self._think_buf:
            line, self._think_buf = self._think_buf.split("\n", 1)
            self._emit_think_line(line)

        flush_w = _W - 7
        while len(self._think_buf) >= flush_w:
            cut = self._think_buf.rfind(" ", 0, flush_w + 1)
            if cut < flush_w // 2:
                cut = flush_w
            chunk = self._think_buf[:cut].rstrip()
            self._think_buf = self._think_buf[cut:].lstrip()
            if chunk:
                self._emit_think_line(chunk)

        # Keep latency low for short reasoning updates. Once a sentence or a
        # reasonably sized phrase is complete, show it without waiting for a
        # full line window.
        stripped = self._think_buf.rstrip()
        if len(stripped) >= 32 and stripped[-1:] in ".!?:":
            self._emit_think_line(stripped)
            self._think_buf = ""

    def _emit_think_line(self, text: str) -> None:
        text = text.rstrip()
        if not text:
            return
        inner_w = _W - 5
        while len(text) > inner_w:
            self._print(f"  {_THD}│{_R}  {_TH}{text[:inner_w]}{_R}")
            text = text[inner_w:]
        if text:
            self._print(f"  {_THD}│{_R}  {_TH}{text}{_R}")

    def _close_thinking(self) -> None:
        if not self._think_open:
            return
        if self._think_buf.strip():
            self._emit_think_line(self._think_buf)
        self._think_buf = ""
        self._print(f"  {_THD}╰{'─' * (_W - 2)}╯{_R}")
        self._think_open = False

    def _open_response(self) -> None:
        if self._resp_open:
            return
        m = self.model if len(self.model) <= 28 else self.model[:25] + "…"
        gap = max(2, _W - 12 - len(m))
        self._print(
            f"\n  {_P2}{_BD}◆{_R}  {_P}{_BD}daena{_R}"
            f"  {_SEP}{'─' * gap}{_R}  {_DIM}{m}{_R}"
        )
        self._resp_open = True

    def _close_response(self) -> None:
        if self._content_buf:
            self._print(f"  {_RESP}{self._content_buf}{_R}")
            self._content_buf = ""
        self._resp_open = False

    def _flush_resp_lines(self) -> None:
        while "\n" in self._content_buf:
            ln, self._content_buf = self._content_buf.split("\n", 1)
            self._print(f"  {_RESP}{ln}{_R}")
        if len(self._content_buf) >= 160:
            self._print(f"  {_RESP}{self._content_buf}{_R}")
            self._content_buf = ""

    def finish(self, result: Any = None, error: BaseException | None = None) -> None:
        if self._finished or self._finished_rendered:
            return

        resp = getattr(result, "response", None) if result is not None else None
        if resp and not self.response_streamed:
            self._open_response()
            for ln in str(resp).splitlines():
                self._print(f"  {_RESP}{ln}{_R}")
            self.response_streamed = True

        self._close_thinking()
        self._close_response()

        elapsed = f"{time.monotonic() - self._start_ts:.1f}s"
        tok = _fmt_tok(self.usage)

        if error is not None:
            badge = f"{_ERR}{_BD}✗ failed{_R}"
        elif self.status == "stopped":
            badge = f"{_FAIL}{_BD}■ stopped{_R}"
        else:
            badge = f"{_OK}{_BD}✓ done{_R}"

        tools_str = f"{_OK}✓{self.success_count}{_R}"
        if self.failure_count:
            tools_str += f"  {_FAIL}✗{self.failure_count}{_R}"

        self._print(f"\n{_sep()}")
        self._print(
            f"  {badge}  "
            f"{_DIM}iter {self.iteration}  ·  tools {_R}"
            f"{tools_str}  "
            f"{_DIM}·  {tok} tok  ·  ctx {self._ctx_str()}  ·  {elapsed}{_R}"
        )
        self._finished = self._finished_rendered = True

    def toolbar(self) -> HTML:
        if self.status == "thinking":
            label = "thinking…"
        elif self.status == "responding":
            label = "responding…"
        elif self.status.startswith("tool:"):
            label = self.status[5:]
        elif self.status.startswith("result:"):
            label = self.status[7:]
        elif self.status in {"completed", "done"}:
            label = "ready"
        else:
            label = self.status

        # Dynamic values must be escaped before being interpolated into
        # Prompt Toolkit's HTML formatter. Tool/status text can legitimately
        # contain characters such as '<', '>' or '&' (for example shell args).
        # Passing those raw turns ordinary status text into malformed XML and
        # can crash the CLI's worker thread.
        label = html.escape(label, quote=False)
        model = html.escape(str(self.model), quote=False)

        itr = f"{self.iteration}/{self.max_iterations}" if self.max_iterations else str(self.iteration)
        think = str(self.think_enabled).lower()
        ctx = self._ctx_str()
        pb = _bar(self.context_tokens, self.context_budget)

        return HTML(
            f"<ansibrightmagenta>  ◆  </ansibrightmagenta>"
            f"<ansibrightwhite><b>{label}</b></ansibrightwhite>"
            f"  <ansiblue>{pb}</ansiblue>"
            f"  <ansicyan>ctx {ctx}</ansicyan>"
            f"  <ansibrightblack>·</ansibrightblack>"
            f"  <ansicyan>iter {itr}</ansicyan>"
            f"  <ansibrightblack>·</ansibrightblack>"
            f"  <ansibrightgreen>✓{self.success_count}</ansibrightgreen>"
            f" <ansired>✗{self.failure_count}</ansired>"
            f"  <ansibrightblack>·</ansibrightblack>"
            f"  <ansibrightblack>{model}  think={think}</ansibrightblack>"
            f"  "
        )

    def _ctx_str(self) -> str:
        if not self.context_budget:
            return _fmt_tok(self.context_tokens)
        pct = (self.context_tokens / self.context_budget) * 100
        return f"{_fmt_tok(self.context_tokens)}/{_fmt_tok(self.context_budget)} ({pct:.0f}%)"

    def _print(self, text: str) -> None:
        # This method is called directly from the Ollama streaming worker.
        # patch_stdout() replaces sys.stdout with PromptToolkit's
        # thread-safe proxy while the prompt is active, so writing to stdout
        # here gives us immediate flush semantics without waiting for the
        # whole generation to finish.
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
