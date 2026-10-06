from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import ANSI, HTML

_P="\x1b[38;2;167;139;250m"; _P2="\x1b[38;2;139;92;246m"; _B="\x1b[38;2;96;165;250m"
_TH="\x1b[38;2;109;101;156m"; _THD="\x1b[38;2;130;110;200m"; _RESP="\x1b[38;2;219;234;254m"
_TOOL="\x1b[38;2;99;179;237m"; _OK="\x1b[38;2;74;222;128m"; _FAIL="\x1b[38;2;251;113;133m"
_ERR="\x1b[38;2;252;165;165m"; _SEP="\x1b[38;2;79;64;124m"; _DIM="\x1b[38;2;107;114;128m"
_MUT="\x1b[38;2;71;85;105m"; _R="\x1b[0m"; _BD="\x1b[1m"; _W=68

def _short(v: Any, lim=500):
    s=" ".join(str(v if isinstance(v,str) else repr(v)).split())
    return s if len(s)<=lim else s[:lim-1]+"…"

def _fmt_tok(n:int):
    if n>=1_000_000:return f"{n/1_000_000:.1f}M"
    if n>=1_000:return f"{n/1_000:.1f}k"
    return str(n)

def _sep_line(char="─"):
    return f"{_SEP}{char*_W}{_R}"

@dataclass
class StreamRenderer:
    model: str
    workspace: str
    session_id: str
    think_enabled: Any="medium"
    iteration:int=0
    max_iterations:int=0
    status:str="ready"
    tool_count:int=0
    success_count:int=0
    failure_count:int=0
    usage:int=0
    context_tokens:int=0
    context_budget:int=0
    response_streamed:bool=False
    _think_open:bool=field(default=False,repr=False)
    _resp_open:bool=field(default=False,repr=False)
    _finished:bool=field(default=False,repr=False)
    _start_ts:float=field(default_factory=time.monotonic,repr=False)

    def handle(self,event:dict[str,Any])->None:
        t=str(event.get("type") or "")
        if t=="run_start":
            self.status="starting"; self.max_iterations=int(event.get("max_iterations") or self.max_iterations)
            self.session_id=str(event.get("session_id") or self.session_id); self._start_ts=time.monotonic()
            self._println(_sep_line())
        elif t=="iteration_start":
            self._close_thinking(); self._close_response()
            self.iteration=int(event.get("iteration") or 0); self.max_iterations=int(event.get("max_iterations") or self.max_iterations)
            self.status="thinking"
        elif t=="context":
            self.context_tokens=int(event.get("estimated_tokens") or 0); self.context_budget=int(event.get("budget") or 0)
        elif t=="thinking_delta":
            self.status="thinking"; self._feed_think(str(event.get("text") or ""))
        elif t=="content_delta":
            self._close_thinking(); self.status="responding"
            text=str(event.get("text") or "")
            if text:self.response_streamed=True; self._feed_content(text)
        elif t=="generation_done":
            self.usage=int(event.get("usage") or self.usage or 0)
        elif t=="tool_call":
            self._close_thinking(); self._close_response(); self.tool_count+=1
            name=_short(event.get("name") or "unknown",32); self.status=f"tool:{name}"
            self._println(f"\n  {_B}{_BD}↳{_R}  {_TOOL}{name}{_R}")
        elif t=="tool_result":
            self._close_response(); name=_short(event.get("name") or "unknown",32); ok=bool(event.get("success"))
            self.success_count += int(ok); self.failure_count += int(not ok)
            icon=f"{_OK}✓{_R}" if ok else f"{_FAIL}✗{_R}"
            self._println(f"  {icon}  {name}  {_DIM}{_short(event.get('summary') or event.get('content') or '',360)}{_R}")
        elif t=="steering":
            self._close_thinking(); self._close_response(); self.status="redirected"
            self._println(f"\n  {_P}↪ redirect{_R}  {_DIM}{_short(event.get('text') or '',600)}{_R}")
        elif t=="run_stopped":
            self._close_thinking(); self._close_response(); self.status="stopped"
            self._println(f"\n  {_FAIL}{_BD}■ stopped{_R}  {_DIM}{_short(event.get('reason') or '',600)}{_R}")
        elif t=="error":
            self._close_thinking(); self._close_response(); self.status="error"
            self._println(f"\n  {_ERR}{_BD}✗ error{_R}  {_DIM}{_short(event.get('message') or '',700)}{_R}")

    def _feed_think(self,delta:str):
        if not delta:return
        if not self._think_open:
            suffix=f" · iter {self.iteration}" if self.iteration else ""
            label=f"─ thinking{suffix} "; dashes=max(2,_W-len(label)-2)
            self._println(f"\n  {_THD}╭{label}{'─'*dashes}╮{_R}")
            self._ink(f"  {_THD}│{_R}  {_TH}"); self._think_open=True
        parts=delta.split("\n")
        for i,p in enumerate(parts):
            if i:self._ink(f"{_R}\n  {_THD}│{_R}  {_TH}")
            if p:self._ink(p)

    def _close_thinking(self):
        if not self._think_open:return
        self._ink(f"{_R}\n"); self._println(f"  {_THD}╰{'─'*(_W-2)}╯{_R}"); self._think_open=False

    def _open_response(self):
        if self._resp_open:return
        m=self.model if len(self.model)<=28 else self.model[:25]+"…"
        self._println(f"\n  {_P2}{_BD}◆{_R}  {_P}{_BD}daena{_R}  {_SEP}{'─'*max(2,_W-18-len(m))}{_R}  {_DIM}{m}{_R}")
        self._ink(f"  {_RESP}"); self._resp_open=True

    def _feed_content(self,delta:str):
        if not delta:return
        if not self._resp_open:self._open_response()
        parts=delta.split("\n")
        for i,p in enumerate(parts):
            if i:self._ink(f"{_R}\n  {_RESP}")
            if p:self._ink(p)

    def _close_response(self):
        if self._resp_open:self._ink(f"{_R}\n"); self._resp_open=False

    def finish(self,result=None,error=None):
        if self._finished:return
        if result is not None and not self.response_streamed:
            text=getattr(result,"response","") or ""
            if text:self._open_response(); self._ink(text); self.response_streamed=True
        self._close_thinking(); self._close_response()
        elapsed=f"{time.monotonic()-self._start_ts:.1f}s"
        badge=f"{_ERR}{_BD}✗ failed{_R}" if error else (f"{_FAIL}{_BD}■ stopped{_R}" if self.status=="stopped" else f"{_OK}{_BD}✓ done{_R}")
        self._println(f"\n{_sep_line()}"); self._println(f"  {badge}  {_DIM}iter {self.iteration} · tools {_R}{_OK}✓{self.success_count}{_R} {_FAIL}✗{self.failure_count}{_R} {_DIM}· {_fmt_tok(self.usage)} tok · {elapsed}{_R}")
        self._finished=True

    def toolbar(self)->HTML:
        return HTML(f"<ansibrightmagenta> ◆ </ansibrightmagenta><ansibrightwhite><b>{self.status}</b></ansibrightwhite>  <ansicyan>ctx {_fmt_tok(self.context_tokens)}</ansicyan>  <ansicyan>iter {self.iteration}/{self.max_iterations or '?'}</ansicyan>")

    @staticmethod
    def _println(text:str): print_formatted_text(ANSI(text))
    @staticmethod
    def _ink(text:str): print_formatted_text(ANSI(text),end="",flush=True)
