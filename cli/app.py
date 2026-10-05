from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.agent.agent import Agent
from src.models.ContextEvent import (
    ContextEvent,
    ContextPriority,
    ContextRole,
    ContextType,
)

from .render import StreamRenderer


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"

STYLE = Style.from_dict(
    {
        "prompt": "ansicyan bold",
        "bottom-toolbar": "ansiwhite bg:ansiblack",
        "completion-menu.completion": "bg:ansiblack fg:ansiwhite",
        "completion-menu.completion.current": "bg:ansiblue fg:ansiwhite",
    }
)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be an object: {path}")
    return data


def _think_value(agent: Agent) -> object:
    config = getattr(agent.llm, "generation_config", {}) or {}
    return config.get(
        "think",
        getattr(agent.llm.model, "defaultConfig", {}).get("think", "medium")
        if agent.llm.model
        else "medium",
    )


def _set_think(agent: Agent, value: object) -> object:
    generation_config = getattr(agent.llm, "generation_config", None)
    if generation_config is None:
        raise RuntimeError("The active provider does not expose generation configuration.")
    generation_config["think"] = value
    return value


def _set_setting(agent: Agent, key: str, value: str) -> object:
    generation_config = getattr(agent.llm, "generation_config", None)
    if generation_config is None:
        raise RuntimeError("The active provider does not expose generation configuration.")

    normalized = key.strip().lower().replace("-", "_")
    if normalized == "temperature":
        parsed: object = float(value)
        if not 0 <= float(parsed) <= 2:
            raise ValueError("temperature must be between 0 and 2")
    elif normalized in {"num_thread", "num_threads"}:
        parsed = int(value)
        if parsed < 1:
            raise ValueError("num_thread must be >= 1")
        normalized = "num_thread"
    elif normalized == "max_iterations":
        parsed = int(value)
        if parsed < 1:
            raise ValueError("max_iterations must be >= 1")
        agent.loop.max_iterations = parsed
        return parsed
    elif normalized == "think":
        raw = value.strip().lower()
        if raw == "auto":
            parsed = None
        elif raw in {"off", "false"}:
            parsed = False
        elif raw in {"low", "medium", "high"}:
            parsed = raw
        else:
            raise ValueError("think must be auto, off, low, medium, or high")
    else:
        raise ValueError(
            "Supported settings: temperature, num_thread, max_iterations, think"
        )

    if parsed is None:
        generation_config.pop(normalized, None)
    else:
        generation_config[normalized] = parsed
    return parsed


def _session_rows(agent: Agent, limit: int = 30) -> list[dict]:
    return agent.loop.stm.list_sessions(limit=limit)


def _session_title(value: object) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return "Untitled conversation"
    if len(text) > 78:
        return text[:75].rstrip() + "..."
    return text


def _format_age(value: object) -> str:
    try:
        when = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return "unknown"
    seconds = max(0, int((datetime.now() - when).total_seconds()))
    if seconds < 60:
        return "now"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    if seconds < 604800:
        return f"{seconds // 86400}d"
    return when.strftime("%Y-%m-%d")


def _new_session(agent: Agent) -> None:
    session_id = uuid4()
    agent.session_id = session_id
    agent.loop.session_id = session_id


def _open_session(agent: Agent, index: int) -> str:
    sessions = _session_rows(agent)
    if index < 1 or index > len(sessions):
        raise ValueError(f"No conversation #{index}.")
    session_id = str(sessions[index - 1]["session_id"])
    agent.session_id = UUID(session_id)
    agent.loop.session_id = agent.session_id
    return _session_title(sessions[index - 1].get("title"))


def _show_sessions(console: Console, agent: Agent) -> None:
    sessions = _session_rows(agent)
    if not sessions:
        console.print("[dim]No conversations yet.[/dim]")
        return

    console.print("\n[bold cyan]Conversations[/bold cyan]")
    for index, session in enumerate(sessions, start=1):
        current = str(session.get("session_id")) == str(agent.session_id)
        marker = "●" if current else " "
        console.print(
            f" {index:>2} {marker}  {_session_title(session.get('title')):<78} "
            f"[dim]{_format_age(session.get('last_activity'))} · "
            f"{session.get('event_count', 0)} events[/dim]"
        )
    console.print("[dim]open: /session N · new: /session new[/dim]\n")


def _show_status(console: Console, agent: Agent) -> None:
    config = getattr(agent.llm, "generation_config", {}) or {}
    console.print(
        f"[cyan]Daena[/cyan]  model=[bold]{agent.llm.llm_config.get('model_name') or 'default'}[/bold] "
        f"provider=[bold]{agent.llm.provider_name}[/bold] "
        f"think=[bold]{_think_value(agent)}[/bold]\n"
        f"[dim]workspace[/dim] {agent.workingdirectory}\n"
        f"[dim]session[/dim] {agent.session_id}\n"
        f"[dim]temperature[/dim] {config.get('temperature', 'default')} · "
        f"iterations {agent.loop.max_iterations}"
    )


def _show_tools(console: Console, agent: Agent) -> None:
    names: list[str] = []
    for item in agent.loop.tool_definitions:
        function = item.get("function", {}) if isinstance(item, dict) else {}
        name = str(function.get("name", "")).strip()
        if name:
            names.append(name)
    console.print("[cyan]tools[/cyan] " + ", ".join(sorted(names)))


def _show_stats(console: Console, agent: Agent) -> None:
    metrics = agent.last_run_metrics
    if not metrics:
        console.print("[dim]No run yet.[/dim]")
        return
    keys = (
        "iterations",
        "llm_calls",
        "tokens",
        "generation_failures",
        "tool_call_attempts",
        "tool_successes",
        "tool_failures",
        "loop_guard_warnings",
        "loop_guard_blocks",
        "duration_ms",
        "stop_reason",
    )
    lines = [f"{key}: {metrics.get(key)}" for key in keys if key in metrics]
    console.print(Panel("\n".join(lines), title="stats", border_style="green"))


def _show_help(console: Console) -> None:
    console.print(
        Panel(
            "\n".join(
                [
                    "/help                 commands",
                    "/status               runtime status",
                    "/sessions             conversation history",
                    "/session N             open conversation N",
                    "/session new           new conversation",
                    "/think low|medium|high|off|auto",
                    "/provider [name]      show/switch provider",
                    "/model [name]         show/set model",
                    "/set key value         change runtime setting",
                    "/tools                available tools",
                    "/stats                last run metrics",
                    "/cwd [path]           show/change workspace",
                    "/interrupt            stop current task",
                    "/clear                clear terminal",
                    "/exit                 quit Daena",
                    "",
                    "While Agent is working: type normal text to redirect it.",
                ]
            ),
            title="Daena",
            border_style="cyan",
        )
    )


def _provider_switch(agent: Agent, name: str) -> None:
    providers = getattr(agent.llm.registry, "providers", {})
    key = name.strip().lower()
    if key not in providers:
        raise ValueError(
            f"Unknown provider '{name}'. Available: {', '.join(sorted(providers))}"
        )
    agent.llm._loadModel(key)
    agent.llm.provider_name = key
    agent.llm.generation_config = dict(
        ((agent.config.get("llm") or {}).get("provider_config") or {}).get(
            "generation_config"
        )
        or {}
    )


def _handle_command(console: Console, agent: Agent, text: str, running: bool) -> str:
    parts = text.split(maxsplit=1)
    command = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""

    if running and command in {"/provider", "/model", "/session", "/sessions", "/cwd"}:
        console.print("[yellow]Wait for the current task or use /interrupt first.[/yellow]")
        return "continue"

    if command in {"/exit", "/quit", "/q"}:
        return "exit"
    if command == "/help":
        _show_help(console, agent)
    elif command == "/clear":
        console.clear()
    elif command in {"/sessions", "/history"}:
        _show_sessions(console, agent)
    elif command == "/session":
        if not argument:
            _show_sessions(console, agent)
        elif argument.lower() == "new":
            _new_session(agent)
            console.print("[green]new conversation[/green]")
        elif argument.isdigit():
            try:
                title = _open_session(agent, int(argument))
                console.print(f"[green]opened #{argument} · {title}[/green]")
            except Exception as exc:
                console.print(f"[red]{exc}[/red]")
        else:
            console.print("[yellow]Usage: /session | /session N | /session new[/yellow]")
    elif command == "/status":
        _show_status(console, agent)
    elif command == "/tools":
        _show_tools(console, agent)
    elif command == "/stats":
        _show_stats(console, agent)
    elif command == "/think":
        value = argument.lower()
        if value == "on":
            value = "medium"
        if value == "auto":
            _set_think(agent, None)
        elif value == "off":
            _set_think(agent, False)
        elif value in {"low", "medium", "high"}:
            _set_think(agent, value)
        else:
            console.print("[yellow]Usage: /think low|medium|high|off|auto[/yellow]")
            return "continue"
        console.print(f"[green]think = {_think_value(agent)}[/green]")
    elif command == "/provider":
        if not argument:
            providers = sorted(getattr(agent.llm.registry, "providers", {}).keys())
            console.print(
                f"[cyan]provider[/cyan] active={agent.llm.provider_name} "
                f"available={', '.join(providers)}"
            )
        else:
            try:
                _provider_switch(agent, argument)
                console.print(f"[green]provider → {agent.llm.provider_name}[/green]")
            except Exception as exc:
                console.print(f"[red]{exc}[/red]")
    elif command == "/model":
        if not argument:
            console.print(str(agent.llm.llm_config.get("model_name") or agent.llm.model.defaultModel))
        else:
            agent.llm.llm_config["model_name"] = argument
            console.print(f"[green]model → {argument}[/green]")
    elif command == "/set":
        bits = argument.split(maxsplit=1)
        if len(bits) != 2:
            console.print("[yellow]Usage: /set key value[/yellow]")
        else:
            try:
                value = _set_setting(agent, bits[0], bits[1])
                console.print(f"[green]{bits[0]} = {value}[/green]")
            except Exception as exc:
                console.print(f"[red]{exc}[/red]")
    elif command == "/cwd":
        if not argument:
            console.print(agent.workingdirectory)
        else:
            agent.set_workingdirectory(argument)
            console.print(f"[green]workspace → {agent.workingdirectory}[/green]")
    else:
        console.print(f"[yellow]Unknown command: {command}[/yellow]")
    return "continue"


def _run_turn(console: Console, agent: Agent, text: str, state: dict) -> None:
    stop_event = threading.Event()
    renderer = StreamRenderer(
        model=str(agent.llm.llm_config.get("model_name") or "default"),
        workspace=agent.workingdirectory,
        session_id=str(agent.session_id),
        think_enabled=_think_value(agent),
    )

    def emit(event: dict) -> None:
        state["renderer"] = renderer
        renderer.handle(event)
        state["toolbar"] = renderer.toolbar()

    task = ContextEvent(
        role=ContextRole.USER,
        type=ContextType.MESSAGE,
        content=text,
        priority=ContextPriority.NORMAL,
        step=0,
        metadata={"source": "cli", "workspace": agent.workingdirectory},
    )

    def worker() -> None:
        try:
            state["result"] = agent.act(
                task,
                on_event=emit,
                stop_event=stop_event,
            )
        except Exception as exc:
            state["error"] = exc
            renderer.handle({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            state["running"] = False
            state["stop_event"] = None
            state["toolbar"] = renderer.toolbar()

    state["running"] = True
    state["stop_event"] = stop_event
    state["renderer"] = renderer
    thread = threading.Thread(target=worker, name="daena-agent", daemon=True)
    state["thread"] = thread
    thread.start()


def _print_turn_result(console: Console, state: dict) -> None:
    result = state.get("result")
    if result is not None:
        response = getattr(result, "response", None)
        if response:
            console.print(Markdown(str(response)))
    metrics = state.get("renderer")
    if metrics:
        console.print(
            f"[dim]done · iter {metrics.iteration} · "
            f"tools {metrics.tool_count} · usage {metrics.usage}[/dim]"
        )
    state["result"] = None
    state["error"] = None
    state["renderer"] = None


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="daena",
        description="Daena interactive coding-agent CLI",
    )
    parser.add_argument("--cwd", dest="cwd", default=None)
    parser.add_argument("--no-think", action="store_true")
    args = parser.parse_args()

    config = _load_config(CONFIG_PATH)
    agent = Agent(config)
    if args.cwd:
        agent.set_workingdirectory(args.cwd)
    if args.no_think:
        _set_think(agent, False)

    console = Console()
    session = PromptSession(style=STYLE)
    state: dict = {
        "running": False,
        "thread": None,
        "stop_event": None,
        "renderer": None,
        "result": None,
        "error": None,
        "toolbar": "ready",
    }

    completer = NestedCompleter.from_nested_dict(
        {
            "help": None,
            "status": None,
            "sessions": None,
            "session": {"new": None},
            "think": {"low": None, "medium": None, "high": None, "off": None, "auto": None},
            "provider": None,
            "model": None,
            "set": {"temperature": None, "num_thread": None, "max_iterations": None, "think": None},
            "tools": None,
            "stats": None,
            "cwd": None,
            "interrupt": None,
            "clear": None,
            "exit": None,
        }
    )

    console.print(
        Panel.fit(
            "[bold cyan]◈ DAENA[/bold cyan]  [white]terminal agent[/white]\n"
            f"[dim]{agent.llm.llm_config.get('model_name') or 'default'} · "
            f"{agent.workingdirectory}\n"
            "Tab: autocomplete · /sessions: history · Ctrl+C: interrupt · /exit: quit[/dim]",
            border_style="cyan",
        )
    )

    try:
        while True:
            running = bool(state["running"])

            try:
                with patch_stdout(raw=True):
                    user_text = session.prompt(
                        f"[cyan]you {'↪' if running else '›'} [/cyan]",
                        completer=completer,
                        bottom_toolbar=lambda: state.get("toolbar", "ready"),
                    ).strip()
            except KeyboardInterrupt:
                if running and state.get("stop_event") is not None:
                    state["stop_event"].set()
                    console.print("[yellow]↯ interrupt requested[/yellow]")
                    continue
                console.print("[yellow]Press /exit to leave Daena.[/yellow]")
                continue
            except EOFError:
                if running and state.get("stop_event") is not None:
                    state["stop_event"].set()
                    thread = state.get("thread")
                    if thread is not None:
                        thread.join()
                break

            if state["running"] and user_text:
                if user_text.startswith("/"):
                    action = _handle_command(console, agent, user_text, running=True)
                    if action == "exit":
                        state["stop_event"].set()
                        thread = state.get("thread")
                        if thread is not None:
                            thread.join()
                        break
                    continue

                agent.steer(user_text)
                continue

            if not user_text:
                continue

            if user_text.startswith("/"):
                action = _handle_command(console, agent, user_text, running=False)
                if action == "exit":
                    break
                continue

            _run_turn(console, agent, user_text, state)

            # Worker is asynchronous so the input prompt remains available.
            # When it finishes, render the final reply before accepting the
            # next normal turn.
            thread = state.get("thread")
            if thread is not None and not thread.is_alive() and state["running"] is False:
                _print_turn_result(console, state)
    finally:
        stop_event = state.get("stop_event")
        if stop_event is not None:
            stop_event.set()
        thread = state.get("thread")
        if thread is not None:
            thread.join()
        _print_turn_result(console, state)
        agent.close()

    console.print("[bold cyan]◈ session closed[/bold cyan]")
