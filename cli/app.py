from __future__ import annotations

import argparse
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.console import Console

# Keep framework INFO logs in the file; the interactive terminal only shows warnings/errors.
os.environ.setdefault("DAENA_CLI", "1")

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
        "prompt": "ansimagenta bold",
        "completion-menu.completion": "bg:ansiblack fg:ansiwhite",
        "completion-menu.completion.current": "bg:ansiblue fg:ansiwhite",
        "scrollbar.background": "bg:ansiblack",
        "scrollbar.button": "bg:ansiblue",
    }
)

COMMANDS = {
    "/help": None,
    "/status": None,
    "/sessions": None,
    "/session": {"new": None},
    "/new": None,
    "/reasoning": {"low": None, "medium": None, "high": None, "off": None, "auto": None},
    "/think": {"low": None, "medium": None, "high": None, "off": None, "auto": None},
    "/busy": {"steer": None, "interrupt": None, "queue": None, "status": None},
    "/provider": None,
    "/model": None,
    "/set": {
        "temperature": None,
        "num_thread": None,
        "max_iterations": None,
        "think": None,
    },
    "/interrupt": None,
    "/clear": None,
    "/exit": None,
}

HELP_LINES = (
    "/help                     commands",
    "/sessions                 conversation history",
    "/session N                open conversation N",
    "/session new              start a new conversation",
    "/reasoning low|medium|high|off",
    "/busy steer|interrupt|queue",
    "/provider [name]          show/switch provider",
    "/model [name]             show/set model",
    "/set key value            runtime setting",
    "/interrupt                stop current task",
    "/clear                    clear terminal",
    "/exit                     quit Daena",
    "",
    "While Daena is working, type normal text to redirect it.",
)

BUSY_MODES = {"steer", "interrupt", "queue"}


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
            raise ValueError("think must be auto, off, low, medium, high, or auto")

    else:
        raise ValueError(
            "Supported settings: temperature, num_thread, max_iterations, think"
        )

    if parsed is None:
        generation_config.pop(normalized, None)
    else:
        generation_config[normalized] = parsed
    return parsed


def _new_session(agent: Agent) -> None:
    session_id = uuid4()
    agent.session_id = session_id
    agent.loop.session_id = session_id


def _session_rows(agent: Agent, limit: int = 30) -> list[dict]:
    return agent.loop.stm.list_sessions(limit=limit)


def _session_title(value: object) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return "Untitled conversation"
    if len(text) > 82:
        return text[:79].rstrip() + "..."
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


def _open_session(agent: Agent, index: int) -> str:
    sessions = _session_rows(agent)
    if index < 1 or index > len(sessions):
        raise ValueError(f"No conversation #{index}.")
    selected = sessions[index - 1]
    session_id = UUID(str(selected["session_id"]))
    agent.session_id = session_id
    agent.loop.session_id = session_id
    return _session_title(selected.get("title"))


def _show_sessions(console: Console, agent: Agent) -> None:
    sessions = _session_rows(agent)
    if not sessions:
        console.print("[dim]No conversations yet.[/dim]")
        return

    console.print("\n[bold cyan]Conversations[/bold cyan]")
    for index, session in enumerate(sessions, start=1):
        current = str(session.get("session_id")) == str(agent.session_id)
        marker = "●" if current else " "
        title = _session_title(session.get("title"))
        workspace = Path(str(session.get("workspace") or "")).name
        details = f"{_format_age(session.get('last_activity'))} · {session.get('event_count', 0)} events"
        if workspace:
            details += f" · {workspace}"
        console.print(
            f" {index:>2} {marker}  {title:<82} [dim]{details}[/dim]"
        )
    console.print("[dim]Enter /session N to open one, or /session new for a fresh chat.[/dim]\n")


def _show_status(console: Console, agent: Agent, busy_mode: str) -> None:
    config = getattr(agent.llm, "generation_config", {}) or {}
    model = str(agent.llm.llm_config.get("model_name") or "default")
    console.print(
        f"[cyan]Daena[/cyan]  [bold]{model}[/bold] · "
        f"{agent.llm.provider_name} · think={_think_value(agent)} · busy={busy_mode}\n"
        f"[dim]workspace[/dim] {agent.workingdirectory}\n"
        f"[dim]session[/dim] {str(agent.session_id)}\n"
        f"[dim]temperature[/dim] {config.get('temperature', 'default')} · "
        f"max_iterations={agent.loop.max_iterations}"
    )


def _show_tools(console: Console, agent: Agent) -> None:
    names = []
    for definition in agent.loop.tool_definitions:
        function = definition.get("function", {}) if isinstance(definition, dict) else {}
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
    console.print("[cyan]last run[/cyan]")
    for key in keys:
        if key in metrics:
            console.print(f"  {key:<24} {metrics[key]}")


def _show_help(console: Console) -> None:
    console.print("[bold magenta]Daena commands[/bold magenta]")
    for line in HELP_LINES:
        console.print(f"  {line}")


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


def _handle_command(
    console: Console,
    agent: Agent,
    text: str,
    busy_mode: str,
    running: bool,
) -> tuple[str, str | None]:
    parts = text.split(maxsplit=1)
    command = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""

    if command in {"/exit", "/quit", "/q"}:
        return "exit", None

    if running and command in {
        "/provider",
        "/model",
        "/session",
        "/sessions",
        "/cwd",
        "/new",
    }:
        console.print("[yellow]Daena is busy. Use /interrupt first for this command.[/yellow]")
        return "continue", busy_mode

    if command == "/help":
        _show_help(console)
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
    elif command == "/new":
        _new_session(agent)
        console.print("[green]new conversation[/green]")
    elif command == "/status":
        _show_status(console, agent, busy_mode)
    elif command in {"/reasoning", "/think"}:
        value = argument.lower() or str(_think_value(agent)).lower()
        if value == "on":
            value = "medium"
        if value == "auto":
            _set_think(agent, None)
        elif value == "off":
            _set_think(agent, False)
        elif value in {"low", "medium", "high"}:
            _set_think(agent, value)
        else:
            console.print("[yellow]Usage: /reasoning low|medium|high|off|auto[/yellow]")
            return "continue", busy_mode
        console.print(f"[green]reasoning = {_think_value(agent)}[/green]")
    elif command == "/busy":
        if not argument or argument == "status":
            console.print(f"[cyan]busy input mode = {busy_mode}[/cyan]")
        elif argument in BUSY_MODES:
            busy_mode = argument
            console.print(f"[green]busy input mode = {busy_mode}[/green]")
        else:
            console.print("[yellow]Usage: /busy steer|interrupt|queue|status[/yellow]")
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
            console.print(
                str(
                    agent.llm.llm_config.get("model_name")
                    or agent.llm.model.defaultModel
                )
            )
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
    elif command == "/interrupt":
        if running and (agent_state := getattr(agent, "_cli_state", None)):
            stop_event = agent_state.get("stop_event")
            if stop_event is not None:
                stop_event.set()
                console.print("[yellow]↯ interrupt requested[/yellow]")
        else:
            console.print("[dim]No active task.[/dim]")
    elif command == "/clear":
        console.clear()
    else:
        console.print(f"[yellow]Unknown command: {command}[/yellow]")

    return "continue", busy_mode


def _start_turn(agent: Agent, text: str, state: dict) -> None:
    stop_event = threading.Event()
    renderer = state.get("renderer")
    if renderer is None:
        renderer = StreamRenderer(
            model=str(agent.llm.llm_config.get("model_name") or "default"),
            workspace=agent.workingdirectory,
            session_id=str(agent.session_id),
            think_enabled=_think_value(agent),
        )
        renderer.begin_user_message(text)

    task = ContextEvent(
        role=ContextRole.USER,
        type=ContextType.MESSAGE,
        content=text,
        priority=ContextPriority.NORMAL,
        step=0,
        metadata={"source": "cli", "workspace": agent.workingdirectory},
    )

    def emit(event: dict) -> None:
        state["renderer"] = renderer
        state["toolbar"] = renderer.toolbar()
        renderer.handle(event)
        state["toolbar"] = renderer.toolbar()

    def worker() -> None:
        try:
            state["result"] = agent.act(
                task,
                on_event=emit,
                stop_event=stop_event,
            )
        except Exception as exc:
            state["error"] = exc
            renderer.handle(
                {
                    "type": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            # Finalize from the worker, not from the main prompt loop. The
            # prompt remains blocked waiting for input, so deferring this until
            # the next iteration makes a completed response look like it needs
            # a second Enter.
            renderer.finish(
                result=state.get("result"),
                error=state.get("error"),
            )
            state["running"] = False
            state["stop_event"] = None
            state["turn_finished"] = True
            state["toolbar"] = renderer.toolbar()

    state["running"] = True
    state["stop_event"] = stop_event
    state["renderer"] = renderer
    state["result"] = None
    state["error"] = None
    state["turn_finished"] = False
    agent._cli_state = state

    thread = threading.Thread(
        target=worker,
        name="daena-agent",
        daemon=True,
    )
    state["thread"] = thread
    thread.start()


def _print_finished_turn(console: Console, state: dict) -> None:
    """Release completed-turn state after the worker already rendered it."""
    state["result"] = None
    state["error"] = None
    state["renderer"] = None
    state["thread"] = None
    state["turn_finished"] = False

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="daena",
        description="Daena terminal agent",
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
    completer = NestedCompleter.from_nested_dict(COMMANDS)

    state: dict = {
        "running": False,
        "stop_event": None,
        "thread": None,
        "renderer": None,
        "result": None,
        "error": None,
        "turn_finished": False,
    }

    model = str(agent.llm.llm_config.get("model_name") or "default")
    console.print(
        f"[bold magenta]◈ DAENA[/bold magenta]  [bold cyan]{model}[/bold cyan]  "
        f"[dim]· {agent.llm.provider_name} · terminal agent[/dim]\n"
        f"[dim]cwd[/dim] {agent.workingdirectory}\n"
        f"[dim]Tab commands · type while working to redirect · Ctrl+C interrupt[/dim]"
    )

    busy_mode = "steer"

    try:
        while True:
            if state.get("turn_finished") and not state.get("running"):
                _print_finished_turn(console, state)

            # Follow-ups queued while the previous run was busy are started
            # only after the Agent becomes idle, never concurrently.
            if not state.get("running"):
                pending_interrupt = state.pop("pending_followup", None)
                if pending_interrupt:
                    _start_turn(agent, pending_interrupt, state)
                    continue

                pending_queue = state.get("pending_followups") or []
                if pending_queue:
                    next_text = pending_queue.pop(0)
                    if pending_queue:
                        state["pending_followups"] = pending_queue
                    else:
                        state.pop("pending_followups", None)
                    _start_turn(agent, next_text, state)
                    continue

            running = bool(state["running"])

            try:
                with patch_stdout(raw=True):
                    user_text = session.prompt(
                        HTML(f"<ansimagenta><b>YOU</b></ansimagenta> <ansicyan><b>{'»' if running else '›'}</b></ansicyan> "),
                        completer=completer,
                    ).strip()
            except KeyboardInterrupt:
                if state.get("running") and state.get("stop_event") is not None:
                    state["stop_event"].set()
                    console.print("[yellow]↯ interrupt requested[/yellow]")
                    continue
                console.print("[yellow]Use /exit to leave Daena.[/yellow]")
                continue
            except EOFError:
                if state.get("stop_event") is not None:
                    state["stop_event"].set()
                break

            if not user_text:
                continue

            running = bool(state["running"])

            if running:
                if user_text.startswith("/"):
                    action, busy_mode = _handle_command(
                        console, agent, user_text, busy_mode, True
                    )
                    if action == "exit":
                        if state.get("stop_event") is not None:
                            state["stop_event"].set()
                        break
                    continue

                if busy_mode == "interrupt":
                    if state.get("stop_event") is not None:
                        state["stop_event"].set()
                    console.print("[yellow]↯ interrupting current task; message will be sent after it stops.[/yellow]")
                    state["pending_followup"] = user_text
                elif busy_mode == "queue":
                    pending = state.setdefault("pending_followups", [])
                    pending.append(user_text)
                    console.print(f"[cyan]queued follow-up #{len(pending)}[/cyan]")
                else:
                    agent.steer(user_text)
                continue

            if user_text.startswith("/"):
                action, busy_mode = _handle_command(
                    console, agent, user_text, busy_mode, False
                )
                if action == "exit":
                    break
                continue

            renderer = StreamRenderer(
                model=str(agent.llm.llm_config.get("model_name") or "default"),
                workspace=agent.workingdirectory,
                session_id=str(agent.session_id),
                think_enabled=_think_value(agent),
            )
            state["renderer"] = renderer
            _start_turn(agent, user_text, state)

            # Busy mode remains active until the worker finishes.
            # The next prompt remains immediately available for steering.
    finally:
        stop_event = state.get("stop_event")
        if stop_event is not None:
            stop_event.set()

        thread = state.get("thread")
        if thread is not None:
            thread.join()

        if state.get("turn_finished"):
            _print_finished_turn(console, state)

        if hasattr(agent, "_cli_state"):
            delattr(agent, "_cli_state")

        agent.close()

    console.print("[bold magenta]◈ session closed[/bold magenta]")