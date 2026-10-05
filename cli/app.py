from __future__ import annotations

import argparse
import json
import queue
import select
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.live import Live

from src.agent.agent import Agent
from src.models.ContextEvent import ContextEvent, ContextPriority, ContextRole, ContextType

from .render import StreamRenderer


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be an object: {path}")
    return data


def _think_value(agent: Agent) -> object:
    config = getattr(agent.llm, "generation_config", {}) or {}
    return config.get("think", getattr(agent.llm.model, "defaultConfig", {}).get("think", "medium") if agent.llm.model else "medium")


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
    elif normalized == "max_iterations":
        parsed = int(value)
        if parsed < 1:
            raise ValueError("max_iterations must be >= 1")
        agent.loop.max_iterations = parsed
        return parsed
    elif normalized == "think":
        value_lower = value.strip().lower()
        if value_lower == "auto":
            parsed = None
        elif value_lower in {"off", "false"}:
            parsed = False
        elif value_lower in {"low", "medium", "high"}:
            parsed = value_lower
        else:
            raise ValueError("think must be auto, off, low, medium, or high")
    else:
        raise ValueError("Supported settings: temperature, num_thread, max_iterations, think")

    if parsed is None:
        generation_config.pop(normalized, None)
    else:
        generation_config[normalized] = parsed
    return parsed


def _show_help(console: Console) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_row("/help", "show commands")
    table.add_row("/status", "show session, model, workspace and runtime state")
    table.add_row("/cwd [path]", "show or change the Agent workspace")
    table.add_row("/think low|medium|high|off|auto", "set reasoning effort for the next turns")
    table.add_row("/provider [name]", "show or switch the active provider")
    table.add_row("/model [name]", "show or switch the model on the active provider")
    table.add_row("/set key value", "change runtime settings (temperature, threads, iterations, think)")
    table.add_row("/tools", "show tools currently exposed to the Agent")
    table.add_row("/stats", "show metrics from the last completed turn")
    table.add_row("/clear", "clear the terminal")
    table.add_row("/exit", "close Daena")
    console.print(Panel(table, title="[bold cyan]Daena commands[/bold cyan]", border_style="cyan"))


def _show_status(console: Console, agent: Agent) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("model", str(agent.llm.llm_config.get("model_name") or agent.llm.model or "default"))
    table.add_row("provider", str(agent.llm.provider_name))
    table.add_row("workspace", agent.workingdirectory)
    table.add_row("session", str(agent.session_id))
    table.add_row("think", str(_think_value(agent)).lower())
    table.add_row("temperature", str((getattr(agent.llm, "generation_config", {}) or {}).get("temperature", "default")))
    table.add_row("max iterations", str(agent.loop.max_iterations))
    console.print(Panel(table, title="[bold cyan]Daena status[/bold cyan]", border_style="cyan"))


def _show_provider(console: Console, agent: Agent) -> None:
    providers = sorted(getattr(agent.llm.registry, "providers", {}).keys())
    active = str(agent.llm.provider_name)
    console.print(Panel(
        f"[bold cyan]active[/bold cyan]  {active}\n"
        f"[bold cyan]available[/bold cyan]  {', '.join(providers) or 'none'}",
        title="[bold magenta]provider[/bold magenta]",
        border_style="magenta",
    ))


def _switch_provider(agent: Agent, provider_name: str) -> None:
    name = provider_name.strip().lower()
    providers = getattr(agent.llm.registry, "providers", {})
    if name not in providers:
        raise ValueError(f"Unknown provider '{name}'. Available: {', '.join(sorted(providers))}")
    agent.llm._loadModel(name)
    agent.llm.provider_name = name
    provider_cfg = (agent.config.get("llm") or {}).get("provider_config") or {}
    agent.llm.generation_config = dict(provider_cfg.get("generation_config") or {})


def _show_tools(console: Console, agent: Agent) -> None:
    names = []
    for item in agent.loop.tool_definitions:
        function = item.get("function", {}) if isinstance(item, dict) else {}
        name = str(function.get("name", "")).strip()
        if name:
            names.append(name)
    names.sort()
    console.print(Panel("\n".join(f"• {name}" for name in names), title="[bold blue]available tools[/bold blue]", border_style="blue"))


def _show_stats(console: Console, agent: Agent) -> None:
    metrics = agent.last_run_metrics
    if not metrics:
        console.print("[dim]No turn has completed yet.[/dim]")
        return
    table = Table(show_header=False, box=None, padding=(0, 1))
    for key in (
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
    ):
        if key in metrics:
            table.add_row(key, str(metrics[key]))
    console.print(Panel(table, title="[bold green]last run metrics[/bold green]", border_style="green"))


def _run_turn(console: Console, agent: Agent, user_text: str) -> None:
    events: queue.Queue[dict] = queue.Queue()
    stop_event = threading.Event()
    holder: dict[str, object] = {}

    task = ContextEvent(
        role=ContextRole.USER,
        type=ContextType.MESSAGE,
        content=user_text,
        priority=ContextPriority.NORMAL,
        step=0,
        metadata={"source": "cli"},
    )

    def emit(event: dict) -> None:
        events.put(event)

    def worker() -> None:
        try:
            holder["result"] = agent.act(
                task,
                on_event=emit,
                stop_event=stop_event,
            )
        except Exception as exc:
            holder["error"] = exc
            events.put({
                "type": "error",
                "message": f"{type(exc).__name__}: {exc}",
            })

    model_name = str(agent.llm.llm_config.get("model_name") or "default")
    renderer = StreamRenderer(
        model=model_name,
        workspace=agent.workingdirectory,
        session_id=str(agent.session_id),
        think_enabled=_think_value(agent),
    )

    thread = threading.Thread(target=worker, name="daena-agent", daemon=True)
    thread.start()

    interrupt_requested = False
    with Live(renderer.render(), console=console, refresh_per_second=10, transient=False) as live:
        while thread.is_alive() or not events.empty():
            try:
                while True:
                    event = events.get_nowait()
                    renderer.handle(event)
            except queue.Empty:
                pass

            live.update(renderer.render())

            if stop_event.is_set():
                renderer.status = "interrupt requested..."

            # Allow operator guidance while the Agent is working. The input
            # reader only feeds a thread-safe queue; Loop consumes it at its
            # own safe iteration boundary, so SQLite/context state stays in
            # the Agent thread.
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0)
                if ready:
                    incoming = sys.stdin.readline().strip()
                    if incoming:
                        if incoming.lower() in {"/interrupt", "/stop", "/cancel"}:
                            stop_event.set()
                            console.print("[bold yellow]↯ Interrupt requested.[/bold yellow]")
                        else:
                            steering = incoming[7:].strip() if incoming.lower().startswith("/steer ") else incoming
                            agent.steer(steering)
                            console.print(f"[bold magenta]➜ steering:[/bold magenta] {steering}")
            except (OSError, ValueError):
                pass

            try:
                time.sleep(0.05)
            except KeyboardInterrupt:
                if not interrupt_requested:
                    stop_event.set()
                    interrupt_requested = True
                    renderer.status = "interrupt requested..."
                    console.print("[bold yellow]↯ Interrupt requested. Daena will stop at the next safe loop boundary.[/bold yellow]")

        while not events.empty():
            renderer.handle(events.get_nowait())
        live.update(renderer.render())

    if "error" in holder:
        console.print(
            Panel(str(holder["error"]), title="[bold red]turn error[/bold red]", border_style="red")
        )
        return

    result = holder.get("result")
    response = getattr(result, "response", None)
    if response:
        console.print(
            Panel(
                Markdown(str(response)),
                title="[bold green]Daena[/bold green]",
                border_style="green",
            )
        )

    metrics = agent.last_run_metrics
    duration = metrics.get("duration_ms", 0)
    tokens = metrics.get("tokens", 0)
    console.print(
        f"[dim]turn complete · {metrics.get('iterations', 0)} iterations · "
        f"{tokens} tokens · {duration:.0f} ms[/dim]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="daena", description="Daena interactive coding-agent CLI")
    parser.add_argument("--cwd", dest="cwd", default=None, help="initial Agent workspace")
    parser.add_argument("--no-think", action="store_true", help="disable Ollama thinking for this session")
    args = parser.parse_args()

    config = _load_config(CONFIG_PATH)
    agent = Agent(config)
    if args.cwd:
        agent.set_workingdirectory(args.cwd)
    if args.no_think:
        _set_think(agent, False)

    console = Console()
    model_name = str(agent.llm.llm_config.get("model_name") or "default")

    console.print(Panel.fit(
        "[bold cyan]DAENA[/bold cyan]  [white]Autonomous Coding Agent[/white]\n"
        f"[dim]model: {model_name} · workspace: {agent.workingdirectory}\n"
        "type /help for commands · /steer text while working · Ctrl+C interrupts · /exit quits[/dim]",
        border_style="cyan",
    ))

    try:
        while True:
            try:
                prompt = console.input("\n[bold cyan]you › [/bold cyan]")
            except KeyboardInterrupt:
                console.print("\n[yellow]Use /exit to leave Daena.[/yellow]")
                continue
            except EOFError:
                break

            text = prompt.strip()
            if not text:
                continue

            if text.startswith("/"):
                parts = text.split(maxsplit=1)
                command = parts[0].lower()
                argument = parts[1].strip() if len(parts) > 1 else ""

                if command in {"/exit", "/quit", "/q"}:
                    break
                if command == "/help":
                    _show_help(console)
                    continue
                if command == "/clear":
                    console.clear()
                    continue
                if command == "/status":
                    _show_status(console, agent)
                    continue
                if command == "/tools":
                    _show_tools(console, agent)
                    continue
                if command == "/stats":
                    _show_stats(console, agent)
                    continue
                if command == "/think":
                    value = argument.lower() or str(_think_value(agent)).lower()
                    if value == "on":
                        value = "medium"
                    elif value == "auto":
                        _set_think(agent, None)
                        console.print("[green]thinking = auto[/green]")
                        continue
                    if value == "off":
                        _set_think(agent, False)
                    elif value in {"low", "medium", "high"}:
                        _set_think(agent, value)
                    else:
                        console.print("[yellow]Usage: /think low|medium|high|off|auto[/yellow]")
                        continue
                    console.print(f"[green]thinking = {value}[/green]")
                    continue
                if command == "/provider":
                    if not argument:
                        _show_provider(console, agent)
                    else:
                        try:
                            _switch_provider(agent, argument)
                            console.print(f"[green]provider → {agent.llm.provider_name}[/green]")
                        except Exception as exc:
                            console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
                    continue
                if command == "/model":
                    if not argument:
                        console.print(str(agent.llm.llm_config.get("model_name") or agent.llm.model.defaultModel))
                    else:
                        agent.llm.llm_config["model_name"] = argument
                        console.print(f"[green]model → {argument}[/green]")
                    continue
                if command == "/set":
                    bits = argument.split(maxsplit=1)
                    if len(bits) != 2:
                        console.print("[yellow]Usage: /set key value[/yellow]")
                    else:
                        try:
                            parsed = _set_setting(agent, bits[0], bits[1])
                            console.print(f"[green]{bits[0]} = {parsed}[/green]")
                        except Exception as exc:
                            console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
                    continue
                if command == "/cwd":
                    if not argument:
                        console.print(agent.workingdirectory)
                    else:
                        agent.set_workingdirectory(argument)
                        console.print(f"[green]workspace → {agent.workingdirectory}[/green]")
                    continue

                console.print(f"[yellow]Unknown command: {command}. Try /help.[/yellow]")
                continue

            _run_turn(console, agent, text)
    finally:
        agent.close()

    console.print("[bold cyan]◈ Daena session closed[/bold cyan]")
