from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from threading import Event, Lock
from typing import Any, Callable
from uuid import uuid4

from src.agent.agentloop import Loop
from src.engine.LlmProviderManager import LlmProvider
from src.models.ContextEvent import ContextEvent


class Agent:
    """Public runtime facade for one Daena session."""

    def __init__(self, config, *, llm=None) -> None:
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        self.config = config
        self._run_lock = Lock()
        self._instruction_lock = Lock()
        self._runtime_instructions: dict[str, str] = {}
        self._closed = False
        workspace_config = config.get("workspace") or {}
        configured_workspace = workspace_config.get("directory")
        self.workingdirectory = str(
            Path(configured_workspace).expanduser().resolve()
            if configured_workspace
            else Path.cwd().resolve()
        )
        self.session_id = uuid4()

        Path(self.workingdirectory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.llm = llm if llm is not None else LlmProvider(self.config)

        self.loop = Loop(
            self.config,
            self.llm,
        )

        self.loop.session_id = self.session_id
        self.loop.set_workspace(self.workingdirectory)
        self.loop.context.contextbuilder.runtime_instruction_source = (
            self.runtime_instructions
        )

    def set_runtime_instruction(self, key: str, text: str | None) -> None:
        """Set a transient instruction, including during an active turn.

        This changes the next provider request without editing configuration,
        conversation history or delegated agents. Passing None removes a mode.
        Callers supply application instructions, never untrusted tool content.
        """
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Runtime instruction key must be a nonempty string")
        if text is not None and not isinstance(text, str):
            raise TypeError("Runtime instruction must be text or None")
        with self._instruction_lock:
            if text and text.strip():
                self._runtime_instructions[key] = text.strip()
            else:
                self._runtime_instructions.pop(key, None)

    def runtime_instructions(self) -> str:
        """Return an immutable snapshot for the context builder's worker thread."""
        with self._instruction_lock:
            return "\n\n".join(self._runtime_instructions.values())

    def act(
        self,
        event: ContextEvent,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        stop_event: Event | None = None,
    ):
        if self._closed:
            raise RuntimeError("Agent is closed")
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError(
                "This Agent is already running; use steer() or a separate Agent"
            )
        try:
            if self._closed:
                raise RuntimeError("Agent is closed")
            # CLI model/provider overrides must also apply to newly delegated
            # agents, which create their own provider instances.
            child_config = deepcopy(self.config)
            if hasattr(self.llm, "provider_name"):
                child_config["llm"] = {
                    "provider": self.llm.provider_name,
                    "provider_config": {
                        **deepcopy(self.llm.llm_config),
                        "generation_config": deepcopy(self.llm.generation_config),
                    },
                }
            self.loop.tool.subagents.config = child_config
            return self.loop.run(
                user_task=event,
                workspace_directory=self.workingdirectory,
                event_sink=on_event,
                stop_event=stop_event,
            )
        finally:
            try:
                self.loop.tool.finish_run()
            finally:
                self._run_lock.release()

    def steer(self, text: str) -> None:
        """Queue guidance for the currently running task."""
        if self._closed:
            raise RuntimeError("Agent is closed")
        self.loop.steer(text)

    def receive_agent_message(self, text: str) -> None:
        """Queue untrusted coordination data for the active runtime turn."""
        self.loop.receive_agent_message(text)

    def approve_background_command(
        self,
        command: list[str],
        workdir: str | None = None,
    ) -> None:
        """Approve one exact background command for this Agent session."""
        self.loop.tool.approve_background_command(
            command=command,
            workdir=workdir,
        )

    def revoke_background_command(
        self,
        command: list[str],
        workdir: str | None = None,
    ) -> None:
        """Revoke one exact background command approval."""
        self.loop.tool.revoke_background_command(
            command=command,
            workdir=workdir,
        )

    def clear_background_approvals(self) -> None:
        """Clear all background command approvals for this session."""
        self.loop.tool.clear_background_approvals()

    def set_workingdirectory(
        self,
        directory: str,
    ) -> None:
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("Cannot change workspace while the Agent is running")
        try:
            if self._closed:
                raise RuntimeError("Agent is closed")
            resolved = str(Path(directory).expanduser().resolve())
            Path(resolved).mkdir(parents=True, exist_ok=True)
            self.loop.set_workspace(resolved)
            self.workingdirectory = resolved
        finally:
            self._run_lock.release()

    @property
    def last_run_metrics(self) -> dict:
        return self.loop.get_metrics()

    def close(self) -> None:
        if self._closed:
            return
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("Stop the running task before closing the Agent")
        try:
            self.loop.close()
            self._closed = True
        finally:
            self._run_lock.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
