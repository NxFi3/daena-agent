"""Worker threads never touch Qt widgets; the UI owns all rendering."""

from __future__ import annotations

from copy import deepcopy
from threading import Event
from PySide6.QtCore import QThread, Signal
from src.agent.agent import Agent
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.studio.blueprint import execute


class RuntimeWorker(QThread):
    event = Signal(dict)
    ready = Signal(object)
    result = Signal(dict)

    def __init__(
        self,
        config,
        prompt,
        *,
        agent=None,
        session_id=None,
        blueprint=None,
        parent=None,
    ):
        super().__init__(parent)
        self.config, self.prompt = deepcopy(config), prompt
        self.agent, self.session_id, self.blueprint = agent, session_id, blueprint
        self.stop = Event()

    def run(self):
        try:
            if self.blueprint is not None:
                result = execute(
                    self.blueprint, self.config, self.stop, self.event.emit
                )
                self.result.emit(
                    {
                        "text": "Automation completed"
                        if result["completed"]
                        else "Automation stopped",
                        "metrics": result,
                    }
                )
                return
            if self.agent is None:
                self.agent = Agent(self.config)
            if self.session_id is not None:
                self.agent.session_id = self.session_id
                self.agent.loop.session_id = self.session_id
            self.ready.emit(self.agent)
            result = self.agent.act(
                ContextEvent(
                    role=ContextRole.USER, type=ContextType.MESSAGE, content=self.prompt
                ),
                stop_event=self.stop,
                on_event=self.event.emit,
            )
            self.result.emit(
                {"text": result.response, "metrics": self.agent.last_run_metrics}
            )
        except Exception as exc:
            self.result.emit(
                {
                    "text": f"{type(exc).__name__}: {exc}",
                    "metrics": {"completed": False, "stop_reason": str(exc)},
                }
            )
