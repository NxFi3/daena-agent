from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.agent.agentloop import Loop
from src.engine.LlmProviderManager import LlmProvider
from src.models.ContextEvent import ContextEvent


class Agent:
    """Public runtime facade for one Daena session."""

    def __init__(self, config) -> None:
        self.config = config
        self.workingdirectory = "EvanaEval"
        self.session_id = uuid4()

        Path(self.workingdirectory).mkdir(parents=True, exist_ok=True)

        self.llm = LlmProvider(self.config)
        self.loop = Loop(self.config, self.llm)
        self.loop.session_id = self.session_id
        self.loop.set_workspace(self.workingdirectory)

    def act(self, event: ContextEvent):
        return self.loop.run(
            user_task=event,
            workspace_directory=self.workingdirectory,
        )

    def set_workingdirectory(self, directory: str) -> None:
        self.workingdirectory = str(
            Path(directory).expanduser().resolve()
        )
        Path(self.workingdirectory).mkdir(parents=True, exist_ok=True)
        self.loop.set_workspace(self.workingdirectory)

    @property
    def last_run_metrics(self) -> dict:
        return self.loop.get_metrics()

    def close(self) -> None:
        self.loop.close()
