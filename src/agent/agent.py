# src.agent/agent.py
from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4
from src.agent.agentloop import Loop
from src.engine.LlmProviderManager import LlmProvider
from src.models.ContextEvent import ContextEvent
from src.utils.logger import get_logger


class Agent:

    def __init__(
        self,
        config,
    ) -> None:

        self.config = config

        self.logger = get_logger("[AGENT]")

        self.workingdirectory = "EvanaEval"
        self.session_id = uuid4()
        Path(self.workingdirectory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.llm = LlmProvider(self.config)

        self.loop = Loop(
            self.config,
            self.llm,
        )
        self.loop.session_id = self.session_id

    def act(
        self,
        event: ContextEvent,
    ):

        return self.loop.run(
            user_task=event,
            workspace_directory=self.workingdirectory,
        )

    def set_workingdirectory(
        self,
        directory: str,
    ) -> None:

        self.workingdirectory = directory

        Path(self.workingdirectory).mkdir(
            parents=True,
            exist_ok=True,
        )

    def close(self) -> None:

        self.loop.close()
