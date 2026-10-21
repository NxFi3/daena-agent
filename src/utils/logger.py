"""Shared loggers with thread-safe setup and user-owned log storage."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from threading import Lock

_STATE_ROOT = Path(os.getenv("XDG_STATE_HOME", str(Path.home() / ".local" / "state")))
LOG_DIR = str(
    Path(os.getenv("DAENA_LOG_DIR", str(_STATE_ROOT / "daena" / "logs"))).expanduser()
)
_SETUP = Lock()


def get_logger(name="Daena"):
    logger = logging.getLogger(name)
    with _SETUP:
        if logger.handlers:
            return logger
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] → %(message)s", datefmt="%H:%M:%S"
        )
        try:
            directory = Path(LOG_DIR)
            directory.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(
                directory / f"{datetime.now().date()}.log", encoding="utf-8", delay=True
            )
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        except OSError:
            # An unwritable optional log directory must not disable the agent.
            pass
        console = logging.StreamHandler()
        level = os.getenv("DAENA_LOG_LEVEL", "WARNING").upper()
        console.setLevel(getattr(logging, level, logging.WARNING))
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger
