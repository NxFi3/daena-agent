#src/Engine/llmManagment/ProvidersInput.py 

from dataclasses import dataclass, field
from typing import Any, Callable, Dict
import numpy as np


@dataclass
class LLMInput:
    model_name: str
    messages:list[Dict[str, Any]]
    tools: list[Dict] = field(default_factory=list)
    images: list[np.ndarray] = field(default_factory=list)
    options: Dict = field(default_factory=dict)
    stream_callback: Callable[[dict[str, Any]], None] | None = None
