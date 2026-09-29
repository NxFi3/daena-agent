from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    id: str = ""

    approved: bool = False
    valid: bool = False

    path: str = ""

    args: dict[str, Any] = field(default_factory=dict)

    action: str = "execute"
    target: str = ""

    security_reason: str = ""
    security_rule: str = ""
