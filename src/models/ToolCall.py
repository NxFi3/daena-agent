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

    # Populated by ToolDispatcher when a model call cannot be normalized or
    # validated. Keeping the reason on the canonical call lets the loop return
    # a precise structured tool error instead of the generic "invalid call".
    validation_error: str = ""

    # Human-readable notes about safe runtime normalization, such as clamping
    # an out-of-range optional integer to the tool's declared bounds.
    normalization_notes: list[str] = field(default_factory=list)
