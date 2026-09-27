"""Data models used by the report generator.

Only the fields that are required for the report are defined.  All
fields are typed with ``str`` or ``dict`` for simplicity.  The
``ToolCallFormat`` and ``ToolResultFormat`` dataclasses capture the
representation details that differ between providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ToolCallFormat:
    """How a tool call is represented in the provider's response.

    Attributes
    ----------
    representation : str
        Human‑readable description of the representation.
    id_field : str
        Name of the field that holds the tool call id.
    arguments : str
        How the arguments are encoded – ``json string`` or ``json
        object``.
    """

    representation: str
    id_field: str
    arguments: str


@dataclass
class ToolResultFormat:
    """How a tool result is represented.

    Attributes
    ----------
    role : str
        The role field used in the response message.
    content : str
        How the content of the tool result is encoded.
    tool_call_id_field : str
        Field that links the result to the originating tool call.
    """

    role: str
    content: str
    tool_call_id_field: str


@dataclass
class Provider:
    """Metadata for a provider.

    Attributes
    ----------
    name : str
        Human‑readable name of the provider.
    doc_url : str
        URL of the official documentation page that was used.
    tool_call_format : ToolCallFormat
    tool_result_format : ToolResultFormat
    """

    name: str
    doc_url: str
    tool_call_format: ToolCallFormat
    tool_result_format: ToolResultFormat

