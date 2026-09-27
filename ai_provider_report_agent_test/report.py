"""Report generation utilities.

The report is intentionally simple – it lists each provider and the
representation details that were extracted from the official
documentation.  The comparison function highlights the key differences
between the providers.
"""

from __future__ import annotations

from typing import List

from .models import Provider
from .sources import providers


def build_report() -> str:
    """Return a human‑readable report for all providers.

    The report is a plain text string that includes the provider name,
    documentation URL, tool‑call representation, tool‑call id field,
    argument representation, tool result role, content representation and
    the tool‑call id field used in the result.
    """
    lines: List[str] = []
    for p in providers:
        lines.append(f"Provider: {p.name}")
        lines.append(f"  Documentation: {p.doc_url}")
        lines.append("  Tool Call Format:")
        lines.append(f"    Representation: {p.tool_call_format.representation}")
        lines.append(f"    ID Field: {p.tool_call_format.id_field}")
        lines.append(f"    Arguments: {p.tool_call_format.arguments}")
        lines.append("  Tool Result Format:")
        lines.append(f"    Role: {p.tool_result_format.role}")
        lines.append(f"    Content: {p.tool_result_format.content}")
        lines.append(f"    Tool Call ID Field: {p.tool_result_format.tool_call_id_field}")
        lines.append("")
    return "\n".join(lines)


def compare_providers() -> str:
    """Return a concise comparison of the three providers.

    The comparison focuses on the differences that are relevant for a
    provider‑neutral agent runtime: the type of the tool‑call ID field,
    whether arguments are JSON strings or objects, and the format of the
    tool result content.
    """
    # Build a table of key attributes
    header = ["Provider", "ID Field", "Args", "Result Content"]
    rows = []
    for p in providers:
        rows.append(
            [
                p.name,
                p.tool_call_format.id_field,
                p.tool_call_format.arguments,
                p.tool_result_format.content,
            ]
        )
    # Format as a simple table
    col_widths = [max(len(row[i]) for row in ([header] + rows)) for i in range(len(header))]
    lines = [" | ".join(f"{header[i].ljust(col_widths[i])}" for i in range(len(header)))]
    lines.append("-+-".join("-" * w for w in col_widths))
    for row in rows:
        lines.append(" | ".join(f"{row[i].ljust(col_widths[i])}" for i in range(len(row))))
    return "\n".join(lines)

