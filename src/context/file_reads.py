"""Exact read receipts: versioned source spans, never semantic task assessments."""

import json


def file_evidence(value):
    if not isinstance(value, dict):
        return None
    evidence = value.get("evidence", value)
    if not isinstance(evidence, dict):
        return None
    if not all(
        k in evidence
        for k in (
            "read_id",
            "file_sha256",
            "path",
            "source_start_offset",
            "source_end_offset",
            "requested_end_offset",
            "end_line",
            "max_output_chars",
        )
    ):
        return None
    if not isinstance(evidence.get("content"), str):
        return None
    if not all(
        isinstance(evidence[k], str) for k in ("read_id", "file_sha256", "path")
    ):
        return None
    if (
        type(evidence["requested_end_offset"]) is not int
        or type(evidence["max_output_chars"]) is not int
    ):
        return None
    start, end = evidence["source_start_offset"], evidence["source_end_offset"]
    if (
        type(start) is not int
        or type(end) is not int
        or start < 0
        or end > evidence["requested_end_offset"]
        or end - start != len(evidence["content"])
    ):
        return None
    return evidence


def trim_file_text(evidence, keep):
    """Update continuation when transport budgeting removes a suffix."""
    evidence["content"] = evidence["content"][:keep]
    end = evidence["source_start_offset"] + len(evidence["content"])
    evidence["source_end_offset"] = end
    evidence["transport_truncated"] = True
    evidence["truncated"] = end < evidence["requested_end_offset"]
    evidence["lines_returned"] = len(evidence["content"].splitlines())
    # A character page can split a physical line; offsets are authoritative.
    evidence["returned_end_line"] = None
    evidence["next_read"] = (
        None
        if not evidence["truncated"]
        else {
            "file_path": evidence["path"],
            "start_offset": end,
            "end_line": evidence["end_line"],
            "expected_sha256": evidence["file_sha256"],
            "max_output_chars": evidence["max_output_chars"],
        }
    )


def delivered_reads(messages):
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            value = json.loads(message.get("content", ""))
        except (ValueError, TypeError):
            continue
        evidence = file_evidence(value)
        if evidence is not None and value.get("success", True):
            yield evidence
