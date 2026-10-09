from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from harness.cases import (
    _OFFLINE_BULK_HEADER,
    _OFFLINE_BULK_FIXTURES,
    _cleanup_offline_bulk_collection,
    _setup_offline_bulk_collection,
    _verify_offline_bulk_collection,
)
from harness.runner import summarize


def _materialize_correct_dataset(root: Path) -> None:
    fixture = _OFFLINE_BULK_FIXTURES[root.resolve()]
    endpoint = fixture["endpoint"]
    all_records = []
    for page in range(1, endpoint["total_pages"] + 1):
        with urlopen(f"{endpoint['base_url']}{endpoint['endpoint']}?page={page}", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["page"] == page
        assert payload["total_pages"] == 40
        raw_dir = root / "raw_pages"
        raw_dir.mkdir(exist_ok=True)
        (raw_dir / f"page_{page:03d}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        all_records.extend(payload["records"])

    assert len(all_records) == 1000
    with (root / "collected_items.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_OFFLINE_BULK_HEADER)
        writer.writeheader()
        for record in all_records:
            row = {key: "" for key in _OFFLINE_BULK_HEADER}
            for key in _OFFLINE_BULK_HEADER:
                if key in record:
                    value = record[key]
                    if key == "available":
                        value = str(value).lower()
                    row[key] = value
            row["retrieved_at"] = datetime.now(timezone.utc).isoformat()
            writer.writerow(row)

    dictionary_lines = [
        "# Data Dictionary",
        "",
        "Missing or unprovided fields are represented by blank cells; no default is inferred.",
        "",
    ]
    for column in _OFFLINE_BULK_HEADER:
        dictionary_lines.append(f"- {column}: source field or retrieval metadata; preserved as provided.")
    (root / "data_dictionary.md").write_text("\\n".join(dictionary_lines) + "\\n", encoding="utf-8")
    (root / "collection_report.md").write_text(
        "# Collection Report\\n\\n"
        "Fetched 40 pages containing 1000 source records and wrote 1000 unique CSV rows. "
        "Missing source fields were left blank. Validation checked header, row count, uniqueness, "
        "source fidelity and timezone-aware UTC retrieval timestamps. Raw page responses are in raw_pages/.\\n",
        encoding="utf-8",
    )


def test_offline_bulk_fixture_serves_all_pages_and_verifies_without_internet(tmp_path):
    _setup_offline_bulk_collection(tmp_path)
    try:
        endpoint = json.loads((tmp_path / "fixture_endpoint.json").read_text(encoding="utf-8"))
        assert endpoint["total_pages"] == 40
        assert endpoint["total_records"] == 1000
        assert endpoint["base_url"].startswith("http://127.0.0.1:")
        _materialize_correct_dataset(tmp_path)
        ok, note = _verify_offline_bulk_collection(tmp_path)
        assert ok, note
        assert "1,000 source records verified" in note
        assert any("price" not in record for record in _OFFLINE_BULK_FIXTURES[tmp_path.resolve()]["records"])
    finally:
        _cleanup_offline_bulk_collection(tmp_path)
    assert tmp_path.resolve() not in _OFFLINE_BULK_FIXTURES


def test_offline_bulk_verifier_rejects_fabricated_values_and_naive_timestamps(tmp_path):
    _setup_offline_bulk_collection(tmp_path)
    try:
        _materialize_correct_dataset(tmp_path)
        csv_path = tmp_path / "collected_items.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["price"] = "999999999"
        rows[0]["retrieved_at"] = "2026-10-09T15:00:00"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_OFFLINE_BULK_HEADER)
            writer.writeheader()
            writer.writerows(rows)
        ok, note = _verify_offline_bulk_collection(tmp_path)
        assert not ok
        assert "values differ from the source" in note
        assert "not timezone-aware" in note
    finally:
        _cleanup_offline_bulk_collection(tmp_path)


def test_harness_summary_reports_per_run_failures_and_duplicate_cache_hits():
    record = {
        "verified": True,
        "elapsed_ms": 50,
        "metrics": {
            "completed": True,
            "iterations": 12,
            "llm_calls": 4,
            "tokens": 2300,
            "tool_call_attempts": 8,
            "tool_failures": 2,
            "tool_blocks": 0,
        },
        "run_metrics": {
            "iterations": 12,
            "tool_failures_by_error_type": {
                "shell_syntax_not_supported": 1,
                "old_string_not_found": 1,
            },
            "duplicate_cache_hits": 3,
            "tokens": 2300,
            "wall_time_ms": 50,
            "completed": True,
        },
    }
    summary = summarize([record])
    assert summary["tool_failures_by_error_type"] == {
        "old_string_not_found": 1,
        "shell_syntax_not_supported": 1,
    }
    assert summary["average_duplicate_cache_hits"] == 3
