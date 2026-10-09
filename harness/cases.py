from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from typing import Callable
from urllib.parse import parse_qs, urlsplit


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    prompt: str
    verifier: Callable[[Path], tuple[bool, str]]
    setup: Callable[[Path], None] | None = None
    tags: tuple[str, ...] = ()
    cleanup: Callable[[Path], None] | None = None



_OFFLINE_BULK_FIXTURES: dict[Path, dict] = {}
_OFFLINE_BULK_HEADER = [
    "record_id", "title", "category", "price", "quantity", "weight_g",
    "rating", "available", "region", "source_url", "retrieved_at",
]


def _offline_bulk_records(base_url: str) -> list[dict]:
    categories = ("office", "garden", "kitchen", "books", "tools")
    regions = ("north", "south", "east", "west", "central")
    records: list[dict] = []
    for index in range(1, 1001):
        record = {
            "record_id": f"SKU-{index:05d}",
            "title": f"Fixture item {index:05d}",
            "category": categories[(index * 7) % len(categories)],
            "source_url": f"{base_url}/items/SKU-{index:05d}",
        }
        if index % 17:
            record["price"] = 499 + index * 37
        if index % 13:
            record["quantity"] = index % 41
        if index % 11:
            record["weight_g"] = 25 + (index * 19) % 2500
        if index % 23:
            record["rating"] = round(1.0 + (index % 40) / 10.0, 1)
        if index % 19:
            record["available"] = index % 3 != 0
        if index % 9:
            record["region"] = regions[(index * 3) % len(regions)]
        records.append(record)
    return records


def _setup_offline_bulk_collection(path: Path) -> None:
    """Start a deterministic, local-only paginated API with 1,000 records."""
    path.mkdir(parents=True, exist_ok=True)

    class FixtureHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path == "/health":
                payload = {"success": True}
                status = 200
            elif parsed.path == "/api/items":
                try:
                    page = int(parse_qs(parsed.query).get("page", ["1"])[0])
                except (TypeError, ValueError):
                    page = 0
                records = self.server.fixture_records
                page_size = self.server.fixture_page_size
                total_pages = (len(records) + page_size - 1) // page_size
                if page < 1 or page > total_pages:
                    payload = {"error": "page out of range", "total_pages": total_pages}
                    status = 400
                else:
                    start = (page - 1) * page_size
                    payload = {
                        "page": page,
                        "page_size": page_size,
                        "total_pages": total_pages,
                        "total_records": len(records),
                        "records": records[start:start + page_size],
                    }
                    status = 200
            elif parsed.path.startswith("/items/"):
                record_id = parsed.path.rsplit("/", 1)[-1]
                payload = next(
                    (item for item in self.server.fixture_records
                     if item["record_id"] == record_id),
                    {"error": "not found"},
                )
                status = 200 if "record_id" in payload else 404
            else:
                payload = {"error": "not found"}
                status = 404

            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    server.daemon_threads = True
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    server.fixture_page_size = 25
    server.fixture_records = _offline_bulk_records(base_url)
    server_thread = __import__("threading").Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.05},
        name="daena-offline-bulk-fixture",
        daemon=True,
    )
    server_thread.start()

    endpoint = {
        "base_url": base_url,
        "endpoint": "/api/items",
        "page_size": 25,
        "total_pages": 40,
        "total_records": 1000,
        "request_interval_ms": 5,
    }
    (path / "fixture_endpoint.json").write_text(
        json.dumps(endpoint, indent=2), encoding="utf-8"
    )
    _OFFLINE_BULK_FIXTURES[path.resolve()] = {
        "server": server,
        "thread": server_thread,
        "records": server.fixture_records,
        "endpoint": endpoint,
    }


def _cleanup_offline_bulk_collection(path: Path) -> None:
    fixture = _OFFLINE_BULK_FIXTURES.pop(path.resolve(), None)
    if fixture is None:
        return
    server = fixture.get("server")
    if server is not None:
        server.shutdown()
        server.server_close()
    thread = fixture.get("thread")
    if thread is not None:
        thread.join(timeout=2)


def _verify_offline_bulk_collection(path: Path) -> tuple[bool, str]:
    fixture = _OFFLINE_BULK_FIXTURES.get(path.resolve())
    if fixture is None:
        return False, "offline fixture state is unavailable; setup must run before verification"

    findings: list[str] = []
    dataset_path = path / "collected_items.csv"
    try:
        with dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        return False, f"could not read collected_items.csv: {exc}"

    expected_records = fixture["records"]
    if header != _OFFLINE_BULK_HEADER:
        findings.append(f"header mismatch: expected {_OFFLINE_BULK_HEADER}, got {header}")
    if len(rows) != len(expected_records):
        findings.append(f"row count mismatch: expected {len(expected_records)}, got {len(rows)}")

    ids = [str(row.get("record_id", "")).strip() for row in rows]
    if len(ids) != len(set(ids)):
        findings.append("record_id contains duplicates")
    expected_by_id = {record["record_id"]: record for record in expected_records}
    if set(ids) != set(expected_by_id):
        missing = len(set(expected_by_id) - set(ids))
        extra = len(set(ids) - set(expected_by_id))
        findings.append(f"record identity mismatch: {missing} missing, {extra} unknown")

    field_map = {
        "record_id": "record_id",
        "title": "title",
        "category": "category",
        "price": "price",
        "quantity": "quantity",
        "weight_g": "weight_g",
        "rating": "rating",
        "available": "available",
        "region": "region",
        "source_url": "source_url",
    }
    fabricated: list[str] = []
    missing_source_values: list[str] = []
    invalid_timestamps: list[str] = []
    for row_index, row in enumerate(rows, start=2):
        record_id = str(row.get("record_id", "")).strip()
        source = expected_by_id.get(record_id)
        if source is None:
            continue
        for column, source_key in field_map.items():
            value = str(row.get(column, "") or "").strip()
            if source_key not in source:
                if value:
                    missing_source_values.append(f"row {row_index} column {column}")
            else:
                expected = source[source_key]
                if column == "available":
                    valid = value.lower() == str(expected).lower()
                else:
                    valid = value == str(expected)
                if not valid:
                    fabricated.append(f"row {row_index} column {column}")
        timestamp = str(row.get("retrieved_at", "") or "").strip()
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("timezone missing")
        except (TypeError, ValueError):
            invalid_timestamps.append(f"row {row_index}")

    if fabricated:
        findings.append(f"values differ from the source on {len(fabricated)} field(s), e.g. {fabricated[:5]}")
    if missing_source_values:
        findings.append(
            f"non-empty values were invented for missing source fields ({len(missing_source_values)}), "
            f"e.g. {missing_source_values[:5]}"
        )
    if invalid_timestamps:
        findings.append(f"{len(invalid_timestamps)} retrieved_at timestamp(s) are not timezone-aware")

    dictionary_path = path / "data_dictionary.md"
    report_path = path / "collection_report.md"
    try:
        dictionary = dictionary_path.read_text(encoding="utf-8")
    except OSError:
        dictionary = ""
    try:
        report = report_path.read_text(encoding="utf-8")
    except OSError:
        report = ""
    if not dictionary.strip():
        findings.append("data_dictionary.md is missing or empty")
    else:
        undocumented = [column for column in _OFFLINE_BULK_HEADER if column not in dictionary]
        if undocumented:
            findings.append(f"data_dictionary.md does not document: {undocumented}")
        if "blank" not in dictionary.lower() and "empty" not in dictionary.lower():
            findings.append("data_dictionary.md does not explain how missing fields are represented")
    if not report.strip():
        findings.append("collection_report.md is missing or empty")
    else:
        if str(len(expected_records)) not in report:
            findings.append("collection_report.md does not state the expected record count")
        if str(fixture["endpoint"]["total_pages"]) not in report:
            findings.append("collection_report.md does not state the page count")

    ok = not findings
    if ok:
        return True, "1,000 source records verified; header, uniqueness, source fidelity, missing values, timezone-aware timestamps and documentation passed"
    return False, "; ".join(findings[:12])


_OFFLINE_BULK_PROMPT = """Collect every record from the offline paginated JSON fixture and produce a verified dataset.
First read fixture_endpoint.json to obtain base_url, endpoint, total_pages and the expected total_records. This server is local to this benchmark; do not use external websites or web search.
Use write_file to create a reusable Python script, then run it with command_exec. Fetch pages 1 through total_pages with urllib, wait at least request_interval_ms between requests, validate each response, and stop on unexpected status or malformed data. Save each raw response under raw_pages/page_NNN.json and append a concise progress line to collection.log.
Create collected_items.csv with exactly these columns in this order:
record_id,title,category,price,quantity,weight_g,rating,available,region,source_url,retrieved_at
Map the source fields exactly. If a source record omits a field, leave the corresponding CSV cell empty; never guess, infer, or fill a default. Emit available as true/false when present. retrieved_at is collection metadata, not a source field: generate a timezone-aware UTC ISO-8601 timestamp for each row. Deduplicate only by record_id and ensure every source record is represented exactly once.
Also create data_dictionary.md documenting every column, types, meaning, and the empty-cell policy; create collection_report.md stating pages fetched, source records, output rows, missing-value handling, validation performed, and limitations. Validate header, field count, row count against total_records, unique IDs, source-value fidelity, blanks for omitted fields, and timezone-aware timestamps before reporting completion. Fix the generating script and rerun it if validation fails; do not hand-edit generated CSV data."""



def _verify_file(path: Path, expected: str) -> tuple[bool, str]:
    try:
        value = (path / "result.txt").read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"missing result.txt: {exc}"

    ok = value.strip() == expected
    return ok, "result.txt matches expected content" if ok else "result.txt mismatch"


def _setup_empty(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _security_source() -> Path:
    import os

    override = os.environ.get("DAENA_SECURITY_BENCHMARK")
    if override:
        return Path(override).expanduser().resolve()
    # This checkout lives beside EVOLUTION/test101 in the user's workspace.
    return Path(__file__).resolve().parents[2] / "EVOLUTION" / "test101"


def _setup_security_audit(path: Path) -> None:
    source = _security_source()
    if not source.is_dir():
        raise FileNotFoundError(
            "Security benchmark source not found. Set DAENA_SECURITY_BENCHMARK "
            f"to the test101 directory. Checked: {source}"
        )
    shutil.copytree(
        source,
        path,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", ".pytest_cache", "*.pyc", "app.db", "tests",
            ".daena", "AGENTS.md"
        ),
    )
    visible_tests = path / "tests"
    visible_tests.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "tests" / "test_regression.py", visible_tests / "test_regression.py")

    # Keep the full verifier outside the agent workspace. The model sees only
    # normal regression tests, not the security oracle or its test names.
    oracle_tests = path.parent / "oracle-tests"
    shutil.copytree(source / "tests", oracle_tests, dirs_exist_ok=True)
    hashes = {
        "test_regression.py": hashlib.sha256(
            (source / "tests" / "test_regression.py").read_bytes()
        ).hexdigest()
    }
    (path.parent / "security-oracle.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8"
    )


def _verify_security_audit(path: Path) -> tuple[bool, str]:
    oracle_dir = path.parent / "oracle-tests"
    manifest_path = path.parent / "security-oracle.json"
    integrity_ok = False
    integrity_note = "oracle manifest missing"
    try:
        expected_hashes = json.loads(manifest_path.read_text(encoding="utf-8"))
        visible_regression = path / "tests" / "test_regression.py"
        actual_hashes = {
            "test_regression.py": hashlib.sha256(visible_regression.read_bytes()).hexdigest()
        } if visible_regression.is_file() else {}
        integrity_ok = actual_hashes == expected_hashes
        integrity_note = (
            "workspace tests unchanged"
            if integrity_ok
            else "workspace tests changed; protected oracle remains authoritative"
        )
    except (OSError, json.JSONDecodeError) as exc:
        integrity_note = f"test integrity check failed: {exc}"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(path) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(oracle_dir)],
            cwd=path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "protected security/regression suite timed out after 120 seconds"
    output = (result.stdout + "\n" + result.stderr).strip()
    note = (
        f"protected pytest exit={result.returncode}; {integrity_note}\n"
        f"{output[-1800:]}"
    )
    return result.returncode == 0 and integrity_ok, note


CASES = (
    BenchmarkCase(
        name="basic_file_creation",
        prompt=(
            "Create a file named result.txt in the workspace containing exactly "
            "the single line: baseline-ok. Verify it by reading the file."
        ),
        verifier=lambda path: _verify_file(path, "baseline-ok"),
        setup=_setup_empty,
        tags=("tool_use", "filesystem"),
    ),
    BenchmarkCase(
        name="basic_file_repair",
        prompt=(
            "Create result.txt with the wrong content, then repair it so it "
            "contains exactly: repaired. Verify the final file."
        ),
        verifier=lambda path: _verify_file(path, "repaired"),
        setup=lambda path: (
            path.mkdir(parents=True, exist_ok=True),
            (path / "result.txt").write_text("wrong", encoding="utf-8"),
        ),
        tags=("tool_use", "recovery"),
    ),
    BenchmarkCase(
        name="offline_bulk_collection_1000",
        prompt=_OFFLINE_BULK_PROMPT,
        verifier=_verify_offline_bulk_collection,
        setup=_setup_offline_bulk_collection,
        tags=("offline", "bulk_collection", "data_quality", "tool_use"),
        cleanup=_cleanup_offline_bulk_collection,
    ),
    BenchmarkCase(
        name="security_audit_test101",
        prompt=(
            "You are auditing a local Flask application for security issues. "
            "Work autonomously and stay strictly inside this workspace. Inspect "
            "the whole source tree and use safe local-only HTTP or test-client "
            "probes where useful. Discover the security weaknesses yourself; "
            "determine root causes, verify them locally, fix the root causes "
            "without removing legitimate functionality, and rerun verification. "
            "Do not modify tests to hide failures or replace functionality with "
            "dummy behavior. Continue until all significant issues are addressed. "
            "Run the complete regression and security test suite, then report "
            "each issue, root cause, fix, and verification result."
        ),
        verifier=_verify_security_audit,
        setup=_setup_security_audit,
        tags=("security", "web", "autonomous_coding", "verification"),
    ),
)
