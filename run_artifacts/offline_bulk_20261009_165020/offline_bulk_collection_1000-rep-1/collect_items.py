#!/usr/bin/env python3
"""Collect items from the local paginated JSON fixture.

The script reads ``fixture_endpoint.json`` to discover the base URL,
endpoint, total pages and request interval.  It then fetches each page,
stores the raw JSON under ``raw_pages/page_XXX.json`` and writes a CSV
``collected_items.csv`` with the required columns.  A log file
``collection.log`` records progress.

The script performs validation:

* HTTP status must be 200.
* JSON must contain a ``records`` list.
* Each record must have a unique ``record_id``.
* The number of rows in the CSV must equal ``total_records``.
* All timestamps are UTC ISO‑8601.

If validation fails the script exits with a non‑zero status.
"""

import json
import csv
import os
import sys
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

FIXTURE = "fixture_endpoint.json"
RAW_DIR = "raw_pages"
CSV_FILE = "collected_items.csv"
LOG_FILE = "collection.log"

REQUIRED_COLUMNS = [
    "record_id",
    "title",
    "category",
    "price",
    "quantity",
    "weight_g",
    "rating",
    "available",
    "region",
    "source_url",
    "retrieved_at",
]


def load_fixture(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_page(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "daena-agent"})
    try:
        with urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} for {url}")
            data = resp.read().decode("utf-8")
            return json.loads(data)
    except HTTPError as e:
        raise RuntimeError(f"HTTP error {e.code} for {url}") from e
    except URLError as e:
        raise RuntimeError(f"URL error for {url}: {e.reason}") from e


def main():
    fixture = load_fixture(FIXTURE)
    base = fixture["base_url"]
    endpoint = fixture["endpoint"]
    total_pages = fixture["total_pages"]
    total_records_expected = fixture["total_records"]
    interval_ms = fixture.get("request_interval_ms", 0)

    os.makedirs(RAW_DIR, exist_ok=True)
    seen_ids = set()
    rows = []

    with open(LOG_FILE, "w", encoding="utf-8") as log:
        for page in range(1, total_pages + 1):
            url = f"{base}{endpoint}?page={page}"
            try:
                data = fetch_page(url)
            except Exception as e:
                log.write(f"Page {page} fetch error: {e}\n")
                sys.exit(1)
            # Save raw
            raw_path = os.path.join(RAW_DIR, f"page_{page:03d}.json")
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            log.write(f"Fetched page {page}\n")
            records = data.get("records")
            if not isinstance(records, list):
                log.write(f"Page {page} missing 'records' list\n")
                sys.exit(1)
            for rec in records:
                rid = rec.get("record_id")
                if rid is None:
                    log.write(f"Record without record_id on page {page}\n")
                    sys.exit(1)
                if rid in seen_ids:
                    log.write(f"Duplicate record_id {rid} on page {page}\n")
                    sys.exit(1)
                seen_ids.add(rid)
                row = {
                    "record_id": rid,
                    "title": rec.get("title", ""),
                    "category": rec.get("category", ""),
                    "price": rec.get("price", ""),
                    "quantity": rec.get("quantity", ""),
                    "weight_g": rec.get("weight_g", ""),
                    "rating": rec.get("rating", ""),
                    "available": str(rec.get("available", "")),
                    "region": rec.get("region", ""),
                    "source_url": rec.get("source_url", ""),
                    "retrieved_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
                }
                rows.append(row)
            time.sleep(interval_ms / 1000.0)

    # Validation
    if len(rows) != total_records_expected:
        print(f"Row count {len(rows)} does not match expected {total_records_expected}")
        sys.exit(1)
    # Write CSV
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Create data dictionary
    with open("data_dictionary.md", "w", encoding="utf-8") as f:
        f.write("# Data Dictionary\n\n")
        f.write(
            "| Column | Type | Meaning | Empty-cell policy |\n"
            "|---|---|---|---|\n"
            "| record_id | string | Unique identifier from source | Must be present |\n"
            "| title | string | Title of item | Empty if omitted |\n"
            "| category | string | Category of item | Empty if omitted |\n"
            "| price | number | Price in USD | Empty if omitted |\n"
            "| quantity | integer | Quantity available | Empty if omitted |\n"
            "| weight_g | number | Weight in grams | Empty if omitted |\n"
            "| rating | number | Rating out of 5 | Empty if omitted |\n"
            "| available | boolean | Availability flag | Empty if omitted |\n"
            "| region | string | Region code | Empty if omitted |\n"
            "| source_url | string | URL of source record | Empty if omitted |\n"
            "| retrieved_at | datetime | UTC ISO‑8601 timestamp | Always present |\n"
        )

    # Create collection report
    with open("collection_report.md", "w", encoding="utf-8") as f:
        f.write("# Collection Report\n\n")
        f.write(f"Pages fetched: {total_pages}\n")
        f.write(f"Source records: {len(rows)}\n")
        f.write(f"Output rows: {len(rows)}\n")
        f.write("Missing-value handling: fields omitted in source are left empty in CSV.\n")
        f.write("Validation performed: HTTP 200, JSON structure, unique record_id, row count match, UTC timestamps.\n")
        f.write("Limitations: assumes source JSON schema as described; no retry logic on transient failures.\n")


if __name__ == "__main__":
    main()
