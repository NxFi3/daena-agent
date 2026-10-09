import json
import csv
import os
from pathlib import Path

DATA_DIR = Path(".")


def test_json_has_records():
    json_file = DATA_DIR / "divar_listings.json"
    assert json_file.exists(), "divar_listings.json not found"
    data = json.load(open(json_file, encoding="utf-8"))
    assert isinstance(data, list), "JSON root is not a list"
    assert len(data) >= 24, f"Expected at least 24 records, got {len(data)}"


def test_csv_columns():
    csv_file = DATA_DIR / "divar_listings_clean.csv"
    assert csv_file.exists(), "divar_listings_clean.csv not found"
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"token", "title", "price", "city", "district", "image_url", "url"}
        assert set(reader.fieldnames) == expected, f"CSV columns mismatch: {reader.fieldnames}"
        rows = list(reader)
        assert len(rows) >= 48, f"Expected at least 48 cleaned rows, got {len(rows)}"


def test_price_is_int():
    csv_file = DATA_DIR / "divar_listings_normalized.csv"
    assert csv_file.exists(), "divar_listings_normalized.csv not found"
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                int(row["price"])
            except ValueError:
                assert False, f"Price not integer: {row['price']}"
