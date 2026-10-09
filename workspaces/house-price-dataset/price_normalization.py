"""Normalize price values in the cleaned CSV.

The original CSV contains price strings that may include Persian
numerals, commas, or the word "مبلغ".  This script converts the price to an
integer number of Iranian Rial.  Non‑numeric values are set to 0.
"""

import csv
import re
from pathlib import Path

INPUT_CSV = Path("divar_listings_clean.csv")
OUTPUT_CSV = Path("divar_listings_normalized.csv")

PERSIAN_NUM_MAP = {
    "۰": "0",
    "۱": "1",
    "۲": "2",
    "۳": "3",
    "۴": "4",
    "۵": "5",
    "۶": "6",
    "۷": "7",
    "۸": "8",
    "۹": "9",
}


def normalize_price(price_str: str) -> int:
    if not price_str:
        return 0
    # Replace Persian digits
    for p, d in PERSIAN_NUM_MAP.items():
        price_str = price_str.replace(p, d)
    # Remove non‑digits
    digits = re.sub(r"[^0-9]", "", price_str)
    return int(digits) if digits else 0


def main() -> None:
    rows = []
    with INPUT_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["price"] = str(normalize_price(row["price"]))
            rows.append(row)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Normalized {len(rows)} records to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
