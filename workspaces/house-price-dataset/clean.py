"""Data cleaning and export for Divar listings.

The script reads the JSON file produced by :mod:`collect_divar` (``divar_listings.json``),
normalises the price field to an integer number of Iranian Rials, removes any
duplicate entries based on the unique ``token`` field, and writes the cleaned
data to ``divar_listings_clean.csv``.

The CSV contains the following columns (in order):

``token``
    Unique identifier for the listing.
``title``
    The title shown on the page.
``price``
    Normalised price as an integer.
``city``
    City name.
``district``
    District name.
``image_url``
    URL of the first image.
``url``
    The URL that the listing links to.

The script is intentionally lightweight and does not depend on any
external libraries beyond the standard library.
"""

import csv
import json
import re
from pathlib import Path

INPUT_JSON = Path("divar_listings.json")
OUTPUT_CSV = Path("divar_listings_clean.csv")


def _parse_price(price_str: str) -> int:
    """Convert a price string like ``"۷,۱۵۰,۰۰۰,۰۰۰ تومان"`` to an int.

    The function removes any non‑digit characters and converts the
    remaining string to an integer.  If the string cannot be parsed,
    ``0`` is returned.
    """
    digits = re.sub(r"[^0-9]", "", price_str)
    return int(digits) if digits else 0


def main() -> None:
    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    seen = set()
    rows = []
    for item in data:
        # Each item is the widget dict; actual data is under dto.data
        d = item.get("dto", {}).get("data", {})
        token = d.get("action", {}).get("payload", {}).get("token")
        if not token or token in seen:
            continue
        seen.add(token)
        title = d.get("title", "")
        price_raw = d.get("middle_description_text", "")
        price = _parse_price(price_raw)
        city = d.get("action", {}).get("payload", {}).get("web_info", {}).get("city_persian", "")
        district = d.get("action", {}).get("payload", {}).get("web_info", {}).get("district_persian", "")
        image_url = d.get("image_url", "")
        url = f"https://divar.ir/s/tehran/buy-residential/{token}"
        rows.append([token, title, price, city, district, image_url, url])

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "title", "price", "city", "district", "image_url", "url"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} cleaned listings to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
