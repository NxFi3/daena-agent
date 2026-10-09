"""Collect listings from Divar Tehran buy‑residential pages.

The collector uses the same extraction logic as :mod:`inspect_state`:
it looks for a ``window.__PRELOADED_STATE__`` JavaScript assignment,
decodes the JSON, and extracts widgets of type ``POST_ROW``.  The
first page is read from the local ``page1.html`` file; subsequent
pages are fetched via HTTP until no more items are returned.  The
collected data is written to ``divar_listings.json``.
"""

import json
import re
import requests
from pathlib import Path

BASE_URL = "https://divar.ir/s/tehran/buy-residential"


def _extract_preloaded_state(html: str) -> dict:
    """Return the JSON object from ``window.__PRELOADED_STATE__``.

    The payload is a JavaScript assignment followed by a semicolon.
    ``json.JSONDecoder().raw_decode`` is used to avoid pulling the
    surrounding text into the JSON parser.
    """
    pattern = r"window\.\_\_PRELOADED_STATE__\s*=\s*({.*?});"
    m = re.search(pattern, html, re.S)
    if not m:
        raise ValueError("PRELOADED_STATE not found")
    json_text = m.group(1)
    return json.JSONDecoder().raw_decode(json_text)[0]


def _post_rows(state: dict) -> list:
    """Return a list of POST_ROW widget data from the state."""
    widgets = state.get("nb", {}).get("listWidgets", [])
    return [w["data"] for w in widgets if w.get("data", {}).get("widgetType") == "POST_ROW"]


def fetch_page(page: int) -> list:
    if page == 1:
        html = Path("page1.html").read_text(encoding="utf-8")
    else:
        url = f"{BASE_URL}?page={page}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        html = r.text
    state = _extract_preloaded_state(html)
    return _post_rows(state)


def main() -> None:
    all_listings: list = []
    page = 1
    while True:
        listings = fetch_page(page)
        if not listings:
            break
        all_listings.extend(listings)
        print(f"Page {page} fetched {len(listings)} items, total {len(all_listings)}")
        page += 1
    Path("divar_listings.json").write_text(json.dumps(all_listings, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", len(all_listings), "listings")


if __name__ == "__main__":
    main()
