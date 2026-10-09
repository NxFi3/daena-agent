# House Price Dataset – Divar Tehran

This repository contains a small, reproducible pipeline that extracts
real‑estate listings from the Iranian classifieds site **Divar** (Tehran
buy‑residential section).  The goal is to provide a clean CSV that can be
used for downstream analysis or machine‑learning experiments.

## Pipeline Overview

1. **collect_divar.py** – Scrapes the first page from a local copy
   (`page1.html`) and subsequent pages via HTTP.  It extracts the
   `window.__PRELOADED_STATE__` JSON, filters for widgets of type
   `POST_ROW`, and writes the raw JSON to `divar_listings.json`.

2. **clean.py** – Reads `divar_listings.json`, normalises the price
   field (removing Persian numerals and non‑digit characters), removes
   duplicates based on the unique `token`, and writes a CSV
   (`divar_listings_clean.csv`).

3. **price_normalization.py** – Post‑processes the CSV to ensure the
   `price` column contains integer values in Iranian Rial.  The output
   is `divar_listings_normalized.csv`.

4. **tests/test_collection.py** – Simple unit tests that verify:
   * `divar_listings.json` contains at least one record.
   * `divar_listings_clean.csv` has the expected columns.
   * `divar_listings_normalized.csv` contains integer prices.

## How to Run

```bash
python3 collect_divar.py   # fetches pages and writes JSON
python3 clean.py           # cleans and writes CSV
python3 price_normalization.py  # normalises prices
pytest tests/               # run tests
```

All scripts are pure Python and depend only on the standard library
plus `requests` for HTTP fetching.

## Data Dictionary

| Column | Description |
|--------|-------------|
| token | Unique identifier for the listing |
| title | Title shown on the page |
| price | Normalised price in Iranian Rial |
| city | City name |
| district | District name |
| image_url | URL of the first image |
| url | Direct link to the listing |

## License

This dataset is derived from publicly available listings on Divar.  The
scripts are released under the MIT license.
