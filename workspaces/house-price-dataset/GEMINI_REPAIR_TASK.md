# Gemini repair pass: fix the existing Divar dataset locally

This is a bounded repair task in the current workspace, not a fresh broad research task. Do real code edits and verify the generated files. A plan or status report without code/data changes is a failure.

Use ONLY existing local files and saved HTML/JSON. Do not make HTTP requests, re-run live collection, bypass website controls, or invent records.

Independent audit evidence from the current files:
- divar_listings.json contains 96 raw rows, across pages 1–4, but only 48 distinct listing tokens; duplicate pages/rows exist.
- divar_listings_clean.csv contains 48 rows, but every value in price is "0".
- divar_listings_normalized.csv contains 48 rows, but every value in price is "0".
- Raw source price strings are Persian-digit values with the unit تومان, for example `۷,۱۵۰,۰۰۰,۰۰۰ تومان`. `clean.py` strips everything except ASCII 0-9, so it turns these valid prices into zero.
- These zero prices must never be treated as valid data. Do not convert Toman prices to Rial unless a separate documented output explicitly requires it; keep the primary value in Toman.

Required actions, in this order:
1. In the first two iterations, make concrete edits to `clean.py`, `collect_divar.py`, and `tests/test_collection.py`. Do not spend more iterations merely reading files already identified above.
2. Parse Persian digits ۰–۹ and Arabic-Indic digits ٠–٩ before numeric extraction. Parse only verified monetary values with explicit units; invalid/missing prices must become empty/invalid and make validation fail, never silently become 0.
3. Deduplicate collection output by a stable listing token. In the collector, detect a repeated page token signature and stop rather than repeatedly collecting duplicates. Do not perform any network access in this repair pass; work from saved source data only.
4. Regenerate cleaned/normalized CSVs from the existing raw JSON. Confirm a known example maps exactly: `۷,۱۵۰,۰۰۰,۰۰۰ تومان` -> `7150000000` Toman.
5. Strengthen tests so they fail on zero/missing prices, duplicate listing tokens, and Persian-digit parsing errors. Add `validate_dataset.py` that checks schema, unique tokens, positive numeric prices, and reports explicit counts; it must exit nonzero for invalid outputs.
6. Add `data_dictionary.md`, `collection_manifest.json`, and a concise `README.md` if the core fixes/tests are complete.
7. Run the workspace tests and validator. If anything fails, fix it and run again. Report actual outputs, row counts, duplicate counts, price counts, and limitations. Do not claim the dataset is ML-ready unless the validator and source-level checks support that conclusion.

Focus on useful edits and independent evidence. The current iteration budget is short; avoid unnecessary broad searches or repeated reads.