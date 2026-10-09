# Daena task: Divar residential sale-price dataset (end-to-end)

You are an autonomous data engineer and ML dataset developer. Execute this task in the current workspace. Do not stop at a plan, a list of suggestions, or an empty scaffold. Inspect real sources, build scripts, run them, validate the actual files independently, fix defects, rerun as needed, and report factual results.

## Goal
Create a trustworthy, reproducible tabular dataset of real residential property-for-sale listings from Divar (divar.ir), prioritizing Tehran and neighborhoods where the source supports them. Make it useful for exploratory analysis and future machine-learning work. Never generate synthetic or fabricated records to fill gaps.

## 1. Investigate actual Divar access
- Inspect the current public Divar website and relevant official documentation using the available web_search and web_fetch tools. A prior failure of one tool call is not proof that all network access is unavailable: inspect the actual error and try another appropriate permitted method.
- Fetch the public homepage and then find/inspect actual residential sale category/search pages. Base URLs, parameters, categories, pagination, and source schema on observed pages/responses; do not invent them.
- Inspect whether documented API access and valid credentials are actually configured. Never print secret values. Use only credentials already present and authorized.
- Respect access rules, rate limits, privacy, and applicable site policies. Do not bypass login, CAPTCHA, access controls, anti-bot measures, or rate limits. Do not use an undocumented private interface as a workaround for unavailable authorized access.
- Do a small real-source connectivity/schema test before collection. If direct bulk access is blocked, try reasonable authorized alternatives and complete all useful work possible; state the precise verified blocker rather than declaring that the whole environment has no internet after one tool failure.
- Prefer genuine Divar listings. Do not silently replace them with an unrelated or synthetic source. If an alternative public historical dataset becomes necessary, explain the change and keep it separate from any live Divar collection.

## 2. Build a reproducible project
Create a sensible, not overengineered Python project with:
- collect_divar.py — repeatable collection from observed, authorized sources.
- process_dataset.py — cleaning, normalization, filters, and derived features.
- validate_dataset.py — independent validation by re-reading actual outputs and comparing them with stored source observations.
- tests/ — tests for parsing, price handling, filtering, deduplication, and derived fields.
- data/ — actual collected and processed data.
- collection.log — useful progress messages and failures.
- collection_manifest.json — source, retrieval time, filters, scope, run ID, page counts, row counts, and configuration.
- data_dictionary.md — fields, meaning, source mapping, types, units, missingness, transformations, and formulas.
- docs.md — full human-readable explanation of the real resulting dataset.
- README.md — environment setup, dependencies, project structure, and reproducible commands.

Inspect existing workspace content first. Keep source, normalized fields, and derived fields distinguishable. Save permitted raw responses or a privacy-conscious source snapshot sufficient to audit the processing. Do not collect unnecessary personal details such as phone numbers.

## 3. Source collection and reliability
- Discover the real schema from observed source responses.
- Collect only the pages and records legitimately accessible under actual site/API constraints.
- Handle transient errors, timeouts, malformed data, incomplete responses, and missing fields.
- Use bounded retries for transient failures, conservative request intervals, and pagination-loop detection.
- Avoid repeated downloads where reliable identifiers make deduplication possible.
- Log attempted, successful, failed, skipped, and processed records.
- Preserve provenance for every output row when possible: source, listing identifier/token, source URL, retrieval timestamp, and run identifier.
- Never claim HTTP success proves collection correctness.
- Never route bulk source data through the model context manually; have the script fetch and process it.
- Do not hand-edit generated CSVs. Fix the generator and rerun it.

## 4. Residential sale schema
First inspect available fields; the following are candidate fields, not assumptions about the source:
- listing_id or token, listing_url, source, source_category
- city, district/neighborhood, observed/retrieved time, published time if actually available
- property_type, area_m2, bedrooms, construction_year, floor, building_floors, parking, elevator, storage, other real available property attributes
- original_price_text, original_price_value, original_price_unit, sale_price_toman, price_per_sqm_toman
- price_validation_status, price_anomaly_flag

Keep residential SALE listings as the primary dataset. Keep rental observations separate if collected. Exclude land, commercial property, and unrelated categories from the primary residential-sale dataset; report excluded counts and reasons.

Do not infer missing area, property type, location, price, identifiers, or dates. Leave unavailable values blank/null. Preserve original price representation when practical. Do not assume a currency/unit from a number alone. Never mix total sale price, monthly rent, rental deposit, and other pricing modes. If unit/currency conversion is uncertain, preserve the original and leave the normalized amount unavailable.

## 5. Price and data-quality validation
Implement and run explicit checks:
- Each row maps to an observed source listing and intended residential-sale category.
- Detect duplicate identifiers and repeated source records.
- Parse localized digits/separators carefully; distinguish missing, undisclosed, negotiable, malformed, and valid prices where supported by the source.
- Never turn undisclosed prices into zero.
- Validate property area and flag implausible/malformed values.
- Compute price_per_sqm_toman only if the record has a verified total sale price in toman and a valid area in square metres.
- Never compute price per square metre using rent or a rental deposit.
- Independently recalculate every derived price value during validation.
- Report descriptive statistics and flag suspicious total price, area, and per-square-metre outliers with explainable, robust rules. Do not automatically discard a valid expensive/inexpensive property merely because it is an outlier; preserve it and flag it unless a documented exclusion rule applies.
- Verify missing source values remain missing and normalization agrees with source values.
- Keep a row-count accounting: fetched, parsed, kept, duplicated, filtered out, and usable for each key price calculation.
- Validate the actual final CSV by reading it back with an independent validator, not merely by trusting the collector or its success message.

## 6. ML suitability
- Use consistent column names, types, units, and missing-value semantics for Pandas and common ML tools.
- Separate source fields, normalized fields, and derived fields.
- Keep provenance available for auditing, but identify identifiers, URLs, and retrieval metadata that must not be used as default model features.
- Analyze missingness and category distributions.
- Explain potential target leakage and duplicated/near-duplicate listing risks.
- If published dates are available and coverage is suitable, recommend time-based train/validation/test splits; do not claim predictive validity from an inadequate sample.
- Clearly state that advertised asking price is not necessarily the final transaction price and that the collection may not represent the entire housing market.
- Do not train an unnecessary model before delivering a validated dataset. An optional baseline is allowed only if there is enough verified real data and it is clearly labelled.

## 7. docs.md must describe actual results
Include:
- purpose, actual source and access method
- retrieval date/period and geographic/category coverage
- exact filters and exclusion counts
- fetched, parsed, retained, duplicate, and excluded counts
- counts with valid sale price, valid area, and valid price per square metre
- units and currency-normalization rules
- key fields, data types, missingness, deduplication, and outlier handling
- independent validation and test results
- how to reproduce collection, processing, and validation
- dependencies and configuration needed
- biases, limitations, legal/privacy/source-access considerations
- recommended future ML targets/features and risks
- a clear distinction between verified facts and assumptions

Do not invent statistics or write success claims before examining results. Generate documentation from the actual run where practical.

## 8. Execute the full end-to-end workflow
1. Inspect the workspace and source.
2. Test real access and inspect schema.
3. Implement the collection script and a small test run.
4. Run the actual permitted collection.
5. Process data and create the residential-sale dataset.
6. Run independent validation and automated tests.
7. Inspect the final CSV and report.
8. Fix concrete issues in the pipeline and rerun it.
9. Ensure docs, manifest, logs, and data refer to the same run.
10. Report what actually worked, what was inaccessible, exact usable record counts, key price-validation counts, test results, and remaining limitations.

If a real external restriction blocks bulk collection, do not bypass it and do not fabricate a dataset. Still implement and test all useful pipeline components and provide an honest partial result plus the specific access requirement needed to finish.

Begin now. Use evidence and execute; do not decline the task simply because the workspace starts empty or one web tool call fails.
