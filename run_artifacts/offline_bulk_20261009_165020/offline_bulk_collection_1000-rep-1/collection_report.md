# Collection Report

Pages fetched: 40
Source records: 1000
Output rows: 1000
Missing-value handling: fields omitted in source are left empty in CSV.
Validation performed: HTTP 200, JSON structure, unique record_id, row count match, UTC timestamps.
Limitations: assumes source JSON schema as described; no retry logic on transient failures.
