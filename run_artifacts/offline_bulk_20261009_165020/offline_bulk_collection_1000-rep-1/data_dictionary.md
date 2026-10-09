# Data Dictionary

| Column | Type | Meaning | Empty-cell policy |
|---|---|---|---|
| record_id | string | Unique identifier from source | Must be present |
| title | string | Title of item | Empty if omitted |
| category | string | Category of item | Empty if omitted |
| price | number | Price in USD | Empty if omitted |
| quantity | integer | Quantity available | Empty if omitted |
| weight_g | number | Weight in grams | Empty if omitted |
| rating | number | Rating out of 5 | Empty if omitted |
| available | boolean | Availability flag | Empty if omitted |
| region | string | Region code | Empty if omitted |
| source_url | string | URL of source record | Empty if omitted |
| retrieved_at | datetime | UTC ISO‑8601 timestamp | Always present |
