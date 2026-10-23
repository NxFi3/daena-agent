"""Keep desktop receipts valid JSON inside the host's 8 KB tool-message limit."""

from copy import deepcopy
import json


def encoded_size(value):
    return len(json.dumps(value, ensure_ascii=False))


def element_page(elements, *, offset=0, limit=40, budget=3500):
    shown = []
    for element in elements[offset : offset + limit]:
        row = element.public()
        if shown and encoded_size([*shown, row]) > budget:
            break
        shown.append(row)
    end = offset + len(shown)
    return {
        "elements": shown,
        "matched_count": len(elements),
        "shown_count": len(shown),
        "offset": offset,
        "next_offset": end if end < len(elements) else None,
        "truncated": end < len(elements) or offset > 0,
    }


def compact_evidence(content, budget=6000):
    result = deepcopy(content)
    # Backend receipts and duplicate raw-image hashes are kept in content/trace.
    result.pop("backend_receipt", None)

    def bound(value):
        if isinstance(value, dict):
            return {k: bound(v) for k, v in value.items()}
        if isinstance(value, list):
            return [bound(v) for v in value[:100]]
        if isinstance(value, str) and len(value) > 900:
            return value[:900] + "… [truncated]"
        return value

    result = bound(result)
    while encoded_size(result) > budget:
        choices = []

        def collect(value):
            if not isinstance(value, dict):
                return
            for key, item in value.items():
                if isinstance(item, list) and item:
                    choices.append((encoded_size(item), value, key))
                elif isinstance(item, dict):
                    collect(item)

        collect(result)
        if not choices:
            # Preserve the action, IDs and failure cause even for oversized diagnostics.
            return {
                k: v
                for k, v in result.items()
                if k
                in {
                    "action",
                    "snapshot_id",
                    "dispatch_status",
                    "error",
                    "recovery",
                    "delivery_mode",
                }
            } | {"details_omitted": True}
        _, parent, key = max(choices, key=lambda row: row[0])
        parent[key].pop()
        if key == "elements":
            parent.update(shown_count=len(parent[key]), truncated=True)
            parent["next_offset"] = parent.get("offset", 0) + len(parent[key])
        else:
            parent[key + "_truncated"] = True
    return result
