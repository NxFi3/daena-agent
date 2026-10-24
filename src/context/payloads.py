"""Reduce structured receipts without corrupting JSON or inventing observations."""

from copy import deepcopy
import json


def fit_json(value, limit):
    value = deepcopy(value)

    def dump(obj):
        return json.dumps(obj, ensure_ascii=False, default=str)

    from src.context.file_reads import file_evidence, trim_file_text

    evidence = file_evidence(value)
    if evidence is not None:
        if len(dump(value)) <= limit:
            return dump(value)
        body = evidence["content"]

        def trim(keep):
            evidence["content"] = body
            trim_file_text(evidence, keep)
            if "summary" in value and value is not evidence:
                value["summary"] = (
                    "Source text reduced for context; the evidence offsets and next_read describe the exact delivered prefix."
                )

        trim(0)
        if len(dump(value)) <= limit:
            low, high, best = 0, len(body), 0
            while low <= high:
                keep = (low + high) // 2
                trim(keep)
                if len(dump(value)) <= limit:
                    best, low = keep, keep + 1
                else:
                    high = keep - 1
            trim(best)
            return dump(value)
        # Metadata cannot fit either. Do not leave a claimed source span whose
        # corresponding text has been removed by the generic reducer.
        minimal = dump(
            {
                "details_omitted": True,
                "read_id": evidence["read_id"],
                "path": evidence["path"],
                "file_text_omitted": True,
            }
        )
        return minimal if len(minimal) <= limit else '{"details_omitted":true}'

    while len(dump(value)) > limit:
        choices = []

        def walk(item):
            if isinstance(item, dict):
                for key, child in item.items():
                    if isinstance(child, list) and child:
                        choices.append((len(dump(child)), item, key, "list"))
                    elif isinstance(child, str) and len(child) > 80:
                        choices.append((len(child), item, key, "text"))
                    elif isinstance(child, dict):
                        walk(child)
            elif isinstance(item, list) and item:
                choices.append((len(dump(item)), None, item, "root_list"))

        walk(value)
        if not choices:
            # At extreme budgets explicitly replace details, never emit broken JSON.
            minimal = {"details_omitted": True}
            if isinstance(value, dict):
                for key in (
                    "success",
                    "summary",
                    "action",
                    "snapshot_id",
                    "dispatch_status",
                ):
                    if key in value:
                        minimal[key] = value[key]
            if len(dump(minimal)) <= limit:
                return dump(minimal)
            return '{"details_omitted":true}'
        _, parent, key, kind = max(choices, key=lambda row: row[0])
        if kind == "root_list":
            key.pop()
        elif kind == "list":
            parent[key].pop()
            if key == "elements":
                parent.update(shown_count=len(parent[key]), truncated=True)
                parent["next_offset"] = parent.get("offset", 0) + len(parent[key])
            else:
                parent[key + "_truncated"] = True
        else:
            text = parent[key]
            keep = max(40, len(text) // 2)
            parent[key] = text[: keep // 2] + " …[omitted]… " + text[-keep // 2 :]
    return dump(value)


def fit_tool_text(text, limit, fallback):
    if len(text) <= limit:
        return text
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return fallback(text, limit)
    return fit_json(value, limit)
