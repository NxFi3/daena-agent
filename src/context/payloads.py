"""Reduce structured receipts without corrupting JSON or inventing observations."""

from copy import deepcopy
import json


def fit_json(value, limit):
    value = deepcopy(value)

    def dump(obj):
        return json.dumps(obj, ensure_ascii=False, default=str)

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
