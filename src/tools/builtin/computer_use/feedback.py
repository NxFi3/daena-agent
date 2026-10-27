"""Bounded action receipts: delivery and observed changes are separate from goals."""


def input_receipt(request, content):
    """Retain targeting context without duplicating typed secrets in observations."""
    target = {
        key: request[key]
        for key in (
            "coordinate",
            "position",
            "element",
            "target",
            "from_coordinate",
            "to_coordinate",
            "button",
            "key",
            "keys",
            "direction",
            "amount",
        )
        if key in request
    }
    if "text" in request:
        target["typed_characters"] = len(request["text"])
    if "path" in request:
        target["stroke"] = {
            "start": request["path"][0],
            "end": request["path"][-1],
            "point_count": len(request["path"]),
        }
    observation = content.get("observation") or {}
    return {
        "action": request["action"],
        "target": target,
        "snapshot_id_used": request.get("snapshot_id"),
        "next_snapshot_id": observation.get("snapshot_id"),
        "dispatch_status": content.get("dispatch_status"),
        "delivery_mode": content.get("delivery_mode"),
        "verification": content.get("verification"),
        "visual_delta": content.get("visual_delta"),
        "accessibility_delta": content.get("accessibility_delta"),
        "settling": observation.get("settling"),
        "error": content.get("error"),
        "meaning": "Input receipt, not proof of task completion. Review focus/value and current image before deciding another input. Repeating the same click is not a substitute for typing or a key.",
    }


def feedback_summary(content):
    parts = []
    pixels = content.get("visual_delta", {}).get("status")
    if pixels:
        parts.append("pixels=" + pixels)
    ax = content.get("accessibility_delta", {})
    changed = ax.get("changed_controls", 0)
    if changed:
        fields = sorted(
            {
                key
                for change in ax.get("changes", [])
                for key in change.get("fields", {})
            }
        )
        parts.append("controls changed: " + ", ".join(fields))
    settling = (content.get("observation") or {}).get("settling", {})
    if settling.get("status") == "deadline":
        parts.append(
            "UI still changing at observation deadline; wait/expect can observe without another click"
        )
    if pixels == "unchanged" and not changed:
        parts.append(
            "No observed change; inspect focus/value or wait before deciding to repeat input"
        )
    return "; ".join(parts)
