"""Normalize accessible controls without website-specific boosts or guessed insets."""

from .contracts import Element, Rect, normalized
from .runtime import ComputerError


def normalize_elements(capture, snapshot_id):
    result = []
    seen = set()
    for row in capture.elements:
        if not isinstance(row, dict) or row.get("visible") is False or row.get("showing") is False:
            continue
        if (
            row.get("window_id") is not None
            and capture.window
            and str(row["window_id"]) != str(capture.window.window_id)
        ):
            continue
        token = str(row.get("element_token") or "")
        identity = token or str(row.get("element_index", len(result)))
        if identity in seen:
            continue
        seen.add(identity)
        role = str(row.get("role") or "")
        label = str(row.get("label") or row.get("name") or "")
        # Unknown is distinct from false. SELECTED never substitutes for CHECKED.
        raw_states = row.get("states")
        raw_states = raw_states if isinstance(raw_states, (list, tuple)) else []
        states = {normalized(s).replace("_", " ") for s in raw_states if isinstance(s, str)}
        checked = row.get("checked")
        if not isinstance(checked, bool):
            checked = True if "checked" in states else False if "checkable" in states else None
        bounds = Rect.parse(row.get("frame") or row.get("bounds"))
        if capture.accessibility_frame is None:
            bounds = None
        elif (
            bounds
            and capture.accessibility_frame == "desktop"
            and capture.coordinate_frame == "window"
        ):
            wb = capture.window.bounds if capture.window else None
            bounds = (
                Rect(bounds.x - wb.x, bounds.y - wb.y, bounds.width, bounds.height) if wb else None
            )
        elif capture.accessibility_frame != capture.coordinate_frame:
            bounds = None
        value = row.get("value")
        # Password contents are never persisted or returned in accessible observations.
        if row.get("protected") is True or normalized(role) in {
            "password",
            "password text",
            "password field",
        }:
            value = None
        result.append(
            Element(
                id=f"{snapshot_id}:{len(result) + 1}",
                token=token,
                role=role,
                label=label,
                bounds=bounds,
                enabled=row.get("enabled") is not False,
                value=str(value) if value is not None else None,
                checked=checked,
                selected=row.get("selected") if isinstance(row.get("selected"), bool) else None,
                focused=row.get("focused") if isinstance(row.get("focused"), bool) else None,
                actions=[str(a) for a in row.get("actions", [])]
                if isinstance(row.get("actions"), list)
                else [],
            )
        )
    return result


def matches(elements, selector):
    if isinstance(selector, str):
        label, role = selector, None
    else:
        label, role = selector["label"], selector.get("role")
    return [
        e
        for e in elements
        if normalized(e.label) == normalized(label)
        and (role is None or normalized(e.role) == normalized(role))
    ]


def resolve(elements, *, element=None, target=None):
    if target is not None and element is not None:
        raise ComputerError("invalid_arguments", "Choose either element or target, not both.")
    if element is not None:
        if isinstance(element, int) and not isinstance(element, bool):
            rows = [e for e in elements if e.id.rsplit(":", 1)[-1] == str(element)]
        else:
            rows = [e for e in elements if e.id == element]
    else:
        rows = matches(elements, target)
    if not rows:
        raise ComputerError(
            "target_missing",
            "Target is absent from this snapshot. Capture again or choose a visible point.",
        )
    if len(rows) != 1:
        raise ComputerError(
            "target_ambiguous",
            f"Target matches {len(rows)} controls; use a specific element ID or label+role.",
        )
    if not rows[0].enabled:
        raise ComputerError("target_disabled", "The target is disabled in the current snapshot.")
    return rows[0]


def evaluate(condition, elements, window, *, complete=True):
    kind = condition["kind"]
    if kind == "window_title":
        return {
            "status": "verified" if window and window.title == condition["value"] else "not_met",
            "kind": kind,
        }
    rows = matches(elements, condition["target"])
    if kind == "exists":
        status = "verified" if rows else "not_met" if complete else "unavailable"
    elif kind == "absent":
        status = "not_met" if rows else "verified" if complete else "unavailable"
    elif len(rows) != 1:
        status = "ambiguous" if len(rows) > 1 else "unavailable"
    else:
        value = getattr(rows[0], kind)
        status = (
            "unavailable"
            if value is None
            else "verified"
            if value == condition["value"]
            else "not_met"
        )
    return {"status": status, "kind": kind}
