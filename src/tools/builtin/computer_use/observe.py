"""Immutable raw screenshots, separately annotated previews and honest visual deltas."""

import hashlib
import json
from PIL import Image, ImageChops, ImageDraw, ImageFont

from .runtime import ComputerError

MAX_PIXELS = 32_000_000


def open_image(path):
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ComputerError(
                    "invalid_screenshot", "The backend must return a PNG screenshot."
                )
            if image.width * image.height > MAX_PIXELS:
                raise ComputerError(
                    "image_too_large", f"Screenshot exceeds {MAX_PIXELS} pixels."
                )
            image.load()
            return image.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ComputerError(
            "invalid_screenshot", f"Unreadable screenshot: {exc}"
        ) from exc


def image_info(path):
    image = open_image(path)
    digest = hashlib.sha256(
        f"{image.width}x{image.height}:".encode() + image.tobytes()
    ).hexdigest()
    return {"width": image.width, "height": image.height, "pixel_hash": digest}


def annotate(path, elements, output):
    image = open_image(path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=14)
    for e in elements:
        b = e.bounds
        if (
            b is None
            or b.x >= image.width
            or b.y >= image.height
            or b.x + b.width <= 0
            or b.y + b.height <= 0
        ):
            continue
        x, y = max(0, int(b.x)), max(0, int(b.y))
        right, bottom = (
            min(image.width - 1, int(b.x + b.width)),
            min(image.height - 1, int(b.y + b.height)),
        )
        draw.rectangle((x, y, right, bottom), outline="#9b8afb", width=2)
        tag = e.id.rsplit(":", 1)[-1]
        box = draw.textbbox((0, 0), tag, font=font)
        tw = box[2] - box[0] + 10
        x = min(x, max(0, image.width - tw))
        y = min(y, max(0, image.height - 23))
        draw.rounded_rectangle((x, y, x + tw, y + 22), radius=4, fill="#5b3ccc")
        draw.text((x + 5, y + 2), tag, fill="white", font=font)
    image.save(output, "PNG")
    return str(output)


def visual_delta(before, after):
    if not before or not after:
        return {"status": "unavailable"}
    old, new = open_image(before), open_image(after)
    if old.size != new.size:
        return {
            "status": "geometry_changed",
            "before_size": old.size,
            "after_size": new.size,
        }
    diff = ImageChops.difference(old, new)
    channels = diff.split()
    mask = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])
    bounds = mask.getbbox()
    hist = mask.histogram()
    return {
        "status": "changed" if bounds else "unchanged",
        "changed_fraction": round(1 - hist[0] / (old.width * old.height), 6),
        "changed_bounds": list(bounds) if bounds else None,
        "meaning": "Pixel change is evidence only; it does not prove task success.",
    }


def crop(path, region, scale, output):
    image = open_image(path)
    x, y, w, h = region
    if (
        x < 0
        or y < 0
        or w <= 0
        or h <= 0
        or x + w > image.width
        or y + h > image.height
    ):
        raise ComputerError(
            "invalid_region", "Zoom region must be inside the native screenshot."
        )
    box = (int(x), int(y), int(x + w), int(y + h))
    if box[2] == box[0] or box[3] == box[1]:
        raise ComputerError(
            "invalid_region", "Zoom region must contain at least one native pixel."
        )
    target_size = (round((box[2] - box[0]) * scale), round((box[3] - box[1]) * scale))
    if target_size[0] * target_size[1] > MAX_PIXELS:
        raise ComputerError(
            "image_too_large", "Zoom image would exceed the pixel limit."
        )
    image.crop(box).resize(target_size, Image.Resampling.LANCZOS).save(output, "PNG")
    return {
        "image_ref": str(output),
        "origin": list(box[:2]),
        "scale_x": target_size[0] / (box[2] - box[0]),
        "scale_y": target_size[1] / (box[3] - box[1]),
        "mapping": "native_x = origin[0] + zoom_x/scale_x; native_y = origin[1] + zoom_y/scale_y",
    }


def _element_state(element):
    """Internal identity/state, independent of per-capture public element IDs."""
    from dataclasses import asdict, is_dataclass

    row = asdict(element) if is_dataclass(element) else dict(element)
    bounds = row.get("bounds")
    if is_dataclass(bounds):
        bounds = asdict(bounds)
    identity = (
        ("token", str(row["token"]))
        if row.get("token")
        else (
            "geometry",
            row.get("role"),
            row.get("label"),
            json.dumps(bounds, sort_keys=True),
        )
    )
    return identity, row


def observation_signature(snapshot):
    """Compare observed pixels and AX state, never capture IDs, paths or timestamps.

    Used only to decide when observation polling has stopped changing. A stable
    signature does not establish application readiness or task completion.
    """
    window = snapshot.get("window") or {}
    elements = []
    for element in snapshot.get("elements", []):
        identity, row = _element_state(element)
        elements.append(
            {
                "identity": identity,
                **{
                    key: row.get(key)
                    for key in (
                        "role",
                        "label",
                        "bounds",
                        "enabled",
                        "value",
                        "checked",
                        "selected",
                        "focused",
                    )
                },
            }
        )
    elements.sort(key=lambda row: json.dumps(row["identity"], sort_keys=True))
    state = {
        "pixel_hash": snapshot.get("pixel_hash"),
        "size": [snapshot.get("width"), snapshot.get("height")],
        "window": {
            key: window.get(key)
            for key in ("pid", "window_id", "title", "bounds", "active")
        },
        "elements": elements,
        "accessibility_complete": snapshot.get("complete"),
    }
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def accessibility_delta(before_elements, after_elements, *, limit=12):
    """Report matched accessible state changes without inferring workflow success.

    Stable backend tokens are used internally; otherwise only uniquely matching
    role/label/geometry combinations are compared. Missing rows mean no longer
    observed, not proof that a control was deleted from the application.
    """

    def index(elements):
        result = {}
        for element in elements:
            key, row = _element_state(element)
            result.setdefault(key, []).append(row)
        return result

    before, after = index(before_elements), index(after_elements)
    changes = []
    count = compared = ambiguous = 0
    for identity in sorted(before.keys() & after.keys(), key=repr):
        old, new = before[identity], after[identity]
        if len(old) != 1 or len(new) != 1:
            ambiguous += 1
            continue
        old, new = old[0], new[0]
        compared += 1
        fields = {}
        for field in ("enabled", "value", "checked", "selected", "focused"):
            previous, current = old.get(field), new.get(field)
            if previous != current:

                def bounded(value):
                    return value[:400] if isinstance(value, str) else value

                fields[field] = {"before": bounded(previous), "after": bounded(current)}
                if any(
                    isinstance(value, str) and len(value) > 400
                    for value in (previous, current)
                ):
                    fields[field]["truncated"] = True
        if fields:
            count += 1
            if len(changes) < limit:
                changes.append(
                    {
                        "element": new.get("id"),
                        "label": str(new.get("label") or "")[:240],
                        "role": str(new.get("role") or "")[:80],
                        "fields": fields,
                    }
                )
    added = sum(len(rows) for key, rows in after.items() if key not in before)
    missing = sum(len(rows) for key, rows in before.items() if key not in after)
    return {
        "status": "changed"
        if count or added or missing
        else "unchanged"
        if compared
        else "unavailable",
        "compared_controls": compared,
        "changed_controls": count,
        "newly_observed": added,
        "no_longer_observed": missing,
        "ambiguous_identities": ambiguous,
        "changes": changes,
        "truncated": count > len(changes),
        "meaning": "Accessible state evidence only. No longer observed does not prove absence; state or pixel changes do not prove task completion.",
    }
