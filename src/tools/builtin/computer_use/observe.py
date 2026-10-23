"""Immutable raw screenshots, separately annotated previews and honest visual deltas."""

import hashlib
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
