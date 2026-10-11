from __future__ import annotations

import mimetypes
from pathlib import Path

MAX_IMAGE_BYTES = 20 * 1024 * 1024
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def read_image_reference(path_value: str, mime_hint: str | None = None) -> tuple[bytes, str]:
    """Load a persisted Daena screenshot reference with strict path/size checks."""
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("Image reference is empty.")
    path = Path(path_value).expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("Image reference must point to a supported image file.")
    parts = path.parts
    if not any(
        parts[index] == ".daena" and index + 1 < len(parts) and parts[index + 1] == "screenshots"
        for index in range(len(parts))
    ):
        raise ValueError("Image reference is outside Daena's .daena/screenshots storage.")
    size = path.stat().st_size
    if size <= 0 or size > MAX_IMAGE_BYTES:
        raise ValueError(f"Image size is invalid or exceeds {MAX_IMAGE_BYTES} bytes.")
    data = path.read_bytes()
    mime = str(mime_hint or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    if mime not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError(f"Unsupported image MIME type: {mime}")
    return data, mime


def image_data_url(path_value: str, mime_hint: str | None = None) -> str:
    import base64

    data, mime = read_image_reference(path_value, mime_hint)
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
