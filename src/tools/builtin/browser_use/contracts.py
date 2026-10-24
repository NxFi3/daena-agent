"""Small, provider-portable browser contract; no site-specific selectors."""

from urllib.parse import urlsplit


class BrowserError(Exception):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code, self.details = code, details


ACTIONS = [
    "open",
    "tabs",
    "select_tab",
    "close_tab",
    "observe",
    "read",
    "screenshot",
    "click",
    "fill",
    "press",
    "select",
    "check",
    "hover",
    "scroll",
    "drag",
    "upload",
    "downloads",
    "save_download",
    "dialog",
    "wait",
    "back",
    "forward",
    "reload",
    "close",
    "doctor",
]
TARGET = {
    "type": "object",
    "properties": {
        "ref": {
            "type": "string",
            "description": "Opaque element reference from the latest observation.",
        },
        "role": {"type": "string"},
        "name": {"type": "string"},
        "label": {"type": "string"},
        "placeholder": {"type": "string"},
        "text": {"type": "string"},
        "css": {"type": "string"},
        "frame_id": {
            "type": "string",
            "description": "Frame from observe; main frame by default.",
        },
    },
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ACTIONS},
        "url": {
            "type": "string",
            "description": "Literal http(s) URL for open; starts the browser if needed.",
        },
        "tab_id": {"type": "string"},
        "new_tab": {"type": "boolean", "default": False},
        "target": TARGET,
        "destination": TARGET,
        "button": {
            "type": "string",
            "enum": ["left", "right", "middle"],
            "default": "left",
        },
        "click_count": {"type": "integer", "minimum": 1, "maximum": 2, "default": 1},
        "position": {
            "type": "object",
            "description": "Optional click point in CSS pixels relative to the target element's padding box; useful for canvas. Use screenshot/read to ground the target.",
            "properties": {
                "x": {"type": "number", "minimum": 0},
                "y": {"type": "number", "minimum": 0},
            },
            "required": ["x", "y"],
            "additionalProperties": False,
        },
        "text": {
            "type": "string",
            "description": "Literal fill text, press key (e.g. Enter or ControlOrMeta+A), dialog response, or wait text.",
        },
        "values": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Option values for select.",
        },
        "checked": {"type": "boolean"},
        "paths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Workspace files for upload.",
        },
        "path": {
            "type": "string",
            "description": "New workspace path for save_download.",
        },
        "download_id": {"type": "string"},
        "accept": {
            "type": "boolean",
            "description": "Explicit choice for a pending JavaScript dialog.",
        },
        "state": {
            "type": "string",
            "enum": [
                "visible",
                "hidden",
                "attached",
                "detached",
                "domcontentloaded",
                "load",
            ],
            "description": "Observable wait condition; target optional for page load.",
        },
        "delta_x": {"type": "integer", "default": 0},
        "delta_y": {"type": "integer", "default": 600},
        "query": {
            "type": "string",
            "description": "Filter observed controls by label/role; no fuzzy action selection.",
        },
        "offset": {"type": "integer", "minimum": 0, "default": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 40},
        "timeout_ms": {
            "type": "integer",
            "minimum": 100,
            "maximum": 30000,
            "default": 10000,
        },
    },
    "required": ["action"],
    "additionalProperties": False,
}


def validate(request):
    from jsonschema import Draft7Validator

    errors = list(Draft7Validator(SCHEMA).iter_errors(request))
    if errors:
        raise BrowserError("invalid_arguments", errors[0].message)
    action = request["action"]
    required = {
        "open": ("url",),
        "select_tab": ("tab_id",),
        "close_tab": ("tab_id",),
        "click": ("target",),
        "fill": ("target", "text"),
        "press": ("text",),
        "select": ("target", "values"),
        "check": ("target", "checked"),
        "hover": ("target",),
        "drag": ("target", "destination"),
        "upload": ("target", "paths"),
        "save_download": ("download_id", "path"),
        "dialog": ("accept",),
    }
    for key in required.get(action, ()):
        if key not in request:
            raise BrowserError("invalid_arguments", f"{action} requires {key}.")
    if action == "open":
        if request.get("new_tab") and request.get("tab_id"):
            raise BrowserError(
                "invalid_arguments", "Choose new_tab or an existing tab_id, not both."
            )
        url = request["url"]
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or any(ord(c) < 32 for c in url)
        ):
            raise BrowserError(
                "invalid_arguments",
                "open requires a literal http(s) URL without control characters.",
            )
    if action == "wait" and not any(k in request for k in ("target", "state", "text")):
        raise BrowserError(
            "invalid_arguments", "wait requires a target, text, or page load state."
        )
    for key in ("target", "destination"):
        if key not in request:
            continue
        target = request[key]
        kinds = [
            k
            for k in ("ref", "role", "label", "placeholder", "text", "css")
            if k in target
        ]
        if len(kinds) != 1 or not target[kinds[0]].strip():
            raise BrowserError(
                "invalid_arguments",
                f"{key} requires exactly one ref, role, label, placeholder, text or css.",
            )
        if "name" in target and "role" not in target:
            raise BrowserError("invalid_arguments", "name requires role.")
    if action == "upload" and not request["paths"]:
        raise BrowserError("invalid_arguments", "upload requires at least one file.")


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("browser must be an object")
    if type(config.get("headless", False)) is not bool:
        raise ValueError("browser.headless must be a boolean")
    for key in ("executable_path", "cdp_url"):
        if key in config and not isinstance(config[key], str):
            raise ValueError(f"browser.{key} must be a string")
    if config.get("cdp_url"):
        url = urlsplit(config["cdp_url"])
        if url.scheme not in {"http", "https", "ws", "wss"} or not url.hostname:
            raise ValueError("browser.cdp_url must be an HTTP or WebSocket endpoint")
    return dict(config)
