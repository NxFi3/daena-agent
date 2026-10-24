"""Transport-neutral models and strict tool schemas. No task/website classifiers."""

from dataclasses import asdict, dataclass, field
import math
import unicodedata

from .runtime import ComputerError


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def number(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ComputerError("invalid_arguments", f"{name} must be a finite number.")
    return float(value)


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict):
            return None
        try:
            result = cls(
                *(
                    number(value.get(k, value.get(a)), k)
                    for k, a in (
                        ("x", "left"),
                        ("y", "top"),
                        ("width", "w"),
                        ("height", "h"),
                    )
                )
            )
            return result if result.width > 0 and result.height > 0 else None
        except ComputerError:
            return None

    def center(self):
        return [self.x + self.width / 2, self.y + self.height / 2]


@dataclass
class Window:
    pid: int
    window_id: int | str
    title: str = ""
    app: str = ""
    bounds: Rect | None = None
    active: bool = False

    def same(self, other):
        return (
            other is not None
            and self.pid == other.pid
            and str(self.window_id) == str(other.window_id)
        )

    def public(self):
        return asdict(self)


@dataclass
class Element:
    id: str
    token: str
    role: str
    label: str
    bounds: Rect | None = None
    enabled: bool = True
    value: str | None = None
    checked: bool | None = None
    selected: bool | None = None
    focused: bool | None = None
    actions: list[str] = field(default_factory=list)

    def public(self):
        row = asdict(self)
        row.pop("token")
        for key, limit in (("label", 240), ("value", 400), ("role", 80)):
            if isinstance(row.get(key), str) and len(row[key]) > limit:
                row[key] = row[key][:limit]
                row[key + "_truncated"] = True
        row["actions"] = [str(action)[:80] for action in row["actions"][:8]]
        return {key: value for key, value in row.items() if value is not None}


@dataclass
class Capture:
    image_path: str | None
    capture_id: str | None
    window: Window | None
    coordinate_frame: str
    elements: list[dict] = field(default_factory=list)
    accessibility_frame: str | None = None
    warnings: list[str] = field(default_factory=list)
    truncated: bool = False


POINT = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
SELECTOR = {
    "type": ["string", "object"],
    "description": "Exact visible label, or exact label+role to disambiguate. No fuzzy click selection.",
    "properties": {"label": {"type": "string"}, "role": {"type": "string"}},
    "required": ["label"],
    "additionalProperties": False,
}
EXPECT = {
    "type": "object",
    "description": "An explicit observable postcondition; never inferred from pixel changes.",
    "properties": {
        "kind": {
            "enum": [
                "exists",
                "absent",
                "checked",
                "selected",
                "focused",
                "value",
                "window_title",
            ]
        },
        "target": SELECTOR,
        "value": {"type": ["string", "boolean"]},
    },
    "required": ["kind"],
    "additionalProperties": False,
}
INPUT_ACTIONS = {
    "click",
    "double_click",
    "right_click",
    "middle_click",
    "type",
    "key",
    "scroll",
    "drag",
    "move",
}
PROPERTIES = {
    "action": {
        "description": "capture observes a window. inspect searches/pages controls in the same snapshot. navigate opens an exact URL in the selected browser using Ctrl+L, literal URL, Return; prefer this for browser navigation. type sends literal text to the focused field or explicit target. key sends a shortcut without a click. Input returns the next observation; use its snapshot_id.",
        "enum": [
            "capture",
            "inspect",
            "click",
            "double_click",
            "right_click",
            "middle_click",
            "type",
            "key",
            "scroll",
            "drag",
            "move",
            "wait",
            "sequence",
            "navigate",
            "launch",
            "list_windows",
            "list_apps",
            "doctor",
            "zoom",
        ],
    },
    "snapshot_id": {
        "type": "string",
        "description": "ID of the observation used to choose this input. Required for input and zoom.",
    },
    "element": {
        "type": ["string", "integer"],
        "description": "Opaque element ID from that snapshot; legacy numbered IDs require snapshot_id too.",
    },
    "target": SELECTOR,
    "coordinate": POINT,
    "from_coordinate": POINT,
    "to_coordinate": POINT,
    "text": {
        "type": "string",
        "maxLength": 100000,
        "description": "Literal text, exactly as it should be typed. For a URL use navigate with url; never wrap it in Markdown link syntax.",
    },
    "key": {"type": "string", "minLength": 1},
    "keys": {"type": ["array", "string"], "items": {"type": "string"}},
    "button": {"enum": ["left", "right", "middle"]},
    "direction": {"enum": ["up", "down", "left", "right"]},
    "amount": {"type": "integer", "minimum": 1, "maximum": 20},
    "duration_ms": {"type": "integer", "minimum": 0, "maximum": 10000},
    "mode": {"enum": ["som", "vision", "ax"]},
    "scope": {
        "enum": ["window", "desktop"],
        "description": "Default window. Explicit desktop enables whole-desktop coordinates.",
    },
    "app": {"type": "string"},
    "window_title": {"type": "string"},
    "pid": {"type": "integer", "minimum": 1},
    "window_id": {"type": ["integer", "string"]},
    "query": {"type": "string", "maxLength": 200},
    "max_results": {"type": "integer", "minimum": 1, "maximum": 100},
    "offset": {
        "type": "integer",
        "minimum": 0,
        "description": "Zero-based offset into filtered controls. Use inspect with query and next_offset without recapturing.",
    },
    "capture_after": {"type": "boolean"},
    "delivery_mode": {
        "enum": ["background", "foreground"],
        "description": "Explicit foreground uses the real focused desktop. This choice persists for the same session and window; omitted calls inherit it. A new target defaults to background. Background refusal never automatically escalates.",
    },
    "include_windows": {"type": "boolean"},
    "expect": EXPECT,
    "timeout": {"type": "number", "minimum": 0.1, "maximum": 120},
    "seconds": {"type": "number", "minimum": 0, "maximum": 10},
    "url": {"type": "string", "maxLength": 8192},
    "region": {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 4,
        "maxItems": 4,
        "description": "Zoom region [x,y,width,height] in the original screenshot; zoom never changes input coordinates.",
    },
    "scale": {"type": "number", "minimum": 1, "maximum": 4},
    "request_id": {
        "type": "string",
        "minLength": 1,
        "maxLength": 120,
        "description": "Optional idempotency key for reconnect/retry. The same request is never dispatched twice.",
    },
}
STEP_KEYS = {
    "action",
    "target",
    "text",
    "key",
    "keys",
    "direction",
    "amount",
    "seconds",
    "expect",
    "button",
}
STEP = {
    "type": "object",
    "properties": {key: value for key, value in PROPERTIES.items() if key in STEP_KEYS},
    "required": ["action"],
    "additionalProperties": False,
}
STEP["properties"]["action"] = {"enum": ["click", "type", "key", "scroll", "wait"]}
PROPERTIES["steps"] = {"type": "array", "items": STEP, "minItems": 1, "maxItems": 6}
SCHEMA = {
    "type": "object",
    "properties": PROPERTIES,
    "required": ["action"],
    "additionalProperties": False,
}


def validate_request(request, *, sequence_step=False):
    from jsonschema import Draft202012Validator

    errors = sorted(
        Draft202012Validator(SCHEMA).iter_errors(request), key=lambda e: str(e.path)
    )
    if errors:
        error = errors[0]
        raise ComputerError(
            "invalid_arguments",
            f"{'.'.join(map(str, error.path)) or 'request'}: {error.message}",
        )

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ComputerError(
                "invalid_arguments", "NaN and infinity are not valid tool arguments."
            )
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        if isinstance(value, list):
            for item in value:
                finite(item)

    finite(request)
    action = request["action"]
    if action == "launch" and not request.get("app", "").strip():
        raise ComputerError(
            "invalid_arguments",
            "launch requires app: an executable name or absolute path, for example app='firefox'. Use browser_use.open(url) for managed website browsing.",
        )
    if (
        action in INPUT_ACTIONS | {"sequence", "navigate", "zoom", "inspect"}
        and not sequence_step
        and not request.get("snapshot_id")
    ):
        raise ComputerError(
            "stale_snapshot", "snapshot_id from the latest observation is required."
        )
    if action in {
        "click",
        "double_click",
        "right_click",
        "middle_click",
        "move",
        "scroll",
    }:
        if (
            sum(request.get(k) is not None for k in ("target", "element", "coordinate"))
            != 1
        ):
            raise ComputerError(
                "invalid_arguments",
                "Exactly one target, element, or coordinate is required.",
            )
    if action == "type" and "text" not in request:
        raise ComputerError("invalid_arguments", "type requires text.")
    if action == "key":
        keys = request.get("keys")
        if isinstance(keys, str):
            keys = keys.split("+")
        if bool(request.get("key")) == bool(keys) or (
            keys and not all(k.strip() for k in keys)
        ):
            raise ComputerError(
                "invalid_arguments", "Provide either key or a nonempty keys chord."
            )
    if action == "drag" and (
        "from_coordinate" not in request or "to_coordinate" not in request
    ):
        raise ComputerError("invalid_arguments", "drag requires both endpoints.")
    if action == "scroll" and "direction" not in request:
        raise ComputerError("invalid_arguments", "scroll requires direction.")
    condition = request.get("expect")
    if condition:
        kind = condition["kind"]
        if kind != "window_title" and not condition.get("target"):
            raise ComputerError(
                "invalid_arguments",
                "expect.target is required for an element condition.",
            )
        if kind in {"checked", "selected", "focused"} and not isinstance(
            condition.get("value"), bool
        ):
            raise ComputerError(
                "invalid_arguments", f"expect.{kind} requires a boolean value."
            )
        if kind in {"value", "window_title"} and not isinstance(
            condition.get("value"), str
        ):
            raise ComputerError(
                "invalid_arguments", f"expect.{kind} requires a string value."
            )
    if request["action"] == "sequence":
        if not request.get("steps"):
            raise ComputerError("invalid_arguments", "sequence requires steps.")
        for step in request.get("steps", []):
            validate_request(step, sequence_step=True)
