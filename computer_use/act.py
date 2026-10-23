"""Translate one validated request. Dispatch once; never infer an application's workflow."""

from .contracts import INPUT_ACTIONS
from .inspect import resolve
from .runtime import ComputerError, checkpoint


def point(value, width, height):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ComputerError("invalid_arguments", "A point must be [x,y] in the native screenshot.")
    x, y = value
    if not (0 <= x < width and 0 <= y < height):
        raise ComputerError(
            "outside_capture", f"Point {value} is outside the {width}×{height} screenshot."
        )
    return x, y


def build_action(request, snapshot, elements):
    action = request["action"]
    if action not in INPUT_ACTIONS:
        raise ComputerError("invalid_action", f"Not an input action: {action}")
    args = {"delivery_mode": request.get("delivery_mode", "background")}
    window = snapshot.get("window")
    if window:
        args.update(pid=window["pid"], window_id=window["window_id"])
    else:
        args["target"] = {"kind": "desktop", "display_id": "primary"}
    target = None
    if request.get("element") is not None or request.get("target") is not None:
        target = resolve(elements, element=request.get("element"), target=request.get("target"))
    if target and request.get("coordinate") is not None:
        raise ComputerError(
            "invalid_arguments", "Use one target mechanism: element, label, or coordinate."
        )
    coordinate = request.get("coordinate")
    if target:
        if target.token and action in {"click", "double_click", "right_click", "type"}:
            args["element_token"] = target.token
        elif target.bounds:
            coordinate = target.bounds.center()
        else:
            raise ComputerError(
                "target_not_actionable",
                "This control has neither an action token nor verified bounds.",
            )
    if coordinate is not None:
        args["x"], args["y"] = point(
            coordinate, snapshot.get("width", 0), snapshot.get("height", 0)
        )
        args["coordinate_frame"] = snapshot["coordinate_frame"]
    if action in {"click", "double_click", "right_click", "middle_click", "move"}:
        if "element_token" not in args and "x" not in args:
            raise ComputerError(
                "target_required", "Choose a current element, exact label, or screenshot point."
            )
        if "x" in args:
            args["capture_id"] = snapshot["capture_id"]
        if action == "middle_click":
            action, args["button"] = "click", "middle"
        elif action == "click":
            args["button"] = request.get("button", "left")
    elif action == "type":
        if "text" not in request:
            raise ComputerError("invalid_arguments", "type requires text.")
        action, args["text"] = "type_text", request["text"]
    elif action == "key":
        keys = request.get("keys")
        if isinstance(keys, str):
            keys = keys.split("+")
        if request.get("key") and keys:
            raise ComputerError("invalid_arguments", "Specify key or keys, not both.")
        if keys and len(keys) > 1 and all(k.strip() for k in keys):
            action, args["keys"] = "hotkey", keys
        elif request.get("key") or (keys and len(keys) == 1 and keys[0]):
            action, args["key"] = "press_key", request.get("key") or keys[0]
        else:
            raise ComputerError("invalid_arguments", "key requires a key name or nonempty chord.")
    elif action == "scroll":
        if "x" not in args:
            raise ComputerError(
                "target_required",
                "scroll needs an explicit point or target inside the scroll area.",
            )
        if "direction" not in request:
            raise ComputerError("invalid_arguments", "scroll requires direction.")
        args.update(direction=request["direction"], amount=request.get("amount", 3), by="line")
    elif action == "drag":
        if target or coordinate:
            raise ComputerError(
                "invalid_arguments", "drag requires from_coordinate and to_coordinate."
            )
        args["from_x"], args["from_y"] = point(
            request.get("from_coordinate"), snapshot.get("width", 0), snapshot.get("height", 0)
        )
        args["to_x"], args["to_y"] = point(
            request.get("to_coordinate"), snapshot.get("width", 0), snapshot.get("height", 0)
        )
        args.update(
            coordinate_frame=snapshot["coordinate_frame"],
            duration_ms=request.get("duration_ms", 500),
            button=request.get("button", "left"),
        )
    return action, args


def dispatch(backend, session, action, args):
    # Native element typing may be replacement, not append. Never chunk it.
    if action != "type_text" or "element_token" in args or len(args["text"]) <= 400:
        checkpoint()
        return backend.dispatch(session, action, args)
    text = args["text"]
    sent = 0
    for index in range(0, len(text), 400):
        checkpoint()
        part = dict(args, text=text[index : index + 400])
        if index:
            for key in ("x", "y", "coordinate_frame"):
                part.pop(key, None)
        try:
            backend.dispatch(session, action, part)
        except ComputerError as exc:
            raise ComputerError(
                exc.code,
                f"Typing is partial: {sent}/{len(text)} characters acknowledged; the failing chunk may also have reached the app. {exc}",
                dispatched=True,
            ) from exc
        sent += len(part["text"])
    return {"status": "dispatched", "characters_acknowledged": sent}
