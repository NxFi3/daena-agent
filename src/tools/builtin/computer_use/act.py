from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool
from src.tools.builtin.computer_use.driver import (
    CuaDriverError, call_driver, pointer_state_path, session_label_from_id,
)


# Long text is typed in bounded chunks: one huge argv/IPC payload can hit
# EAGAIN (os error 11) or exceed the per-call timeout inside the driver.
_TYPE_CHUNK_CHARS = 400
_TYPE_TOTAL_BUDGET_SECONDS = 120.0


class ComputerActTool(Tool):
    name = "computer_act"
    action = "interact"
    description = (
        "Perform exactly one explicit mouse/keyboard action on the desktop through Cua Driver. "
        "Prefer accessibility actions: call computer_inspect on the target window, then pass the "
        "matching element_token here for click or text entry. This avoids guessing coordinates. "
        "Only use x/y from the latest computer_observe screenshot when inspection cannot identify "
        "the target. For pixel actions, use the exact desktop capture ID automatically bound by the "
        "runtime. Observe after actions and verify the requested state change; this tool blocks "
        "repeated coordinates and excessive pixel-click loops. Never treat a dispatched click as proof "
        "that the desired UI state changed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["click", "double_click", "right_click", "drag", "type_text", "press_key", "hotkey", "scroll"],
                "description": "Exactly one action to perform.",
            },
            "x": {"type": "integer", "description": "Desktop screenshot X coordinate for click, or focus-target for typing/keys."},
            "y": {"type": "integer", "description": "Desktop screenshot Y coordinate for pixel actions; use only from the latest screenshot."},
            "element_token": {"type": "string", "description": "Preferred exact UI element handle returned by computer_inspect. For click or type_text, use this instead of x/y."},
            "from_x": {"type": "number", "description": "Start X for drag, from the latest desktop screenshot when coordinate_frame='desktop'."},
            "from_y": {"type": "number", "description": "Start Y for drag."},
            "to_x": {"type": "number", "description": "End X for drag."},
            "to_y": {"type": "number", "description": "End Y for drag."},
            "duration_ms": {"type": "integer", "minimum": 0, "maximum": 10000, "default": 500, "description": "Duration of drag gesture."},
            "text": {"type": "string", "description": "Text for type_text; may be combined with element_token from computer_inspect."},
            "key": {"type": "string", "description": "Single key for press_key, e.g. ENTER or ESC."},
            "keys": {
                "type": "array", "items": {"type": "string"},
                "description": "Hotkey chord such as ['ctrl', 'l'].",
            },
            "button": {"type": "string", "enum": ["left", "right", "middle"], "default": "left"},
            "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
            "amount": {"type": "integer", "minimum": 1, "maximum": 20, "default": 3},
            "delivery_mode": {
                "type": "string", "enum": ["background", "foreground"], "default": "background",
                "description": "Background is the default, following Hermes' no-focus-steal contract. Use foreground only after a structured driver refusal recommends escalation and it is appropriate to bring the app forward.",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.session_id_provider = None
        self.workspace_root = Path.cwd().resolve()

    def set_workspace(self, directory: str | Path) -> None:
        self.workspace_root = Path(directory).expanduser().resolve()

    def set_context_store(self, _stm, session_id_provider) -> None:
        self.session_id_provider = session_id_provider

    def _session_label(self) -> str:
        raw = "default"
        if callable(self.session_id_provider):
            try:
                raw = str(self.session_id_provider() or "default")
            except Exception:
                raw = "default"
        return session_label_from_id(raw)

    def _prime_drag_screenshot_context(
        self, session: str, pointer_state: dict[str, Any],
    ) -> str:
        """Refresh CUA's active capture immediately before a pixel drag.

        Computer Use reads the accessibility tree after obtaining the screenshot.
        On Wayland, that tree read invalidates the driver's active screenshot
        context, so a later drag is refused as screenshot_context_missing. Refresh
        only the pixels (no AX tree) in the same CUA session, and refuse to dispatch
        if the native target or coordinate geometry changed in the meantime.
        """
        target = pointer_state.get("latest_screenshot_target")
        target = target if isinstance(target, dict) else {}
        frame = str(
            target.get("coordinate_frame")
            or pointer_state.get("latest_screenshot_coordinate_frame")
            or "desktop"
        ).strip().casefold()
        if frame not in {"window", "desktop"}:
            raise ValueError(
                "Blocked drag: the latest screenshot has no supported coordinate frame. "
                "Capture the target again before drawing."
            )

        desktop_capture = target.get("desktop_capture") is True
        use_window_capture = frame == "window" and not desktop_capture
        capture_args: dict[str, Any] = {"session": session}
        if use_window_capture:
            try:
                pid = int(target.get("pid"))
                window_id = int(target.get("window_id"))
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    "Blocked drag: the latest window screenshot has no verified PID/window ID. "
                    "Capture the intended app again."
                )
            if pid <= 0 or window_id <= 0:
                raise ValueError(
                    "Blocked drag: the latest window screenshot has no verified PID/window ID. "
                    "Capture the intended app again."
                )
            action = "get_window_state"
            capture_args.update({
                "pid": pid,
                "window_id": window_id,
                "include_accessibility_tree": False,
                "include_screenshot": True,
            })
        else:
            action = "get_desktop_state"

        session_key = hashlib.sha256(str(session).encode("utf-8")).hexdigest()[:16]
        screenshot_path = (
            self.workspace_root / ".daena" / "computer_use"
            / f"drag_context_{session_key}.png"
        )
        try:
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"Cannot prepare a safe drag-context screenshot: {exc}") from exc
        capture_args["screenshot_out_file"] = str(screenshot_path)

        raw = call_driver(action, capture_args, timeout=30)
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise CuaDriverError(
                f"Cua Driver returned invalid drag-context screenshot metadata: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise CuaDriverError("Cua Driver returned no drag-context screenshot metadata.")
        if payload.get("code") or payload.get("isError"):
            raise CuaDriverError(
                json.dumps(payload, ensure_ascii=False)[:1500]
            )
        if payload.get("screenshot_frame_valid") is False:
            detail = payload.get("screenshot_error") or payload.get("frame_note") or (
                "the driver could not verify the screenshot frame"
            )
            raise CuaDriverError(f"Drag-context screenshot is not verified: {detail}")
        capture_id = payload.get("capture_id")
        if not isinstance(capture_id, str) or not capture_id.strip():
            raise CuaDriverError("Cua Driver did not provide a capture ID for the drag context.")
        if not screenshot_path.is_file() or screenshot_path.stat().st_size <= 24:
            raise CuaDriverError("Cua Driver did not write a readable drag-context screenshot.")

        try:
            width = int(payload.get("screenshot_width") or payload.get("screen_width") or 0)
            height = int(payload.get("screenshot_height") or payload.get("screen_height") or 0)
        except (TypeError, ValueError, OverflowError):
            width = height = 0
        if width <= 0 or height <= 0:
            raise CuaDriverError("The drag-context screenshot has no verified dimensions.")

        try:
            old_width = int(pointer_state.get("latest_screenshot_width"))
            old_height = int(pointer_state.get("latest_screenshot_height"))
        except (TypeError, ValueError, OverflowError):
            old_width = old_height = 0
        if (old_width, old_height) != (width, height):
            raise ValueError(
                f"Blocked drag: screenshot size changed from {old_width}x{old_height} "
                f"to {width}x{height}. Capture the resized screen and recompute the drag coordinates."
            )

        if use_window_capture:
            try:
                captured_pid = int(payload.get("pid"))
                captured_window_id = int(payload.get("window_id"))
            except (TypeError, ValueError, OverflowError):
                captured_pid = captured_window_id = 0
            if (captured_pid, captured_window_id) != (pid, window_id):
                raise ValueError(
                    "Blocked drag: the refreshed screenshot belongs to a different native window. "
                    "Capture the intended window again before dragging."
                )
            refreshed_bounds = payload.get("window_bounds")
            expected_bounds = target.get("bounds")
            if not isinstance(expected_bounds, dict):
                expected_bounds = target
            if isinstance(refreshed_bounds, dict) and isinstance(expected_bounds, dict):
                for key in ("width", "height"):
                    try:
                        before = float(expected_bounds[key])
                        after = float(refreshed_bounds[key])
                    except (KeyError, TypeError, ValueError, OverflowError):
                        continue
                    if abs(before - after) > 1:
                        raise ValueError(
                            "Blocked drag: the target window was resized after the screenshot. "
                            "Capture it again and recompute the coordinates."
                        )
        elif not desktop_capture:
            # The screenshot frame is desktop-wide but the action is bound to a
            # particular native window. Verify that window did not move/resize since
            # the model saw its screenshot, otherwise desktop pixels are stale.
            try:
                expected_pid = int(target.get("pid"))
                expected_window_id = int(target.get("window_id"))
            except (TypeError, ValueError, OverflowError):
                expected_pid = expected_window_id = 0
            windows = payload.get("windows")
            windows = windows if isinstance(windows, list) else []
            current = next((
                row for row in windows if isinstance(row, dict)
                and int(row.get("pid") or 0) == expected_pid
                and int(row.get("window_id") or 0) == expected_window_id
            ), None)
            if current is None:
                raise ValueError(
                    "Blocked drag: the selected native window was not found in the refreshed desktop. "
                    "Capture the intended app again."
                )
            expected_bounds = target.get("bounds")
            if not isinstance(expected_bounds, dict):
                expected_bounds = target
            current_bounds = current.get("bounds")
            if not isinstance(current_bounds, dict):
                current_bounds = current
            for key in ("x", "y", "width", "height"):
                try:
                    before = float(expected_bounds[key])
                    after = float(current_bounds[key])
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue
                if abs(before - after) > 1:
                    raise ValueError(
                        "Blocked drag: the target window moved or resized after the screenshot. "
                        "Capture it again and recompute the coordinates."
                    )

        # The new native read invalidates the prior accessibility snapshot. Prevent a
        # subsequent token action from reusing stale AX IDs when capture_after=False.
        pointer_state["latest_capture_id"] = capture_id
        pointer_state["latest_screenshot_width"] = width
        pointer_state["latest_screenshot_height"] = height
        pointer_state["latest_screenshot_captured_at"] = time.time()
        pointer_state["latest_screenshot_coordinate_frame"] = frame
        pointer_state["latest_element_ids"] = {}
        pointer_state["latest_accessibility_elements"] = {}
        pointer_state.pop("latest_accessibility_pid", None)
        pointer_state.pop("latest_accessibility_window_id", None)
        pointer_state.pop("latest_accessibility_window_title", None)
        pointer_state.pop("latest_accessibility_captured_at", None)
        return capture_id

    @staticmethod
    def _driver_args(
        action: str, arguments: dict[str, Any], session: str,
        pointer_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build input args against the exact screenshot/accessibility target.

        Hermes binds every normal window action to PID + window_id. Desktop-wide
        coordinates are retained only for an explicitly full-desktop capture.
        """
        state = pointer_state if isinstance(pointer_state, dict) else {}
        delivery_mode = str(arguments.get("delivery_mode") or "background").strip().lower()
        if delivery_mode not in {"background", "foreground"}:
            raise ValueError("delivery_mode must be 'background' or 'foreground'.")

        screenshot_target = state.get("latest_screenshot_target")
        screenshot_target = screenshot_target if isinstance(screenshot_target, dict) else {}
        access_pid = state.get("latest_accessibility_pid")
        access_window_id = state.get("latest_accessibility_window_id")

        def positive_int(value: Any) -> int | None:
            if value is None or isinstance(value, bool):
                return None
            try:
                number = int(value)
            except (TypeError, ValueError, OverflowError):
                return None
            return number if number > 0 else None

        screenshot_pid = positive_int(screenshot_target.get("pid"))
        screenshot_window_id = positive_int(screenshot_target.get("window_id"))
        frame = str(
            screenshot_target.get("coordinate_frame")
            or state.get("latest_screenshot_coordinate_frame")
            or "desktop"
        ).strip().casefold()
        screenshot_is_window = (
            screenshot_target.get("desktop_capture") is not True
            and screenshot_pid is not None
            and screenshot_window_id is not None
        )

        # Coordinates belong to the screenshot frame; keyboard-only actions use
        # the current accessibility target, then the last verified window target.
        has_point = any(key in arguments for key in ("x", "y", "from_x", "from_y", "to_x", "to_y"))
        target_pid = screenshot_pid if has_point and screenshot_is_window else positive_int(access_pid)
        target_window_id = screenshot_window_id if has_point and screenshot_is_window else positive_int(access_window_id)
        if target_pid is None or target_window_id is None:
            target_pid = screenshot_pid
            target_window_id = screenshot_window_id
        targeted_window = (
            screenshot_is_window if has_point
            else target_pid is not None and target_window_id is not None
        )

        common: dict[str, Any] = {"session": session, "delivery_mode": delivery_mode}
        if targeted_window and target_pid is not None and target_window_id is not None:
            common.update({"pid": target_pid, "window_id": target_window_id})
        else:
            common["target"] = {"kind": "desktop", "display_id": "primary"}

        def coordinate_args() -> dict[str, Any]:
            if targeted_window and target_pid is not None and target_window_id is not None:
                return {"coordinate_frame": frame if frame in {"window", "desktop"} else "desktop"}
            return {"coordinate_frame": "desktop"}

        element_token = str(arguments.get("element_token") or "").strip()
        if element_token and action in {"click", "double_click", "right_click"}:
            return {"session": session, "element_token": element_token, "delivery_mode": delivery_mode}
        if element_token and action == "type_text":
            text = arguments.get("text")
            if not isinstance(text, str):
                raise ValueError("type_text with element_token requires text.")
            return {"session": session, "element_token": element_token, "text": text, "delivery_mode": delivery_mode}

        if action in {"click", "double_click", "right_click"}:
            if not isinstance(arguments.get("x"), (int, float)) or isinstance(arguments.get("x"), bool) or not isinstance(arguments.get("y"), (int, float)) or isinstance(arguments.get("y"), bool):
                raise ValueError(f"{action} requires numeric x and y from the latest screenshot.")
            args = {**common, **coordinate_args(), "x": arguments["x"], "y": arguments["y"]}
            if action == "click":
                args["button"] = arguments.get("button", "left")
            return args

        if action == "drag":
            coordinates = ("from_x", "from_y", "to_x", "to_y")
            if not all(isinstance(arguments.get(key), (int, float)) and not isinstance(arguments.get(key), bool) for key in coordinates):
                raise ValueError("drag requires from_x/from_y and to_x/to_y from the latest screenshot.")
            return {
                **common, **coordinate_args(),
                "from_x": arguments["from_x"], "from_y": arguments["from_y"],
                "to_x": arguments["to_x"], "to_y": arguments["to_y"],
                "duration_ms": max(0, min(10000, int(arguments.get("duration_ms", 500)))),
                "button": arguments.get("button", "left"),
            }

        if action == "type_text":
            text = arguments.get("text")
            if not isinstance(text, str):
                raise ValueError("type_text requires text.")
            args = {**common, "text": text}
            if "x" in arguments or "y" in arguments:
                if not isinstance(arguments.get("x"), (int, float)) or isinstance(arguments.get("x"), bool) or not isinstance(arguments.get("y"), (int, float)) or isinstance(arguments.get("y"), bool):
                    raise ValueError("type_text requires both x and y when targeting a text field.")
                args.update(coordinate_args())
                args.update({"x": arguments["x"], "y": arguments["y"]})
            return args

        if action == "press_key":
            key = str(arguments.get("key") or "").strip()
            if not key:
                raise ValueError("press_key requires key.")
            args = {**common, "key": key}
            if "x" in arguments or "y" in arguments:
                if not isinstance(arguments.get("x"), (int, float)) or isinstance(arguments.get("x"), bool) or not isinstance(arguments.get("y"), (int, float)) or isinstance(arguments.get("y"), bool):
                    raise ValueError("press_key requires both x and y when either is supplied.")
                args.update(coordinate_args())
                args.update({"x": arguments["x"], "y": arguments["y"]})
            return args

        if action == "hotkey":
            keys = arguments.get("keys")
            if not isinstance(keys, list) or len(keys) < 2 or not all(isinstance(k, str) and k.strip() for k in keys):
                raise ValueError("hotkey requires a list of at least two key names.")
            args = {**common, "keys": [key.strip() for key in keys]}
            if "x" in arguments or "y" in arguments:
                if not isinstance(arguments.get("x"), (int, float)) or isinstance(arguments.get("x"), bool) or not isinstance(arguments.get("y"), (int, float)) or isinstance(arguments.get("y"), bool):
                    raise ValueError("hotkey requires both x and y when either is supplied.")
                args.update(coordinate_args())
                args.update({"x": arguments["x"], "y": arguments["y"]})
            return args

        if action == "scroll":
            direction = str(arguments.get("direction") or "").strip().lower()
            if direction not in {"up", "down", "left", "right"}:
                raise ValueError("scroll requires direction: up, down, left, or right.")
            if not isinstance(arguments.get("x"), (int, float)) or isinstance(arguments.get("x"), bool) or not isinstance(arguments.get("y"), (int, float)) or isinstance(arguments.get("y"), bool):
                raise ValueError("scroll requires x/y for a point inside the intended scroll area, from the latest capture.")
            return {
                **common, **coordinate_args(),
                "x": arguments["x"], "y": arguments["y"],
                "direction": direction,
                "amount": max(1, min(20, int(arguments.get("amount", 3)))),
                "by": "line",
            }

        raise ValueError(f"Unsupported computer action: {action}")

    @staticmethod
    def validate_pixel_coordinates(action: str, arguments: dict[str, Any], pointer_state: dict[str, Any]) -> None:
        """Fail closed before native desktop input if its screenshot frame is untrusted."""
        action = str(action or "").strip().casefold()
        if arguments.get("element_token") and action in {"click", "double_click", "right_click", "type_text"}:
            return
        coordinate_actions = {"click", "double_click", "right_click", "drag", "type_text", "press_key", "hotkey", "scroll"}
        if action not in coordinate_actions:
            return

        points: list[tuple[str, float, float]] = []
        if action == "drag":
            for prefix in ("from", "to"):
                x, y = arguments.get(f"{prefix}_x"), arguments.get(f"{prefix}_y")
                if x is None and y is None:
                    continue
                if (not isinstance(x, (int, float)) or isinstance(x, bool)
                        or not isinstance(y, (int, float)) or isinstance(y, bool)):
                    raise ValueError(f"Blocked unsafe drag: {prefix} point must contain numeric x/y from the latest screenshot.")
                points.append((prefix, float(x), float(y)))
        elif "x" in arguments or "y" in arguments:
            x, y = arguments.get("x"), arguments.get("y")
            if (not isinstance(x, (int, float)) or isinstance(x, bool)
                    or not isinstance(y, (int, float)) or isinstance(y, bool)):
                raise ValueError(f"Blocked unsafe {action}: coordinates must be numeric x/y from the latest screenshot.")
            points.append(("target", float(x), float(y)))
        if not points:
            return

        capture_id = pointer_state.get("latest_capture_id")
        if not isinstance(capture_id, str) or not capture_id.strip():
            raise ValueError("No current desktop capture is recorded. Capture the current screen before sending coordinate input.")
        try:
            width = int(pointer_state.get("latest_screenshot_width"))
            height = int(pointer_state.get("latest_screenshot_height"))
        except (TypeError, ValueError):
            width = height = 0
        if width <= 0 or height <= 0:
            raise ValueError("Blocked coordinate input: the latest screenshot has no verified native dimensions. Capture the current screen before acting.")

        captured_at = pointer_state.get("latest_screenshot_captured_at")
        if not isinstance(captured_at, (int, float)) or isinstance(captured_at, bool):
            raise ValueError("Blocked coordinate input: screenshot freshness is unknown. Capture the current screen before acting.")
        age = time.time() - float(captured_at)
        if age < -5 or age > 180:
            raise ValueError("Blocked coordinate input: the screenshot is older than 180 seconds or has an invalid timestamp. Capture the current screen before acting.")

        target = pointer_state.get("latest_screenshot_target")
        target = target if isinstance(target, dict) else {}
        if not target:
            raise ValueError("Blocked coordinate input: no verified screenshot target is recorded. Capture the current screen/window before acting.")
        bounds = target.get("bounds")
        if bounds is None and all(key in target for key in ("x", "y", "width", "height")):
            bounds = {key: target.get(key) for key in ("x", "y", "width", "height")}
        rect: tuple[float, float, float, float] | None = None
        coordinate_frame = str(
            target.get("coordinate_frame")
            or pointer_state.get("latest_screenshot_coordinate_frame")
            or "desktop"
        ).strip().casefold()
        # Window screenshots use local x/y (0..screenshot_width/height). Their
        # native screen-space bounds describe placement, not the coordinate frame.
        if target.get("desktop_capture") is not True and coordinate_frame != "window":
            if not isinstance(bounds, dict):
                raise ValueError("Blocked coordinate input: the selected screenshot window has no verified bounds. Capture the target window again.")
            try:
                rect = tuple(float(bounds[key]) for key in ("x", "y", "width", "height"))
            except (KeyError, TypeError, ValueError):
                raise ValueError("Blocked coordinate input: target window bounds are malformed. Capture the target window again.")
            if rect[2] <= 0 or rect[3] <= 0:
                raise ValueError("Blocked coordinate input: target window bounds are empty. Capture the target window again.")

        for label, x, y in points:
            if not (float("-inf") < x < float("inf") and float("-inf") < y < float("inf")):
                raise ValueError(f"Blocked unsafe {action}: {label} point contains a non-finite coordinate.")
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"Blocked unsafe {action}: {label} point ({x:g},{y:g}) is outside the latest native screenshot ({width}x{height}); no input was dispatched."
                )
            if rect is not None:
                left, top, win_width, win_height = rect
                if not (left <= x < left + win_width and top <= y < top + win_height):
                    raise ValueError(
                        f"Blocked unsafe {action}: {label} point ({x:g},{y:g}) is outside the selected window "
                        f"'{target.get('title') or target.get('app') or '(unknown)'}'; no input was dispatched."
                    )

    @staticmethod
    def _type_in_chunks(args: dict[str, Any]) -> str:
        """Type long text in bounded chunks; stop at the first failure, never retry."""
        text = str(args["text"])
        chunks = [text[i:i + _TYPE_CHUNK_CHARS] for i in range(0, len(text), _TYPE_CHUNK_CHARS)]
        deadline = time.monotonic() + _TYPE_TOTAL_BUDGET_SECONDS
        sent = 0
        last_output = ""
        for index, chunk in enumerate(chunks, start=1):
            if time.monotonic() > deadline:
                raise CuaDriverError(
                    f"Typing stopped after {sent} of {len(text)} characters: total time budget exceeded. "
                    "The text on screen is partial; do not resend the whole text."
                )
            chunk_args = {**args, "text": chunk}
            if index > 1:
                # Only the first chunk may carry a focus point; later chunks go to the focused target.
                for key in ("x", "y", "coordinate_frame"):
                    chunk_args.pop(key, None)
            try:
                last_output = call_driver("type_text", chunk_args, timeout=15)
            except CuaDriverError as exc:
                raise CuaDriverError(
                    f"Typing stopped after {sent} of {len(text)} characters (chunk {index}/{len(chunks)}). "
                    "The text on screen is partial and the failing chunk may or may not have been typed. "
                    f"Do not resend the whole text. Original error: {exc}"
                ) from exc
            sent += len(chunk)
            time.sleep(0.05)
        return last_output

    @staticmethod
    def _save_pointer_state(path: Path, state: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            pass

    def execute(self, action: str, **kwargs: Any) -> ToolResult:
        action = str(action or "").strip().lower()
        element_token = str(kwargs.get("element_token") or "").strip()
        click_actions = {"click", "double_click", "right_click"}
        try:
            session = self._session_label()
            state_path = pointer_state_path(self.workspace_root, session)
            try:
                loaded = json.loads(state_path.read_text(encoding="utf-8"))
                pointer_state = loaded if isinstance(loaded, dict) else {}
            except (OSError, ValueError, TypeError):
                pointer_state = {}

            now = time.time()
            last_click = pointer_state.get("last_click_timestamp")
            if not isinstance(last_click, (int, float)) or now - float(last_click) > 20:
                # Persistent workspace state must not make a fresh user task inherit
                # an old click-loop lockout. Retain only guards created in the last 20s.
                pointer_state["pixel_clicks_since_progress"] = 0
                pointer_state["recent_pixel_clicks"] = [
                    item for item in pointer_state.get("recent_pixel_clicks", [])
                    if isinstance(item, dict)
                    and isinstance(item.get("timestamp"), (int, float))
                    and now - float(item["timestamp"]) <= 20
                ] if isinstance(pointer_state.get("recent_pixel_clicks", []), list) else []
                pointer_state["recent_element_actions"] = [
                    item for item in pointer_state.get("recent_element_actions", [])
                    if isinstance(item, dict)
                    and isinstance(item.get("timestamp"), (int, float))
                    and now - float(item["timestamp"]) <= 20
                ] if isinstance(pointer_state.get("recent_element_actions", []), list) else []

            if element_token and action not in click_actions | {"type_text"}:
                raise ValueError("element_token is supported for click/right_click/double_click and type_text; use press_key after the element has focus.")

            args = self._driver_args(action, kwargs, session, pointer_state)
            self.validate_pixel_coordinates(action, args, pointer_state)
            if element_token:
                target_pid = pointer_state.get("latest_accessibility_pid")
                if not isinstance(target_pid, int) or target_pid <= 0:
                    raise ValueError("element_token requires computer_inspect first. Inspect the target window and use a token from that exact snapshot.")
                element_map = pointer_state.get("latest_accessibility_elements", {})
                if not isinstance(element_map, dict) or element_token not in element_map:
                    raise ValueError("This element_token is not in the latest inspect result. Re-run computer_inspect and use one of its current tokens.")
                args["pid"] = target_pid
                target_window_id = pointer_state.get("latest_accessibility_window_id")
                if isinstance(target_window_id, int) and not isinstance(target_window_id, bool) and target_window_id > 0:
                    args["window_id"] = target_window_id
                # Persist accessibility click attempts BEFORE dispatch so timeouts cannot
                # trigger blind retries. Compare semantic label/role as well as opaque token:
                # a fresh screenshot produces a new token even when the UI did not change.
                if action in click_actions:
                    target_element = element_map.get(element_token, {})
                    target_role = str(target_element.get("role", "")).casefold()
                    target_label = str(target_element.get("label", "")).strip().casefold()
                    now = time.time()
                    recent_elements = pointer_state.get("recent_element_actions", [])
                    if not isinstance(recent_elements, list):
                        recent_elements = []
                    # Expire guard state quickly so a completed/future task can interact
                    # with a familiar control again. A token is snapshot-scoped: only the
                    # exact same token repeated immediately is blocked here. Blocking by
                    # semantic label across fresh snapshots prevented legitimate later tasks
                    # from clicking the address bar again (the bug seen in real logs).
                    recent_elements = [
                        item for item in recent_elements
                        if isinstance(item, dict)
                        and isinstance(item.get("timestamp"), (int, float))
                        and now - float(item["timestamp"]) <= 20
                    ]
                    repeated = any(
                        item.get("action") == action and item.get("token") == element_token
                        for item in recent_elements[-6:]
                    )
                    if repeated:
                        raise ValueError(
                            "Blocked immediate reuse of the same accessibility token. The token belongs to "
                            "one snapshot; inspect the current UI and verify the previous action before retrying."
                        )
                    recent_elements.append({
                        "token": element_token,
                        "action": action,
                        "role": target_role,
                        "label": target_label,
                        "timestamp": now,
                    })
                    pointer_state["recent_element_actions"] = recent_elements[-8:]
                    # Count this as a click attempt until an input/key/scroll action offers
                    # a plausible state transition; an AX dispatch can also be ineffective.
                    pointer_state["pixel_clicks_since_progress"] = min(
                        4, int(pointer_state.get("pixel_clicks_since_progress", 0) or 0) + 1
                    )
                    pointer_state["last_click_timestamp"] = now
                    self._save_pointer_state(state_path, pointer_state)
                # The accessibility token carries its own snapshot/window. Do not attach pixel target/frame fields.
            elif action == "drag":
                # The accessibility-tree read performed by Computer Use invalidates
                # CUA's active screenshot context. Re-capture pixels in this same
                # session immediately before drag; this action's schema has no
                # capture_id field, so passing one would be rejected by the driver.
                capture_id = pointer_state.get("latest_capture_id")
                if not isinstance(capture_id, str) or not capture_id:
                    raise ValueError("No current desktop capture is recorded. Call computer_use(action='capture') before dragging.")
                target = pointer_state.get("latest_screenshot_target")
                target = target if isinstance(target, dict) else {}
                frame = str(
                    target.get("coordinate_frame")
                    or pointer_state.get("latest_screenshot_coordinate_frame")
                    or "desktop"
                ).strip().casefold()
                args.setdefault("coordinate_frame", frame if frame in {"window", "desktop"} else "desktop")
                capture_id = self._prime_drag_screenshot_context(session, pointer_state)
                pointer_state["last_drag_timestamp"] = now
                pointer_state["last_drag_capture_id"] = capture_id
                self._save_pointer_state(state_path, pointer_state)
            elif action in click_actions:
                # Bind the click to the exact screenshot and keep its coordinate
                # frame (window-local for app captures, desktop only for full-screen).
                target = pointer_state.get("latest_screenshot_target")
                target = target if isinstance(target, dict) else {}
                frame = str(
                    target.get("coordinate_frame")
                    or pointer_state.get("latest_screenshot_coordinate_frame")
                    or "desktop"
                ).strip().casefold()
                args.setdefault("coordinate_frame", frame if frame in {"window", "desktop"} else "desktop")
                capture_id = pointer_state.get("latest_capture_id")
                if not isinstance(capture_id, str) or not capture_id:
                    raise ValueError("No current desktop capture is recorded. Call computer_observe before any pixel click.")
                args["capture_id"] = capture_id

                x, y = int(args["x"]), int(args["y"])
                # A no-op is tied to its screenshot hash and target identity. Do not
                # dispatch the same pixel again while that exact visual state persists.
                no_op = pointer_state.get("last_unverified_pixel_action")
                if isinstance(no_op, dict):
                    age = now - float(no_op.get("timestamp", 0) or 0)
                    same_target = (
                        str(no_op.get("pid") or "") == str(target.get("pid") or args.get("pid") or "")
                        and str(no_op.get("window_id") or "") == str(target.get("window_id") or args.get("window_id") or "")
                        and str(no_op.get("coordinate_frame") or "desktop") == frame
                    )
                    if (
                        0 <= age <= 60
                        and same_target
                        and pointer_state.get("latest_screenshot_hash") == no_op.get("after_screenshot_hash")
                        and int(no_op.get("x", -1)) == x
                        and int(no_op.get("y", -1)) == y
                    ):
                        raise ValueError(
                            f"Blocked repeat click at ({x},{y}): the same target screenshot showed no visible change "
                            "after the previous click. The annotated screenshot identifies the previous point; choose "
                            "a different, visibly supported target or use accessibility controls. No input was dispatched."
                        )
                click_target = dict(target)
                click_target.setdefault("coordinate_frame", frame)
                pointer_state["last_agent_click_target"] = {
                    key: click_target.get(key)
                    for key in ("pid", "window_id", "coordinate_frame", "desktop_capture", "title", "app")
                    if key in click_target
                }
                recent = pointer_state.get("recent_pixel_clicks", [])
                if not isinstance(recent, list):
                    recent = []
                recent = [
                    item for item in recent
                    if isinstance(item, dict)
                    and "x" in item and "y" in item
                    and isinstance(item.get("timestamp"), (int, float))
                    and now - float(item["timestamp"]) <= 20
                ]
                if action in {"click", "right_click"} and any(int(item["x"]) == x and int(item["y"]) == y for item in recent[-5:]):
                    raise ValueError(
                        f"Blocked repeated pixel click at ({x},{y}). Screen animations do not prove progress. "
                        "Stop coordinate guessing; call computer_inspect for the target window and use an element_token, "
                        "or choose a clearly different, verified target. Use double_click only when a double-click is intended."
                    )
                attempts = int(pointer_state.get("pixel_clicks_since_progress", 0) or 0)
                if attempts >= 4:
                    raise ValueError(
                        "Blocked this pixel click after four clicks without a confirmed progress action. "
                        "Do not continue guessing coordinates; inspect the window and use a current element_token, "
                        "or report that the target could not be identified."
                    )
                recent.append({"x": x, "y": y, "action": action, "timestamp": now})
                pointer_state["recent_pixel_clicks"] = recent[-6:]
                pointer_state["recent_element_actions"] = []
                pointer_state["pixel_clicks_since_progress"] = attempts + 1
                pointer_state.update({"x": x, "y": y, "action": action, "last_click_timestamp": now})
                self._save_pointer_state(state_path, pointer_state)

            delivery_mode_escalated = False
            chunked_typing = (
                action == "type_text"
                and not element_token
                and isinstance(args.get("text"), str)
                and len(args["text"]) > _TYPE_CHUNK_CHARS
            )
            try:
                output = (
                    self._type_in_chunks(args) if chunked_typing
                    else call_driver(action, args, timeout=20)
                )
            except CuaDriverError as exc:
                detail = str(exc)
                if chunked_typing:
                    # Part of the text may already be on screen. Never replay it.
                    raise
                # Auto-escalate only when the driver explicitly rejects background
                # desktop input and recommends foreground delivery. Keep this to one
                # internal retry to avoid making the model repeat a protected click.
                if (
                    str(args.get("delivery_mode") or "background").casefold() == "background"
                    and "background_unavailable" in detail
                    and "foreground" in detail.casefold()
                    and action in {"click", "double_click", "right_click", "drag", "type_text", "press_key", "hotkey", "scroll"}
                ):
                    args["delivery_mode"] = "foreground"
                    output = call_driver(action, args, timeout=20)
                    delivery_mode_escalated = True
                else:
                    raise

            if element_token and action in click_actions:
                # Draw the element's center on subsequent screenshots where its bounds are available.
                element_map = pointer_state.get("latest_accessibility_elements", {})
                element = element_map.get(element_token, {}) if isinstance(element_map, dict) else {}
                frame = element.get("frame", {}) if isinstance(element, dict) else {}
                try:
                    x = round(float(frame["x"]) + float(frame["w"]) / 2)
                    y = round(float(frame["y"]) + float(frame["h"]) / 2)
                    if x >= 0 and y >= 0:
                        pointer_state.update({"x": x, "y": y, "action": action})
                        click_target = pointer_state.get("latest_screenshot_target")
                        if isinstance(click_target, dict):
                            pointer_state["last_agent_click_target"] = {
                                key: click_target.get(key)
                                for key in ("pid", "window_id", "coordinate_frame", "desktop_capture", "title", "app")
                                if key in click_target
                            }
                except (KeyError, TypeError, ValueError):
                    pointer_state.pop("x", None)
                    pointer_state.pop("y", None)
                # The attempt was already persisted before dispatch. Keep the pixel-loop
                # budget too: an accessibility click is not verified progress by itself.
                self._save_pointer_state(state_path, pointer_state)
            elif action not in click_actions:
                # Input/keyboard/scroll actions end a click-only sequence; the next observation
                # must still verify that the intended state transition really occurred.
                pointer_state["recent_pixel_clicks"] = []
                pointer_state["recent_element_actions"] = []
                pointer_state["pixel_clicks_since_progress"] = 0
                pointer_state.pop("last_unverified_pixel_action", None)
                pointer_state.pop("last_click_timestamp", None)
                pointer_state.pop("x", None)
                pointer_state.pop("y", None)
                self._save_pointer_state(state_path, pointer_state)

            return ToolResult(
                success=True,
                name=self.name,
                content={
                    "action": action,
                    "target_mode": "accessibility_element" if element_token else "pixel_coordinates" if action in click_actions | {"drag"} else "focused_input",
                    "arguments": {key: value for key, value in args.items() if key != "session"},
                    "driver_output": output[:5000],
                },
                metadata={"kind": "computer_action", "action": action, "target_mode": "accessibility_element" if element_token else "pixel_coordinates" if action in click_actions | {"drag"} else "focused_input", "delivery_mode_escalated": delivery_mode_escalated, "delivery_mode": args.get("delivery_mode", "background")},
                summary=f"Computer action '{action}' was dispatched. This does not prove the intended UI state changed; observe and verify it.",
                evidence={"action": action, "target_mode": "accessibility_element" if element_token else "pixel_coordinates" if action in click_actions | {"drag"} else "focused_input", "driver_output": output[:1800]},
                effects=[{"action": "interact", "target": f"desktop:{action}"}],
            )
        except (CuaDriverError, ValueError, TypeError, KeyError) as exc:
            return ToolResult(
                success=False,
                name=self.name,
                content={"error": {"type": "computer_action_failed", "message": str(exc)}},
                metadata={"recovery_hint": (
                    "Do not repeat the same pixel click or oscillate between guesses. Refresh the desktop observation, "
                    "inspect the target window for a current element_token, and use that token when available. "
                    "If the driver times out or refuses the action, report the exact error rather than retrying blindly."
                )},
                summary=f"Computer action '{action}' failed: {exc}",
            )
