from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool
from src.tools.builtin.computer_use.driver import (
    CuaDriverError, call_driver, pointer_state_path, session_label_from_id,
)


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
                "enum": ["click", "double_click", "right_click", "type_text", "press_key", "hotkey", "scroll"],
                "description": "Exactly one action to perform.",
            },
            "x": {"type": "integer", "description": "Desktop screenshot X coordinate for click, or focus-target for typing/keys."},
            "y": {"type": "integer", "description": "Desktop screenshot Y coordinate for pixel actions; use only from the latest screenshot."},
            "element_token": {"type": "string", "description": "Preferred exact UI element handle returned by computer_inspect. For click or type_text, use this instead of x/y."},
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

    @staticmethod
    def _driver_args(action: str, arguments: dict[str, Any], session: str) -> dict[str, Any]:
        # Coordinates come from get_desktop_state. The tagged desktop target
        # selects screen coordinates; CUA rejects mixing it with legacy `scope`.
        # Keep delivery_mode explicit so callers can opt into foreground input only
        # when the active compositor has a verified, target-addressable adapter.
        delivery_mode = str(arguments.get("delivery_mode") or "background").strip().lower()
        if delivery_mode not in {"background", "foreground"}:
            raise ValueError("delivery_mode must be 'background' or 'foreground'.")
        common = {
            "session": session,
            "target": {"kind": "desktop", "display_id": "primary"},
            "delivery_mode": delivery_mode,
        }

        element_token = str(arguments.get("element_token") or "").strip()
        if element_token and action in {"click", "double_click", "right_click"}:
            # Accessibility tokens are preferred and do not use pixel coordinates.
            return {"session": session, "element_token": element_token, "delivery_mode": delivery_mode}
        if element_token and action == "type_text":
            text = arguments.get("text")
            if not isinstance(text, str):
                raise ValueError("type_text with element_token requires text.")
            return {"session": session, "element_token": element_token, "text": text, "delivery_mode": delivery_mode}

        if action in {"click", "double_click", "right_click"}:
            if not isinstance(arguments.get("x"), (int, float)) or not isinstance(arguments.get("y"), (int, float)):
                raise ValueError(f"{action} requires numeric x and y from the latest screenshot.")
            args = {**common, "x": arguments["x"], "y": arguments["y"]}
            if action == "click":
                args["button"] = arguments.get("button", "left")
            return args

        if action == "type_text":
            text = arguments.get("text")
            if not isinstance(text, str):
                raise ValueError("type_text requires text.")
            args = {**common, "text": text}
            if "x" in arguments or "y" in arguments:
                if not isinstance(arguments.get("x"), (int, float)) or not isinstance(arguments.get("y"), (int, float)):
                    raise ValueError("type_text requires both x and y when targeting a text field.")
                args.update({"x": arguments["x"], "y": arguments["y"]})
            return args

        if action == "press_key":
            key = str(arguments.get("key") or "").strip()
            if not key:
                raise ValueError("press_key requires key.")
            args = {**common, "key": key}
            if "x" in arguments or "y" in arguments:
                if not isinstance(arguments.get("x"), (int, float)) or not isinstance(arguments.get("y"), (int, float)):
                    raise ValueError("press_key requires both x and y when either is supplied.")
                args.update({"x": arguments["x"], "y": arguments["y"]})
            return args

        if action == "hotkey":
            keys = arguments.get("keys")
            if not isinstance(keys, list) or len(keys) < 2 or not all(isinstance(k, str) and k.strip() for k in keys):
                raise ValueError("hotkey requires a list of at least two key names.")
            args = {**common, "keys": [key.strip() for key in keys]}
            if "x" in arguments or "y" in arguments:
                if not isinstance(arguments.get("x"), (int, float)) or not isinstance(arguments.get("y"), (int, float)):
                    raise ValueError("hotkey requires both x and y when either is supplied.")
                args.update({"x": arguments["x"], "y": arguments["y"]})
            return args

        if action == "scroll":
            direction = str(arguments.get("direction") or "").strip().lower()
            if direction not in {"up", "down", "left", "right"}:
                raise ValueError("scroll requires direction: up, down, left, or right.")
            if "x" in arguments or "y" in arguments:
                raise ValueError("Cua Driver scroll acts at the current pointer/focused target and does not accept x/y. Click the target first, then scroll.")
            return {
                **common,
                "direction": direction,
                "amount": max(1, min(20, int(arguments.get("amount", 3)))),
                "by": "line",
            }

        raise ValueError(f"Unsupported computer action: {action}")

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

            args = self._driver_args(action, kwargs, session)
            if element_token:
                target_pid = pointer_state.get("latest_accessibility_pid")
                if not isinstance(target_pid, int) or target_pid <= 0:
                    raise ValueError("element_token requires computer_inspect first. Inspect the target window and use a token from that exact snapshot.")
                element_map = pointer_state.get("latest_accessibility_elements", {})
                if not isinstance(element_map, dict) or element_token not in element_map:
                    raise ValueError("This element_token is not in the latest inspect result. Re-run computer_inspect and use one of its current tokens.")
                args["pid"] = target_pid
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
            elif action in click_actions:
                # Pixel coordinates must refer to the latest exact desktop capture.
                args["coordinate_frame"] = "desktop"
                capture_id = pointer_state.get("latest_capture_id")
                if not isinstance(capture_id, str) or not capture_id:
                    raise ValueError("No current desktop capture is recorded. Call computer_observe before any pixel click.")
                args["capture_id"] = capture_id

                x, y = int(args["x"]), int(args["y"])
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
            try:
                output = call_driver(action, args, timeout=20)
            except CuaDriverError as exc:
                detail = str(exc)
                # Auto-escalate only when the driver explicitly rejects background
                # desktop input and recommends foreground delivery. Keep this to one
                # internal retry to avoid making the model repeat a protected click.
                if (
                    str(args.get("delivery_mode") or "background").casefold() == "background"
                    and "background_unavailable" in detail
                    and "foreground" in detail.casefold()
                    and action in {"click", "double_click", "right_click", "type_text", "press_key", "hotkey", "scroll"}
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
                pointer_state.pop("last_click_timestamp", None)
                pointer_state.pop("x", None)
                pointer_state.pop("y", None)
                self._save_pointer_state(state_path, pointer_state)

            return ToolResult(
                success=True,
                name=self.name,
                content={
                    "action": action,
                    "target_mode": "accessibility_element" if element_token else "pixel_coordinates" if action in click_actions else "focused_input",
                    "arguments": {key: value for key, value in args.items() if key != "session"},
                    "driver_output": output[:5000],
                },
                metadata={"kind": "computer_action", "action": action, "target_mode": "accessibility_element" if element_token else "pixel_coordinates" if action in click_actions else "focused_input", "delivery_mode_escalated": delivery_mode_escalated, "delivery_mode": args.get("delivery_mode", "background")},
                summary=f"Computer action '{action}' was dispatched. This does not prove the intended UI state changed; observe and verify it.",
                evidence={"action": action, "target_mode": "accessibility_element" if element_token else "pixel_coordinates" if action in click_actions else "focused_input", "driver_output": output[:1800]},
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
