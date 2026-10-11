from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool
from src.tools.builtin.computer_use.driver import (
    CuaDriverError,
    call_driver,
    pointer_state_path,
    _desktop_session_environment,
)
from src.tools.builtin.computer_use.observe import ComputerObserveTool
from src.tools.builtin.computer_use.inspect import ComputerInspectTool
from src.tools.builtin.computer_use.act import ComputerActTool


class ComputerUseTool(Tool):
    """One model-facing computer-use tool with a stable, concise action contract."""

    name = "computer_use"
    action = "interact"
    description = (
        "One unified desktop-control tool. Start with computer_use(action='capture', mode='som', app='Firefox'). "
        "Capture returns a screenshot plus a short list of numbered accessible controls (#1, #2, ...). Use "
        "computer_use(action='click', element=2) with a number from the latest capture; IDs expire after "
        "every new capture. Use mode='ax' for text-only controls or mode='vision' for pixels only. After a "
        "state-changing actions are captured automatically and should be verified before continuing. "
        "For URLs, action='navigate', url='https://…' opens a new tab through the selected browser executable and then captures the result. "
        "Other actions: type, key, scroll, double_click, right_click, wait, list_apps, list_windows. "
        "Prefer numbered elements over coordinates. Input is background-first; never guess pixels "
        "or repeat an action after a timeout/no-op. This is the only Computer Use tool exposed to the model. "
        "If the app is not open, use action='launch', app='firefox' instead of clicking the taskbar. "
        "If a capture reports 0 numbered controls, NEVER use element=N. "
        "Every click/type/key is captured and checked automatically unless capture_after=false is explicitly supplied. "
        "When a text field is listed, use action='type', element=N, text='...' to type directly into that field; "
        "do not click it first unless required. "
        "The red ring marks where the LAST click landed (not the live cursor). "
        "Read the numbered controls list in the capture result and choose the element whose LABEL matches your goal; "
        "never assume element=1. To find a specific control use capture with query='<visible label>' (e.g. query='ورود'). "
        "Never click the same element twice in a row."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "capture",
                    "launch",
                    "navigate",
                    "click",
                    "double_click",
                    "right_click",
                    "middle_click",
                    "type",
                    "key",
                    "scroll",
                    "wait",
                    "list_apps",
                    "list_windows",
                ],
                "description": "Hermes-style single-tool action. Capture first; use only element IDs from the latest capture.",
            },
            "url": {
                "type": "string",
                "description": "http(s) URL for action='navigate'.",
            },
            "mode": {
                "type": "string",
                "enum": ["som", "vision", "ax"],
                "default": "som",
                "description": "capture: som returns screenshot + numbered controls; vision returns screenshot only; ax returns controls only.",
            },
            "app": {
                "type": "string",
                "description": "Application name for capture/launch or browser to use for navigate, e.g. 'Firefox' or 'Chrome'.",
            },
            "query": {
                "type": "string",
                "description": "Optional target label filter for capture, e.g. 'address bar' or 'جست‌و‌جو در آپارات'. Leave empty to list controls.",
            },
            "element": {
                "type": "integer",
                "minimum": 1,
                "description": "Numbered element from the latest capture. For text entry, use action='type', element=N, text='...' to target the field directly. IDs expire after a new capture; never invent them.",
            },
            "coordinate": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": "[x,y] from the latest full-desktop screenshot; use only if capture provides no accessible target.",
            },
            "capture_after": {
                "type": "boolean",
                "default": True,
                "description": "Defaults to true: capture the screen after an interaction and return fresh numbered elements in the same call. Set false only when intentionally skipping verification.",
            },
            "window_title": {
                "type": "string",
                "description": "Optional visible-window title substring to disambiguate inspect/focus.",
            },
            "pid": {
                "type": "integer",
                "description": "Optional process id from the current window list. Usually omit.",
            },
            "window_id": {
                "type": ["integer", "string"],
                "description": "Optional native window id. Usually omit; exact whole-number scientific notation is accepted.",
            },
            "text": {"type": "string", "description": "Text for action='type'."},
            "key": {
                "type": "string",
                "description": "Single key for action='key', e.g. ENTER or ESC.",
            },
            "keys": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "description": "Key or chord for action='key', e.g. 'ctrl+l', 'return', or ['ctrl','l'].",
            },
            "from_element": {
                "type": "integer",
                "minimum": 1,
                "description": "Start element ID for action='drag'.",
            },
            "to_element": {
                "type": "integer",
                "minimum": 1,
                "description": "End element ID for action='drag'.",
            },
            "from_coordinate": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
            "to_coordinate": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
            "button": {
                "type": "string",
                "enum": ["left", "right", "middle"],
                "default": "left",
            },
            "delivery_mode": {
                "type": "string",
                "enum": ["background", "foreground"],
                "default": "background",
                "description": "Background is attempted first. If the driver explicitly reports background_unavailable and recommends foreground, the tool retries that action once automatically.",
            },
            "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
            "amount": {"type": "integer", "minimum": 1, "maximum": 20, "default": 3},
            "seconds": {"type": "number", "minimum": 0, "maximum": 10, "default": 1},
            "include_windows": {"type": "boolean", "default": True},
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 30,
                "default": 24,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self._observe = ComputerObserveTool()
        self._inspect = ComputerInspectTool()
        self._act = ComputerActTool()
        self.workspace_root = self._observe.workspace_root
        self.session_id_provider = None

    def set_workspace(self, directory: str) -> None:
        self.workspace_root = Path(directory).expanduser().resolve()
        for helper in (self._observe, self._inspect, self._act):
            helper.set_workspace(self.workspace_root)

    def set_context_store(self, stm, session_id_provider) -> None:
        self.session_id_provider = session_id_provider
        for helper in (self._observe, self._inspect, self._act):
            helper.set_context_store(stm, session_id_provider)

    @staticmethod
    def _copy_result(result: ToolResult, action: str) -> ToolResult:
        content = dict(result.content)
        content["action"] = action
        metadata = dict(result.metadata)
        metadata["action"] = action
        return ToolResult(
            success=result.success,
            name="computer_use",
            content=content,
            metadata=metadata,
            summary=result.summary,
            evidence=result.evidence,
            effects=result.effects,
        )

    def _current_state(self) -> tuple[Path, dict[str, Any]]:
        path = pointer_state_path(self.workspace_root, self._observe._session_label())
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            state = value if isinstance(value, dict) else {}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            state = {}
        return path, state

    @staticmethod
    def _save_state(path: Path, state: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            pass

    def _element_record(self, element_id: Any) -> tuple[str, dict[str, Any]]:
        if isinstance(element_id, bool):
            raise ValueError(
                "element must be a numbered control from the latest capture."
            )
        try:
            numeric_id = int(element_id)
        except (TypeError, ValueError):
            raise ValueError(
                "element must be a numbered control from the latest capture, e.g. element=2."
            )
        if numeric_id <= 0 or str(numeric_id) != str(element_id).strip():
            raise ValueError(
                "element must be a positive integer from the latest capture."
            )
        _path, state = self._current_state()
        mapping = state.get("latest_element_ids")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(
                "The latest capture had ZERO numbered controls, so element=N cannot be used at all. "
                "To open an app use action='launch' app='<name>'. Otherwise click with coordinate=[x,y] "
                "taken from the screenshot, with capture_after=true, and check the red ring."
            )
        if str(numeric_id) not in mapping:
            raise ValueError(
                f"Element #{numeric_id} is not in the latest capture (valid: 1..{len(mapping)}). Capture again."
            )
        token = str(mapping[str(numeric_id)])
        token_map = state.get("latest_accessibility_elements")
        record = token_map.get(token) if isinstance(token_map, dict) else None
        if not isinstance(record, dict):
            raise ValueError(
                f"Element #{numeric_id} no longer exists in the latest accessibility snapshot. Capture again."
            )
        return token, record

    @staticmethod
    def _reading_order(row: Any) -> tuple[float, float]:
        """Sort controls top-to-bottom (20px bands), then left-to-right."""
        frame = row.get("frame") if isinstance(row, dict) else None
        try:
            return (round(float(frame["y"]) / 20), float(frame["x"]))
        except (TypeError, KeyError, ValueError):
            return (1e9, 0.0)

    def _capture(
        self,
        *,
        mode: str = "som",
        app: str = "",
        query: str = "",
        window_title: str = "",
        pid: Any = None,
        window_id: Any = None,
        max_results: int = 24,
        include_windows: bool = True,
    ) -> ToolResult:
        mode = str(mode or "som").strip().casefold()
        if mode not in {"som", "vision", "ax"}:
            raise ValueError("mode must be 'som', 'vision', or 'ax'.")
        hint = str(window_title or app or "").strip()
        elements: list[dict[str, Any]] = []
        element_id_map: dict[str, str] = {}
        target: dict[str, Any] | None = None
        inspection_error: str | None = None

        # Screenshots are captured only for modes that need them; AX mode avoids the
        # image in model context, which matters for text-only local models.
        observed: ToolResult | None = None
        if mode != "ax":
            observed = self._observe.execute(include_windows=include_windows)
            if not observed.success:
                return self._copy_result(observed, "capture")

        inspected: ToolResult | None = None
        query_fallback_used = False
        if mode != "vision":
            inspected = self._inspect.execute(
                query=str(query or ""),
                window_title=hint,
                pid=pid,
                window_id=window_id,
                max_results=max(1, min(30, int(max_results))),
            )

            source = inspected.content.get("elements", []) if isinstance(inspected.content, dict) else []
            # Query labels can be in another language (e.g. "phone number" vs Persian
            # "شماره تلفن"). If the specific query fails, re-read the same window without
            # filtering, so the model can choose among actual controls instead of guessing pixels.
            if query and (
                not inspected.success or not isinstance(source, list) or not source
            ):
                query_fallback_used = True
                query_failure = str(inspected.summary or "No controls matched the requested query.")
                broad = self._inspect.execute(
                    query="",
                    window_title=hint,
                    pid=pid,
                    window_id=window_id,
                    max_results=max(1, min(30, int(max_results))),
                )
                if broad.success:
                    inspected = broad
                    inspection_error = (
                        f"Requested query {query!r} did not identify a control; "
                        "showing the unfiltered accessible controls instead. "
                        + query_failure
                    )

            if inspected.success:
                target = (
                    inspected.content.get("window")
                    if isinstance(inspected.content, dict)
                    else None
                )
                source_elements = (
                    inspected.content.get("elements", [])
                    if isinstance(inspected.content, dict)
                    else []
                )
                if isinstance(source_elements, list):
                    # Number controls in reading order (top to bottom) so #1 is the
                    # top-most control, not whatever the ranker scored highest.
                    source_elements = sorted(source_elements, key=self._reading_order)
                    for row in source_elements:
                        if not isinstance(row, dict) or not row.get("element_token"):
                            continue
                        number = len(elements) + 1
                        element_id_map[str(number)] = str(row["element_token"])
                        elements.append(
                            {
                                "id": number,
                                "role": row.get("role", ""),
                                "label": row.get("label", ""),
                                "actions": row.get("actions", []),
                                "enabled": bool(row.get("enabled", True)),
                                "in_web_content": bool(row.get("in_web_content", False)),
                                **({"value": row["value"]} if row.get("value") else {}),
                                **({"bounds": row["frame"]} if isinstance(row.get("frame"), dict) else {}),
                            }
                        )
                state_path, state = self._current_state()
                # Element numbers are scoped to one capture. Replacing this map
                # invalidates every old number, exactly as in Hermes' wrapper.
                state["latest_element_ids"] = element_id_map
                state["latest_computer_use_target"] = target or {}
                state["latest_computer_use_mode"] = mode
                self._save_state(state_path, state)
            else:
                inspection_error = str(inspected.summary or "Accessibility inspection unavailable.")
                # A fresh capture invalidates old IDs even when the tree read failed.
                state_path, state = self._current_state()
                state["latest_element_ids"] = {}
                self._save_state(state_path, state)

        if mode == "vision":
            state_path, state = self._current_state()
            state["latest_element_ids"] = {}
            self._save_state(state_path, state)

        no_controls = mode != "vision" and not elements
        content: dict[str, Any] = {
            "action": "capture",
            "mode": mode,
            "window": target,
            "element_count": len(elements) if mode != "vision" else 0,
            "elements": elements if mode != "vision" else [],
            "note": "Element IDs are valid only until the next capture. Use computer_use(action='click', element=N), never an element_token.",
        }
        if no_controls:
            content["warning"] = (
                "NO numbered controls were found: click/type with element=N is impossible and will fail. "
                + (f"Reason: {inspection_error}. " if inspection_error else "")
                + "If the target app is not open, use action='launch' app='<name>'. "
                "Otherwise use coordinate=[x,y] from the screenshot with capture_after=true."
            )
        metadata: dict[str, Any] = {
            "kind": "computer_use_capture",
            "action": "capture",
            "mode": mode,
        }
        if observed is not None:
            content.update(
                {
                    "image_ref": observed.metadata.get("image_ref")
                    or observed.content.get("image_ref"),
                    "image_mime_type": observed.metadata.get("image_mime_type")
                    or observed.content.get("image_mime_type", "image/png"),
                    "width": observed.content.get("width"),
                    "height": observed.content.get("height"),
                    "screenshot_hash": observed.content.get("screenshot_hash"),
                    "visual_hash": observed.content.get("visual_hash"),
                    "windows": observed.content.get("windows", ""),
                    "cursor_position": observed.content.get("cursor_position"),
                    "cursor_hint": observed.content.get("cursor_hint"),
                    "cursor_overlay": observed.content.get("cursor_overlay"),
                }
            )
            if observed.content.get("pointer_overlay"):
                content["pointer_hint"] = (
                    "A red ring on the screenshot marks where the agent's LAST click landed (not the live cursor). "
                    "If it is not on the intended target, the click missed: do not repeat it, pick a different target."
                )
            metadata.update(
                {
                    "image_ref": content["image_ref"],
                    "image_mime_type": content["image_mime_type"],
                    "width": content["width"],
                    "height": content["height"],
                    "screenshot_hash": content.get("screenshot_hash"),
                    "visual_hash": content.get("visual_hash"),
                    "cursor_position": content.get("cursor_position"),
                    "cursor_overlay": content.get("cursor_overlay"),
                }
            )
        if inspection_error:
            if query_fallback_used:
                content["query_fallback"] = True
                content["query_requested"] = query
                content["selection_warning"] = inspection_error
            else:
                content["accessibility_error"] = inspection_error

        if no_controls:
            resolution = ""
            if observed is not None:
                screenshot_width = observed.content.get("width")
                screenshot_height = observed.content.get("height")
                if (
                    isinstance(screenshot_width, int) and screenshot_width > 0
                    and isinstance(screenshot_height, int) and screenshot_height > 0
                ):
                    resolution = (
                        f" Screenshot resolution is {screenshot_width}x{screenshot_height}; "
                        f"valid pixel coordinates are x=0..{screenshot_width - 1}, "
                        f"y=0..{screenshot_height - 1}."
                    )
            summary = (
                "Captured the desktop but 0 numbered controls are available, so element=N will FAIL. "
                + (f"Inspection said: {inspection_error} " if inspection_error else "")
                + "Use action='launch' to open an app, or use coordinate=[x,y] only when the intended point is clearly visible in this exact screenshot."
                + resolution
            )
        elif mode == "vision":
            summary = "Captured desktop screenshot (vision-only mode; no element IDs were generated)."
        else:
            summary = f"Captured {(target or {}).get('title') or (target or {}).get('app') or 'desktop'}: {len(elements)} numbered UI controls."

        if query_fallback_used and inspection_error:
            summary += (
                "\nNOTICE: the requested query did not match; these are the unfiltered accessible controls. "
                "Choose by the displayed label and role, not by element number alone."
            )

        # Put the actual control list in the text the model reads. Small models do not
        # reliably dig through structured `content`, so they used to guess element=1.
        if elements:
            lines = []
            for e in elements:
                label = str(e.get("label") or e.get("value") or "").replace("\n", " ")[
                    :60
                ]
                b = e.get("bounds") or {}
                try:
                    where = f" @({int(float(b['x']) + float(b['w']) / 2)},{int(float(b['y']) + float(b['h']) / 2)})"
                except (KeyError, TypeError, ValueError):
                    where = ""
                lines.append(f"#{e['id']} {e.get('role', '')} \"{label}\"{where}")
            content["elements_text"] = "\n".join(lines)
            summary += (
                "\nControls (pick by LABEL, not by guess):\n" + content["elements_text"]
            )
            summary += "\nIf the control you need is not listed, capture again with query='<its label>'."

        return ToolResult(
            success=(
                observed.success
                if observed is not None
                else bool(inspected and inspected.success)
            ),
            name=self.name,
            content=content,
            metadata=metadata,
            summary=summary,
            evidence={
                "kind": "computer_use_capture",
                "mode": mode,
                "window": target,
                "element_count": len(elements),
                "image_ref": metadata.get("image_ref"),
                "accessibility_error": inspection_error,
            },
        )

    def _act_once(self, action: str, **kwargs: Any) -> ToolResult:
        action = str(action or "").strip().casefold()
        args = dict(kwargs)
        # Translate Hermes' compact action vocabulary into the existing CUA adapter.
        if action == "type":
            action = "type_text"
        elif action == "key":
            keys = args.pop("keys", None)
            key = args.pop("key", None)
            if isinstance(keys, list):
                keys = [str(part).strip() for part in keys if str(part).strip()]
            elif isinstance(keys, str):
                keys = [part for part in keys.replace("+", " ").split() if part]
            else:
                keys = []
            if len(keys) > 1:
                action, args["keys"] = "hotkey", keys
            elif key:
                action, args["key"] = "press_key", key
            elif keys:
                action, args["key"] = "press_key", keys[0]
            else:
                raise ValueError("key requires key='ENTER' or keys='ctrl+l'.")
        elif action == "middle_click":
            action, args["button"] = "click", "middle"

        click_like = {"click", "double_click", "right_click"}
        click_signature: str | None = None
        typing_fallback_token: str | None = None
        typing_fallback_text: str | None = None
        element_id = args.pop("element", None)
        state_path, state = self._current_state()
        if element_id is not None:
            token, record = self._element_record(element_id)
            args["element_token"] = token
            if action == "type_text":
                role = str(record.get("role") or "").casefold().strip()
                actions = record.get("actions", [])
                actions = [str(item).casefold().replace("_", "") for item in actions] if isinstance(actions, list) else []
                supports_direct_value = any(
                    item in {"setvalue", "settext", "inserttext", "replacevalue", "editabletext"}
                    for item in actions
                )
                # Some web form entries only expose "activate" in AT-SPI. Sending
                # text to their element token may be a successful no-op. Focus the
                # exact current element first, then type through the keyboard channel.
                if role in {"entry", "text field", "text input", "editable text", "password text", "combo box"} and not supports_direct_value:
                    text_value = args.get("text")
                    if isinstance(text_value, str):
                        typing_fallback_token = token
                        typing_fallback_text = text_value
            if action in click_like:
                click_signature = f"{record.get('role')}|{record.get('label')}"
                if state.get("last_element_click") == click_signature:
                    raise ValueError(
                        f"You already clicked this same control ({record.get('label')!r}) and the page did not "
                        "change as intended. Do not click it again. Read the controls list, choose a DIFFERENT "
                        "control, or capture with query='<label>' (e.g. query='ورود')."
                    )
        if action not in click_like:
            # Typing / keys / scroll are progress even when directly targeting an
            # accessibility element; clear the repeated-click guard either way.
            if state.pop("last_element_click", None) is not None:
                self._save_state(state_path, state)

        coordinate = args.pop("coordinate", None)
        if coordinate is not None:
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                raise ValueError(
                    "coordinate must be [x, y] from the latest capture screenshot."
                )
            args["x"], args["y"] = coordinate

        allowed = {
            "element_token",
            "x",
            "y",
            "text",
            "key",
            "keys",
            "button",
            "direction",
            "amount",
            "delivery_mode",
        }
        args = {
            key: value
            for key, value in args.items()
            if key in allowed and value is not None
        }
        if typing_fallback_token is not None and typing_fallback_text is not None:
            delivery_mode = str(args.get("delivery_mode") or "background")
            focus_result = self._act.execute(
                action="click",
                element_token=typing_fallback_token,
                delivery_mode=delivery_mode,
            )
            if not focus_result.success:
                return ToolResult(
                    success=False,
                    name="computer_use",
                    content={
                        "action": "type",
                        "target_mode": "accessibility_focus_then_keyboard",
                        "failed_stage": "focus_text_field",
                        "error": focus_result.content.get("error", focus_result.summary),
                    },
                    metadata={"recovery_hint": "The input field could not be focused. Read the error and refreshed controls before trying another method."},
                    summary=f"Could not focus the selected text field before typing: {focus_result.summary}",
                    evidence={"focus_result": focus_result.evidence},
                )
            type_result = self._act.execute(
                action="type_text",
                text=typing_fallback_text,
                delivery_mode=delivery_mode,
            )
            return ToolResult(
                success=type_result.success,
                name="computer_use",
                content={
                    "action": "type",
                    "target_mode": "accessibility_focus_then_keyboard",
                    "field_role": str(record.get("role") or ""),
                    "field_label": str(record.get("label") or ""),
                    "focus_succeeded": focus_result.success,
                    "driver_output": type_result.content.get("driver_output", ""),
                    **({"error": type_result.content.get("error")} if not type_result.success else {}),
                },
                metadata={
                    "action": "type",
                    "target_mode": "accessibility_focus_then_keyboard",
                    "field_role": str(record.get("role") or ""),
                    "delivery_mode": type_result.metadata.get("delivery_mode", delivery_mode),
                    "delivery_mode_escalated": type_result.metadata.get("delivery_mode_escalated", False),
                },
                summary=(
                    "Focused the selected text field and sent keyboard text. Capture the result and verify the visible field value before proceeding."
                    if type_result.success
                    else f"Focused the field, but keyboard text entry failed: {type_result.summary}"
                ),
                evidence={
                    "target_mode": "accessibility_focus_then_keyboard",
                    "focus_result": focus_result.evidence,
                    "type_result": type_result.evidence,
                },
                effects=type_result.effects,
            )
        result = self._act.execute(action=action, **args)

        if click_signature is not None and result.success:
            # Remember only clicks that were actually dispatched. Re-read state because
            # the act tool rewrites the same file.
            state_path, state = self._current_state()
            state["last_element_click"] = click_signature
            self._save_state(state_path, state)
        return result

    _APP_ALIASES = {
        "firefox": ["firefox", "firefox-esr"],
        "chrome": [
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
        ],
        "vscode": ["code"],
        "files": ["nautilus", "dolphin", "thunar"],
        "terminal": ["gnome-terminal", "konsole", "xterm"],
    }

    def _launch(self, app: str, seconds: Any) -> ToolResult:
        name = str(app or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._+-]{0,59}", name):
            return ToolResult(
                success=False,
                name=self.name,
                content={
                    "error": {
                        "type": "invalid_app",
                        "message": "launch needs app='firefox' (a plain application name).",
                    }
                },
                summary="Launch refused: invalid app name.",
            )
        key = name.casefold()
        candidates = self._APP_ALIASES.get(key, [key.replace(" ", "-")])
        exe = next((path for c in candidates if (path := shutil.which(c))), None)
        gtk = shutil.which("gtk-launch")
        try:
            env = _desktop_session_environment()
            if exe:
                command = [exe]
            elif gtk:
                command = [gtk, candidates[0]]
            else:
                raise CuaDriverError(
                    f"Could not find an executable or .desktop entry for '{name}'."
                )
            subprocess.Popen(
                command,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
        except (CuaDriverError, OSError) as exc:
            return ToolResult(
                success=False,
                name=self.name,
                content={
                    "action": "launch",
                    "error": {"type": "launch_failed", "message": str(exc)},
                },
                summary=f"Launch of '{name}' failed: {exc}",
            )
        try:
            wait = max(3.0, min(float(seconds), 10.0))
        except (TypeError, ValueError):
            wait = 4.0
        time.sleep(wait)
        captured = self._capture(
            mode="som", app=name, max_results=24, include_windows=True
        )
        content = dict(captured.content)
        content.update({"action": "launch", "launched": name})
        metadata = dict(captured.metadata)
        metadata["action"] = "launch"
        return ToolResult(
            success=captured.success,
            name=self.name,
            content=content,
            metadata=metadata,
            summary=f"Started '{name}' and captured the screen. "
            + str(captured.summary),
            evidence=captured.evidence,
            effects=[{"action": "interact", "target": f"launch:{name}"}],
        )

    _BROWSER_ALIASES = {
        "firefox": ["firefox", "firefox-esr"],
        "chrome": ["google-chrome", "google-chrome-stable"],
        "chromium": ["chromium", "chromium-browser"],
        "brave": ["brave-browser", "brave"],
        "edge": ["microsoft-edge", "microsoft-edge-stable"],
    }

    def _navigation_browser(self, requested_app: str = "") -> tuple[str, str]:
        requested = str(requested_app or "").strip().casefold()
        aliases = {
            "mozilla firefox": "firefox",
            "google chrome": "chrome",
            "ms edge": "edge",
            "microsoft edge": "edge",
        }
        requested = aliases.get(requested, requested)
        keys: list[str] = []
        if requested:
            if requested in self._BROWSER_ALIASES:
                keys.append(requested)
            else:
                safe = re.sub(r"[^a-z0-9._+-]", "", requested)
                if safe:
                    keys.append(safe)
        else:
            # Prefer a browser already visible in the window inventory.
            try:
                windows = self._inspect._windows()
            except (CuaDriverError, ValueError, TypeError):
                windows = []
            windows = sorted(
                windows,
                key=lambda row: (
                    int(row.get("z_index") or 0),
                    int(row.get("width") or 0) * int(row.get("height") or 0),
                ),
                reverse=True,
            )
            signatures = [
                f"{row.get('app_name', '')} {row.get('title', '')}".casefold()
                for row in windows if self._inspect._is_browser(row)
            ]
            for signature in signatures:
                for key, words in (
                    ("firefox", ("firefox",)),
                    ("chrome", ("google chrome", "chrome")),
                    ("chromium", ("chromium",)),
                    ("brave", ("brave",)),
                    ("edge", ("edge",)),
                ):
                    if any(word in signature for word in words) and key not in keys:
                        keys.append(key)
            keys.extend(key for key in ("firefox", "chrome", "chromium", "brave", "edge") if key not in keys)

        for key in keys:
            candidates = self._BROWSER_ALIASES.get(key, [key])
            exe = next((path for candidate in candidates if (path := shutil.which(candidate))), None)
            if exe:
                return key, exe
        raise CuaDriverError(
            f"Could not find a browser executable for {requested_app or 'the visible browser'}. "
            "Set app='Firefox' or install a supported browser."
        )

    def _navigate(self, url: str, seconds: float, app: str = "") -> ToolResult:
        raw_url = str(url or "").strip()
        parsed = urlparse(raw_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            return ToolResult(
                success=False,
                name=self.name,
                content={"error": {"type": "invalid_url", "message": "navigate accepts only a complete http(s) URL without embedded credentials."}},
                summary="Navigation refused because the URL was not a valid http(s) URL.",
            )

        try:
            browser_key, executable = self._navigation_browser(app)
            env = _desktop_session_environment()
            # Do not send Ctrl+L/type/Enter to whichever window happens to have focus.
            # Browser CLIs route this URL to their own existing process and open a new tab.
            subprocess.Popen(
                [executable, "--new-tab", raw_url],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
        except (CuaDriverError, OSError) as exc:
            return ToolResult(
                success=False,
                name=self.name,
                content={"action": "navigate", "url": raw_url, "error": {"type": "navigation_failed", "message": str(exc)}},
                summary=f"Could not open {parsed.hostname} in the browser: {exc}",
            )

        try:
            time.sleep(max(0.0, min(float(seconds), 5.0)))
        except (TypeError, ValueError):
            time.sleep(1.0)

        captured = self._capture(
            mode="som", app=browser_key, max_results=24, include_windows=True
        )
        content = dict(captured.content)
        title = str((content.get("window") or {}).get("title") or "")
        title_confirms_host = parsed.hostname.casefold() in title.casefold()
        content.update({
            "action": "navigate",
            "url_requested": raw_url,
            "browser": browser_key,
            "verification": (
                "The browser was asked to open this URL in a new tab. Inspect the screenshot and page title "
                "before claiming the destination loaded."
            ),
            "destination_in_title": title_confirms_host,
        })
        metadata = dict(captured.metadata)
        metadata.update({"action": "navigate", "url_requested": raw_url, "browser": browser_key})
        return ToolResult(
            success=captured.success,
            name=self.name,
            content=content,
            metadata=metadata,
            summary=(
                f"Opened a new tab request for {parsed.hostname} in {browser_key}; captured the browser window. "
                "Verify the page visually before claiming it loaded.\n" + str(captured.summary)
                if captured.success
                else f"Browser launch request was sent, but the follow-up capture failed: {captured.summary}"
            ),
            evidence={
                "action": "navigate",
                "url_requested": raw_url,
                "browser": browser_key,
                "destination_in_title": title_confirms_host,
                "screenshot": metadata.get("image_ref"),
                "window": content.get("window"),
            },
        )

    def execute(
        self,
        action: str,
        **kwargs: Any,
    ) -> ToolResult:
        action = str(action or "").strip().casefold()
        # Compatibility aliases are accepted internally, but never exposed in the
        # model-facing schema: the model sees one Hermes-style computer_use tool.
        if action in {"observe", "inspect"}:
            action = "capture"
        elif action == "type_text":
            action = "type"

        try:
            if action == "capture":
                return self._capture(
                    mode=str(kwargs.get("mode") or "som"),
                    app=str(kwargs.get("app") or ""),
                    query=str(kwargs.get("query") or ""),
                    window_title=str(kwargs.get("window_title") or ""),
                    pid=kwargs.get("pid"),
                    window_id=kwargs.get("window_id"),
                    max_results=kwargs.get("max_results", 24),
                    include_windows=bool(kwargs.get("include_windows", True)),
                )

            if action == "launch":
                return self._launch(
                    str(kwargs.get("app") or ""), kwargs.get("seconds", 4.0)
                )

            if action == "navigate":
                return self._navigate(
                    str(kwargs.get("url") or ""),
                    kwargs.get("seconds", 1.0),
                    str(kwargs.get("app") or ""),
                )

            if action in {
                "click",
                "double_click",
                "right_click",
                "middle_click",
                "type",
                "key",
                "scroll",
                "press_key",
                "hotkey",
            }:
                before_state = self._current_state()[1]
                before_screen_hash = before_state.get("latest_screenshot_hash")
                result = self._act_once(action, **kwargs)
                if not result.success or not bool(kwargs.get("capture_after", True)):
                    return self._copy_result(result, action)

                # Re-capture the same app/window immediately after the action. No
                # stale element number is retained by the new capture.
                _state_path, state = self._current_state()
                target = state.get("latest_computer_use_target", {})
                target = target if isinstance(target, dict) else {}
                captured = self._capture(
                    mode=str(kwargs.get("mode") or "som"),
                    pid=target.get("pid"),
                    window_id=target.get("window_id"),
                    max_results=kwargs.get("max_results", 24),
                    include_windows=bool(kwargs.get("include_windows", True)),
                )
                after_state = self._current_state()[1]
                after_screen_hash = after_state.get("latest_screenshot_hash")
                screen_hash_changed = (
                    before_screen_hash != after_screen_hash
                    if isinstance(before_screen_hash, str) and isinstance(after_screen_hash, str)
                    else None
                )
                content = dict(result.content)
                content["action"] = action
                content["capture_after"] = {
                    "success": captured.success,
                    "screen_hash_changed": screen_hash_changed,
                    "window": captured.content.get("window"),
                    "elements": captured.content.get("elements", []),
                    "elements_text": captured.content.get("elements_text", ""),
                    "element_count": captured.content.get("element_count", 0),
                    "mode": captured.content.get("mode", kwargs.get("mode", "som")),
                    "error": captured.content.get("accessibility_error"),
                    "warning": captured.content.get("warning") or captured.content.get("selection_warning"),
                    "query_fallback": captured.content.get("query_fallback", False),
                    "no_visible_change_warning": (
                        "No screenshot pixel change was detected after this action. Its intended effect is unverified; "
                        "do not repeat the same click. Read the fresh controls, try direct text entry into a listed field, "
                        "or stop if the target cannot be identified."
                        if screen_hash_changed is False else None
                    ),
                }
                metadata = dict(result.metadata)
                metadata["screen_hash_changed"] = screen_hash_changed
                metadata["action"] = action
                if captured.content.get("image_ref"):
                    content["image_ref"] = captured.content["image_ref"]
                    content["screenshot_hash"] = captured.content.get("screenshot_hash")
                    content["visual_hash"] = captured.content.get("visual_hash")
                    content["cursor_position"] = captured.content.get("cursor_position")
                    content["cursor_hint"] = captured.content.get("cursor_hint")
                    content["image_mime_type"] = captured.content.get(
                        "image_mime_type", "image/png"
                    )
                    metadata["image_ref"] = captured.content["image_ref"]
                    metadata["image_mime_type"] = captured.content.get(
                        "image_mime_type", "image/png"
                    )
                if screen_hash_changed is False:
                    metadata["verification_warning"] = (
                        "No screenshot pixel change was detected; the intended effect is not verified. "
                        "Do not repeat the same click."
                    )
                metadata["capture_after"] = True
                return ToolResult(
                    success=result.success and captured.success,
                    name=self.name,
                    content=content,
                    metadata=metadata,
                    summary=(
                        ("WARNING: No screenshot pixel change detected; do not repeat the same click. "
                         if screen_hash_changed is False else "")
                        + f"Action '{action}' was dispatched and the latest screen was captured. "
                        "This does not prove the intended state changed; inspect the screenshot and fresh controls.\n"
                        + str(captured.summary)
                        if captured.success
                        else f"Action '{action}' was dispatched, but the follow-up capture failed: {captured.summary}"
                    ),
                    evidence={
                        "action": action,
                        "capture_after": captured.evidence,
                        "driver_result": result.evidence,
                    },
                    effects=result.effects,
                )

            if action == "wait":
                seconds = max(0.0, min(10.0, float(kwargs.get("seconds", 1.0))))
                time.sleep(seconds)
                return ToolResult(
                    success=True,
                    name=self.name,
                    content={"action": "wait", "seconds": seconds},
                    metadata={"action": "wait"},
                    summary=f"Waited {seconds:g} second(s). Observe the desktop before deciding the next step.",
                    evidence={"action": "wait", "seconds": seconds},
                )

            if action == "list_apps":
                session = self._observe._session_label()
                raw = call_driver("list_apps", {"session": session}, timeout=12)
                return ToolResult(
                    success=True,
                    name=self.name,
                    content={"action": action, "apps": raw[:5000]},
                    metadata={"action": action},
                    summary="Listed running applications.",
                    evidence={"action": action, "apps": raw[:1000]},
                )

            if action == "list_windows":
                session = self._observe._session_label()
                raw = call_driver(
                    "list_windows",
                    {"session": session, "on_screen_only": True},
                    timeout=12,
                )
                try:
                    payload = json.loads(raw)
                    windows = (
                        payload.get("windows", [])
                        if isinstance(payload, dict)
                        else payload
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    windows = []
                compact = [
                    {
                        key: row.get(key)
                        for key in (
                            "app_name",
                            "title",
                            "pid",
                            "window_id",
                            "x",
                            "y",
                            "width",
                            "height",
                        )
                    }
                    for row in windows
                    if isinstance(row, dict)
                    and row.get("is_on_screen", True)
                    and int(row.get("width") or 0) > 0
                    and int(row.get("height") or 0) > 0
                ][:12]
                return ToolResult(
                    success=True,
                    name=self.name,
                    content={"action": action, "windows": compact},
                    metadata={"action": action, "window_count": len(compact)},
                    summary=f"Found {len(compact)} visible windows. Pass a visible window title to the next capture to target it.",
                    evidence={"action": action, "window_count": len(compact)},
                )

            return ToolResult(
                success=False,
                name=self.name,
                content={
                    "error": {
                        "type": "unsupported_action",
                        "message": f"Unsupported computer_use action: {action!r}.",
                    }
                },
                summary=f"Unsupported computer-use action: {action!r}.",
            )
        except (CuaDriverError, ValueError, TypeError, KeyError) as exc:
            return ToolResult(
                success=False,
                name=self.name,
                content={
                    "action": action,
                    "error": {"type": "computer_use_failed", "message": str(exc)},
                },
                metadata={
                    "recovery_hint": "Stop on a driver timeout or refusal; do not fall back to repeated pixel guesses. Observe again and report the exact blocker."
                },
                summary=f"Computer Use action '{action}' failed: {exc}",
            )
