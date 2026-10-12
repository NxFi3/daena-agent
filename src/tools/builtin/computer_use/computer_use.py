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
    ensure_driver_running,
    pointer_state_path,
    _desktop_session_environment,
)
from src.tools.builtin.computer_use.observe import ComputerObserveTool
from src.tools.builtin.computer_use.inspect import ComputerInspectTool
from src.tools.builtin.computer_use.act import ComputerActTool


_DRIVER_BACKED_ACTIONS = frozenset({
    "capture", "launch", "navigate", "sequence", "click", "double_click",
    "drag", "right_click", "middle_click", "type", "key", "scroll",
    "press_key", "hotkey", "list_apps", "list_windows",
})


class ComputerUseTool(Tool):
    """One model-facing computer-use tool with a stable, concise action contract."""

    name = "computer_use"
    model_exposure = "search"
    action = "interact"
    description = (
        "General-purpose desktop control for browsers, editors, terminals, drawing apps, dialogs, and other GUI work. "
        "If a destination URL is known, use action='navigate', url='https://…'; otherwise start with action='capture', mode='som'. "
        "Controls are snapshot-scoped. For click/type, prefer target='exact visible label' when it uniquely identifies the intended control; otherwise use an element number from the latest capture, never guess an ID. Make only one Computer Use call per model turn; use action='sequence' for 2-6 predictable steps so GUI calls do not race each other. "
        "Use action='type', target='field label', text='…' to type into an accessible field. If the field is visible but missing from the controls list, use coordinate=[x,y] from the latest screenshot; the tool focuses that point before typing. For action='scroll', use coordinate=[x,y] inside the intended scrollable area when it is not inferable from current accessibility state; the driver requires an explicit scroll anchor. "
        "For continuous pointer gestures—drawing in Paint, dragging objects, selecting regions—use action='drag' with from_coordinate=[x,y] and to_coordinate=[x,y] from the latest screenshot. Each drag is a real press-move-release gesture; use separate strokes for separate drawing segments. "
        "Use action='sequence' for 2-6 predictable steps with label targets; it refreshes state between steps and stops on an ambiguous target or failure. Keep decisions requiring visual interpretation or user input (for example verification codes) as separate turns. "
        "After meaningful actions, inspect the returned fresh screenshot and controls. If screen_hash_changed is false, do not repeat the same click; re-capture/re-evaluate the current window. 'Dispatched' only means an input was sent, not that it worked; verify the visible result before continuing or claiming success. "
        "If the user replies with a value you requested (especially an OTP), first capture the current UI and resume the pending flow; do not navigate or launch another tab just because the reply is short. "
        "Never reuse an old element number after capture changes, never repeat an action after timeout/no-op, and never keep trying guessed coordinates. "
        "The tool aligns the target window before capture; if it cannot align screenshot and controls or the driver refuses/times out, report the concrete error rather than guessing. If a scroll lacks a usable anchor, it reports that explicitly so a fresh screenshot can supply one. "
        "Use mode='ax' for controls only and mode='vision' when visual pixels are needed or controls are inaccessible. If the app is not open, use action='launch', app='<application>' instead of clicking the taskbar. "
        "If a capture reports zero numbered controls, element=N cannot be used; use a fresh screenshot and a clearly visible coordinate target only when appropriate. "
        "For URLs, action='navigate' opens the destination in a browser and returns the selected window's fresh screenshot and controls; use it instead of clicking/typing into the address bar, which may append to existing text. Do not immediately call capture again unless the returned state is stale or ambiguous. If navigation fails, inspect the returned state and retry only when the evidence supports a recovery. Repeating the exact same URL without a preceding failed attempt is blocked, so inspect the current page and continue from evidence instead. "
        "The red ring marks the last agent click, not the live cursor. This is the only Computer Use tool exposed to the model."
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
                    "sequence",
                    "click",
                    "double_click",
                    "drag",
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
            "target": {
                "type": "string",
                "description": "Visible accessible control label for a click/type action, e.g. 'Continue' or 'ادامه'. Resolves only against a unique current control; ambiguous matches are refused.",
            },
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": 6,
                "description": "Model-planned sequential UI actions. Every target is re-resolved from current accessibility controls after the preceding action; the sequence stops at the first ambiguous/missing target or failed step.",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["click", "type", "key", "scroll", "drag", "wait"]},
                        "target": {"type": "string", "description": "Visible accessible label; required for click/type."},
                        "text": {"type": "string"},
                        "key": {"type": "string"},
                        "keys": {"type": ["string", "array"], "items": {"type": "string"}},
                        "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                        "coordinate": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "Optional point inside the target scroll area from the latest screenshot."},
                        "amount": {"type": "integer", "minimum": 1, "maximum": 20},
                        "seconds": {"type": "number", "minimum": 0, "maximum": 10},
                        "button": {"type": "string", "enum": ["left", "right", "middle"]},
                        "from_coordinate": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "to_coordinate": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "duration_ms": {"type": "integer", "minimum": 0, "maximum": 10000}
                    },
                    "required": ["action"],
                    "additionalProperties": False
                }
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
                "description": "[x,y] from the latest full-desktop screenshot: use for coordinate-only click/type targets or as an anchor inside the intended scroll area. Never estimate coordinates from a typical layout.",
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
            "duration_ms": {
                "type": "integer",
                "minimum": 0,
                "maximum": 10000,
                "description": "Duration of a drag gesture in milliseconds (default 500). Use drag for drawing, moving, selecting, or other continuous mouse gestures.",
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
    def _scroll_anchor_from_state(state: dict[str, Any]) -> list[float] | None:
        """Choose a point inside the largest current accessible content region."""
        elements = state.get("latest_accessibility_elements", {})
        if not isinstance(elements, dict):
            return None
        preferred = ("scroll pane", "scroll area", "document web", "web area", "web view", "canvas", "document")
        candidates: list[tuple[int, float, list[float]]] = []
        for element in elements.values():
            if not isinstance(element, dict):
                continue
            frame = element.get("frame")
            if not isinstance(frame, dict):
                continue
            try:
                x, y, width, height = (float(frame[key]) for key in ("x", "y", "w", "h"))
            except (KeyError, TypeError, ValueError):
                continue
            # Prefer a real container, but some browser/accessibility backends expose
            # only a small content link or text node. Keep those as a last-resort
            # anchor so ordinary page scrolling still works without guessed pixels.
            if x < 0 or y < 0 or width < 1 or height < 1:
                continue
            role = str(element.get("role") or "").casefold()
            in_content = bool(element.get("in_web_content"))
            if (width < 80 or height < 60) and not in_content:
                continue
            priority = next((len(preferred) - i for i, token in enumerate(preferred) if token in role), 0)
            if not in_content and priority == 0:
                continue
            if not in_content:
                priority = max(1, priority - 2)
            area = min(width * height, 100_000_000)
            candidates.append((priority, area, [x + width / 2, y + height / 2]))
        if candidates:
            # Prefer semantically scrollable/content roles, then the largest visible area.
            return max(candidates, key=lambda item: (item[0], item[1]))[2]
        return None

    @staticmethod
    def _save_state(path: Path, state: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            pass

    def _ensure_window_front(self, target: dict[str, Any]) -> bool:
        """Activate the exact target window when it is covered, then verify z-order."""
        try:
            pid = int(target.get("pid"))
            window_id = int(target.get("window_id"))
        except (TypeError, ValueError):
            raise CuaDriverError("Cannot align screenshot and controls: target window has no valid pid/window_id.")

        def find_target(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
            exact = next(
                (
                    row for row in rows
                    if int(row.get("pid") or 0) == pid
                    and int(row.get("window_id") or 0) == window_id
                    and row.get("is_on_screen", True)
                ),
                None,
            )
            if exact is not None:
                return exact

            # Browser launch/tab changes can invalidate a native window id between
            # inventory and activation. Re-resolve only within the same process, and
            # prefer a title match; if multiple unrelated windows remain ambiguous,
            # fail rather than focus an arbitrary page.
            same_process = [
                row for row in rows
                if int(row.get("pid") or 0) == pid
                and row.get("is_on_screen", True)
                and int(row.get("width") or 0) >= 600
                and int(row.get("height") or 0) >= 400
            ]
            if not same_process:
                return None

            wanted_title = " ".join(str(target.get("title") or target.get("app") or "").casefold().split())
            if wanted_title:
                title_matches = [
                    row for row in same_process
                    if wanted_title in " ".join(
                        f"{row.get('title', '')} {row.get('app_name', '')}".casefold().split()
                    )
                    or " ".join(
                        f"{row.get('title', '')} {row.get('app_name', '')}".casefold().split()
                    ) in wanted_title
                ]
                if len(title_matches) == 1:
                    return title_matches[0]
                if len(title_matches) > 1:
                    return max(
                        title_matches,
                        key=lambda row: (
                            int(row.get("z_index") or 0),
                            int(row.get("width") or 0) * int(row.get("height") or 0),
                        ),
                    )
            if len(same_process) == 1:
                return same_process[0]
            return None

        rows = self._inspect._windows()
        current = find_target(rows)
        if current is None:
            raise CuaDriverError(
                f"Cannot align screenshot and controls: target window pid={pid}, window_id={window_id} is no longer visible and could not be re-resolved safely. Capture again."
            )
        target.update(current)
        window_id = int(current.get("window_id") or window_id)

        def highest_foreign_z(rows: list[dict[str, Any]]) -> int:
            # Some desktop backends report auxiliary/tab entries for one browser
            # process as multiple rows. The accessibility action is PID-scoped, so
            # those same-PID rows are not independently focusable occluders here.
            # Compare against other processes only; otherwise a stale/auxiliary row
            # can make a visible Firefox window fail every capture.
            return max(
                (
                    int(row.get("z_index") or 0)
                    for row in rows
                    if int(row.get("pid") or 0) != pid
                    and row.get("is_on_screen", True)
                    and int(row.get("width") or 0) >= 250
                    and int(row.get("height") or 0) >= 150
                ),
                default=0,
            )

        highest_z = highest_foreign_z(rows)
        if int(current.get("z_index") or 0) < highest_z:
            call_driver(
                "bring_to_front",
                {
                    "pid": pid,
                    "window_id": window_id,
                    "session": self._observe._session_label(),
                },
                timeout=12,
            )
            time.sleep(0.15)
            rows = self._inspect._windows()
            current = find_target(rows)
            if current is None:
                raise CuaDriverError(
                    f"Target window pid={pid}, window_id={window_id} disappeared while being activated."
                )
            target.update(current)
            window_id = int(current.get("window_id") or window_id)
            highest_z = highest_foreign_z(rows)
            if int(current.get("z_index") or 0) < highest_z:
                raise CuaDriverError(
                    f"Could not bring target window '{current.get('title') or ''}' to the front of other applications. No screenshot or UI action was treated as verified."
                )
        return True

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

        # Select the target first. On desktops with several browser windows, an
        # accessibility tree can describe a covered window while the screenshot shows
        # another app. Activate the selected native window before taking either view.
        observed: ToolResult | None = None
        inspected: ToolResult | None = None
        query_fallback_used = False
        selection: ToolResult | None = None
        selection_target_ready = False
        if mode != "vision" or hint or pid is not None or window_id is not None:
            selection = self._inspect.execute(
                query="",
                window_title=hint,
                pid=pid,
                window_id=window_id,
                max_results=1,
            )
            if not selection.success or not isinstance(selection.content, dict):
                # A failed target selection is a hard stop. Capturing a generic desktop
                # and then inspecting another window would recreate the old mismatch.
                return self._copy_result(selection, "capture")
            selected_window = selection.content.get("window")
            if not isinstance(selected_window, dict) or selected_window.get("pid") is None or selected_window.get("window_id") is None:
                return ToolResult(
                    success=False,
                    name=self.name,
                    content={"action": "capture", "error": {"type": "window_target_unverified", "message": "Could not resolve a concrete native window; screenshot and controls were not combined."}},
                    summary="Capture stopped because the target window could not be identified safely.",
                )
            target = selected_window
            pid = selected_window.get("pid")
            window_id = selected_window.get("window_id")
            # If this fails, fail the capture closed instead of mixing a hidden app's
            # accessibility controls with another window's pixels.
            self._ensure_window_front(selected_window)
            # _ensure_window_front may safely re-resolve a browser window id that
            # changed while the browser was starting; use the updated target for both
            # the screenshot and the follow-up accessibility read.
            pid = selected_window.get("pid")
            window_id = selected_window.get("window_id")
            selection_target_ready = True

        if mode != "ax":
            observed = self._observe.execute(include_windows=include_windows)
            if not observed.success:
                return self._copy_result(observed, "capture")

        if mode != "vision":
            # A failed preliminary query already contains the concrete error; reuse it
            # and go directly to the existing unfiltered recovery instead of issuing
            # the same failing inspection twice.
            inspect_title = "" if pid is not None and window_id is not None else hint
            if selection is not None and not selection.success:
                inspected = selection
            else:
                # A fresh read after activation makes the control IDs and screenshot refer
                # to the same window/tab state. Once selected, pin this exact native window.
                inspected = self._inspect.execute(
                    query=str(query or ""),
                    window_title=inspect_title,
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
                    window_title=inspect_title,
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
        if action == "drag" and kwargs.get("element") is not None:
            raise ValueError("drag uses from_coordinate and to_coordinate from the latest capture; element IDs are not drag paths.")
        click_signature: str | None = None
        typing_fallback_token: str | None = None
        typing_fallback_text: str | None = None
        element_record: dict[str, Any] = {}
        element_id = args.pop("element", None)
        state_path, state = self._current_state()
        if action == "scroll" and args.get("coordinate") is None and not ("x" in args and "y" in args):
            anchor = self._scroll_anchor_from_state(state)
            if anchor is None:
                return ToolResult(
                    success=False,
                    name=self.name,
                    content={"success": False, "error": {
                        "type": "missing_scroll_anchor",
                        "message": "The current accessibility snapshot has no reliable scrollable region. Capture the screen and retry with coordinate=[x,y] inside the intended scroll area.",
                    }},
                    metadata={"recovery_hint": "Capture a fresh screenshot and pass a coordinate inside the intended scroll area; do not retry the same scroll without an anchor."},
                    summary="Scroll was not sent because no reliable target area could be resolved.",
                )
            args["coordinate"] = anchor
        if element_id is not None:
            token, record = self._element_record(element_id)
            element_record = record
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

        for point_name, axis_names in (
            ("from_coordinate", ("from_x", "from_y")),
            ("to_coordinate", ("to_x", "to_y")),
        ):
            point = args.pop(point_name, None)
            if point is not None:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError(f"{point_name} must be [x,y] from the latest screenshot.")
                try:
                    args[axis_names[0]], args[axis_names[1]] = float(point[0]), float(point[1])
                except (TypeError, ValueError):
                    raise ValueError(f"{point_name} must contain numeric x/y values.")

        allowed = {
            "element_token",
            "x",
            "y",
            "from_x",
            "from_y",
            "to_x",
            "to_y",
            "duration_ms",
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
            # When AT-SPI exposes an input only as "activate", its AX click route may
            # fail on Wayland (or focus a different native surface). Use the exact
            # visible field bounds from the latest capture: foreground pixel click first,
            # then send text to the now-focused app. Never type if that click failed.
            frame = element_record.get("frame")
            try:
                fx, fy, fw, fh = (float(frame[key]) for key in ("x", "y", "w", "h"))
                if fw <= 0 or fh <= 0 or fx < 0 or fy < 0:
                    raise ValueError("field bounds are not visible positive desktop coordinates")
                x, y = round(fx + fw / 2), round(fy + fh / 2)
            except (KeyError, TypeError, ValueError):
                return ToolResult(
                    success=False,
                    name="computer_use",
                    content={
                        "action": "type",
                        "target_mode": "unavailable",
                        "failed_stage": "resolve_visible_field_bounds",
                        "field_role": str(record.get("role") or ""),
                        "field_label": str(record.get("label") or ""),
                        "error": "This field exposes no reliable visible bounds, so text was not sent.",
                    },
                    metadata={"recovery_hint": "Capture the current screen, select a visible field with usable bounds, and do not type into an unverified target."},
                    summary=f"Cannot safely type into {record.get('label') or 'the selected field'} because its visible bounds are unavailable.",
                    evidence={"field": {"role": record.get("role"), "label": record.get("label"), "frame": frame}},
                )

            delivery_mode = str(args.get("delivery_mode") or "foreground")
            focus_result = self._act.execute(
                action="click",
                x=x,
                y=y,
                delivery_mode="foreground",
            )
            if not focus_result.success:
                return ToolResult(
                    success=False,
                    name="computer_use",
                    content={
                        "action": "type",
                        "target_mode": "visible_field_coordinates",
                        "failed_stage": "focus_visible_field",
                        "field_label": str(record.get("label") or ""),
                        "field_center": [x, y],
                        "error": focus_result.content.get("error", focus_result.summary),
                    },
                    metadata={"recovery_hint": "The coordinate click did not focus the selected field. Re-capture and inspect; do not send text or retry a timed-out click."},
                    summary=f"Could not focus the visible field '{record.get('label') or ''}': {focus_result.summary}",
                    evidence={"focus_result": focus_result.evidence, "field": {"label": record.get("label"), "frame": frame}},
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
                    "target_mode": "visible_field_coordinates_then_keyboard",
                    "field_role": str(record.get("role") or ""),
                    "field_label": str(record.get("label") or ""),
                    "field_center": [x, y],
                    "focus_succeeded": focus_result.success,
                    "driver_output": type_result.content.get("driver_output", ""),
                    **({"error": type_result.content.get("error")} if not type_result.success else {}),
                },
                metadata={
                    "action": "type",
                    "target_mode": "visible_field_coordinates_then_keyboard",
                    "field_role": str(record.get("role") or ""),
                    "delivery_mode": type_result.metadata.get("delivery_mode", delivery_mode),
                    "delivery_mode_escalated": type_result.metadata.get("delivery_mode_escalated", False),
                },
                summary=(
                    f"Focused the visible field '{record.get('label') or ''}' using its current screenshot bounds and sent text. Inspect the fresh screen and verify the value before continuing."
                    if type_result.success
                    else f"The field was focused, but text entry failed for '{record.get('label') or ''}': {type_result.summary}"
                ),
                evidence={
                    "target_mode": "visible_field_coordinates_then_keyboard",
                    "field": {"role": record.get("role"), "label": record.get("label"), "frame": frame, "center": [x, y]},
                    "focus_result": focus_result.evidence,
                    "type_result": type_result.evidence,
                },
                effects=type_result.effects,
            )
        if (
            action == "type_text"
            and not args.get("element_token")
            and isinstance(args.get("x"), (int, float))
            and isinstance(args.get("y"), (int, float))
        ):
            # Coordinate-targeted typing is the fallback for apps whose editable
            # controls are absent from AT-SPI (custom canvas/toolkit widgets). CUA's
            # type_text(x,y) route can report "typed" without focusing the field on
            # Wayland. Use a verified point from the latest screenshot to focus first,
            # then type to the focused application. Never send text if focus fails.
            text_value = args.get("text")
            if not isinstance(text_value, str):
                raise ValueError("Coordinate-targeted type requires text.")
            x, y = float(args["x"]), float(args["y"])
            focus_result = self._act.execute(
                action="click",
                x=x,
                y=y,
                delivery_mode="foreground",
            )
            if not focus_result.success:
                return ToolResult(
                    success=False,
                    name="computer_use",
                    content={
                        "action": "type",
                        "target_mode": "visible_coordinate_then_keyboard",
                        "failed_stage": "focus_coordinate",
                        "coordinate": [x, y],
                        "error": focus_result.content.get("error", focus_result.summary),
                    },
                    metadata={"recovery_hint": "The coordinate click failed. Refresh the screenshot, identify the field again, and do not send text or retry the timed-out click."},
                    summary=f"Could not focus the selected coordinate ({round(x)}, {round(y)}): {focus_result.summary}",
                    evidence={"focus_result": focus_result.evidence, "coordinate": [x, y]},
                )
            delivery_mode = str(args.get("delivery_mode") or "background")
            type_result = self._act.execute(
                action="type_text",
                text=text_value,
                delivery_mode=delivery_mode,
            )
            return ToolResult(
                success=type_result.success,
                name="computer_use",
                content={
                    "action": "type",
                    "target_mode": "visible_coordinate_then_keyboard",
                    "coordinate": [round(x), round(y)],
                    "focus_succeeded": focus_result.success,
                    "driver_output": type_result.content.get("driver_output", ""),
                    **({"error": type_result.content.get("error")} if not type_result.success else {}),
                },
                metadata={
                    "action": "type",
                    "target_mode": "visible_coordinate_then_keyboard",
                    "delivery_mode": type_result.metadata.get("delivery_mode", delivery_mode),
                    "delivery_mode_escalated": type_result.metadata.get("delivery_mode_escalated", False),
                },
                summary=(
                    f"Focused the visible point ({round(x)}, {round(y)}) and sent text. Inspect the fresh screen to confirm it appeared in the intended field before continuing."
                    if type_result.success
                    else f"The point was focused, but text entry failed: {type_result.summary}"
                ),
                evidence={
                    "target_mode": "visible_coordinate_then_keyboard",
                    "coordinate": [round(x), round(y)],
                    "focus_result": focus_result.evidence,
                    "type_result": type_result.evidence,
                },
                effects=type_result.effects,
            )

        result = self._act.execute(action=action, **args)

        # Accessibility handles can expire inside the desktop driver between the
        # inspection and dispatch. Recover only from an explicit stale-token refusal
        # that guarantees no input was delivered; never retry an ambiguous timeout.
        if (
            not result.success
            and element_id is not None
            and action in click_like
            and "stale_element_token" in str(
                (result.content.get("error") or {}).get("message", "")
                if isinstance(result.content, dict) else ""
            ).casefold()
            and '"effect":"none"' in str(
                (result.content.get("error") or {}).get("message", "")
                if isinstance(result.content, dict) else ""
            ).replace(" ", "").casefold()
        ):
            old_label = str(element_record.get("label") or "").strip().casefold()
            old_role = str(element_record.get("role") or "").strip().casefold()
            _state_path, latest_state = self._current_state()
            latest_target = latest_state.get("latest_computer_use_target", {})
            latest_target = latest_target if isinstance(latest_target, dict) else {}
            refreshed = self._capture(
                mode="ax",
                pid=latest_target.get("pid"),
                window_id=latest_target.get("window_id"),
                max_results=30,
                include_windows=False,
            )
            fresh_elements = refreshed.content.get("elements", []) if refreshed.success else []
            matches = [
                item for item in fresh_elements
                if isinstance(item, dict)
                and str(item.get("label") or "").strip().casefold() == old_label
                and str(item.get("role") or "").strip().casefold() == old_role
                and item.get("id") is not None
            ] if old_label and old_role and isinstance(fresh_elements, list) else []
            if len(matches) == 1:
                fresh_id = matches[0]["id"]
                try:
                    fresh_token, _fresh_record = self._element_record(fresh_id)
                    retry_args = {
                        "element_token": fresh_token,
                        "delivery_mode": str(kwargs.get("delivery_mode") or "background"),
                    }
                    result = self._act.execute(action=action, **retry_args)
                    if result.success:
                        result.content.setdefault("stale_handle_recovered", True)
                        result.metadata["stale_handle_recovered"] = True
                        result.summary = (
                            "Recovered from a stale accessibility handle by re-resolving the same "
                            "label and role and retrying once. " + result.summary
                        )
                except (ValueError, TypeError, KeyError):
                    pass

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

    @staticmethod
    def _site_identity_tokens(hostname: str) -> set[str]:
        """Derive likely site-brand tokens from a hostname without domain-specific rules."""
        labels = [part.casefold() for part in str(hostname or "").split(".") if part]
        ignored = {
            "www", "m", "amp", "com", "org", "net", "edu", "gov", "mil",
            "int", "io", "dev", "app", "info", "biz", "co", "uk", "us",
            "ir", "az", "ru", "de", "fr", "jp", "au", "ca", "me", "tv",
        }
        return {label for label in labels if label not in ignored and len(label) > 1}

    @staticmethod
    def _observed_browser_url(content: dict[str, Any]) -> str | None:
        """Read the URL only from the browser's address/location control."""
        elements = content.get("elements")
        if not isinstance(elements, list):
            return None
        for item in elements:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").casefold().strip()
            label = " ".join(str(item.get("label") or "").casefold().split())
            if role not in {"combo box", "address bar", "url bar"}:
                continue
            if not any(marker in label for marker in (
                "address", "location", "url", "enter address", "search with google",
                "نوار آدرس", "آدرس وب", "نشانی",
            )):
                continue
            value = str(item.get("value") or "").strip()
            if not value or len(value) > 1000:
                continue
            candidate = value if re.match(r"^https?://", value, re.IGNORECASE) else "https://" + value
            if not re.match(r"^https?://[a-z0-9.-]+(?::[0-9]{1,5})?(?:[/#?][^\s]*)?$", candidate, re.IGNORECASE):
                continue
            parsed = urlparse(candidate)
            if parsed.hostname and "." in parsed.hostname:
                # Preserve the address bar's literal value; don't infer its scheme.
                return value
        return None

    @staticmethod
    def _navigation_address_status(requested_url: str, observed_url: str | None) -> tuple[str, bool]:
        """Compare observed browser address with strict origin/path semantics.

        A capture cannot prove a page is functional. When a browser hides the
        scheme, host/path can match, but the exact secure origin is not verified.
        """
        if not observed_url:
            return "url_not_observable", False

        requested = urlparse(requested_url)
        observed_has_scheme = bool(re.match(r"^https?://", observed_url, re.IGNORECASE))
        observed = urlparse(
            observed_url if observed_has_scheme else "https://" + observed_url
        )
        requested_host = (requested.hostname or "").casefold()
        observed_host = (observed.hostname or "").casefold()
        if not requested_host or not observed_host or observed_host != requested_host:
            return "redirected_to_different_host", False

        if observed_has_scheme and requested.scheme.casefold() != observed.scheme.casefold():
            return "scheme_changed", False

        def effective_port(parsed: Any) -> int | None:
            try:
                explicit = parsed.port
            except ValueError:
                return None
            if explicit is not None:
                return explicit
            return 443 if parsed.scheme.casefold() == "https" else 80 if parsed.scheme.casefold() == "http" else None

        if observed_has_scheme and effective_port(requested) != effective_port(observed):
            return "origin_changed", False

        requested_path = (requested.path or "/").rstrip("/") or "/"
        observed_path = (observed.path or "/").rstrip("/") or "/"
        same_path_and_query = (
            requested_path == observed_path
            and requested.query == observed.query
            and (not requested.fragment or requested.fragment == observed.fragment)
        )
        if not same_path_and_query:
            return "redirected_within_same_host", False

        if not observed_has_scheme:
            return "address_host_path_matches_scheme_unobservable", False
        return "address_matches_request", True

    def _existing_site_window(
        self, windows: list[dict[str, Any]], hostname: str
    ) -> dict[str, Any] | None:
        tokens = self._site_identity_tokens(hostname)
        if not tokens:
            return None
        candidates = [
            row for row in windows
            if self._inspect._is_browser(row)
            and row.get("is_on_screen", True)
            and int(row.get("width") or 0) >= 600
            and int(row.get("height") or 0) >= 400
            and any(
                token in str(row.get("title") or "").casefold()
                or token in str(row.get("app_name") or "").casefold()
                for token in tokens
            )
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda row: (
                int(row.get("z_index") or 0),
                int(row.get("width") or 0) * int(row.get("height") or 0),
            ),
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

        before_windows: list[dict[str, Any]] = []
        try:
            browser_key, executable = self._navigation_browser(app)
            try:
                before_windows = self._inspect._windows()
            except (CuaDriverError, ValueError, TypeError):
                before_windows = []

            # Continuation safety: models sometimes repeat the site's root URL when a
            # user replies with an OTP or other requested input. Do not open a duplicate
            # tab and abandon a live form when a matching site is already visible and
            # the requested URL is only the origin/root. Reuse and inspect that window.
            is_root_url = parsed.path in {"", "/"} and not parsed.query and not parsed.fragment
            existing_site = (
                self._existing_site_window(before_windows, parsed.hostname or "")
                if is_root_url else None
            )
            if existing_site is not None:
                try:
                    self._ensure_window_front(existing_site)
                    captured = self._capture(
                        mode="som",
                        pid=existing_site.get("pid"),
                        window_id=existing_site.get("window_id"),
                        max_results=24,
                        include_windows=True,
                    )
                except (CuaDriverError, ValueError, TypeError) as exc:
                    return ToolResult(
                        success=False,
                        name=self.name,
                        content={"action": "navigate", "url": raw_url,
                                 "error": {"type": "existing_site_window_unavailable", "message": str(exc)}},
                        summary=f"An existing browser window appears to match {parsed.hostname}, but it could not be safely inspected: {exc}",
                    )
                content = dict(captured.content) if isinstance(captured.content, dict) else {}
                title = str((content.get("window") or {}).get("title") or existing_site.get("title") or "")
                current_address = self._observed_browser_url(content)
                current_status, current_address_matches = self._navigation_address_status(raw_url, current_address)
                content.update({
                    "action": "navigate",
                    "url_requested": raw_url,
                    "observed_url": current_address,
                    "navigation_status": "existing_site_window_reused_without_navigation",
                    "destination_verified": current_address_matches,
                    "browser": browser_key,
                    "navigation_reused_existing_site_window": True,
                    "navigation_reused_existing_browser_window": True,
                    "destination_in_title": (parsed.hostname or "").casefold() in title.casefold(),
                    "verification": (
                        "No navigation was performed and no new tab was opened: an existing visible window appeared to match this site's hostname and the request was for the root URL. "
                        f"Current address: {current_address or 'not observable'} (address comparison: {current_status}). This protects an in-progress form; do not claim the root URL was opened unless the current address confirms it. Inspect the page and resume from its actual state."
                    ),
                })
                metadata = dict(captured.metadata)
                metadata.update({
                    "action": "navigate",
                    "url_requested": raw_url,
                    "observed_url": current_address,
                    "navigation_status": "existing_site_window_reused_without_navigation",
                    "destination_verified": current_address_matches,
                    "browser": browser_key,
                    "navigation_reused_existing_site_window": True,
                    "navigation_reused_existing_browser_window": True,
                })
                return ToolResult(
                    success=captured.success,
                    name=self.name,
                    content=content,
                    metadata=metadata,
                    summary=(
                        f"Reused the existing {browser_key} window matching {parsed.hostname}; no duplicate tab was opened, so any in-progress form remains intact. "
                        "Inspect the returned screenshot and current controls, then resume the task from its present step.\n" + str(captured.summary)
                    ),
                    evidence={
                        "action": "navigate",
                        "url_requested": raw_url,
                        "navigation_reused_existing_site_window": True,
                        "window": content.get("window"),
                        "screenshot": metadata.get("image_ref"),
                    },
                )

            # Reuse the visible browser window whenever possible. Spawning a new
            # native window for every navigate call caused tab/window accumulation when
            # the model retried after an uncertain result.
            browser_candidates = [
                row for row in before_windows
                if self._inspect._is_browser(row)
                and row.get("is_on_screen", True)
                and int(row.get("width") or 0) >= 600
                and int(row.get("height") or 0) >= 400
            ]
            if browser_key == "firefox":
                matching_browser = [
                    row for row in browser_candidates
                    if "firefox" in f"{row.get('app_name', '')} {row.get('title', '')}".casefold()
                ]
            elif browser_key in {"chrome", "chromium"}:
                matching_browser = [
                    row for row in browser_candidates
                    if any(name in f"{row.get('app_name', '')} {row.get('title', '')}".casefold()
                           for name in ("chrome", "chromium", "brave", "edge"))
                ]
            else:
                matching_browser = browser_candidates

            reused_existing_browser_window = bool(matching_browser)
            selected_browser: dict[str, Any] | None = None
            if reused_existing_browser_window:
                selected_browser = max(
                    matching_browser,
                    key=lambda row: (
                        int(row.get("z_index") or 0),
                        int(row.get("width") or 0) * int(row.get("height") or 0),
                    ),
                )
                self._ensure_window_front(selected_browser)
                pid = int(selected_browser.get("pid") or 0)
                window_id = int(selected_browser.get("window_id") or 0)
                if pid <= 0 or window_id <= 0:
                    raise CuaDriverError("The existing browser window has no usable pid/window_id.")
                session = self._observe._session_label()
                target = {"kind": "window", "pid": pid, "window_id": window_id}
                # Navigate in-place through the browser's location bar. The target
                # is explicit and each call is checked; a failure does not silently
                # fall back to opening yet another window.
                call_driver(
                    "hotkey",
                    {"session": session, "target": target, "keys": ["ctrl", "l"], "delivery_mode": "foreground"},
                    timeout=12,
                )
                call_driver(
                    "type_text",
                    {"session": session, "target": target, "text": raw_url, "delivery_mode": "foreground"},
                    timeout=12,
                )
                call_driver(
                    "press_key",
                    {"session": session, "target": target, "key": "enter", "delivery_mode": "foreground"},
                    timeout=12,
                )
            else:
                env = _desktop_session_environment()
                subprocess.Popen(
                    [executable, "--new-window", raw_url],
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
            requested_wait = float(seconds)
        except (TypeError, ValueError):
            requested_wait = 1.0
        # Browser startup is asynchronous (and Firefox can take several seconds to
        # create a native window on a cold launch). Poll the real window inventory.
        # Reusing an existing window needs a shorter settle time and must not lose
        # the original PID while native window IDs change during navigation.
        reused_pid = (
            int(selected_browser.get("pid") or 0)
            if reused_existing_browser_window and isinstance(selected_browser, dict)
            else 0
        )
        # Keep the cold-start fallback bounded: the loop exits early as soon as
        # the target window is observable, so this cap only affects genuinely slow
        # launches. Long unconditional waits added noticeable latency to each retry.
        wait_floor = 1.5 if reused_pid else 4.0
        wait_extra = 1.0 if reused_pid else 2.0
        wait_budget = max(wait_floor, min(6.0, requested_wait + wait_extra))
        poll_count = max(1, int(wait_budget / 0.25) + 1)
        # Always allow the navigation at least ~0.75s to settle. A changed native
        # window ID/title can appear before the requested page has finished loading.
        minimum_settle_attempt = max(3, int(max(0.75, min(requested_wait, 3.0)) / 0.25))

        before_titles = {
            (int(row.get("pid") or 0), int(row.get("window_id") or 0)): str(row.get("title") or "")
            for row in before_windows if isinstance(row, dict)
        }
        host = str(parsed.hostname or "").casefold()
        after_windows: list[dict[str, Any]] = []
        browser_rows: list[dict[str, Any]] = []
        changed_rows: list[dict[str, Any]] = []
        host_matches: list[dict[str, Any]] = []
        for attempt in range(poll_count):
            try:
                after_windows = self._inspect._windows()
            except (CuaDriverError, ValueError, TypeError):
                after_windows = []
            browser_rows = [
                row for row in after_windows
                if self._inspect._is_browser(row)
                and row.get("is_on_screen", True)
                and int(row.get("width") or 0) >= 600
                and int(row.get("height") or 0) >= 400
            ]
            changed_rows = [
                row for row in browser_rows
                if before_titles.get(
                    (int(row.get("pid") or 0), int(row.get("window_id") or 0))
                ) != str(row.get("title") or "")
            ]
            host_matches = [
                row for row in browser_rows
                if host and (
                    host in str(row.get("title") or "").casefold()
                    or host in str(row.get("app_name") or "").casefold()
                )
            ]
            # A changed/new browser window means launch/navigation produced
            # observable UI state. When reusing a process, prefer that exact browser.
            same_process_rows = [
                row for row in browser_rows
                if reused_pid and int(row.get("pid") or 0) == reused_pid
            ]
            has_target_window = bool(same_process_rows) if reused_pid else bool(changed_rows)
            if has_target_window and attempt >= minimum_settle_attempt:
                break
            if attempt + 1 < poll_count:
                time.sleep(0.25)
        if reused_pid:
            candidates = [
                row for row in browser_rows
                if int(row.get("pid") or 0) == reused_pid
            ]
        else:
            candidates = host_matches or changed_rows or browser_rows
        if candidates:
            selected_browser = max(
                candidates,
                key=lambda row: (
                    int((int(row.get("pid") or 0), int(row.get("window_id") or 0)) not in before_titles),
                    int(str(row.get("title") or "") != before_titles.get(
                        (int(row.get("pid") or 0), int(row.get("window_id") or 0)), "")),
                    int(row.get("z_index") or 0),
                    int(row.get("width") or 0) * int(row.get("height") or 0),
                ),
            )
        elif reused_pid:
            return ToolResult(
                success=False,
                name=self.name,
                content={
                    "action": "navigate",
                    "url_requested": raw_url,
                    "error": {
                        "type": "browser_window_lost_after_navigation",
                        "message": f"The existing browser process {reused_pid} no longer has a visible, verifiable window. No additional browser window was spawned.",
                    },
                },
                summary="Navigation was requested in the existing browser, but its window could not be re-identified safely. Re-capture the desktop before any further action; no duplicate window was opened.",
            )
        else:
            selected_browser = None

        if selected_browser is not None:
            try:
                self._ensure_window_front(selected_browser)
                captured = self._capture(
                    mode="som",
                    pid=selected_browser.get("pid"),
                    window_id=selected_browser.get("window_id"),
                    max_results=24,
                    include_windows=True,
                )
            except (CuaDriverError, ValueError, TypeError) as exc:
                return ToolResult(
                    success=False,
                    name=self.name,
                    content={"action": "navigate", "url": raw_url, "error": {"type": "navigation_target_unavailable", "message": str(exc)}},
                    summary=f"The URL was opened, but Daena could not safely activate and inspect its browser window: {exc}",
                )
        else:
            captured = self._capture(
                mode="som", app=browser_key, max_results=24, include_windows=True
            )
        content = dict(captured.content)
        title = str((content.get("window") or {}).get("title") or "")
        observed_url = self._observed_browser_url(content)
        navigation_status, destination_verified = self._navigation_address_status(raw_url, observed_url)
        title_confirms_host = parsed.hostname.casefold() in title.casefold()
        status_explanations = {
            "address_matches_request": "The visible browser address matches the requested HTTPS origin, path, and query. Page functionality still needs visual/content verification.",
            "address_host_path_matches_scheme_unobservable": "The host and path match, but the browser hides the URL scheme. Do not assume HTTPS/security was verified; inspect browser security state and page content before any sign-in.",
            "redirected_within_same_host": "The browser address redirected to a different path or query on the same host. Do not assume the requested page loaded; inspect this page and its controls.",
            "redirected_to_different_host": "The browser address is on a different host than requested. Treat this as a redirect/security/error state, inspect it, and do not continue a login flow automatically.",
            "scheme_changed": "The browser address uses a different scheme than requested (for example HTTPS to HTTP). Treat this as a security-relevant redirect and do not enter credentials.",
            "origin_changed": "The browser address uses a different port/origin than requested. Treat it as unverified and inspect before continuing.",
            "url_not_observable": "The current browser address could not be read from accessible controls. The request is unverified; inspect the screenshot/current controls before claiming success or navigating again.",
        }
        verification = (
            f"Requested URL: {raw_url}. Observed address: {observed_url or 'not available'}. "
            + status_explanations[navigation_status]
            + " A successful capture proves only that the browser state was captured, not that the intended page or user task succeeded."
        )
        content.update({
            "action": "navigate",
            "url_requested": raw_url,
            "observed_url": observed_url,
            "navigation_status": navigation_status,
            "destination_verified": destination_verified,
            "navigation_reused_existing_browser_window": reused_existing_browser_window,
            "browser": browser_key,
            "verification": verification,
            "destination_in_title": title_confirms_host,
        })
        metadata = dict(captured.metadata)
        metadata.update({
            "action": "navigate",
            "url_requested": raw_url,
            "observed_url": observed_url,
            "navigation_status": navigation_status,
            "destination_verified": destination_verified,
            "browser": browser_key,
            "navigation_reused_existing_browser_window": reused_existing_browser_window,
        })
        verb = "Reused the existing browser window" if reused_existing_browser_window else "Opened a browser window"
        return ToolResult(
            success=captured.success,
            name=self.name,
            content=content,
            metadata=metadata,
            summary=(
                f"{verb} in {browser_key} and captured the current page. Navigation verification: {navigation_status}; "
                f"observed address: {observed_url or 'unavailable'}. "
                "Inspect the page before claiming the requested destination loaded.\n" + str(captured.summary)
                if captured.success
                else f"Browser action was requested, but the follow-up capture failed: {captured.summary}"
            ),
            evidence={
                "action": "navigate",
                "url_requested": raw_url,
                "observed_url": observed_url,
                "navigation_status": navigation_status,
                "destination_verified": destination_verified,
                "navigation_reused_existing_browser_window": reused_existing_browser_window,
                "browser": browser_key,
                "destination_in_title": title_confirms_host,
                "screenshot": metadata.get("image_ref"),
                "window": content.get("window"),
            },
        )

    @staticmethod
    def _normalize_target_label(value: Any) -> str:
        """Normalize accessible labels conservatively without translating them."""
        text = str(value or "").casefold().replace("\u200c", "").replace("\ufeff", "")
        return " ".join(text.split())

    def _target_candidates(self, target: str) -> tuple[list[int], list[int]]:
        """Return exact and unique-substring candidate IDs from the latest snapshot."""
        wanted = self._normalize_target_label(target)
        _path, state = self._current_state()
        ids = state.get("latest_element_ids", {})
        records = state.get("latest_accessibility_elements", {})
        if not wanted or not isinstance(ids, dict) or not isinstance(records, dict):
            return [], []
        exact: list[int] = []
        partial: list[int] = []
        for raw_id, token in ids.items():
            try:
                number = int(raw_id)
            except (TypeError, ValueError):
                continue
            record = records.get(str(token))
            if not isinstance(record, dict) or record.get("enabled") is False:
                continue
            labels = [record.get("label", ""), record.get("value", "")]
            normalized = [self._normalize_target_label(item) for item in labels]
            if wanted in normalized:
                exact.append(number)
            elif any(wanted in label for label in normalized if label):
                partial.append(number)
        return exact, partial

    def _resolve_target_id(self, target: str, *, refresh_if_missing: bool = True) -> int:
        """Resolve a visible label to one current control; refuse ambiguity."""
        exact, partial = self._target_candidates(target)
        matches = exact if exact else partial
        if not matches and refresh_if_missing:
            _state_path, state = self._current_state()
            latest_target = state.get("latest_computer_use_target", {})
            latest_target = latest_target if isinstance(latest_target, dict) else {}
            refreshed = self._capture(
                mode="ax",
                pid=latest_target.get("pid"),
                window_id=latest_target.get("window_id"),
                max_results=30,
                include_windows=False,
            )
            if refreshed.success:
                exact, partial = self._target_candidates(target)
                matches = exact if exact else partial
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Target label {target!r} matches multiple current controls {matches}; "
                "use a more specific visible label. No action was dispatched."
            )
        raise ValueError(
            f"No current accessible control uniquely matches target label {target!r}. "
            "Read the latest controls and use an exact visible label or a separate visual action."
        )

    def _execute_sequence(
        self,
        steps: Any,
        *,
        final_mode: str = "som",
        delivery_mode: str = "background",
        max_results: int = 24,
        include_windows: bool = True,
    ) -> ToolResult:
        """Execute a short model-planned UI sequence with fresh semantic resolution at every step."""
        if not isinstance(steps, list) or not 1 <= len(steps) <= 6:
            raise ValueError("sequence requires between 1 and 6 steps.")

        normalized_steps: list[dict[str, Any]] = []
        allowed_actions = {"click", "type", "key", "scroll", "drag", "wait"}
        allowed_fields = {"action", "target", "text", "key", "keys", "direction", "coordinate", "amount", "seconds", "button", "from_coordinate", "to_coordinate", "duration_ms"}
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict) or not set(step).issubset(allowed_fields):
                raise ValueError(f"sequence step {index} must be an object with only documented step fields.")
            current = dict(step)
            current_action = str(current.get("action", "")).strip().casefold()
            if current_action not in allowed_actions:
                raise ValueError(f"sequence step {index} has unsupported action {current_action!r}.")
            target = str(current.get("target") or "").strip()
            if current_action in {"click", "type"} and not target:
                raise ValueError(f"sequence step {index} ({current_action}) requires a visible-label target.")
            if current_action not in {"click", "type"} and target:
                raise ValueError(f"sequence step {index} ({current_action}) cannot use target; target is supported only for click/type.")
            if current_action == "type" and not isinstance(current.get("text"), str):
                raise ValueError(f"sequence step {index} (type) requires text.")
            if current_action == "key" and not (current.get("key") or current.get("keys")):
                raise ValueError(f"sequence step {index} (key) requires key or keys.")
            if current_action == "scroll" and str(current.get("direction", "")).casefold() not in {"up", "down", "left", "right"}:
                raise ValueError(f"sequence step {index} (scroll) requires direction up/down/left/right.")
            if current_action == "drag":
                for coordinate_name in ("from_coordinate", "to_coordinate"):
                    point = current.get(coordinate_name)
                    if not isinstance(point, (list, tuple)) or len(point) != 2:
                        raise ValueError(f"sequence step {index} (drag) requires {coordinate_name}=[x,y] from the latest capture.")
                    try:
                        [float(value) for value in point]
                    except (TypeError, ValueError):
                        raise ValueError(f"sequence step {index} (drag) has invalid {coordinate_name} values.")
            normalized_steps.append(current)

        step_results: list[dict[str, Any]] = []
        completed_steps = 0
        failure = ""
        for index, step in enumerate(normalized_steps, start=1):
            step_action = str(step["action"]).strip().casefold()
            target = str(step.get("target") or "").strip()
            if step_action == "wait":
                try:
                    seconds = max(0.0, min(float(step.get("seconds", 1.0)), 10.0))
                except (TypeError, ValueError):
                    seconds = 1.0
                time.sleep(seconds)
                _state_path, state = self._current_state()
                latest_target = state.get("latest_computer_use_target", {})
                latest_target = latest_target if isinstance(latest_target, dict) else {}
                step_result = self._capture(
                    mode="ax", pid=latest_target.get("pid"),
                    window_id=latest_target.get("window_id"), max_results=30,
                    include_windows=False,
                )
                step_summary = (
                    f"Waited {seconds:g}s and refreshed accessibility state."
                    if step_result.success
                    else f"Waited {seconds:g}s, but accessibility refresh failed: {step_result.summary}"
                )
            else:
                action_args: dict[str, Any] = {
                    "action": step_action,
                    # Every state-changing batch step needs a fresh visual postcondition;
                    # AX-only steps cannot distinguish a successful interaction from a no-op.
                    "mode": "som",
                    "capture_after": True,
                    "include_windows": False,
                    "max_results": 30,
                    "delivery_mode": delivery_mode,
                }
                if target:
                    action_args["target"] = target
                for key in ("text", "key", "keys", "direction", "coordinate", "amount", "button", "from_coordinate", "to_coordinate", "duration_ms"):
                    if key in step:
                        action_args[key] = step[key]
                step_result = self.execute(**action_args)
                step_summary = str(step_result.summary or "")[:280]

            succeeded = bool(step_result.success)
            step_entry = {
                "index": index,
                "action": step_action,
                "target": target or None,
                "success": succeeded,
                "summary": step_summary,
            }
            step_results.append(step_entry)
            if not succeeded:
                failure = f"Step {index} ({step_action}{' ' + repr(target) if target else ''}) failed: {step_summary}"
                break
            completed_steps += 1

            # A batched sequence cannot pause for the model to inspect the UI between
            # actions. Never chain a click/submit after text that the tool itself cannot
            # read back exactly: stop so the model can inspect the returned screenshot.
            if step_action == "type":
                latest_state = self._current_state()[1]
                id_map = latest_state.get("latest_element_ids", {})
                records = latest_state.get("latest_accessibility_elements", {})
                exact, partial = self._target_candidates(target)
                candidates = exact if exact else partial
                observed_value: Any = None
                value_readable = False
                if len(candidates) == 1 and isinstance(id_map, dict) and isinstance(records, dict):
                    token = id_map.get(str(candidates[0]))
                    record = records.get(str(token)) if token is not None else None
                    if isinstance(record, dict) and "value" in record:
                        observed_value = record.get("value")
                        value_readable = observed_value == step.get("text")
                if not value_readable:
                    step_entry["verification"] = "text_value_not_read_back"
                    step_entry["summary"] = (
                        "Text input was dispatched, but the exact value is not available from the field's accessibility readback. "
                        "Sequence paused before the next step; inspect the final screenshot and verify the field visually."
                    )
                    failure = (
                        f"Step {index} (type {target!r}) has unverified text: the exact typed value was not readable from the current field. "
                        "The remaining steps were not run; inspect the screenshot before submitting or continuing."
                    )
                    break

            if step_action != "wait":
                step_content = step_result.content if isinstance(step_result.content, dict) else {}
                post_capture = step_content.get("capture_after")
                visual_change = post_capture.get("screen_hash_changed") if isinstance(post_capture, dict) else None
                if visual_change is False:
                    step_entry["verification"] = "no_visible_change"
                    failure = (
                        f"Step {index} ({step_action}{' ' + repr(target) if target else ''}) was dispatched but no screenshot change was detected. "
                        "The remaining steps were not run; inspect current controls and choose a different evidence-based action."
                    )
                    break
                if visual_change is None:
                    step_entry["verification"] = "visual_effect_unavailable"
                    failure = (
                        f"Step {index} ({step_action}{' ' + repr(target) if target else ''}) has no usable visual postcondition. "
                        "The remaining steps were not run; capture the screen and verify before continuing."
                    )
                    break

        _state_path, state = self._current_state()
        latest_target = state.get("latest_computer_use_target", {})
        latest_target = latest_target if isinstance(latest_target, dict) else {}
        final_capture = self._capture(
            mode=final_mode if final_mode in {"som", "vision", "ax"} else "som",
            pid=latest_target.get("pid"),
            window_id=latest_target.get("window_id"),
            max_results=max(1, min(30, int(max_results))),
            include_windows=include_windows,
        )
        final_content = final_capture.content if isinstance(final_capture.content, dict) else {}
        content: dict[str, Any] = {
            "action": "sequence",
            "complete": not failure and completed_steps == len(normalized_steps),
            "requested_steps": len(normalized_steps),
            "completed_steps": completed_steps,
            "step_results": step_results,
            "window": final_content.get("window"),
            "element_count": final_content.get("element_count", 0),
            "elements": final_content.get("elements", []),
            "elements_text": final_content.get("elements_text", ""),
            "screenshot_hash": final_content.get("screenshot_hash"),
            "width": final_content.get("width"),
            "height": final_content.get("height"),
            "warning": final_content.get("warning") or final_content.get("selection_warning"),
        }
        for key in ("image_ref", "image_mime_type", "cursor_position", "cursor_hint", "pointer_hint"):
            if final_content.get(key) is not None:
                content[key] = final_content[key]
        succeeded = not failure and completed_steps == len(normalized_steps) and final_capture.success
        if failure:
            summary = f"Sequence stopped after {completed_steps}/{len(normalized_steps)} step(s). {failure}"
        elif completed_steps == len(normalized_steps):
            summary = f"Sequence completed {completed_steps}/{len(normalized_steps)} planned UI step(s)."
        else:
            summary = f"Sequence stopped after {completed_steps}/{len(normalized_steps)} step(s)."
        if final_capture.success and final_capture.summary:
            summary += "\nFinal UI state:\n" + str(final_capture.summary)
        elif not final_capture.success:
            summary += " Final UI capture failed: " + str(final_capture.summary)
        effects = [
            {"action": str(item["action"]), "target": str(item.get("target") or "desktop")}
            for item in step_results if item.get("success")
        ]
        return ToolResult(
            success=succeeded,
            name=self.name,
            content=content,
            metadata={
                "kind": "computer_use_sequence",
                "action": "sequence",
                "completed_steps": completed_steps,
                "requested_steps": len(normalized_steps),
                **({"image_ref": content["image_ref"], "image_mime_type": content.get("image_mime_type", "image/png")} if content.get("image_ref") else {}),
            },
            summary=summary,
            evidence={
                "completed_steps": completed_steps,
                "requested_steps": len(normalized_steps),
                "step_results": step_results,
                "final_window": content.get("window"),
                "final_elements": content.get("elements_text", ""),
                "failure": failure,
            },
            effects=effects,
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
            # Make driver readiness deterministic: start/verify Cua Driver before
            # launching a browser or attempting the first screenshot/window lookup.
            # This avoids relying on a particular call returning a daemon-not-running
            # error before startup recovery is triggered.
            if action in _DRIVER_BACKED_ACTIONS:
                ensure_driver_running(self._observe._session_label())

            if action == "sequence":
                return self._execute_sequence(
                    steps=kwargs.get("steps"),
                    final_mode=str(kwargs.get("mode") or "som"),
                    delivery_mode=str(kwargs.get("delivery_mode") or "background"),
                    max_results=kwargs.get("max_results", 24),
                    include_windows=bool(kwargs.get("include_windows", True)),
                )

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
                "drag",
                "type",
                "key",
                "scroll",
                "press_key",
                "hotkey",
            }:
                target_label = str(kwargs.pop("target", "") or "").strip()
                kwargs.pop("_target_snapshot_fresh", None)
                if target_label:
                    if action not in {"click", "double_click", "right_click", "middle_click", "type"}:
                        raise ValueError("target='visible label' is supported only for click/type actions; drag uses from_coordinate/to_coordinate.")
                    if kwargs.get("element") is not None:
                        raise ValueError("Supply either target='visible label' or element=N, not both.")
                    # Reuse the last model-visible accessibility snapshot. Refresh only if
                    # the label is absent; stale-token failures have their own one-shot recovery.
                    kwargs["element"] = self._resolve_target_id(target_label)
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
                capture_mode = str(kwargs.get("mode") or "som").strip().casefold()
                screen_hash_changed = (
                    before_screen_hash != after_screen_hash
                    if capture_mode != "ax" and isinstance(before_screen_hash, str) and isinstance(after_screen_hash, str)
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
