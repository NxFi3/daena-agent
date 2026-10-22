from __future__ import annotations

import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool
from src.tools.builtin.computer_use.driver import (
    CuaDriverError,
    call_driver,
    pointer_state_path,
    session_label_from_id,
)


_INPUT_ROLES = {
    "combo box", "text field", "entry", "search box", "password text",
    "editable text", "text input", "input",
}
_ACTION_ROLES = _INPUT_ROLES | {
    "button", "push button", "toggle button", "check box", "radio button",
    "link", "menu item", "tab", "page tab", "list item", "tree item",
    "spin button", "slider", "option", "switch", "select", "landmark",
    "panel", "section",
}
_BROWSER_NAMES = ("firefox", "chrome", "chromium", "brave", "edge", "browser")
_NOISE_APPS = ("gjs", "update-notifier", "ibus-", "xdg-desktop-portal")
_BROWSER_INTENT_PHRASES = (
    "browser", "website", "web page", "webpage", "url", "address bar",
    "navigate", "search", "sign in", "log in", "login", "internet",
    "مرورگر", "وب", "سایت", "جست", "آدرس", "صفحه وب",
)
_QUERY_STOP_WORDS = {
    "the", "and", "for", "with", "from", "please", "open", "go", "to",
    "page", "site", "website", "browser", "login", "log", "sign", "in",
    "on", "at", "my", "your", "into", "there", "this", "that", "find",
    "visit", "navigate", "web", "url", "search", "enter", "type",
    "در", "به", "را", "از", "برای", "روی", "یک", "این", "آن", "بازکن",
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normal(value: Any) -> str:
    return (
        _text(value).casefold()
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\u064a", "\u06cc")
        .replace("\u0643", "\u06a9")
    )


def _integer(value: Any, name: str) -> int:
    """LLMs sometimes serialize large integer IDs as '4.27360257e+08'."""
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer from the live window list.")
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f"{name} must be an integer; received {value!r}.")
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"{name} must be a whole number; received {value!r}.")
    result = int(number)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


class ComputerInspectTool(Tool):
    name = "computer_inspect"
    action = "inspect"
    description = (
        "Find real UI controls in a live window. For browser tasks, normally call this with only a "
        "specific query (e.g. 'جست‌و‌جو در آپارات') and omit pid/window_id; it selects a visible "
        "browser window, queries its accessibility tree, and returns only a few matching controls with "
        "element_token handles. Then use computer_act with that token. Pass window_title only to disambiguate "
        "multiple windows. Never read a made-up snapshot file or re-inspect just to recover a token hidden in "
        "a long output: best_match contains the token directly. A new inspect invalidates old tokens."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Visible element label/name to find, e.g. 'جست‌و‌جو در آپارات'.",
            },
            "window_title": {
                "type": "string",
                "description": "Optional visible-window title substring to select, e.g. 'Firefox' or 'Paint'.",
            },
            "pid": {
                "type": "integer",
                "description": "Optional process id from computer_observe; normally omit for browser tasks.",
            },
            "window_id": {
                "type": ["integer", "string"],
                "description": "Optional native window id; normally omit. Scientific-notation strings are safely converted only when they represent an exact whole number.",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 30,
                "default": 24,
                "description": "Number of controls to return for a unified capture. The model-facing wrapper numbers these controls and does not expose opaque driver tokens.",
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.workspace_root = Path.cwd().resolve()
        self.session_id_provider = None

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
                pass
        return session_label_from_id(raw)

    @staticmethod
    def _windows() -> list[dict[str, Any]]:
        raw = call_driver("list_windows", {"on_screen_only": True}, timeout=10)
        payload = json.loads(raw)
        windows = payload.get("windows", []) if isinstance(payload, dict) else payload
        if not isinstance(windows, list):
            raise CuaDriverError("Cua Driver returned an invalid window inventory.")
        return [item for item in windows if isinstance(item, dict)]

    @staticmethod
    def _is_browser(window: dict[str, Any]) -> bool:
        signature = _normal(f"{window.get('app_name', '')} {window.get('title', '')}")
        return any(name in signature for name in _BROWSER_NAMES)

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        normalized = _normal(query)
        return {
            token for token in re.findall(r"[^\W_]{3,}", normalized, flags=re.UNICODE)
            if token not in _QUERY_STOP_WORDS
        }

    @classmethod
    def _matching_browser_windows(
        cls,
        browser_rows: list[dict[str, Any]],
        query: str,
    ) -> list[dict[str, Any]]:
        """Match named browser content generically; no site-specific routing rules."""
        terms = cls._query_terms(query)
        if terms:
            matches = [
                row for row in browser_rows
                if terms & cls._query_terms(
                    f"{row.get('title', '')} {row.get('app_name', '')}"
                )
            ]
            if matches:
                return matches
        query_norm = _normal(query)
        if any(phrase in query_norm for phrase in _BROWSER_INTENT_PHRASES):
            return browser_rows
        return []

    @classmethod
    def _matching_app_windows(
        cls, windows: list[dict[str, Any]], app_name_hint: str,
    ) -> list[dict[str, Any]]:
        """Match application intent against app identity, not arbitrary page titles.

        Browser inventories sometimes expose the active tab title as app_name. A page
        titled "Paint A House — Mozilla Firefox" must not win a request to capture
        the native Paint application merely because its page title contains "Paint".
        """
        requested = _normal(app_name_hint).strip()
        if not requested:
            return windows

        browser_families = {
            "firefox": ("firefox", "firefox-esr", "mozilla firefox"),
            "mozilla firefox": ("firefox", "firefox-esr", "mozilla firefox"),
            "chrome": ("chrome", "google chrome", "chromium", "chromium-browser"),
            "google chrome": ("chrome", "google chrome", "chromium", "chromium-browser"),
            "chromium": ("chrome", "google chrome", "chromium", "chromium-browser"),
            "brave": ("brave", "brave browser"),
            "edge": ("edge", "microsoft edge"),
            "microsoft edge": ("edge", "microsoft edge"),
        }
        family = next((values for key, values in browser_families.items() if requested == key), None)
        if family is not None:
            return [
                row for row in windows
                if cls._is_browser(row)
                and any(alias in _normal(f"{row.get('app_name', '')} {row.get('title', '')}") for alias in family)
            ]

        aliases = {
            "files": ("files", "nautilus", "dolphin", "thunar", "file manager"),
            "file manager": ("files", "nautilus", "dolphin", "thunar", "file manager"),
            "terminal": ("terminal", "gnome-terminal", "konsole", "xterm", "ptyxis"),
            "vscode": ("vscode", "visual studio code", "code"),
            "visual studio code": ("vscode", "visual studio code", "code"),
        }
        needles = aliases.get(requested, (requested,))
        # For ordinary applications, app_name is the identity field. Searching the
        # whole window title is unsafe: a terminal running `python paint_app.py`
        # would otherwise be mistaken for Paint just because its title contains the
        # script filename. Fall back to title only when the inventory has no app name.
        def identity(row: dict[str, Any]) -> str:
            app_name = _normal(row.get("app_name"))
            return app_name or _normal(row.get("title"))

        return [
            row for row in windows
            if not cls._is_browser(row)
            and any(needle in identity(row) for needle in needles)
        ]

    def _resolve_target(
        self,
        pid_value: Any,
        window_id_value: Any,
        window_title: str,
        query: str,
        app_name_hint: str = "",
    ) -> tuple[int, int | None, str, str]:
        session = self._session_label()
        windows = self._windows()
        target_title = _normal(window_title)
        app_hint = _normal(app_name_hint)
        query_norm = _normal(query)

        if pid_value is not None:
            pid = _integer(pid_value, "pid")
            window_id = _integer(window_id_value, "window_id") if window_id_value is not None else None
            known = [
                window for window in windows
                if int(window.get("pid") or 0) == pid
                and (window_id is None or int(window.get("window_id") or 0) == window_id)
            ]
            if known:
                selected = max(
                    known,
                    key=lambda row: int(row.get("width") or 0) * int(row.get("height") or 0),
                )
                app = _text(selected.get("app_name"))
                title = _text(selected.get("title"))
                # An explicit application hint is stronger than page-title/query
                # heuristics: never switch away from the requested application here.
                if app_hint and not self._matching_app_windows([selected], app_name_hint):
                    raise ValueError(
                        f"The selected window '{title}' does not belong to requested application "
                        f"'{app_name_hint}'. Capture was refused rather than selecting a similarly named browser tab."
                    )
                # The model can mistake a file dialog for a browser. Recover only when
                # no explicit application was requested and the query clearly names web content.
                if not app_hint and not self._is_browser(selected):
                    browser_rows = [
                        row for row in windows
                        if self._is_browser(row) and bool(row.get("is_on_screen", True))
                        and int(row.get("width") or 0) >= 600
                        and int(row.get("height") or 0) >= 400
                    ]
                    browser_matches = self._matching_browser_windows(browser_rows, query_norm)
                    if browser_matches:
                        selected = self._choose_window(browser_matches, target_title, query_norm)
                        pid = _integer(selected.get("pid"), "pid")
                        window_id = _integer(selected.get("window_id"), "window_id")
                        return pid, window_id, _text(selected.get("title")), _text(selected.get("app_name"))
                if target_title and target_title not in _normal(title) and target_title not in _normal(app):
                    raise ValueError(
                        f"The selected window is '{title}', not a match for '{window_title}'. "
                        "Call computer_inspect without numeric IDs or specify a matching window_title."
                    )
                return pid, window_id, title, app
            if window_id is not None:
                raise ValueError(
                    f"No visible window matches pid={pid}, window_id={window_id}. "
                    "Omit numeric IDs so computer_inspect can resolve the current browser window."
                )
            return pid, None, "", ""

        candidates = [
            row for row in windows
            if bool(row.get("is_on_screen", True))
            and int(row.get("width") or 0) >= 300
            and int(row.get("height") or 0) >= 200
            and not any(noise in _normal(row.get("app_name")) for noise in _NOISE_APPS)
        ]
        browser_rows = [row for row in candidates if self._is_browser(row)]

        if app_hint:
            app_matches = self._matching_app_windows(candidates, app_name_hint)
            if target_title:
                app_matches = [
                    row for row in app_matches
                    if target_title in _normal(row.get("title"))
                    or target_title in _normal(row.get("app_name"))
                ]
            if not app_matches:
                available = "; ".join(
                    f"{_text(row.get('app_name'))}: {_text(row.get('title'))}"
                    for row in candidates[:8]
                )
                requested = f"application '{app_name_hint}'"
                if window_title:
                    requested += f" and window title '{window_title}'"
                raise ValueError(
                    f"No visible window matches requested {requested}. Visible windows: {available or 'none'}."
                )
            selected = self._choose_window(app_matches, target_title, query_norm)
        elif target_title:
            matches = [
                row for row in candidates
                if target_title in _normal(row.get("title"))
                or target_title in _normal(row.get("app_name"))
            ]
            if not matches:
                available = "; ".join(
                    f"{_text(row.get('app_name'))}: {_text(row.get('title'))}"
                    for row in candidates[:8]
                )
                raise ValueError(
                    f"No visible window title matches '{window_title}'. Visible windows: {available or 'none'}."
                )
            selected = self._choose_window(matches, target_title, query_norm)
        elif browser_rows and (browser_matches := self._matching_browser_windows(browser_rows, query_norm)):
            # Follow a named page/tab or clear web intent. Otherwise, generic desktop
            # work follows the actual foreground app (Paint, an editor, and so on).
            selected = self._choose_window(browser_matches, "", query_norm)
        elif candidates:
            selected = self._choose_window(candidates, "", query_norm)
        else:
            raise CuaDriverError(
                "No suitable visible application window was found. Call computer_observe and select a live window."
            )

        return (
            _integer(selected.get("pid"), "pid"),
            _integer(selected.get("window_id"), "window_id"),
            _text(selected.get("title")),
            _text(selected.get("app_name")),
        )

    @staticmethod
    def _choose_window(
        windows: list[dict[str, Any]],
        title_hint: str,
        query: str,
    ) -> dict[str, Any]:
        def score(window: dict[str, Any]) -> tuple[int, int, int]:
            title = _normal(window.get("title"))
            app = _normal(window.get("app_name"))
            hint_match = 1000 if title_hint and title_hint in title else 0
            query_match = 500 if query and query in title else 0
            area = int(window.get("width") or 0) * int(window.get("height") or 0)
            # Do not globally prefer browsers. Generic desktop tasks must select the
            # foreground app (Paint, an editor, etc.); website preference is applied
            # explicitly in _resolve_target only when the query names a website.
            return hint_match + query_match, int(window.get("z_index") or 0), area
        return max(windows, key=score)

    @staticmethod
    def _read_snapshot(
        pid: int,
        window_id: int | None,
        query: str,
        session: str,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {
            "pid": pid,
            "include_accessibility_tree": True,
            "include_screenshot": False,
            "max_elements": 5000,
            "max_depth": 20,
            "timeout_ms": 9000,
            "session": session,
        }
        if window_id is not None:
            args["window_id"] = window_id
        if query:
            args["query"] = query
        raw = call_driver("get_window_state", args, timeout=14)
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Cua Driver returned an invalid accessibility snapshot.")
        if payload.get("code") or payload.get("isError"):
            raise CuaDriverError(json.dumps(payload, ensure_ascii=False)[:1000])
        return payload

    @staticmethod
    def _accessibility_frame_space(
        payload: dict[str, Any], items: list[dict[str, Any]],
    ) -> str | None:
        """Infer whether AT-SPI bounds are screen-space or window-local.

        Cua Driver normalizes some native toolkit trees to window-local coordinates,
        while browser AT-SPI trees may expose screen-space bounds. The native window
        placement is always screen-space, so comparing local child frames to that
        rectangle hides controls whenever the window is not positioned at (0, 0).
        """
        bounds = payload.get("window_bounds")
        if not isinstance(bounds, dict):
            return None
        try:
            bx, by, bw, bh = (
                float(bounds[key]) for key in ("x", "y", "width", "height")
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if bw <= 0 or bh <= 0:
            return None

        title = _normal(payload.get("window_title"))
        # A top-level frame matching native screen bounds is strong evidence that
        # its descendants use screen coordinates. A root at (0, 0) is window-local.
        for item in items:
            if (
                _normal(item.get("role")) not in {"frame", "window"}
                or _normal(item.get("label")) != title
                or not isinstance(item.get("frame"), dict)
            ):
                continue
            frame = item["frame"]
            try:
                fx, fy, fw, fh = (
                    float(frame.get(key, frame.get(alt)))
                    for key, alt in (
                        ("x", "left"), ("y", "top"),
                        ("w", "width"), ("h", "height"),
                    )
                )
            except (TypeError, ValueError, OverflowError):
                continue
            if (
                abs(fx - bx) <= 2 and abs(fy - by) <= 2
                and abs(fw - bw) <= 2 and abs(fh - bh) <= 2
            ):
                return "desktop"
            if (
                abs(fx) <= 2 and abs(fy) <= 2
                and abs(fw - bw) <= 2 and abs(fh - bh) <= 2
            ):
                return "window"

        declared = _normal(payload.get("coordinate_frame"))
        if declared in {"window", "window-local", "window local"}:
            return "window"
        if declared in {"desktop", "screen", "screen-space"}:
            return "desktop"

        def intersects(frame: Any, rect: tuple[float, float, float, float]) -> bool:
            if not isinstance(frame, dict):
                return False
            try:
                x = float(frame.get("x", frame.get("left")))
                y = float(frame.get("y", frame.get("top")))
                w = float(frame.get("w", frame.get("width")))
                h = float(frame.get("h", frame.get("height")))
            except (TypeError, ValueError, OverflowError):
                return False
            left, top, right, bottom = rect
            return w > 0 and h > 0 and x + w > left and x < right and y + h > top and y < bottom

        desktop_rect = (bx, by, bx + bw, by + bh)
        local_rect = (0.0, 0.0, bw, bh)
        desktop_hits = sum(intersects(item.get("frame"), desktop_rect) for item in items)
        local_hits = sum(intersects(item.get("frame"), local_rect) for item in items)
        return "window" if local_hits > desktop_hits else "desktop"

    @staticmethod
    def _wayland_window_top_inset() -> int:
        """Return the compositor titlebar inset for full-window Wayland screenshots.

        On GNOME/Wayland, screenshots include the server-side titlebar while native
        AT-SPI child frames are relative to the client surface. The screenshot and
        frame_note currently claim the same origin, but Paint demonstrates a 38 px
        mismatch. Allow tuning for different themes with DAENA_CUA_WAYLAND_TOP_INSET.
        """
        is_wayland = (
            str(os.environ.get("XDG_SESSION_TYPE", "")).strip().casefold() == "wayland"
            or bool(os.environ.get("WAYLAND_DISPLAY"))
        )
        if not is_wayland:
            return 0
        try:
            inset = int(os.environ.get("DAENA_CUA_WAYLAND_TOP_INSET", "38"))
        except (TypeError, ValueError):
            inset = 38
        return max(0, min(100, inset))

    @staticmethod
    def _checked_from_states(item: dict[str, Any]) -> bool | None:
        """Read CHECKED from a driver-provided AT-SPI state list (if that is its schema)."""
        for key in ("states", "state", "state_set", "flags"):
            states = item.get(key)
            if isinstance(states, (list, tuple, set)) and states:
                names = {
                    _normal(name).replace("_", " ").replace("-", " ")
                    for name in states if isinstance(name, str)
                }
                if names & {"checked", "pressed"}:
                    return True
                if "checkable" in names:
                    return False
        return None

    @staticmethod
    def _rank_elements(payload: dict[str, Any], query: str, limit: int = 8) -> list[dict[str, Any]]:
        items = payload.get("elements", [])
        if not isinstance(items, list):
            return []
        items = [item for item in items if isinstance(item, dict)]
        # Firefox can expose controls from several native windows and browser tabs
        # in one process-wide AT-SPI tree. Scope first to the exact top-level frame,
        # then to that frame's selected browser tab. Nested modal/iframe documents
        # remain usable when they descend from the active page.
        by_index = {
            item.get("element_index"): item
            for item in items if isinstance(item.get("element_index"), int)
        }
        title = _normal(payload.get("window_title"))
        frame_roots = {
            item["element_index"] for item in items
            if isinstance(item.get("element_index"), int)
            and _normal(item.get("role")) in {"frame", "window"}
            and _normal(item.get("label")) == title
            and item.get("parent_index") is None
        }
        if not frame_roots:
            frame_roots = {
                item["element_index"] for item in items
                if isinstance(item.get("element_index"), int)
                and _normal(item.get("role")) in {"frame", "window"}
                and _normal(item.get("label")) == title
            }

        def has_ancestor_in(item: dict[str, Any], roots: set[int]) -> bool:
            current = item
            visited: set[int] = set()
            for _ in range(100):
                idx = current.get("element_index")
                if isinstance(idx, int) and idx in roots:
                    return True
                parent = current.get("parent_index")
                if not isinstance(parent, int) or parent in visited or parent not in by_index:
                    return False
                visited.add(parent)
                current = by_index[parent]
            return False

        def belongs_to_frame(item: dict[str, Any]) -> bool:
            return not frame_roots or has_ancestor_in(item, frame_roots)

        selected_tab_labels: set[str] = set()
        for item in items:
            if _normal(item.get("role")) not in {"page tab", "tab"} or item.get("selected") is not True:
                continue
            if not belongs_to_frame(item):
                continue
            current = item
            visited: set[int] = set()
            for _ in range(12):
                parent = current.get("parent_index")
                if not isinstance(parent, int) or parent in visited or parent not in by_index:
                    break
                visited.add(parent)
                current = by_index[parent]
                if _normal(current.get("role")) == "tool bar" and _normal(current.get("label")) == "browser tabs":
                    label = _normal(item.get("label"))
                    if label:
                        selected_tab_labels.add(label)
                    break

        page_title = title
        for suffix in (
            " — mozilla firefox", " - mozilla firefox",
            " — google chrome", " - google chrome",
            " — chromium", " - chromium",
            " — brave", " - brave",
            " — microsoft edge", " - microsoft edge",
        ):
            if page_title.endswith(suffix):
                page_title = page_title[:-len(suffix)].strip()
                break
        active_labels = selected_tab_labels if len(selected_tab_labels) == 1 else {page_title}
        document_roles = {"document", "web document", "document web"}
        page_roots = {
            item["element_index"] for item in items
            if isinstance(item.get("element_index"), int)
            and _normal(item.get("role")) in document_roles
            and _normal(item.get("label")) in active_labels
            and belongs_to_frame(item)
            # Only top-level page documents define a browser tab's content root.
            and not (
                isinstance(item.get("parent_index"), int)
                and item.get("parent_index") in by_index
                and _normal(by_index[item["parent_index"]].get("role")) in document_roles
            )
        }

        def belongs_to_target_window(item: dict[str, Any]) -> bool:
            if not belongs_to_frame(item):
                return False
            if not page_roots:
                return True
            role = _normal(item.get("role"))
            in_page = (
                item.get("in_web_content") is True
                or role in document_roles
                or role in {"internal frame", "scroll pane"}
            )
            if not in_page:
                # Address bars, browser tabs, toolbars, and other browser chrome.
                return True
            # Main-page controls and nested modal/frame documents must descend from
            # the active page root; other tabs are siblings and are rejected.
            return has_ancestor_in(item, page_roots)

        query_norm = _normal(query)
        # Reject descendants outside the target window, comparing rectangles
        # in the same coordinate space. Native toolkit frames may be window-local
        # even though window_bounds contains the window's desktop placement.
        window_bounds = payload.get("window_bounds")
        frame_space = ComputerInspectTool._accessibility_frame_space(payload, items)
        visible_rect = None
        if isinstance(window_bounds, dict):
            try:
                vx = float(window_bounds.get("x", 0))
                vy = float(window_bounds.get("y", 0))
                vw = float(window_bounds.get("width", 0))
                vh = float(window_bounds.get("height", 0))
                if vw > 0 and vh > 0:
                    if frame_space == "window":
                        visible_rect = (0.0, 0.0, vw, vh)
                    else:
                        visible_rect = (vx, vy, vx + vw, vy + vh)
            except (TypeError, ValueError, OverflowError):
                visible_rect = None
        address_query = any(
            hint in query_norm
            for hint in ("address", "url", "location bar", "navigation bar", "آدرس", "نوارآدرس")
        )
        # Treat words such as "field", "bar", and "input" as descriptions of the
        # requested control, not as required literal words in its accessible name.
        descriptor_words = {
            "a", "an", "the", "or", "and", "please", "find", "open", "click",
            "field", "input", "box", "bar", "text", "control", "controls", "element",
            "button", "browser", "in", "on", "with", "the", "my",
        }
        query_terms = [
            term for term in query_norm.split()
            if len(term) > 1 and term not in descriptor_words
        ]
        candidates: list[tuple[float, int, dict[str, Any]]] = []

        for order, item in enumerate(items):
            if not belongs_to_target_window(item):
                continue
            role = _normal(item.get("role"))
            label = _text(item.get("label"))
            value = _text(item.get("value"))
            description = _text(item.get("description"))
            text = _normal(f"{label} {value} {description}")
            actions = item.get("actions", [])
            if not isinstance(actions, list):
                actions = []
            actions = [str(action) for action in actions if _normal(action) != "clickancestor"]
            token = _text(item.get("element_token"))
            if not token:
                continue
            # Sections/landmarks are often giant web-page containers. Keep them only
            # when AT-SPI exposes a real action; otherwise a chat transcript mentioning
            # the requested label must not masquerade as the target control.
            if role in {"section", "panel", "landmark"} and not actions:
                continue

            in_page = bool(item.get("in_web_content", False))
            exact_match = bool(query_norm and query_norm in text)
            matched_terms = [term for term in query_terms if term in text or term in role]
            term_match = bool(query_terms and len(matched_terms) == len(query_terms))
            # Address-bar labels vary across platforms. For an address/URL request,
            # select a browser-chrome input/combobox rather than demanding that its
            # accessible name literally contain every word in the user's sentence.
            address_control = (
                address_query
                and not in_page
                and role in _INPUT_ROLES
                and (
                    any(hint in text for hint in ("address", "url", "location", "search with google", "enter address", "آدرس"))
                    or role in {"combo box", "text field", "entry", "search box"}
                )
            )
            if query_norm and not (exact_match or term_match or address_control):
                continue

            if role in _INPUT_ROLES:
                score = 120
            elif role in {"button", "push button", "toggle button", "check box", "radio button"}:
                score = 75
            elif role in {"link", "menu item", "tab", "page tab"}:
                score = 55
            elif role in {"landmark", "panel", "section"} and actions:
                score = 40
            elif actions:
                score = 25
            elif role in _ACTION_ROLES:
                score = 10
            else:
                continue

            score += 40 if in_page else -15
            score += 8 if label else 0
            score += 220 if address_control else 0
            score += 160 if "جست" in text and "آپارات" in text else 0
            score += 120 if "search" in text and role in _INPUT_ROLES and in_page else 0
            score += 100 if exact_match else (30 if term_match else 0)

            frame = item.get("frame")
            if isinstance(frame, dict):
                try:
                    x = float(frame.get("x", -1))
                    y = float(frame.get("y", -1))
                    w = float(frame.get("w", 0))
                    h = float(frame.get("h", 0))
                    if w <= 0 or h <= 0:
                        continue
                    if visible_rect is not None:
                        left, top, right, bottom = visible_rect
                        if x + w <= left or x >= right or y + h <= top or y >= bottom:
                            continue
                    elif x + w <= 0 or y + h <= 0 or x > 10000 or y > 10000:
                        continue
                except (TypeError, ValueError):
                    continue

            row = {
                "element_token": token,
                "role": _text(item.get("role")),
                "label": label[:120],
                "actions": actions[:5],
                "enabled": bool(item.get("enabled", True)),
                "in_web_content": in_page,
            }
            if value and value != label:
                row["value"] = value[:100]
            if isinstance(item.get("selected"), bool):
                # Checkbox/toggle selection is state, not merely a successful click.
                # Keep it so the model can see when a drawing tool is already active.
                row["selected"] = item["selected"]
            checked_value = item.get("checked")
            if not isinstance(checked_value, bool):
                checked_value = ComputerInspectTool._checked_from_states(item)
            if isinstance(checked_value, bool):
                # AT-SPI's CHECKED state is distinct from SELECTED (often focus/selection).
                # Preserve it separately; never infer checkbox state from `selected`.
                row["checked"] = checked_value
            if isinstance(frame, dict):
                normalized_frame = {key: frame[key] for key in ("x", "y", "w", "h") if key in frame}
                if frame_space == "window":
                    inset = ComputerInspectTool._wayland_window_top_inset()
                    if inset and isinstance(normalized_frame.get("y"), (int, float)):
                        normalized_frame["y"] = round(float(normalized_frame["y"]) + inset, 2)
                row["frame"] = normalized_frame
                row["frame_coordinate_space"] = frame_space or "desktop"
            candidates.append((score, order, row))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [row for _score, _order, row in candidates[:limit]]

    def execute(
        self,
        query: str = "",
        window_title: str = "",
        pid: Any = None,
        window_id: Any = None,
        max_results: int = 8,
        **_kwargs: Any,
    ) -> ToolResult:
        try:
            query = _text(query)[:160]
            window_title = _text(window_title)[:160]
            max_results = max(1, min(30, _integer(max_results, "max_results")))
            session = self._session_label()
            app_name_hint = _text(_kwargs.get("app_name_hint"))[:120]
            target_pid, target_window_id, inventory_title, app_name = self._resolve_target(
                pid, window_id, window_title, query, app_name_hint=app_name_hint
            )

            payload = self._read_snapshot(target_pid, target_window_id, query, session)
            elements = self._rank_elements(payload, query, max_results)

            # Some custom controls have an inaccessible/empty name. If a requested label
            # returns no matches, one bounded full-tree pass lets us do local fuzzy matching.
            if query and not elements:
                payload = self._read_snapshot(target_pid, target_window_id, "", session)
                elements = self._rank_elements(payload, query, max_results)

            actual_title = _text(payload.get("window_title")) or inventory_title
            actual_pid = _integer(payload.get("pid", target_pid), "pid")
            actual_window = payload.get("window_id", target_window_id)
            if actual_window is not None:
                actual_window = _integer(actual_window, "window_id")

            state_path = pointer_state_path(self.workspace_root, session)
            state: dict[str, Any] = {}
            try:
                state_path.parent.mkdir(parents=True, exist_ok=True)
                previous = json.loads(state_path.read_text(encoding="utf-8"))
                if isinstance(previous, dict):
                    state = previous
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

            token_map = {
                row["element_token"]: {
                    "role": row["role"],
                    "label": row["label"],
                    "value": row.get("value", ""),
                    "actions": row.get("actions", []),
                    "frame": row.get("frame"),
                    "frame_coordinate_space": row.get("frame_coordinate_space", "desktop"),
                    **({"selected": row["selected"]} if isinstance(row.get("selected"), bool) else {}),
                    **({"checked": row["checked"]} if isinstance(row.get("checked"), bool) else {}),
                    "in_web_content": row["in_web_content"],
                }
                for row in elements
            }
            state.update({
                "latest_accessibility_pid": actual_pid,
                "latest_accessibility_window_id": actual_window,
                "latest_accessibility_window_title": actual_title,
                "latest_accessibility_elements": token_map,
            })
            try:
                temp = state_path.with_suffix(".tmp")
                temp.write_text(json.dumps(state), encoding="utf-8")
                temp.replace(state_path)
            except OSError:
                pass

            best_match = elements[0] if elements else None
            content = {
                "window": {
                    "app": app_name or _text(payload.get("app_name")),
                    "title": actual_title,
                    "pid": actual_pid,
                    "window_id": actual_window,
                },
                "query": query,
                "match_count": len(elements),
                "best_match": best_match,
                "elements": elements,
                "truncated": bool(payload.get("truncated", False)),
                "result": (
                    "MATCH_FOUND" if best_match else
                    "NO_MATCH_FOUND: do not click; confirm the intended page/window first."
                ),
            }
            summary = (
                f"Found {len(elements)} matching UI element(s) in {actual_title or app_name or 'window'}. "
                f"Best match: {best_match['role']} '{best_match['label']}' ({best_match['element_token']})."
                if best_match else
                f"No UI element matched '{query}' in {actual_title or app_name or 'window'}. Do not guess coordinates."
            )
            return ToolResult(
                success=True,
                name=self.name,
                content=content,
                metadata={
                    "kind": "computer_accessibility_snapshot",
                    "pid": actual_pid,
                    "window_id": actual_window,
                    "window_title": actual_title,
                    "query": query,
                    "match_count": len(elements),
                },
                summary=summary,
                evidence={
                    "kind": "computer_accessibility_snapshot",
                    "pid": actual_pid,
                    "window_id": actual_window,
                    "match_count": len(elements),
                    "best_match_token": best_match["element_token"] if best_match else None,
                },
            )
        except (CuaDriverError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return ToolResult(
                success=False,
                name=self.name,
                content={"error": {"type": "computer_inspect_failed", "message": str(exc)}},
                metadata={
                    "recovery_hint": (
                        "Do not repeat the same inspection with broken window IDs. Omit pid/window_id to let "
                        "computer_inspect select the visible browser, or specify window_title. If no match is "
                        "found, report that result instead of reading a nonexistent snapshot file."
                    )
                },
                summary=f"Computer inspection failed: {exc}",
            )
