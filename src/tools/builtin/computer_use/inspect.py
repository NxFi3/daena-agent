from __future__ import annotations

import json
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
                "description": "Optional visible-window title substring to select, e.g. 'Aparat'.",
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

    def _resolve_target(
        self,
        pid_value: Any,
        window_id_value: Any,
        window_title: str,
        query: str,
    ) -> tuple[int, int | None, str, str]:
        session = self._session_label()
        windows = self._windows()
        target_title = _normal(window_title)
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
                # The model often mistakes a file dialog for the browser in a screenshot.
                # For an explicit web-page query, recover to a visible browser instead of
                # silently inspecting/clicking the file manager.
                if (
                    not self._is_browser(selected)
                    and any(word in query_norm for word in ("aparat", "search", "جست"))
                ):
                    browser_rows = [
                        row for row in windows
                        if self._is_browser(row) and bool(row.get("is_on_screen", True))
                        and int(row.get("width") or 0) >= 600
                        and int(row.get("height") or 0) >= 400
                    ]
                    if browser_rows:
                        selected = self._choose_window(browser_rows, target_title, query_norm)
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

        if target_title:
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
        elif browser_rows:
            # If the task names a site, prefer its titled window when one is visible.
            site_terms = []
            if "aparat" in query_norm or "آپارات" in query_norm:
                site_terms = ["aparat", "آپارات"]
            site_matches = [
                row for row in browser_rows
                if any(term in _normal(row.get("title")) for term in site_terms)
            ]
            selected = self._choose_window(site_matches or browser_rows, "", query_norm)
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
            browser = 1 if any(name in app for name in _BROWSER_NAMES) else 0
            return hint_match + query_match + browser, area, int(window.get("z_index") or 0)
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
    def _rank_elements(payload: dict[str, Any], query: str, limit: int = 8) -> list[dict[str, Any]]:
        items = payload.get("elements", [])
        if not isinstance(items, list):
            return []
        items = [item for item in items if isinstance(item, dict)]
        # A browser process can expose controls from several tabs. Keep only nodes
        # descending from the exact top-level frame represented by this snapshot.
        by_index = {
            item.get("element_index"): item
            for item in items if isinstance(item.get("element_index"), int)
        }
        title = _normal(payload.get("window_title"))
        roots = {
            item["element_index"] for item in items
            if isinstance(item.get("element_index"), int)
            and _normal(item.get("role")) in {"frame", "window"}
            and _normal(item.get("label")) == title
        }

        def belongs_to_target_window(item: dict[str, Any]) -> bool:
            if not roots:
                return True
            current = item
            visited: set[int] = set()
            for _ in range(80):
                idx = current.get("element_index")
                if isinstance(idx, int) and idx in roots:
                    return True
                parent = current.get("parent_index")
                if not isinstance(parent, int) or parent in visited or parent not in by_index:
                    return False
                visited.add(parent)
                current = by_index[parent]
            return False

        query_norm = _normal(query)
        # AT-SPI can expose descendants from background browser tabs with large
        # negative/otherwise off-screen coordinates. A score penalty alone leaves
        # them selectable, so reject bounds that cannot be clicked in this window.
        window_bounds = payload.get("window_bounds")
        visible_rect = None
        if isinstance(window_bounds, dict):
            try:
                vx = float(window_bounds.get("x", 0))
                vy = float(window_bounds.get("y", 0))
                vw = float(window_bounds.get("width", 0))
                vh = float(window_bounds.get("height", 0))
                if vw > 0 and vh > 0:
                    visible_rect = (vx, vy, vx + vw, vy + vh)
            except (TypeError, ValueError):
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
            if isinstance(frame, dict):
                row["frame"] = {key: frame[key] for key in ("x", "y", "w", "h") if key in frame}
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
            target_pid, target_window_id, inventory_title, app_name = self._resolve_target(
                pid, window_id, window_title, query
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
