"""Backend contract and the Cua Driver adapter. No UI/task inference lives here."""

from pathlib import Path
import json
import os
import shutil
import subprocess
import uuid
from typing import Protocol

from .contracts import Capture, Rect, Window
from . import driver
from .runtime import ComputerError, checkpoint


class Backend(Protocol):
    name: str

    def ready(self, session: str) -> None: ...
    def windows(self, session: str) -> list[Window]: ...
    def capture(
        self,
        session: str,
        window: Window | None,
        output: Path,
        *,
        pixels: bool,
        accessibility: bool,
    ) -> Capture: ...
    def dispatch(self, session: str, action: str, arguments: dict) -> dict: ...
    def doctor(self) -> dict: ...
    def capabilities(
        self, session: str | None = None, window: Window | None = None
    ) -> dict: ...


def decode(raw):
    try:
        result = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ComputerError(
            "invalid_driver_response", "Expected a complete JSON driver response."
        ) from exc
    refusal = driver._driver_refusal_detail(raw)
    if refusal:
        raise driver.response_error(refusal[:1000], raw)
    return result


def parse_window(row):
    try:
        pid = int(row["pid"])
        wid = row["window_id"]
        if (
            isinstance(row["pid"], bool)
            or isinstance(wid, bool)
            or pid <= 0
            or not str(wid)
        ):
            return None
    except (ValueError, TypeError, KeyError):
        return None
    return Window(
        pid,
        wid,
        str(row.get("title") or row.get("window_title") or ""),
        str(row.get("app_name") or row.get("app") or ""),
        Rect.parse(row.get("window_bounds") or row.get("bounds") or row),
        row.get("is_active") is True
        or row.get("active") is True
        or row.get("is_focused") is True,
    )


class CuaBackend:
    name = "cua"

    def __init__(self):
        self._delivery_routes = {}

    @staticmethod
    def _route(session, action, arguments):
        target_kind = (
            "element"
            if arguments.get("element_token")
            else "coordinate"
            if "x" in arguments or action == "drag"
            else "focused"
        )
        return (
            str(session),
            arguments.get("pid"),
            str(arguments.get("window_id")),
            action,
            target_kind,
            arguments.get("delivery_mode", "background"),
        )

    def _remember_route(self, session, action, arguments, *, supported, code=None):
        key = self._route(session, action, arguments)
        self._delivery_routes[key] = {
            "action": action,
            "target_kind": key[-2],
            "delivery_mode": key[-1],
            "supported": supported,
            **({"code": code} if code else {}),
        }
        while len(self._delivery_routes) > 64:
            self._delivery_routes.pop(next(iter(self._delivery_routes)))

    def capabilities(self, session=None, window=None):
        # Declarations and observed transport routes are distinct. A success
        # attests delivery support, not application progress or goal completion.
        known = [
            dict(value)
            for key, value in self._delivery_routes.items()
            if (session is None or key[0] == str(session))
            and (
                window is None
                or (key[1] == window.pid and key[2] == str(window.window_id))
            )
        ]
        return {
            "backend": self.name,
            "actions": sorted(
                driver.DRIVER_TOOLS
                - {
                    "start_session",
                    "get_desktop_state",
                    "get_cursor_position",
                    "get_window_state",
                    "list_windows",
                    "list_apps",
                    "bring_to_front",
                }
            ),
            "coordinate_frames": ["window", "desktop"],
            "delivery_modes": {
                "foreground": "driver-dependent",
                "background": "driver-dependent",
            },
            "preferred_delivery_mode": "foreground",
            "accessibility": True,
            "pointer_move": False,
            "drag_path": False,
            "window_scope": "driver-dependent",
            "native_wayland": "driver-dependent",
            "known_delivery_routes": known[-8:],
        }

    def preflight(self, action, arguments):
        if action == "drag" and arguments.get("path") is not None:
            raise ComputerError(
                "unsupported",
                "This Cua adapter supports endpoint drag only, not a continuous multi-point stroke. Choose a backend declaring drag_path support.",
                not_dispatched=True,
            )
        if action == "move":
            raise ComputerError(
                "unsupported",
                "The installed Cua adapter has no documented pointer-move action.",
            )

    def prepare_action(self, session, action, arguments, snapshot):
        if action != "drag":
            return
        # Drag has no capture_id field in Cua's documented contract. Re-prime
        # pixels without reading AX; do not replay a failed gesture.
        from .observe import image_info

        window = parse_window(snapshot["window"]) if snapshot.get("window") else None
        path = Path(snapshot["raw_image_ref"]).with_name(
            ".drag-prime-" + uuid.uuid4().hex + ".png"
        )
        try:
            cap = self.capture(session, window, path, pixels=True, accessibility=False)
            if image_info(cap.image_path).get("pixel_hash") != snapshot.get(
                "pixel_hash"
            ):
                raise ComputerError(
                    "stale_snapshot",
                    "The screen changed before drag preparation. Capture again and choose the current endpoints.",
                )
        finally:
            path.unlink(missing_ok=True)

    def ready(self, session):
        driver.ensure_driver_running(session)

    def windows(self, session):
        data = decode(
            driver.call_driver(
                "list_windows", {"session": session, "on_screen_only": True}, timeout=12
            )
        )
        rows = data.get("windows") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ComputerError(
                "invalid_driver_response", "list_windows did not return a window list."
            )
        return [
            w
            for row in rows
            if isinstance(row, dict) and row.get("is_on_screen") is not False
            if (w := parse_window(row)) is not None
        ]

    def capture(self, session, window, output, *, pixels=True, accessibility=True):
        args = {"session": session}
        if window:
            args.update(
                pid=window.pid,
                window_id=window.window_id,
                include_accessibility_tree=accessibility,
                include_screenshot=pixels,
                max_elements=5000,
                max_depth=20,
                timeout_ms=9000,
            )
            action = "get_window_state"
        else:
            action = "get_desktop_state"
            if not pixels:
                raise ComputerError(
                    "unsupported",
                    "Desktop AX-only capture is unavailable. Select a native window.",
                )
        if pixels:
            args["screenshot_out_file"] = str(output)
        payload = decode(driver.call_driver(action, args, timeout=20))
        if not isinstance(payload, dict):
            raise ComputerError(
                "invalid_driver_response", "Capture response must be an object."
            )
        if window:
            # Do not substitute a similar title or another window's screenshot.
            actual = parse_window(payload)
            if actual is None or not window.same(actual):
                raise ComputerError(
                    "window_mismatch",
                    "The capture did not attest the requested PID/window_id.",
                )
            window = Window(
                window.pid,
                window.window_id,
                actual.title or window.title,
                actual.app or window.app,
                actual.bounds or window.bounds,
                window.active,
            )
        if pixels and not output.is_file():
            raise ComputerError(
                "screenshot_missing",
                "The driver did not write the requested screenshot file.",
            )
        capture_id = payload.get("capture_id")
        if pixels and (not isinstance(capture_id, str) or not capture_id.strip()):
            raise ComputerError(
                "capture_id_missing", "The driver returned pixels without a capture ID."
            )
        frame = str(
            payload.get("accessibility_coordinate_frame")
            or payload.get("coordinate_frame")
            or ""
        )
        frame = {
            "window-local": "window",
            "screen": "desktop",
            "screen-space": "desktop",
        }.get(frame, frame)
        elements = payload.get("elements", []) if accessibility else []
        warnings = []
        if accessibility and not isinstance(elements, list):
            warnings.append(
                "The driver returned no structured accessibility element list."
            )
            elements = []
        if accessibility and not elements:
            warnings.append(
                "Accessibility is empty; use visible pixels or inspect another window."
            )
        if frame not in {"window", "desktop"}:
            frame = None
            if elements:
                warnings.append(
                    "Accessibility coordinate frame is undeclared; element bounds are not used for pixel input."
                )
        return Capture(
            str(output) if pixels else None,
            capture_id,
            window,
            "window" if window else "desktop",
            elements,
            frame,
            warnings,
            bool(payload.get("truncated", False)),
        )

    def dispatch(self, session, action, arguments):
        if action == "move":
            raise ComputerError(
                "unsupported", "This Cua adapter has no documented pointer-move action."
            )
        self.preflight(action, arguments)
        checkpoint()
        # No retry on background refusal, timeout, partial typing, or unknown errors.
        try:
            raw = driver.call_driver(
                action, {"session": session, **arguments}, timeout=20
            )
            if raw.lstrip().startswith(("{", "[")):
                decode(raw)
        except ComputerError as exc:
            if exc.code == "background_unavailable" and exc.not_dispatched:
                self._remember_route(
                    session, action, arguments, supported=False, code=exc.code
                )
            exc.backend_capabilities = self.capabilities(session)
            raise
        self._remember_route(session, action, arguments, supported=True)
        return {
            "status": "dispatched",
            "backend": self.name,
            "delivery_mode": arguments.get("delivery_mode", "background"),
        }

    def doctor(self):
        try:
            executable = driver.driver_executable()
            running, _ = driver._daemon_status(executable)
            return {
                "backend": self.name,
                "installed": True,
                "executable": executable,
                "daemon_running": running,
                "accessibility": True,
                "background_input": "driver-dependent",
                "pointer_move": False,
                "next_step": "cua-driver doctor",
                "capabilities": self.capabilities(),
            }
        except ComputerError as exc:
            return {
                "backend": self.name,
                "installed": False,
                "error": str(exc),
                "next_step": "Install Cua Driver and run cua-driver doctor in your desktop session.",
            }

    def launch(self, app):
        # App is a single executable or executable path, never a shell command.
        executable = shutil.which(app)
        if executable is None:
            raise ComputerError(
                "application_missing",
                f"Executable {app!r} is not on PATH. Pass one executable name, without shell arguments.",
            )
        process = subprocess.Popen(
            [executable],
            env=driver._desktop_session_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return {"pid": process.pid, "app": app, "status": "launched"}


def create_backend(name=None):
    name = name or os.environ.get("DAENA_COMPUTER_BACKEND", "cua")
    if name == "cua":
        return CuaBackend()
    if name == "x11":
        from .x11 import X11Backend

        return X11Backend()
    raise ComputerError(
        "unknown_backend", f"Unknown backend {name!r}; choose cua or x11."
    )
