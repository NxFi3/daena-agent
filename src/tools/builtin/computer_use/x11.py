"""Optional Linux/X11 adapter. Explicit foreground input, no silent Wayland fallback."""

import os
import math
from contextlib import closing
import shutil
import subprocess
import tempfile
import time
import uuid

from PIL import Image

from .backend import CuaBackend
from .contracts import Capture, Rect, Window
from .driver import _run_bounded
from .runtime import ComputerError, checkpoint


class X11Backend:
    name = "x11"

    @staticmethod
    def _wayland_session():
        session = os.environ.get("XDG_SESSION_TYPE", "").lower()
        return session == "wayland" or (
            session not in {"x11", "tty"} and bool(os.environ.get("WAYLAND_DISPLAY"))
        )

    def empty_window_error(self):
        if self._wayland_session():
            raise ComputerError(
                "native_wayland_unavailable",
                "This X11 backend cannot enumerate or control native Wayland windows. No eligible Xwayland windows were found; this does not mean the user's app is closed or uninstalled. Installing another app or retrying capture will not add native Wayland support. Use a supported native Wayland backend, or log out and select an X11/Xorg desktop session before restarting Daena. Setting DAENA_COMPUTER_BACKEND=x11 does not change the desktop session.",
            )

    def capabilities(self, session=None, window=None):
        return {
            "backend": self.name,
            "actions": [
                "click",
                "double_click",
                "right_click",
                "move",
                "type_text",
                "press_key",
                "hotkey",
                "scroll",
                "drag",
            ],
            "coordinate_frames": ["window", "desktop"],
            "delivery_modes": {"foreground": True, "background": False},
            "preferred_delivery_mode": "foreground",
            "accessibility": False,
            "pointer_move": True,
            "drag_path": True,
            "window_scope": "Xwayland clients only"
            if self._wayland_session()
            else "X11 clients",
            "native_wayland": False,
            "known_delivery_routes": [],
        }

    def preflight(self, action, arguments):
        if action == "drag" and arguments.get("path") is not None:
            self._validate_drag_path(arguments["path"])
        if not shutil.which("xdotool"):
            raise ComputerError(
                "dependency_missing",
                "Install xdotool for X11 input. Window discovery and screenshots remain available.",
            )
        if arguments.get("delivery_mode") != "foreground":
            raise ComputerError(
                "background_unavailable",
                "X11 requires explicit foreground delivery; no input was sent.",
                not_dispatched=True,
            )

    def ready(self, session):
        if not os.environ.get("DISPLAY"):
            raise ComputerError(
                "display_missing",
                "X11 requires DISPLAY. Native Wayland requires a backend that supports your compositor or an X11/Xorg login session.",
            )
        try:
            from Xlib.display import Display

            with closing(Display()) as display:
                display.sync()
        except ImportError as exc:
            raise ComputerError(
                "dependency_missing", "Install the x11 extra (python-xlib)."
            ) from exc
        except Exception as exc:
            raise ComputerError(
                "display_unavailable", f"Cannot connect to X11: {exc}"
            ) from exc

    @staticmethod
    def _local_client_pid(display, handle):
        """Ask XRes for identity when a client omits optional EWMH PID metadata."""
        try:
            from Xlib.ext import res

            if not display.has_extension("X-Resource"):
                return None
            reply = display.res_query_client_ids(
                [{"client": handle.id, "mask": res.LocalClientPIDMask}]
            )
            for value in reply.ids:
                if value.spec.mask & res.LocalClientPIDMask and value.value:
                    pid = int(value.value[0])
                    if 0 < pid < 2**31:
                        return pid
        except Exception:
            pass
        return None

    def windows(self, session):
        from Xlib import X
        from Xlib.display import Display

        rows = []
        with closing(Display()) as display:
            root = display.screen().root
            listing = root.get_full_property(
                display.intern_atom("_NET_CLIENT_LIST"), X.AnyPropertyType
            )
            handles = (
                [
                    display.create_resource_object("window", int(i))
                    for i in listing.value
                ]
                if listing is not None
                else []
            )
            self.window_diagnostics = {
                "scope": "Xwayland clients only"
                if self._wayland_session()
                else "X11 clients",
                "session_type": os.environ.get("XDG_SESSION_TYPE", "unknown"),
                "wayland_display_configured": bool(os.environ.get("WAYLAND_DISPLAY")),
                "missing_identity": 0,
                "source": "ewmh" if handles else "window_tree",
            }
            if not handles:
                # Window managers can reparent clients beneath decoration windows;
                # looking at root children alone silently loses those applications.
                pending = list(root.query_tree().children)
                seen = set()
                while pending and len(seen) < 4096:
                    handle = pending.pop(0)
                    if handle.id in seen:
                        continue
                    seen.add(handle.id)
                    try:
                        attrs = handle.get_attributes()
                        if attrs.map_state != X.IsViewable or attrs.override_redirect:
                            continue
                        if (
                            handle.get_wm_class()
                            or handle.get_wm_name()
                            or handle.get_full_property(
                                display.intern_atom("WM_STATE"), X.AnyPropertyType
                            )
                        ):
                            handles.append(handle)
                        pending.extend(handle.query_tree().children)
                    except Exception:
                        continue
                self.window_diagnostics["tree_truncated"] = bool(pending)
            active_prop = root.get_full_property(
                display.intern_atom("_NET_ACTIVE_WINDOW"), X.AnyPropertyType
            )
            focused = (
                int(active_prop.value[0])
                if active_prop is not None and len(active_prop.value)
                else getattr(display.get_input_focus().focus, "id", None)
            )
            for handle in handles:
                try:
                    if handle.get_attributes().map_state != X.IsViewable:
                        continue
                    pid = handle.get_full_property(
                        display.intern_atom("_NET_WM_PID"), X.AnyPropertyType
                    )
                    process_id = (
                        int(pid.value[0])
                        if pid is not None and len(pid.value)
                        else self._local_client_pid(display, handle)
                    )
                    if process_id is None or process_id <= 0:
                        self.window_diagnostics["missing_identity"] += 1
                        continue
                    title = handle.get_wm_name() or ""
                    name = handle.get_full_property(
                        display.intern_atom("_NET_WM_NAME"),
                        display.intern_atom("UTF8_STRING"),
                    )
                    if name:
                        title = bytes(name.value).decode("utf-8", "replace")
                    cls = handle.get_wm_class() or ("", "")
                    geometry = handle.get_geometry()
                    origin = root.translate_coords(handle, 0, 0)
                    rows.append(
                        Window(
                            process_id,
                            handle.id,
                            str(title),
                            str(cls[-1]),
                            Rect(origin.x, origin.y, geometry.width, geometry.height),
                            handle.id == focused,
                        )
                    )
                except Exception:
                    # Windows can close while enumerating; no substitute is chosen.
                    continue
        return rows

    def capture(self, session, window, output, *, pixels=True, accessibility=True):
        if window is None and self._wayland_session():
            raise ComputerError(
                "native_wayland_unavailable",
                "An Xwayland root screenshot is not a capture of the Wayland desktop. Use a supported native capture backend or an X11/Xorg desktop session. An explicitly selected Xwayland window can still be captured.",
            )
        from Xlib import X
        from Xlib.display import Display

        if not pixels:
            raise ComputerError(
                "unsupported",
                "The X11 fallback is pixel-only; use Cua for accessibility.",
            )
        with closing(Display()) as display:
            screen = display.screen()
            if window:
                current = next(
                    (w for w in self.windows(session) if window.same(w)), None
                )
                if current is None:
                    raise ComputerError(
                        "window_missing", "The target window closed before capture."
                    )
                window = current
                drawable = display.create_resource_object(
                    "window", int(window.window_id)
                )
            else:
                drawable = screen.root
            geometry = drawable.get_geometry()
            if geometry.width * geometry.height > 32_000_000:
                raise ComputerError(
                    "image_too_large", "X11 capture exceeds the pixel limit."
                )
            try:
                raw = drawable.get_image(
                    0, 0, geometry.width, geometry.height, X.ZPixmap, 0xFFFFFFFF
                )
                if raw.depth not in {24, 32}:
                    raise ComputerError(
                        "unsupported_pixel_format",
                        f"X11 depth {raw.depth} is unsupported.",
                    )
                image = Image.frombytes(
                    "RGB", (geometry.width, geometry.height), raw.data, "raw", "BGRX"
                )
                image.save(output, "PNG")
            except ComputerError:
                raise
            except Exception as exc:
                raise ComputerError(
                    "capture_failed", f"X11 capture failed: {exc}"
                ) from exc
        return Capture(
            str(output),
            "x11-" + uuid.uuid4().hex,
            window,
            "window" if window else "desktop",
            warnings=[
                "X11 fallback has no accessibility tree. Window pixels may be occlusion-sensitive; keep the target visible."
            ],
            truncated=False,
        )

    def _run(self, arguments):
        executable = shutil.which("xdotool")
        if not executable:
            raise ComputerError("dependency_missing", "xdotool is unavailable.")
        try:
            result = _run_bounded([executable, *map(str, arguments)], 20)
        except subprocess.TimeoutExpired as exc:
            raise ComputerError(
                "input_timeout",
                "X11 input timed out; outcome is unknown.",
                dispatched=True,
            ) from exc
        if result.returncode:
            raise ComputerError(
                "input_failed",
                (result.stderr or "xdotool failed")[:800],
                dispatched=True,
            )
        return result.stdout

    @staticmethod
    def _validate_drag_path(path):
        if (
            not isinstance(path, list)
            or not 2 <= len(path) <= 256
            or any(
                not isinstance(point, (list, tuple))
                or len(point) != 2
                or any(
                    isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or not math.isfinite(v)
                    for v in point
                )
                for point in path
            )
        ):
            raise ComputerError(
                "invalid_arguments",
                "drag path needs 2–256 finite [x,y] points.",
                not_dispatched=True,
            )
        return path

    def _drag(self, args, move, button):
        path = args.get("path")
        if path is None:
            path = [[args["from_x"], args["from_y"]], [args["to_x"], args["to_y"]]]
        self._validate_drag_path(path)
        duration = args["duration_ms"] / 1000
        lengths = [math.dist(start, end) for start, end in zip(path, path[1:])]
        total_length = sum(lengths)
        move(*path[0])
        try:
            # A failed/timeout acknowledgement may still have pressed the
            # button. Keep even that first dispatch inside cleanup coverage.
            self._run(["mousedown", button])
            for start, end, length in zip(path, path[1:], lengths):
                segment_duration = (
                    duration * length / total_length
                    if total_length
                    else duration / len(lengths)
                )
                steps = max(1, min(600, round(segment_duration * 60)))
                for i in range(1, steps + 1):
                    checkpoint()
                    t = i / steps
                    move(
                        start[0] + (end[0] - start[0]) * t,
                        start[1] + (end[1] - start[1]) * t,
                    )
                    time.sleep(segment_duration / steps)
        finally:
            # Releasing is mandatory even after cancellation. It is a new
            # bounded cleanup helper, not an input retry or continuation.
            executable = shutil.which("xdotool")
            try:
                released = subprocess.run(
                    [executable, "mouseup", str(button)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
                if getattr(released, "returncode", 0):
                    raise OSError("xdotool rejected mouseup")
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ComputerError(
                    "button_release_failed",
                    "Drag cleanup could not confirm mouseup; the button may still be held. Stop input and inspect the desktop before continuing.",
                    dispatched=True,
                ) from exc

    def dispatch(self, session, action, arguments):
        checkpoint()
        args = dict(arguments)
        if args.get("delivery_mode") != "foreground":
            raise ComputerError(
                "background_unavailable",
                "X11 input requires delivery_mode='foreground'; no input was sent.",
                not_dispatched=True,
            )
        window = None
        if args.get("window_id") is not None:
            window = next(
                (
                    w
                    for w in self.windows(session)
                    if w.pid == args.get("pid")
                    and str(w.window_id) == str(args["window_id"])
                ),
                None,
            )
            if window is None:
                raise ComputerError(
                    "window_missing", "The selected X11 window is gone."
                )
            self._run(
                [
                    "windowraise",
                    window.window_id,
                    "windowfocus",
                    "--sync",
                    window.window_id,
                ]
            )

        def move(x, y):
            if args.get("coordinate_frame") == "window":
                if window is None:
                    raise ComputerError(
                        "invalid_target", "Window-local input needs a native window."
                    )
                x, y = x + window.bounds.x, y + window.bounds.y
            # xdotool --sync waits for a movement event even when the pointer
            # is already at this point (common for consecutive drawing strokes).
            # A server acknowledgement also handles no-op moves without that wait.
            from Xlib import X
            from Xlib.display import Display
            from Xlib.ext import xtest

            with closing(Display()) as display:
                xtest.fake_input(display, X.MotionNotify, x=round(x), y=round(y))
                display.sync()

        button = {"left": 1, "middle": 2, "right": 3}[args.get("button", "left")]
        if "x" in args:
            move(args["x"], args["y"])
        if action in {"click", "double_click", "right_click"}:
            button = 3 if action == "right_click" else button
            self._run(
                [
                    "click",
                    "--repeat",
                    2 if action == "double_click" else 1,
                    "--delay",
                    100,
                    button,
                ]
            )
        elif action == "move":
            pass
        elif action == "type_text":
            if "x" in args:
                self._run(["click", 1])
            from .x11_text import type_stable

            def send(value):
                # A private file keeps text out of process command-line arguments.
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as text:
                    text.write(value)
                    text.flush()
                    self._run(
                        ["type", "--clearmodifiers", "--delay", 25, "--file", text.name]
                    )

            type_stable(args["text"], window.window_id if window else None, send)
        elif action in {"press_key", "hotkey"}:
            keys = args.get("keys") or [args["key"]]
            aliases = {
                "enter": "Return",
                "return": "Return",
                "esc": "Escape",
                "escape": "Escape",
                "ctrl": "ctrl",
                "control": "ctrl",
                "backspace": "BackSpace",
                "tab": "Tab",
                "left": "Left",
                "right": "Right",
                "up": "Up",
                "down": "Down",
                "delete": "Delete",
            }
            chord = "+".join(aliases.get(k.casefold(), k) for k in keys)
            self._run(["key", "--clearmodifiers", "--", chord])
        elif action == "scroll":
            button = {"up": 4, "down": 5, "left": 6, "right": 7}[args["direction"]]
            self._run(["click", "--repeat", args["amount"], "--delay", 50, button])
        elif action == "drag":
            self._drag(args, move, button)
        else:
            raise ComputerError("unsupported", f"X11 action {action!r} is unsupported.")
        return {
            "status": "dispatched",
            "backend": self.name,
            "delivery_mode": "foreground",
        }

    def doctor(self):
        return {
            "backend": self.name,
            "installed": bool(shutil.which("xdotool")),
            "display_configured": bool(os.environ.get("DISPLAY")),
            "native_desktop_supported": not self._wayland_session(),
            "whole_desktop_capture_supported": not self._wayland_session(),
            "window_scope": "Xwayland clients only"
            if self._wayland_session()
            else "X11 clients",
            "changing_backend_changes_desktop_session": False,
            "accessibility": False,
            "background_input": False,
            "pointer_move": True,
            "capabilities": self.capabilities(),
            "session_type": os.environ.get("XDG_SESSION_TYPE", "unknown"),
            "wayland_display_configured": bool(os.environ.get("WAYLAND_DISPLAY")),
            "window_diagnostics": getattr(self, "window_diagnostics", {}),
            "next_step": (
                "Native Wayland control is unsupported by this X11 backend. Select a supported native backend or log out and select an X11/Xorg desktop session; an environment variable does not change the login session."
                if self._wayland_session()
                else "Select foreground delivery explicitly. Read the actual session_type and window_diagnostics; an empty list does not prove no apps are open or installed."
            ),
        }

    launch = CuaBackend.launch
