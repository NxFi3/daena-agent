"""Optional Linux/X11 adapter. Explicit foreground input, no silent Wayland fallback."""

import os
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

    def preflight(self, action, arguments):
        if arguments.get("delivery_mode") != "foreground":
            raise ComputerError(
                "background_unavailable",
                "X11 requires explicit foreground delivery; no input was sent.",
            )

    def ready(self, session):
        if not os.environ.get("DISPLAY"):
            raise ComputerError(
                "display_missing", "X11 requires DISPLAY. Use Cua Driver for native Wayland."
            )
        if not shutil.which("xdotool"):
            raise ComputerError("dependency_missing", "Install xdotool for X11 input.")
        try:
            from Xlib.display import Display

            with closing(Display()) as display:
                display.sync()
        except ImportError as exc:
            raise ComputerError(
                "dependency_missing", "Install the x11 extra (python-xlib)."
            ) from exc
        except Exception as exc:
            raise ComputerError("display_unavailable", f"Cannot connect to X11: {exc}") from exc

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
                [display.create_resource_object("window", int(i)) for i in listing.value]
                if listing
                else root.query_tree().children
            )
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
                    if pid is None or not len(pid.value):
                        continue
                    title = handle.get_wm_name() or ""
                    name = handle.get_full_property(
                        display.intern_atom("_NET_WM_NAME"), display.intern_atom("UTF8_STRING")
                    )
                    if name:
                        title = bytes(name.value).decode("utf-8", "replace")
                    cls = handle.get_wm_class() or ("", "")
                    geometry = handle.get_geometry()
                    origin = root.translate_coords(handle, 0, 0)
                    rows.append(
                        Window(
                            int(pid.value[0]),
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
        from Xlib import X
        from Xlib.display import Display

        if not pixels:
            raise ComputerError(
                "unsupported", "The X11 fallback is pixel-only; use Cua for accessibility."
            )
        with closing(Display()) as display:
            screen = display.screen()
            if window:
                current = next((w for w in self.windows(session) if window.same(w)), None)
                if current is None:
                    raise ComputerError(
                        "window_missing", "The target window closed before capture."
                    )
                window = current
                drawable = display.create_resource_object("window", int(window.window_id))
            else:
                drawable = screen.root
            geometry = drawable.get_geometry()
            if geometry.width * geometry.height > 32_000_000:
                raise ComputerError("image_too_large", "X11 capture exceeds the pixel limit.")
            try:
                raw = drawable.get_image(
                    0, 0, geometry.width, geometry.height, X.ZPixmap, 0xFFFFFFFF
                )
                if raw.depth not in {24, 32}:
                    raise ComputerError(
                        "unsupported_pixel_format", f"X11 depth {raw.depth} is unsupported."
                    )
                image = Image.frombytes(
                    "RGB", (geometry.width, geometry.height), raw.data, "raw", "BGRX"
                )
                image.save(output, "PNG")
            except ComputerError:
                raise
            except Exception as exc:
                raise ComputerError("capture_failed", f"X11 capture failed: {exc}") from exc
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
                "input_timeout", "X11 input timed out; outcome is unknown.", dispatched=True
            ) from exc
        if result.returncode:
            raise ComputerError(
                "input_failed", (result.stderr or "xdotool failed")[:800], dispatched=True
            )
        return result.stdout

    def dispatch(self, session, action, arguments):
        checkpoint()
        args = dict(arguments)
        if args.get("delivery_mode") != "foreground":
            raise ComputerError(
                "background_unavailable",
                "X11 input requires delivery_mode='foreground'; no input was sent.",
            )
        window = None
        if args.get("window_id") is not None:
            window = next(
                (
                    w
                    for w in self.windows(session)
                    if w.pid == args.get("pid") and str(w.window_id) == str(args["window_id"])
                ),
                None,
            )
            if window is None:
                raise ComputerError("window_missing", "The selected X11 window is gone.")
            self._run(["windowraise", window.window_id, "windowfocus", "--sync", window.window_id])

        def move(x, y):
            if args.get("coordinate_frame") == "window":
                if window is None:
                    raise ComputerError(
                        "invalid_target", "Window-local input needs a native window."
                    )
                x, y = x + window.bounds.x, y + window.bounds.y
            self._run(["mousemove", "--sync", round(x), round(y)])

        button = {"left": 1, "middle": 2, "right": 3}[args.get("button", "left")]
        if "x" in args:
            move(args["x"], args["y"])
        if action in {"click", "double_click", "right_click"}:
            button = 3 if action == "right_click" else button
            self._run(
                ["click", "--repeat", 2 if action == "double_click" else 1, "--delay", 100, button]
            )
        elif action == "move":
            pass
        elif action == "type_text":
            if "x" in args:
                self._run(["click", 1])
            # A private file avoids placing typed text in the process command line.
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as text:
                text.write(args["text"])
                text.flush()
                self._run(["type", "--clearmodifiers", "--delay", 25, "--file", text.name])
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
            move(args["from_x"], args["from_y"])
            self._run(["mousedown", button])
            try:
                duration = args["duration_ms"] / 1000
                steps = max(2, min(120, round(duration * 60)))
                for i in range(1, steps + 1):
                    checkpoint()
                    t = i / steps
                    move(
                        args["from_x"] + (args["to_x"] - args["from_x"]) * t,
                        args["from_y"] + (args["to_y"] - args["from_y"]) * t,
                    )
                    time.sleep(duration / steps)
            finally:
                # Release even when the operation's cancellation/deadline is set.
                executable = shutil.which("xdotool")
                subprocess.run(
                    [executable, "mouseup", str(button)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
        else:
            raise ComputerError("unsupported", f"X11 action {action!r} is unsupported.")
        return {"status": "dispatched", "backend": self.name, "delivery_mode": "foreground"}

    def doctor(self):
        return {
            "backend": self.name,
            "installed": bool(shutil.which("xdotool")),
            "display_configured": bool(os.environ.get("DISPLAY")),
            "accessibility": False,
            "background_input": False,
            "pointer_move": True,
            "next_step": "Select foreground delivery explicitly. Native Wayland needs the cua backend.",
        }

    launch = CuaBackend.launch
