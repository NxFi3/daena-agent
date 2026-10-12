from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw
except ImportError:  # Screenshot capture must still work without optional annotation support.
    Image = ImageDraw = None

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool
from src.tools.builtin.computer_use.driver import (
    CuaDriverError, call_driver, png_dimensions, pointer_state_path, session_label_from_id,
)


class ComputerObserveTool(Tool):
    name = "computer_observe"
    action = "observe"
    description = (
        "Observe the current desktop or a specific native window through Cua Driver. When pid/window_id are "
        "provided, capture that exact window without activating it and return window-local coordinates; otherwise "
        "capture the full desktop. Save a real PNG under the active workspace's .daena/screenshots directory and "
        "return a persistent image reference for the next vision-capable model call. Use before choosing coordinates "
        "or verifying an action. When available, the saved image "
        "includes a red ring at Daena's last click and, when available, a separate marker for the cursor "
        "position reported by Cua Driver. The cursor marker includes the driver's source label (synthetic "
        "does not guarantee the physical user's pointer). The driver must be connected to the logged-in desktop."
    )
    parameters = {
        "type": "object",
        "properties": {
            "include_windows": {
                "type": "boolean",
                "default": True,
                "description": "Include a compact list of visible windows when available (full-desktop capture only).",
            },
            "pid": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional native PID to capture one specific window without activating it. Provide together with window_id.",
            },
            "window_id": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional native window ID from the latest visible-window inventory; provide together with pid.",
            }
        },
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
                raw = "default"
        return session_label_from_id(raw)

    def _overlay_last_agent_pointer(
        self, image_path: Path, *, expected_pid: int | None = None,
        expected_window_id: int | None = None,
    ) -> bool:
        """Mark the previous click, but never transfer screen coordinates across windows."""
        if Image is None or ImageDraw is None:
            return False
        state_path = pointer_state_path(self.workspace_root, self._session_label())
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if expected_pid is not None or expected_window_id is not None:
                target = state.get("last_agent_click_target")
                if not isinstance(target, dict):
                    return False
                if (
                    str(target.get("pid")) != str(expected_pid)
                    or str(target.get("window_id")) != str(expected_window_id)
                    or str(target.get("coordinate_frame") or "") != "window"
                ):
                    return False
            x, y = int(state["x"]), int(state["y"])
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            width, height = image.size
            if not (0 <= x < width and 0 <= y < height):
                return False
            draw = ImageDraw.Draw(image)
            # Red ring and white outer ring remain visible on both light and dark UI.
            draw.ellipse((x - 15, y - 15, x + 15, y + 15), outline=(255, 255, 255), width=5)
            draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline=(235, 45, 65), width=3)
            draw.line((x - 19, y, x - 7, y), fill=(255, 255, 255), width=3)
            draw.line((x + 7, y, x + 19, y), fill=(255, 255, 255), width=3)
            draw.line((x, y - 19, x, y - 7), fill=(255, 255, 255), width=3)
            draw.line((x, y + 7, x, y + 19), fill=(255, 255, 255), width=3)
            draw.line((x - 18, y, x - 8, y), fill=(235, 45, 65), width=2)
            draw.line((x + 8, y, x + 18, y), fill=(235, 45, 65), width=2)
            draw.line((x, y - 18, x, y - 8), fill=(235, 45, 65), width=2)
            draw.line((x, y + 8, x, y + 18), fill=(235, 45, 65), width=2)
            image.save(image_path, format="PNG")
            return True
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return False

    def _overlay_coordinate_grid(self, image_path: Path, spacing: int = 100) -> bool:
        """Draw labeled window-local coordinate lines for visually ambiguous UIs."""
        if Image is None or ImageDraw is None:
            return False
        spacing = max(50, min(200, int(spacing)))
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGBA")
            width, height = image.size
            grid = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(grid)

            for x in range(0, width, spacing):
                draw.line((x, 0, x, height - 1), fill=(0, 115, 150, 62), width=1)
                label = f"x={x}"
                bbox = draw.textbbox((x + 3, 2), label)
                draw.rounded_rectangle(
                    (bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1),
                    radius=2, fill=(8, 24, 32, 205),
                )
                draw.text((x + 3, 2), label, fill=(235, 250, 255, 255))

            for y in range(0, height, spacing):
                draw.line((0, y, width - 1, y), fill=(0, 115, 150, 62), width=1)
                label = f"y={y}"
                bbox = draw.textbbox((3, y + 3), label)
                draw.rounded_rectangle(
                    (bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1),
                    radius=2, fill=(8, 24, 32, 205),
                )
                draw.text((3, y + 3), label, fill=(235, 250, 255, 255))

            image = Image.alpha_composite(image, grid).convert("RGB")
            image.save(image_path, format="PNG")
            return True
        except (OSError, ValueError, TypeError):
            return False

    def _overlay_cursor_position(self, image_path: Path, position: dict[str, Any] | None) -> bool:
        """Annotate the driver-reported cursor without mislabelling synthetic positions as physical."""
        if Image is None or ImageDraw is None or not isinstance(position, dict):
            return False
        try:
            x_raw, y_raw = position.get("x"), position.get("y")
            if isinstance(x_raw, bool) or isinstance(y_raw, bool):
                return False
            x, y = int(round(float(x_raw))), int(round(float(y_raw)))
            source = str(position.get("source") or "unknown").strip().casefold()
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
            width, height = image.size
            if not (0 <= x < width and 0 <= y < height):
                return False
            draw = ImageDraw.Draw(image)
            # Cyan crosshair is intentionally different from the red last-click ring.
            draw.ellipse((x - 11, y - 11, x + 11, y + 11), outline=(255, 255, 255), width=4)
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(30, 220, 235), width=3)
            draw.line((x - 16, y, x - 5, y), fill=(30, 220, 235), width=2)
            draw.line((x + 5, y, x + 16, y), fill=(30, 220, 235), width=2)
            draw.line((x, y - 16, x, y - 5), fill=(30, 220, 235), width=2)
            draw.line((x, y + 5, x, y + 16), fill=(30, 220, 235), width=2)
            label = "CUA cursor" if source in {"physical", "native", "os"} else f"CUA cursor ({source})"
            try:
                box = draw.textbbox((x + 14, y + 12), label)
                draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(15, 30, 38))
            except AttributeError:
                pass
            draw.text((x + 14, y + 12), label, fill=(30, 220, 235))
            image.save(image_path, format="PNG")
            return True
        except (OSError, ValueError, TypeError, KeyError):
            return False

    def _execute_window_capture(self, pid: Any, window_id: Any) -> ToolResult:
        """Capture pixels from the exact native window, without activating it.

        This mirrors Hermes' key invariant: the pixels and accessibility snapshot
        refer to one PID/window pair. Cua Driver reports coordinates in this
        screenshot's window-local frame, even when another app is in front.
        """
        try:
            if isinstance(pid, bool) or isinstance(window_id, bool):
                raise ValueError("Window capture requires a real pid and window_id.")
            target_pid, target_window_id = int(pid), int(window_id)
            if target_pid <= 0 or target_window_id <= 0:
                raise ValueError("Window capture requires positive pid and window_id.")
        except (TypeError, ValueError, OverflowError) as exc:
            return ToolResult(
                success=False, name=self.name,
                content={"error": {"type": "invalid_window_target", "message": str(exc)}},
                summary=f"Window screenshot refused: {exc}",
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        session_label = self._session_label()
        session_dir = hashlib.sha256(session_label.encode("utf-8")).hexdigest()[:16]
        screenshot_dir = (self.workspace_root / ".daena" / "screenshots" / session_dir).resolve()
        image_path = (screenshot_dir / f"window_{timestamp}.png").resolve()
        try:
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            output = call_driver("get_window_state", {
                "session": session_label,
                "pid": target_pid,
                "window_id": target_window_id,
                "include_accessibility_tree": False,
                "include_screenshot": True,
                "screenshot_out_file": str(image_path),
            }, timeout=30)
            try:
                payload = json.loads(output)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise CuaDriverError(f"Cua Driver returned invalid window-capture JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise CuaDriverError("Cua Driver returned no structured window-capture metadata.")
            if payload.get("screenshot_frame_valid") is False:
                detail = payload.get("screenshot_error") or payload.get("frame_note") or "the driver could not verify the screenshot frame"
                raise CuaDriverError(f"Window screenshot frame is unavailable: {detail}")
            if not image_path.is_file() or image_path.stat().st_size <= 24:
                reported = payload.get("screenshot_file_path")
                raise CuaDriverError(
                    "Cua Driver did not write a readable screenshot for the selected window"
                    + (f" (reported path: {reported})" if reported else "")
                )

            width, height = png_dimensions(image_path)
            width = int(payload.get("screenshot_width") or width or 0)
            height = int(payload.get("screenshot_height") or height or 0)
            if width <= 0 or height <= 0:
                raise CuaDriverError("The selected-window screenshot has no verified dimensions.")
            screenshot_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
            capture_id = payload.get("capture_id")
            if not isinstance(capture_id, str) or not capture_id.strip():
                raise CuaDriverError("Cua Driver did not provide a capture_id for the window screenshot.")

            state_path = pointer_state_path(self.workspace_root, session_label)
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if not isinstance(state, dict):
                    state = {}
            except (OSError, ValueError, TypeError):
                state = {}
            state.update({
                "latest_capture_id": capture_id,
                "latest_screenshot_hash": screenshot_hash,
                "latest_screenshot_coordinate_frame": "window",
            })
            try:
                state_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = state_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(state), encoding="utf-8")
                temporary.replace(state_path)
            except OSError:
                pass

            # Mark the prior click only when it belongs to this exact native
            # window. The coordinate grid is added by ComputerUse when the AX
            # tree cannot expose enough controls to guide the model.
            pointer_overlay = self._overlay_last_agent_pointer(
                image_path, expected_pid=target_pid, expected_window_id=target_window_id,
            )
            visual_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()

            raw_bounds = payload.get("window_bounds")
            bounds = raw_bounds if isinstance(raw_bounds, dict) else {}
            window = {
                "app": str(payload.get("app_name") or ""),
                "app_name": str(payload.get("app_name") or ""),
                "title": str(payload.get("window_title") or payload.get("app_name") or ""),
                "pid": target_pid,
                "window_id": target_window_id,
                "bounds": {
                    "x": bounds.get("x", 0), "y": bounds.get("y", 0),
                    "width": bounds.get("width", width), "height": bounds.get("height", height),
                },
            }
            size_bytes = image_path.stat().st_size
            captured_at = datetime.now(timezone.utc).isoformat()
            content = {
                "image_ref": str(image_path),
                "image_mime_type": str(payload.get("screenshot_mime_type") or "image/png"),
                "width": width, "height": height, "size_bytes": size_bytes,
                "screenshot_hash": screenshot_hash, "visual_hash": visual_hash,
                "pointer_overlay": "last_agent_click" if pointer_overlay else None,
                "capture_id": capture_id, "coordinate_frame": "window", "window": window,
                "windows": "[]",
            }
            return ToolResult(
                success=True, name=self.name, content=content,
                metadata={
                    "kind": "screen_observation", "image_ref": str(image_path),
                    "image_mime_type": content["image_mime_type"], "width": width, "height": height,
                    "size_bytes": size_bytes, "screenshot_hash": screenshot_hash,
                    "visual_hash": visual_hash,
                    "pointer_overlay": "last_agent_click" if pointer_overlay else None,
                    "coordinate_frame": "window", "capture_id": capture_id,
                    "window": window, "captured_at": captured_at,
                },
                summary=(
                    f"Captured {window['title'] or 'the selected window'} directly "
                    f"({width}x{height}); coordinates are window-local. The window was not activated."
                    + (" Previous click is marked with a red ring." if pointer_overlay else "")
                ),
                evidence={
                    "kind": "screen_observation", "image_ref": str(image_path),
                    "width": width, "height": height, "capture_id": capture_id,
                    "coordinate_frame": "window", "window": window,
                },
            )
        except (OSError, CuaDriverError, TypeError, ValueError) as exc:
            return ToolResult(
                success=False, name=self.name,
                content={"error": {"type": "window_capture_failed", "message": str(exc)}},
                metadata={"recovery_hint": (
                    "The exact-window screenshot could not be verified. Do not pair its controls with a desktop image "
                    "or guess coordinates. Check the returned driver error; use an explicitly verified desktop capture only as a recovery."
                )},
                summary=f"Could not capture the selected window without switching focus: {exc}",
            )

    def execute(self, include_windows: bool = True, **_kwargs: Any) -> ToolResult:
        pid, window_id = _kwargs.get("pid"), _kwargs.get("window_id")
        if pid is not None or window_id is not None:
            if pid is None or window_id is None:
                return ToolResult(
                    success=False, name=self.name,
                    content={"error": {"type": "invalid_window_target", "message": "Window capture requires both pid and window_id."}},
                    summary="Window screenshot refused because the target was incomplete.",
                )
            return self._execute_window_capture(pid, window_id)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        session_label = self._session_label()
        session_dir = hashlib.sha256(session_label.encode("utf-8")).hexdigest()[:16]
        screenshot_dir = (self.workspace_root / ".daena" / "screenshots" / session_dir).resolve()
        image_path = (screenshot_dir / f"screen_{timestamp}.png").resolve()

        try:
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            output = call_driver("get_desktop_state", {
                "session": session_label,
                "screenshot_out_file": str(image_path),
            })
            if not image_path.is_file() or image_path.stat().st_size <= 24:
                return ToolResult(
                    success=False,
                    name=self.name,
                    content={"error": {"type": "screenshot_missing",
                        "message": f"Cua Driver returned without a readable screenshot file. Response: {output[:500]}"}},
                    summary="Desktop observation failed: screenshot was not saved.",
                )

            # Keep the capture ID produced by CUA, plus a digest of the raw PNG.
            # Actions can then bind coordinates to this exact screenshot instead
            # of using a stale or ambiguous coordinate frame.
            try:
                driver_payload = json.loads(output)
                capture_id = driver_payload.get("capture_id") if isinstance(driver_payload, dict) else None
            except (json.JSONDecodeError, TypeError, ValueError):
                capture_id = None
            try:
                state_path = pointer_state_path(self.workspace_root, session_label)
                pointer_state = json.loads(state_path.read_text(encoding="utf-8"))
                if not isinstance(pointer_state, dict):
                    pointer_state = {}
            except (OSError, ValueError, TypeError):
                state_path = pointer_state_path(self.workspace_root, session_label)
                pointer_state = {}
            screenshot_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
            try:
                pointer_state["latest_capture_id"] = capture_id
                pointer_state["latest_screenshot_hash"] = screenshot_hash
                state_path.parent.mkdir(parents=True, exist_ok=True)
                temp_state_path = state_path.with_suffix(".tmp")
                temp_state_path.write_text(json.dumps(pointer_state), encoding="utf-8")
                temp_state_path.replace(state_path)
            except OSError:
                pass

            cursor_position: dict[str, Any] | None = None
            try:
                cursor_payload = json.loads(call_driver(
                    "get_cursor_position", {"session": session_label}, timeout=5
                ))
                if (
                    isinstance(cursor_payload, dict)
                    and isinstance(cursor_payload.get("x"), (int, float))
                    and isinstance(cursor_payload.get("y"), (int, float))
                ):
                    cursor_position = {
                        "x": cursor_payload["x"],
                        "y": cursor_payload["y"],
                        "source": str(cursor_payload.get("source") or "unknown"),
                    }
            except Exception:
                # Cursor reporting is an optional visual aid, never a reason for the
                # primary screenshot capture to fail (including older CUA backends).
                cursor_position = None

            pointer_overlay = self._overlay_last_agent_pointer(image_path)
            cursor_overlay = self._overlay_cursor_position(image_path, cursor_position)
            # Raw screenshot_hash intentionally excludes overlays for no-op detection.
            # visual_hash includes the visible cursor markers, so frames remain distinct
            # when the CUA-reported pointer moves over an otherwise static screen.
            visual_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()

            windows_text = ""
            if include_windows:
                try:
                    raw_windows = call_driver(
                        "list_windows",
                        {"on_screen_only": True},
                        timeout=12,
                    )
                    parsed = json.loads(raw_windows)
                    rows = parsed.get("windows", []) if isinstance(parsed, dict) else parsed
                    if not isinstance(rows, list):
                        raise ValueError("invalid windows list")
                    rows = [row for row in rows if isinstance(row, dict)]
                    rows = [
                        row for row in rows
                        if not any(
                            token in str(row.get("app_name") or "").casefold()
                            for token in ("gjs", "update-notifier", "ibus-", "xdg-desktop-portal")
                        )
                    ]
                    # Do not truncate arbitrary JSON: that hides the browser near the end
                    # of the list and tempts the model to guess another app's window IDs.
                    def window_rank(row: dict[str, Any]) -> tuple[int, int]:
                        app = str(row.get("app_name") or "").casefold()
                        title = str(row.get("title") or "").casefold()
                        browser = int(any(name in f"{app} {title}" for name in ("firefox", "chrome", "chromium", "brave", "edge")))
                        area = int(row.get("width") or 0) * int(row.get("height") or 0)
                        return browser, area
                    rows = sorted(rows, key=window_rank, reverse=True)
                    compact = []
                    for row in rows:
                        width, height = int(row.get("width") or 0), int(row.get("height") or 0)
                        if not row.get("is_on_screen", True) or width < 250 or height < 150:
                            continue
                        compact.append({
                            "app": str(row.get("app_name") or "")[:60],
                            "title": str(row.get("title") or "")[:140],
                            "pid": row.get("pid"),
                            "window_id": row.get("window_id"),
                            "bounds": {"x": row.get("x", 0), "y": row.get("y", 0), "width": width, "height": height},
                        })
                        if len(compact) >= 8:
                            break
                    windows_text = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
                except (CuaDriverError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    windows_text = f"Window inventory unavailable: {exc}"

            width, height = png_dimensions(image_path)
            size_bytes = image_path.stat().st_size
            captured_at = datetime.now(timezone.utc).isoformat()
            metadata = {
                "kind": "screen_observation",
                "image_ref": str(image_path),
                "image_mime_type": "image/png",
                "screenshot_id": image_path.stem,
                "captured_at": captured_at,
                "width": width,
                "height": height,
                "size_bytes": size_bytes,
                "session_label": session_label,
                "screenshot_hash": screenshot_hash,
                "visual_hash": visual_hash,
                "pointer_overlay": "last_agent_click" if pointer_overlay else None,
                "cursor_position": cursor_position,
                "cursor_overlay": "cua_cursor_position" if cursor_overlay else None,
            }
            return ToolResult(
                success=True,
                name=self.name,
                content={
                    "image_ref": str(image_path),
                    "image_mime_type": "image/png",
                    "width": width,
                    "height": height,
                    "size_bytes": size_bytes,
                    "screenshot_hash": screenshot_hash,
                    "visual_hash": visual_hash,
                    "pointer_overlay": "last_agent_click" if pointer_overlay else None,
                    "cursor_position": cursor_position,
                    "cursor_overlay": "cua_cursor_position" if cursor_overlay else None,
                    "cursor_hint": (
                        f"CUA-reported cursor position is ({cursor_position['x']}, {cursor_position['y']}); "
                        f"source={cursor_position['source']}. A synthetic source is not proof of the physical user's pointer."
                        if cursor_position else "Cursor position unavailable from Cua Driver; only the last-agent-click marker may be shown."
                    ),
                    "windows": windows_text,
                    "driver_output": output[:1500],
                },
                metadata=metadata,
                summary=(
                    f"Captured desktop screenshot ({width or '?'}x{height or '?'}, {size_bytes} bytes). "
                    "The image is saved and will be attached to the next model context."
                ),
                evidence={
                    "kind": "screen_observation",
                    "image_ref": str(image_path),
                    "width": width,
                    "height": height,
                    "size_bytes": size_bytes,
                },
            )
        except OSError as exc:
            return ToolResult(
                success=False,
                name=self.name,
                content={"error": {"type": "screenshot_storage_failed", "message": f"Could not save screenshot under workspace: {exc}"}},
                metadata={"recovery_hint": "Check workspace permissions and free disk space before retrying screenshot capture."},
                summary=f"Desktop observation failed because screenshot storage was unavailable: {exc}",
            )
        except CuaDriverError as exc:
            detail = str(exc)
            lowered = detail.lower()
            if any(marker in lowered for marker in ("getimage", "badmatch", "xwayland", "rootless")):
                error_type = "desktop_capture_backend_unavailable"
                recovery_hint = (
                    "The running Cua Driver cannot capture the rootless Wayland desktop through XWayland. "
                    "Daena automatically starts a missing daemon with CUA_DRIVER_RS_ENABLE_WAYLAND=1, "
                    "but it does not silently restart a daemon that is already running with the wrong backend. "
                    "Stop that existing daemon and let computer_observe restart it, then retry."
                )
                summary = "Desktop observation failed: the active driver capture backend cannot capture this Wayland desktop."
            else:
                error_type = "computer_use_unavailable"
                recovery_hint = (
                    "Check cua-driver status and cua-driver doctor from the logged-in graphical session. "
                    "Do not keep retrying screenshots while the daemon or desktop capture access is unavailable."
                )
                summary = f"Desktop observation unavailable: {exc}"
            return ToolResult(
                success=False,
                name=self.name,
                content={"error": {"type": error_type, "message": detail}},
                metadata={"recovery_hint": recovery_hint},
                summary=summary,
            )
