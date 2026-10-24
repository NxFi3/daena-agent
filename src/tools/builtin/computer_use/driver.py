from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import tempfile

from .runtime import ComputerError, checkpoint, pause
from pathlib import Path
from typing import Any, Mapping

DRIVER_TOOLS = frozenset(
    {
        # Session lifecycle must be recoverable after the daemon expires an idle run.
        "start_session",
        "get_desktop_state",
        "get_cursor_position",
        "get_window_state",
        "list_windows",
        "list_apps",
        "bring_to_front",
        "click",
        "double_click",
        "right_click",
        "drag",
        "type_text",
        "press_key",
        "hotkey",
        "scroll",
    }
)

_SESSION_ENV_KEYS = (
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "XAUTHORITY",
    "XDG_RUNTIME_DIR",
    "DBUS_SESSION_BUS_ADDRESS",
    "XDG_SESSION_TYPE",
    "XDG_CURRENT_DESKTOP",
    "DESKTOP_SESSION",
)
_SESSION_PROCESS_MARKERS = (
    "gnome-session",
    "gdm-wayland-session",
    "xfce4-session",
    "mate-session",
    "cinnamon",
    "plasmashell",
    "kwin_wayland",
    "sway",
    "hyprland",
)
_DAEMON_START_LOCK = threading.Lock()
_DAEMON_START_TIMEOUT_SECONDS = 8.0


class CuaDriverError(ComputerError):
    """A bounded Cua Driver invocation failed."""

    def __init__(
        self, message, *, code="driver_error", dispatched=False, not_dispatched=False
    ):
        super().__init__(
            code, message, dispatched=dispatched, not_dispatched=not_dispatched
        )


def response_error(message, *outputs):
    """Preserve structured refusals without treating text fragments as delivery evidence."""
    for output in outputs:
        try:
            payload = json.loads(output)
        except (ValueError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        details = payload.get("error")
        details = details if isinstance(details, dict) else payload
        code = details.get("code") or payload.get("code") or "driver_refused"
        # Cua's background_unavailable means no focus-free backend exists;
        # the driver refuses BEFORE input. Other failures need explicit evidence.
        refused = code == "background_unavailable" or any(
            p.get("delivered") is False or p.get("dispatch_status") == "not_dispatched"
            for p in (payload, details)
        )
        error = CuaDriverError(message, code=str(code), not_dispatched=refused)
        return error
    return CuaDriverError(message)


def driver_executable() -> str:
    executable = shutil.which("cua-driver")
    if executable:
        return executable
    fallback = Path.home() / ".local" / "bin" / "cua-driver"
    if fallback.is_file() and os.access(fallback, os.X_OK):
        return str(fallback)
    raise CuaDriverError(
        "Cua Driver is not installed or not on PATH. Install Cua Driver before using Computer Use."
    )


def session_label_from_id(value: str | None) -> str:
    raw = str(value or "default")
    label = re.sub(r"[^A-Za-z0-9_-]+", "-", raw).strip("-")[:32] or "default"
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"daena-{label[:20]}-{digest}"


def _read_process_environment(proc_root: Path = Path("/proc")) -> list[dict[str, str]]:
    """Find a same-user graphical-session environment without importing shell state."""
    environments: list[tuple[int, dict[str, str]]] = []
    current_uid = os.getuid() if hasattr(os, "getuid") else None
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return []

    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            if current_uid is not None and entry.stat().st_uid != current_uid:
                continue
            command_line = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\x00", b" ")
                .decode("utf-8", errors="replace")
                .lower()
            )
            priority = next(
                (
                    len(_SESSION_PROCESS_MARKERS) - index
                    for index, marker in enumerate(_SESSION_PROCESS_MARKERS)
                    if marker in command_line
                ),
                0,
            )
            if priority <= 0:
                continue
            raw = (entry / "environ").read_bytes().split(b"\x00")
            env: dict[str, str] = {}
            for item in raw:
                if b"=" not in item:
                    continue
                key, value = item.split(b"=", 1)
                name = key.decode("utf-8", errors="ignore")
                text = value.decode("utf-8", errors="replace")
                if name in _SESSION_ENV_KEYS and text:
                    env[name] = text
            if env:
                environments.append((priority * 100 + len(env), env))
        except (OSError, PermissionError, ProcessLookupError, ValueError):
            continue

    environments.sort(key=lambda item: item[0], reverse=True)
    return [env for _, env in environments]


def _compose_desktop_environment(
    base_environment: Mapping[str, str] | None = None,
    session_environments: list[Mapping[str, str]] | None = None,
    *,
    runtime_dir: Path | None = None,
    home: Path | None = None,
    x11_socket_dir: Path = Path("/tmp/.X11-unix"),
) -> dict[str, str]:
    """Recover display/session variables for a CLI started outside the GUI's shell."""
    env = {
        str(key): str(value) for key, value in (base_environment or {}).items() if value
    }
    for candidate in session_environments or []:
        for key in _SESSION_ENV_KEYS:
            if not env.get(key) and candidate.get(key):
                env[key] = str(candidate[key])

    user_home = (home or Path.home()).expanduser()
    uid = os.getuid() if hasattr(os, "getuid") else 0
    runtime = Path(
        env.get("XDG_RUNTIME_DIR")
        or (runtime_dir if runtime_dir is not None else Path(f"/run/user/{uid}"))
    ).expanduser()
    if runtime.is_dir():
        env.setdefault("XDG_RUNTIME_DIR", str(runtime))
        if not env.get("DBUS_SESSION_BUS_ADDRESS") and (runtime / "bus").exists():
            env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={runtime / 'bus'}"

        if not env.get("WAYLAND_DISPLAY"):
            wayland_sockets = sorted(runtime.glob("wayland-[0-9]*"))
            if wayland_sockets:
                env["WAYLAND_DISPLAY"] = wayland_sockets[0].name

        if not env.get("XAUTHORITY"):
            auth_files = sorted(runtime.glob(".mutter-Xwaylandauth.*"))
            if auth_files:
                env["XAUTHORITY"] = str(auth_files[0])
            elif (user_home / ".Xauthority").is_file():
                env["XAUTHORITY"] = str(user_home / ".Xauthority")

    if not env.get("DISPLAY") and x11_socket_dir.is_dir():
        x11_sockets = sorted(
            path for path in x11_socket_dir.glob("X*") if path.name[1:].isdigit()
        )
        if x11_sockets:
            env["DISPLAY"] = f":{x11_sockets[0].name[1:]}"

    if not env.get("XDG_SESSION_TYPE"):
        env["XDG_SESSION_TYPE"] = (
            "wayland"
            if env.get("WAYLAND_DISPLAY")
            else ("x11" if env.get("DISPLAY") else "")
        )
    if (
        env.get("WAYLAND_DISPLAY")
        and env.get("XDG_SESSION_TYPE", "").lower() == "wayland"
    ):
        # Native Wayland avoids rootless XWayland's unreadable root drawable
        # for desktop screenshots. This is an explicit Cua Driver opt-in.
        env["CUA_DRIVER_RS_ENABLE_WAYLAND"] = "1"

    return env


def _desktop_session_environment() -> dict[str, str]:
    candidates = _read_process_environment()
    env = _compose_desktop_environment(os.environ, candidates)
    if not env.get("DISPLAY") and not env.get("WAYLAND_DISPLAY"):
        raise CuaDriverError(
            "Cua Driver is not running and no active graphical session could be found. "
            "Run Daena from the logged-in desktop session, or start Cua Driver there."
        )
    if not env.get("XDG_RUNTIME_DIR"):
        raise CuaDriverError(
            "A display was found but XDG_RUNTIME_DIR is missing; cannot connect Cua Driver "
            "to the graphical user's D-Bus/Wayland session."
        )
    return env


def _kill_process_group(process: subprocess.Popen) -> None:
    # The helper runs in its own session, so this only ever kills that one
    # `cua-driver call` client (and anything it spawned), never the daemon,
    # the browser, or other user processes.
    try:
        os.killpg(process.pid, 9)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except OSError:
            pass


def _run_bounded(command: list[str], timeout: float) -> subprocess.CompletedProcess:
    """Drain into temporary files, cap replies, and terminate the helper on cancellation.

    The daemon may already have accepted an input when the client is stopped. A
    timeout is an unknown outcome, never a reason to replay the input.
    """
    checkpoint()
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                checkpoint()
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(
                        [command[0], "<arguments omitted>"], timeout
                    )
                if (
                    os.fstat(stdout.fileno()).st_size
                    + os.fstat(stderr.fileno()).st_size
                    > 16_000_000
                ):
                    raise CuaDriverError(
                        "Driver response exceeded 16 MB.", code="response_too_large"
                    )
                time.sleep(0.025)
            checkpoint()
        except BaseException:
            _kill_process_group(process)
            process.wait(timeout=2)
            raise
        stdout.seek(0)
        stderr.seek(0)
        out, err = stdout.read(16_000_001), stderr.read(16_000_001)
        if len(out) + len(err) > 16_000_000:
            raise CuaDriverError(
                "Driver response exceeded 16 MB.", code="response_too_large"
            )
        return subprocess.CompletedProcess(
            command,
            process.returncode,
            out.decode("utf-8", "replace"),
            err.decode("utf-8", "replace"),
        )


def _daemon_status(executable: str) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            [executable, "status"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
    lowered = output.casefold()
    running = "daemon is running" in lowered and "daemon is not running" not in lowered
    return running, output


def _ensure_daemon_running(executable: str) -> None:
    """Start a default-policy driver only when a graphical desktop is discoverable."""
    with _DAEMON_START_LOCK:
        running, status_text = _daemon_status(executable)
        if running:
            return

        env = _desktop_session_environment()
        socket_path = Path.home() / ".cache" / "cua-driver" / "cua-driver.sock"
        try:
            socket_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                log_target: Any = open(socket_path.parent / "daemon.log", "ab")
            except OSError:
                log_target = subprocess.DEVNULL
            try:
                process = subprocess.Popen(
                    [executable, "serve", "--socket", str(socket_path)],
                    stdin=subprocess.DEVNULL,
                    stdout=log_target,
                    stderr=log_target,
                    env=env,
                    close_fds=True,
                    start_new_session=True,
                )
            finally:
                if log_target is not subprocess.DEVNULL:
                    log_target.close()
        except OSError as exc:
            raise CuaDriverError(
                f"Cua Driver daemon could not be started automatically: {exc}"
            ) from exc

        deadline = time.monotonic() + _DAEMON_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            running, _detail = _daemon_status(executable)
            if running:
                return
            exit_code = process.poll()
            if exit_code not in (None, 0):
                raise CuaDriverError(
                    f"Cua Driver daemon exited immediately with code {exit_code}. "
                    "See ~/.cache/cua-driver/daemon.log and run cua-driver doctor in the "
                    f"logged-in desktop session. Initial status: {status_text[:300]}"
                )
            pause(0.2)

        raise CuaDriverError(
            "Cua Driver daemon did not become ready after automatic startup. "
            "Run the command cua-driver doctor in the logged-in desktop session "
            f"to diagnose capture access. Initial status: {status_text[:500]}"
        )


def ensure_driver_running(session: str | None = None) -> None:
    """Ensure the daemon and the requested lifecycle session are ready.

    Cua Driver can keep the daemon running while ending an idle named session.
    Re-create/return the session before the first action instead of letting the
    first accessibility request fail with ``session has ended``.
    """
    _ensure_daemon_running(driver_executable())
    label = str(session or "").strip()
    if label:
        call_driver("start_session", {"session": label}, timeout=5.0)


def _driver_refusal_detail(output: str) -> str | None:
    """Return a useful error when Cua Driver encodes a refusal in stdout.

    `cua-driver call` may exit with code 0 even for structured refusals such
    as {"code":"invalid_action_target"}; treating every zero exit as success
    makes the agent believe an input event was delivered when it was not.
    """
    text = str(output or "").strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        lowered = text.casefold()
        refusal_markers = (
            "invalid_action_target",
            "background_unavailable",
            "window_target_not_found",
            "stale_element_token",
            "permission_denied",
        )
        return text if any(marker in lowered for marker in refusal_markers) else None

    if not isinstance(payload, dict):
        return None

    status = str(payload.get("status") or "").casefold()
    effect = str(payload.get("effect") or "").casefold()
    code = str(payload.get("code") or "").strip()
    is_error = payload.get("isError") is True
    has_refusal = bool(payload.get("refusal"))
    explicit_negative = (
        payload.get("ok") is False
        or payload.get("success") is False
        or payload.get("delivered") is False
    )
    if (
        is_error
        or has_refusal
        or explicit_negative
        or status in {"refused", "error", "failed"}
        or effect == "refused"
        or code
    ):
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return None


def call_driver(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    timeout: float = 30.0,
) -> str:
    """Call one explicitly allowlisted Cua Driver action, starting it on demand if needed."""
    name = str(tool_name or "").strip()
    if name not in DRIVER_TOOLS:
        raise CuaDriverError(f"Unsupported Cua Driver action: {name or '(empty)'}")
    args = arguments if isinstance(arguments, dict) else {}
    executable = driver_executable()
    command = [
        executable,
        "call",
        name,
        json.dumps(args, ensure_ascii=False, separators=(",", ":"), default=str),
    ]

    def invoke() -> subprocess.CompletedProcess[str]:
        try:
            return _run_bounded(command, max(1.0, min(float(timeout), 60.0)))
        except subprocess.TimeoutExpired as exc:
            raise CuaDriverError(
                f"Cua Driver action '{name}' timed out after {timeout:g} seconds."
            ) from exc
        except OSError as exc:
            raise CuaDriverError(
                f"Could not start Cua Driver: {type(exc).__name__}: {exc}"
            ) from exc

    completed = invoke()
    output = (completed.stdout or "").strip()
    error = (completed.stderr or "").strip()
    combined = error + "\n" + output
    lowered = combined.casefold()

    if (
        name
        in {
            "start_session",
            "list_windows",
            "list_apps",
            "get_window_state",
            "get_desktop_state",
            "get_cursor_position",
        }
        and completed.returncode != 0
        and ("daemon is not running" in lowered or "daemon not running" in lowered)
    ):
        _ensure_daemon_running(executable)
        completed = invoke()
        output = (completed.stdout or "").strip()
        error = (completed.stderr or "").strip()

    if completed.returncode == 0:
        refusal = _driver_refusal_detail(output)
        if refusal:
            raise response_error(
                f"Cua Driver action '{name}' was refused by the driver: {refusal[:2500]}",
                output,
            )

    if completed.returncode != 0:
        detail = error or output or f"exit code {completed.returncode}"
        lowered = detail.lower()
        if "daemon is not running" in lowered or "daemon not running" in lowered:
            detail += (
                "\nCua Driver is still unavailable. Start it from the logged-in desktop session "
                "and run cua-driver doctor."
            )
        raise response_error(
            f"Cua Driver action '{name}' failed (exit {completed.returncode}): {detail[:2500]}",
            error,
            output,
        )
    # Accessibility snapshots can contain thousands of structured elements plus a
    # compatibility tree. Keep enough of the JSON intact for computer_inspect to
    # parse it; truncating mid-JSON silently turns a valid tree into a parse error.
    return output


def pointer_state_path(workspace_root: str | Path, session_label: str) -> Path:
    """Return the workspace-local state file for the agent's last pixel action."""
    session_key = hashlib.sha256(str(session_label).encode("utf-8")).hexdigest()[:16]
    return (
        Path(workspace_root).expanduser().resolve()
        / ".daena"
        / "computer_use"
        / f"pointer_{session_key}.json"
    )


def png_dimensions(path: str | Path) -> tuple[int | None, int | None]:
    """Read PNG dimensions without requiring Pillow."""
    try:
        with Path(path).open("rb") as image:
            header = image.read(24)
        if (
            len(header) < 24
            or header[:8] != b"\x89PNG\r\n\x1a\n"
            or header[12:16] != b"IHDR"
        ):
            return None, None
        return int.from_bytes(header[16:20], "big"), int.from_bytes(
            header[20:24], "big"
        )
    except (OSError, ValueError):
        return None, None
