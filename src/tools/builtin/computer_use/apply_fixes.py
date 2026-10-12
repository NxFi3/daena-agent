#!/usr/bin/env python3
"""Apply the confirmed Computer Use fixes to a Daena working tree.

Run from the repo root:   python3 apply_fixes.py --check   (dry run)
                          python3 apply_fixes.py           (apply)

Each patch is an exact-string replacement that must match exactly once. If any
patch fails to match, NOTHING is written. No git commands are run; nothing is
staged or committed; unrelated files are never touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE = Path("src/tools/builtin/computer_use")

PATCHES: list[tuple[str, str, str]] = []


def patch(file: str, old: str, new: str) -> None:
    PATCHES.append((file, old, new))


# ---------------------------------------------------------------- computer_use.py
# NameError: `hint` is undefined (should be title_hint).
patch(
    "computer_use.py",
    '            inspect_title = "" if pid is not None and window_id is not None else hint\n',
    '            inspect_title = "" if pid is not None and window_id is not None else title_hint\n',
)

# ---------------------------------------------------------------- act.py
patch(
    "act.py",
    "\n\nclass ComputerActTool(Tool):\n",
    '''

# Long text is typed in bounded chunks: one huge argv/IPC payload can hit
# EAGAIN (os error 11) or exceed the per-call timeout inside the driver.
_TYPE_CHUNK_CHARS = 400
_TYPE_TOTAL_BUDGET_SECONDS = 120.0


class ComputerActTool(Tool):
''',
)

patch(
    "act.py",
    '''            delivery_mode_escalated = False
            try:
                output = call_driver(action, args, timeout=20)
            except CuaDriverError as exc:
                detail = str(exc)
''',
    '''            delivery_mode_escalated = False
            chunked_typing = (
                action == "type_text"
                and not element_token
                and isinstance(args.get("text"), str)
                and len(args["text"]) > _TYPE_CHUNK_CHARS
            )
            try:
                output = (
                    self._type_in_chunks(args) if chunked_typing
                    else call_driver(action, args, timeout=20)
                )
            except CuaDriverError as exc:
                detail = str(exc)
                if chunked_typing:
                    # Part of the text may already be on screen. Never replay it.
                    raise
''',
)

patch(
    "act.py",
    '''    @staticmethod
    def _save_pointer_state(path: Path, state: dict[str, Any]) -> None:
''',
    '''    @staticmethod
    def _type_in_chunks(args: dict[str, Any]) -> str:
        """Type long text in bounded chunks; stop at the first failure, never retry."""
        text = str(args["text"])
        chunks = [text[i:i + _TYPE_CHUNK_CHARS] for i in range(0, len(text), _TYPE_CHUNK_CHARS)]
        deadline = time.monotonic() + _TYPE_TOTAL_BUDGET_SECONDS
        sent = 0
        last_output = ""
        for index, chunk in enumerate(chunks, start=1):
            if time.monotonic() > deadline:
                raise CuaDriverError(
                    f"Typing stopped after {sent} of {len(text)} characters: total time budget exceeded. "
                    "The text on screen is partial; do not resend the whole text."
                )
            chunk_args = {**args, "text": chunk}
            if index > 1:
                # Only the first chunk may carry a focus point; later chunks go to the focused target.
                for key in ("x", "y", "coordinate_frame"):
                    chunk_args.pop(key, None)
            try:
                last_output = call_driver("type_text", chunk_args, timeout=15)
            except CuaDriverError as exc:
                raise CuaDriverError(
                    f"Typing stopped after {sent} of {len(text)} characters (chunk {index}/{len(chunks)}). "
                    "The text on screen is partial and the failing chunk may or may not have been typed. "
                    f"Do not resend the whole text. Original error: {exc}"
                ) from exc
            sent += len(chunk)
            time.sleep(0.05)
        return last_output

    @staticmethod
    def _save_pointer_state(path: Path, state: dict[str, Any]) -> None:
''',
)

# ---------------------------------------------------------------- driver.py
patch(
    "driver.py",
    '''def _daemon_status(executable: str) -> tuple[bool, str]:
''',
    '''def _kill_process_group(process: subprocess.Popen) -> None:
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
    """Run a helper with a hard deadline, no inherited stdin, and cleanup on timeout."""
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        try:
            process.communicate(timeout=2.0)
        except Exception:
            pass
        raise subprocess.TimeoutExpired(command, timeout)
    except BaseException:
        _kill_process_group(process)
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _daemon_status(executable: str) -> tuple[bool, str]:
''',
)

patch(
    "driver.py",
    '''            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=max(1.0, min(float(timeout), 60.0)),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
''',
    '''            return _run_bounded(command, max(1.0, min(float(timeout), 60.0)))
        except subprocess.TimeoutExpired as exc:
''',
)

patch(
    "driver.py",
    '''        try:
            socket_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.Popen(
                [executable, "serve", "--socket", str(socket_path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                close_fds=True,
                start_new_session=True,
            )
        except OSError as exc:
''',
    '''        try:
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
''',
)

patch(
    "driver.py",
    '''            running, _detail = _daemon_status(executable)
            if running:
                return
            time.sleep(0.2)
''',
    '''            running, _detail = _daemon_status(executable)
            if running:
                return
            exit_code = process.poll()
            if exit_code not in (None, 0):
                raise CuaDriverError(
                    f"Cua Driver daemon exited immediately with code {exit_code}. "
                    "See ~/.cache/cua-driver/daemon.log and run cua-driver doctor in the "
                    f"logged-in desktop session. Initial status: {status_text[:300]}"
                )
            time.sleep(0.2)
''',
)


# ---------------------------------------------------------------- computer_use.py
# PRIMARY BUG gap: only check box / radio / toggle / switch roles were verified
# semantically. A toolbar *push button* that exposes `selected=True` (focus/selection
# outline) but no real CHECKED state was accepted as success purely because the
# screenshot hash changed. Treat "selected-only" state on a non-selection role as
# UNVERIFIABLE (fail closed, no dependent action) instead of trusting it.
patch(
    "computer_use.py",
    '''                            toggle_target = {
                                "role": str(record.get("role") or ""),
                                "label": str(record.get("label") or ""),
                                "checked_before": checked_before if isinstance(checked_before, bool) else None,
                            }
''',
    '''                            toggle_target = {
                                "role": str(record.get("role") or ""),
                                "label": str(record.get("label") or ""),
                                "checked_before": checked_before if isinstance(checked_before, bool) else None,
                            }
                        elif (
                            isinstance(record.get("selected"), bool)
                            and not isinstance(record.get("checked"), bool)
                            and role not in self._SELECTION_STATE_ROLES
                        ):
                            # A button-like control that only reports SELECTED. For these
                            # roles SELECTED is focus/selection, not "the action took effect".
                            toggle_target = {
                                "role": str(record.get("role") or ""),
                                "label": str(record.get("label") or ""),
                                "checked_before": None,
                                "selected_only": True,
                            }
''',
)

patch(
    "computer_use.py",
    '''                if toggle_target is not None and captured.success:
''',
    '''                if (
                    toggle_target is not None and toggle_target.get("selected_only")
                    and captured.success
                ):
                    # Only the "became/remained selected" signature is unverifiable.
                    # A selected-capable button that is not selected afterwards is
                    # judged by the ordinary visual evidence.
                    if self._selected_control_state(
                        captured.content.get("elements", []),
                        toggle_target["role"], toggle_target["label"],
                    ) is not True:
                        toggle_target = None
                if toggle_target is not None and captured.success:
''',
)

patch(
    "computer_use.py",
    '''    def _capture(
        self,
        *,
        mode: str = "som",
''',
    '''    # Roles for which AT-SPI SELECTED genuinely is the state (a chosen tab or row).
    _SELECTION_STATE_ROLES = frozenset({
        "page tab", "tab", "list item", "tree item", "option", "table cell",
        "table row", "menu item", "combo box",
    })

    @staticmethod
    def _selected_control_state(elements: Any, role: str, label: str) -> bool | None:
        """Read the SELECTED flag of exactly one matching control, else None."""
        if not isinstance(elements, list):
            return None
        wanted_role = str(role or "").strip().casefold()
        wanted_label = str(label or "").strip().casefold()
        matches = [
            item for item in elements
            if isinstance(item, dict)
            and str(item.get("role") or "").strip().casefold() == wanted_role
            and str(item.get("label") or "").strip().casefold() == wanted_label
            and isinstance(item.get("selected"), bool)
        ]
        return matches[0]["selected"] if len(matches) == 1 else None

    def _capture(
        self,
        *,
        mode: str = "som",
''',
)


def main() -> int:
    check_only = "--check" in sys.argv
    contents: dict[str, str] = {}
    for name, old, new in PATCHES:
        path = BASE / name
        if name not in contents:
            if not path.is_file():
                print(f"ERROR: {path} not found (run from repo root).")
                return 1
            contents[name] = path.read_text(encoding="utf-8")
        count = contents[name].count(old)
        if count != 1:
            print(f"ERROR: patch for {name} matched {count} times (expected 1):\n{old[:160]!r}")
            print("Nothing was written. The file differs from the version this patch was made for.")
            return 1
        contents[name] = contents[name].replace(old, new, 1)
    if check_only:
        print(f"OK: all {len(PATCHES)} patches match exactly once. Nothing written (--check).")
        return 0
    for name, text in contents.items():
        (BASE / name).write_text(text, encoding="utf-8")
    print(f"Applied {len(PATCHES)} patches to: {', '.join(sorted(contents))}")
    print("Review with: git diff   (nothing was staged or committed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
