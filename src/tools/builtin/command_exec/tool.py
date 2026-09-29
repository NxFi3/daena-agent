from __future__ import annotations

import os
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool

_EXIT_CODE_HINTS: dict[str, dict[int, str]] = {
    "pytest": {
        0: "all tests passed",
        1: "one or more tests failed",
        2: "test collection error (import or syntax error)",
        3: "internal pytest error",
        4: "usage error (bad command-line args)",
        5: "no tests were collected — check test file naming (test_*.py)",
    },
    "npm": {
        1: "npm error",
    },
    "cargo": {
        101: "cargo test found failing tests",
    },
}


class CommandExec(Tool):
    """
    Execute local commands for the Evana agent runtime.

    Features:
        - Foreground command execution.
        - Optional background execution.
        - Cross-platform process-group isolation.
        - Timeout handling.
        - Process-group termination on timeout.
        - Bounded stdout/stderr.
        - Structured JSON-compatible ToolResult.content.
        - No full stdout/stderr duplication in metadata.
        - Human-readable exit-code hints for known tools.
    """

    name = "command_exec"
    action = "run"

    DEFAULT_TIMEOUT_MS = 120_000
    MAX_TIMEOUT_MS = 600_000

    DEFAULT_MAX_OUTPUT_CHARS = 8_000
    MAX_OUTPUT_CHARS = 32_000
    MIN_OUTPUT_CHARS = 512

    BACKGROUND_STARTUP_GRACE_MS = 150

    description = (
        "Run a local command given as an argv array (not a shell string). "
        "Returns exit status, bounded stdout/stderr, duration, and timeout info.\n"
        "\n"
        "Foreground (default): use for commands that finish on their own, such as "
        "installs, builds, tests, scripts, and file or git operations. Set "
        "timeout_ms for slow ones (max 600000). Never repeat a command that timed "
        "out unchanged; change something first, for example a larger timeout.\n"
        "\n"
        "Background (background=true): use for anything that keeps running until "
        "stopped, such as servers, watchers, and dev tools. A long-running command "
        "in the foreground blocks until timeout and is killed, so the tool may "
        "reject it. Background returns pid and log_file. Read the log to check "
        "startup, then verify with a short foreground command (for example curl "
        "with a timeout). Stop it when you are done.\n"
        "\n"
        "Always set workdir explicitly."
    )

    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "array",
                "description": (
                    "Executable and its arguments. Each element is passed as "
                    "a separate argument. Example: "
                    '["python", "-m", "pytest", "-q"].'
                ),
                "items": {
                    "type": "string",
                },
                "minItems": 1,
            },
            "workdir": {
                "type": "string",
                "description": (
                    "Optional working directory. If omitted, the current "
                    "working directory is inherited."
                ),
            },
            "timeout_ms": {
                "type": "integer",
                "description": (
                    "Maximum execution time for foreground commands in "
                    f"milliseconds. Default: {DEFAULT_TIMEOUT_MS}."
                ),
                "default": DEFAULT_TIMEOUT_MS,
                "minimum": 1,
                "maximum": MAX_TIMEOUT_MS,
            },
            "max_output_chars": {
                "type": "integer",
                "description": (
                    "Maximum output characters returned to the agent context. "
                    f"Default: {DEFAULT_MAX_OUTPUT_CHARS}."
                ),
                "default": DEFAULT_MAX_OUTPUT_CHARS,
                "minimum": MIN_OUTPUT_CHARS,
                "maximum": MAX_OUTPUT_CHARS,
            },
            "background": {
                "type": "boolean",
                "description": (
                    "Start the command without waiting for it to finish. "
                    "The result returns the process ID and log file path."
                ),
                "default": False,
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def execute(
        self,
        command: list[str],
        workdir: str | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        background: bool = False,
    ) -> ToolResult:

        validation_error = self._validate_arguments(
            command=command,
            timeout_ms=timeout_ms,
            max_output_chars=max_output_chars,
            background=background,
        )

        if validation_error is not None:
            return validation_error

        resolved_workdir, workdir_error = self._resolve_workdir(workdir)

        if workdir_error is not None:
            return workdir_error

        if background:
            return self._execute_background(
                command=command,
                workdir=resolved_workdir,
                max_output_chars=max_output_chars,
            )

        return self._execute_foreground(
            command=command,
            workdir=resolved_workdir,
            timeout_ms=timeout_ms,
            max_output_chars=max_output_chars,
        )

    def _validate_arguments(
        self,
        *,
        command: Any,
        timeout_ms: Any,
        max_output_chars: Any,
        background: Any,
    ) -> ToolResult | None:

        if not isinstance(command, list):
            return self._error(
                error_type="invalid_argument",
                message="command must be a list of strings.",
            )

        if not command:
            return self._error(
                error_type="invalid_argument",
                message="command cannot be empty.",
            )

        if not all(isinstance(argument, str) for argument in command):
            return self._error(
                error_type="invalid_argument",
                message="Every command argument must be a string.",
            )

        if not command[0].strip():
            return self._error(
                error_type="invalid_argument",
                message="The executable cannot be empty.",
            )

        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int):
            return self._error(
                error_type="invalid_argument",
                message="timeout_ms must be an integer.",
            )

        if not (1 <= timeout_ms <= self.MAX_TIMEOUT_MS):
            return self._error(
                error_type="invalid_argument",
                message=("timeout_ms must be between 1 and " f"{self.MAX_TIMEOUT_MS}."),
            )

        if isinstance(max_output_chars, bool) or not isinstance(max_output_chars, int):
            return self._error(
                error_type="invalid_argument",
                message="max_output_chars must be an integer.",
            )

        if not (self.MIN_OUTPUT_CHARS <= max_output_chars <= self.MAX_OUTPUT_CHARS):
            return self._error(
                error_type="invalid_argument",
                message=(
                    "max_output_chars must be between "
                    f"{self.MIN_OUTPUT_CHARS} and "
                    f"{self.MAX_OUTPUT_CHARS}."
                ),
            )

        if not isinstance(background, bool):
            return self._error(
                error_type="invalid_argument",
                message="background must be a boolean.",
            )

        return None

    def _execute_foreground(
        self,
        *,
        command: list[str],
        workdir: Path | None,
        timeout_ms: int,
        max_output_chars: int,
    ) -> ToolResult:

        started = time.perf_counter()

        try:
            process = self._spawn(
                command=command,
                workdir=workdir,
            )

        except FileNotFoundError as exc:
            return self._execution_error(
                command=command,
                workdir=workdir,
                started=started,
                error_type="command_not_found",
                message=str(exc),
            )

        except PermissionError as exc:
            return self._execution_error(
                command=command,
                workdir=workdir,
                started=started,
                error_type="permission_error",
                message=str(exc),
            )

        except OSError as exc:
            return self._execution_error(
                command=command,
                workdir=workdir,
                started=started,
                error_type="execution_error",
                message=str(exc),
            )

        except Exception as exc:
            return self._execution_error(
                command=command,
                workdir=workdir,
                started=started,
                error_type="unexpected_error",
                message=str(exc),
            )

        try:
            stdout_bytes, stderr_bytes = process.communicate(
                timeout=timeout_ms / 1000.0
            )

        except subprocess.TimeoutExpired as exc:
            self._terminate_process_tree(process)

            stdout_bytes, stderr_bytes = process.communicate()

            stdout = self._decode(
                stdout_bytes if stdout_bytes is not None else exc.stdout
            )

            stderr = self._decode(
                stderr_bytes if stderr_bytes is not None else exc.stderr
            )

            stdout, stdout_truncated = self._truncate(
                stdout,
                max_output_chars,
            )

            stderr, stderr_truncated = self._truncate(
                stderr,
                max_output_chars,
            )

            return ToolResult(
                success=False,
                name=self.name,
                content={
                    "success": False,
                    "command": command,
                    "workdir": self._stringify_workdir(workdir),
                    "exit_code": None,
                    "exit_code_hint": "",
                    "stdout": stdout,
                    "stderr": stderr,
                    "timed_out": True,
                    "background": False,
                    "duration_ms": self._duration_ms(started),
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                },
                metadata={},
            )

        except Exception as exc:
            return self._execution_error(
                command=command,
                workdir=workdir,
                started=started,
                error_type="communication_error",
                message=str(exc),
            )

        stdout = self._decode(stdout_bytes)
        stderr = self._decode(stderr_bytes)

        stdout, stdout_truncated = self._truncate(
            stdout,
            max_output_chars,
        )

        stderr, stderr_truncated = self._truncate(
            stderr,
            max_output_chars,
        )

        exit_code = process.returncode

        hint = self._exit_code_hint(
            command,
            exit_code,
        )

        return ToolResult(
            success=(exit_code == 0),
            name=self.name,
            content={
                "success": exit_code == 0,
                "command": command,
                "workdir": self._stringify_workdir(workdir),
                "exit_code": exit_code,
                "exit_code_hint": hint,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": False,
                "background": False,
                "duration_ms": self._duration_ms(started),
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            },
            metadata={},
        )

    def _execute_background(
        self,
        *,
        command: list[str],
        workdir: Path | None,
        max_output_chars: int,
    ) -> ToolResult:

        started = time.perf_counter()

        log_directory = workdir if workdir is not None else Path.cwd()

        try:
            log_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        except OSError as exc:
            return self._execution_error(
                command=command,
                workdir=workdir,
                started=started,
                error_type="log_directory_error",
                message=str(exc),
                extra={
                    "background": True,
                },
            )

        log_path = log_directory / f".evana_command_{uuid.uuid4().hex}.log"

        log_file = None

        try:
            log_file = open(
                log_path,
                "w",
                encoding="utf-8",
                errors="replace",
            )

            process = self._spawn(
                command=command,
                workdir=workdir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

        except FileNotFoundError as exc:
            if log_file is not None:
                log_file.close()

            return self._execution_error(
                command=command,
                workdir=workdir,
                started=started,
                error_type="command_not_found",
                message=str(exc),
                extra={
                    "background": True,
                },
            )

        except PermissionError as exc:
            if log_file is not None:
                log_file.close()

            return self._execution_error(
                command=command,
                workdir=workdir,
                started=started,
                error_type="permission_error",
                message=str(exc),
                extra={
                    "background": True,
                },
            )

        except OSError as exc:
            if log_file is not None:
                log_file.close()

            return self._execution_error(
                command=command,
                workdir=workdir,
                started=started,
                error_type="execution_error",
                message=str(exc),
                extra={
                    "background": True,
                },
            )

        except Exception as exc:
            if log_file is not None:
                log_file.close()

            return self._execution_error(
                command=command,
                workdir=workdir,
                started=started,
                error_type="unexpected_error",
                message=str(exc),
                extra={
                    "background": True,
                },
            )

        finally:
            if log_file is not None:
                log_file.close()

        time.sleep(self.BACKGROUND_STARTUP_GRACE_MS / 1000.0)

        exit_code = process.poll()

        if exit_code is not None:
            log = self._read_log(
                log_path,
                max_output_chars,
            )

            success = exit_code == 0

            hint = self._exit_code_hint(
                command,
                exit_code,
            )

            return ToolResult(
                success=success,
                name=self.name,
                content={
                    "success": success,
                    "command": command,
                    "workdir": self._stringify_workdir(workdir),
                    "exit_code": exit_code,
                    "exit_code_hint": hint,
                    "stdout": log["content"],
                    "stderr": "",
                    "timed_out": False,
                    "background": True,
                    "status": "exited",
                    "pid": process.pid,
                    "log_file": str(log_path),
                    "duration_ms": self._duration_ms(started),
                    "stdout_truncated": log["truncated"],
                    "stderr_truncated": False,
                },
                metadata={},
            )

        return ToolResult(
            success=True,
            name=self.name,
            content={
                "success": True,
                "command": command,
                "workdir": self._stringify_workdir(workdir),
                "exit_code": None,
                "exit_code_hint": "",
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "background": True,
                "status": "running",
                "pid": process.pid,
                "log_file": str(log_path),
                "duration_ms": self._duration_ms(started),
                "stdout_truncated": False,
                "stderr_truncated": False,
            },
            metadata={},
        )

    def _spawn(
        self,
        *,
        command: list[str],
        workdir: Path | None,
        stdout: Any = subprocess.PIPE,
        stderr: Any = subprocess.PIPE,
    ) -> subprocess.Popen:

        kwargs: dict[str, Any] = {
            "args": command,
            "cwd": str(workdir) if workdir is not None else None,
            "stdin": subprocess.DEVNULL,
            "stdout": stdout,
            "stderr": stderr,
            "text": False,
        }

        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True

        return subprocess.Popen(**kwargs)

    @staticmethod
    def _exit_code_hint(
        command: list[str],
        exit_code: int | None,
    ) -> str:

        if exit_code is None or not command:
            return ""

        exe = ""

        for i, arg in enumerate(command):
            if arg in ("-m", "-c", "--module"):
                if i + 1 < len(command):
                    exe = command[i + 1]
                    break
            elif not arg.startswith("-"):
                exe = arg.split("/")[-1]
                break

        exe = exe.lower().replace(".py", "")

        hints = _EXIT_CODE_HINTS.get(exe)

        if not hints:
            return ""

        return hints.get(exit_code, "")

    @staticmethod
    def _terminate_process_tree(
        process: subprocess.Popen,
    ) -> None:

        if process.poll() is not None:
            return

        try:
            if os.name == "nt":

                try:
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                except (
                    OSError,
                    ValueError,
                ):
                    pass

                try:
                    process.wait(timeout=2.0)
                    return
                except subprocess.TimeoutExpired:
                    pass

                try:
                    process.kill()
                except (
                    ProcessLookupError,
                    OSError,
                ):
                    pass

                try:
                    process.wait(timeout=2.0)
                except (
                    subprocess.TimeoutExpired,
                    ProcessLookupError,
                    OSError,
                ):
                    pass

                return

            try:
                os.killpg(
                    process.pid,
                    signal.SIGTERM,
                )
            except ProcessLookupError:
                pass

            try:
                process.wait(timeout=2.0)
                return
            except subprocess.TimeoutExpired:
                pass

            try:
                os.killpg(
                    process.pid,
                    signal.SIGKILL,
                )
            except ProcessLookupError:
                pass

            try:
                process.wait(timeout=2.0)
            except (
                subprocess.TimeoutExpired,
                ProcessLookupError,
                OSError,
            ):
                pass

        except (
            ProcessLookupError,
            PermissionError,
            OSError,
        ):
            try:
                process.kill()
            except (
                ProcessLookupError,
                PermissionError,
                OSError,
            ):
                pass

            try:
                process.wait(timeout=2.0)
            except (
                subprocess.TimeoutExpired,
                ProcessLookupError,
                OSError,
            ):
                pass

    def _resolve_workdir(
        self,
        workdir: str | None,
    ) -> tuple[
        Path | None,
        ToolResult | None,
    ]:

        if workdir is None:
            return None, None

        if not isinstance(workdir, str):
            return (
                None,
                self._error(
                    error_type="invalid_argument",
                    message="workdir must be a string.",
                ),
            )

        workdir = workdir.strip()

        if not workdir:
            return None, None

        path = Path(workdir).expanduser()

        try:
            path = path.resolve()
        except OSError as exc:
            return (
                None,
                self._error(
                    error_type="invalid_workdir",
                    message=f"Could not resolve workdir: {exc}",
                ),
            )

        if not path.exists():
            return (
                None,
                self._error(
                    error_type="invalid_workdir",
                    message=f"Working directory does not exist: {path}",
                ),
            )

        if not path.is_dir():
            return (
                None,
                self._error(
                    error_type="invalid_workdir",
                    message=f"Working directory is not a directory: {path}",
                ),
            )

        return path, None

    @staticmethod
    def _decode(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        if isinstance(value, bytes):
            return value.decode(
                "utf-8",
                errors="replace",
            )

        return str(value)

    @staticmethod
    def _truncate(
        value: str,
        limit: int,
    ) -> tuple[str, bool]:

        value = value or ""

        if len(value) <= limit:
            return value, False

        head = int(limit * 0.60)
        tail = limit - head

        omitted = len(value) - head - tail

        bounded = (
            value[:head].rstrip()
            + "\n\n"
            + f"... {omitted} characters omitted ..."
            + "\n\n"
            + value[-tail:].lstrip()
        )

        return bounded, True

    @staticmethod
    def _read_log(
        path: Path,
        max_output_chars: int,
    ) -> dict[str, Any]:

        try:
            content = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

        except OSError:
            return {
                "content": "",
                "truncated": False,
            }

        content, truncated = CommandExec._truncate(
            content,
            max_output_chars,
        )

        return {
            "content": content,
            "truncated": truncated,
        }

    def _execution_error(
        self,
        *,
        command: list[str],
        workdir: Path | None,
        started: float,
        error_type: str,
        message: str,
        extra: dict[str, Any] | None = None,
    ) -> ToolResult:

        content: dict[str, Any] = {
            "success": False,
            "command": command,
            "workdir": self._stringify_workdir(workdir),
            "exit_code": None,
            "exit_code_hint": "",
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "background": False,
            "duration_ms": self._duration_ms(started),
            "error": {
                "type": error_type,
                "message": message,
            },
        }

        if extra:
            content.update(extra)

        return ToolResult(
            success=False,
            name=self.name,
            content=content,
            metadata={},
        )

    def _error(
        self,
        *,
        error_type: str,
        message: str,
    ) -> ToolResult:

        return ToolResult(
            success=False,
            name=self.name,
            content={
                "success": False,
                "error": {
                    "type": error_type,
                    "message": message,
                },
            },
            metadata={},
        )

    @staticmethod
    def _stringify_workdir(
        workdir: Path | None,
    ) -> str | None:

        if workdir is None:
            return None

        return str(workdir)

    @staticmethod
    def _duration_ms(
        started: float,
    ) -> int:

        return int((time.perf_counter() - started) * 1000)

    def __repr__(self) -> str:
        return "<Tool name='command_exec'>"
