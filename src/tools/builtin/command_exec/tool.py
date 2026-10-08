from __future__ import annotations

import errno
import subprocess
import time
from pathlib import Path
from typing import Any

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool
from src.tools.builtin.command_exec.process_manager import PROCESS_MANAGER

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
        - Yield-based managed process execution.
        - Process-group termination for managed process cleanup.
        - Bounded stdout/stderr.
        - Structured JSON-compatible ToolResult.content.
        - No full stdout/stderr duplication in metadata.
        - Human-readable exit-code hints for known tools.
    """

    # ToolManager replaces this with its session-scoped ProcessManager.
    process_manager = PROCESS_MANAGER

    name = "command_exec"
    action = "run"

    DEFAULT_YIELD_TIME_MS = 1_000
    MAX_YIELD_TIME_MS = 30_000

    DEFAULT_MAX_OUTPUT_CHARS = 8_000
    MAX_OUTPUT_CHARS = 32_000
    MIN_OUTPUT_CHARS = 512

    BACKGROUND_STARTUP_GRACE_MS = 150

    def __init__(self) -> None:
        self._workspace_root: Path | None = None

    def set_workspace(self, directory: str | Path) -> None:
        self._workspace_root = Path(directory).expanduser().resolve()

    description = (
        "Run a local command given as an argv array (not a shell command string). "
        "Shell operators such as >, |, &&, ||, $(...), heredocs, and shell globbing are "
        "not interpreted unless you explicitly invoke a shell executable such as sh or bash. "
        "Use apply_patch for file creation/editing instead of relying on echo output. "
        "For directory inspection, prefer one concise listing when the workspace "
        "inventory is insufficient; do not repeat an identical successful ls/find "
        "observation. The command waits up to yield_time_ms (default 1000ms). Optional numeric "
        "arguments are normalized to their declared bounds before execution. If it exits in that "
        "window, the result contains exit_code and completed output. If it is still "
        "running, the process is kept alive and the result returns status=running "
        "with an opaque process_id. Use process_poll to wait for completion or inspect "
        "new output, process_write for stdin, and process_stop to terminate it. "
        "A slow one-shot command and a long-lived server use the same process lifecycle; "
        "do not treat elapsed time as proof that a command is a server.\n"
        "\n"
        "Set background=true only when you want the command to return immediately "
        "without the initial yield wait. Set pipe_stdin=true for non-TTY processes that "
        "need later stdin input through process_write. Set tty=true for terminal-aware "
        "interactive programs such as ssh, sudo, password prompts, shells, and REPLs. "
        "TTY processes keep a real pseudo-terminal and can be written to with process_write. "
        "Managed processes are never killed merely because they remain alive.\n"
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
                    "Working directory relative to the active workspace. Defaults to "
                    "the workspace root ('.')."
                ),
            },
            "yield_time_ms": {
                "type": "integer",
                "description": (
                    "How long to wait synchronously before returning a managed "
                    "process_id when the command is still running."
                ),
                "default": DEFAULT_YIELD_TIME_MS,
                "minimum": 0,
                "maximum": MAX_YIELD_TIME_MS,
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
                    "Start the command without the initial yield wait."
                ),
                "default": False,
            },
            "pipe_stdin": {
                "type": "boolean",
                "description": (
                    "Keep stdin writable for a non-TTY process through process_write. "
                    "Ignored when tty=true."
                ),
                "default": False,
            },
            "tty": {
                "type": "boolean",
                "description": (
                    "Run the command inside a real pseudo-terminal (PTY). Use this "
                    "for terminal-aware interactive programs that need TTY input, "
                    "password prompts, shells, REPLs, or terminal control. PTY input "
                    "remains writable through process_write."
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
        workdir: str | None = ".",
        yield_time_ms: int = DEFAULT_YIELD_TIME_MS,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        background: bool = False,
        pipe_stdin: bool = False,
        tty: bool = False,
    ) -> ToolResult:

        validation_error = self._validate_arguments(
            command=command,
            yield_time_ms=yield_time_ms,
            max_output_chars=max_output_chars,
            background=background,
            pipe_stdin=pipe_stdin,
            tty=tty,
        )

        if validation_error is not None:
            return validation_error

        resolved_workdir, workdir_error = self._resolve_workdir(workdir)

        if workdir_error is not None:
            return workdir_error

        return self._execute_managed(
            command=command,
            workdir=resolved_workdir,
            yield_time_ms=(0 if background else yield_time_ms),
            max_output_chars=max_output_chars,
            pipe_stdin=pipe_stdin,
            tty=tty,
            background=background,
        )

    def _validate_arguments(
        self,
        *,
        command: Any,
        yield_time_ms: Any,
        max_output_chars: Any,
        background: Any,
        pipe_stdin: Any,
        tty: Any,
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

        if isinstance(yield_time_ms, bool) or not isinstance(yield_time_ms, int):
            return self._error(
                error_type="invalid_argument",
                message="yield_time_ms must be an integer.",
            )

        if not (0 <= yield_time_ms <= self.MAX_YIELD_TIME_MS):
            return self._error(
                error_type="invalid_argument",
                message=(
                    "yield_time_ms must be between 0 and "
                    f"{self.MAX_YIELD_TIME_MS}."
                ),
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

        if not isinstance(pipe_stdin, bool):
            return self._error(
                error_type="invalid_argument",
                message="pipe_stdin must be a boolean.",
            )

        if not isinstance(tty, bool):
            return self._error(
                error_type="invalid_argument",
                message="tty must be a boolean.",
            )

        return None

    def _execute_managed(
        self,
        *,
        command: list[str],
        workdir: Path | None,
        yield_time_ms: int,
        max_output_chars: int,
        pipe_stdin: bool = False,
        tty: bool = False,
        background: bool = False,
    ) -> ToolResult:
        started = time.perf_counter()

        try:
            result = self.process_manager.start(
                command=command,
                workdir=workdir,
                yield_time_ms=yield_time_ms,
                max_output_chars=max_output_chars,
                pipe_stdin=pipe_stdin,
                tty=tty,
                background=background,
            )
        except FileNotFoundError as exc:
            return self._execution_error(
                command=command,
                workdir=workdir,
                started=started,
                error_type="command_not_found",
                message=str(exc),
            )
        except NotImplementedError as exc:
            return self._execution_error(
                command=command,
                workdir=workdir,
                started=started,
                error_type="unsupported_tty",
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
            error_type = "resource_in_use" if exc.errno == errno.EADDRINUSE else "execution_error"
            extra = {
                "resource": "address"
            } if exc.errno == errno.EADDRINUSE else None
            return self._execution_error(
                command=command,
                workdir=workdir,
                started=started,
                error_type=error_type,
                message=str(exc),
                extra=extra,
            )
        except Exception as exc:
            return self._execution_error(
                command=command,
                workdir=workdir,
                started=started,
                error_type="unexpected_error",
                message=str(exc),
            )

        status = result.get("status")
        # Launching a managed process is not the same as completing the command.
        # A running process must be resolved by process_poll/process_stop.
        success = status == "exited" and result.get("exit_code") == 0

        hint = self._exit_code_hint(command, result.get("exit_code"))

        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        failure_type = ""
        if not success and status in {"exited", "terminated"}:
            failure_type = self._classify_failure(stdout=stdout, stderr=stderr)

        content = {
            "success": success,
            "command": command,
            "workdir": self._stringify_workdir(workdir),
            "exit_code": result.get("exit_code"),
            "exit_code_hint": hint,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": False,
            # Preserve the caller's explicit intent. A long-running
            # foreground command is not a background task just because it
            # crossed the synchronous yield boundary.
            "background": bool(background),
            "managed": True,
            "operation_complete": status == "exited",
            "process_state": status,
            "status": status,
            "process_id": result.get("process_id"),
            "pid": result.get("pid"),
            "tty": bool(result.get("tty", tty)),
            "reused_existing_process": result.get(
                "reused_existing_process",
                False,
            ),
            "duration_ms": result.get(
                "duration_ms",
                self._duration_ms(started),
            ),
            "stdout_truncated": False,
            "stderr_truncated": False,
        }

        if failure_type:
            diagnostic = self._failure_diagnostic(
                stdout=stdout,
                stderr=stderr,
            )
            content["error"] = {
                "type": failure_type,
                "message": diagnostic
                or "The process reported a known execution failure.",
            }

        return ToolResult(
            success=success,
            name=self.name,
            content=content,
            metadata={},
        )

    @staticmethod
    def _classify_failure(
        *,
        stdout: Any,
        stderr: Any,
    ) -> str:
        text = "\n".join(
            value for value in (stdout, stderr) if isinstance(value, str)
        )
        lowered = text.lower()
        if "eaddrinuse" in lowered or "address already in use" in lowered:
            return "resource_in_use"
        if "permission denied" in lowered:
            return "permission_error"
        if "command not found" in lowered or "not found" in lowered and "sh:" in lowered:
            return "command_not_found"
        return ""

    @staticmethod
    def _failure_diagnostic(
        *,
        stdout: Any,
        stderr: Any,
        limit: int = 600,
    ) -> str:
        """Return concrete, model-useful diagnostics for a failed process."""
        sources = [
            stderr if isinstance(stderr, str) else "",
            stdout if isinstance(stdout, str) else "",
        ]

        selected: list[str] = []
        for source in sources:
            for raw_line in source.splitlines():
                line = raw_line.strip()
                if not line or line in selected:
                    continue

                lowered = line.lower()
                if any(
                    marker in lowered
                    for marker in (
                        "error",
                        "fail",
                        "exception",
                        "traceback",
                        "assert",
                        "expected",
                        "received",
                        "address already in use",
                        "not found",
                        "permission denied",
                        "panic",
                    )
                ):
                    selected.append(line)

                if len(selected) >= 4:
                    break

            if len(selected) >= 4:
                break

        if not selected:
            for source in sources:
                lines = [line.strip() for line in source.splitlines() if line.strip()]
                if lines:
                    selected.append(lines[-1])
                    break

        diagnostic = " | ".join(selected)
        if len(diagnostic) <= limit:
            return diagnostic
        return diagnostic[: limit - 3].rstrip() + "..."

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

    def _resolve_workdir(
        self,
        workdir: str | None,
    ) -> tuple[
        Path | None,
        ToolResult | None,
    ]:

        if workdir is None:
            return self._workspace_root, None

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
            return self._workspace_root, None

        path = Path(workdir).expanduser()
        if not path.is_absolute() and self._workspace_root is not None:
            path = self._workspace_root / path

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
