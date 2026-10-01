from __future__ import annotations

import atexit
import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO


@dataclass
class ManagedProcess:
    process_id: str
    command: list[str]
    workdir: Path | None
    process: subprocess.Popen
    stdout_file: IO[bytes]
    stderr_file: IO[bytes]
    started_at: float
    stdout_offset: int = 0
    stderr_offset: int = 0
    pipe_stdin: bool = False
    status: str = "running"
    exit_code: int | None = None





def _set_parent_death_signal() -> None:
    """
    On Linux, make managed children terminate when the owning Python process
    disappears unexpectedly (for example SIGKILL on a Jupyter kernel).

    Windows keeps its existing process-group cleanup path.
    """
    if os.name != "posix":
        return

    try:
        import ctypes

        libc = ctypes.CDLL(None)
        libc.prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG
    except Exception:
        # Normal runtime cleanup still handles managed processes when close()
        # is reached; this hook is only an extra safety net for abrupt exits.
        return


class ProcessManager:
    """Own the lifecycle of Daena-managed local processes.

    A process can remain alive after its initial command call returns. The
    manager gives it an opaque process_id so later tool calls can poll, write
    to stdin, or terminate it without relying on a raw PID.
    """

    MAX_RETAINED_FINISHED = 64

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._processes: dict[str, ManagedProcess] = {}

    def start(
        self,
        *,
        command: list[str],
        workdir: Path | None,
        yield_time_ms: int,
        max_output_chars: int,
        pipe_stdin: bool = True,
    ) -> dict:
        # Reuse an already-running identical process instead of spawning a
        # second copy. The agent should poll the existing process.
        with self._lock:
            existing = self._find_running_locked(
                command=command,
                workdir=workdir,
                pipe_stdin=pipe_stdin,
            )
            if existing is not None:
                self._refresh_locked(existing)
                output = self._read_incremental(existing, max_output_chars)
                return {
                    **self._entry_result(existing, output),
                    "reused_existing_process": True,
                }

        stdout_file = tempfile.TemporaryFile(mode="w+b")
        stderr_file = tempfile.TemporaryFile(mode="w+b")

        try:
            process = self._spawn(
                command=command,
                workdir=workdir,
                stdin=subprocess.PIPE if pipe_stdin else subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
            )
        except Exception:
            stdout_file.close()
            stderr_file.close()
            raise

        started = time.perf_counter()

        try:
            process.wait(timeout=yield_time_ms / 1000.0)
        except subprocess.TimeoutExpired:
            pass

        if process.poll() is not None:
            entry = ManagedProcess(
                process_id="",
                command=list(command),
                workdir=workdir,
                process=process,
                stdout_file=stdout_file,
                stderr_file=stderr_file,
                started_at=started,
                pipe_stdin=pipe_stdin,
                status="exited",
                exit_code=process.returncode,
            )
            output = self._read_incremental(entry, max_output_chars)
            stdout_file.close()
            stderr_file.close()
            return {
                "status": "exited",
                "process_id": None,
                "pid": process.pid,
                "exit_code": process.returncode,
                "stdout": output["stdout"],
                "stderr": output["stderr"],
                "duration_ms": self._duration_ms(started),
            }

        process_id = f"proc-{os.urandom(6).hex()}"

        entry = ManagedProcess(
            process_id=process_id,
            command=list(command),
            workdir=workdir,
            process=process,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            started_at=started,
            pipe_stdin=pipe_stdin,
        )

        with self._lock:
            # A defensive collision check; process ids are intentionally opaque.
            while process_id in self._processes:
                process_id = f"proc-{os.urandom(6).hex()}"
                entry.process_id = process_id

            self._processes[process_id] = entry

            self._refresh_locked(entry)
            output = self._read_incremental(entry, max_output_chars)
            self._prune_finished_locked()

        return {
            "status": entry.status,
            "command": list(entry.command),
            "workdir": str(entry.workdir) if entry.workdir is not None else None,
            "process_id": entry.process_id,
            "pid": entry.process.pid,
            "exit_code": entry.exit_code,
            "stdout": output["stdout"],
            "stderr": output["stderr"],
            "stdout_truncated": bool(output.get("stdout_truncated", False)),
            "stderr_truncated": bool(output.get("stderr_truncated", False)),
            "duration_ms": self._duration_ms(started),
        }

    def poll(
        self,
        *,
        process_id: str,
        wait_ms: int,
        max_output_chars: int,
    ) -> dict:
        deadline = time.perf_counter() + wait_ms / 1000.0

        while True:
            with self._lock:
                entry = self._processes.get(process_id)

                if entry is None:
                    return {
                        "status": "unknown",
                        "process_id": process_id,
                        "error": "Unknown process_id.",
                    }

                self._refresh_locked(entry)
                output = self._read_incremental(entry, max_output_chars)

                if entry.status != "running":
                    return self._entry_result(entry, output)

                if output["stdout"] or output["stderr"]:
                    return self._entry_result(entry, output)

            if time.perf_counter() >= deadline:
                with self._lock:
                    entry = self._processes.get(process_id)
                    if entry is None:
                        return {
                            "status": "unknown",
                            "process_id": process_id,
                            "error": "Unknown process_id.",
                        }
                    self._refresh_locked(entry)
                    output = self._read_incremental(entry, max_output_chars)
                    return self._entry_result(entry, output)

            time.sleep(min(0.05, max(0.0, deadline - time.perf_counter())))

    def write(
        self,
        *,
        process_id: str,
        input_text: str,
    ) -> dict:
        with self._lock:
            entry = self._processes.get(process_id)

            if entry is None:
                return {
                    "status": "unknown",
                    "process_id": process_id,
                    "error": "Unknown process_id.",
                }

            self._refresh_locked(entry)

            if entry.status != "running":
                return {
                    "status": entry.status,
                    "process_id": process_id,
                    "error": "Process is not running.",
                    "exit_code": entry.exit_code,
                }

            if entry.process.stdin is None:
                return {
                    "status": "failed",
                    "process_id": process_id,
                    "error": "Process stdin is not writable.",
                }

            try:
                entry.process.stdin.write(input_text.encode("utf-8"))
                entry.process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._refresh_locked(entry)
                return {
                    "status": entry.status if entry.status != "running" else "failed",
                    "process_id": process_id,
                    "error": str(exc),
                    "exit_code": entry.exit_code,
                }

            return {
                "status": "accepted",
                "process_id": process_id,
            }

    def stop(
        self,
        *,
        process_id: str,
    ) -> dict:
        with self._lock:
            entry = self._processes.get(process_id)

            if entry is None:
                return {
                    "status": "unknown",
                    "process_id": process_id,
                    "error": "Unknown process_id.",
                }

            self._refresh_locked(entry)

            if entry.status == "running":
                self._terminate_process_tree(entry.process)
                self._refresh_locked(entry)

                if entry.status == "running":
                    entry.status = "terminated"

            output = self._read_incremental(entry, 8_000)

            return self._entry_result(entry, output)

    def close(self) -> None:
        with self._lock:
            for entry in list(self._processes.values()):
                if entry.status == "running":
                    self._terminate_process_tree(entry.process)
                    self._refresh_locked(entry)

            self._prune_finished_locked()

    def _spawn(
        self,
        *,
        command: list[str],
        workdir: Path | None,
        stdin,
        stdout,
        stderr,
    ) -> subprocess.Popen:
        kwargs = {
            "args": command,
            "cwd": str(workdir) if workdir is not None else None,
            "stdin": stdin,
            "stdout": stdout,
            "stderr": stderr,
            "text": False,
        }

        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
            kwargs["preexec_fn"] = _set_parent_death_signal

        return subprocess.Popen(**kwargs)

    def _find_running_locked(
        self,
        *,
        command: list[str],
        workdir: Path | None,
        pipe_stdin: bool,
    ) -> ManagedProcess | None:
        normalized_workdir = str(workdir) if workdir is not None else None
        for entry in self._processes.values():
            if entry.status != "running":
                continue
            if entry.command != list(command):
                continue
            entry_workdir = str(entry.workdir) if entry.workdir is not None else None
            if entry_workdir != normalized_workdir:
                continue
            if entry.pipe_stdin != pipe_stdin:
                continue
            return entry
        return None

    def _refresh_locked(self, entry: ManagedProcess) -> None:
        exit_code = entry.process.poll()

        if exit_code is None:
            entry.status = "running"
            return

        entry.exit_code = exit_code

        if entry.status == "terminated":
            return

        entry.status = "exited"

    @staticmethod
    def _read_file_incremental(
        file: IO[bytes],
        offset: int,
        max_bytes: int,
    ) -> tuple[str, int, bool]:
        file.seek(0, os.SEEK_END)
        end = file.tell()
        if end <= offset:
            return "", end, False

        available = end - offset
        if available <= max_bytes:
            file.seek(offset)
            data = file.read(available)
            return data.decode("utf-8", errors="replace"), end, False

        # Keep both the beginning and end of each newly observed chunk. This
        # preserves command headers and the final traceback/test summary.
        head_bytes = max(1, int(max_bytes * 0.60))
        tail_bytes = max(1, max_bytes - head_bytes)
        file.seek(offset)
        head = file.read(head_bytes)
        file.seek(max(offset + head_bytes, end - tail_bytes))
        tail = file.read(tail_bytes)
        data = head + b"\n\n... output omitted ...\n\n" + tail
        return data.decode("utf-8", errors="replace"), end, True

    def _read_incremental(
        self,
        entry: ManagedProcess,
        max_output_chars: int,
    ) -> dict[str, object]:
        max_bytes = max(1, max_output_chars)

        stdout, entry.stdout_offset, stdout_truncated = self._read_file_incremental(
            entry.stdout_file, entry.stdout_offset, max_bytes
        )
        stderr, entry.stderr_offset, stderr_truncated = self._read_file_incremental(
            entry.stderr_file, entry.stderr_offset, max_bytes
        )

        return {
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }

    def _entry_result(
        self,
        entry: ManagedProcess,
        output: dict[str, str],
    ) -> dict:
        return {
            "status": entry.status,
            "command": list(entry.command),
            "workdir": str(entry.workdir) if entry.workdir is not None else None,
            "process_id": entry.process_id,
            "pid": entry.process.pid,
            "exit_code": entry.exit_code,
            "stdout": output["stdout"],
            "stderr": output["stderr"],
            "duration_ms": self._duration_ms(entry.started_at),
        }

    def _prune_finished_locked(self) -> None:
        finished = [
            entry
            for entry in self._processes.values()
            if entry.status != "running"
        ]

        overflow = len(finished) - self.MAX_RETAINED_FINISHED

        if overflow <= 0:
            return

        finished.sort(key=lambda item: item.started_at)

        for entry in finished[:overflow]:
            self._processes.pop(entry.process_id, None)
            try:
                entry.stdout_file.close()
            finally:
                entry.stderr_file.close()

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return

        try:
            if os.name == "nt":
                try:
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                except (OSError, ValueError):
                    pass

                try:
                    process.wait(timeout=2.0)
                    return
                except subprocess.TimeoutExpired:
                    pass

                try:
                    process.kill()
                except (ProcessLookupError, OSError):
                    pass

                try:
                    process.wait(timeout=2.0)
                except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                    pass

                return

            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

            try:
                process.wait(timeout=2.0)
                return
            except subprocess.TimeoutExpired:
                pass

            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

            try:
                process.wait(timeout=2.0)
            except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                pass

        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill()
            except (ProcessLookupError, PermissionError, OSError):
                pass

    @staticmethod
    def _duration_ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)


PROCESS_MANAGER = ProcessManager()
atexit.register(PROCESS_MANAGER.close)
