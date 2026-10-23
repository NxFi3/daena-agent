"""Bounded operations, cancellation, cross-process desktop ownership and state IO."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time


class ComputerError(RuntimeError):
    def __init__(self, code, message, *, dispatched=False):
        super().__init__(message)
        self.code = code
        self.dispatched = dispatched


@dataclass
class Operation:
    deadline: float
    cancelled: threading.Event

    def check(self):
        if self.cancelled.is_set():
            raise ComputerError(
                "cancelled", "Operation cancelled; observe before deciding whether to retry."
            )
        if time.monotonic() >= self.deadline:
            raise ComputerError(
                "deadline_exceeded",
                "Operation deadline exceeded; inputs are never replayed automatically.",
            )

    def sleep(self, seconds):
        end = time.monotonic() + max(0, seconds)
        while time.monotonic() < end:
            self.check()
            time.sleep(min(0.05, max(0, end - time.monotonic())))
        self.check()


class StopToken:
    """Also accepts Daena's child cancellation token (which only has is_set)."""

    def __init__(self, local, parent=None):
        self.local, self.parent = local, parent

    def is_set(self):
        return self.local.is_set() or bool(self.parent and self.parent.is_set())


CURRENT: ContextVar[Operation | None] = ContextVar("computer_operation", default=None)


def checkpoint():
    op = CURRENT.get()
    if op:
        op.check()


def pause(seconds):
    op = CURRENT.get()
    if op:
        op.sleep(seconds)
    else:
        time.sleep(seconds)


@contextmanager
def operation(seconds, event):
    token = CURRENT.set(Operation(time.monotonic() + seconds, event))
    try:
        checkpoint()
        yield
    finally:
        CURRENT.reset(token)


def private_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def atomic_json(path, value):
    path = Path(path)
    private_directory(path.parent)
    fd, temporary = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_json(path, default=None):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError:
        return default
    except (ValueError, OSError) as exc:
        raise ComputerError("state_unreadable", f"Cannot read state {path}: {exc}") from exc


_THREAD_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def desktop_lock_path():
    # One physical desktop is shared by all agents/workspaces, not one lock per tool.
    identity = os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY") or "default"
    key = hashlib.sha256(identity.encode()).hexdigest()[:16]
    root = private_directory(Path.home() / ".cache" / "daena-computer-use")
    return root / f"desktop-{key}.lock"


@contextmanager
def desktop_lease(path=None):
    """Serialize capture+input+verification, including different Python processes."""
    path = Path(path or desktop_lock_path())
    private_directory(path.parent)
    with _LOCKS_GUARD:
        lock = _THREAD_LOCKS.setdefault(str(path.resolve()), threading.Lock())
    while not lock.acquire(timeout=0.05):
        checkpoint()
    fd = None
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        if os.name == "posix":
            import fcntl

            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    pause(0.05)
        else:
            import msvcrt

            if os.fstat(fd).st_size == 0:
                os.write(fd, b"0")
            while True:
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    pause(0.05)
        checkpoint()
        yield
    finally:
        if fd is not None:
            os.close(fd)
        lock.release()
