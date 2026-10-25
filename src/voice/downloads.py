"""Resumable transfers for immutable model URLs. Never publish partial weights."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

from .providers import VoiceError, _check_stop


@contextmanager
def _lock(path):
    if path.is_symlink():
        raise VoiceError("Download lock must not be a symbolic link")
    with path.open("a+b") as stream:
        stream.seek(0)
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise VoiceError(
                "This model file is already downloading in another process"
            ) from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _digest(path, stop):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            _check_stop(stop)
            digest.update(block)
    return digest.hexdigest()


def transfer(
    client,
    url,
    cache,
    *,
    size=None,
    sha256=None,
    headers=None,
    stop_event=None,
    progress=None,
    retries=3,
):
    """Retain partial bytes on cancellation/network failure across app restarts.

    Callers must supply an immutable commit/asset URL. Resume requires a correct
    206 Content-Range. A server ignoring Range is rejected without destroying the
    existing prefix or silently spending bandwidth on a fresh full download.
    """
    import httpx

    _check_stop(stop_event)
    identity = json.dumps([url, size, sha256, headers], sort_keys=True)
    root = Path(cache).resolve()
    root.mkdir(parents=True, exist_ok=True)
    folder = root / hashlib.sha256(identity.encode()).hexdigest()
    folder.mkdir(exist_ok=True)
    for name in ("payload.part", "payload.complete", "validator.json", "transfer.lock"):
        if (
            not (folder / name).resolve().is_relative_to(root)
            or (folder / name).is_symlink()
        ):
            raise VoiceError("Download cache path escapes its selected directory")
    part, complete, validator = (
        folder / name for name in ("payload.part", "payload.complete", "validator.json")
    )
    limit = size if size is not None else 4 * 1024**3
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 0 <= limit <= 4 * 1024**3
    ):
        raise VoiceError("Model download has an unsupported size")
    with _lock(folder / "transfer.lock"):
        if complete.is_file():
            checksum = _digest(complete, stop_event)
            try:
                recorded = json.loads(validator.read_text()).get("complete_sha256")
            except (OSError, ValueError, AttributeError):
                recorded = None
            if (size is None or complete.stat().st_size == size) and checksum == (
                sha256 or recorded
            ):
                if progress:
                    progress(
                        {
                            "stage": "cached",
                            "received": complete.stat().st_size,
                            "total": size,
                        }
                    )
                return complete
            complete.unlink()
        for attempt in range(retries):
            _check_stop(stop_event)
            offset = part.stat().st_size if part.exists() else 0
            if offset > limit:
                raise VoiceError(
                    "Partial model exceeds its declared size; cache was preserved"
                )
            try:
                state = json.loads(validator.read_text())
                if not isinstance(state, dict):
                    state = {}
            except (OSError, ValueError):
                state = {}
            # A fully received file can have been canceled just before final
            # verification. Do not request bytes beyond the end (HTTP 416).
            finished = part.is_file() and (
                (size is not None and offset == size)
                or (state.get("finished_size") == offset)
            )
            if not finished:
                request_headers = {**(headers or {}), "Accept-Encoding": "identity"}
                if offset:
                    request_headers["Range"] = f"bytes={offset}-"
                    if state.get("etag"):
                        request_headers["If-Range"] = state["etag"]
                if progress:
                    progress(
                        {
                            "stage": "resuming" if offset else "connecting",
                            "received": offset,
                            "total": size,
                        }
                    )
                try:
                    with client.stream("GET", url, headers=request_headers) as response:
                        response.raise_for_status()
                        if (
                            response.headers.get("content-encoding", "identity")
                            != "identity"
                        ):
                            raise VoiceError(
                                "Model source returned encoded bytes; partial download was preserved"
                            )
                        total = size
                        response_end = None
                        if offset:
                            match = re.fullmatch(
                                r"bytes (\d+)-(\d+)/(\d+)",
                                response.headers.get("content-range", ""),
                            )
                            if response.status_code != 206 or not match:
                                raise VoiceError(
                                    "Model source did not honor Resume (HTTP Range). Partial bytes were preserved; no restart from zero was performed"
                                )
                            start, end, reported_total = map(int, match.groups())
                            if (
                                start != offset
                                or end < start
                                or end >= reported_total
                                or (size is not None and reported_total != size)
                                or reported_total > limit
                            ):
                                raise VoiceError(
                                    "Invalid model Content-Range; partial bytes were preserved"
                                )
                            total = reported_total
                            response_end = end + 1
                        elif response.status_code != 200:
                            raise VoiceError(
                                "Model source returned an unexpected initial response"
                            )
                        etag = response.headers.get("etag")
                        if (
                            offset
                            and state.get("etag")
                            and etag
                            and etag != state["etag"]
                        ):
                            raise VoiceError(
                                "Model source changed its validator; partial bytes were preserved"
                            )
                        if etag and not etag.startswith("W/"):
                            state["etag"] = etag
                        validator.write_text(json.dumps(state))
                        received = offset
                        with part.open("ab", buffering=0) as output:
                            try:
                                for block in response.iter_bytes(chunk_size=64 * 1024):
                                    _check_stop(stop_event)
                                    if (
                                        received + len(block) > limit
                                        or (
                                            total is not None
                                            and received + len(block) > total
                                        )
                                        or (
                                            response_end is not None
                                            and received + len(block) > response_end
                                        )
                                    ):
                                        raise VoiceError(
                                            "Model download exceeded its declared size"
                                        )
                                    output.write(block)
                                    received += len(block)
                                    if progress:
                                        progress(
                                            {
                                                "stage": "downloading",
                                                "received": received,
                                                "total": total,
                                            }
                                        )
                                if total is not None and received != total:
                                    raise httpx.ReadError("Model stream ended early")
                                state["finished_size"] = received
                                validator.write_text(json.dumps(state))
                            finally:
                                os.fsync(output.fileno())
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    if isinstance(
                        exc, httpx.HTTPStatusError
                    ) and exc.response.status_code not in {
                        408,
                        429,
                        500,
                        502,
                        503,
                        504,
                    }:
                        raise
                    if attempt + 1 == retries:
                        raise VoiceError(
                            f"Network download stopped; partial bytes saved at {part}. Retry the same model to resume. {exc}"
                        ) from exc
                    if progress:
                        progress(
                            {
                                "stage": "retrying",
                                "received": part.stat().st_size if part.exists() else 0,
                                "total": size,
                            }
                        )
                    delay = 0.5 * (attempt + 1)
                    if stop_event is not None:
                        stop_event.wait(delay)
                    else:
                        time.sleep(delay)
                    continue
            _check_stop(stop_event)
            checksum = _digest(part, stop_event)
            if sha256 and checksum != sha256:
                part.unlink()
                raise VoiceError(
                    "Model download checksum mismatch; invalid bytes discarded, previous installation preserved"
                )
            state["complete_sha256"] = checksum
            validator.write_text(json.dumps(state))
            os.replace(part, complete)
            return complete
    raise VoiceError("Model transfer did not complete")


def materialize(source, destination):
    """Copy verified bytes atomically; published files never share writable inodes."""
    fd, temporary = tempfile.mkstemp(prefix=".voice-download-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
