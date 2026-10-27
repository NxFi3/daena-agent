"""Network interruptions preserve verified prefixes without publishing partials."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread

import httpx
import pytest

from src.voice.downloads import transfer
from src.voice.providers import VoiceError, VoiceCancelled


BODY = b"a" * 65536 + b"b" * 65536 + b"c" * 16000
SHA = hashlib.sha256(BODY).hexdigest()


class Broken(httpx.SyncByteStream):
    def __iter__(self):
        yield BODY[:65536]
        raise httpx.ReadTimeout("temporary stall")


def test_timeout_keeps_bytes_and_new_attempt_resumes_exact_offset(tmp_path):
    requests = []

    def handle(request):
        requests.append(dict(request.headers))
        if len(requests) == 1:
            return httpx.Response(200, headers={"ETag": '"v1"'}, stream=Broken())
        assert request.headers["Range"] == "bytes=65536-"
        assert request.headers["If-Range"] == '"v1"'
        return httpx.Response(
            206,
            headers={
                "Content-Range": f"bytes 65536-{len(BODY) - 1}/{len(BODY)}",
                "ETag": '"v1"',
            },
            content=BODY[65536:],
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(VoiceError, match="partial bytes saved"):
            transfer(
                client,
                "https://example.org/immutable/revision/model",
                tmp_path,
                size=len(BODY),
                sha256=SHA,
                retries=1,
            )
    assert next(tmp_path.rglob("payload.part")).read_bytes() == BODY[:65536]
    # A new client/process does not need in-memory state to resume.
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = transfer(
            client,
            "https://example.org/immutable/revision/model",
            tmp_path,
            size=len(BODY),
            sha256=SHA,
        )
    assert result.read_bytes() == BODY and len(requests) == 2


@pytest.mark.parametrize("size", [8, None])
def test_cancel_retains_even_fully_received_partial_and_verifies_without_redownload(
    tmp_path,
    size,
):
    stop = Event()
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"complete")
        )
    ) as client:
        with pytest.raises(VoiceCancelled):
            transfer(
                client,
                "https://example.org/id",
                tmp_path,
                size=size,
                stop_event=stop,
                progress=lambda event: (
                    stop.set() if event["stage"] == "downloading" else None
                ),
            )
    assert next(tmp_path.rglob("payload.part")).read_bytes() == b"complete"
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: pytest.fail("unnecessary network request")
        )
    ) as client:
        assert (
            transfer(client, "https://example.org/id", tmp_path, size=size).read_bytes()
            == b"complete"
        )


@pytest.mark.parametrize(
    "status,headers",
    [
        (200, {}),
        (206, {"Content-Range": f"bytes 0-{len(BODY) - 1}/{len(BODY)}"}),
        (
            206,
            {
                "Content-Range": f"bytes 65536-{len(BODY) - 1}/{len(BODY)}",
                "ETag": '"changed"',
            },
        ),
    ],
)
def test_invalid_resume_never_overwrites_or_appends_to_saved_prefix(
    tmp_path, status, headers
):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"ETag": '"v1"'}, stream=Broken()
            )
        )
    ) as client:
        with pytest.raises(VoiceError):
            transfer(
                client, "https://example.org/id", tmp_path, size=len(BODY), retries=1
            )
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status, headers=headers, content=BODY)
        )
    ) as client:
        with pytest.raises(VoiceError):
            transfer(client, "https://example.org/id", tmp_path, size=len(BODY))
    assert next(tmp_path.rglob("payload.part")).read_bytes() == BODY[:65536]


def test_real_http_connection_drop_retries_with_range_and_preserves_hash(tmp_path):
    ranges = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            value = self.headers.get("Range")
            ranges.append(value)
            offset = int(value[6:-1]) if value else 0
            self.send_response(206 if value else 200)
            self.send_header("Content-Length", str(len(BODY) - offset))
            if value:
                self.send_header(
                    "Content-Range", f"bytes {offset}-{len(BODY) - 1}/{len(BODY)}"
                )
            self.end_headers()
            self.wfile.write(BODY[offset:] if value else BODY[:65536])
            self.wfile.flush()
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(trust_env=False, timeout=2) as client:
            result = transfer(
                client,
                f"http://127.0.0.1:{server.server_port}/pinned",
                tmp_path,
                size=len(BODY),
                sha256=SHA,
            )
        assert result.read_bytes() == BODY and ranges == [None, "bytes=65536-"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_completed_cache_is_checked_and_empty_assets_are_supported(tmp_path):
    calls = []

    def handle(request):
        calls.append(1)
        return httpx.Response(200, content=b"")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        first = transfer(client, "https://example.org/empty", tmp_path, size=0)
        assert transfer(client, "https://example.org/empty", tmp_path, size=0) == first
    assert first.read_bytes() == b"" and len(calls) == 1


def test_corrupt_final_hash_is_not_published_as_resumable_valid_content(tmp_path):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"invalid")
        )
    ) as client:
        with pytest.raises(VoiceError, match="checksum mismatch"):
            transfer(
                client, "https://example.org/model", tmp_path, size=7, sha256="a" * 64
            )
    assert not list(tmp_path.rglob("payload.complete"))
    assert not list(tmp_path.rglob("payload.part"))


@pytest.mark.parametrize("source", ["huggingface", "github-release"])
def test_model_installer_resumes_canceled_transfer_without_publishing_partial(
    tmp_path, monkeypatch, source
):
    from src.voice import models

    spec = models.ModelSpec(
        "test-resume",
        "test",
        "stt",
        "en",
        "test/model",
        "pinned",
        ("model.bin",),
        "model.bin",
        source,
        (("model.bin", 123),) if source == "github-release" else (),
    )
    monkeypatch.setitem(models._SPEC_BY_KEY, spec.key, spec)
    requests = []

    def handle(request):
        if "/api/models/" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "sha": "a" * 40,
                    "siblings": [
                        {
                            "rfilename": "model.bin",
                            "size": len(BODY),
                            "lfs": {"sha256": SHA},
                        }
                    ],
                },
            )
        if request.headers.get("Accept") == "application/vnd.github+json":
            return httpx.Response(
                200,
                json={
                    "id": 123,
                    "name": "model.bin",
                    "size": len(BODY),
                    "digest": "sha256:" + SHA,
                },
            )
        value = request.headers.get("Range")
        requests.append(value)
        if value:
            assert value == "bytes=65536-"
            return httpx.Response(
                206,
                headers={"Content-Range": f"bytes 65536-{len(BODY) - 1}/{len(BODY)}"},
                content=BODY[65536:],
            )
        return httpx.Response(200, content=BODY)

    stop = Event()
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(VoiceCancelled):
            models.download_model(
                spec.key,
                tmp_path,
                client=client,
                stop_event=stop,
                progress=lambda event: (
                    stop.set() if event["stage"] == "downloading" else None
                ),
            )
        assert not (tmp_path / "installed" / "test-resume.json").exists()
        assert next(tmp_path.rglob("payload.part")).stat().st_size == 65536
        result = models.download_model(spec.key, tmp_path, client=client)
    from pathlib import Path

    assert Path(result["path"]).read_bytes() == BODY
    assert requests == [None, "bytes=65536-"]


def test_concurrent_process_contract_rejects_second_writer(tmp_path):
    def progress(event):
        if event["stage"] == "connecting":
            with pytest.raises(VoiceError, match="already downloading"):
                transfer(client, "https://example.org/pinned", tmp_path, size=1)

    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"a"))
    ) as client:
        assert (
            transfer(
                client,
                "https://example.org/pinned",
                tmp_path,
                size=1,
                progress=progress,
            ).read_bytes()
            == b"a"
        )
