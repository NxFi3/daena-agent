"""Explicit offline-model setup never publishes partial or untrusted assets."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
from threading import Event

import httpx
import pytest

from src.voice.models import download_model, installed_models, model_catalog
from src.voice.providers import VoiceCancelled, VoiceError


def archive_bytes(members):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:bz2") as archive:
        for name, data, kind in members:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(data) if kind == tarfile.REGTYPE else 0
            if kind == tarfile.SYMTYPE:
                member.linkname = "../../outside"
            archive.addfile(member, io.BytesIO(data))
    return output.getvalue()


def fixture_client(key, *, corrupt=False, wrong_identity=False, archive_members=None):
    spec = next(row for row in model_catalog() if row["key"] == key)
    if key == "audio-events":
        members = (
            archive_members
            if archive_members is not None
            else [
                (spec["files"][0], b"int8 model", tarfile.REGTYPE),
                (
                    spec["files"][1],
                    b"index,mid,display_name\n0,m,speech\n",
                    tarfile.REGTYPE,
                ),
                ("unused-fp32.onnx", b"not extracted", tarfile.REGTYPE),
            ]
        )
        bodies = {spec["assets"][0][0]: archive_bytes(members)}
    else:
        bodies = {name: f"model bytes {name}".encode() for name, _ in spec["assets"]}
    requests = []

    def handle(request):
        asset_id = int(request.url.path.rsplit("/", 1)[1])
        name = next(
            name for name, identifier in spec["assets"] if identifier == asset_id
        )
        body = bodies[name]
        requests.append((asset_id, request.headers["accept"]))
        assert request.url.host == "api.github.com"
        if request.headers["accept"] == "application/vnd.github+json":
            return httpx.Response(
                200,
                json={
                    "id": asset_id + int(wrong_identity),
                    "name": name,
                    "size": len(body),
                    "digest": "sha256:" + hashlib.sha256(body).hexdigest(),
                    "browser_download_url": "https://untrusted.example/never-follow-this",
                },
            )
        assert request.headers["accept"] == "application/octet-stream"
        if corrupt:
            body = b"x" * len(body)
        return httpx.Response(200, content=body)

    return httpx.Client(transport=httpx.MockTransport(handle)), requests


def test_catalog_separates_languages_capabilities_and_model_terms():
    catalog = {row["key"]: row for row in model_catalog()}
    assert set(catalog) == {
        "whisper-turbo",
        "whisper-small",
        "piper-fa",
        "piper-en",
        "kokoro-en",
        "speaker-embedding",
        "audio-events",
    }
    assert catalog["kokoro-en"]["language"] == "en"
    assert "Persian is not supported" in catalog["kokoro-en"]["license_note"]
    assert "Piper software is GPLv3" in catalog["piper-fa"]["license_note"]
    assert catalog["audio-events"]["kind"] == "sound"
    assert catalog["speaker-embedding"]["kind"] == "speaker"


def test_kokoro_download_pins_assets_verifies_and_reuses_complete_install(tmp_path):
    client, requests = fixture_client("kokoro-en")
    with client:
        record = download_model("kokoro-en", tmp_path, client=client)
        reused = download_model("kokoro-en", tmp_path, client=client)
    assert reused["path"] == record["path"]
    assert len(requests) == 6  # first resolve+download both; second resolve only
    assert Path(record["path"]).name == "kokoro-v1.0.onnx"
    assert Path(record["path"]).with_name("voices-v1.0.bin").is_file()
    assert record["source_type"] == "github-release"
    assert {row["id"] for row in record["source_assets"]} == {520133700, 520134664}
    assert (
        installed_models(tmp_path, verify=True)["kokoro-en"]["path"] == record["path"]
    )
    source = json.loads(Path(record["path"]).with_name("SOURCE.json").read_text())
    assert "Apache-2.0" in source["license_note"]


def test_archive_installs_only_selected_models_and_labels(tmp_path):
    client, _ = fixture_client("audio-events")
    with client:
        record = download_model("audio-events", tmp_path, client=client)
    path = Path(record["path"])
    assert path.name == "model.int8.onnx"
    assert path.read_bytes() == b"int8 model"
    assert path.with_name("class_labels_indices.csv").is_file()
    assert {p.name for p in path.parent.iterdir()} == {
        "model.int8.onnx",
        "class_labels_indices.csv",
        "SOURCE.json",
    }
    assert installed_models(tmp_path, verify=True)["audio-events"]["path"] == str(path)


@pytest.mark.parametrize(
    "bad_member",
    [
        ("../../outside", b"malicious", tarfile.REGTYPE),
        ("link", b"", tarfile.SYMTYPE),
    ],
)
def test_archive_rejects_traversal_and_links_without_publishing(tmp_path, bad_member):
    spec = next(row for row in model_catalog() if row["key"] == "audio-events")
    members = [(name, b"valid bytes", tarfile.REGTYPE) for name in spec["files"]]
    members.append(bad_member)
    client, _ = fixture_client("audio-events", archive_members=members)
    with client, pytest.raises(VoiceError, match="unsafe"):
        download_model("audio-events", tmp_path, client=client)
    assert installed_models(tmp_path) == {}
    assert not list((tmp_path / "models" / "audio-events").iterdir())
    assert not (tmp_path.parent / "outside").exists()


def test_archive_extraction_is_bounded_before_allocating_large_output(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("src.voice.models._MAX_ARCHIVE_OUTPUT", 4)
    client, _ = fixture_client("audio-events")
    with client, pytest.raises(VoiceError, match="size limit"):
        download_model("audio-events", tmp_path, client=client)
    assert installed_models(tmp_path) == {}


def test_replaced_release_asset_is_rejected_before_download(tmp_path):
    client, requests = fixture_client("speaker-embedding", wrong_identity=True)
    with client, pytest.raises(VoiceError, match="identity"):
        download_model("speaker-embedding", tmp_path, client=client)
    assert len(requests) == 1
    assert installed_models(tmp_path) == {}


def test_cancelled_asset_download_leaves_previous_installation_untouched(tmp_path):
    client, _ = fixture_client("speaker-embedding")
    with client:
        original = download_model("speaker-embedding", tmp_path, client=client)
    manifest = tmp_path / "installed" / "speaker-embedding.json"
    original_manifest = manifest.read_bytes()
    # Corrupt the old file to force a replacement download. Cancel while reading
    # the new asset; even the old (corrupt) installation is not partially rewritten.
    Path(original["path"]).write_bytes(b"changed")
    cancel = Event()
    client, _ = fixture_client("speaker-embedding")
    with client, pytest.raises(VoiceCancelled):
        download_model(
            "speaker-embedding",
            tmp_path,
            client=client,
            stop_event=cancel,
            progress=lambda row: (
                cancel.set() if row["stage"] == "downloading" else None
            ),
        )
    assert manifest.read_bytes() == original_manifest
    assert Path(original["path"]).read_bytes() == b"changed"
    assert len(list((tmp_path / "models" / "speaker-embedding").iterdir())) == 1


def test_bad_checksum_never_publishes_download(tmp_path):
    client, _ = fixture_client("kokoro-en", corrupt=True)
    with client, pytest.raises(VoiceError, match="checksum"):
        download_model("kokoro-en", tmp_path, client=client)
    assert installed_models(tmp_path) == {}
    assert not list((tmp_path / "models" / "kokoro-en").iterdir())


def test_incomplete_manifest_is_not_treated_as_an_installed_model(tmp_path):
    client, _ = fixture_client("kokoro-en")
    with client:
        download_model("kokoro-en", tmp_path, client=client)
    manifest = tmp_path / "installed" / "kokoro-en.json"
    record = json.loads(manifest.read_text())
    record["files"] = record["files"][:1]
    manifest.write_text(json.dumps(record))
    assert installed_models(tmp_path) == {}
