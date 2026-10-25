"""Explicit, cancellable downloads of public local speech models.

Inference providers never call this module automatically. The Hub's selected
revision is resolved to an immutable commit before files are streamed. A complete
installation is published only after required files and local hashes are ready.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import quote

from .providers import VoiceCancelled, VoiceError, _check_stop


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    kind: str
    language: str
    repo_id: str
    revision: str
    files: tuple[str, ...]
    path_file: str = ""

    @property
    def source_url(self):
        return f"https://huggingface.co/{self.repo_id}/tree/{self.revision}"


_SPECS = (
    ModelSpec(
        "whisper-turbo",
        "Whisper large-v3-turbo · quality",
        "stt",
        "fa,en",
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "main",
        ("config.json", "model.bin", "tokenizer.json"),
    ),
    ModelSpec(
        "whisper-small",
        "Whisper small · lower memory",
        "stt",
        "fa,en",
        "Systran/faster-whisper-small",
        "main",
        ("config.json", "model.bin", "tokenizer.json"),
    ),
    ModelSpec(
        "piper-fa",
        "Persian · Amir medium",
        "tts",
        "fa",
        "rhasspy/piper-voices",
        "v1.0.0",
        (
            "fa/fa_IR/amir/medium/fa_IR-amir-medium.onnx",
            "fa/fa_IR/amir/medium/fa_IR-amir-medium.onnx.json",
            "fa/fa_IR/amir/medium/MODEL_CARD",
        ),
        "fa_IR-amir-medium.onnx",
    ),
    ModelSpec(
        "piper-en",
        "English · Lessac high",
        "tts",
        "en",
        "rhasspy/piper-voices",
        "v1.0.0",
        (
            "en/en_US/lessac/high/en_US-lessac-high.onnx",
            "en/en_US/lessac/high/en_US-lessac-high.onnx.json",
            "en/en_US/lessac/high/MODEL_CARD",
        ),
        "en_US-lessac-high.onnx",
    ),
)
_SPEC_BY_KEY = {spec.key: spec for spec in _SPECS}


def model_catalog():
    return [
        asdict(spec)
        | {
            "source_url": spec.source_url,
            "license_note": "Review the downloaded model card and license documents; Piper software is GPLv3 and individual voice/data licenses differ.",
        }
        for spec in _SPECS
    ]


def default_cache_dir():
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "daena" / "voice"


def _root(cache_dir=None):
    return Path(cache_dir or default_cache_dir()).expanduser().resolve()


def _inside(root, path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise VoiceError("Voice cache path escapes its selected directory")
    return resolved


def _sha256(path, stop_event=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(1024 * 1024):
            _check_stop(stop_event)
            digest.update(data)
    return digest.hexdigest()


def installed_models(cache_dir=None, *, verify=False):
    root = _root(cache_dir)
    result = {}
    for spec in _SPECS:
        manifest = root / "installed" / f"{spec.key}.json"
        if not manifest.is_file():
            continue
        try:
            record = json.loads(_inside(root, manifest).read_text(encoding="utf-8"))
            if record.get("key") != spec.key or not re.fullmatch(
                r"[0-9a-f]{40,64}", record.get("commit", "")
            ):
                continue
            path = _inside(root, root / record["relative_path"])
            if not path.exists():
                continue
            for row in record["files"]:
                file = _inside(root, root / row["relative_path"])
                if not file.is_file() or file.stat().st_size != row["size"]:
                    raise ValueError("missing or changed model file")
                if verify and _sha256(file) != row["sha256"]:
                    raise ValueError("model file hash mismatch")
            result[spec.key] = {**record, "path": str(path)}
        except (OSError, ValueError, TypeError, KeyError, VoiceError):
            continue
    return result


def voice_paths(cache_dir=None):
    installed = installed_models(cache_dir)
    return {
        language: installed[key]["path"]
        for language, key in (("fa", "piper-fa"), ("en", "piper-en"))
        if key in installed
    }


def _publish_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".voice-manifest-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _files_for(spec, siblings):
    by_name = {
        row["rfilename"]: row
        for row in siblings
        if isinstance(row, dict) and isinstance(row.get("rfilename"), str)
    }
    missing = set(spec.files) - set(by_name)
    if missing:
        raise VoiceError(
            "The model repository is missing required files: "
            + ", ".join(sorted(missing))
        )
    files = list(spec.files)
    if spec.kind == "stt":
        files.extend(
            name
            for name in (
                "preprocessor_config.json",
                "vocabulary.json",
                "vocabulary.txt",
            )
            if name in by_name
        )
    files.extend(
        name
        for name in (
            "README.md",
            "LICENSE",
            "LICENSE.txt",
            "LICENSE.md",
            "LICENSE_MODEL",
        )
        if name in by_name
    )
    return list(dict.fromkeys(files)), by_name


def download_model(
    model_key, cache_dir=None, *, stop_event=None, progress=None, client=None
):
    """Download only after an explicit user action; returns an installed manifest.

    ``progress`` receives dictionaries containing stage/file/received/total.
    Cancellation checks occur between streamed blocks and checksum blocks. A
    stalled network read is bounded by a 15-second read timeout.
    """
    _check_stop(stop_event)
    spec = _SPEC_BY_KEY.get(str(model_key))
    if spec is None:
        raise VoiceError("Choose a model key from the local voice catalog")
    root = _root(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    if progress:
        progress({"stage": "resolving", "model": spec.key})
    own_client = client is None
    if own_client:
        import httpx

        client = httpx.Client(
            follow_redirects=True, timeout=httpx.Timeout(15.0, connect=10.0)
        )
    try:
        url = f"https://huggingface.co/api/models/{spec.repo_id}/revision/{quote(spec.revision, safe='')}?blobs=true"
        response = client.get(url)
        response.raise_for_status()
        metadata = response.json()
        _check_stop(stop_event)
        commit = metadata.get("sha", "")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
            raise VoiceError("Model source did not return an immutable revision")
        files, source = _files_for(spec, metadata.get("siblings", []))
        destination = _inside(root, root / "models" / spec.key / commit)
        destination.mkdir(parents=True, exist_ok=True)
        rows = []
        for remote in files:
            _check_stop(stop_event)
            # Catalog paths are trusted constants; metadata never chooses disk paths.
            path = _inside(root, destination / Path(remote).name)
            info = source[remote]
            lfs = info.get("lfs") or {}
            expected = lfs.get("sha256") if isinstance(lfs, dict) else None
            expected_size = info.get("size")
            if expected_size is not None and (
                not isinstance(expected_size, int)
                or expected_size < 0
                or expected_size > 4 * 1024**3
            ):
                raise VoiceError("Model source reported an unsupported file size")
            cached = (
                path.is_file() and expected and _sha256(path, stop_event) == expected
            )
            if not cached:
                fd, temporary = tempfile.mkstemp(
                    prefix=".voice-download-", dir=destination
                )
                received = 0
                digest = hashlib.sha256()
                try:
                    download_url = f"https://huggingface.co/{spec.repo_id}/resolve/{commit}/{quote(remote, safe='/')}"
                    with os.fdopen(fd, "wb") as output:
                        with client.stream("GET", download_url) as data:
                            data.raise_for_status()
                            for block in data.iter_bytes(chunk_size=64 * 1024):
                                _check_stop(stop_event)
                                received += len(block)
                                if received > 4 * 1024**3 or (
                                    expected_size is not None
                                    and received > expected_size
                                ):
                                    raise VoiceError(
                                        "Model download exceeded its declared size"
                                    )
                                output.write(block)
                                digest.update(block)
                                if progress:
                                    progress(
                                        {
                                            "stage": "downloading",
                                            "model": spec.key,
                                            "file": remote,
                                            "received": received,
                                            "total": expected_size,
                                        }
                                    )
                        output.flush()
                        os.fsync(output.fileno())
                    _check_stop(stop_event)
                    checksum = digest.hexdigest()
                    if expected and checksum != expected:
                        raise VoiceError(
                            "Model download checksum mismatch; the previous installation was preserved"
                        )
                    if expected_size is not None and received != expected_size:
                        raise VoiceError(
                            "Model download was incomplete; the previous installation was preserved"
                        )
                    # JSON files must parse before being published.
                    if remote.endswith(".json"):
                        json.loads(Path(temporary).read_text(encoding="utf-8"))
                    os.replace(temporary, path)
                finally:
                    Path(temporary).unlink(missing_ok=True)
            checksum = _sha256(path, stop_event)
            rows.append(
                {
                    "source_path": remote,
                    "relative_path": str(path.relative_to(root)),
                    "sha256": checksum,
                    "size": path.stat().st_size,
                    "source_sha256": expected,
                }
            )
        _check_stop(stop_event)
        model_path = destination / spec.path_file if spec.path_file else destination
        record = {
            "schema_version": 1,
            "key": spec.key,
            "repo_id": spec.repo_id,
            "requested_revision": spec.revision,
            "commit": commit,
            "relative_path": str(model_path.relative_to(root)),
            "files": rows,
            "source_url": f"https://huggingface.co/{spec.repo_id}/tree/{commit}",
            "license_note": "Consult the downloaded model card/license before distribution or commercial use.",
        }
        manifest = _inside(root, root / "installed" / f"{spec.key}.json")
        _publish_json(manifest, record)
        if progress:
            progress({"stage": "ready", "model": spec.key})
        return {**record, "path": str(model_path)}
    except VoiceCancelled:
        raise
    except VoiceError:
        raise
    except Exception as exc:
        raise VoiceError(
            "Local voice model download failed. Check network access to Hugging Face, free disk space and the source URL. "
            + str(exc)
        ) from exc
    finally:
        if own_client:
            client.close()
