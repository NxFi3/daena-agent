"""Local speech provider contracts without downloaded weights or hardware."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from src.voice.models import (
    download_model,
    installed_models,
    model_catalog,
    voice_paths,
)
from src.voice.providers import (
    AudioChunk,
    FasterWhisperProvider,
    LanguageRouter,
    PiperProvider,
    VoiceCancelled,
    VoiceError,
    VoiceLanguageUnavailable,
    language_runs,
)


def test_whisper_is_lazy_cached_offline_and_uses_bounded_cpu():
    calls = []
    transcriptions = []

    class Model:
        def __init__(self, path, **kwargs):
            calls.append((path, kwargs))

        def transcribe(self, audio, **kwargs):
            transcriptions.append((audio, kwargs))
            return iter(
                [SimpleNamespace(text=" سلام "), SimpleNamespace(text="hello")]
            ), None

    provider = FasterWhisperProvider("local/model", model_factory=Model)
    assert calls == []
    pcm = np.array([-32768, 0, 32767], dtype="<i2").tobytes()
    assert provider.transcribe(pcm) == "سلام hello"
    assert provider.transcribe(pcm) == "سلام hello"
    assert len(calls) == 1
    assert calls[0][1] == {
        "device": "cpu",
        "compute_type": "int8",
        "cpu_threads": 4,
        "num_workers": 1,
        "local_files_only": True,
        "download_root": None,
    }
    assert transcriptions[0][0].dtype == np.float32
    assert transcriptions[0][0][0] == -1.0
    assert transcriptions[0][1]["language"] is None
    assert transcriptions[0][1]["task"] == "transcribe"
    assert transcriptions[0][1]["condition_on_previous_text"] is False
    assert transcriptions[0][1]["beam_size"] == 3


def test_whisper_cancellation_discards_result_and_never_loads_when_already_stopped():
    stop = Event()
    stop.set()
    provider = FasterWhisperProvider(
        model_factory=lambda *a, **k: pytest.fail("loaded")
    )
    with pytest.raises(VoiceCancelled):
        provider.transcribe(b"\0\0", stop_event=stop)
    stop.clear()

    class Model:
        def transcribe(self, *args, **kwargs):
            def segments():
                stop.set()
                yield SimpleNamespace(text="must not be delivered")

            return segments(), None

    provider = FasterWhisperProvider(model_factory=lambda *a, **k: Model())
    with pytest.raises(VoiceCancelled):
        provider.transcribe(b"\0\0", stop_event=stop)


@pytest.mark.parametrize(
    "rate,pcm", [(48000, b"\0\0"), (16000, b"\0"), (16000, "not bytes")]
)
def test_whisper_requires_explicit_pcm16_contract(rate, pcm):
    provider = FasterWhisperProvider(
        model_factory=lambda *a, **k: pytest.fail("loaded")
    )
    with pytest.raises(VoiceError):
        provider.transcribe(pcm, sample_rate=rate)


def test_whisper_rejects_implicit_downloads():
    with pytest.raises(VoiceError, match="explicitly"):
        FasterWhisperProvider(local_files_only=False)


def voices(tmp_path):
    result = {}
    for language in ("fa", "en"):
        path = tmp_path / f"{language}.onnx"
        path.write_bytes(b"fake model")
        Path(str(path) + ".json").write_text("{}")
        result[language] = path
    return result


def test_language_router_uses_one_contract_for_piper_and_kokoro_style_providers():
    events = []

    class FakeProvider:
        def __init__(self, name):
            self.name = name

        def prepare(self, stop_event=None):
            events.append(("prepare", self.name))

        def synthesize(self, text, language="auto", stop_event=None):
            events.append(("speak", self.name, language, text))
            yield AudioChunk(16000, b"\0\0")

    persian = FakeProvider("piper-fa")
    english = FakeProvider("kokoro-en")
    router = LanguageRouter(
        {"fa": persian, "en": english},
        language="auto",
        required_languages=("fa", "en"),
    )

    router.prepare()
    chunks = list(router.synthesize("سلام Daena، خوبی؟"))
    assert len(chunks) == 3
    assert events[:2] == [("prepare", "piper-fa"), ("prepare", "kokoro-en")]
    assert events[2:] == [
        ("speak", "piper-fa", "fa", "سلام"),
        ("speak", "kokoro-en", "en", "Daena،"),
        ("speak", "piper-fa", "fa", "خوبی؟"),
    ]


def test_language_router_fails_before_start_when_automatic_language_is_missing():
    events = []

    class EnglishOnly:
        def prepare(self, stop_event=None):
            events.append("prepared")

        def synthesize(self, text, language="auto", stop_event=None):
            events.append(("spoken", text, language))
            yield AudioChunk(16000, b"\0\0")

    router = LanguageRouter(
        {"fa": None, "en": EnglishOnly()},
        language="auto",
        required_languages=("fa", "en"),
    )
    assert router.available_languages == ("en",)
    with pytest.raises(VoiceLanguageUnavailable, match="Persian output is not configured"):
        router.prepare()
    with pytest.raises(VoiceLanguageUnavailable, match="does not include Persian"):
        list(router.synthesize("سلام"))
    assert events == []


def test_piper_streams_current_chunk_api_and_caches_both_language_voices(tmp_path):
    loaded = []
    spoken = []

    class Voice:
        def __init__(self, path):
            self.path = path

        def synthesize(self, text):
            spoken.append((Path(self.path).name, text))
            yield SimpleNamespace(
                sample_rate=22050,
                sample_width=2,
                sample_channels=1,
                audio_int16_bytes=b"\x01\x00" * 10,
            )

    def load(path, **kwargs):
        loaded.append((path, kwargs))
        return Voice(path)

    provider = PiperProvider(voices(tmp_path), voice_loader=load)
    assert loaded == []
    chunks = list(provider.synthesize("سلام hello دنیا"))
    assert len(chunks) == 3
    assert all(isinstance(chunk, AudioChunk) for chunk in chunks)
    assert [row[0] for row in spoken] == ["fa.onnx", "en.onnx", "fa.onnx"]
    assert len(loaded) == 2
    assert all(row[1]["use_cuda"] is False for row in loaded)
    assert all(row[1]["config_path"] == row[0] + ".json" for row in loaded)
    assert len(list(provider.synthesize("English again."))) == 1
    assert len(loaded) == 2


def test_piper_missing_voice_is_actionable_and_never_loads(tmp_path):
    provider = PiperProvider(
        {"fa": tmp_path / "missing.onnx"},
        voice_loader=lambda *a, **k: pytest.fail("loaded"),
    )
    with pytest.raises(VoiceError, match="download"):
        list(provider.synthesize("سلام"))


def test_piper_checks_cancellation_between_native_chunks(tmp_path):
    stop = Event()

    class Voice:
        def synthesize(self, text):
            yield SimpleNamespace(
                sample_rate=22050,
                sample_width=2,
                sample_channels=1,
                audio_int16_bytes=b"\0\0",
            )
            stop.set()
            yield SimpleNamespace(
                sample_rate=22050,
                sample_width=2,
                sample_channels=1,
                audio_int16_bytes=b"\0\0",
            )

    provider = PiperProvider(voices(tmp_path), voice_loader=lambda *a, **k: Voice())
    stream = provider.synthesize("hello", stop_event=stop)
    assert next(stream).pcm16 == b"\0\0"
    with pytest.raises(VoiceCancelled):
        next(stream)


def test_language_runs_preserve_order_and_neutral_punctuation():
    assert list(language_runs("سلام Python 3 خوب است.")) == [
        ("fa", "سلام"),
        ("en", "Python 3"),
        ("fa", "خوب است."),
    ]
    assert list(language_runs("123!", fallback="fa")) == [("fa", "123!")]


def model_fixture(*, commit="a" * 40, corrupt=False):
    catalog = next(row for row in model_catalog() if row["key"] == "piper-fa")
    files = {
        catalog["files"][0]: b"fake onnx bytes",
        catalog["files"][1]: json.dumps({"audio": {"sample_rate": 22050}}).encode(),
        catalog["files"][2]: b"License: see model source",
        "LICENSE": b"Example license text",
    }
    metadata = {
        "sha": commit,
        "siblings": [
            {
                "rfilename": name,
                "size": len(body),
                "lfs": {"sha256": hashlib.sha256(body).hexdigest()},
            }
            for name, body in files.items()
        ],
    }
    requests = []

    def handle(request):
        requests.append(str(request.url))
        if "/api/models/" in request.url.path:
            return httpx.Response(200, json=metadata)
        assert f"/resolve/{commit}/" in request.url.path
        path = request.url.path.split(f"/resolve/{commit}/", 1)[1]
        return httpx.Response(
            200,
            content=b"x" * len(files[path])
            if corrupt and path.endswith(".onnx")
            else files[path],
        )

    return httpx.Client(
        transport=httpx.MockTransport(handle), follow_redirects=True
    ), requests


def test_explicit_download_pins_commit_checks_hashes_and_publishes_manifest(tmp_path):
    client, requests = model_fixture()
    progress = []
    with client:
        record = download_model(
            "piper-fa", tmp_path, client=client, progress=progress.append
        )
    assert record["commit"] == "a" * 40
    assert Path(record["path"]).is_file()
    assert len(requests) == 5  # metadata, weights, config, card, license
    assert progress[-1]["stage"] == "ready"
    installed = installed_models(tmp_path, verify=True)
    assert installed["piper-fa"]["path"] == record["path"]
    assert voice_paths(tmp_path)["fa"] == record["path"]
    assert any(row["source_path"] == "LICENSE" for row in record["files"])
    assert not list(tmp_path.rglob(".voice-download-*"))
    Path(record["path"]).write_bytes(b"corrupt")
    assert installed_models(tmp_path, verify=True) == {}


def test_failed_new_download_preserves_previous_installation(tmp_path):
    client, _ = model_fixture()
    with client:
        first = download_model("piper-fa", tmp_path, client=client)
    manifest = tmp_path / "installed" / "piper-fa.json"
    before = manifest.read_bytes()
    client, _ = model_fixture(commit="b" * 40, corrupt=True)
    with client, pytest.raises(VoiceError, match="checksum"):
        download_model("piper-fa", tmp_path, client=client)
    assert manifest.read_bytes() == before
    assert installed_models(tmp_path)["piper-fa"]["path"] == first["path"]
    assert not list(tmp_path.rglob(".voice-download-*"))


def test_cancelled_download_never_publishes_partial_install(tmp_path):
    client, _ = model_fixture()
    stop = Event()

    def progress(event):
        if event["stage"] == "downloading":
            stop.set()

    with client, pytest.raises(VoiceCancelled):
        download_model(
            "piper-fa", tmp_path, client=client, progress=progress, stop_event=stop
        )
    assert installed_models(tmp_path) == {}
    assert not list(tmp_path.rglob(".voice-download-*"))


def test_model_cache_rejects_symlink_escape(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "models").symlink_to(outside, target_is_directory=True)
    client, _ = model_fixture()
    with client, pytest.raises(VoiceError, match="escapes"):
        download_model("piper-fa", root, client=client)
    assert list(outside.iterdir()) == []


def test_prepare_caches_models_and_rejects_cancelled_native_load(tmp_path):
    loads = []
    provider = FasterWhisperProvider(
        model_factory=lambda *a, **k: loads.append(1) or object()
    )
    provider.prepare()
    provider.prepare()
    assert loads == [1]
    loaded = []
    piper = PiperProvider(
        voices(tmp_path), voice_loader=lambda path, **k: loaded.append(path) or object()
    )
    piper.prepare()
    piper.prepare()
    assert len(loaded) == 2
    stop = Event()
    provider = FasterWhisperProvider(
        model_factory=lambda *a, **k: stop.set() or object()
    )
    with pytest.raises(VoiceCancelled):
        provider.prepare(stop)
