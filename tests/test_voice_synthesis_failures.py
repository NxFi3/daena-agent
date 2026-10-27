"""Speech failures must be visible instead of reporting successful silence."""

from queue import Empty
from threading import Event
import time
from types import SimpleNamespace

import pytest

from src.voice.kokoro import BilingualSynthesizer, KokoroProvider
from src.voice.providers import AudioChunk, VoiceError
from src.voice.session import VoiceSession


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


def drain(queue):
    rows = []
    while True:
        try:
            rows.append(queue.get_nowait())
        except Empty:
            return rows


def close(session):
    session.close()
    wait_for(lambda: session.retired)


@pytest.mark.parametrize("empty_chunks", [False, True])
def test_nonempty_prose_with_no_pcm_reports_error_not_done_or_latency(empty_chunks):
    class Silent:
        def synthesize(self, *args, **kwargs):
            if empty_chunks:
                yield AudioChunk(24000, b"")

    session = VoiceSession(None, Silent())
    try:
        session.speak("Hello, this should be spoken.")
        wait_for(lambda: any(e.kind == "error" for e in list(session.events.queue)))
        events = drain(session.events)
        assert not any(e.kind in {"tts_done", "latency"} for e in events)
        assert "produced no audio" in next(e.data for e in events if e.kind == "error")
        assert session.audio.empty()
    finally:
        close(session)


def test_empty_native_frames_are_skipped_before_first_audio_receipt():
    class Voice:
        def synthesize(self, *args, **kwargs):
            yield AudioChunk(24000, b"")
            yield AudioChunk(24000, b"\1\0" * 100)
            yield AudioChunk(24000, b"")

    session = VoiceSession(None, Voice())
    try:
        session.speak("A short answer.")
        wait_for(lambda: any(e.kind == "tts_done" for e in list(session.events.queue)))
        chunks = drain(session.audio)
        assert len(chunks) == 2 and chunks[-1][1] is None
        assert chunks[0][1].pcm16 == b"\1\0" * 100
        events = drain(session.events)
        assert len([e for e in events if e.kind == "latency"]) == 1
        assert not any(e.kind == "error" for e in events)
    finally:
        close(session)


def test_intentionally_omitted_code_has_no_synthesis_error_or_phantom_latency():
    class Unused:
        def synthesize(self, *args, **kwargs):
            pytest.fail("Code-only content must not reach the speech engine")

    session = VoiceSession(None, Unused())
    try:
        session.speak("```python\nprint('not prose')\n```")
        wait_for(lambda: any(e.kind == "tts_done" for e in list(session.events.queue)))
        events = drain(session.events)
        assert not any(e.kind in {"error", "latency"} for e in events)
        assert all(chunk is None for _, chunk in drain(session.audio))
    finally:
        close(session)


def test_cancellation_during_empty_output_does_not_report_a_new_failure():
    entered, release = Event(), Event()

    class Voice:
        def synthesize(self, *args, **kwargs):
            yield AudioChunk(24000, b"")
            entered.set()
            release.wait(2)

    session = VoiceSession(None, Voice())
    try:
        session.speak("An interrupted passage.")
        assert entered.wait(1)
        session.interrupt()
        release.set()
        time.sleep(0.03)
        assert not any(
            e.kind in {"error", "tts_done", "latency"} for e in drain(session.events)
        )
    finally:
        release.set()
        close(session)


def test_invalid_adapter_audio_cannot_be_counted_as_successful_playback():
    class Invalid:
        def synthesize(self, *args, **kwargs):
            yield SimpleNamespace(sample_rate=0, pcm16=b"\1\0")

    session = VoiceSession(None, Invalid())
    try:
        session.speak("Invalid adapter output.")
        wait_for(lambda: any(e.kind == "error" for e in list(session.events.queue)))
        assert session.audio.empty()
        events = drain(session.events)
        assert not any(e.kind in {"latency", "tts_done"} for e in events)
        assert "unsupported sample rate" in next(
            e.data for e in events if e.kind == "error"
        )
    finally:
        close(session)


@pytest.mark.parametrize("language", ["auto", "en"])
@pytest.mark.parametrize("text", ["سلام", "Hello سلام", "helloسلام"])
def test_explicit_english_does_not_bypass_kokoro_language_capability(language, text):
    voice = KokoroProvider(
        "unused.onnx",
        "unused.bin",
        engine_factory=lambda *a, **k: pytest.fail(
            "Unsupported script loaded the English engine"
        ),
    )
    with pytest.raises(VoiceError, match="Persian text requires"):
        list(voice.synthesize(text, language))


def test_valid_bilingual_router_keeps_persian_and_english_on_separate_engines():
    spoken = []

    class Voice:
        def __init__(self, language):
            self.language = language

        def synthesize(self, text, language, stop_event=None):
            spoken.append((self.language, language, text))
            yield AudioChunk(24000, b"\1\0")

    router = BilingualSynthesizer(english=Voice("en"), persian=Voice("fa"))
    assert len(list(router.synthesize("سلام Hello دوست من"))) == 3
    assert spoken == [
        ("fa", "fa", "سلام"),
        ("en", "en", "Hello"),
        ("fa", "fa", "دوست من"),
    ]
