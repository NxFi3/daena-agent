"""Voice inference lifecycle without downloading speech weights."""

from queue import Empty
from threading import Event
import struct
import time

from src.voice.providers import AudioChunk
from src.voice.session import VoiceSession
from src.voice.streaming import SpeechSegmenter, Utterance, speech_segments


def wait_until(predicate, seconds=2):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def drain(queue):
    values = []
    while True:
        try:
            values.append(queue.get_nowait())
        except Empty:
            return values


def test_endpointing_preroll_silence_and_short_noise():
    frame = struct.pack("<320h", *([1000] * 320))
    quiet = b"\0" * 640
    segmenter = SpeechSegmenter(silence_ms=200, vad=lambda data: data != quiet)
    assert not segmenter.feed(quiet * 20 + frame * 2 + quiet * 10)
    result = segmenter.feed(quiet * 20 + frame * 12 + quiet * 10)
    assert len(result) == 1 and not result[0].truncated
    assert result[0].pcm16.endswith(quiet * 10)
    assert frame * 12 in result[0].pcm16


def test_long_utterance_is_bounded_and_marked_for_review():
    frame = b"\1\0" * 320
    segmenter = SpeechSegmenter(max_seconds=1, vad=lambda data: True)
    utterances = segmenter.feed(frame * 300)
    assert len(utterances) == 1
    assert all(u.truncated and len(u.pcm16) == 640 * 50 for u in utterances)


def test_speech_cleanup_preserves_bilingual_prose_but_skips_code():
    text = "## سلام\n\n[Hello](https://example.org) **دوست من**.\n```python\nsecret_code()\n```\nپایان."
    output = " ".join(speech_segments(text, max_chars=40))
    assert "سلام" in output and "Hello" in output and "دوست من" in output
    assert "secret_code" not in output and "https://" not in output


def test_cancel_drops_late_native_recognition_without_blocking_close():
    entered, release = Event(), Event()

    class STT:
        def transcribe(self, *args, **kwargs):
            entered.set()
            release.wait(2)
            return "do not submit late"

    class TTS:
        def synthesize(self, *args, **kwargs):
            return iter(())

    session = VoiceSession(STT(), TTS())
    try:
        session.submit_audio(Utterance(b"\0\0" * 1600))
        assert entered.wait(1)
        started = time.monotonic()
        session.close()
        assert time.monotonic() - started < 0.1
        release.set()
        wait_until(lambda: session.retired)
        assert not any(event.kind == "transcript" for event in drain(session.events))
    finally:
        release.set()
        session.close()


def test_audio_backpressure_and_interrupt_epoch():
    class STT:
        def transcribe(self, *args, **kwargs):
            return ""

    class TTS:
        def synthesize(self, *args, **kwargs):
            yield AudioChunk(16000, b"\0\0" * 16000 * 5)

    session = VoiceSession(STT(), TTS())
    try:
        session.speak("A sentence to synthesize.")
        wait_until(lambda: session.audio.full())
        assert session.audio.qsize() == 6
        before = session.epoch
        session.interrupt()
        assert session.epoch == before + 1
        time.sleep(0.08)
        assert all(epoch != session.epoch for epoch, _ in drain(session.audio))
        session.speak("Another sentence.")
        wait_until(lambda: not session.audio.empty())
        epoch, chunk = session.audio.get_nowait()
        assert epoch == session.epoch and len(chunk.pcm16) <= 3200
    finally:
        session.close()
        wait_until(lambda: session.retired)


def test_explicit_silence_releases_capped_utterance_quarantine():
    frame = b"\1\0" * 320
    quiet = b"\0" * 640
    segmenter = SpeechSegmenter(
        max_seconds=1, silence_ms=200, vad=lambda data: data != quiet
    )
    first = segmenter.feed(frame * 80)
    assert len(first) == 1 and first[0].truncated
    second = segmenter.feed(quiet * 10 + frame * 12 + quiet * 10)
    assert len(second) == 1 and not second[0].truncated


def test_voice_cli_writes_real_pcm_wav_without_recording(tmp_path, monkeypatch):
    import wave
    from src.voice.__main__ import main

    class TTS:
        def __init__(self, *args, **kwargs):
            pass

        def synthesize(self, *args, **kwargs):
            yield AudioChunk(22050, b"\0\0" * 2205)

    monkeypatch.setattr("src.voice.providers.PiperProvider", TTS)
    output = tmp_path / "audition.wav"
    assert (
        main(
            [
                "synthesize",
                "--text",
                "سلام",
                "--language",
                "fa",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    with wave.open(str(output), "rb") as wav:
        assert wav.getframerate() == 22050 and wav.getnchannels() == 1
        assert wav.getsampwidth() == 2 and wav.getnframes() == 2205
