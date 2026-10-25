"""Optional acoustic evidence stays independent from critical STT/playback."""

from queue import Empty
from threading import Event
from types import SimpleNamespace
import time

from src.voice.session import VoiceSession
from src.voice.streaming import Utterance


def wait_until(predicate, seconds=2):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


def drain(queue):
    output = []
    while True:
        try:
            output.append(queue.get_nowait())
        except Empty:
            return output


class Recognizer:
    def transcribe(self, *args, **kwargs):
        return "recognized speech"


class Synthesizer:
    def synthesize(self, *args, **kwargs):
        return iter(())


class Perception:
    def __init__(self):
        self.calls = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def analyze(self, pcm16, sample_rate, *, stop_event, speech):
        self.calls.append((pcm16, sample_rate, speech))
        return {
            "speaker": {"id": "speaker_1" if speech else None},
            "sounds": {"status": "unknown", "events": []},
        }


def close(session):
    session.close()
    wait_until(lambda: session.retired)


def test_non_speech_raw_audio_reaches_optional_classifier_without_transcription():
    perception = Perception()
    session = VoiceSession(
        Recognizer(), Synthesizer(), perception=perception, sound_events_enabled=True
    )
    try:
        wait_until(lambda: session.recognition_ready.is_set())
        session.segmenter._vad = lambda frame: False
        pcm = b"\x10\0" * 32000
        session.feed(pcm[:16000])
        session.feed(pcm[16000:])
        wait_until(lambda: len(perception.calls) == 1)
        events = drain(session.events)
        acoustic = [event for event in events if event.kind == "perception"]
        assert len(acoustic) == 1
        assert acoustic[0].data["source"] == "ambient"
        assert acoustic[0].data["utterance_id"] is None
        assert acoustic[0].data["voice_session_id"] == session.acoustic_session_id
        assert len(session.acoustic_session_id) == 32
        assert acoustic[0].data["sounds"]["status"] == "unknown"
        assert perception.calls == [(pcm, 16000, False)]
        assert not any(event.kind == "transcript" for event in events)
        assert not session._ambient_buffer
    finally:
        close(session)


def test_speech_evidence_and_transcript_correlate_without_waiting_for_analysis():
    entered, release = Event(), Event()

    class Slow(Perception):
        def analyze(self, *args, **kwargs):
            entered.set()
            release.wait(2)
            return super().analyze(*args, **kwargs)

    perception = Slow()
    session = VoiceSession(Recognizer(), Synthesizer(), perception=perception)
    try:
        pcm = b"\1\0" * 24000
        session.submit_audio(Utterance(pcm))
        assert entered.wait(1)
        wait_until(
            lambda: any(e.kind == "transcript" for e in list(session.events.queue))
        )
        first = drain(session.events)
        transcript = next(e.data for e in first if e.kind == "transcript")
        assert transcript["text"] == "recognized speech"
        assert not any(e.kind == "perception" for e in first)
        release.set()
        wait_until(
            lambda: any(e.kind == "perception" for e in list(session.events.queue))
        )
        evidence = next(e.data for e in drain(session.events) if e.kind == "perception")
        assert evidence["utterance_id"] == transcript["utterance_id"]
        assert evidence["source"] == "speech"
        assert perception.calls == [(pcm, 16000, True)]
        assert perception.resets == 1
    finally:
        release.set()
        close(session)


def test_slow_acoustics_keep_only_latest_ambient_and_prioritize_speech():
    entered, release = Event(), Event()

    class Slow(Perception):
        def analyze(self, *args, **kwargs):
            if not self.calls:
                entered.set()
                release.wait(2)
            return super().analyze(*args, **kwargs)

    perception = Slow()
    session = VoiceSession(
        Recognizer(), Synthesizer(), perception=perception, sound_events_enabled=True
    )
    try:
        wait_until(lambda: session.recognition_ready.is_set())
        session.segmenter._vad = lambda frame: False
        first, latest = b"\1\0" * 32000, b"\3\0" * 32000
        session.feed(first)
        assert entered.wait(1)
        session.feed(b"\2\0" * 32000)
        session.feed(latest)
        assert session._ambient.qsize() == 1
        assert len(session._ambient_buffer) < 64000
        spoken = b"\4\0" * 24000
        session.submit_audio(Utterance(spoken))
        release.set()
        wait_until(lambda: len(perception.calls) == 3)
        assert perception.calls == [
            (first, 16000, False),
            (spoken, 16000, True),
            (latest, 16000, False),
        ]
    finally:
        release.set()
        close(session)


def test_interruption_discards_late_acoustics_and_resets_partial_input():
    entered, release = Event(), Event()

    class Slow(Perception):
        def analyze(self, *args, **kwargs):
            entered.set()
            release.wait(2)
            return super().analyze(*args, **kwargs)

    perception = Slow()
    session = VoiceSession(
        Recognizer(), Synthesizer(), perception=perception, sound_events_enabled=True
    )
    try:
        session.feed(b"\0\0" * 32000)
        assert entered.wait(1)
        session.feed(b"\0\0" * 100)
        session.interrupt()
        assert not session._ambient_buffer
        assert not session._ambient.qsize()
        release.set()
        wait_until(lambda: len(perception.calls) == 1)
        time.sleep(0.03)
        assert not any(e.kind == "perception" for e in drain(session.events))
        assert perception.resets == 1  # Cancel is not a new speaker session.
    finally:
        release.set()
        close(session)


def test_mute_or_playback_reset_invalidates_inflight_ambient_without_stopping_stt():
    entered, release = Event(), Event()

    class Slow(Perception):
        def analyze(self, *args, **kwargs):
            entered.set()
            release.wait(2)
            return super().analyze(*args, **kwargs)

    session = VoiceSession(
        Recognizer(), Synthesizer(), perception=Slow(), sound_events_enabled=True
    )
    try:
        session.feed(b"\0\0" * 32000)
        assert entered.wait(1)
        epoch = session.epoch
        session.reset_input()
        assert session.epoch == epoch
        session.submit_audio(Utterance(b"\0\0" * 24000))
        release.set()
        wait_until(lambda: len(session.perception.calls) == 2)
        time.sleep(0.02)
        events = drain(session.events)
        assert any(e.kind == "transcript" for e in events)
        assert [e.data["source"] for e in events if e.kind == "perception"] == [
            "speech"
        ]
    finally:
        release.set()
        close(session)


def test_optional_acoustic_load_failure_does_not_disable_speech_recognition():
    class Broken(Perception):
        def prepare(self, **kwargs):
            raise ValueError("missing optional acoustic model")

    session = VoiceSession(Recognizer(), Synthesizer(), perception=Broken())
    try:
        session.submit_audio(Utterance(b"\0\0" * 24000))
        wait_until(
            lambda: any(e.kind == "transcript" for e in list(session.events.queue))
        )
        events = drain(session.events)
        assert any(e.kind == "perception_error" for e in events)
        assert not any(e.kind == "error" for e in events)
        assert any(e.kind == "transcript" for e in events)
    finally:
        close(session)


def test_ambient_analysis_is_opt_in_and_new_session_resets_cached_speaker_ids():
    perception = Perception()
    sessions = set()
    for _ in range(2):
        session = VoiceSession(Recognizer(), Synthesizer(), perception=perception)
        sessions.add(session.acoustic_session_id)
        try:
            session.segmenter._vad = lambda frame: False
            session.feed(b"\0\0" * 64000)
            session.submit_audio(Utterance(b"\1\0" * 24000))
            wait_until(lambda: len(perception.calls) == perception.resets)
            assert not session._ambient_buffer
            assert len(session._threads) == 3
        finally:
            close(session)
    assert perception.resets == 2
    assert len(sessions) == 2
    assert all(speech for _, _, speech in perception.calls)


def test_microphone_during_slow_stt_warmup_does_not_overflow_command_queue():
    entered, release = Event(), Event()

    class Slow(Recognizer):
        def prepare(self, **kwargs):
            entered.set()
            release.wait(2)

    session = VoiceSession(Slow(), Synthesizer(), silence_ms=200)
    try:
        assert entered.wait(1)
        session.segmenter._vad = lambda frame: frame != b"\0" * 640
        frame = b"\1\0" * 320
        for _ in range(5):
            session.feed(frame * 12 + b"\0" * 640 * 10)
            session.finish_utterance()
        assert session._speech.qsize() == 0
        assert not session.recognition_ready.is_set()
        assert not any(e.kind == "error" for e in drain(session.events))
        release.set()
        wait_until(lambda: session.recognition_ready.is_set())
        session.feed(frame * 12 + b"\0" * 640 * 10)
        wait_until(
            lambda: any(e.kind == "transcript" for e in list(session.events.queue))
        )
    finally:
        release.set()
        close(session)


def test_canceled_stt_warmup_can_finish_for_the_new_epoch():
    entered, release = Event(), Event()

    class Slow(Recognizer):
        def __init__(self):
            self.prepares = 0

        def prepare(self, *, stop_event):
            self.prepares += 1
            if self.prepares == 1:
                entered.set()
                release.wait(2)

    recognizer = Slow()
    session = VoiceSession(recognizer, Synthesizer())
    try:
        assert entered.wait(1)
        session.interrupt()
        release.set()
        wait_until(lambda: session.recognition_ready.is_set())
        assert recognizer.prepares == 2
        session.submit_audio(Utterance(b"\0\0" * 24000))
        wait_until(
            lambda: any(e.kind == "transcript" for e in list(session.events.queue))
        )
        assert all(e.epoch == session.epoch for e in drain(session.events))
    finally:
        release.set()
        close(session)


def test_optional_evidence_cannot_fill_the_entire_critical_event_queue():
    session = VoiceSession(Recognizer(), Synthesizer())
    try:
        wait_until(lambda: session.recognition_ready.is_set())
        drain(session.events)
        for index in range(100):
            session.emit("perception", {"index": index}, session.epoch)
        assert session.events.qsize() == 32
        session.emit("transcript", {"text": "preserved"}, session.epoch)
        assert any(e.kind == "transcript" for e in drain(session.events))
    finally:
        close(session)


def test_perception_freshness_uses_capture_time_before_native_or_queue_delay(
    monkeypatch,
):
    import src.voice.session as session_module

    entered, release = Event(), Event()
    clock = [100.0]
    monkeypatch.setattr(
        session_module, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )

    class Slow(Perception):
        def analyze(self, *args, **kwargs):
            entered.set()
            release.wait(2)
            return super().analyze(*args, **kwargs)

    session = VoiceSession(Recognizer(), Synthesizer(), perception=Slow())
    try:
        session.submit_audio(Utterance(b"\1\0" * 24000))
        assert entered.wait(1)
        clock[0] = 800.0
        release.set()
        wait_until(
            lambda: any(e.kind == "perception" for e in list(session.events.queue))
        )
        evidence = next(e.data for e in drain(session.events) if e.kind == "perception")
        assert evidence["captured_at"] == 100.0
        assert evidence["analyzed_ms"] == 700_000
    finally:
        release.set()
        close(session)


def test_recognition_ready_is_set_before_its_ready_event_is_published():
    observed = []

    class ObservedSession(VoiceSession):
        def emit(self, kind, data, epoch):
            if kind == "ready" and data == "Recognizer":
                observed.append(self.recognition_ready.is_set())
            return super().emit(kind, data, epoch)

    session = ObservedSession(Recognizer(), Synthesizer())
    try:
        wait_until(lambda: bool(observed))
        assert observed == [True]
    finally:
        close(session)
