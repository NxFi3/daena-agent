"""Conversational interruption preserves new speech and cannot submit stale drafts."""

import os
from queue import Queue
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QPlainTextEdit, QWidget

from src.studio.voice_controller import VoiceController
from src.studio.voice_dialog import VoiceDialog
from src.voice.providers import AudioChunk
from src.voice.session import VoiceEvent
from tests.test_voice_controller import Audio


class DuplexAudio(Audio):
    speech_started = Signal()
    duplex_status = Signal(dict)

    def configure_duplex(self, mode="auto"):
        return {"mode": mode, "active": True, "backend": "scripted", "message": "Ready"}


class Session:
    def __init__(self, *args, **kwargs):
        self.epoch = 0
        self.events, self.audio = Queue(), Queue()
        self.spoken, self.interruptions, self.captured = [], [], []
        self.retired = True
        self.segmenter = SimpleNamespace(reset=self.reset_input)
        self.recognition_ready = SimpleNamespace(is_set=lambda: True)
        self.input_resets = 0

    def speak(self, text, language):
        self.spoken.append(text)

    def interrupt(self, *, preserve_input=False):
        self.interruptions.append(preserve_input)
        self.epoch += 1
        if not preserve_input:
            self.reset_input()
        for queue in (self.events, self.audio):
            while not queue.empty():
                queue.get_nowait()

    def reset_input(self):
        self.input_resets += 1
        self.captured.clear()

    def close(self):
        self.interrupt()

    def feed(self, data):
        self.captured.append(data)

    def finish_utterance(self):
        pass


@pytest.fixture
def conversation(qt_app, tmp_path):
    w = QWidget()
    w.workspace, w.session_id = tmp_path, "conversation"
    w.store = SimpleNamespace(root=tmp_path / "studio")
    w.input = QPlainTextEdit(w)
    w.busy = w._automation_run = False
    w.sent, w.stop_requests = [], []
    c = VoiceController(
        w,
        audio_factory=DuplexAudio,
        session_factory=Session,
        provider_factory=lambda settings: (object(), object()),
    )
    w.voice_controller = c
    c.dialog = VoiceDialog(parent=w)

    def send():
        w.sent.append(w.input.toPlainText())
        w.input.clear()
        if not w.busy:
            c.begin_turn()
            w.busy = True

    def stop():
        w.stop_requests.append("stop")
        c.cancel_speech()

    w.send, w.stop = send, stop
    c.start(
        {"input_mode": "microphone", "auto_send": True, "duplex_mode": "headphones"}
    )
    assert c.session is not None
    yield c, w
    c.close()
    w.close()
    qt_app.processEvents()


def play_reply(c):
    c.on_event({"type": "final_response", "text": "A reply that is still playing."})
    c.audio.play_pcm(b"\1\0" * 320, 16000)
    c.playback_started()


def transcript(c, text="Actually, make that shorter.", *, truncated=False):
    c.session.events.put(
        VoiceEvent(
            "transcript",
            {"text": text, "truncated": truncated, "stt_ms": 4},
            c.session.epoch,
        )
    )
    c.poll()


def drain_events():
    for _ in range(4):
        QApplication.processEvents()


def test_speech_onset_clears_old_audio_and_preserves_new_microphone_input(conversation):
    c, w = conversation
    play_reply(c)
    c.feed(b"new utterance pre-roll")
    resets = c.session.input_resets
    c._pending_pcm = (c.session.epoch, AudioChunk(16000, b"\1\0" * 100))
    c.audio.speech_started.emit()
    c.feed(b"new utterance continuation")
    assert c.session.interruptions == [True]
    assert c.session.input_resets == resets
    assert c.session.captured == [
        b"new utterance pre-roll",
        b"new utterance continuation",
    ]
    assert not c.audio.played and c._pending_pcm is None
    assert not w.stop_requests


def test_barge_in_requests_busy_agent_stop_once_and_suppresses_late_reply(conversation):
    c, w = conversation
    play_reply(c)
    w.busy = True
    c.audio.speech_started.emit()
    c.audio.speech_started.emit()
    assert w.stop_requests == ["stop"]
    assert c.session.interruptions == [True]
    spoken = list(c.session.spoken)
    c.on_event({"type": "run_start"})
    c.on_event({"type": "final_response", "text": "Canceled response arriving late"})
    assert c.session.spoken == spoken and not c.audio.played
    c.begin_turn()
    c.on_event({"type": "final_response", "text": "Fresh requested reply"})
    assert c.session.spoken[-1] == "Fresh requested reply"


def test_recognized_barge_in_waits_for_stopping_worker_then_submits_once(conversation):
    c, w = conversation
    play_reply(c)
    w.busy = True
    c.barge_in()
    transcript(c)
    assert not w.sent
    w.busy = False
    c.agent_idle()
    # Studio still has to retire its old worker after this callback returns.
    assert not w.sent
    drain_events()
    assert w.sent == ["Actually, make that shorter."]
    c.agent_idle()
    drain_events()
    assert len(w.sent) == 1


@pytest.mark.parametrize("intervention", ["edit", "mute", "session", "stop"])
def test_deferred_speech_cannot_submit_after_user_or_session_changed(
    conversation, intervention
):
    c, w = conversation
    play_reply(c)
    w.busy = True
    c.barge_in()
    transcript(c)
    if intervention == "edit":
        w.input.setPlainText("Keep this edited draft")
    elif intervention == "mute":
        c.mute(True)
    elif intervention == "session":
        w.session_id = "another conversation"
    else:
        c.stop()
    w.busy = False
    c.agent_idle()
    drain_events()
    assert not w.sent
    if intervention == "edit":
        assert w.input.toPlainText() == "Keep this edited draft"


def test_truncated_barge_in_stays_for_review_after_agent_stops(conversation):
    c, w = conversation
    play_reply(c)
    w.busy = True
    c.barge_in()
    transcript(c, "A recording that hit its limit", truncated=True)
    w.busy = False
    c.agent_idle()
    drain_events()
    assert not w.sent
    assert "hit its limit" in w.input.toPlainText()


def test_voice_text_submission_preempts_previous_spoken_answer(conversation):
    c, w = conversation
    play_reply(c)
    old_epoch = c.session.epoch
    c.send_review("Now answer another question")
    assert not c.audio.played and c.session.epoch > old_epoch
    assert w.sent == ["Now answer another question"]


def test_speech_onset_never_cancels_an_automation(conversation):
    c, w = conversation
    w.busy = w._automation_run = True
    c.audio.speech_started.emit()
    assert not w.stop_requests


def test_real_session_retains_first_syllables_across_interruption():
    from src.voice.session import VoiceSession
    from src.voice.streaming import SpeechSegmenter
    from tests.test_voice_session import drain, wait_until

    heard = []

    class Recognizer:
        def transcribe(self, pcm16, sample_rate, **kwargs):
            heard.append(pcm16)
            return "New utterance starts before the interruption"

    class Synthesizer:
        def synthesize(self, *args, **kwargs):
            return iter(())

    session = VoiceSession(Recognizer(), Synthesizer())
    first = b"\x10\x00" * 320 * 8
    continuation = b"\x20\x00" * 320 * 8
    quiet = b"\0" * 640
    try:
        assert session.recognition_ready.wait(1)
        session.segmenter = SpeechSegmenter(
            silence_ms=200, vad=lambda frame: frame != quiet
        )
        session.feed(first)
        epoch = session.interrupt(preserve_input=True)
        session.feed(continuation + quiet * 10)
        wait_until(lambda: bool(heard))
        assert heard == [first + continuation + quiet * 10]
        wait_until(lambda: not session.events.empty())
        events = drain(session.events)
        transcripts = [event for event in events if event.kind == "transcript"]
        assert len(transcripts) == 1 and transcripts[0].epoch == epoch
    finally:
        session.close()
        wait_until(lambda: session.retired)


def test_other_queued_work_prevents_deferred_voice_from_starting_a_turn(conversation):
    c, w = conversation
    play_reply(c)
    w.busy = True
    c.barge_in()
    transcript(c)
    w.queued = ["Previously queued work"]
    w.busy = False
    c.agent_idle()
    drain_events()
    assert not w.sent and w.queued == ["Previously queued work"]
    assert "make that shorter" in w.input.toPlainText()


def test_new_direct_preview_can_be_interrupted_after_a_previous_barge_in(conversation):
    c, _ = conversation
    play_reply(c)
    c.barge_in()
    c.speak_text("A newly requested direct audition")
    c.audio.play_pcm(b"\1\0" * 320, 16000)
    c.playback_started()
    old_epoch = c.session.epoch
    c.audio.speech_started.emit()
    assert c.session.epoch == old_epoch + 1 and not c.audio.played
    assert c.session.interruptions[-1] is True


def test_voice_text_steering_stops_old_audio_but_keeps_new_final_audible(conversation):
    c, w = conversation
    play_reply(c)
    w.busy = True
    c.send_review("Please change the answer you are working on")
    assert w.sent == ["Please change the answer you are working on"]
    assert not c.audio.played
    c.on_event(
        {"type": "final_response", "text": "The answer incorporates your steering"}
    )
    assert c.session.spoken[-1] == "The answer incorporates your steering"


def test_typed_request_waits_for_interrupted_worker_and_preserves_chat_draft(
    conversation,
):
    c, w = conversation
    play_reply(c)
    w.busy = True
    c.barge_in()
    w.input.setPlainText("An unrelated unsent draft")
    c.send_review("A new typed Voice request")
    assert not w.sent
    assert w.input.toPlainText() == "An unrelated unsent draft"
    w.busy = False
    c.agent_idle()
    assert not w.sent
    drain_events()
    assert w.sent == ["A new typed Voice request"]
    assert w.input.toPlainText() == "An unrelated unsent draft"


def test_ending_voice_discards_typed_request_waiting_for_interrupted_worker(
    conversation,
):
    c, w = conversation
    play_reply(c)
    w.busy = True
    c.barge_in()
    c.send_review("Do not send this after ending Voice")
    c.stop()
    w.busy = False
    c.agent_idle()
    drain_events()
    assert not w.sent
