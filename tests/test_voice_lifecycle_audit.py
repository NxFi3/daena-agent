"""Voice reconfiguration retains user intent across retiring native workers."""

import json

import pytest

from tests.test_voice_live_integration import Output, Session, voice


@pytest.fixture
def restarting(voice):
    c, w, synth = voice
    sessions = []

    class RetiringSession(Session):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            sessions.append(self)

        def close(self):
            super().close()
            self.retired = False

    c._session_factory = RetiringSession
    yield c, w, sessions
    # Native retirement is controlled explicitly in these tests. The owner
    # fixture still exercises ordinary controller shutdown afterward.
    for session in sessions:
        session.retired = True
        session.close = lambda owner=session: setattr(owner, "retired", True)


def test_latest_valid_apply_replaces_pending_configuration(restarting):
    c, w, sessions = restarting
    c.start(c.dialog.values())
    old = c.session
    c.apply_settings({**c.dialog.values(), "speech_speed": 1.1})
    assert c.session is None
    c.apply_settings({**c.dialog.values(), "speech_speed": 1.3})
    c.on_event({"type": "final_response", "text": "Reply retained during restart"})
    old.retired = True
    c.poll()
    assert len(sessions) == 2
    assert c._settings["speech_speed"] == 1.3
    assert c.session.spoken == ["Reply retained during restart"]
    assert not w.stops


def test_invalid_apply_and_unapplied_draft_cannot_replace_pending_selection(restarting):
    c, _, _ = restarting
    c.start(c.dialog.values())
    old = c.session
    c.apply_settings({**c.dialog.values(), "speech_speed": 1.2})
    selected = dict(c._restart_settings)
    c.dialog.kokoro_model_path.setText("unapplied-model.onnx")
    assert json.loads(c._settings_path().read_text()) == selected

    def validate(settings):
        if settings.get("speech_speed") == 1.4:
            raise ValueError("Invalid selected voice")
        return object(), object()

    c._provider_factory = validate
    c.apply_settings({**c.dialog.values(), "speech_speed": 1.4})
    assert c._restart_settings == selected
    assert json.loads(c._settings_path().read_text()) == selected
    old.retired = True
    c.poll()
    assert c._settings == selected and c.session is not None
    assert "not applied" in c.dialog.error_label.text()


def test_input_mode_change_during_restart_is_the_mode_that_resumes(restarting):
    c, _, _ = restarting
    c.start(c.dialog.values())
    old = c.session
    c.apply_settings({**c.dialog.values(), "speech_speed": 1.1})
    c.switch_input_mode("microphone")
    assert c._restart_settings["input_mode"] == "microphone"
    old.retired = True
    c.poll()
    assert c.session is not None and c._settings["input_mode"] == "microphone"


def test_stop_reply_while_restarting_keeps_late_answer_suppressed(restarting):
    c, _, _ = restarting
    c.start(c.dialog.values())
    old = c.session
    c.apply_settings({**c.dialog.values(), "speech_speed": 1.1})
    c.cancel_speech()
    old.retired = True
    c.poll()
    c.on_event({"type": "final_response", "text": "Canceled answer arriving late"})
    assert c.session is not None and c._speech_suppressed
    assert not c.session.spoken


def test_model_restart_preserves_mute_before_opening_replacement_microphone(restarting):
    c, _, _ = restarting
    devices = []

    class TrackedOutput(Output):
        def __init__(self, parent):
            super().__init__(parent)
            self.muted = False
            self.captured_while_muted = []
            devices.append(self)

        def set_muted(self, value):
            self.muted = bool(value)

        def start_capture(self, *args):
            self.captured_while_muted.append(self.muted)
            return True

    c._audio_factory = TrackedOutput
    c.start({"input_mode": "microphone", "language": "en"})
    old = c.session
    c.mute(True)
    c.apply_settings(
        {"input_mode": "microphone", "language": "en", "speech_speed": 1.1}
    )
    old.retired = True
    c.poll()
    assert len(devices) == 2
    assert devices[-1].captured_while_muted == [True]
    assert c._muted and c.dialog.mute_button.isChecked()
    c.output_idle()
    assert c.dialog.status.text() == "Microphone muted"


def test_retired_audio_signals_cannot_stop_or_feed_replacement_session(restarting):
    c, _, _ = restarting
    c.start(c.dialog.values())
    old_session, old_audio = c.session, c.audio
    c.apply_settings({**c.dialog.values(), "speech_speed": 1.2})
    old_session.retired = True
    c.poll()
    active = c.session
    received = []
    active.feed = received.append
    old_audio.pcm_ready.emit(b"old audio")
    old_audio.playback_started.emit()
    old_audio.fault.emit("An old queued output failure")
    assert c.session is active and not received and not c._playback_active
    c.audio.pcm_ready.emit(b"current audio")
    assert received == [b"current audio"]


def test_deferred_text_from_previous_conversation_is_not_restored(voice):
    c, w, _ = voice
    c.start(c.dialog.values())
    c._deferred_voice_text = (c.session, c._signature(), "Old conversation message")
    w.session_id = "new-session"
    c.dialog.set_transcript("New conversation draft")
    c._send_deferred_text()
    assert c.dialog.transcript.toPlainText() == "New conversation draft"
    assert not w.sent
