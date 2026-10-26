"""Voice settings and acoustic evidence stay local, optional and reviewable."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from src.studio.voice_dialog import VoiceDialog


def test_turn_taking_defaults_and_saved_review_choice_round_trip(qt_app):
    dialog = VoiceDialog()
    assert dialog.values()["auto_send"] is True
    assert dialog.values()["duplex_mode"] == "auto"
    changes = []
    dialog.settings_changed.connect(changes.append)
    dialog.populate({"auto_send": False, "duplex_mode": "headphones"})
    assert not changes
    assert not dialog.values()["auto_send"]
    assert dialog.values()["duplex_mode"] == "headphones"
    assert "Wear headphones" in dialog.input_note.text()
    dialog.set_active(True)
    dialog._select(dialog.duplex_mode, "off")
    assert changes[-1]["duplex_mode"] == "off"
    assert dialog.duplex_mode.isEnabled()
    assert "Stop reply" in dialog.input_note.text()
    dialog._select(dialog.input_mode, "text")
    assert not dialog.duplex_mode.isEnabled()
    assert "Whisper and microphone are off" in dialog.input_note.text()
    dialog._select(dialog.input_mode, "microphone")
    assert dialog.duplex_mode.isEnabled()
    dialog.close()


def test_applied_duplex_status_survives_draft_mode_changes_and_is_plain_text(qt_app):
    dialog = VoiceDialog()
    dialog.set_duplex_status(
        {
            "mode": "auto",
            "active": True,
            "backend": "WebRTC",
            "message": "Echo control ready <b>locally</b>",
        }
    )
    status = dialog.duplex_status.text()
    assert status == "Active: Echo control ready <b>locally</b> · WebRTC"
    assert dialog.duplex_status.textFormat() == Qt.PlainText
    dialog._select(dialog.duplex_mode, "off")
    assert dialog.duplex_status.text() == status
    assert "Manual" in dialog.input_note.text()
    dialog.close()


def test_advanced_voice_settings_are_opt_in_and_downloads_are_explicit(qt_app):
    dialog = VoiceDialog()
    downloads, starts = [], []
    dialog.download_requested.connect(downloads.append)
    dialog.start_requested.connect(starts.append)
    settings = dialog.values()
    assert settings["english_engine"] == "piper"
    assert settings["kokoro_voice"] == "af_heart"
    assert settings["kokoro_cpu_threads"] == 2
    assert settings["speaker_threshold"] == 0.88
    assert settings["sound_threshold"] == 0.25
    assert settings["speech_speed"] == 1.0
    assert not settings["perception_enabled"]
    assert not settings["sound_events_enabled"]
    assert dialog.kokoro_page.isHidden()
    dialog._select(dialog.english_engine, "kokoro")
    assert not dialog.kokoro_page.isHidden()
    assert not dialog.tts_en_model.isEnabled()
    assert dialog.tts_fa_model.isEnabled()
    assert not downloads and not starts
    dialog.kokoro_download.click()
    dialog.speaker_download.click()
    dialog.sound_download.click()
    assert downloads == ["kokoro-en", "speaker-embedding", "audio-events"]
    assert not starts
    dialog.close()


def test_advanced_settings_round_trip_and_emit_single_coherent_start(qt_app):
    settings = {
        "english_engine": "kokoro",
        "kokoro_model_path": "/models/kokoro.onnx",
        "kokoro_voices_path": "/models/voices.bin",
        "kokoro_voice": "am_adam",
        "kokoro_cpu_threads": 3,
        "speech_speed": 0.85,
        "perception_enabled": True,
        "speaker_model_path": "/models/speaker.onnx",
        "sound_model_path": "/models/events.onnx",
        "sound_labels_path": "/models/events.csv",
        "speaker_threshold": 0.91,
        "sound_threshold": 0.35,
        "sound_events_enabled": True,
        "tts_fa_model": "/models/persian.onnx",
    }
    dialog = VoiceDialog()
    changed, starts = [], []
    dialog.settings_changed.connect(changed.append)
    dialog.start_requested.connect(starts.append)
    dialog.populate(settings)
    assert not changed, "Restoring settings must not publish intermediate values"
    assert {key: dialog.values()[key] for key in settings} == settings
    dialog.speech_speed.setValue(1.1)
    assert len(changed) == 1 and changed[-1]["speech_speed"] == 1.1
    dialog.start_button.click()
    assert starts == [dialog.values()]
    dialog.set_active(True)
    assert dialog.perception_page.isEnabled()
    assert dialog.kokoro_cpu_threads.isEnabled()
    assert dialog.speaker_threshold.isEnabled()
    assert dialog.sound_threshold.isEnabled()
    dialog.set_active(False)
    assert dialog.perception_page.isEnabled()
    dialog.close()


def test_model_evidence_is_plain_bounded_and_does_not_claim_real_identity(qt_app):
    dialog = VoiceDialog()
    dialog.set_perception(
        {
            "speaker": {
                "id": "speaker_2",
                "status": "matched",
                "similarity": 0.834,
                "basis": "session_embedding",
            },
            "sounds": {
                "status": "known",
                "events": [
                    {"label": "Speech", "score": 0.91},
                    {"label": "<b>Music</b>", "score": 0.43},
                ],
            },
        }
    )
    assert dialog.speaker_observation.text() == "Speaker 2 · cosine similarity 0.83"
    assert "91%" in dialog.sound_observation.text()
    assert "<b>Music</b>" in dialog.sound_observation.text()
    assert dialog.sound_observation.textFormat() == Qt.PlainText
    assert "Speaker 2" in dialog.perception_summary.text()
    assert "confidence" not in dialog.speaker_observation.text().lower()
    assert not dialog.transcript.toPlainText(), (
        "Acoustic observations are not user commands"
    )
    dialog.set_perception(
        {
            "speaker": {"id": None, "status": "too_short"},
            "sounds": {"status": "unknown", "events": []},
        }
    )
    assert "too short" in dialog.speaker_observation.text()
    assert "no class passed" in dialog.sound_observation.text()
    assert "Speaker uncertain" in dialog.perception_summary.text()
    dialog.close()
