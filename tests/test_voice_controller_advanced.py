"""Speech evidence follows the matching request without blocking ordinary chat."""

from threading import Event
from types import SimpleNamespace

import pytest

from tests.test_voice_controller import controller
from src.voice.session import VoiceEvent


def transcript(controller, text="سلام", utterance_id=1):
    controller.session.events.put(
        VoiceEvent(
            "transcript",
            {
                "text": text,
                "truncated": False,
                "stt_ms": 12,
                "utterance_id": utterance_id,
            },
            controller.session.epoch,
        )
    )


def evidence(
    controller, *, source="speech", utterance_id=1, captured_at=None, analyzed_ms=0
):
    data = {
        "source": source,
        "utterance_id": utterance_id if source == "speech" else None,
        "speaker": {"id": "speaker_1", "status": "matched", "similarity": 0.93},
        "sounds": {"status": "known", "events": [{"label": "Speech", "score": 0.92}]},
        "analyzed_ms": analyzed_ms,
    }
    if captured_at is not None:
        data["captured_at"] = captured_at
    controller.session.events.put(
        VoiceEvent("perception", data, controller.session.epoch)
    )


def local_clock(monkeypatch, controller):
    controller.timer.stop()
    now = [100.0]
    monkeypatch.setattr(
        "src.studio.voice_controller.time", SimpleNamespace(monotonic=lambda: now[0])
    )
    return now


def send_with_context(controller, window):
    requests = []

    def send():
        text = window.input.toPlainText()
        requests.append((text, controller.context_for_prompt(text)))
        window.input.clear()
        controller.begin_turn()

    window.send = send
    return requests


def test_autosend_waits_briefly_for_correlated_waveform_evidence(
    controller, monkeypatch
):
    c, w = controller
    local_clock(monkeypatch, c)
    c._settings.update(auto_send=True, perception_enabled=True)
    requests = send_with_context(c, w)
    transcript(c)
    c.session.events.put(VoiceEvent("state", "Listening", c.session.epoch))
    c.poll()
    assert not requests and "waiting briefly" in c.dialog.status.text()
    evidence(c, captured_at=99.9)
    c.poll()
    assert requests[0][0] == "سلام"
    assert (
        requests[0][1]["observations"][0]["speaker"]["anonymous_session_label"]
        == "speaker_1"
    )
    assert not c._pending_transcript


def test_missing_acoustic_result_never_delays_autosend_beyond_deadline(
    controller, monkeypatch
):
    c, w = controller
    now = local_clock(monkeypatch, c)
    c._settings.update(auto_send=True, perception_enabled=True)
    requests = send_with_context(c, w)
    transcript(c)
    c.poll()
    assert not requests
    now[0] += 0.41
    c.poll()
    assert requests == [("سلام", None)]


def test_interrupt_discards_waiting_transcript_and_late_evidence(
    controller, monkeypatch
):
    c, w = controller
    now = local_clock(monkeypatch, c)
    c._settings.update(auto_send=True, perception_enabled=True)
    transcript(c)
    c.poll()
    old_epoch = c.session.epoch
    c.cancel_speech()
    c.session.events.put(
        VoiceEvent("perception", {"source": "speech", "utterance_id": 1}, old_epoch)
    )
    now[0] += 1
    c.poll()
    assert not w.sent and not w.input.toPlainText() and not c._pending_transcript


@pytest.mark.parametrize("captured_at, analyzed_ms", [(90.0, 1), (None, 6000)])
def test_delayed_ambient_evidence_uses_capture_age_not_gui_receipt(
    controller, monkeypatch, captured_at, analyzed_ms
):
    c, w = controller
    local_clock(monkeypatch, c)
    c._settings["perception_enabled"] = True
    evidence(c, source="ambient", captured_at=captured_at, analyzed_ms=analyzed_ms)
    transcript(c)
    c.poll()
    assert c.context_for_prompt("سلام") is None


def test_matching_speech_survives_newer_room_observations(controller, monkeypatch):
    c, w = controller
    local_clock(monkeypatch, c)
    c._settings["perception_enabled"] = True
    evidence(c, captured_at=90)
    for _ in range(4):
        evidence(c, source="ambient", captured_at=99.9)
    transcript(c)
    c.poll()
    context = c.context_for_prompt("سلام")
    assert len(context["observations"]) <= 3
    speech = [row for row in context["observations"] if row["source"] == "speech"]
    assert (
        len(speech) == 1
        and speech[0]["speaker"]["anonymous_session_label"] == "speaker_1"
    )


def test_combined_voice_draft_is_not_attributed_to_last_speaker(controller):
    c, w = controller
    c._settings["perception_enabled"] = True
    evidence(c)
    evidence(c, utterance_id=2)
    transcript(c, "First speaker", 1)
    transcript(c, "Second speaker", 2)
    c.poll()
    assert w.input.toPlainText() == "First speaker\nSecond speaker"
    assert c.context_for_prompt(w.input.toPlainText()) is None


def test_unrelated_typed_prompt_never_receives_room_or_speaker_context(controller):
    c, w = controller
    c._settings["perception_enabled"] = True
    evidence(c)
    evidence(c, source="ambient")
    c.poll()
    assert c.context_for_prompt("Ordinary typed request") is None
    transcript(c)
    c.poll()
    assert c.context_for_prompt("A different typed request") is None


def test_perception_failure_keeps_chat_live_without_repeated_deadline_waits(
    controller, monkeypatch
):
    c, w = controller
    local_clock(monkeypatch, c)
    c._settings.update(auto_send=True, perception_enabled=True)
    requests = send_with_context(c, w)
    c.session.events.put(
        VoiceEvent("perception_error", "Model unavailable", c.session.epoch)
    )
    transcript(c)
    c.poll()
    assert c.session is not None and requests == [("سلام", None)]
    # A later successful analysis restores bounded evidence pairing.
    evidence(c)
    c.poll()
    transcript(c, "Next utterance", 2)
    c.poll()
    assert len(requests) == 1 and c._pending_transcript


def test_first_playback_metric_does_not_restart_at_sentence_underruns(
    controller, monkeypatch
):
    c, w = controller
    now = local_clock(monkeypatch, c)
    c.begin_turn()
    now[0] += 0.1
    c.playback_started()
    assert c._metrics["first_playback_ms"] == 100
    now[0] += 3
    c.playback_started()
    assert c._metrics["first_playback_ms"] == 100
    c.begin_turn()
    assert "first_playback_ms" not in c._metrics
    now[0] += 0.2
    c.playback_started()
    assert c._metrics["first_playback_ms"] == 200


def test_synthesizer_readiness_does_not_claim_recognizer_ready(controller):
    c, w = controller
    c.session.recognition_ready = Event()
    c.session.events.put(VoiceEvent("ready", "Synthesizer", c.session.epoch))
    c.poll()
    assert "Preparing speech recognition" in c.dialog.status.text()
    c.session.recognition_ready.set()
    c.session.events.put(VoiceEvent("ready", "Recognizer", c.session.epoch))
    c.poll()
    assert "recognition ready" in c.dialog.status.text()


def test_kokoro_warmup_does_not_require_unused_stale_english_piper(
    controller, tmp_path
):
    c, w = controller
    c._provider_factory = None
    c._cached_providers = None
    model = tmp_path / "kokoro.onnx"
    voices = tmp_path / "voices.bin"
    persian = tmp_path / "persian.onnx"
    for path in (model, voices, persian, tmp_path / "persian.onnx.json"):
        path.write_text("fixture")
    _, synth = c._providers(
        {
            "language": "auto",
            "stt_model": str(tmp_path),
            "english_engine": "kokoro",
            "kokoro_model_path": str(model),
            "kokoro_voices_path": str(voices),
            "tts_fa_model": str(persian),
            "tts_en_model": str(tmp_path / "old-missing-en.onnx"),
        }
    )
    loaded = []
    synth.providers["fa"]._loader = lambda path, **kwargs: (
        loaded.append(path) or object()
    )
    synth.providers["en"]._engine = object()
    synth.prepare()
    assert loaded == [str(persian)]


def test_switch_between_kokoro_and_piper_without_using_stale_optional_voices(
    controller, tmp_path, monkeypatch
):
    c, _ = controller
    c._provider_factory = None
    model, voices = tmp_path / "kokoro.onnx", tmp_path / "voices.bin"
    english = tmp_path / "english.onnx"
    for path in (model, voices, english, tmp_path / "english.onnx.json"):
        path.touch()

    created = []

    class English:
        def __init__(self, *args, **kwargs):
            created.append("kokoro")

        def prepare(self, stop_event=None):
            pass

    class Piper:
        def __init__(self, paths, **kwargs):
            created.append(("piper", tuple(paths)))

        def prepare(self, stop_event=None):
            pass

    monkeypatch.setattr("src.voice.kokoro.KokoroProvider", English)
    monkeypatch.setattr("src.studio.voice_controller.PiperProvider", Piper)
    monkeypatch.setattr(
        "src.studio.voice_controller.FasterWhisperProvider",
        lambda *args, **kwargs: object(),
    )
    shared = {
        "input_mode": "microphone",
        "language": "auto",
        "stt_model": str(tmp_path),
        "kokoro_model_path": str(model),
        "kokoro_voices_path": str(voices),
        "tts_fa_model": str(tmp_path / "stale-persian.onnx"),
    }
    kokoro = {**shared, "english_engine": "kokoro", "tts_en_model": "/stale"}
    piper = {**shared, "english_engine": "piper", "tts_en_model": str(english)}

    _, synth = c._providers(kokoro)
    synth.prepare()
    assert created == ["kokoro"]
    assert synth.providers["fa"] is None
    with pytest.raises(Exception, match="local fa voice"):
        list(synth.synthesize("سلام", "auto"))

    c.apply_settings(kokoro)
    c.poll()
    assert c.session is not None and c._settings["english_engine"] == "kokoro"
    c.apply_settings(piper)
    c.poll()
    assert c.session is not None and c._settings["english_engine"] == "piper"
    assert created == ["kokoro", ("piper", ("en",))]
    c.apply_settings(kokoro)
    c.poll()
    assert c.session is not None and c._settings["english_engine"] == "kokoro"
    assert created[-1] == "kokoro"
    assert [item for item in created if isinstance(item, tuple)] == [("piper", ("en",))]


def test_voice_language_routes_automatically_even_if_old_setting_says_english(
    controller, tmp_path
):
    c, _ = controller
    c._provider_factory = None
    persian = tmp_path / "persian.onnx"
    english = tmp_path / "english.onnx"
    for path in (
        persian,
        english,
        tmp_path / "persian.onnx.json",
        tmp_path / "english.onnx.json",
    ):
        path.touch()

    recognizer, synthesizer = c._providers(
        {
            "input_mode": "microphone",
            "language": "en",  # A previously saved setting must not force English.
            "stt_model": str(tmp_path),
            "english_engine": "piper",
            "tts_fa_model": str(persian),
            "tts_en_model": str(english),
        }
    )

    assert recognizer.language == "auto"
    assert synthesizer.language == "auto"
    assert set(synthesizer.model_paths) == {"fa", "en"}


def test_voice_start_requires_at_least_one_configured_output_voice(
    controller, tmp_path
):
    c, _ = controller
    c._provider_factory = None
    with pytest.raises(ValueError, match="at least one local output voice"):
        c._providers(
            {
                "input_mode": "text",
                "language": "fa",
                "english_engine": "kokoro",
                "tts_fa_model": str(tmp_path / "missing.onnx"),
            }
        )


def test_cli_installed_paths_fill_empty_settings_but_preserve_custom_paths(
    controller, monkeypatch
):
    c, _ = controller
    monkeypatch.setattr(
        "src.studio.voice_controller.installed_models",
        lambda: {
            "kokoro-en": {"path": "/cache/kokoro.onnx"},
            "speaker-embedding": {"path": "/cache/speaker.onnx"},
            "audio-events": {"path": "/cache/sound.onnx"},
        },
    )
    settings = {
        "kokoro_model_path": "",
        "kokoro_voices_path": "",
        "speaker_model_path": "/custom/speaker.onnx",
    }
    c._installed_settings(settings)
    assert settings["kokoro_model_path"] == "/cache/kokoro.onnx"
    assert settings["kokoro_voices_path"] == "/cache/voices-v1.0.bin"
    assert settings["sound_labels_path"] == "/cache/class_labels_indices.csv"
    assert settings["speaker_model_path"] == "/custom/speaker.onnx"
