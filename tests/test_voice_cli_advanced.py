"""Offline audition and raw-audio diagnostics, including installed engine contracts."""

import json
import os
from pathlib import Path
import wave

import pytest

from src.voice.__main__ import main
from src.voice.providers import AudioChunk


def test_kokoro_cli_does_not_load_unused_persian_or_download(tmp_path, monkeypatch):
    import src.voice.kokoro

    calls = []

    class English:
        def __init__(self, model, voices, **options):
            calls.append((model, voices, options))

        def synthesize(self, text, language="en", stop_event=None):
            yield AudioChunk(24000, b"\0\0" * 2400)

    monkeypatch.setattr(src.voice.kokoro, "KokoroProvider", English)
    monkeypatch.setattr("src.voice.models.installed_models", lambda *args: {})
    output = tmp_path / "english.wav"
    assert (
        main(
            [
                "synthesize",
                "--engine",
                "kokoro",
                "--language",
                "en",
                "--text",
                "Hello",
                "--kokoro-model",
                "local.onnx",
                "--kokoro-voices",
                "voices.bin",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    with wave.open(str(output), "rb") as wav:
        assert wav.getframerate() == 24000 and wav.getnframes() == 2400
    assert calls[0][2]["cpu_threads"] == 2


def test_cli_acoustic_analysis_reads_waveform_without_transcription(
    tmp_path, monkeypatch, capsys
):
    import src.voice.perception

    recording = tmp_path / "room.wav"
    with wave.open(str(recording), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\1\0" * 16000 * 10)
    calls = []

    class Perception:
        def __init__(self, *args):
            pass

        def analyze(self, pcm, **options):
            calls.append((len(pcm), options))
            return {"sounds": {"status": "unknown", "events": []}}

    monkeypatch.setattr(src.voice.perception, "LocalAudioPerception", Perception)
    monkeypatch.setattr("src.voice.models.installed_models", lambda: {})
    assert (
        main(
            [
                "analyze",
                str(recording),
                "--sound-model",
                "local.onnx",
                "--sound-labels",
                "labels.csv",
                "--ambient",
            ]
        )
        == 0
    )
    assert calls == [(16000 * 8 * 2, {"speech": False})]
    result = json.loads(capsys.readouterr().out)
    assert result["file_truncated"] and result["sounds"]["status"] == "unknown"


def test_invalid_speaking_speed_does_not_create_output(tmp_path):
    output = tmp_path / "invalid.wav"
    with pytest.raises(SystemExit):
        main(["synthesize", "--text", "hello", "--speed", "0", "--output", str(output)])
    assert not output.exists()


def test_native_kokoro_local_model_emits_finite_pcm_without_network():
    directory = os.environ.get("DAENA_TEST_VOICE_MODELS")
    if not directory:
        pytest.skip(
            "Set DAENA_TEST_VOICE_MODELS to explicitly provisioned local models"
        )
    from src.voice.kokoro import KokoroProvider

    root = Path(directory)
    provider = KokoroProvider(
        root / "kokoro-v1.0.onnx", root / "voices-v1.0.bin", cpu_threads=2
    )
    provider.prepare()
    chunks = list(provider.synthesize("Hello, I am Daena.", "en"))
    assert chunks and all(chunk.sample_rate == 24000 for chunk in chunks)
    assert all(0 < len(chunk.pcm16) <= 4800 for chunk in chunks)
    assert any(any(chunk.pcm16) for chunk in chunks)
