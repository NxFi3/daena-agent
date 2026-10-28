"""Local voice I/O contracts without opening host microphones or speakers."""

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QObject, Signal
from PySide6.QtTest import QTest

from src.studio.voice_audio import AudioDevices, PCMNormalizer, _PlaybackBuffer
from src.studio.voice_dialog import VoiceDialog


def test_normalizer_preserves_pcm16_and_partial_channel_frames():
    values = np.arange(-300, 300, dtype="<i2")
    normalizer = PCMNormalizer(16000)
    raw = values.tobytes()
    assert normalizer.feed(raw[:5]) + normalizer.feed(raw[5:]) == raw
    stereo = np.stack((values, -values), axis=1).astype("<i2").tobytes()
    normalizer = PCMNormalizer(16000, channels=2)
    output = normalizer.feed(stereo[:9]) + normalizer.feed(stereo[9:])
    assert np.all(np.frombuffer(output, dtype="<i2") == 0)


def test_streaming_resample_keeps_fractional_clock_across_irregular_reads():
    samples = (np.sin(np.arange(44100) * (2 * np.pi * 330 / 44100)) * 0.8).astype("<f4")
    raw = samples.tobytes()
    reference = PCMNormalizer(44100, sample_format="float32", use_soxr=False).feed(raw)
    normalizer = PCMNormalizer(44100, sample_format="float32", use_soxr=False)
    chunks = [normalizer.feed(raw[i : i + 997]) for i in range(0, len(raw), 997)]
    result = b"".join(chunks)
    assert len(result) == len(reference)
    assert len(result) // 2 in {15999, 16000}
    assert (
        np.max(
            np.abs(
                np.frombuffer(result, dtype="<i2").astype(int)
                - np.frombuffer(reference, dtype="<i2")
            )
        )
        <= 1
    )
    assert len(normalizer._samples) < 8
    assert len(normalizer._raw) < 4


def test_normalizer_handles_nonfinite_float_and_clips_without_wrap():
    raw = np.array([np.nan, np.inf, -np.inf, 2.0, -2.0], dtype="<f4").tobytes()
    assert list(
        np.frombuffer(
            PCMNormalizer(16000, sample_format="float32").feed(raw), dtype="<i2"
        )
    ) == [0, 32767, -32768, 32767, -32768]


def test_playback_buffer_is_bounded_fifo_and_can_be_cancelled(qt_app):
    buffer = _PlaybackBuffer(12)
    assert buffer.append(b"12345")
    assert buffer.append(b"6789")
    assert not buffer.append(b"overflow")
    assert bytes(buffer.read(7)) == b"1234567"
    assert buffer.bytesAvailable() == 2
    assert buffer.append(b"ab")
    assert bytes(buffer.read(8)) == b"89ab"
    buffer.clear()
    assert buffer.bytesAvailable() == 0
    buffer.close()


def test_audio_module_does_not_import_multimedia_before_start():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.studio.voice_audio; assert 'PySide6.QtMultimedia' not in sys.modules",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_audio_errors_are_actionable_and_close_is_idempotent(qt_app, monkeypatch):
    import src.studio.voice_audio as module

    def unavailable():
        raise ImportError("optional audio library missing")

    monkeypatch.setattr(module, "_qt_audio", unavailable)
    audio = AudioDevices()
    errors = []
    audio.fault.connect(errors.append)
    assert not audio.start_capture()
    assert errors and "Microphone" in errors[0]
    assert AudioDevices.devices()["error"]
    audio.close()
    audio.close()
    assert not audio.play_pcm(b"\0\0", 16000)


class Format:
    Int16 = SimpleNamespace(name="Int16")

    def __init__(self):
        self.rate, self.channels, self.kind = 16000, 1, self.Int16

    def setSampleRate(self, value):
        self.rate = value

    def setChannelCount(self, value):
        self.channels = value

    def setSampleFormat(self, value):
        self.kind = value

    def sampleRate(self):
        return self.rate

    def channelCount(self):
        return self.channels

    def sampleFormat(self):
        return self.kind

    def bytesForDuration(self, us):
        return int(self.rate * self.channels * 2 * us / 1000000)

    def bytesPerFrame(self):
        return self.channels * 2


class Device:
    def isNull(self):
        return False

    def isFormatSupported(self, fmt):
        return True

    def preferredFormat(self):
        return Format()

    def id(self):
        return b"test-device"

    def description(self):
        return "Test device"


class Media:
    @staticmethod
    def audioInputs():
        return [Device()]

    @staticmethod
    def audioOutputs():
        return [Device()]

    @staticmethod
    def defaultAudioInput():
        return Device()

    @staticmethod
    def defaultAudioOutput():
        return Device()


class Audio:
    StoppedState, ActiveState, IdleState, SuspendedState = range(4)
    NoError = 0


class FakeSink(QObject):
    stateChanged = Signal(object)

    def __init__(self, device, fmt, parent=None):
        super().__init__(parent)
        self.current = Audio.StoppedState
        self.source = None
        self.stops = 0
        self.starts = 0
        self.resumes = 0

    def setBufferSize(self, count):
        self.buffer_size = count

    def state(self):
        return self.current

    def error(self):
        return Audio.NoError

    def start(self, source):
        self.starts += 1
        self.source = source
        self.current = Audio.ActiveState
        self.stateChanged.emit(self.current)

    def resume(self):
        self.resumes += 1
        assert self.current == Audio.SuspendedState
        self.current = Audio.ActiveState

    def stop(self):
        self.stops += 1
        self.current = Audio.StoppedState
        self.stateChanged.emit(self.current)

    def drain(self):
        self.source.readAll()
        self.current = Audio.IdleState
        self.stateChanged.emit(self.current)


def test_playback_echo_gate_backpressure_and_explicit_interrupt(qt_app, monkeypatch):
    import src.studio.voice_audio as module

    monkeypatch.setattr(
        module, "_qt_audio", lambda: (Audio, Format, FakeSink, None, Media)
    )
    audio = AudioDevices()
    started, stopped, errors = [], [], []
    audio.playback_started.connect(lambda: started.append(True))
    audio.playback_stopped.connect(lambda: stopped.append(True))
    audio.fault.connect(errors.append)
    frame = np.zeros(1600, dtype="<i2").tobytes()
    assert audio.play_pcm(frame, 16000)
    sink = audio._sink
    assert audio._echo_blocked and len(started) == 1
    assert audio.queued_seconds == pytest.approx(0.1)
    # Temporary backpressure retains the caller's chunk and does not stop output.
    assert audio._playback.append(b"\0" * (16000 * 2 * 19))
    before = audio._playback.bytesAvailable()
    assert not audio.play_pcm(b"\0" * (16000 * 2), 16000)
    assert audio._playback.bytesAvailable() == before
    assert not errors and sink.stops == 0
    sink.drain()
    assert len(stopped) == 1 and audio._echo_blocked
    QTest.qWait(220)
    assert not audio._echo_blocked
    assert audio.play_pcm(frame, 16000)
    assert len(started) == 2
    assert sink.starts == 2 and sink.resumes == 0
    assert sink.source.bytesAvailable() == len(frame)
    audio.stop_playback()
    assert not audio._echo_blocked and audio._sink is None
    assert len(stopped) == 2
    audio.close()


def test_voice_dialog_controls_do_not_start_devices_or_download_models(qt_app):
    dialog = VoiceDialog({"language": "fa", "auto_send": False})
    starts, downloads, sends = [], [], []
    dialog.start_requested.connect(starts.append)
    dialog.download_requested.connect(downloads.append)
    dialog.send_requested.connect(sends.append)
    assert dialog.values()["language"] == "auto"
    assert dialog.language.count() == 1
    assert dialog.values()["stt_model"] == "large-v3-turbo"
    assert not dialog.values()["auto_send"]
    assert not dialog.values()["speak_progress"]
    assert not starts and not downloads
    dialog.stt_download.click()
    assert downloads == ["whisper-turbo"]
    dialog.start_button.click()
    assert len(starts) == 1
    dialog.set_active(True)
    assert dialog.model_page.isEnabled()  # Live settings use explicit Apply.
    assert dialog.interrupt_button.isEnabled()
    # Text-only mode does not expose microphone/Whisper settings or start
    # recognition; switch back to microphone mode to change those controls.
    dialog._select(dialog.input_mode, "text")
    dialog._update_input_mode()
    assert dialog.values()["input_mode"] == "text"
    assert not dialog.stt_model.isEnabled() and not dialog.stt_download.isEnabled()
    assert not dialog.stt_device.isEnabled()
    assert not dialog.input_device.isEnabled()
    dialog.set_transcript("سلام Daena")
    dialog.send_button.click()
    assert sends == ["سلام Daena"]
    dialog.set_metrics({"stt_ms": 260, "first_pcm_ms": 180, "queued_seconds": 0.4})
    assert "260 ms" in dialog.metrics.text()
    dialog.set_devices(
        {"inputs": [{"id": "123", "name": "Mic", "default": True}], "outputs": []}
    )
    assert dialog.input_device.count() == 2
    dialog.close()


def test_soxr_streaming_keeps_chunk_boundaries_and_flushes_filter_tail():
    pytest.importorskip("soxr")
    raw = (
        (np.sin(np.arange(48000) * (2 * np.pi * 440 / 48000)) * 28000)
        .astype("<i2")
        .tobytes()
    )
    reference = PCMNormalizer(48000)
    expected = reference.feed(raw) + reference.flush()
    converter = PCMNormalizer(48000)
    actual = (
        b"".join(converter.feed(raw[i : i + 997]) for i in range(0, len(raw), 997))
        + converter.flush()
    )
    assert actual == expected
    assert len(actual) == 16000 * 2
    assert converter.flush() == b""
    with pytest.raises(ValueError):
        converter.feed(b"\0\0")


def test_mixed_voice_rates_preserve_previous_resampler_tail(qt_app):
    pytest.importorskip("soxr")
    audio = AudioDevices()
    audio._output_format = Format()
    first = (
        (np.sin(np.arange(9600) * (2 * np.pi * 330 / 48000)) * 25000)
        .astype("<i2")
        .tobytes()
    )
    second = (
        (np.sin(np.arange(2400) * (2 * np.pi * 550 / 24000)) * 25000)
        .astype("<i2")
        .tobytes()
    )
    reference_first = PCMNormalizer(48000)
    reference_second = PCMNormalizer(24000)
    expected = (
        reference_first.feed(first)
        + reference_first.flush()
        + reference_second.feed(second)
        + reference_second.flush()
    )
    actual = audio._encode_output(first, 48000) + audio._encode_output(second, 24000)
    actual += audio._format_output(audio._output_normalizer.flush())
    assert actual == expected
    assert len(actual) == 4800 * 2
    audio.close()


def test_final_filter_tail_can_start_playback_after_a_tiny_first_chunk(
    qt_app, monkeypatch
):
    pytest.importorskip("soxr")
    import src.studio.voice_audio as module

    class Native16k(Device):
        def isFormatSupported(self, fmt):
            return False

    class NativeMedia(Media):
        @staticmethod
        def defaultAudioOutput():
            return Native16k()

    monkeypatch.setattr(
        module, "_qt_audio", lambda: (Audio, Format, FakeSink, None, NativeMedia)
    )
    audio = AudioDevices()
    raw = np.zeros(120, dtype="<i2").tobytes()
    assert audio.play_pcm(raw, 48000)
    assert audio._sink.starts == 0
    audio.finish_playback()
    assert audio._sink.starts == 1
    assert audio._playback.bytesAvailable() == 40 * 2
    assert audio._echo_blocked
    audio.close()


def test_missing_pipewire_config_selects_available_safe_default_before_qt(monkeypatch):
    import ctypes.util
    import src.studio.voice_audio as module

    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(module, "_pipewire_config_paths", lambda: [])
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: "libpulse.so.0")
    monkeypatch.delenv("QT_AUDIO_BACKEND", raising=False)
    module._prepare_audio_backend()
    assert os.environ["QT_AUDIO_BACKEND"] == "pulseaudio"


def test_explicit_pipewire_configuration_failure_is_reported_without_native_call(
    monkeypatch,
):
    import src.studio.voice_audio as module

    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(module, "_pipewire_config_paths", lambda: [])
    monkeypatch.setenv("QT_AUDIO_BACKEND", "pipewire")
    with pytest.raises(RuntimeError, match="client.conf"):
        module._prepare_audio_backend()
    assert os.environ["QT_AUDIO_BACKEND"] == "pipewire"


def test_backend_preflight_respects_configured_backend_and_native_platform(monkeypatch):
    import src.studio.voice_audio as module

    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setenv("QT_AUDIO_BACKEND", "alsa")

    def unexpected():
        raise AssertionError("Another selected backend must not inspect PipeWire")

    monkeypatch.setattr(module, "_pipewire_config_paths", unexpected)
    module._prepare_audio_backend()
    assert os.environ["QT_AUDIO_BACKEND"] == "alsa"
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.delenv("QT_AUDIO_BACKEND")
    module._prepare_audio_backend()
    assert "QT_AUDIO_BACKEND" not in os.environ


def test_native_device_enumeration_finishes_safely_in_a_fresh_process():
    env = dict(os.environ)
    env.pop("QT_AUDIO_BACKEND", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from PySide6.QtCore import QCoreApplication; "
            "from src.studio.voice_audio import AudioDevices; "
            "app=QCoreApplication([]); result=AudioDevices.devices(); "
            "assert isinstance(result.get('inputs'),list); print('AUDIO_ENUMERATION_COMPLETE')",
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "AUDIO_ENUMERATION_COMPLETE" in result.stdout


def test_manual_finish_capture_flushes_short_native_tail_before_utterance_end(qt_app):
    pytest.importorskip("soxr")
    raw = (
        (np.sin(np.arange(1440) * (2 * np.pi * 440 / 48000)) * 24000)
        .astype("<i2")
        .tobytes()
    )
    reference = PCMNormalizer(48000)
    expected = reference.feed(raw) + reference.flush()
    audio = AudioDevices()
    audio._normalizer = PCMNormalizer(48000)
    # A device callback has delivered the beginning, and one final native read
    # remains at the moment the user presses Finish speaking.
    audio._input_pending.extend(audio._normalizer.feed(raw[:1000]))
    remainder = raw[1000:]
    audio._input_io = SimpleNamespace(readAll=lambda: remainder)
    frames = []
    audio.pcm_ready.connect(frames.append)
    assert audio.finish_capture()
    assert frames and all(len(frame) == 640 for frame in frames)
    actual = b"".join(frames)
    assert actual[: len(expected)] == expected
    assert set(actual[len(expected) :]) <= {0}
    assert len(actual) == 2 * 640
    assert not audio._normalizer._flushed
    assert not audio._input_pending
    # No final PCM is replayed on the next manual finish.
    audio._input_io = None
    assert audio.finish_capture()
    assert len(frames) == 2
    audio.close()


@pytest.mark.parametrize("gate", ["_muted", "_echo_blocked", "_closed"])
def test_finish_capture_does_not_emit_while_microphone_is_gated(qt_app, gate):
    audio = AudioDevices()
    audio._normalizer = PCMNormalizer(16000)
    audio._input_pending.extend(b"\x01\x02" * 40)
    setattr(audio, gate, True)
    frames = []
    audio.pcm_ready.connect(frames.append)
    assert not audio.finish_capture()
    assert not frames
    audio.close()
