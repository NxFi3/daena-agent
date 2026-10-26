"""Interruptible native I/O keeps capture and consumed output synchronized."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from src.studio.voice_audio import AudioDevices, PCMNormalizer, _PlaybackBuffer
from tests.test_voice_audio import Audio, FakeSink, Format, Media


class FakeDuplex:
    def __init__(self, mode="auto", *, delay_ms=80):
        self.mode, self.delay_ms = mode, delay_ms
        self.can_listen_during_playback = mode != "off"
        self.status = {
            "mode": mode,
            "backend": "test-aec" if mode == "auto" else "none",
            "active": self.can_listen_during_playback,
            "message": "Test local echo control",
        }
        self.renders, self.frames, self.order = [], [], []
        self.resets = 0
        self.closed = False

    def feed_render(self, pcm, timestamp=None):
        self.renders.append((pcm, timestamp))
        self.order.append("render")

    def process_capture(self, pcm, *, playback_active=False):
        self.frames.append((pcm, playback_active))
        self.order.append("capture")
        return pcm, len(self.frames) == 2

    def reset_input(self):
        self.resets += 1

    def close(self):
        self.closed = True


@pytest.fixture
def duplex_audio(qt_app, monkeypatch):
    import src.studio.voice_audio as module
    import src.voice.duplex as engine

    monkeypatch.setattr(engine, "DuplexProcessor", FakeDuplex)
    monkeypatch.setattr(
        module, "_qt_audio", lambda: (Audio, Format, FakeSink, None, Media)
    )
    audio = AudioDevices()
    yield audio
    audio.close()


def test_render_tap_sees_only_native_pulls_and_preserves_partial_reads(qt_app):
    rendered = []
    buffer = _PlaybackBuffer(30, read_tap=rendered.append)
    assert buffer.append(b"queued-but-not-played")
    assert rendered == []
    assert bytes(buffer.read(7)) == b"queued-"
    assert rendered == [b"queued-"]
    buffer.clear()
    assert rendered == [b"queued-"]
    buffer.close()


def test_duplex_processes_consumed_render_before_live_capture_and_preserves_onset(
    duplex_audio,
):
    audio = duplex_audio
    statuses, order, captured = [], [], []
    audio.duplex_status.connect(statuses.append)
    assert audio.configure_duplex("auto")["active"]
    processor = audio._duplex
    assert statuses[-1]["backend"] == "test-aec"
    audio._normalizer = PCMNormalizer(16000)
    normalizer = audio._normalizer
    frame = np.full(320, 2400, dtype="<i2").tobytes()
    assert audio.play_pcm(frame * 3, 16000)
    sink = audio._sink
    assert sink.buffer_size == 1280  # 40ms at16k, not a queued TTS duration.
    assert not audio._echo_blocked
    assert not processor.renders
    sink.source.read(640)
    assert not processor.renders  # Native callback only queues reference bytes.

    def onset():
        order.append("onset")
        audio.stop_playback()

    audio.speech_started.connect(onset)
    audio.pcm_ready.connect(lambda pcm: (order.append("pcm"), captured.append(pcm)))
    audio._input_io = SimpleNamespace(readAll=lambda: frame * 3)
    audio._read_capture()
    assert processor.order == ["render", "capture", "capture", "capture"]
    assert processor.renders[0][0] == frame
    assert isinstance(processor.renders[0][1], float)
    assert order == ["pcm", "onset", "pcm", "pcm"]
    assert captured == [frame] * 3
    assert audio._normalizer is normalizer
    assert processor.resets == 0
    assert not audio._echo_blocked
    assert audio._tail.isActive()


def test_native_stereo_float_reference_is_downmixed_resampled_and_bounded(
    duplex_audio,
):
    audio = duplex_audio
    audio.configure_duplex("auto")
    frame = np.full(960, 0.25, dtype="=f4")
    stereo = np.repeat(frame[:, None], 2, axis=1).tobytes()
    generation = audio._render_generation
    audio._tap_render(stereo[:37], (48000, 2, "float32"), generation, 16000)
    audio._tap_render(stereo[37:], (48000, 2, "float32"), generation, 16000)
    audio._drain_render()
    rendered = b"".join(pcm for pcm, _ in audio._duplex.renders)
    assert 319 <= len(rendered) // 2 <= 320
    assert np.all(np.frombuffer(rendered, dtype="<i2") == 8192)
    assert audio._render_bytes == 0
    assert not audio._render_chunks


def test_reference_overflow_falls_back_without_leaking_echo_or_stopping_output(
    duplex_audio,
):
    audio = duplex_audio
    audio.configure_duplex("auto")
    processor = audio._duplex
    frame = b"\x01\x10" * 320
    assert audio.play_pcm(frame, 16000)
    audio._normalizer = PCMNormalizer(16000)
    audio._input_io = SimpleNamespace(readAll=lambda: frame * 2)
    generation = audio._render_generation
    audio._tap_render(frame * 3, (16000, 1, "int16"), generation, 640)
    captured, statuses = [], []
    audio.pcm_ready.connect(captured.append)
    audio.duplex_status.connect(statuses.append)
    sink = audio._sink
    audio._read_capture()
    assert processor.closed
    assert not audio.listening_during_playback
    assert audio._echo_blocked
    assert audio._sink is sink and sink.stops == 0
    assert not captured
    assert not statuses[-1]["active"]
    assert "reference" in statuses[-1]["message"]


def test_native_aec_failure_is_visible_and_does_not_emit_unprocessed_echo(duplex_audio):
    audio = duplex_audio
    audio.configure_duplex("auto")
    frame = b"\x01\x10" * 320
    assert audio.play_pcm(frame, 16000)

    def failed(*args, **kwargs):
        raise RuntimeError("native echo processor failed")

    audio._duplex.process_capture = failed
    audio._normalizer = PCMNormalizer(16000)
    audio._input_io = SimpleNamespace(readAll=lambda: frame * 3)
    captured, statuses = [], []
    audio.pcm_ready.connect(captured.append)
    audio.duplex_status.connect(statuses.append)
    audio._read_capture()
    assert not captured
    assert audio._echo_blocked
    assert "native echo processor failed" in statuses[-1]["message"]


def test_mute_resets_onset_and_old_native_callback_cannot_repopulate_closed_audio(
    duplex_audio,
):
    audio = duplex_audio
    audio.configure_duplex("headphones")
    processor = audio._duplex
    generation = audio._render_generation
    audio.set_muted(True)
    assert processor.resets == 1
    audio.close()
    audio._tap_render(b"\0\0" * 320, (16000, 1, "int16"), generation, 640)
    assert not audio._render_chunks
    assert processor.closed


def test_idle_stop_does_not_invent_echo_tail_or_drop_the_next_microphone_prefix(
    duplex_audio,
):
    audio = duplex_audio
    audio.configure_duplex("auto")
    audio.stop_playback()
    assert not audio._tail.isActive()
    assert not audio._echo_blocked


def test_disabled_interruption_does_not_cancel_an_agent_while_microphone_is_idle(
    duplex_audio,
):
    audio = duplex_audio
    audio.configure_duplex("off")
    frame = b"\x01\x10" * 320
    audio._normalizer = PCMNormalizer(16000)
    audio._input_io = SimpleNamespace(readAll=lambda: frame * 2)
    onsets, captured = [], []
    audio.speech_started.connect(lambda: onsets.append(True))
    audio.pcm_ready.connect(captured.append)
    audio._read_capture()
    assert captured == [frame, frame]
    assert not onsets
