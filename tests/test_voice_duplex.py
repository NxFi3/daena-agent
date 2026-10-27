"""Barge-in onset, bounded render references, and native echo-control checks."""

import os
from pathlib import Path
import wave

import numpy as np
import pytest

from src.voice.duplex import DuplexProcessor, _Speex, _WebRTC


SILENCE = bytes(640)


def pcm(values):
    return np.clip(np.asarray(values) * 32768, -32768, 32767).astype("<i2").tobytes()


class IdentityBackend:
    name = "identity"

    def __init__(self):
        self.rendered = []
        self.closed = False

    def render(self, frame):
        self.rendered.append(frame)

    def process(self, frame):
        return frame

    def close(self):
        self.closed = True


def test_unavailable_aec_stays_half_duplex_without_silently_using_raw_microphone(
    monkeypatch,
):
    monkeypatch.setattr("src.voice.duplex._make_backend", lambda delay: None)
    processor = DuplexProcessor()
    assert processor.status["active"] is False
    assert "Half duplex" in processor.status["message"]
    assert processor.process_capture(b"\xff\x7f" * 320, playback_active=True) == (
        b"",
        False,
    )
    assert processor.process_capture(SILENCE) == (SILENCE, False)
    processor.close()


def test_headphones_onset_is_debounced_and_rearms_after_silence():
    processor = DuplexProcessor("headphones", vad=lambda frame: True)
    voice = pcm(np.ones(320) * 0.1)
    onsets = [
        processor.process_capture(voice, playback_active=True)[1] for _ in range(20)
    ]
    assert [i for i, value in enumerate(onsets) if value] == [5]
    for _ in range(15):
        assert not processor.process_capture(SILENCE, playback_active=True)[1]
    assert [processor.process_capture(voice)[1] for _ in range(6)] == [False] * 5 + [
        True
    ]
    processor.close()


def test_clicks_and_short_noise_do_not_interrupt():
    processor = DuplexProcessor("headphones", vad=lambda frame: True)
    click = pcm(np.ones(320) * 0.3)
    for frame in [click, SILENCE, click, SILENCE, click, SILENCE] * 4:
        assert not processor.process_capture(frame, playback_active=True)[1]
    processor.close()


def test_render_is_framed_and_history_is_bounded_even_for_large_native_read_burst():
    backend = IdentityBackend()
    processor = DuplexProcessor(backend=backend, vad=lambda frame: True)
    with pytest.raises(RuntimeError, match="500ms"):
        processor.feed_render(bytes(200000), timestamp=1)
    assert not processor._render
    processor.feed_render(bytes(16000), timestamp=1)
    assert len(processor._render) < 640
    assert len(processor._reference_history) <= 16640
    assert backend.rendered and all(len(frame) == 640 for frame in backend.rendered)
    processor.close()
    assert backend.closed


def test_stale_or_missing_render_reference_never_allows_speaker_input():
    processor = DuplexProcessor(backend=IdentityBackend(), vad=lambda frame: True)
    assert processor.process_capture(SILENCE, playback_active=True, timestamp=1) == (
        b"",
        False,
    )
    processor.feed_render(SILENCE, timestamp=1)
    assert processor.process_capture(SILENCE, playback_active=True, timestamp=1.1) == (
        SILENCE,
        False,
    )
    assert processor.process_capture(SILENCE, playback_active=True, timestamp=1.6) == (
        b"",
        False,
    )
    processor.close()


def test_coherent_delayed_echo_is_not_a_user_turn_even_before_native_aec_converges():
    rng = np.random.default_rng(42)
    reference = rng.normal(0, 0.1, 320 * 100)
    echoed = np.concatenate([np.zeros(1280), reference[:-1280] * 0.6])
    processor = DuplexProcessor(backend=IdentityBackend(), vad=lambda frame: True)
    output, onsets = [], []
    for i in range(100):
        processor.feed_render(
            pcm(reference[i * 320 : (i + 1) * 320]), timestamp=i * 0.02
        )
        cleaned, onset = processor.process_capture(
            pcm(echoed[i * 320 : (i + 1) * 320]),
            playback_active=True,
            timestamp=i * 0.02,
        )
        output.append(cleaned)
        onsets.append(onset)
    assert not any(onsets)
    assert b"".join(output[10:]) == bytes(640 * 90)
    processor.close()


def test_independent_near_end_speech_passes_while_playback_continues():
    rng = np.random.default_rng(43)
    reference = rng.normal(0, 0.04, 320 * 60)
    near = rng.normal(0, 0.16, len(reference))
    processor = DuplexProcessor(backend=IdentityBackend(), vad=lambda frame: True)
    onsets = []
    for i in range(60):
        processor.feed_render(
            pcm(reference[i * 320 : (i + 1) * 320]), timestamp=i * 0.02
        )
        _, onset = processor.process_capture(
            pcm(near[i * 320 : (i + 1) * 320]), playback_active=True, timestamp=i * 0.02
        )
        if onset:
            onsets.append(i)
    assert len(onsets) == 1 and 0.2 <= onsets[0] * 0.02 <= 0.32
    processor.close()


def test_invalid_frames_modes_and_closed_processor():
    with pytest.raises(ValueError):
        DuplexProcessor("guess")
    processor = DuplexProcessor("headphones")
    with pytest.raises(ValueError):
        processor.process_capture(bytes(320))
    processor.close()
    processor.close()
    assert not processor.status["active"]
    assert processor.process_capture(SILENCE) == (b"", False)


@pytest.mark.parametrize("backend_class", [_Speex, _WebRTC], ids=["speex", "webrtc"])
@pytest.mark.parametrize("burst_frames", [1, 4], ids=["20ms", "80ms_native_burst"])
def test_native_aec_delayed_echo_and_double_talk(backend_class, burst_frames):
    if os.environ.get("DAENA_TEST_VOICE_DUPLEX") != "1":
        pytest.skip("Enable DAENA_TEST_VOICE_DUPLEX=1 for actual native AEC")
    try:
        backend = backend_class(80)
    except (ImportError, OSError) as exc:
        pytest.skip(f"Optional native echo backend unavailable: {exc}")
    rng = np.random.default_rng(491)
    reference = rng.normal(0, 0.08, 16000 * 5)
    # Keep the delayed speaker tail after the native renderer has stopped.
    microphone = np.concatenate([np.zeros(1280), reference * 0.55, np.zeros(1920)])
    microphone[16000 * 3 : 16000 * 4] += rng.normal(0, 0.12, 16000)
    processor = DuplexProcessor(backend=backend, vad=lambda frame: True)
    output, onsets = [], []
    for i in range(len(microphone) // 320):
        if i * 320 < len(reference) and i % burst_frames == 0:
            processor.feed_render(
                pcm(reference[i * 320 : (i + burst_frames) * 320]),
                timestamp=i * 0.02,
            )
        cleaned, onset = processor.process_capture(
            pcm(microphone[i * 320 : (i + 1) * 320]),
            playback_active=True,
            timestamp=i * 0.02,
        )
        output.extend(np.frombuffer(cleaned, "<i2").astype(np.float32) / 32768)
        if onset:
            onsets.append(i * 0.02)
    processor.close()
    output = np.asarray(output)
    assert not any(onset < 3 for onset in onsets), onsets
    assert any(3 <= onset < 3.5 for onset in onsets), onsets
    assert np.sqrt(np.mean(output[16000:48000] ** 2)) < 0.002
    assert np.sqrt(np.mean(output[50000:60000] ** 2)) > 0.02
    assert np.sqrt(np.mean(output[80000:] ** 2)) < 0.002


def test_real_speech_delayed_echo_has_no_self_interruption_and_preserves_second_speaker():
    fixture_root = os.environ.get("DAENA_TEST_VOICE_MODELS")
    if not fixture_root or os.environ.get("DAENA_TEST_VOICE_DUPLEX") != "1":
        pytest.skip("Enable native duplex and real voice model fixtures")
    soxr = pytest.importorskip("soxr")

    def load(name):
        with wave.open(str(Path(fixture_root) / name)) as stream:
            values = (
                np.frombuffer(stream.readframes(stream.getnframes()), "<i2").astype(
                    np.float32
                )
                / 32768
            )
            if stream.getnchannels() > 1:
                values = values.reshape(-1, stream.getnchannels()).mean(axis=1)
            if stream.getframerate() != 16000:
                values = soxr.resample(values, stream.getframerate(), 16000)
        return np.tile(values, 5)[: 16000 * 12]

    reference, near = load("fangjun1.wav"), load("leijun.wav")
    microphone = np.concatenate([np.zeros(1280), reference[:-1280] * 0.55])
    microphone[96000:160000] += near[96000:160000] * 0.9
    processor = DuplexProcessor()
    assert processor.status["active"], processor.status
    onsets, near_pcm = [], []
    for i in range(len(reference) // 320):
        processor.feed_render(
            pcm(reference[i * 320 : (i + 1) * 320]), timestamp=i * 0.02
        )
        cleaned, onset = processor.process_capture(
            pcm(microphone[i * 320 : (i + 1) * 320]),
            playback_active=True,
            timestamp=i * 0.02,
        )
        if 6 <= i * 0.02 < 10:
            near_pcm.append(np.frombuffer(cleaned, "<i2"))
        if onset:
            onsets.append(i * 0.02)
    processor.close()
    assert onsets and all(6 <= onset < 10 for onset in onsets), onsets
    assert np.sqrt(np.mean(np.concatenate(near_pcm).astype(np.float64) ** 2)) > 300
