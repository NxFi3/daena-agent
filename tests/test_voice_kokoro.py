"""Offline Kokoro contracts and bilingual routing without downloaded weights."""

from threading import Event

import numpy as np
import pytest

from src.voice.kokoro import BilingualSynthesizer, KokoroProvider
from src.voice.providers import AudioChunk, VoiceCancelled, VoiceError


def files(tmp_path):
    model = tmp_path / "kokoro.onnx"
    voices = tmp_path / "voices.bin"
    model.write_bytes(b"scripted model")
    voices.write_bytes(b"scripted voices")
    return model, voices


class Engine:
    def __init__(self):
        self.spoken = []
        self.closed = 0

    def get_voices(self):
        return ["af_heart", "bm_george"]

    async def create_stream(self, text, **kwargs):
        self.spoken.append((text, kwargs))
        try:
            yield np.array([-2, -1, 0, 0.5, 1, 2], dtype=np.float32), 24000
            yield np.zeros(24000, dtype=np.float32), 24000
        finally:
            self.closed += 1


def provider(tmp_path, engine=None, **kwargs):
    engine = engine or Engine()
    loaded = []

    def factory(model, voices, **opts):
        loaded.append((model, voices, opts))
        return engine

    return (
        KokoroProvider(*files(tmp_path), engine_factory=factory, **kwargs),
        engine,
        loaded,
    )


def test_kokoro_lazy_cached_stream_clips_pcm_and_bounds_cpu_and_chunks(tmp_path):
    voice, engine, loaded = provider(
        tmp_path, voice="bm_george", speed=0.9, cpu_threads=2
    )
    assert loaded == []
    voice.prepare()
    chunks = list(voice.synthesize("Hello, world.", "en"))
    assert len(loaded) == 1
    assert loaded[0][2] == {"cpu_threads": 2}
    assert engine.spoken == [
        ("Hello, world.", {"voice": "bm_george", "speed": 0.9, "lang": "en-gb"})
    ]
    assert all(
        isinstance(chunk, AudioChunk) and chunk.sample_rate == 24000 for chunk in chunks
    )
    assert all(len(chunk.pcm16) <= 4800 for chunk in chunks)
    assert np.frombuffer(chunks[0].pcm16, dtype="<i2").tolist() == [
        -32768,
        -32768,
        0,
        16384,
        32767,
        32767,
    ]
    assert engine.closed == 1
    list(voice.synthesize("Again."))
    assert len(loaded) == 1


def test_kokoro_missing_files_and_voice_give_actionable_errors(tmp_path):
    voice = KokoroProvider(
        tmp_path / "missing",
        tmp_path / "missing",
        engine_factory=lambda *a, **k: pytest.fail("loaded"),
    )
    with pytest.raises(VoiceError, match="Select/download"):
        voice.prepare()
    voice, engine, _ = provider(tmp_path, voice="af_missing")
    with pytest.raises(VoiceError, match="absent"):
        voice.prepare()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"voice": "fa_unsupported"},
        {"speed": 0.4},
        {"speed": float("nan")},
        {"speed": float("inf")},
        {"cpu_threads": 0},
        {"cpu_threads": 1.5},
        {"cpu_threads": True},
    ],
)
def test_kokoro_rejects_unsupported_settings(kwargs):
    with pytest.raises(VoiceError):
        KokoroProvider("model", "voices", **kwargs)


def test_kokoro_cancellation_before_load_and_between_pcm_frames(tmp_path):
    voice, engine, loaded = provider(tmp_path)
    stop = Event()
    stop.set()
    with pytest.raises(VoiceCancelled):
        voice.prepare(stop)
    assert loaded == []
    stop.clear()
    stream = voice.synthesize("Hello", stop_event=stop)
    next(stream)
    next(stream)
    stop.set()
    with pytest.raises(VoiceCancelled):
        next(stream)
    assert engine.closed == 1


def test_kokoro_close_cleans_async_generator_and_propagates_native_error(tmp_path):
    voice, engine, _ = provider(tmp_path)
    stream = voice.synthesize("Hello")
    next(stream)
    stream.close()
    assert engine.closed == 1

    class FailedEngine(Engine):
        async def create_stream(self, text, **kwargs):
            raise RuntimeError("native inference error")
            yield

    voice, _, _ = provider(tmp_path, FailedEngine())
    with pytest.raises(VoiceError, match="native inference error"):
        list(voice.synthesize("Hello"))


def test_kokoro_rejects_persian_without_running_english_model(tmp_path):
    voice, engine, loaded = provider(tmp_path)
    for text, language in [("سلام", "fa"), ("hello سلام", "auto")]:
        with pytest.raises(VoiceError, match="Persian"):
            list(voice.synthesize(text, language))
    assert loaded == []


@pytest.mark.parametrize(
    "samples",
    [
        np.array([np.nan], dtype=np.float32),
        np.zeros((2, 2)),
        np.array([1], dtype=np.int16),
    ],
)
def test_kokoro_rejects_invalid_native_pcm(tmp_path, samples):
    class InvalidEngine(Engine):
        async def create_stream(self, text, **kwargs):
            yield samples, 24000

    voice, _, _ = provider(tmp_path, InvalidEngine())
    with pytest.raises(VoiceError, match="invalid mono"):
        list(voice.synthesize("Hello"))


def test_kokoro_half_precision_samples_do_not_wrap_positive_peak(tmp_path):
    class HalfEngine(Engine):
        async def create_stream(self, text, **kwargs):
            yield np.array([-1, 0, 1], dtype=np.float16), 24000

    voice, _, _ = provider(tmp_path, HalfEngine())
    chunks = list(voice.synthesize("Hello"))
    assert np.frombuffer(chunks[0].pcm16, dtype="<i2").tolist() == [-32768, 0, 32767]


def test_bilingual_router_preserves_language_order_and_neutral_tokens():
    spoken, prepared = [], []

    class Voice:
        def __init__(self, name):
            self.name = name

        def prepare(self, stop_event=None):
            prepared.append(self.name)

        def synthesize(self, text, language, stop_event=None):
            spoken.append((self.name, language, text))
            yield AudioChunk(24000 if language == "en" else 22050, b"\x01\x00")

    router = BilingualSynthesizer(Voice("kokoro"), Voice("piper"))
    router.prepare()
    chunks = list(router.synthesize("سلام Python 3 خوب است."))
    assert prepared == ["piper", "kokoro"]
    assert spoken == [
        ("piper", "fa", "سلام"),
        ("kokoro", "en", "Python 3"),
        ("piper", "fa", "خوب است."),
    ]
    assert [chunk.sample_rate for chunk in chunks] == [22050, 24000, 22050]


def test_bilingual_router_missing_language_is_explicit_and_single_language_does_not_warm_other():
    prepared = []

    class Voice:
        def prepare(self, stop_event=None):
            prepared.append("en")

        def synthesize(self, text, language, stop_event=None):
            yield AudioChunk(24000, b"\0\0")

    router = BilingualSynthesizer(english=Voice(), language="en")
    router.prepare()
    assert prepared == ["en"]
    assert list(router.synthesize("Hello"))
    with pytest.raises(VoiceError, match="local fa voice"):
        list(router.synthesize("سلام", "fa"))
    with pytest.raises(VoiceError, match="local fa voice"):
        BilingualSynthesizer(english=Voice()).prepare()
