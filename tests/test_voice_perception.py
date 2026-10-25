"""Acoustic evidence comes from waveform models, with bounded volatile state."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from src.voice.perception import LocalAudioPerception
from src.voice.providers import VoiceCancelled, VoiceError


def waveform(seconds=2.0):
    samples = np.sin(np.arange(int(16000 * seconds)) * 0.17) * 12000
    return samples.astype("<i2").tobytes()


def vector(axis=0, secondary=None):
    result = np.zeros(16, dtype=np.float32)
    result[axis] = 1
    if secondary is not None:
        result[1] = secondary
    return result


class Stream:
    def __init__(self):
        self.finished = False
        self.waveforms = []

    def accept_waveform(self, *, sample_rate, waveform):
        self.waveforms.append((sample_rate, waveform))

    def input_finished(self):
        self.finished = True


class Speaker:
    def __init__(self, values=None):
        self.values = values or [vector()]
        self.streams = []
        self.before_compute = None

    def create_stream(self):
        result = Stream()
        self.streams.append(result)
        return result

    def is_ready(self, stream):
        assert stream.finished
        return True

    def compute(self, stream):
        if self.before_compute:
            self.before_compute()
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


class Tagger:
    def __init__(self, events=None):
        self.events = events or [SimpleNamespace(name="Doorbell", prob=0.85)]
        self.streams = []
        self.before_compute = None

    def create_stream(self):
        result = Stream()
        self.streams.append(result)
        return result

    def compute(self, stream):
        if self.before_compute:
            self.before_compute()
        return self.events


def models(tmp_path, *, speaker=None, tagger=None, **kwargs):
    speaker = speaker or Speaker()
    tagger = tagger or Tagger()
    paths = {}
    for key in ("speaker_model_path", "sound_model_path", "sound_labels_path"):
        path = tmp_path / key
        path.write_bytes(b"local fixture")
        paths[key] = path
    calls = []

    def speaker_factory(path, **options):
        calls.append(("speaker", path, options))
        return speaker

    def sound_factory(path, labels, **options):
        calls.append(("sound", path, labels, options))
        return tagger

    perception = LocalAudioPerception(
        **paths,
        speaker_factory=speaker_factory,
        sound_factory=sound_factory,
        **kwargs,
    )
    return perception, speaker, tagger, calls


def test_native_adapters_are_lazy_cached_and_waveform_based(tmp_path):
    perception, speaker, tagger, calls = models(tmp_path)
    assert calls == []
    perception.prepare()
    result = perception.analyze(waveform())
    perception.prepare()
    assert [call[0] for call in calls] == ["speaker", "sound"]
    assert calls[0][2] == {"num_threads": 2}
    assert calls[1][3] == {"model_type": "zipformer", "num_threads": 2}
    assert result["speaker"]["status"] == "new"
    assert result["speaker"]["id"] == "speaker_1"
    assert result["sounds"] == {
        "status": "known",
        "events": [{"label": "Doorbell", "score": 0.85}],
    }
    assert speaker.streams[0].waveforms[0][0] == 16000
    assert speaker.streams[0].waveforms[0][1].dtype == np.float32
    assert np.max(np.abs(tagger.streams[0].waveforms[0][1])) < 1
    json.dumps(result, allow_nan=False)


def test_anonymous_session_clusters_repeated_and_new_voices(tmp_path):
    speaker = Speaker([vector(0), vector(0), vector(1), vector(1), vector(0)])
    perception, *_ = models(tmp_path, speaker=speaker)
    evidence = [perception.analyze(waveform())["speaker"] for _ in range(5)]
    assert [(row["id"], row["status"]) for row in evidence] == [
        ("speaker_1", "new"),
        ("speaker_1", "matched"),
        ("speaker_2", "new"),
        ("speaker_2", "matched"),
        ("speaker_1", "matched"),
    ]
    assert evidence[0]["similarity"] is None
    assert evidence[1]["similarity"] == 1.0
    perception.reset()
    assert perception.analyze(waveform())["speaker"]["status"] == "new"


def test_ambiguous_embedding_does_not_force_match_or_enroll(tmp_path):
    ambiguous = np.zeros(16, dtype=np.float32)
    ambiguous[:2] = [0.7, np.sqrt(1 - 0.7**2)]
    perception, *_ = models(
        tmp_path,
        speaker=Speaker([vector(0), ambiguous, vector(0)]),
        speaker_threshold=0.72,
    )
    assert perception.analyze(waveform())["speaker"]["id"] == "speaker_1"
    assert perception.analyze(waveform())["speaker"]["status"] == "uncertain"
    assert len(perception._centroids) == 1
    assert perception.analyze(waveform())["speaker"]["id"] == "speaker_1"


def test_close_two_matches_are_uncertain_without_mutating_clusters(tmp_path):
    midway = vector(0) + vector(1)
    perception, *_ = models(
        tmp_path,
        speaker=Speaker([vector(0), vector(1), midway]),
        speaker_threshold=0.65,
    )
    perception.analyze(waveform())
    perception.analyze(waveform())
    counts = list(perception._observations)
    result = perception.analyze(waveform())
    assert result["speaker"]["id"] is None
    assert result["speaker"]["status"] == "uncertain"
    assert perception._observations == counts


def test_speaker_capacity_is_bounded_and_never_relabels_existing_people(tmp_path):
    perception, *_ = models(
        tmp_path, max_speakers=1, speaker=Speaker([vector(0), vector(1)])
    )
    perception.analyze(waveform())
    result = perception.analyze(waveform())
    assert result["speaker"]["id"] is None
    assert result["speaker"]["status"] == "uncertain"
    assert len(perception._centroids) == 1


def test_ambient_sounds_do_not_create_speaker_embeddings(tmp_path):
    perception, speaker, tagger, calls = models(tmp_path)
    result = perception.analyze(waveform(), speech=False)
    assert result["speaker"]["status"] == "not_speech"
    assert not speaker.streams
    assert tagger.streams
    assert [call[0] for call in calls] == ["sound"]
    assert not perception._centroids


def test_low_sound_scores_are_unknown_and_bad_scores_excluded(tmp_path):
    tagger = Tagger(
        [
            SimpleNamespace(name="Speech", prob=0.04),
            SimpleNamespace(name="Siren", prob=float("nan")),
            SimpleNamespace(name="Music", prob=0.12),
            SimpleNamespace(name="Bad", prob=3),
        ]
    )
    perception, *_ = models(tmp_path, tagger=tagger)
    result = perception.analyze(waveform(), speech=False)
    assert result["sounds"] == {
        "status": "unknown",
        "events": [
            {"label": "Music", "score": 0.12},
            {"label": "Speech", "score": 0.04},
        ],
    }


def test_silence_and_short_speech_do_not_enroll_speakers(tmp_path):
    perception, speaker, tagger, calls = models(tmp_path)
    result = perception.analyze(b"\0\0" * 32000)
    assert result["sounds"]["status"] == "silence"
    assert result["speaker"]["status"] == "silence"
    assert calls == []
    result = perception.analyze(waveform(0.6))
    assert result["speaker"]["status"] == "too_short"
    assert not speaker.streams
    assert tagger.streams


def test_large_waveform_only_analyzes_bounded_tail_and_reports_clip(tmp_path):
    perception, speaker, tagger, _ = models(tmp_path, max_audio_seconds=2)
    result = perception.analyze(waveform(4))
    assert result["duration_ms"] == 4000
    assert result["sampled_ms"] == 2000
    assert result["truncated"] is True
    assert len(speaker.streams[0].waveforms[0][1]) == 32000
    clipped = perception.analyze(np.full(32000, 32767, dtype="<i2").tobytes())
    assert clipped["signal"]["clipped_fraction"] == 1
    assert clipped["signal"]["rms_dbfs"] == 0


def test_cancel_during_native_sound_inference_never_enrolls_speaker(tmp_path):
    stop = Event()
    tagger = Tagger()
    tagger.before_compute = stop.set
    perception, speaker, *_ = models(tmp_path, tagger=tagger)
    with pytest.raises(VoiceCancelled):
        perception.analyze(waveform(), stop_event=stop)
    assert not speaker.streams
    assert not perception._centroids


def test_cancel_during_embedding_never_publishes_new_cluster(tmp_path):
    stop = Event()
    speaker = Speaker()
    speaker.before_compute = stop.set
    perception, *_ = models(tmp_path, speaker=speaker)
    with pytest.raises(VoiceCancelled):
        perception.analyze(waveform(), stop_event=stop)
    assert not perception._centroids


def test_reset_during_inference_discards_result_without_reenrolling(tmp_path):
    speaker = Speaker()
    perception, *_ = models(tmp_path, speaker=speaker)
    speaker.before_compute = perception.reset
    with pytest.raises(VoiceCancelled, match="reset"):
        perception.analyze(waveform())
    assert not perception._centroids


@pytest.mark.parametrize("embedding", [[float("nan")] * 16, [1, 2], [[1] * 16]])
def test_invalid_embedding_is_rejected_without_session_pollution(tmp_path, embedding):
    perception, *_ = models(tmp_path, speaker=Speaker([embedding]))
    with pytest.raises(VoiceError, match="embedding"):
        perception.analyze(waveform())
    assert not perception._centroids


def test_pre_cancel_and_missing_local_model_never_initialize(tmp_path):
    stop = Event()
    stop.set()
    perception, *_ = models(tmp_path)
    with pytest.raises(VoiceCancelled):
        perception.prepare(stop)
    with pytest.raises(VoiceError, match="local"):
        LocalAudioPerception(speaker_model_path=tmp_path / "missing.onnx").prepare()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sound_model_path": "sound.onnx"},
        {"sound_model_type": "imagined"},
        {"num_threads": 0},
        {"speaker_threshold": 1},
        {"sound_threshold": float("nan")},
        {"max_audio_seconds": 0.01},
    ],
)
def test_invalid_config_is_actionable(kwargs):
    with pytest.raises(VoiceError):
        LocalAudioPerception(**kwargs)


@pytest.mark.parametrize("pcm,rate", [("no", 16000), (b"\0", 16000), (b"\0\0", 44100)])
def test_input_contract_rejects_malformed_audio(pcm, rate):
    with pytest.raises(VoiceError):
        LocalAudioPerception().analyze(pcm, sample_rate=rate)


def test_no_configured_models_is_valid_json_evidence():
    result = LocalAudioPerception().analyze(waveform())
    assert result["speaker"]["status"] == "disabled"
    assert result["sounds"]["status"] == "disabled"
    assert "text" not in result
    json.dumps(result, allow_nan=False)


def test_default_is_conservative_about_impostor_similarities(tmp_path):
    # A real cross-person example produced cosine .8265. Similarity is not an
    # identity probability; the default must not turn this into a positive match.
    other = np.zeros(16, dtype=np.float32)
    other[:2] = [0.8265, np.sqrt(1 - 0.8265**2)]
    perception, *_ = models(tmp_path, speaker=Speaker([vector(0), other]))
    first = perception.analyze(waveform())["speaker"]
    second = perception.analyze(waveform())["speaker"]
    assert first["id"] == "speaker_1"
    assert second["status"] != "matched"
    assert second["id"] != first["id"]


def test_real_local_acoustic_models_when_explicitly_available():
    """Opt-in downloaded-model smoke; default test runs never download weights."""
    import os
    import wave

    directory = os.environ.get("DAENA_TEST_VOICE_MODELS")
    if not directory:
        pytest.skip(
            "Set DAENA_TEST_VOICE_MODELS to explicitly provisioned local fixtures"
        )
    root = Path(directory)
    required = [
        "speaker.onnx",
        "model.int8.onnx",
        "class_labels_indices.csv",
        "fangjun1.wav",
        "4.wav",
        "5.wav",
    ]
    assert all((root / name).is_file() for name in required), (
        "Incomplete local acoustic fixtures"
    )
    import soxr

    def read(name):
        with wave.open(str(root / name)) as wav:
            assert wav.getsampwidth() == 2
            rate, channels = wav.getframerate(), wav.getnchannels()
            samples = (
                np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(
                    np.float32
                )
                / 32768
            )
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        if rate != 16000:
            samples = soxr.resample(samples, rate, 16000)
        return np.clip(np.rint(samples * 32768), -32768, 32767).astype("<i2").tobytes()

    perception = LocalAudioPerception(
        speaker_model_path=root / "speaker.onnx",
        sound_model_path=root / "model.int8.onnx",
        sound_labels_path=root / "class_labels_indices.csv",
    )
    perception.prepare()
    first = perception.analyze(read("fangjun1.wav"))
    second = perception.analyze(read("fangjun1.wav"))
    assert first["speaker"]["status"] == "new"
    assert second["speaker"]["status"] == "matched"
    assert second["speaker"]["similarity"] > 0.99
    for evidence in (first, second):
        labels = [event["label"] for event in evidence["sounds"]["events"]]
        assert not any(
            word in label
            for label in labels
            for word in ("Male", "Female", "Child", "Baby")
        )
    for name, expected in [("4.wav", "Laughter"), ("5.wav", "Finger snapping")]:
        evidence = perception.analyze(read(name), speech=False)
        assert evidence["sampled_ms"] <= 8000
        assert evidence["speaker"]["status"] == "not_speech"
        assert any(event["label"] == expected for event in evidence["sounds"]["events"])
        json.dumps(evidence, allow_nan=False)


def test_sound_taxonomy_projects_demographic_classes_by_stable_mid_and_deduplicates(
    tmp_path,
):
    # Renamed display text proves the projection is bound to the model's stable
    # AudioSet class ID, rather than English word matching or transcript rules.
    tagger = Tagger(
        [
            SimpleNamespace(index=101, name="custom voice label one", prob=0.6),
            SimpleNamespace(index=102, name="custom voice label two", prob=0.9),
            SimpleNamespace(index=103, name="Speech", prob=0.7),
            SimpleNamespace(index=104, name="Finger snapping", prob=0.8),
        ]
    )
    perception, *_ = models(tmp_path, tagger=tagger)
    Path(perception.sound_labels_path).write_text(
        'index,mid,display_name\n101,/m/05zppz,"Male speech, man speaking"\n'
        '102,/m/02zsn,"Female speech, woman speaking"\n'
        "103,/m/09x0r,Speech\n104,/m/025_jnm,Finger snapping\n"
    )
    result = perception.analyze(waveform(), speech=False)
    assert result["sounds"]["events"] == [
        {"label": "Speech", "score": 0.9},
        {"label": "Finger snapping", "score": 0.8},
    ]
    assert "custom voice label" not in json.dumps(result)
    assert "Male" not in json.dumps(result)
    assert "Female" not in json.dumps(result)


@pytest.mark.parametrize(
    "source,parent",
    [
        ("Male speech, man speaking", "Speech"),
        ("Female speech, woman speaking", "Speech"),
        ("Child speech, kid speaking", "Speech"),
        ("Children shouting", "Shout"),
        ("Baby laughter", "Laughter"),
        ("Baby cry, infant cry", "Crying, sobbing"),
        ("Male singing", "Singing"),
        ("Female singing", "Singing"),
        ("Child singing", "Singing"),
        ("Children playing", "Human group actions"),
    ],
)
def test_official_class_name_fallback_projects_without_demographic_claims(
    tmp_path, source, parent
):
    perception, *_ = models(
        tmp_path, tagger=Tagger([SimpleNamespace(name=source, prob=0.75)])
    )
    evidence = perception.analyze(waveform(), speech=False)
    assert evidence["sounds"]["events"] == [{"label": parent, "score": 0.75}]
