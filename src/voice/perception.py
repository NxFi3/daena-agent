"""Optional, offline acoustic evidence from waveform models, never transcripts.

The speaker labels are volatile anonymous clusters, not real identities or an
access-control mechanism. AudioSet labels describe a fixed set of sound classes;
``unknown`` means insufficient model score, not verified open-world novelty.
Age/gender-specific AudioSet vocal labels are projected to their sound-category
parents before publication; duplicate parent scores use their maximum, not sum.
"""

from __future__ import annotations

import math
import csv
from pathlib import Path
from threading import RLock

from .providers import VoiceCancelled, VoiceError, _check_stop


# Projection to parent sound classes in the official AudioSet ontology:
# https://github.com/audioset/ontology/blob/master/ontology.json
# These are class identifiers, not user-phrase rules or inferred demographics.
_AUDIOSET_PARENT_LABELS = {
    "/m/05zppz": "Speech",  # Male speech
    "/m/02zsn": "Speech",  # Female speech
    "/m/0ytgt": "Speech",  # Child speech
    "/t/dd00135": "Shout",  # Children shouting
    "/t/dd00001": "Laughter",  # Baby laughter
    "/t/dd00002": "Crying, sobbing",  # Baby cry
    "/t/dd00003": "Singing",  # Male singing
    "/t/dd00004": "Singing",  # Female singing
    "/t/dd00005": "Singing",  # Child singing
    "/t/dd00013": "Human group actions",  # Children playing
}
_AUDIOSET_PARENT_NAMES = {
    "Male speech, man speaking": "Speech",
    "Female speech, woman speaking": "Speech",
    "Child speech, kid speaking": "Speech",
    "Children shouting": "Shout",
    "Baby laughter": "Laughter",
    "Baby cry, infant cry": "Crying, sobbing",
    "Male singing": "Singing",
    "Female singing": "Singing",
    "Child singing": "Singing",
    "Children playing": "Human group actions",
}


def _local_file(value, description):
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_file():
        raise VoiceError(f"Select/download the local {description}: {path}")
    return path


def _sherpa():
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise VoiceError(
            "Install local audio perception: python -m pip install 'daena-agent[voice-perception]'"
        ) from exc
    return sherpa_onnx


class LocalAudioPerception:
    """Bounded CPU acoustic analysis with lazy local ONNX models.

    ``speaker_factory(path, num_threads=...)`` and
    ``sound_factory(path, labels_path, model_type=..., num_threads=...)`` are
    dependency-injection hooks returning sherpa-compatible extractors/taggers.
    Loading and inference belong on a background worker. Native ONNX calls cannot
    be force-killed, but cancelled or reset operations never publish a cluster.
    """

    def __init__(
        self,
        speaker_model_path="",
        sound_model_path="",
        sound_labels_path="",
        *,
        sound_model_type="zipformer",
        speaker_threshold=0.88,
        sound_threshold=0.25,
        speaker_margin=0.04,
        min_speaker_seconds=1.2,
        max_audio_seconds=8.0,
        max_speakers=8,
        num_threads=2,
        top_k=5,
        speaker_factory=None,
        sound_factory=None,
    ):
        if sound_model_type not in {"zipformer", "ced"}:
            raise VoiceError("Audio tagging model type must be zipformer or ced")
        if not 0.1 <= float(speaker_threshold) <= 0.99:
            raise VoiceError(
                "Speaker similarity threshold must be between 0.1 and 0.99"
            )
        if not 0 <= float(sound_threshold) <= 1:
            raise VoiceError("Sound score threshold must be between 0 and 1")
        if not 0 <= float(speaker_margin) <= 0.3:
            raise VoiceError("Speaker ambiguity margin must be between 0 and 0.3")
        if not 0.5 <= float(min_speaker_seconds) <= float(max_audio_seconds) <= 30:
            raise VoiceError(
                "Acoustic analysis duration must be between 0.5 and 30 seconds"
            )
        if not 1 <= int(num_threads) <= 8 or not 1 <= int(max_speakers) <= 32:
            raise VoiceError(
                "Audio perception needs 1–8 threads and 1–32 speaker clusters"
            )
        if not 1 <= int(top_k) <= 10:
            raise VoiceError("Audio tagging top_k must be between 1 and 10")
        if bool(sound_model_path) != bool(sound_labels_path):
            raise VoiceError("Select both the sound ONNX model and its labels CSV")
        self.speaker_model_path = str(speaker_model_path) if speaker_model_path else ""
        self.sound_model_path = str(sound_model_path) if sound_model_path else ""
        self.sound_labels_path = str(sound_labels_path) if sound_labels_path else ""
        self.sound_model_type = sound_model_type
        self.speaker_threshold = float(speaker_threshold)
        self.sound_threshold = float(sound_threshold)
        self.speaker_margin = float(speaker_margin)
        self.min_speaker_seconds = float(min_speaker_seconds)
        self.max_audio_seconds = float(max_audio_seconds)
        self.max_speakers = int(max_speakers)
        self.num_threads = int(num_threads)
        self.top_k = int(top_k)
        self._speaker_factory = speaker_factory
        self._sound_factory = sound_factory
        self._speaker = None
        self._sound = None
        self._label_mids = {}
        self._native_lock = RLock()
        self._session_lock = RLock()
        self._generation = 0
        self._centroids = []
        self._observations = []

    def reset(self):
        """Forget all speaker embeddings without waiting on native inference."""
        with self._session_lock:
            self._generation += 1
            self._centroids.clear()
            self._observations.clear()

    def _load_speaker(self):
        if self._speaker is not None or not self.speaker_model_path:
            return self._speaker
        path = _local_file(self.speaker_model_path, "speaker embedding ONNX model")
        try:
            if self._speaker_factory is not None:
                model = self._speaker_factory(str(path), num_threads=self.num_threads)
            else:
                sherpa = _sherpa()
                config = sherpa.SpeakerEmbeddingExtractorConfig(
                    model=str(path),
                    num_threads=self.num_threads,
                    provider="cpu",
                    debug=False,
                )
                if not config.validate():
                    raise VoiceError(
                        "Invalid local speaker embedding model/configuration"
                    )
                model = sherpa.SpeakerEmbeddingExtractor(config)
            self._speaker = model
        except VoiceError:
            raise
        except Exception as exc:
            raise VoiceError(
                f"Cannot load local speaker embedding model: {exc}"
            ) from exc
        return self._speaker

    def _load_sound(self):
        if self._sound is not None or not self.sound_model_path:
            return self._sound
        path = _local_file(self.sound_model_path, "audio tagging ONNX model")
        labels = _local_file(self.sound_labels_path, "audio tagging labels CSV")
        try:
            # Bind model indices to stable taxonomy IDs. Exact official names are
            # a fallback for compatible adapters that do not expose an index.
            with labels.open(encoding="utf-8-sig", newline="") as stream:
                for count, row in enumerate(csv.DictReader(stream)):
                    if count >= 10000:
                        raise VoiceError(
                            "Audio tagging labels CSV is unexpectedly large"
                        )
                    if row.get("index", "").strip().isdigit() and row.get("mid"):
                        self._label_mids[int(row["index"])] = row["mid"].strip()
            if self._sound_factory is not None:
                model = self._sound_factory(
                    str(path),
                    str(labels),
                    model_type=self.sound_model_type,
                    num_threads=self.num_threads,
                )
            else:
                sherpa = _sherpa()
                model_args = dict(
                    num_threads=self.num_threads, provider="cpu", debug=False
                )
                if self.sound_model_type == "ced":
                    model_args["ced"] = str(path)
                else:
                    model_args["zipformer"] = (
                        sherpa.OfflineZipformerAudioTaggingModelConfig(model=str(path))
                    )
                config = sherpa.AudioTaggingConfig(
                    model=sherpa.AudioTaggingModelConfig(**model_args),
                    labels=str(labels),
                    top_k=self.top_k,
                )
                if not config.validate():
                    raise VoiceError("Invalid local audio tagging model/configuration")
                model = sherpa.AudioTagging(config)
            self._sound = model
        except VoiceError:
            raise
        except Exception as exc:
            raise VoiceError(f"Cannot load local audio tagging model: {exc}") from exc
        return self._sound

    def _sound_label(self, event):
        """Expose sound categories without age/gender-specific vocal labels."""
        label = str(event.name).strip()
        mid = self._label_mids.get(getattr(event, "index", None))
        return _AUDIOSET_PARENT_LABELS.get(
            mid, _AUDIOSET_PARENT_NAMES.get(label, label)
        )

    def prepare(self, stop_event=None):
        _check_stop(stop_event)
        with self._native_lock:
            _check_stop(stop_event)
            self._load_speaker()
            _check_stop(stop_event)
            self._load_sound()
        _check_stop(stop_event)

    @staticmethod
    def _speaker_result(status, *, speaker_id=None, similarity=None):
        return {
            "id": speaker_id,
            "status": status,
            "similarity": round(float(similarity), 4)
            if similarity is not None
            else None,
            "basis": "session_embedding",
        }

    def _cluster(self, embedding, generation):
        import numpy as np

        vector = np.asarray(embedding, dtype=np.float32)
        if (
            vector.ndim != 1
            or not 16 <= vector.size <= 4096
            or not np.isfinite(vector).all()
        ):
            raise VoiceError("Speaker model returned an invalid embedding")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm < 1e-8:
            return self._speaker_result("uncertain")
        vector = vector / norm
        with self._session_lock:
            if generation != self._generation:
                raise VoiceCancelled("Audio perception session was reset")
            if self._centroids and vector.shape != self._centroids[0].shape:
                raise VoiceError("Speaker model embedding dimensions changed")
            scores = [float(np.dot(vector, center)) for center in self._centroids]
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            best = scores[order[0]] if order else None
            # Similarity is a cosine score, never a calibrated identity probability.
            if order and best >= self.speaker_threshold:
                if len(order) > 1 and best - scores[order[1]] < self.speaker_margin:
                    return self._speaker_result("uncertain", similarity=best)
                index = order[0]
                self._observations[index] += 1
                weight = min(0.2, 1.0 / self._observations[index])
                center = self._centroids[index] * (1 - weight) + vector * weight
                self._centroids[index] = center / np.linalg.norm(center)
                return self._speaker_result(
                    "matched", speaker_id=f"speaker_{index + 1}", similarity=best
                )
            if order and best > self.speaker_threshold - self.speaker_margin:
                return self._speaker_result("uncertain", similarity=best)
            if len(self._centroids) >= self.max_speakers:
                return self._speaker_result("uncertain", similarity=best)
            self._centroids.append(vector)
            self._observations.append(1)
            return self._speaker_result(
                "new", speaker_id=f"speaker_{len(self._centroids)}", similarity=best
            )

    def analyze(self, pcm16, sample_rate=16000, stop_event=None, *, speech=True):
        """Return JSON-safe measured evidence from a bounded waveform window.

        ``speech=True`` is appropriate for a VAD speech utterance with one dominant
        speaker. For ambient/non-speech windows pass ``speech=False``; those never
        enroll a speaker. This is not overlapping-speaker diarization.
        """
        _check_stop(stop_event)
        if sample_rate != 16000:
            raise VoiceError("Audio perception expects 16000 Hz mono PCM16")
        if not isinstance(pcm16, bytes) or len(pcm16) % 2:
            raise VoiceError(
                "Audio perception input must contain complete PCM16 samples"
            )
        if not isinstance(speech, bool):
            raise VoiceError("Audio perception speech flag must be a boolean")
        import numpy as np

        original_samples = len(pcm16) // 2
        cap = int(self.max_audio_seconds * sample_rate)
        pcm16 = pcm16[-cap * 2 :]
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        with self._session_lock:
            generation = self._generation
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        result = {
            "duration_ms": round(original_samples * 1000 / sample_rate, 1),
            "sampled_ms": round(samples.size * 1000 / sample_rate, 1),
            "truncated": original_samples > cap,
            "speaker": self._speaker_result(
                "disabled" if not self.speaker_model_path else "not_speech"
            ),
            "sounds": {"status": "disabled", "events": []},
            "signal": {
                "rms_dbfs": round(20 * math.log10(max(rms, 1e-10)), 1),
                "clipped_fraction": round(float(np.mean(np.abs(samples) >= 0.999)), 5)
                if samples.size
                else 0.0,
            },
        }
        # This is only a silence optimization, never a speaker classifier.
        if rms < 1e-4:
            if self.speaker_model_path:
                result["speaker"] = self._speaker_result("silence")
            if self.sound_model_path:
                result["sounds"]["status"] = "silence"
            return result
        with self._native_lock:
            _check_stop(stop_event)
            sound = self._load_sound()
            if sound is not None:
                try:
                    stream = sound.create_stream()
                    stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
                    events = sound.compute(stream)
                    _check_stop(stop_event)
                    by_label = {}
                    for event in events:
                        label = self._sound_label(event)
                        score = float(event.prob)
                        if label and math.isfinite(score) and 0 <= score <= 1:
                            # Several children may share a parent. Keep the max
                            # model score; adding dependent scores overstates it.
                            label = label[:160]
                            by_label[label] = max(by_label.get(label, 0), score)
                    scores = [
                        {"label": label, "score": round(score, 4)}
                        for label, score in by_label.items()
                    ]
                    scores.sort(key=lambda row: row["score"], reverse=True)
                    scores = scores[: self.top_k]
                    result["sounds"] = {
                        "status": "known"
                        if scores and scores[0]["score"] >= self.sound_threshold
                        else "unknown",
                        "events": scores,
                    }
                except VoiceCancelled:
                    raise
                except Exception as exc:
                    raise VoiceError(f"Local audio tagging failed: {exc}") from exc
            if self.speaker_model_path and speech:
                if samples.size < self.min_speaker_seconds * sample_rate:
                    result["speaker"] = self._speaker_result("too_short")
                else:
                    _check_stop(stop_event)
                    speaker = self._load_speaker()
                    try:
                        stream = speaker.create_stream()
                        stream.accept_waveform(
                            sample_rate=sample_rate, waveform=samples
                        )
                        stream.input_finished()
                        if speaker.is_ready(stream):
                            embedding = speaker.compute(stream)
                            _check_stop(stop_event)
                            result["speaker"] = self._cluster(embedding, generation)
                        else:
                            result["speaker"] = self._speaker_result("too_short")
                    except (VoiceCancelled, VoiceError):
                        raise
                    except Exception as exc:
                        raise VoiceError(
                            f"Local speaker embedding failed: {exc}"
                        ) from exc
        _check_stop(stop_event)
        with self._session_lock:
            if generation != self._generation:
                raise VoiceCancelled("Audio perception session was reset")
        return result
