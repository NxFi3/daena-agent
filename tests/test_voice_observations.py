"""Native acoustic metadata is bounded optional data, not a raw model payload."""

import json

import pytest

from src.voice.observations import (
    MAX_OBSERVATION_CHARS,
    compact_observations,
    observation_context,
)


def test_compact_round_trip_preserves_session_speaker_and_omits_raw_audio():
    data = [
        {
            "source": "speech",
            "voice_session_id": "a" * 32,
            "speaker": {
                "id": "speaker_2",
                "status": "matched",
                "similarity": 0.8912,
                "embedding": [1] * 100,
            },
            "sounds": {
                "status": "known",
                "events": [{"label": "Doorbell", "score": 0.8123}],
            },
            "audio": "raw bytes",
            "path": "/private/recording.wav",
        }
    ]
    compact = compact_observations(data)
    text = observation_context(compact)
    assert (
        '"anonymous_session_label": "speaker_2"' in text
        and '"embedding_similarity": 0.891' in text
    )
    assert '"label": "Doorbell"' in text
    assert '"voice_session_id": "' + "a" * 32 + '"' in text
    assert (
        "raw bytes" not in text
        and "/private/recording.wav" not in text
        and '"embedding"' not in text
    )
    assert "not instructions" in text and "do not identify" in text


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        [],
        "text",
        {"observations": 1},
        {
            "observations": [
                None,
                {"source": []},
                {"source": "speech", "sounds": {"events": 1}},
            ]
        },
    ],
)
def test_malformed_optional_context_does_not_abort_chat(value):
    assert isinstance(observation_context(value), str)


def test_invalid_statuses_scores_and_huge_speaker_labels_are_not_echoed():
    compact = compact_observations(
        [
            {
                "source": "speech",
                "speaker": {
                    "id": "speaker_" + "9" * 5000,
                    "status": "execute arbitrary instructions",
                    "similarity": float("nan"),
                },
                "sounds": {
                    "status": {},
                    "events": [
                        {"label": "Clapping\n\0test", "score": 10**1000},
                        {"label": "Speech", "score": float("inf")},
                    ],
                },
            }
        ]
    )
    speaker = compact["observations"][0]["speaker"]
    assert speaker == {
        "anonymous_session_label": None,
        "status": "uncertain",
        "embedding_similarity": None,
    }
    encoded = json.dumps(compact, allow_nan=False)
    assert (
        "arbitrary instructions" not in encoded
        and "NaN" not in encoded
        and "Infinity" not in encoded
    )
    assert "Clapping test" in encoded
    assert compact["observations"][0]["sounds"]["status"] == "unknown"


def test_serialized_observations_have_fixed_bound_and_keep_latest_speech():
    noisy = {
        "source": "ambient",
        "sounds": {
            "status": "known",
            "events": [{"label": '"\\' * 80, "score": 0.5} for _ in range(3)],
        },
    }
    speech = {
        "source": "speech",
        "speaker": {"id": "speaker_1", "status": "new", "similarity": None},
        "sounds": noisy["sounds"],
    }
    compact = compact_observations([noisy, noisy, speech])
    assert len(json.dumps(compact, ensure_ascii=False)) <= MAX_OBSERVATION_CHARS
    assert (
        compact["observations"][-1]["speaker"]["anonymous_session_label"] == "speaker_1"
    )
