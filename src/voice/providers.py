"""Lazy, local-only speech adapters. Importing Daena never loads a speech model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from threading import RLock
from typing import Iterator, Mapping


class VoiceError(RuntimeError):
    """An actionable optional voice dependency/model error."""


class VoiceCancelled(VoiceError):
    pass


class VoiceLanguageUnavailable(VoiceError):
    """A requested passage has no configured local voice; session can continue."""


def _check_stop(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise VoiceCancelled("Voice operation cancelled")


@dataclass(frozen=True)
class AudioChunk:
    sample_rate: int
    pcm16: bytes

    def __post_init__(self):
        if (
            not isinstance(self.sample_rate, int)
            or not 8000 <= self.sample_rate <= 192000
        ):
            raise VoiceError("Speech output has an unsupported sample rate")
        if not isinstance(self.pcm16, bytes) or len(self.pcm16) % 2:
            raise VoiceError(
                "Speech output must contain complete signed 16-bit mono samples"
            )


class FasterWhisperProvider:
    """Whisper transcription on mono PCM16; no network access during inference.

    Native CTranslate2 inference is not forcibly killed. Cancellation is checked
    before loading/inference and between yielded segments; its output is discarded
    if cancellation arrived during an inference call.
    """

    def __init__(
        self,
        model_path="large-v3-turbo",
        *,
        device="cpu",
        compute_type="auto",
        language="auto",
        local_files_only=True,
        download_root=None,
        cpu_threads=4,
        beam_size=3,
        model_factory=None,
    ):
        if language not in {"auto", "fa", "en"}:
            raise VoiceError("Speech language must be auto, fa or en")
        if device not in {"cpu", "cuda", "auto"}:
            raise VoiceError("Speech device must be cpu, cuda or auto")
        if not 1 <= int(cpu_threads) <= 64 or not 1 <= int(beam_size) <= 10:
            raise VoiceError("Voice cpu_threads must be 1–64 and beam_size 1–10")
        # Inference must remain offline even if a caller accidentally requests
        # implicit model downloads. Explicit downloads belong to models.py.
        if not local_files_only:
            raise VoiceError(
                "Download speech models explicitly before enabling Voice Mode"
            )
        self.model_path = str(model_path)
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.download_root = str(download_root) if download_root else None
        self.cpu_threads = int(cpu_threads)
        self.beam_size = int(beam_size)
        self._factory = model_factory
        self._model = None
        self._lock = RLock()

    def _load(self):
        if self._model is not None:
            return self._model
        factory = self._factory
        if factory is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise VoiceError(
                    "Install local speech recognition: python -m pip install 'daena-agent[voice]'"
                ) from exc
            factory = WhisperModel
        device = self.device
        # CPU is the conservative auto choice while Ollama may occupy the GPU.
        # CUDA is available only when explicitly selected by the user.
        if device == "auto":
            device = "cpu"
        compute = (
            self.compute_type
            if self.compute_type != "auto"
            else ("int8" if device == "cpu" else "float16")
        )
        try:
            self._model = factory(
                self.model_path,
                device=device,
                compute_type=compute,
                cpu_threads=self.cpu_threads,
                num_workers=1,
                local_files_only=True,
                download_root=self.download_root,
            )
        except Exception as exc:
            raise VoiceError(
                "Cannot load the local Whisper model. Download/select its CTranslate2 folder in Voice settings; check device/compute support. "
                + str(exc)
            ) from exc
        return self._model

    def warmup(self, stop_event=None):
        _check_stop(stop_event)
        with self._lock:
            self._load()
        _check_stop(stop_event)

    def prepare(self, stop_event=None):
        """Load a local model once on the caller's background worker."""
        self.warmup(stop_event)

    def transcribe(self, pcm16: bytes, sample_rate=16000, stop_event=None) -> str:
        _check_stop(stop_event)
        if sample_rate != 16000:
            raise VoiceError(
                "Local transcription expects 16000 Hz mono PCM16; resample before transcription"
            )
        if not isinstance(pcm16, bytes) or len(pcm16) % 2:
            raise VoiceError("Microphone input must contain complete PCM16 samples")
        if not pcm16:
            return ""
        import numpy as np

        audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        with self._lock:
            _check_stop(stop_event)
            model = self._load()
            try:
                segments, _info = model.transcribe(
                    audio,
                    language=None if self.language == "auto" else self.language,
                    task="transcribe",
                    beam_size=self.beam_size,
                    condition_on_previous_text=False,
                    temperature=0.0,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 300},
                    word_timestamps=False,
                )
                parts = []
                for segment in segments:
                    _check_stop(stop_event)
                    text = str(segment.text).strip()
                    if text:
                        parts.append(text)
            except VoiceCancelled:
                raise
            except Exception as exc:
                raise VoiceError("Local transcription failed: " + str(exc)) from exc
            _check_stop(stop_event)
            return " ".join(parts)


def text_language(text: str, fallback="en") -> str:
    """Select one available voice by written language, not task semantics."""
    persian = len(
        re.findall(
            r"[\u0621-\u063a\u0641-\u064a\u067e\u0686\u0698\u06a9\u06af\u06cc]", text
        )
    )
    latin = len(re.findall(r"[A-Za-z]", text))
    return "fa" if persian > latin else "en" if latin else fallback


def language_runs(text: str, fallback="en"):
    """Keep adjacent words with the same writing system in one synthesis run."""
    words = re.findall(r"\S+\s*", text)
    current = fallback
    buffer = []
    for word in words:
        language = text_language(word, current)
        if buffer and language != current:
            yield current, "".join(buffer).strip()
            buffer = []
        current = language
        buffer.append(word)
    if buffer:
        yield current, "".join(buffer).strip()


class PiperProvider:
    """Sentence streaming through the maintained Piper 1.3+ AudioChunk API."""

    def __init__(
        self,
        model_paths: Mapping[str, str | Path] | None = None,
        *,
        language="auto",
        length_scale=1.0,
        voice_loader=None,
    ):
        if language not in {"auto", "fa", "en"}:
            raise VoiceError("Speech language must be auto, fa or en")
        if not 0.5 <= float(length_scale) <= 2:
            raise VoiceError("Voice length scale must be between 0.5 and 2")
        self.model_paths = {
            key: Path(value).expanduser()
            for key, value in (model_paths or {}).items()
            if value
        }
        self.language = language
        self.length_scale = float(length_scale)
        self._loader = voice_loader
        self._voices = {}
        self._lock = RLock()

    def _load(self, language):
        if language in self._voices:
            return self._voices[language]
        path = self.model_paths.get(language)
        if (
            path is None
            or not path.is_file()
            or not Path(str(path) + ".json").is_file()
        ):
            raise VoiceError(
                f"Select/download the {language} Piper .onnx voice and matching .onnx.json in Voice settings"
            )
        loader = self._loader
        if loader is None:
            try:
                from piper import PiperVoice
            except ImportError as exc:
                raise VoiceError(
                    "Install local speech synthesis: python -m pip install 'daena-agent[voice]'"
                ) from exc
            loader = PiperVoice.load
        try:
            voice = loader(str(path), config_path=str(path) + ".json", use_cuda=False)
        except Exception as exc:
            raise VoiceError(f"Cannot load the {language} Piper voice: {exc}") from exc
        self._voices[language] = voice
        return voice

    def warmup(self, stop_event=None):
        with self._lock:
            languages = (
                [self.language] if self.language != "auto" else list(self.model_paths)
            )
            if not languages:
                raise VoiceError(
                    "Select/download a Persian or English Piper voice in Voice settings"
                )
            for language in languages:
                _check_stop(stop_event)
                self._load(language)
        _check_stop(stop_event)

    def prepare(self, stop_event=None):
        """Warm only explicitly configured local voices, without downloading."""
        self.warmup(stop_event)

    def synthesize(
        self, text, language="auto", stop_event=None
    ) -> Iterator[AudioChunk]:
        _check_stop(stop_event)
        selected = self.language if language == "auto" else language
        if selected not in {"auto", "fa", "en"}:
            raise VoiceError("Speech language must be auto, fa or en")
        if not str(text).strip():
            return
        runs = (
            language_runs(str(text), text_language(str(text)))
            if selected == "auto"
            else [(selected, str(text))]
        )
        with self._lock:
            for current, run in runs:
                _check_stop(stop_event)
                voice = self._load(current)
                try:
                    if self.length_scale == 1.0:
                        chunks = voice.synthesize(run)
                    else:
                        from piper.config import SynthesisConfig

                        chunks = voice.synthesize(
                            run,
                            syn_config=SynthesisConfig(length_scale=self.length_scale),
                        )
                    for chunk in chunks:
                        _check_stop(stop_event)
                        if chunk.sample_width != 2 or chunk.sample_channels != 1:
                            raise VoiceError(
                                "Piper returned unsupported audio; expected mono PCM16"
                            )
                        yield AudioChunk(
                            sample_rate=int(chunk.sample_rate),
                            pcm16=bytes(chunk.audio_int16_bytes),
                        )
                except VoiceCancelled:
                    raise
                except VoiceError:
                    raise
                except Exception as exc:
                    raise VoiceError(
                        f"Local {current} speech synthesis failed: {exc}"
                    ) from exc
                _check_stop(stop_event)
