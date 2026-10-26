"""Optional local English Kokoro synthesis and explicit bilingual routing.

Kokoro has no official Persian voice. Persian is delegated to the caller's
configured Persian synthesizer, never transliterated or sent to an online API.
Importing this module does not import ONNX Runtime or Kokoro and cannot download
anything. Model files and the matching voices archive must already exist.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from threading import RLock

from .providers import (
    AudioChunk,
    VoiceCancelled,
    VoiceError,
    _check_stop,
    language_runs,
    text_language,
)


class KokoroProvider:
    """Kokoro ONNX's bounded async stream, consumed by the existing TTS worker.

    Native inference cannot be forcibly interrupted. The iterator checks the
    cancellation event before/after each native batch and every 100 ms PCM frame.
    Closing it joins the engine's current executor job before releasing the lock,
    so a second call cannot race a still-running model after interruption.
    """

    def __init__(
        self,
        model_path,
        voices_path,
        *,
        voice="af_heart",
        speed=1.0,
        cpu_threads=2,
        engine_factory=None,
    ):
        if not isinstance(voice, str) or not voice.startswith(
            ("af_", "am_", "bf_", "bm_")
        ):
            raise VoiceError("Select an English Kokoro voice (af_, am_, bf_ or bm_)")
        try:
            speed = float(speed)
            threads = int(cpu_threads)
        except (TypeError, ValueError, OverflowError) as exc:
            raise VoiceError("Kokoro speed and CPU threads must be numbers") from exc
        if not 0.5 <= speed <= 2.0:
            raise VoiceError("Kokoro speed must be between 0.5 and 2.0")
        if (
            isinstance(cpu_threads, bool)
            or threads != cpu_threads
            or not 1 <= threads <= 64
        ):
            raise VoiceError("Kokoro CPU threads must be an integer between 1 and 64")
        self.model_path = Path(model_path).expanduser()
        self.voices_path = Path(voices_path).expanduser()
        self.voice, self.speed, self.cpu_threads = voice, speed, threads
        self._factory = engine_factory
        self._engine = None
        self._lock = RLock()

    def _load(self):
        if self._engine is not None:
            return self._engine
        if not self.model_path.is_file() or not self.voices_path.is_file():
            raise VoiceError(
                "Select/download the local Kokoro .onnx model and matching voices .bin archive in Voice settings"
            )
        try:
            if self._factory is not None:
                engine = self._factory(
                    str(self.model_path),
                    str(self.voices_path),
                    cpu_threads=self.cpu_threads,
                )
            else:
                try:
                    import onnxruntime as ort
                    from kokoro_onnx import Kokoro
                except ImportError as exc:
                    raise VoiceError(
                        "Install optional local Kokoro: python -m pip install 'daena-agent[voice-kokoro]'"
                    ) from exc
                options = ort.SessionOptions()
                options.intra_op_num_threads = self.cpu_threads
                options.inter_op_num_threads = 1
                # Keep CPU use bounded while Ollama may occupy the GPU. Neither
                # from_session nor InferenceSession downloads models or assets.
                session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                engine = Kokoro.from_session(session, str(self.voices_path))
            if self.voice not in engine.get_voices():
                raise VoiceError(
                    f"Kokoro voice {self.voice!r} is absent from this archive; choose a matching English voice and model release"
                )
            if not callable(getattr(engine, "create_stream", None)):
                raise VoiceError(
                    "Kokoro requires the create_stream API from kokoro-onnx 0.6.1+"
                )
        except VoiceError:
            raise
        except Exception as exc:
            raise VoiceError(
                "Cannot load local Kokoro. Check the matching model/voices files and espeak-ng installation: "
                + str(exc)
            ) from exc
        self._engine = engine
        return engine

    def prepare(self, stop_event=None):
        _check_stop(stop_event)
        with self._lock:
            self._load()
        _check_stop(stop_event)

    def synthesize(self, text, language="auto", stop_event=None):
        _check_stop(stop_event)
        if language not in {"auto", "en"}:
            raise VoiceError(
                "Kokoro in Daena speaks English; configure Piper for Persian"
            )
        text = str(text).strip()
        if not text:
            return
        # Selecting English explicitly does not make unsupported Persian text
        # pronounceable. Check letters, including mixed-script words, before
        # loading the engine rather than letting its phonemizer guess.
        if any(text_language(char, "en") == "fa" for char in text):
            raise VoiceError(
                "Persian text requires the configured Persian voice; Kokoro has no official Persian model"
            )
        with self._lock:
            _check_stop(stop_event)
            engine = self._load()
            loop = asyncio.new_event_loop()
            stream = None
            try:
                stream = engine.create_stream(
                    text,
                    voice=self.voice,
                    speed=self.speed,
                    lang="en-gb" if self.voice.startswith("b") else "en-us",
                )
                while True:
                    _check_stop(stop_event)
                    try:
                        samples, rate = loop.run_until_complete(anext(stream))
                    except StopAsyncIteration:
                        break
                    _check_stop(stop_event)
                    import numpy as np

                    samples = np.asarray(samples)
                    if (
                        samples.ndim != 1
                        or not np.issubdtype(samples.dtype, np.floating)
                        or not np.isfinite(samples).all()
                    ):
                        raise VoiceError(
                            "Kokoro returned invalid mono floating-point PCM"
                        )
                    # In particular, float16 cannot represent 32767: clipping
                    # in that dtype would round +1.0 to -32768 on conversion.
                    samples = samples.astype(np.float64, copy=False)
                    pcm = (
                        np.clip(np.rint(np.clip(samples, -1, 1) * 32768), -32768, 32767)
                        .astype("<i2")
                        .tobytes()
                    )
                    # Validate the sample rate even when a faulty native engine
                    # produces an empty chunk. No unbounded whole-answer buffer.
                    checked = AudioChunk(int(rate), pcm)
                    frame_size = max(2, int(checked.sample_rate * 0.1) * 2)
                    for start in range(0, len(pcm), frame_size):
                        _check_stop(stop_event)
                        yield AudioChunk(
                            checked.sample_rate, pcm[start : start + frame_size]
                        )
            except (VoiceCancelled, VoiceError, GeneratorExit):
                raise
            except Exception as exc:
                raise VoiceError(
                    "Local English Kokoro synthesis failed: " + str(exc)
                ) from exc
            finally:
                try:
                    if stream is not None:
                        loop.run_until_complete(stream.aclose())
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.run_until_complete(loop.shutdown_default_executor())
                finally:
                    loop.close()
            _check_stop(stop_event)


class BilingualSynthesizer:
    """Route writing systems to explicitly configured local voice engines.

    ``english`` may be Kokoro or Piper; ``persian`` is normally Piper. A missing
    engine is an actionable error, never a silent language or cloud fallback.
    """

    def __init__(
        self, english=None, persian=None, *, language="auto", require_all_languages=True
    ):
        if language not in {"auto", "fa", "en"}:
            raise VoiceError("Speech language must be auto, fa or en")
        self.providers = {"en": english, "fa": persian}
        self.language = language
        self.require_all_languages = bool(require_all_languages)
        self._lock = RLock()

    def _provider(self, language):
        provider = self.providers.get(language)
        if provider is None:
            raise VoiceError(f"Select a local {language} voice in Voice settings")
        return provider

    def prepare(self, stop_event=None):
        selected = ("fa", "en") if self.language == "auto" else (self.language,)
        if self.language == "auto" and not self.require_all_languages:
            selected = tuple(
                code for code in selected if self.providers[code] is not None
            )
        if not selected:
            raise VoiceError("Select a local output voice in Voice settings")
        with self._lock:
            for language in selected:
                _check_stop(stop_event)
                provider = self._provider(language)
                prepare = getattr(provider, "prepare", None)
                if callable(prepare):
                    prepare(stop_event=stop_event)
        _check_stop(stop_event)

    def synthesize(self, text, language="auto", stop_event=None):
        _check_stop(stop_event)
        selected = self.language if language == "auto" else language
        if selected not in {"auto", "fa", "en"}:
            raise VoiceError("Speech language must be auto, fa or en")
        text = str(text).strip()
        if not text:
            return
        runs = (
            language_runs(text, text_language(text))
            if selected == "auto"
            else [(selected, text)]
        )
        with self._lock:
            for code, run in runs:
                _check_stop(stop_event)
                provider = self._provider(code)
                yield from provider.synthesize(run, code, stop_event=stop_event)
        _check_stop(stop_event)
