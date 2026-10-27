"""Local inference workers with bounded queues and cancellation epochs.

No Qt, audio hardware, cloud calls or model downloads in this module.
"""

from dataclasses import dataclass
from queue import Queue, Empty, Full
from threading import Event, Lock, Thread
import time
from uuid import uuid4

from .streaming import SpeechSegmenter, speech_segments
from .providers import AudioChunk, VoiceError, VoiceLanguageUnavailable


@dataclass(frozen=True)
class VoiceEvent:
    kind: str
    data: object
    epoch: int


class VoiceSession:
    def __init__(
        self,
        recognizer,
        synthesizer,
        *,
        silence_ms=600,
        max_seconds=20,
        perception=None,
        sound_events_enabled=False,
    ):
        self.recognizer, self.synthesizer = recognizer, synthesizer
        self.perception = perception
        self.acoustic_session_id = uuid4().hex
        self.sound_events_enabled = bool(sound_events_enabled and perception)
        self.segmenter = (
            SpeechSegmenter(silence_ms=silence_ms, max_seconds=max_seconds)
            if recognizer is not None
            else None
        )
        self.recognition_ready = Event()
        self.events = Queue(maxsize=64)
        self.audio = Queue(maxsize=6)
        self._speech = Queue(maxsize=2)
        self._text = Queue(maxsize=6)
        # Acoustic work cannot delay recognition or playback. Speech observations
        # have priority; continuous ambient observations are latest-only.
        self._perception_speech = Queue(maxsize=2)
        self._ambient = Queue(maxsize=1)
        self._perception_wake = Event()
        self._ambient_buffer = bytearray()
        self._ambient_bytes = 16000 * 2 * 2  # Two seconds of mono PCM16.
        self._input_generation = 0
        self._utterance_id = 0
        self._stop = Event()
        self._cancel = Event()
        self._lock = Lock()
        self.epoch = 0
        self._closed = False
        self._threads = [Thread(target=self._synthesize, daemon=True, name="daena-tts")]
        if recognizer is not None:
            self._threads.insert(
                0, Thread(target=self._recognize, daemon=True, name="daena-stt")
            )
        if self.perception is not None:
            self._threads.append(
                Thread(target=self._perceive, daemon=True, name="daena-acoustics")
            )
        for thread in self._threads:
            thread.start()

    def emit(self, kind, data, epoch):
        if epoch != self.epoch or self._stop.is_set():
            return
        if kind.startswith("perception") and self.events.qsize() >= 32:
            # Optional observations must not crowd out transcripts/errors while
            # the GUI is temporarily busy. The next acoustic window replaces it.
            return
        try:
            self.events.put_nowait(VoiceEvent(kind, data, epoch))
        except Full:
            # Status updates may be dropped; text/errors must not silently submit
            # partial commands. Queue overflow is explicit to the controller.
            try:
                self.events.get_nowait()
                self.events.put_nowait(
                    VoiceEvent(
                        "error",
                        "Voice event queue overflow; review the transcript.",
                        self.epoch,
                    )
                )
            except (Empty, Full):
                pass

    def feed(self, pcm16):
        if self._closed:
            return
        if self.sound_events_enabled:
            self._feed_ambient(pcm16)
        if self.segmenter is None:
            return
        # A large local model may take seconds to warm. Do not queue commands
        # while it is unavailable and turn normal warmup into a queue overflow.
        if not self.recognition_ready.is_set():
            self.segmenter.reset()
            return
        for utterance in self.segmenter.feed(pcm16):
            self.submit_audio(utterance)

    def finish_utterance(self):
        if self.segmenter is None:
            return
        if self._closed or not self.recognition_ready.is_set():
            self.segmenter.reset()
            return
        utterance = self.segmenter.finish()
        if utterance:
            self.submit_audio(utterance)

    def submit_audio(self, utterance):
        if self._closed or self.recognizer is None:
            return
        self._utterance_id += 1
        utterance_id = self._utterance_id
        job = (self.epoch, self._cancel, utterance, time.monotonic(), utterance_id)
        try:
            self._speech.put_nowait(job)
            if self.perception is not None:
                self._submit_perception(utterance.pcm16, "speech", utterance_id)
            self.emit("state", "Transcribing…", self.epoch)
        except Full:
            self.emit(
                "error",
                "Speech queue is full; that utterance was not submitted. Please try again.",
                self.epoch,
            )

    def _feed_ambient(self, pcm16):
        # Never retain an arbitrary microphone batch or an unbounded ring. The
        # classifier receives non-speech too, even when the VAD stays silent.
        data = memoryview(pcm16).cast("B")
        offset = 0
        while offset < len(data):
            size = min(
                self._ambient_bytes - len(self._ambient_buffer), len(data) - offset
            )
            self._ambient_buffer.extend(data[offset : offset + size])
            offset += size
            if len(self._ambient_buffer) == self._ambient_bytes:
                self._submit_perception(bytes(self._ambient_buffer), "ambient")
                self._ambient_buffer.clear()

    def _submit_perception(self, pcm16, source, utterance_id=None):
        queue = self._perception_speech if source == "speech" else self._ambient
        job = (
            self.epoch,
            self._cancel,
            pcm16,
            source,
            utterance_id,
            time.monotonic(),
            self._input_generation,
        )
        try:
            queue.put_nowait(job)
        except Full:
            # Perception is supplementary. A slow model drops its oldest pending
            # observation, never a transcript or an audio playback chunk.
            try:
                queue.get_nowait()
                queue.put_nowait(job)
            except (Empty, Full):
                pass
        self._perception_wake.set()

    def reset_input(self):
        """Discard partial microphone windows when muted or playback begins.

        This is nonblocking: ongoing native acoustic work is invalidated for
        ambient observations, while completed speech remains eligible for STT.
        """
        if self.segmenter is not None:
            self.segmenter.reset()
        self._input_generation += 1
        self._ambient_buffer.clear()
        while True:
            try:
                self._ambient.get_nowait()
            except Empty:
                break

    def speak(self, text, language="auto"):
        if self._closed:
            return
        if len(str(text)) > 8000:
            self.emit(
                "notice",
                "Read-aloud is limited to the first 8000 characters; full text remains in chat.",
                self.epoch,
            )
        text = str(text)[:8000]
        try:
            self._text.put_nowait(
                (self.epoch, self._cancel, text, language, time.monotonic())
            )
        except Full:
            self.emit(
                "error",
                "Read-aloud queue is full. Interrupt to clear it; the full answer remains in chat.",
                self.epoch,
            )

    def interrupt(self, *, preserve_input=False):
        with self._lock:
            self._cancel.set()
            self.epoch += 1
            self._cancel = Event()
            if self._closed:
                self._cancel.set()
            for queue in (
                self._speech,
                self._text,
                self.audio,
                self.events,
                self._perception_speech,
            ):
                while True:
                    try:
                        queue.get_nowait()
                    except Empty:
                        break
            if preserve_input:
                # A barge-in starts before endpointing completes. Keep its
                # pre-roll and first syllables while invalidating old inference.
                self._input_generation += 1
                self._ambient_buffer.clear()
                while True:
                    try:
                        self._ambient.get_nowait()
                    except Empty:
                        break
            else:
                self.reset_input()
            self._perception_wake.set()
        return self.epoch

    def close(self):
        if not self._closed:
            self._closed = True
            self._stop.set()
            self.interrupt()
        # Native inference is not forcibly killed; workers are daemonized and
        # outputs are invalidated. The GUI never blocks waiting for a model.

    @property
    def retired(self):
        return all(not thread.is_alive() for thread in self._threads)

    def _prepare(self, provider, *, notify=True):
        cancel = self._cancel
        epoch = self.epoch
        if self._stop.is_set():
            return False
        try:
            prepare = getattr(provider, "prepare", None)
            if callable(prepare):
                prepare(stop_event=cancel)
            if not cancel.is_set():
                if notify:
                    self.emit("ready", type(provider).__name__, epoch)
                return True
        except Exception as exc:
            if not cancel.is_set():
                self.emit(
                    "error",
                    f"Local voice model could not load: {type(exc).__name__}: {exc}",
                    epoch,
                )
        return False

    def _recognize(self):
        while not self._stop.is_set() and not self.recognition_ready.is_set():
            cancel = self._cancel
            if self._prepare(self.recognizer, notify=False):
                self.recognition_ready.set()
                self.emit("ready", type(self.recognizer).__name__, self.epoch)
                break
            if not cancel.is_set():
                return
        while not self._stop.is_set():
            try:
                epoch, cancel, utterance, started, utterance_id = self._speech.get(
                    timeout=0.1
                )
            except Empty:
                continue
            if cancel.is_set():
                continue
            try:
                text = self.recognizer.transcribe(
                    utterance.pcm16, 16000, stop_event=cancel
                ).strip()
                if text and not cancel.is_set():
                    self.emit(
                        "transcript",
                        {
                            "text": text,
                            "truncated": utterance.truncated,
                            "stt_ms": round((time.monotonic() - started) * 1000),
                            "utterance_id": utterance_id,
                        },
                        epoch,
                    )
                if not cancel.is_set():
                    self.emit("state", "Listening", epoch)
            except Exception as exc:
                if not cancel.is_set():
                    self.emit(
                        "error",
                        f"Speech recognition failed: {type(exc).__name__}: {exc}",
                        epoch,
                    )

    def _perceive(self):
        # Models may be cached by the controller, but speaker labels always start
        # afresh for this VoiceSession. Never acquire a native model lock on Qt.
        try:
            reset = getattr(self.perception, "reset", None)
            if callable(reset):
                reset()
            while not self._stop.is_set():
                cancel = self._cancel
                epoch = self.epoch
                try:
                    prepare = getattr(self.perception, "prepare", None)
                    if callable(prepare):
                        prepare(stop_event=cancel)
                    if not cancel.is_set():
                        self.emit("perception_ready", None, epoch)
                        break
                except Exception:
                    if not cancel.is_set():
                        raise
        except Exception as exc:
            self.emit(
                "perception_error",
                f"Acoustic analysis is unavailable: {type(exc).__name__}: {exc}",
                self.epoch,
            )
            return
        while not self._stop.is_set():
            self._perception_wake.clear()
            try:
                job = self._perception_speech.get_nowait()
            except Empty:
                try:
                    job = self._ambient.get_nowait()
                except Empty:
                    self._perception_wake.wait(0.1)
                    continue
            epoch, cancel, pcm16, source, utterance_id, started, generation = job
            if cancel.is_set() or (
                source == "ambient" and generation != self._input_generation
            ):
                continue
            try:
                evidence = self.perception.analyze(
                    pcm16, 16000, stop_event=cancel, speech=source == "speech"
                )
                if not cancel.is_set() and (
                    source != "ambient" or generation == self._input_generation
                ):
                    self.emit(
                        "perception",
                        {
                            **evidence,
                            "source": source,
                            "utterance_id": utterance_id,
                            "captured_at": started,
                            "voice_session_id": self.acoustic_session_id,
                            "analyzed_ms": round((time.monotonic() - started) * 1000),
                        },
                        epoch,
                    )
            except Exception as exc:
                if not cancel.is_set():
                    self.emit(
                        "perception_error",
                        f"Acoustic observation failed: {type(exc).__name__}: {exc}",
                        epoch,
                    )

    def _synthesize(self):
        while not self._stop.is_set():
            cancel = self._cancel
            if self._prepare(self.synthesizer):
                break
            if not cancel.is_set():
                return
        while not self._stop.is_set():
            try:
                epoch, cancel, text, language, started = self._text.get(timeout=0.1)
            except Empty:
                continue
            if cancel.is_set():
                continue
            try:
                first = True
                for sentence in speech_segments(text):
                    if cancel.is_set() or self._stop.is_set():
                        break
                    sentence_has_audio = False
                    for chunk in self.synthesizer.synthesize(
                        sentence, language, stop_event=cancel
                    ):
                        if cancel.is_set() or epoch != self.epoch:
                            break
                        # Empty native chunks are not playback and must not
                        # report a first-audio latency or successful speech.
                        chunk = AudioChunk(chunk.sample_rate, chunk.pcm16)
                        if not chunk.pcm16:
                            continue
                        sentence_has_audio = True
                        if first:
                            self.emit(
                                "latency",
                                {
                                    "first_pcm_ms": round(
                                        (time.monotonic() - started) * 1000
                                    )
                                },
                                epoch,
                            )
                            first = False
                        # Queue short PCM frames rather than whole sentences:
                        # bounded memory, quick playback start, immediate interruption.
                        size = max(2, int(chunk.sample_rate * 0.1) * 2)
                        for offset in range(0, len(chunk.pcm16), size):
                            part = AudioChunk(
                                sample_rate=chunk.sample_rate,
                                pcm16=chunk.pcm16[offset : offset + size],
                            )
                            while not cancel.is_set() and not self._stop.is_set():
                                try:
                                    self.audio.put((epoch, part), timeout=0.05)
                                    break
                                except Full:
                                    continue
                            if cancel.is_set() or self._stop.is_set():
                                break
                    if (
                        not sentence_has_audio
                        and not cancel.is_set()
                        and not self._stop.is_set()
                    ):
                        raise VoiceError(
                            "The local speech engine produced no audio for this passage. "
                            "Check the selected output voice, language and model files, then try the voice preview."
                        )
                while not cancel.is_set() and not self._stop.is_set():
                    try:
                        self.audio.put((epoch, None), timeout=0.05)
                        break
                    except Full:
                        continue
                if not cancel.is_set():
                    self.emit("tts_done", None, epoch)
            except VoiceLanguageUnavailable as exc:
                if not cancel.is_set():
                    # Missing an optional language is local to this answer.
                    # Finalize any preceding PCM and retain STT/TTS readiness.
                    while not cancel.is_set() and not self._stop.is_set():
                        try:
                            self.audio.put((epoch, None), timeout=0.05)
                            break
                        except Full:
                            continue
                    self.emit("speech_unavailable", str(exc), epoch)
            except Exception as exc:
                if not cancel.is_set():
                    self.emit(
                        "error",
                        f"Speech synthesis failed: {type(exc).__name__}: {exc}",
                        epoch,
                    )
