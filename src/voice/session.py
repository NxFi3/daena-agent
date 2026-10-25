"""Local inference workers with bounded queues and cancellation epochs.

No Qt, audio hardware, cloud calls or model downloads in this module.
"""

from dataclasses import dataclass
from queue import Queue, Empty, Full
from threading import Event, Lock, Thread
import time

from .streaming import SpeechSegmenter, speech_segments


@dataclass(frozen=True)
class VoiceEvent:
    kind: str
    data: object
    epoch: int


class VoiceSession:
    def __init__(self, recognizer, synthesizer, *, silence_ms=600, max_seconds=20):
        self.recognizer, self.synthesizer = recognizer, synthesizer
        self.segmenter = SpeechSegmenter(silence_ms=silence_ms, max_seconds=max_seconds)
        self.events = Queue(maxsize=64)
        self.audio = Queue(maxsize=6)
        self._speech = Queue(maxsize=2)
        self._text = Queue(maxsize=6)
        self._stop = Event()
        self._cancel = Event()
        self._lock = Lock()
        self.epoch = 0
        self._closed = False
        self._threads = [
            Thread(target=self._recognize, daemon=True, name="daena-stt"),
            Thread(target=self._synthesize, daemon=True, name="daena-tts"),
        ]
        for thread in self._threads:
            thread.start()

    def emit(self, kind, data, epoch):
        if epoch != self.epoch or self._stop.is_set():
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
        for utterance in self.segmenter.feed(pcm16):
            self.submit_audio(utterance)

    def finish_utterance(self):
        utterance = self.segmenter.finish()
        if utterance:
            self.submit_audio(utterance)

    def submit_audio(self, utterance):
        if self._closed:
            return
        job = (self.epoch, self._cancel, utterance, time.monotonic())
        try:
            self._speech.put_nowait(job)
            self.emit("state", "Transcribing…", self.epoch)
        except Full:
            self.emit(
                "error",
                "Speech queue is full; that utterance was not submitted. Please try again.",
                self.epoch,
            )

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

    def interrupt(self):
        with self._lock:
            self._cancel.set()
            self.epoch += 1
            self._cancel = Event()
            if self._closed:
                self._cancel.set()
            for queue in (self._speech, self._text, self.audio, self.events):
                while True:
                    try:
                        queue.get_nowait()
                    except Empty:
                        break
            self.segmenter.reset()
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

    def _prepare(self, provider):
        cancel = self._cancel
        if self._stop.is_set():
            return
        try:
            prepare = getattr(provider, "prepare", None)
            if callable(prepare):
                prepare(stop_event=cancel)
            if not cancel.is_set():
                self.emit("ready", type(provider).__name__, self.epoch)
        except Exception as exc:
            if not cancel.is_set():
                self.emit(
                    "error",
                    f"Local voice model could not load: {type(exc).__name__}: {exc}",
                    self.epoch,
                )

    def _recognize(self):
        self._prepare(self.recognizer)
        while not self._stop.is_set():
            try:
                epoch, cancel, utterance, started = self._speech.get(timeout=0.1)
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

    def _synthesize(self):
        self._prepare(self.synthesizer)
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
                    for chunk in self.synthesizer.synthesize(
                        sentence, language, stop_event=cancel
                    ):
                        if cancel.is_set() or epoch != self.epoch:
                            break
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
                            part = type(chunk)(
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
                while not cancel.is_set() and not self._stop.is_set():
                    try:
                        self.audio.put((epoch, None), timeout=0.05)
                        break
                    except Full:
                        continue
                if not cancel.is_set():
                    self.emit("tts_done", None, epoch)
            except Exception as exc:
                if not cancel.is_set():
                    self.emit(
                        "error",
                        f"Speech synthesis failed: {type(exc).__name__}: {exc}",
                        epoch,
                    )
