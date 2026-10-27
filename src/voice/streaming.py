"""Bounded speech endpointing and presentation-only Markdown cleanup."""

from collections import deque
from dataclasses import dataclass
import math
import re
import struct


@dataclass(frozen=True)
class Utterance:
    pcm16: bytes
    truncated: bool = False


class SpeechSegmenter:
    """20ms mono16k frames; endpointing never interprets command semantics."""

    frame_bytes = 640

    def __init__(self, *, silence_ms=600, max_seconds=20, min_speech_ms=120, vad=None):
        self.silence_frames = max(10, int(silence_ms) // 20)
        self.max_frames = max(50, int(float(max_seconds) * 50))
        self.min_speech_frames = max(3, int(min_speech_ms) // 20)
        self._vad = vad
        self.backend = "adaptive_energy"
        if vad is None:
            try:
                import webrtcvad

                detector = webrtcvad.Vad(2)
                self._vad = lambda frame: detector.is_speech(frame, 16000)
                self.backend = "webrtc"
            except ImportError:
                pass
        self.reset()

    def reset(self):
        self._buffer = bytearray()
        self._pre = deque(maxlen=15)
        self._frames = []
        self._voiced = self._silent = 0
        self._noise = 0.002
        self._quarantine = False
        self._quarantine_silence = 0

    def _speech(self, frame):
        if self._vad:
            return bool(self._vad(frame))
        samples = struct.unpack("<320h", frame)
        rms = math.sqrt(sum(v * v for v in samples) / 320) / 32768
        active = rms > max(0.007, self._noise * 3)
        if not active and not self._frames:
            self._noise = 0.98 * self._noise + 0.02 * rms
        return active

    def feed(self, pcm16):
        self._buffer.extend(pcm16)
        output = []
        while len(self._buffer) >= self.frame_bytes:
            frame = bytes(self._buffer[: self.frame_bytes])
            del self._buffer[: self.frame_bytes]
            speech = self._speech(frame)
            if self._quarantine:
                self._quarantine_silence = 0 if speech else self._quarantine_silence + 1
                if self._quarantine_silence >= self.silence_frames:
                    self._quarantine = False
                    self._pre.clear()
                continue
            if not self._frames:
                if not speech:
                    self._pre.append(frame)
                    continue
                self._frames = list(self._pre)
                self._pre.clear()
            self._frames.append(frame)
            self._voiced += int(speech)
            self._silent = 0 if speech else self._silent + 1
            capped = len(self._frames) >= self.max_frames
            if capped or self._silent >= self.silence_frames:
                if self._voiced >= self.min_speech_frames:
                    output.append(Utterance(b"".join(self._frames), capped))
                self._frames = []
                self._voiced = self._silent = 0
                self._pre.clear()
                if capped:
                    # An uninterrupted tail is part of the SAME capped command,
                    # never a new auto-sendable request. Wait for actual silence.
                    self._quarantine = True
                    self._quarantine_silence = 0
        return output

    def finish(self):
        output = (
            Utterance(b"".join(self._frames))
            if self._voiced >= self.min_speech_frames
            else None
        )
        self.reset()
        return output


def speech_segments(text, max_chars=220):
    """Read visible prose, omitting fenced code/URL targets; never read reasoning."""
    text = str(text)
    text = re.sub(r"```[^\n]*\n.*?(?:```|\Z)", " ", text, flags=re.S)
    text = re.sub(r"~~~[^\n]*\n.*?(?:~~~|\Z)", " ", text, flags=re.S)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"(?m)^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s*)", "", text)
    text = re.sub(r"[*_`]+", "", text)
    for sentence in re.split(r"(?<=[.!?؟؛])\s+|\n+", text):
        sentence = " ".join(sentence.split())
        while len(sentence) > max_chars:
            end = sentence.rfind(" ", 0, max_chars + 1)
            end = end if end > max_chars // 3 else max_chars
            yield sentence[:end].strip()
            sentence = sentence[end:].strip()
        if sentence:
            yield sentence
