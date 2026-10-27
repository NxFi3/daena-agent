"""Local echo control and speech-onset detection for interruptible voice.

The reverse stream is audio pulled by the output device, not queued TTS. Native
WebRTC AEC is preferred; the stable SpeexDSP C ABI is an optional system fallback.
Without either backend, speakers stay half duplex. Headphones are an explicit
user choice: software cannot reliably infer acoustic isolation from device names.
"""

from collections import deque
import ctypes
import ctypes.util
import math
import os
import struct
import time


_FRAME_BYTES = 640  # 20ms, 16kHz mono signed little-endian PCM16.
_ZERO = bytes(_FRAME_BYTES)


def _rms(pcm):
    samples = struct.unpack("<320h", pcm)
    return math.sqrt(sum(sample * sample for sample in samples) / 320) / 32768


class _WebRTC:
    name = "webrtc"

    def __init__(self, delay_ms):
        from aec_audio_processing import AudioProcessor

        self.processor = AudioProcessor(
            enable_aec=True, enable_ns=True, enable_agc=False, enable_vad=False
        )
        self.processor.set_stream_format(16000, 1)
        self.processor.set_reverse_stream_format(16000, 1)
        self.processor.set_stream_delay(delay_ms)

    def render(self, frame):
        for offset in (0, 320):
            self.processor.process_reverse_stream(frame[offset : offset + 320])

    def process(self, frame):
        return b"".join(
            self.processor.process_stream(frame[offset : offset + 320])
            for offset in (0, 320)
        )

    def close(self):
        self.processor = None


class _Speex:
    name = "speexdsp"

    def __init__(self, delay_ms):
        name = ctypes.util.find_library("speexdsp")
        if not name:
            # find_library does not always include LD_LIBRARY_PATH / Windows PATH.
            name = "libspeexdsp.so.1" if os.name != "nt" else "libspeexdsp.dll"
        self.lib = ctypes.CDLL(name)
        self.echo = self.pre = None
        self.references = deque(maxlen=25)
        pointer = ctypes.c_void_p
        short_p = ctypes.POINTER(ctypes.c_int16)
        signatures = {
            "speex_echo_state_init": ([ctypes.c_int, ctypes.c_int], pointer),
            "speex_echo_state_destroy": ([pointer], None),
            "speex_echo_ctl": ([pointer, ctypes.c_int, pointer], ctypes.c_int),
            "speex_echo_cancellation": ([pointer, short_p, short_p, short_p], None),
            "speex_preprocess_state_init": ([ctypes.c_int, ctypes.c_int], pointer),
            "speex_preprocess_state_destroy": ([pointer], None),
            "speex_preprocess_ctl": ([pointer, ctypes.c_int, pointer], ctypes.c_int),
            "speex_preprocess_run": ([pointer, short_p], ctypes.c_int),
        }
        for symbol, (args, result) in signatures.items():
            function = getattr(self.lib, symbol)
            function.argtypes, function.restype = args, result
        try:
            # A 500ms FIR tail accommodates USB/Bluetooth and native buffering.
            self.echo = self.lib.speex_echo_state_init(320, 8000)
            self.pre = self.lib.speex_preprocess_state_init(320, 16000)
            if not self.echo or not self.pre:
                raise RuntimeError("SpeexDSP could not allocate echo processing")
            rate = ctypes.c_int(16000)
            self.lib.speex_echo_ctl(self.echo, 24, ctypes.byref(rate))
            self.lib.speex_preprocess_ctl(self.pre, 24, self.echo)
            for request, value in ((0, 1), (2, 0), (18, -20), (20, -40), (22, -15)):
                setting = ctypes.c_int(value)
                self.lib.speex_preprocess_ctl(self.pre, request, ctypes.byref(setting))
        except Exception:
            self.close()
            raise

    def render(self, frame):
        self.references.append(frame)

    def process(self, frame):
        reference = self.references.popleft() if self.references else _ZERO
        samples = ctypes.c_int16 * 320
        microphone = samples.from_buffer_copy(frame)
        output = samples()
        self.lib.speex_echo_cancellation(
            self.echo, microphone, samples.from_buffer_copy(reference), output
        )
        self.lib.speex_preprocess_run(self.pre, output)
        return bytes(output)

    def close(self):
        if self.pre:
            self.lib.speex_preprocess_state_destroy(self.pre)
            self.pre = None
        if self.echo:
            self.lib.speex_echo_state_destroy(self.echo)
            self.echo = None
        self.references.clear()


def _make_backend(delay_ms):
    for factory in (_WebRTC, _Speex):
        try:
            return factory(delay_ms)
        except (ImportError, OSError, AttributeError, RuntimeError, ValueError):
            continue
    return None


def backend_status(mode="auto"):
    """Inspect native echo capability without recording or opening audio devices."""
    processor = DuplexProcessor(mode)
    try:
        return processor.status
    finally:
        processor.close()


class DuplexProcessor:
    """Bounded 20ms processing; call all methods on the capture/GUI thread.

    Output-device callbacks should enqueue consumed PCM in their own bounded
    thread-safe queue, then drain that queue into ``feed_render`` before capture.
    Native AEC inference is tiny and performs no model loading or networking.
    """

    def __init__(self, mode="auto", *, delay_ms=80, backend=None, vad=None):
        if mode not in {"auto", "headphones", "off"}:
            raise ValueError("Duplex mode must be auto, headphones, or off")
        self.mode = mode
        self.delay_ms = max(0, min(500, int(delay_ms)))
        self._backend = (
            backend
            if backend is not None
            else (_make_backend(self.delay_ms) if mode == "auto" else None)
        )
        self.backend = self._backend.name if self._backend else "none"
        self._closed = False
        self._render = bytearray()
        self._reference_history = bytearray()
        self._microphone_history = bytearray()
        self._last_reference = None
        self._playing = False
        self._play_frames = 0
        self._vad = vad
        if vad is None:
            try:
                import webrtcvad

                detector = webrtcvad.Vad(3)
                self._vad = lambda frame: detector.is_speech(frame, 16000)
            except ImportError:
                pass
        self.reset_input()

    @property
    def can_listen_during_playback(self):
        return not self._closed and (
            self.mode == "headphones"
            or (self.mode == "auto" and self._backend is not None)
        )

    @property
    def status(self):
        active = self.can_listen_during_playback
        if self.mode == "headphones":
            message = (
                "Interruptible listening · headphones; no software echo cancellation"
            )
        elif active:
            message = f"Interruptible listening · {self.backend} echo cancellation"
        elif self.mode == "off":
            message = "Half duplex · interruption by microphone is disabled"
        else:
            message = (
                "Half duplex · install the voice-duplex extra or SpeexDSP for speaker echo "
                "cancellation; with headphones choose Headphones"
            )
        return {
            "mode": self.mode,
            "backend": self.backend,
            "active": active,
            "message": message,
        }

    def feed_render(self, pcm16, timestamp=None):
        if self._closed or self._backend is None:
            return
        if len(pcm16) % 2:
            raise ValueError("Render PCM16 must contain complete samples")
        # Native read bursts must not retain unbounded history after a stall.
        data = memoryview(pcm16).cast("B")
        if len(data) > 16000:
            raise RuntimeError("Playback reference exceeds the 500ms safety limit")
        self._render.extend(data)
        while len(self._render) >= _FRAME_BYTES:
            frame = bytes(self._render[:_FRAME_BYTES])
            del self._render[:_FRAME_BYTES]
            self._backend.render(frame)
            self._reference_history.extend(frame)
            del self._reference_history[:-16640]  # 520ms maximum.
            self._last_reference = time.monotonic() if timestamp is None else timestamp

    def _echo_only(self, frame):
        """Reject coherent far-end leakage while the native filter converges.

        A normalized 40ms cross-correlation searches a bounded 520ms reference.
        This is an echo-presence gate, not a replacement cancellation filter:
        independent near-end speech lowers coherence and passes the native AEC.
        """
        import numpy as np

        self._microphone_history.extend(frame)
        del self._microphone_history[:-1280]
        if len(self._microphone_history) < 1280 or len(self._reference_history) < 1280:
            return False
        microphone = np.frombuffer(self._microphone_history, "<i2").astype(np.float32)
        reference = np.frombuffer(self._reference_history, "<i2").astype(np.float32)
        microphone -= microphone.mean()
        power = float(np.dot(microphone, microphone))
        if power < 1:
            return False
        count = len(microphone)
        sums = np.concatenate(([0.0], np.cumsum(reference, dtype=np.float64)))
        squares = np.concatenate(
            ([0.0], np.cumsum(reference * reference, dtype=np.float64))
        )
        window_power = squares[count:] - squares[:-count]
        window_sum = sums[count:] - sums[:-count]
        window_power -= window_sum * window_sum / count
        correlation = np.correlate(reference, microphone, "valid")
        coherence = np.abs(correlation) / np.sqrt(np.maximum(window_power * power, 1))
        return bool(np.max(coherence) >= 0.75)

    def process_capture(
        self, pcm16, *, playback_active=False, playing=None, timestamp=None
    ):
        if len(pcm16) != _FRAME_BYTES:
            raise ValueError("Duplex capture requires one 20ms mono16k PCM16 frame")
        if self._closed:
            return b"", False
        if playing is not None:
            playback_active = playing
        if playback_active and not self.can_listen_during_playback:
            self.reset_input()
            return b"", False
        if playback_active and not self._playing:
            self._play_frames = 0
        self._playing = bool(playback_active)
        self._play_frames += int(playback_active)
        now = time.monotonic() if timestamp is None else timestamp
        echo_only = False
        if self._backend is not None:
            # A broken/missing render tap cannot safely become raw speaker input.
            if playback_active and (
                self._last_reference is None or now - self._last_reference > 0.5
            ):
                self.reset_input()
                return b"", False
            if playback_active:
                echo_only = self._echo_only(pcm16)
            else:
                self._microphone_history.clear()
            pcm16 = self._backend.process(pcm16)
            if len(pcm16) != _FRAME_BYTES:
                raise RuntimeError("Echo backend returned an invalid PCM frame")
        if echo_only:
            # Endpointing/ASR must not turn the assistant's leaked words into a
            # new user command. Silence preserves the capture clock and hangover.
            pcm16 = _ZERO
        energy = _rms(pcm16)
        speech = energy >= 0.007 and (self._vad(pcm16) if self._vad else energy > 0.02)
        # Give AEC's adaptive filter 200ms to converge when a new render begins;
        # still feed those cleaned frames to endpointing to preserve first words.
        if playback_active and self._backend and self._play_frames < 10:
            speech = False
        self._recent.append(bool(speech))
        if speech:
            self._quiet = 0
        else:
            self._quiet += 1
        onset = not self._talking and len(self._recent) == 6 and sum(self._recent) >= 5
        if onset:
            self._talking = True
        elif self._quiet >= 15:
            self._talking = False
            self._recent.clear()
        return pcm16, onset

    def reset_input(self):
        self._recent = deque(maxlen=6)
        self._talking = False
        self._quiet = 0
        self._microphone_history.clear()

    def reset_render(self):
        self._render.clear()
        self._reference_history.clear()
        self._last_reference = None
        self._playing = False
        self._play_frames = 0

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._backend is not None:
            self._backend.close()
        self.reset_input()
        self.reset_render()
