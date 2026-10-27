"""Optional Qt audio I/O with bounded playback and optional local echo control.

Importing Studio does not import QtMultimedia. Device access only starts after an
explicit Voice Mode action; PCM passed to speech recognition is always 16 kHz,
mono, signed little-endian 16-bit audio.
"""

from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import sys
from threading import Lock
import time

import numpy as np
from PySide6.QtCore import QIODevice, QObject, QTimer, Signal, Slot


class PCMNormalizer:
    """Incremental interleaved PCM decode/downmix/resampling without audioop.

    Keep fractional sample position and the interpolation boundary between reads;
    chunk boundaries must not restart the resampling clock.
    """

    def __init__(
        self,
        sample_rate,
        channels=1,
        sample_format="int16",
        output_rate=16000,
        *,
        use_soxr=True,
    ):
        if sample_rate <= 0 or channels <= 0 or output_rate <= 0:
            raise ValueError("Audio rates and channel count must be positive")
        formats = {
            "int16": np.dtype("=i2"),
            "int32": np.dtype("=i4"),
            "uint8": np.dtype("u1"),
            "float32": np.dtype("=f4"),
        }
        if sample_format not in formats:
            raise ValueError(f"Unsupported PCM sample format: {sample_format}")
        self.sample_rate, self.output_rate = int(sample_rate), int(output_rate)
        self.channels, self.sample_format = int(channels), sample_format
        self.dtype = formats[sample_format]
        self._raw = b""
        self._samples = np.empty(0, dtype=np.float32)
        self._position = 0.0
        self._resampler = None
        self._flushed = False
        if use_soxr and self.sample_rate != self.output_rate:
            try:
                import soxr

                self._resampler = soxr.ResampleStream(
                    self.sample_rate, self.output_rate, 1, dtype="float32", quality="HQ"
                )
            except ImportError:
                pass  # The optional voice extra supplies HQ anti-aliased conversion.

    @staticmethod
    def _pcm(values):
        return (np.clip(values, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()

    def flush(self):
        if self._flushed:
            return b""
        self._flushed = True
        if self._resampler is not None:
            return self._pcm(
                self._resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)
            )
        if self._samples.size and self.sample_rate != self.output_rate:
            positions = np.arange(
                self._position, len(self._samples), self.sample_rate / self.output_rate
            )
            return self._pcm(
                np.interp(positions, np.arange(len(self._samples)), self._samples)
            )
        return b""

    def feed(self, raw):
        if self._flushed:
            raise ValueError("A finalized audio converter cannot accept more samples")
        self._raw += bytes(raw)
        frame_size = self.channels * self.dtype.itemsize
        length = len(self._raw) // frame_size * frame_size
        if not length:
            return b""
        data, self._raw = self._raw[:length], self._raw[length:]
        values = np.frombuffer(data, dtype=self.dtype).astype(np.float32)
        if self.sample_format == "uint8":
            values = (values - 128.0) / 128.0
        elif self.sample_format == "int16":
            values /= 32768.0
        elif self.sample_format == "int32":
            values /= 2147483648.0
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
        values = values.reshape(-1, self.channels).mean(axis=1)
        if self.sample_rate == self.output_rate:
            return (np.clip(values, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()
        if self._resampler is not None:
            return self._pcm(self._resampler.resample_chunk(values, last=False))
        self._samples = np.concatenate((self._samples, values))
        step = self.sample_rate / self.output_rate
        count = max(0, int(np.ceil((len(self._samples) - 1 - self._position) / step)))
        if not count:
            return b""
        positions = self._position + np.arange(count) * step
        output = np.interp(positions, np.arange(len(self._samples)), self._samples)
        self._position += count * step
        discard = min(int(self._position), max(0, len(self._samples) - 1))
        self._samples = self._samples[discard:]
        self._position -= discard
        return (np.clip(output, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()


def _pipewire_config_paths():
    name = os.environ.get("PIPEWIRE_CONFIG_NAME", "client.conf")
    if Path(name).is_absolute():
        return [Path(name)]
    prefix = os.environ.get("PIPEWIRE_CONFIG_PREFIX", "pipewire")
    directories = [
        Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / prefix,
        Path("/etc") / prefix,
        Path("/usr/local/share") / prefix,
        Path("/usr/share") / prefix,
    ]
    if os.environ.get("PIPEWIRE_CONFIG_DIR"):
        directory = Path(os.environ["PIPEWIRE_CONFIG_DIR"])
        directories[:0] = [directory, directory / prefix]
    return [directory / name for directory in directories]


def _prepare_audio_backend():
    """Guard a native Qt/PipeWire null-config crash before touching devices.

    Qt 6.12 can load libpipewire on minimal systems without PipeWire's client
    configuration, then crash in native enumeration instead of returning an
    error. Select the compiled PulseAudio route only when the user has not
    selected a backend; ordinary Studio imports never run this code.
    """
    if not sys.platform.startswith("linux"):
        return
    selected = os.environ.get("QT_AUDIO_BACKEND", "").strip().lower()
    if selected and selected != "pipewire":
        return
    if any(path.is_file() for path in _pipewire_config_paths()):
        return
    if selected == "pipewire":
        raise RuntimeError(
            "Qt's PipeWire audio backend needs client.conf, but no client configuration was found. "
            "Repair the PipeWire installation or explicitly select QT_AUDIO_BACKEND=pulseaudio."
        )
    from ctypes.util import find_library

    if not find_library("pulse"):
        raise RuntimeError(
            "No usable Qt audio backend: PipeWire client.conf and the PulseAudio runtime are missing. "
            "Install your desktop audio runtime before starting Voice Mode."
        )
    os.environ["QT_AUDIO_BACKEND"] = "pulseaudio"


def _qt_audio():
    _prepare_audio_backend()
    from PySide6.QtMultimedia import (
        QAudio,
        QAudioFormat,
        QAudioSink,
        QAudioSource,
        QMediaDevices,
    )

    return QAudio, QAudioFormat, QAudioSink, QAudioSource, QMediaDevices


def _audio_enum_is(value, expected):
    """Match Qt audio enum aliases across PySide versions.

    Qt exposes the same C++ enums through QAudio and QtAudio. Some PySide builds
    wrap those aliases as different Python Enum classes: identical repr/value,
    but inequality. Sink/source state() and error() can return either wrapper.
    Keep the enum member name as well as its numeric value to avoid mixing
    unrelated state/error members, while retaining injected integer test APIs.
    """
    return value == expected or (
        getattr(value, "name", None) is not None
        and value.name == getattr(expected, "name", None)
        and value.value == getattr(expected, "value", None)
    )


class _PlaybackBuffer(QIODevice):
    """Thread-safe sequential pull buffer consumed by Qt's audio backend."""

    def __init__(self, maximum, parent=None, *, read_tap=None):
        super().__init__(parent)
        self.maximum = int(maximum)
        self._chunks, self._length = deque(), 0
        self._lock = Lock()
        self.read_tap = read_tap
        self.open(QIODevice.ReadOnly | QIODevice.Unbuffered)

    def append(self, data):
        data = bytes(data)
        with self._lock:
            if self._length + len(data) > self.maximum:
                return False
            if data:
                self._chunks.append(data)
                self._length += len(data)
        self.readyRead.emit()
        return True

    def isSequential(self):
        return True

    def bytesAvailable(self):
        with self._lock:
            count = self._length
        return count + super().bytesAvailable()

    def readData(self, maximum):
        result = bytearray()
        with self._lock:
            while self._chunks and len(result) < maximum:
                chunk = self._chunks.popleft()
                take = min(len(chunk), maximum - len(result))
                result.extend(chunk[:take])
                self._length -= take
                if take < len(chunk):
                    self._chunks.appendleft(chunk[take:])
        result = bytes(result)
        # Qt can pull from a native audio thread. The tap only places bytes in
        # a bounded thread-safe queue; device access and DSP stay off this path.
        tap = self.read_tap
        if result and tap is not None:
            tap(result)
        return result

    def writeData(self, data):
        return -1

    def clear(self):
        with self._lock:
            self._chunks.clear()
            self._length = 0


class AudioDevices(QObject):
    """Main-thread Qt audio owner; speech model work belongs to worker threads."""

    pcm_ready = Signal(bytes)
    level = Signal(float)
    fault = Signal(str)
    playback_started = Signal()
    playback_stopped = Signal()
    speech_started = Signal()
    duplex_status = Signal(dict)
    MAX_BUFFERED_SECONDS = 20
    FRAME_BYTES = 640  # 20 ms of 16 kHz mono PCM16

    def __init__(self, parent=None):
        super().__init__(parent)
        self._retired = {}
        self._source = self._input_io = None
        self._sink = self._playback = None
        self._normalizer = None
        self._capture_needs_reset = False
        self._output_normalizer = None
        self._input_pending = bytearray()
        self._muted = False
        self._echo_blocked = False
        self._playing = False
        self._closed = False
        self._output_id = ""
        self._output_format = None
        self._source_rate = None
        self._duplex = None
        self._duplex_info = {
            "mode": "off",
            "backend": "none",
            "active": False,
            "message": "Microphone pauses during speech playback",
        }
        self._render_lock = Lock()
        self._render_chunks = deque()
        self._render_bytes = 0
        self._render_generation = 0
        self._render_dropped = False
        self._render_normalizer = None
        self._render_kind = None
        self._tail = QTimer(self)
        self._tail.setSingleShot(True)
        self._tail.timeout.connect(self._release_echo_gate)

    @property
    def listening_during_playback(self):
        return bool(
            self._duplex is not None and self._duplex.can_listen_during_playback
        )

    def configure_duplex(self, mode="auto", *, delay_ms=80):
        """Configure interruption before capture starts; no models or downloads.

        ``auto`` requires a usable local echo canceller. If it is unavailable,
        preserve half-duplex instead of routing speaker echo back into the Agent.
        ``headphones`` is an explicit opt-in for acoustically isolated output.
        """
        if mode not in {"auto", "headphones", "off"}:
            raise ValueError("Voice interruption mode must be auto, headphones or off")
        previous, self._duplex = self._duplex, None
        if previous is not None:
            previous.close()
        self._clear_render()
        try:
            from src.voice.duplex import DuplexProcessor

            self._duplex = DuplexProcessor(mode=mode, delay_ms=delay_ms)
            self._duplex_info = dict(self._duplex.status)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            self._duplex_info = {
                "mode": mode,
                "backend": "none",
                "active": False,
                "message": f"Voice interruption unavailable; microphone pauses during playback: {exc}",
            }
        self._echo_blocked = bool(
            (self._playing or self._tail.isActive())
            and not self.listening_during_playback
        )
        self.duplex_status.emit(dict(self._duplex_info))
        return dict(self._duplex_info)

    def _clear_render(self):
        with self._render_lock:
            self._render_generation += 1
            self._render_chunks.clear()
            self._render_bytes = 0
            self._render_dropped = False
        self._render_normalizer = self._render_kind = None

    def _tap_render(self, data, kind, generation, maximum):
        # This callback is allowed on Qt's pull thread. Never call a Qt device,
        # a resampler or an acoustic model here, and never retain queued TTS.
        with self._render_lock:
            if self._closed or generation != self._render_generation:
                return
            if self._render_bytes + len(data) > maximum:
                self._render_chunks.clear()
                self._render_bytes = 0
                self._render_dropped = True
                return
            self._render_chunks.append((bytes(data), kind, time.monotonic()))
            self._render_bytes += len(data)

    def _drain_render(self):
        if self._duplex is None:
            return
        with self._render_lock:
            chunks = list(self._render_chunks)
            self._render_chunks.clear()
            self._render_bytes = 0
            dropped, self._render_dropped = self._render_dropped, False
        if dropped:
            # Losing the reference makes echo cancellation untrustworthy. Keep
            # speaking normally, but fail closed for automatic interruption.
            self._duplex_failed("The playback reference fell behind")
            return
        try:
            for raw, kind, timestamp in chunks:
                if self._render_kind != kind:
                    if self._render_normalizer is not None:
                        tail = self._render_normalizer.flush()
                        if tail:
                            self._duplex.feed_render(tail, timestamp=timestamp)
                    self._render_kind = kind
                    # This is an echo reference, not the audible output. Avoid
                    # the HQ resampler's look-ahead delay on this timing path.
                    self._render_normalizer = PCMNormalizer(*kind, use_soxr=False)
                pcm = self._render_normalizer.feed(raw)
                if pcm:
                    self._duplex.feed_render(pcm, timestamp=timestamp)
        except (OSError, RuntimeError, ValueError) as exc:
            self._duplex_failed(str(exc))

    def _duplex_failed(self, reason):
        previous, self._duplex = self._duplex, None
        if previous is not None:
            previous.close()
        self._clear_render()
        self._duplex_info = {
            "mode": self._duplex_info.get("mode", "auto"),
            "backend": "none",
            "active": False,
            "message": f"Automatic interruption paused; microphone pauses during playback: {reason}",
        }
        self._echo_blocked = bool(self._playing or self._tail.isActive())
        self.duplex_status.emit(dict(self._duplex_info))

    def _retire(self, resource):
        # PySide virtual QIODevice implementations need their Python wrapper
        # alive until Qt processes DeferredDelete, even when C++ has a parent.
        key = id(resource)
        self._retired[key] = resource
        resource.destroyed.connect(
            lambda *args, identity=key: QTimer.singleShot(
                0, lambda: self._retired.pop(identity, None)
            )
        )
        resource.deleteLater()

    @staticmethod
    def devices(*, include_inputs=True):
        try:
            _, _, _, _, media = _qt_audio()

            def rows(devices, default):
                return [
                    {
                        "id": bytes(d.id()).hex(),
                        "name": d.description(),
                        "default": d.id() == default.id(),
                    }
                    for d in devices
                ]

            return {
                "inputs": rows(media.audioInputs(), media.defaultAudioInput())
                if include_inputs
                else [],
                "outputs": rows(media.audioOutputs(), media.defaultAudioOutput()),
            }
        except (ImportError, OSError, RuntimeError) as exc:
            return {"inputs": [], "outputs": [], "error": str(exc)}

    def _device(self, device_id, *, output=False):
        _, _, _, _, media = _qt_audio()
        devices = media.audioOutputs() if output else media.audioInputs()
        if device_id:
            result = next(
                (d for d in devices if bytes(d.id()).hex() == device_id), None
            )
            if result is None:
                raise RuntimeError(
                    "The selected audio device is no longer available; select another device"
                )
        else:
            result = media.defaultAudioOutput() if output else media.defaultAudioInput()
        if result.isNull():
            raise RuntimeError(
                "No audio output device is available"
                if output
                else "No microphone is available"
            )
        return result

    @staticmethod
    def _format_kind(fmt):
        name = fmt.sampleFormat().name.lower()
        return {
            "float": "float32",
            "int16": "int16",
            "int32": "int32",
            "uint8": "uint8",
        }.get(name, "unsupported")

    @Slot(str)
    def start_capture(self, input_device_id=""):
        self.stop_capture()
        self._closed = False
        try:
            _, format_class, _, source_class, _ = _qt_audio()
            device = self._device(input_device_id)
            fmt = format_class()
            fmt.setSampleRate(16000)
            fmt.setChannelCount(1)
            fmt.setSampleFormat(format_class.Int16)
            if not device.isFormatSupported(fmt):
                fmt = device.preferredFormat()
            self._normalizer = PCMNormalizer(
                fmt.sampleRate(), fmt.channelCount(), self._format_kind(fmt)
            )
            source = source_class(device, fmt, self)
            source.setBufferSize(
                max(
                    fmt.bytesForDuration(
                        40000 if self.listening_during_playback else 100000
                    ),
                    fmt.bytesPerFrame(),
                )
            )
            self._source = source
            source.stateChanged.connect(
                lambda state, owner=source: self._capture_state(owner, state)
            )
            self._input_io = source.start()
            if self._input_io is None:
                raise RuntimeError(
                    "Could not open the microphone; check device permissions and system audio settings"
                )
            self._input_io.readyRead.connect(self._read_capture)
            if self._muted:
                source.suspend()
            return True
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            self.stop_capture()
            self.fault.emit(f"Microphone: {exc}")
            return False

    def _capture_state(self, owner, state):
        if owner is not self._source:
            return
        audio, _, _, _, _ = _qt_audio()
        if _audio_enum_is(state, audio.StoppedState) and not _audio_enum_is(
            owner.error(), audio.NoError
        ):
            self.fault.emit(
                "Microphone stopped unexpectedly; check permissions or reconnect the device"
            )

    def _read_capture(self):
        if self._input_io is None:
            return
        raw = bytes(self._input_io.readAll())
        if self._muted or self._echo_blocked or self._closed:
            self._input_pending.clear()
            return
        try:
            self._drain_render()
            if self._capture_needs_reset:
                old = self._normalizer
                self._normalizer = PCMNormalizer(
                    old.sample_rate, old.channels, old.sample_format
                )
                self._capture_needs_reset = False
            self._input_pending.extend(self._normalizer.feed(raw))
            self._emit_capture_frames()
        except (ValueError, TypeError) as exc:
            self.fault.emit(f"Microphone format: {exc}")
            self.stop_capture()

    def _emit_capture_frames(self, *, pad=False):
        if pad and self._input_pending and len(self._input_pending) % self.FRAME_BYTES:
            self._input_pending.extend(
                b"\0" * (self.FRAME_BYTES - len(self._input_pending) % self.FRAME_BYTES)
            )
        while len(self._input_pending) >= self.FRAME_BYTES:
            if self._echo_blocked:
                self._input_pending.clear()
                self.level.emit(0.0)
                break
            frame = bytes(self._input_pending[: self.FRAME_BYTES])
            del self._input_pending[: self.FRAME_BYTES]
            onset = False
            if self._duplex is not None:
                try:
                    frame, onset = self._duplex.process_capture(
                        frame,
                        playback_active=self._playing or self._tail.isActive(),
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    self._duplex_failed(str(exc))
                    if self._echo_blocked:
                        frame = b""
                if not frame:
                    self.level.emit(0.0)
                    continue
            if onset and self.listening_during_playback:
                # The controller cancels old synthesis/output synchronously,
                # preserving the microphone prefix already sent to its VAD.
                self.speech_started.emit()
            if self._closed or self._muted:
                break
            samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768
            self.level.emit(float(min(1.0, np.sqrt(np.mean(samples * samples)))))
            self.pcm_ready.emit(frame)

    @Slot()
    def finish_capture(self):
        """Drain and finalize the current utterance, preserving final phonemes.

        The recognition consumer receives final padded 20ms frames synchronously
        before it is asked to finish its utterance. The native device remains
        open and the next microphone samples start a fresh resampling stream.
        """
        if (
            self._closed
            or self._muted
            or self._echo_blocked
            or self._normalizer is None
        ):
            return False
        self._read_capture()
        if (
            self._closed
            or self._muted
            or self._echo_blocked
            or self._normalizer is None
        ):
            return False
        try:
            old = self._normalizer
            fresh = PCMNormalizer(old.sample_rate, old.channels, old.sample_format)
            if self._capture_needs_reset:
                self._input_pending.clear()
            else:
                self._input_pending.extend(old.flush())
                # Partial native frames are not guessed or padded into samples;
                # keep those bytes until the device supplies the remaining frame.
                fresh._raw = old._raw
            self._normalizer = fresh
            self._capture_needs_reset = False
            self._emit_capture_frames(pad=True)
            return True
        except (ValueError, TypeError) as exc:
            self.fault.emit(f"Microphone finalization: {exc}")
            self.stop_capture()
            return False

    @Slot(bool)
    def set_muted(self, muted):
        self._muted = bool(muted)
        self._input_pending.clear()
        self._capture_needs_reset = True
        if self._duplex is not None:
            self._duplex.reset_input()
        if self._source:
            if self._muted:
                self._source.suspend()
            else:
                if self._input_io:
                    self._input_io.readAll()
                self._source.resume()
        self.level.emit(0.0)

    @Slot()
    def stop_capture(self):
        source, self._source = self._source, None
        self._input_io = None
        if source:
            source.stop()
            self._retire(source)
        self._normalizer = None
        self._capture_needs_reset = False
        self._input_pending.clear()
        if self._duplex is not None:
            self._duplex.reset_input()
        self.level.emit(0.0)

    @Slot(str)
    def set_output_device(self, device_id=""):
        if device_id != self._output_id:
            self.stop_playback()
            self._output_id = str(device_id)

    def prepare_output(self):
        """Validate the selected speaker without opening or inspecting a microphone.

        This is a main-thread preflight, not proof of audible sound. Playback's
        ActiveState and an explicit user output test provide separate evidence.
        Native device failures deliberately raise for the client to present.
        """
        device = self._device(self._output_id, output=True)
        fmt = device.preferredFormat()
        return {
            "id": bytes(device.id()).hex(),
            "name": device.description(),
            "sample_rate": fmt.sampleRate(),
            "channels": fmt.channelCount(),
            "volume": float(self._sink.volume()) if self._sink is not None else 1.0,
        }

    def _encode_output(self, pcm16, sample_rate):
        fmt = self._output_format
        tail = b""
        if self._source_rate != sample_rate or self._output_normalizer is None:
            if self._output_normalizer is not None:
                tail = self._format_output(self._output_normalizer.flush())
            self._output_normalizer = PCMNormalizer(
                sample_rate, output_rate=fmt.sampleRate()
            )
            self._source_rate = sample_rate
        return tail + self._format_output(self._output_normalizer.feed(pcm16))

    def _format_output(self, mono):
        fmt = self._output_format
        values = np.frombuffer(mono, dtype="<i2").astype(np.float32) / 32768
        if fmt.channelCount() > 1:
            values = np.repeat(values[:, None], fmt.channelCount(), axis=1).reshape(-1)
        kind = self._format_kind(fmt)
        if kind == "float32":
            return values.astype("=f4").tobytes()
        if kind == "int32":
            return (values.astype(np.float64) * 2147483648).astype("=i4").tobytes()
        if kind == "uint8":
            return np.clip(values * 128 + 128, 0, 255).astype("u1").tobytes()
        if kind == "int16":
            return (values * 32768).astype("=i2").tobytes()
        raise ValueError("The output device uses an unsupported PCM format")

    @Slot()
    def finish_playback(self):
        """Flush resampler latency after the final synthesis chunk, without stopping audio."""
        if self._output_normalizer is not None and self._playback is not None:
            tail = self._format_output(self._output_normalizer.flush())
            self._output_normalizer = None
            self._source_rate = None
            if tail:
                if not self._playback.append(tail):
                    self.stop_playback()
                    self.fault.emit(
                        "Playback resampling tail exceeded the buffer limit"
                    )
                elif self._sink is not None:
                    self._begin_output(_qt_audio()[0])

    @property
    def queued_seconds(self):
        if self._playback is None or self._output_format is None:
            return 0.0
        return self._playback.bytesAvailable() / max(
            1, self._output_format.bytesForDuration(1000000)
        )

    def can_accept(self, pcm16, sample_rate):
        return (
            sample_rate > 0
            and self.queued_seconds + len(pcm16) / (sample_rate * 2)
            <= self.MAX_BUFFERED_SECONDS - 0.1
        )

    @Slot(bytes, int)
    def play_pcm(self, pcm16, sample_rate):
        if self._closed or not pcm16:
            return False
        try:
            if (
                not isinstance(sample_rate, int)
                or not 8000 <= sample_rate <= 192000
                or len(pcm16) % 2
            ):
                raise ValueError("Invalid PCM16 audio chunk or sample rate")
            audio, format_class, sink_class, _, _ = _qt_audio()
            if self._sink is None:
                device = self._device(self._output_id, output=True)
                fmt = format_class()
                fmt.setSampleRate(sample_rate)
                fmt.setChannelCount(1)
                fmt.setSampleFormat(format_class.Int16)
                if not device.isFormatSupported(fmt):
                    fmt = device.preferredFormat()
                self._output_format = fmt
                kind = (
                    fmt.sampleRate(),
                    fmt.channelCount(),
                    self._format_kind(fmt),
                )
                generation = self._render_generation
                maximum = max(fmt.bytesForDuration(500000), fmt.bytesPerFrame())
                self._playback = _PlaybackBuffer(
                    fmt.bytesForDuration(self.MAX_BUFFERED_SECONDS * 1000000),
                    self,
                    read_tap=(
                        lambda data: self._tap_render(data, kind, generation, maximum)
                    )
                    if self.listening_during_playback
                    and self._duplex_info.get("mode") == "auto"
                    else None,
                )
                self._sink = sink_class(device, fmt, self)
                self._sink.setBufferSize(
                    max(
                        fmt.bytesForDuration(
                            40000 if self.listening_during_playback else 80000
                        ),
                        fmt.bytesPerFrame(),
                    )
                )
                self._sink.stateChanged.connect(
                    lambda state, owner=self._sink: self._playback_state(owner, state)
                )
            if len(pcm16) / (sample_rate * 2) > self.MAX_BUFFERED_SECONDS - 0.1:
                raise ValueError(
                    "A speech chunk exceeds the playback limit; split synthesis into shorter chunks"
                )
            if not self.can_accept(pcm16, sample_rate):
                return False  # Caller retains the unmodified chunk and retries after draining.
            encoded = self._encode_output(pcm16, sample_rate)
            if not self._playback.append(encoded):
                raise RuntimeError(
                    "Playback buffer accounting failed; speech was stopped"
                )
            if self._playback.bytesAvailable():
                self._begin_output(audio)
            return True
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            self.stop_playback()
            self.fault.emit(f"Playback: {exc}")
            return False

    def _begin_output(self, audio):
        self._tail.stop()
        self._echo_blocked = not self.listening_during_playback
        if self._echo_blocked:
            self._capture_needs_reset = True
            self._input_pending.clear()
        state = self._sink.state()
        if _audio_enum_is(state, audio.StoppedState) or _audio_enum_is(
            state, audio.IdleState
        ):
            # resume() is defined for SuspendedState, not underrun/IdleState.
            # If readyRead did not already wake the pull backend, restart
            # against the same still-queued buffer; Active audio is untouched.
            self._sink.start(self._playback)
        elif _audio_enum_is(state, audio.SuspendedState):
            self._sink.resume()
        # Backends may emit ActiveState synchronously from start()/resume(), or
        # later after native setup. A queued PCM block alone is not playback.
        if self._sink is not None and _audio_enum_is(
            self._sink.state(), audio.ActiveState
        ):
            self._mark_playing()

    def _mark_playing(self):
        if not self._playing:
            self._playing = True
            self.playback_started.emit()

    def _playback_state(self, owner, state):
        if owner is not self._sink:
            return
        audio, _, _, _, _ = _qt_audio()
        if _audio_enum_is(state, audio.ActiveState):
            self._mark_playing()
        elif _audio_enum_is(state, audio.StoppedState) and not _audio_enum_is(
            owner.error(), audio.NoError
        ):
            self.stop_playback()
            self.fault.emit(
                "Audio output stopped unexpectedly; check the selected speaker device"
            )
        elif (
            _audio_enum_is(state, audio.IdleState)
            and self._playback
            and not self._playback.bytesAvailable()
        ):
            if self._playing:
                self._playing = False
                self.playback_stopped.emit()
            self._tail.start(200)

    def _release_echo_gate(self):
        if not self._playing:
            self._echo_blocked = False
            if not self.listening_during_playback:
                self._input_pending.clear()

    @Slot()
    def stop_playback(self):
        had_output = self._playing or self._tail.isActive()
        sink, self._sink = self._sink, None
        playback, self._playback = self._playback, None
        if sink:
            sink.stop()
            self._retire(sink)
        if playback:
            playback.read_tap = None
            playback.clear()
            playback.close()
            self._retire(playback)
        with self._render_lock:
            # Invalidate any tap already captured by the retired native pull
            # thread while retaining the reference for audio it already read.
            self._render_generation += 1
        self._output_normalizer = self._output_format = self._source_rate = None
        self._tail.stop()
        self._echo_blocked = False
        if had_output and self.listening_during_playback and not self._closed:
            # A speaker's acoustic tail can outlive sink.stop(). Continue echo
            # cancellation through that tail, preserving near-end speech.
            self._tail.start(200)
        if self._playing:
            self._playing = False
            self.playback_stopped.emit()

    @Slot()
    def close(self):
        self._closed = True
        self.stop_capture()
        self.stop_playback()
        previous, self._duplex = self._duplex, None
        if previous is not None:
            previous.close()
        self._clear_render()
