"""Optional Voice Mode bridge; ordinary agent runs never depend on speech."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from queue import Queue, Empty, Full
from threading import Event, Thread
import time

from PySide6.QtCore import QObject, QTimer

from src.studio.state import write_json
from src.voice.models import (
    default_cache_dir,
    download_model,
    installed_models,
    voice_paths,
)
from src.voice.providers import FasterWhisperProvider, PiperProvider
from src.voice.session import VoiceSession


class VoiceController(QObject):
    def __init__(
        self,
        window,
        *,
        audio_factory=None,
        session_factory=VoiceSession,
        provider_factory=None,
    ):
        super().__init__(window)
        self.window = window
        self.dialog = None
        self.audio = None
        self.session = None
        self._audio_factory = audio_factory
        self._session_factory = session_factory
        self._provider_factory = provider_factory
        self._cached_providers = None
        self._retiring = []
        self._pending_pcm = None
        self._spoken = deque(maxlen=64)
        self._context = None
        self._download = None
        self._download_stop = Event()
        self._download_events = Queue(maxsize=32)
        self._closed = False
        self._settings = {}
        self._metrics = {}
        self._review_inserted = None
        self._speech_suppressed = False
        self.timer = QTimer(self)
        self.timer.setInterval(25)
        self.timer.timeout.connect(self.poll)
        self.timer.start()

    def _signature(self):
        return str(self.window.workspace), str(self.window.session_id)

    def _settings_path(self):
        return self.window.store.root / "voice.json"

    def open(self):
        from src.studio.voice_audio import AudioDevices
        from src.studio.voice_dialog import VoiceDialog

        if self._closed:
            return
        if self.dialog is None:
            try:
                settings = json.loads(self._settings_path().read_text())
                if not isinstance(settings, dict):
                    settings = {}
            except (OSError, ValueError):
                settings = {}
            paths = voice_paths()
            settings.setdefault("tts_fa_model", paths.get("fa", ""))
            settings.setdefault("tts_en_model", paths.get("en", ""))
            self.dialog = VoiceDialog(settings, self.window)
            self.dialog.start_requested.connect(self.start)
            self.dialog.stop_requested.connect(self.stop)
            self.dialog.mute_changed.connect(self.mute)
            self.dialog.interrupt_requested.connect(self.interrupt)
            self.dialog.end_utterance_requested.connect(self.finish_utterance)
            self.dialog.send_requested.connect(self.send_review)
            self.dialog.download_requested.connect(self.download)
            self.dialog.settings_changed.connect(self.save_settings)
            self.dialog.finished.connect(self.release_models)
        try:
            devices = AudioDevices.devices()
            self.dialog.set_devices(devices)
            if devices.get("error"):
                self.error(devices["error"])
        except Exception as exc:
            self.dialog.show_error(f"Audio devices unavailable: {exc}")
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def save_settings(self, settings):
        if self.session:
            for key in ("auto_send", "speak_progress"):
                if key in settings:
                    self._settings[key] = bool(settings[key])
        try:
            write_json(self._settings_path(), settings)
        except (OSError, ValueError) as exc:
            self.error(f"Could not save voice settings: {exc}")

    def error(self, message):
        if self.dialog:
            self.dialog.show_error(str(message))

    def _providers(self, settings):
        if self._provider_factory:
            return self._provider_factory(settings)
        signature = tuple(
            (key, str(settings.get(key, "")))
            for key in (
                "stt_model",
                "stt_device",
                "stt_compute_type",
                "language",
                "tts_fa_model",
                "tts_en_model",
            )
        )
        if self._cached_providers and self._cached_providers[0] == signature:
            return self._cached_providers[1]
        models = installed_models()
        selected = settings.get("stt_model", "large-v3-turbo")
        key = {"large-v3-turbo": "whisper-turbo", "small": "whisper-small"}.get(
            selected
        )
        if key:
            if key not in models:
                raise ValueError(
                    "Download the selected speech model first, or choose its local directory. No model downloads happen on Start."
                )
            selected = models[key]["path"]
        elif not Path(selected).expanduser().is_dir():
            raise ValueError(
                "Select an installed local Whisper directory or download a catalog model."
            )
        language = settings.get("language", "auto")
        paths = {lang: settings.get(f"tts_{lang}_model", "") for lang in ("fa", "en")}
        required = ("fa", "en") if language == "auto" else (language,)
        for lang in required:
            path = Path(paths.get(lang) or "").expanduser()
            if not path.is_file() or not Path(str(path) + ".json").is_file():
                raise ValueError(
                    f"Select/download the {lang} Piper voice (.onnx plus .onnx.json). Auto language needs both voices."
                )
        providers = (
            FasterWhisperProvider(
                selected,
                device=settings.get("stt_device", "cpu"),
                compute_type=settings.get("stt_compute_type", "int8"),
                language=language,
                local_files_only=True,
                cpu_threads=4,
            ),
            PiperProvider(paths, language=language),
        )
        self._cached_providers = signature, providers
        return providers

    def start(self, settings):
        if self.session or self._closed:
            return
        self._retiring = [s for s in self._retiring if not s.retired]
        if self._retiring:
            self.error(
                "A canceled native inference call is still finishing. Wait briefly before restarting; no second model is loaded."
            )
            return
        if getattr(self.window, "_automation_run", False):
            self.error("Stop automation before starting conversational Voice Mode.")
            return
        try:
            recognizer, synthesizer = self._providers(settings)
            if self._audio_factory is None:
                from src.studio.voice_audio import AudioDevices

                self.audio = AudioDevices(self)
            else:
                self.audio = self._audio_factory(self)
            self.audio.pcm_ready.connect(self.feed)
            self.audio.level.connect(self.dialog.set_level)
            self.audio.fault.connect(self.audio_fault)
            self.audio.playback_started.connect(self.playback_started)
            self.audio.playback_stopped.connect(
                lambda: self.dialog.set_state("Listening") if self.session else None
            )
            self.audio.set_output_device(settings.get("output_device_id", ""))
            audio = self.audio
            if not audio.start_capture(settings.get("input_device_id", "")):
                if self.audio is audio:
                    audio.close()
                    audio.deleteLater()
                    self.audio = None
                return
            if self.audio is not audio:
                return
            self._settings = dict(settings)
            self._context = self._signature()
            self._spoken.clear()
            self._speech_suppressed = False
            self._metrics.clear()
            self.session = self._session_factory(
                recognizer,
                synthesizer,
                silence_ms=settings.get("silence_ms", 550),
                max_seconds=settings.get("max_utterance_seconds", 30),
            )
            self.dialog.set_active(True)
            if hasattr(self.window, "voice_button"):
                self.window.voice_button.setText("◉ Live")
            self.dialog.set_state(
                "Loading local models · microphone active; review recognized text before sending"
            )
            self.save_settings(settings)
        except Exception as exc:
            self.stop()
            self.error(f"Voice could not start: {type(exc).__name__}: {exc}")

    def audio_fault(self, message):
        self.stop()
        self.error(message)

    def feed(self, pcm16):
        if self.session:
            self.session.feed(pcm16)

    def mute(self, muted):
        if self.audio:
            self.audio.set_muted(muted)
        if self.session:
            self.session.segmenter.reset()
        if self.dialog:
            self.dialog.set_state("Microphone muted" if muted else "Listening")

    def finish_utterance(self):
        if self.audio:
            flush = getattr(self.audio, "finish_capture", None)
            if callable(flush):
                flush()
        if self.session:
            self.session.finish_utterance()

    def acknowledge_submission(self):
        self._review_inserted = None
        if self.dialog:
            self.dialog.set_transcript("")

    def begin_turn(self):
        self.acknowledge_submission()
        # Called synchronously when Studio starts an actual new agent turn.
        # Queued run_start events from a canceled turn cannot re-enable speech.
        self._speech_suppressed = False
        self._spoken.clear()

    def playback_started(self):
        if self.session:
            self.session.segmenter.reset()
        if self.dialog:
            self.dialog.set_state("Speaking · microphone paused to avoid echo")

    def cancel_speech(self):
        self._speech_suppressed = True
        if self.session:
            self.session.interrupt()
        self._pending_pcm = None
        if self.audio:
            self.audio.stop_playback()
        if self.dialog:
            self.dialog.set_state(
                "Interrupted · listening; unfinished agent work is stopping"
            )

    def interrupt(self):
        if self.window.busy and not getattr(self.window, "_automation_run", False):
            self.window.stop()
        else:
            self.cancel_speech()

    def stop(self):
        self._download_stop.set()
        session, self.session = self.session, None
        if session:
            session.close()
            self._retiring.append(session)
        if self.audio:
            self.audio.close()
            self.audio.deleteLater()
            self.audio = None
        self._pending_pcm = None
        if hasattr(self.window, "voice_button"):
            self.window.voice_button.setText("◉ Voice")
        if self.dialog:
            self.dialog.set_active(False)
            self.dialog.set_state("Voice is off · microphone is not recording")

    def send_review(self, text):
        text = str(text).strip()
        if not text:
            return
        current = self.window.input.toPlainText().strip()
        if self._review_inserted and current == self._review_inserted[0]:
            base = self._review_inserted[1]
            self.window.input.setPlainText(base + "\n" + text if base else text)
        elif current and current != text:
            self.error(
                "The chat composer has changed. Review and send it there to preserve your draft."
            )
            return
        else:
            self.window.input.setPlainText(text)
        if not self.window.busy:
            self.window.send()
            self.dialog.set_transcript("")
            self._review_inserted = None
        else:
            self.dialog.set_state(
                "Agent busy · transcript is ready in the chat composer"
            )

    def _deliver(self, text, *, auto=False, truncated=False):
        if not text:
            return
        current = self.window.input.toPlainText().strip()
        inserted = current + "\n" + text if current else text
        self.window.input.setPlainText(inserted)
        self._review_inserted = (inserted, current)
        if auto and not truncated and not current and not self.window.busy:
            self.window.send()
            self.dialog.set_transcript("")
            self._review_inserted = None
        elif self.dialog:
            self.dialog.set_state(
                "Review the transcript in the composer and press Enter."
                + (
                    " Recording limit reached; this was not sent automatically."
                    if truncated
                    else ""
                )
            )

    def on_event(self, event):
        if not self.session or getattr(self.window, "_automation_run", False):
            return
        kind = event.get("type")
        if self._speech_suppressed:
            return
        if kind != "final_response" and not (
            kind == "assistant_message" and self._settings.get("speak_progress", False)
        ):
            return
        text = str(event.get("text") or "").strip()
        if not text or text in self._spoken:
            return
        self._spoken.append(text)
        self.session.speak(text, self._settings.get("language", "auto"))

    def on_result(self, result):
        # A stopped/failed turn is not silently narrated as a successful reply.
        if (result.get("metrics") or {}).get("completed"):
            self.on_event({"type": "final_response", "text": result.get("text", "")})

    def poll(self):
        self._retiring = [s for s in self._retiring if not s.retired]
        if self.session:
            if self._signature() != self._context:
                self.stop()
                return
            for _ in range(24):
                try:
                    event = self.session.events.get_nowait()
                except Empty:
                    break
                if event.epoch != self.session.epoch:
                    continue
                if event.kind == "transcript":
                    data = event.data
                    self.dialog.set_transcript(data["text"])
                    self._metrics["stt_ms"] = data["stt_ms"]
                    self._deliver(
                        data["text"],
                        auto=self._settings.get("auto_send", False),
                        truncated=data.get("truncated", False),
                    )
                elif event.kind == "error":
                    self.stop()
                    self.error(event.data)
                    return
                elif event.kind == "notice":
                    self.error(event.data)
                elif event.kind in {"state", "ready"}:
                    if self._review_inserted:
                        continue
                    self.dialog.set_state(
                        event.data
                        if event.kind == "state"
                        else "Local model ready · listening"
                    )
                elif event.kind == "latency":
                    self._metrics.update(event.data)
            for _ in range(4):
                if not self.session or not self.audio:
                    break
                if self._pending_pcm is None:
                    try:
                        self._pending_pcm = self.session.audio.get_nowait()
                    except Empty:
                        break
                epoch, chunk = self._pending_pcm
                if epoch != self.session.epoch:
                    self._pending_pcm = None
                    continue
                if chunk is None:
                    self.audio.finish_playback()
                    self._pending_pcm = None
                    continue
                if not self.audio.can_accept(chunk.pcm16, chunk.sample_rate):
                    break
                if not self.audio.play_pcm(chunk.pcm16, chunk.sample_rate):
                    break
                self._pending_pcm = None
            if self.audio:
                self._metrics["queued_seconds"] = round(self.audio.queued_seconds, 2)
            self.dialog.set_metrics(self._metrics)
        for _ in range(8):
            try:
                kind, data = self._download_events.get_nowait()
            except Empty:
                break
            if kind == "error":
                self.error(data)
            elif kind == "done" and self.dialog:
                values = self.dialog.values()
                if data["key"] == "piper-fa":
                    values["tts_fa_model"] = data["path"]
                if data["key"] == "piper-en":
                    values["tts_en_model"] = data["path"]
                self.dialog.populate(values)
                self.save_settings(values)
                self.dialog.set_state("Model downloaded · choose Start when ready")
            elif self.dialog:
                self.dialog.set_state(str(data))

    def download(self, key):
        if self._download and self._download.is_alive():
            self.error("A model download is already running. Closing Voice cancels it.")
            return
        if self.session:
            self.error("Stop Voice before changing model files.")
            return
        self._download_stop = Event()
        cancel = self._download_stop

        def report(kind, data):
            try:
                self._download_events.put_nowait((kind, data))
            except Full:
                if kind != "progress":
                    try:
                        self._download_events.get_nowait()
                        self._download_events.put_nowait((kind, data))
                    except (Empty, Full):
                        pass

        last_progress = [0.0]

        def progress(data):
            now = time.monotonic()
            if now - last_progress[0] < 0.2:
                return
            last_progress[0] = now
            received = data.get("received", 0) / 1048576
            total = data.get("total")
            suffix = (
                f" / {total / 1048576:.1f} MiB"
                if isinstance(total, (int, float))
                else " MiB"
            )
            report(
                "progress",
                f"Downloading {data.get('file', 'model')}: {received:.1f}{suffix}",
            )

        def run():
            try:
                record = download_model(key, stop_event=cancel, progress=progress)
                if not cancel.is_set():
                    report("done", record)
            except Exception as exc:
                if not cancel.is_set():
                    report(
                        "error",
                        f"Model download failed: {exc}. You can select models downloaded elsewhere.",
                    )

        self._download = Thread(target=run, daemon=True, name="daena-voice-download")
        self._download.start()
        if self.dialog:
            self.dialog.set_state(
                "Downloading model explicitly · first download may be large; no API fees"
            )

    def release_models(self, *args):
        self.stop()
        # Retiring workers retain references until native work ends.
        self._cached_providers = None

    def close(self):
        self._closed = True
        self._download_stop.set()
        self.release_models()
        self.timer.stop()
