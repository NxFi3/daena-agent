"""Optional Voice Mode bridge; ordinary agent runs never depend on speech."""

from __future__ import annotations

from collections import deque
import json
import math
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
        perception_factory=None,
    ):
        super().__init__(window)
        self.window = window
        self.dialog = None
        self.audio = None
        self.session = None
        self._audio_factory = audio_factory
        self._session_factory = session_factory
        self._provider_factory = provider_factory
        self._perception_factory = perception_factory
        self._cached_providers = None
        self._cached_perception = None
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
        self._acoustic = deque(maxlen=32)
        self._review_utterance_id = None
        self._turn_started = None
        self._reply_started = None
        self._pending_transcript = None
        self._perception_failed = False
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
            for language in ("fa", "en"):
                field = f"tts_{language}_model"
                if not settings.get(field):
                    settings[field] = paths.get(language, "")
            self._installed_settings(settings)
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

    @staticmethod
    def _installed_settings(settings):
        """Offer only explicitly installed files; opening Voice never downloads."""
        models = installed_models()
        for key, field in (
            ("kokoro-en", "kokoro_model_path"),
            ("speaker-embedding", "speaker_model_path"),
            ("audio-events", "sound_model_path"),
        ):
            if key in models and not settings.get(field):
                settings[field] = models[key]["path"]
        if settings.get("kokoro_model_path"):
            if not settings.get("kokoro_voices_path"):
                settings["kokoro_voices_path"] = str(
                    Path(settings["kokoro_model_path"]).parent / "voices-v1.0.bin"
                )
        if settings.get("sound_model_path"):
            if not settings.get("sound_labels_path"):
                settings["sound_labels_path"] = str(
                    Path(settings["sound_model_path"]).parent
                    / "class_labels_indices.csv"
                )

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
                "english_engine",
                "kokoro_model_path",
                "kokoro_voices_path",
                "kokoro_voice",
                "kokoro_cpu_threads",
                "speech_speed",
                "stt_beam_size",
                "stt_cpu_threads",
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
        english_engine = settings.get("english_engine", "piper")
        if english_engine not in {"piper", "kokoro"}:
            raise ValueError("Choose Piper or Kokoro as the English voice engine")
        speed = float(settings.get("speech_speed", 1.0))
        if not 0.7 <= speed <= 1.4:
            raise ValueError("Speech speed must be between 0.7 and 1.4")
        for lang in required:
            if lang == "en" and english_engine == "kokoro":
                for field in ("kokoro_model_path", "kokoro_voices_path"):
                    if not Path(settings.get(field) or "").expanduser().is_file():
                        raise ValueError(
                            "Select/download the local Kokoro model and matching voices archive"
                        )
                continue
            path = Path(paths.get(lang) or "").expanduser()
            if not path.is_file() or not Path(str(path) + ".json").is_file():
                raise ValueError(
                    f"Select/download the {lang} Piper voice (.onnx plus .onnx.json). Auto language needs both voices."
                )
        piper_paths = (
            {"fa": paths.get("fa", "")}
            if english_engine == "kokoro" and "en" in required
            else paths
        )
        piper_language = (
            "fa" if english_engine == "kokoro" and "en" in required else language
        )
        piper = PiperProvider(
            piper_paths, language=piper_language, length_scale=1.0 / speed
        )
        synthesizer = piper
        if english_engine == "kokoro" and "en" in required:
            from src.voice.kokoro import KokoroProvider, BilingualSynthesizer

            english = KokoroProvider(
                settings["kokoro_model_path"],
                settings["kokoro_voices_path"],
                voice=settings.get("kokoro_voice", "af_heart"),
                speed=speed,
                cpu_threads=settings.get("kokoro_cpu_threads", 2),
            )
            synthesizer = BilingualSynthesizer(
                english=english,
                persian=piper if "fa" in required else None,
                language=language,
            )
        providers = (
            FasterWhisperProvider(
                selected,
                device=settings.get("stt_device", "cpu"),
                compute_type=settings.get("stt_compute_type", "int8"),
                language=language,
                local_files_only=True,
                cpu_threads=settings.get("stt_cpu_threads", 4),
                beam_size=settings.get("stt_beam_size", 3),
            ),
            synthesizer,
        )
        self._cached_providers = signature, providers
        return providers

    def _perception(self, settings):
        if not settings.get("perception_enabled", False):
            return None
        signature = tuple(
            str(settings.get(key, ""))
            for key in (
                "speaker_model_path",
                "sound_model_path",
                "sound_labels_path",
                "sound_events_enabled",
                "speaker_threshold",
                "sound_threshold",
            )
        )
        if self._cached_perception and self._cached_perception[0] == signature:
            return self._cached_perception[1]
        if self._perception_factory:
            perception = self._perception_factory(settings)
        else:
            from src.voice.perception import LocalAudioPerception

            speaker = settings.get("speaker_model_path", "")
            sound = settings.get("sound_model_path", "")
            labels = settings.get("sound_labels_path", "")
            if not speaker and not sound:
                raise ValueError(
                    "Select/download a local speaker or sound-event model before enabling audio perception"
                )
            for path in (speaker, sound, labels if sound else ""):
                if path and not Path(path).expanduser().is_file():
                    raise ValueError(
                        "Audio perception model/labels file is missing; select an installed local file"
                    )
            if sound and not labels:
                raise ValueError(
                    "The sound-event model needs its matching class labels CSV"
                )
            perception = LocalAudioPerception(
                speaker_model_path=speaker,
                sound_model_path=sound,
                sound_labels_path=labels,
                num_threads=2,
                speaker_threshold=settings.get("speaker_threshold", 0.88),
                sound_threshold=settings.get("sound_threshold", 0.25),
            )
        self._cached_perception = signature, perception
        return perception

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
            perception = self._perception(settings)
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
            self._turn_started = self._reply_started = None
            self._perception_failed = False
            self._acoustic.clear()
            self._review_utterance_id = None
            self.dialog.set_perception({})
            self.session = self._session_factory(
                recognizer,
                synthesizer,
                silence_ms=settings.get("silence_ms", 550),
                max_seconds=settings.get("max_utterance_seconds", 30),
                perception=perception,
                sound_events_enabled=bool(settings.get("sound_events_enabled", False)),
            )
            self.dialog.set_active(True)
            if hasattr(self.window, "voice_button"):
                self.window.voice_button.setText("◉ Live")
            self.dialog.set_state(
                "Preparing local models · speech input starts when recognition is ready"
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
            self.reset_input()
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
        self._review_utterance_id = None
        self._pending_transcript = None
        if self.dialog:
            self.dialog.set_transcript("")

    def begin_turn(self):
        self.acknowledge_submission()
        # Called synchronously when Studio starts an actual new agent turn.
        # Queued run_start events from a canceled turn cannot re-enable speech.
        self._speech_suppressed = False
        self._spoken.clear()
        self._turn_started = time.monotonic()
        self._reply_started = None
        for key in ("agent_ms", "first_pcm_ms", "first_playback_ms"):
            self._metrics.pop(key, None)

    def reset_input(self):
        reset = getattr(self.session, "reset_input", None)
        if callable(reset):
            reset()
        elif self.session:
            self.session.segmenter.reset()

    def playback_started(self):
        if self.session:
            self.reset_input()
        if self._turn_started is not None and "first_playback_ms" not in self._metrics:
            self._metrics["first_playback_ms"] = round(
                (time.monotonic() - self._turn_started) * 1000
            )
        if self.dialog:
            self.dialog.set_state("Speaking · microphone paused to avoid echo")

    def cancel_speech(self):
        self._speech_suppressed = True
        if self.session:
            self.session.interrupt()
        self._pending_pcm = None
        self._acoustic.clear()
        self._review_utterance_id = None
        self._pending_transcript = None
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
        self._acoustic.clear()
        self._review_utterance_id = None
        self._pending_transcript = None
        if hasattr(self.window, "voice_button"):
            self.window.voice_button.setText("◉ Voice")
        if self.dialog:
            self.dialog.set_active(False)
            self.dialog.set_perception({})
            self.dialog.set_state("Voice is off · microphone is not recording")

    def send_review(self, text):
        text = str(text).strip()
        if not text:
            return
        if self._pending_transcript:
            self._review_utterance_id = self._pending_transcript[0].get("utterance_id")
            self._pending_transcript = None
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
            self._review_inserted = (text, "")
        if self._review_inserted:
            self._review_inserted = (
                self.window.input.toPlainText().strip(),
                self._review_inserted[1],
            )
        if not self.window.busy:
            self.window.send()
            self.dialog.set_transcript("")
            self._review_inserted = None
        else:
            self.dialog.set_state(
                "Agent busy · transcript is ready in the chat composer"
            )

    def _deliver(self, text, *, auto=False, truncated=False, utterance_id=None):
        if not text:
            return
        current = self.window.input.toPlainText().strip()
        # A combined draft can contain multiple speakers or ordinary typing;
        # do not attribute the entire combined request to its last utterance.
        self._review_utterance_id = None if current else utterance_id
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

    def _has_speech_evidence(self, utterance_id):
        return utterance_id is not None and any(
            data.get("source") == "speech" and data.get("utterance_id") == utterance_id
            for _, data in self._acoustic
        )

    def _deliver_transcript(self, data, *, force_review=False):
        self._deliver(
            data["text"],
            auto=self._settings.get("auto_send", False) and not force_review,
            truncated=data.get("truncated", False),
            utterance_id=data.get("utterance_id"),
        )

    def _flush_transcript(self):
        if not self._pending_transcript or not self.session:
            return
        data, epoch, deadline = self._pending_transcript
        if epoch != self.session.epoch:
            self._pending_transcript = None
        elif (
            self._has_speech_evidence(data.get("utterance_id"))
            or self._perception_failed
            or time.monotonic() >= deadline
        ):
            self._pending_transcript = None
            self._deliver_transcript(data)

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
        if self._reply_started is None:
            self._reply_started = time.monotonic()
            if self._turn_started is not None:
                self._metrics["agent_ms"] = round(
                    (self._reply_started - self._turn_started) * 1000
                )
        self.session.speak(text, self._settings.get("language", "auto"))

    def context_for_prompt(self, prompt):
        """Bounded acoustic observations for a reviewed/recognized request only.

        Never attach an unrelated room sound or old speaker to ordinary typing.
        The agent receives model evidence, not the original microphone recording.
        """
        if (
            not self.session
            or not self._review_inserted
            or not self._settings.get("perception_enabled", False)
        ):
            return None
        if str(prompt).strip() != self._review_inserted[0]:
            return None
        now = time.monotonic()
        speech, ambient = [], []
        for received, data in self._acoustic:
            if data.get("source") == "speech":
                if now - received > 120:
                    continue
                if (
                    self._review_utterance_id is None
                    or data.get("utterance_id") != self._review_utterance_id
                ):
                    continue
                speech.append(data)
            elif data.get("source") != "ambient":
                continue
            elif now - received > 5:
                continue
            else:
                ambient.append(data)
        # Continuous room observations must never displace the exact speech
        # receipt associated with the reviewed request.
        observations = ambient[-2:] + speech[-1:] if speech else ambient[-3:]
        if not observations:
            return None
        # Serialization strips Qt/native objects and caps supplemental context.
        from src.voice.observations import compact_observations

        return compact_observations(observations[-3:])

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
                    if self._pending_transcript:
                        previous = self._pending_transcript[0]
                        self._pending_transcript = None
                        self._deliver_transcript(previous, force_review=True)
                    self.dialog.set_transcript(data["text"])
                    self._metrics["stt_ms"] = data["stt_ms"]
                    if (
                        self._settings.get("auto_send", False)
                        and self._settings.get("perception_enabled", False)
                        and not self._perception_failed
                        and data.get("utterance_id") is not None
                        and not data.get("truncated", False)
                        and not self.window.busy
                        and not self.window.input.toPlainText().strip()
                        and not self._has_speech_evidence(data["utterance_id"])
                    ):
                        # Give independent local perception a small, bounded
                        # chance to attach evidence. Never wait on native work.
                        self._pending_transcript = (
                            data,
                            event.epoch,
                            time.monotonic() + 0.4,
                        )
                        self.dialog.set_state(
                            "Recognized · waiting briefly for acoustic observations"
                        )
                    else:
                        self._deliver_transcript(data)
                elif event.kind == "error":
                    self.stop()
                    self.error(event.data)
                    return
                elif event.kind == "notice":
                    self.error(event.data)
                elif event.kind == "perception":
                    self._perception_failed = False
                    received = time.monotonic()
                    captured = event.data.get("captured_at")
                    if (
                        isinstance(captured, bool)
                        or not isinstance(captured, (float, int))
                        or not math.isfinite(captured)
                        or captured < 0
                        or captured > received
                    ):
                        # Compatibility with older local adapters: include their
                        # queue/inference time rather than making delayed audio
                        # appear newly captured when the GUI polls its receipt.
                        delay = event.data.get("analyzed_ms", 0)
                        if (
                            isinstance(delay, bool)
                            or not isinstance(delay, (float, int))
                            or not math.isfinite(delay)
                            or delay < 0
                        ):
                            delay = 0
                        captured = received - delay / 1000
                    self._acoustic.append((captured, event.data))
                    self.dialog.set_perception(event.data)
                elif event.kind == "perception_ready":
                    self._perception_failed = False
                elif event.kind == "perception_error":
                    self._perception_failed = True
                    self.error(
                        "Audio perception unavailable; speech continues: "
                        + str(event.data)
                    )
                elif event.kind in {"state", "ready"}:
                    if self._review_inserted or self._pending_transcript:
                        continue
                    self.dialog.set_state(
                        event.data
                        if event.kind == "state"
                        else (
                            "Listening · recognition ready"
                            if getattr(self.session, "recognition_ready", None)
                            is not None
                            and self.session.recognition_ready.is_set()
                            else "Preparing speech recognition · microphone input is not submitted yet"
                        )
                    )
                elif event.kind == "latency":
                    self._metrics.update(event.data)
            self._flush_transcript()
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
                if data["key"] == "kokoro-en":
                    values["kokoro_model_path"] = data["path"]
                    values["kokoro_voices_path"] = str(
                        Path(data["path"]).parent / "voices-v1.0.bin"
                    )
                if data["key"] == "speaker-embedding":
                    values["speaker_model_path"] = data["path"]
                if data["key"] == "audio-events":
                    values["sound_model_path"] = data["path"]
                    values["sound_labels_path"] = str(
                        Path(data["path"]).parent / "class_labels_indices.csv"
                    )
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
            stage = {
                "resuming": "Resuming",
                "retrying": "Retrying · partial saved",
                "cached": "Using verified cache",
                "connecting": "Connecting",
            }.get(data.get("stage"), "Downloading")
            report(
                "progress",
                f"{stage} {data.get('file', 'model')}: {received:.1f}{suffix}",
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
        self._cached_perception = None

    def close(self):
        self._closed = True
        self._download_stop.set()
        self.release_models()
        self.timer.stop()
