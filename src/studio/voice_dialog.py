"""Local voice controls. No audio device, model or network is opened by this UI."""

from __future__ import annotations

from math import ceil
from src.studio.widgets import Composer

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QLayout,
    QScrollArea,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class VoiceMeter(QWidget):
    """Small amplitude history; displays energy rather than pretending to show text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = [0.0] * 38
        self.setMinimumHeight(66)
        self.setAccessibleName("Microphone level")
        self.setToolTip("Microphone signal level")

    def set_level(self, level):
        self.values = [*self.values[1:], min(1.0, max(0.0, float(level)) * 4)]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width = max(1.0, (self.width() - 20) / len(self.values))
        for i, value in enumerate(self.values):
            height = max(3, ceil(value * (self.height() - 12)))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#d97757" if value > 0.03 else "#777570"))
            painter.drawRoundedRect(
                QRectF(
                    10 + i * width,
                    (self.height() - height) / 2,
                    max(2.0, width - 3),
                    height,
                ),
                2,
                2,
            )


class VoiceEditor(Composer):
    """Enter performs the selected voice action; Shift+Enter keeps a new line."""

    def _resize(self, *args):
        self.setFixedHeight(
            max(95, min(180, int(self.document().size().height()) + 19))
        )


class VoiceDialog(QDialog):
    start_requested = Signal(dict)
    stop_requested = Signal()
    mute_changed = Signal(bool)
    interrupt_requested = Signal()
    end_utterance_requested = Signal()
    send_requested = Signal(str)
    speak_requested = Signal(str)
    download_requested = Signal(str)
    settings_changed = Signal(dict)
    apply_requested = Signal(dict)
    open_requested = Signal()
    test_output_requested = Signal()
    refresh_requested = Signal()
    state_changed = Signal(str)
    active_changed = Signal(bool)
    level_changed = Signal(float)
    transcript_changed = Signal(str)
    metrics_changed = Signal(dict)
    error_raised = Signal(str)

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Daena Voice")
        self.resize(710, 690)
        self.setMinimumSize(550, 560)
        self._active = False
        self._configuration = None
        self._populating = True
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 22, 24, 20)
        outer.setSpacing(12)
        title = QLabel("Talk with Daena")
        title.setObjectName("pageTitle")
        outer.addWidget(title)
        subtitle = QLabel("Local speech · فارسی / English · no paid API required")
        subtitle.setObjectName("muted")
        outer.addWidget(subtitle)
        self.meter = VoiceMeter()
        outer.addWidget(self.meter)
        self.status = QLabel("Voice is off · microphone is not recording")
        self.status.setTextFormat(Qt.PlainText)
        self.status.setWordWrap(True)
        self.status.setObjectName("muted")
        outer.addWidget(self.status)
        self.metrics = QLabel("Latency appears after recognition and playback begin")
        self.metrics.setObjectName("muted")
        self.metrics.setWordWrap(True)
        outer.addWidget(self.metrics)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        conversation = QWidget()
        chat = QVBoxLayout(conversation)
        chat.setContentsMargins(14, 16, 14, 12)
        chat.setSpacing(10)
        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("Input"))
        self.input_mode = QComboBox()
        self.input_mode.addItem("Microphone · speech recognition", "microphone")
        self.input_mode.addItem("Text only · no Whisper or microphone", "text")
        input_row.addWidget(self.input_mode, 1)
        chat.addLayout(input_row)
        self.transcript_title = QLabel("Recognized speech")
        self.transcript_title.setObjectName("sectionTitle")
        chat.addWidget(self.transcript_title)
        self.transcript = VoiceEditor()
        self.transcript.setPlaceholderText(
            "Your words appear here. Review or edit them before sending."
        )
        self.transcript.setMinimumHeight(95)
        chat.addWidget(self.transcript, 1)
        review_row = QHBoxLayout()
        self.finish_button = QPushButton("Finish speaking")
        self.finish_button.setToolTip(
            "End this utterance now instead of waiting for silence"
        )
        self.finish_button.clicked.connect(self.end_utterance_requested)
        review_row.addWidget(self.finish_button)
        self.speak_button = QPushButton("Speak text")
        self.speak_button.setToolTip(
            "Read this text directly using the selected local voice; does not call the Agent"
        )
        self.speak_button.clicked.connect(self._speak_text)
        review_row.addWidget(self.speak_button)
        review_row.addStretch()
        self.send_button = QPushButton("Send to Daena")
        self.send_button.setObjectName("primary")
        self.send_button.setStyleSheet(
            "QPushButton:disabled {background: transparent; color: #8f8c84; border: 1px solid #6b6860;}"
        )
        self.send_button.clicked.connect(self._send_transcript)
        review_row.addWidget(self.send_button)
        chat.addLayout(review_row)
        self.auto_send = QCheckBox(
            "Hands-free conversation · send recognized speech automatically"
        )
        self.auto_send.setToolTip(
            "Send complete utterances automatically using the same Agent and tool permissions as typed chat. Turn this off to review every transcript before sending."
        )
        chat.addWidget(self.auto_send)
        self.speak_progress = QCheckBox("Read progress updates aloud")
        self.speak_progress.setToolTip(
            "Only committed assistant messages; model reasoning is never spoken"
        )
        chat.addWidget(self.speak_progress)
        self.perception_summary = QLabel(
            "Audio perception is optional · configure it in Perception"
        )
        self.perception_summary.setTextFormat(Qt.PlainText)
        self.perception_summary.setObjectName("muted")
        self.perception_summary.setWordWrap(True)
        chat.addWidget(self.perception_summary)
        self.input_note = QLabel()
        self.input_note.setObjectName("muted")
        self.input_note.setWordWrap(True)
        chat.addWidget(self.input_note)
        self.tabs.addTab(conversation, "Conversation")
        self.model_page = QWidget()
        form = QFormLayout(self.model_page)
        form.setContentsMargins(14, 16, 14, 12)
        form.setVerticalSpacing(11)
        self.language = QComboBox()
        for label, value in [
            ("Automatic · فارسی / English", "auto"),
            ("فارسی", "fa"),
            ("English", "en"),
        ]:
            self.language.addItem(label, value)
        form.addRow("Language", self.language)
        self.stt_model = QComboBox()
        self.stt_model.setEditable(True)
        self.stt_model.addItem(
            "Quality first · large-v3-turbo (~1.6 GB)", "large-v3-turbo"
        )
        self.stt_model.addItem("Faster CPU · small (~0.5 GB)", "small")
        form.addRow("Speech recognition", self.stt_model)
        self.stt_download = QPushButton("Download selected speech model")
        self.stt_download.clicked.connect(self._download_speech)
        form.addRow("", self.stt_download)
        row = QHBoxLayout()
        self.stt_device = QComboBox()
        self.stt_device.addItems(["cpu", "cuda", "auto"])
        self.stt_compute_type = QComboBox()
        self.stt_compute_type.addItems(
            ["int8", "float16", "int8_float16", "float32", "auto"]
        )
        row.addWidget(self.stt_device)
        row.addWidget(self.stt_compute_type)
        form.addRow("Device / precision", row)
        recognition_tuning = QHBoxLayout()
        self.stt_cpu_threads = QSpinBox()
        self.stt_cpu_threads.setRange(1, 32)
        self.stt_cpu_threads.setToolTip("CPU threads used by local speech recognition")
        self.stt_beam_size = QSpinBox()
        self.stt_beam_size.setRange(1, 5)
        self.stt_beam_size.setToolTip(
            "1 decodes faster; a wider beam may improve recognition but takes longer"
        )
        recognition_tuning.addWidget(self.stt_cpu_threads)
        recognition_tuning.addWidget(self.stt_beam_size)
        form.addRow("CPU threads / beam", recognition_tuning)
        self.tts_fa_model = QLineEdit()
        self.tts_fa_model.setPlaceholderText("Local Persian .onnx model")
        form.addRow("Persian voice", self._model_picker(self.tts_fa_model))
        self.fa_download = QPushButton("Download Persian voice")
        self.fa_download.clicked.connect(
            lambda: self.download_requested.emit("piper-fa")
        )
        form.addRow("", self.fa_download)
        self.english_engine = QComboBox()
        self.english_engine.addItem("Piper · lightweight", "piper")
        self.english_engine.addItem("Kokoro · expressive English", "kokoro")
        form.addRow("English engine", self.english_engine)
        self.tts_en_model = QLineEdit()
        self.tts_en_model.setPlaceholderText("Local English .onnx model")
        form.addRow("English voice", self._model_picker(self.tts_en_model))
        self.en_download = QPushButton("Download English voice")
        self.en_download.clicked.connect(
            lambda: self.download_requested.emit("piper-en")
        )
        form.addRow("", self.en_download)
        self.kokoro_page = QWidget()
        kokoro = QFormLayout(self.kokoro_page)
        kokoro.setContentsMargins(0, 2, 0, 2)
        kokoro.setVerticalSpacing(9)
        self.kokoro_model_path = QLineEdit()
        self.kokoro_model_path.setPlaceholderText("Local Kokoro .onnx model")
        kokoro.addRow("Kokoro model", self._model_picker(self.kokoro_model_path))
        self.kokoro_voices_path = QLineEdit()
        self.kokoro_voices_path.setPlaceholderText("Local voices .bin file")
        kokoro.addRow(
            "Kokoro voices",
            self._model_picker(
                self.kokoro_voices_path, "Kokoro voices (*.bin);;All files (*)"
            ),
        )
        self.kokoro_voice = QComboBox()
        self.kokoro_voice.setEditable(True)
        for label, value in (
            ("Heart · American English", "af_heart"),
            ("Bella · American English", "af_bella"),
            ("Michael · American English", "am_michael"),
            ("Emma · British English", "bf_emma"),
            ("George · British English", "bm_george"),
        ):
            self.kokoro_voice.addItem(label, value)
        kokoro.addRow("Speaking voice", self.kokoro_voice)
        self.kokoro_cpu_threads = QSpinBox()
        self.kokoro_cpu_threads.setRange(1, 16)
        self.kokoro_cpu_threads.setToolTip(
            "Kokoro inference threads; more threads can compete with the Agent and slow responses"
        )
        kokoro.addRow("Kokoro CPU threads", self.kokoro_cpu_threads)
        self.kokoro_download = QPushButton("Download Kokoro English model + voices")
        self.kokoro_download.clicked.connect(
            lambda: self.download_requested.emit("kokoro-en")
        )
        kokoro.addRow("", self.kokoro_download)
        kokoro_note = QLabel(
            "Kokoro is used for English. Persian keeps its own Piper voice; Kokoro does not officially support Persian."
        )
        kokoro_note.setObjectName("muted")
        kokoro_note.setWordWrap(True)
        kokoro.addRow(kokoro_note)
        form.addRow(self.kokoro_page)
        self.speech_speed = QDoubleSpinBox()
        self.speech_speed.setRange(0.7, 1.4)
        self.speech_speed.setSingleStep(0.05)
        self.speech_speed.setDecimals(2)
        self.speech_speed.setSuffix(" ×")
        self.speech_speed.setToolTip("1.00 is the voice’s natural speaking speed")
        form.addRow("Speaking speed", self.speech_speed)
        download_note = QLabel(
            "Models are downloaded only when you select Download. Recognition and synthesis then run locally. Quality, RAM/VRAM use and first-response time depend on the selected model and your hardware."
        )
        download_note.setObjectName("muted")
        download_note.setWordWrap(True)
        form.addRow(download_note)
        form.setSizeConstraint(QLayout.SetMinimumSize)
        self.model_scroll = QScrollArea()
        self.model_scroll.setWidgetResizable(True)
        self.model_scroll.setFrameShape(QFrame.NoFrame)
        self.model_scroll.setWidget(self.model_page)
        self.tabs.addTab(self.model_scroll, "Models")
        self.audio_page = QWidget()
        audio = QFormLayout(self.audio_page)
        audio.setContentsMargins(14, 16, 14, 12)
        audio.setVerticalSpacing(12)
        self.input_device = QComboBox()
        self.output_device = QComboBox()
        self.input_device.addItem("System default microphone", "")
        self.output_device.addItem("System default speakers", "")
        audio.addRow("Microphone", self.input_device)
        audio.addRow("Speakers", self.output_device)
        self.duplex_mode = QComboBox()
        self.duplex_mode.addItem("Automatic · local echo cancellation", "auto")
        self.duplex_mode.addItem("Headphones · listen while speaking", "headphones")
        self.duplex_mode.addItem("Manual · pause microphone during replies", "off")
        self.duplex_mode.setToolTip(
            "Automatic uses the available local echo canceller; without one it pauses the microphone during playback. Headphones keeps the microphone open without echo cancellation and requires headphones to avoid hearing Daena's voice."
        )
        audio.addRow("Interrupt by speaking", self.duplex_mode)
        self.duplex_status = QLabel("Active turn-taking mode appears when Voice starts")
        self.duplex_status.setTextFormat(Qt.PlainText)
        self.duplex_status.setObjectName("muted")
        self.duplex_status.setWordWrap(True)
        audio.addRow(self.duplex_status)
        self.silence_ms = QSpinBox()
        self.silence_ms.setRange(200, 2000)
        self.silence_ms.setSingleStep(50)
        self.silence_ms.setSuffix(" ms")
        audio.addRow("End-of-speech silence", self.silence_ms)
        self.max_utterance_seconds = QSpinBox()
        self.max_utterance_seconds.setRange(5, 120)
        self.max_utterance_seconds.setSuffix(" seconds")
        audio.addRow("Maximum utterance", self.max_utterance_seconds)
        audio_note = QLabel(
            "Shorter silence reduces turn delay but may split natural pauses. Stop Voice releases the microphone and clears queued playback. Audio recordings are not saved by this dialog."
        )
        audio_note.setObjectName("muted")
        audio_note.setWordWrap(True)
        audio.addRow(audio_note)
        audio.setSizeConstraint(QLayout.SetMinimumSize)
        self.audio_scroll = QScrollArea()
        self.audio_scroll.setWidgetResizable(True)
        self.audio_scroll.setFrameShape(QFrame.NoFrame)
        self.audio_scroll.setWidget(self.audio_page)
        self.tabs.addTab(self.audio_scroll, "Audio")
        self.perception_page = QWidget()
        perception = QFormLayout(self.perception_page)
        perception.setContentsMargins(14, 16, 14, 12)
        perception.setVerticalSpacing(11)
        self.perception_enabled = QCheckBox("Analyze voices and sounds locally")
        self.perception_enabled.setToolTip(
            "Uses optional learned models on microphone audio, independently of the transcript"
        )
        perception.addRow(self.perception_enabled)
        perception_note = QLabel(
            "Distinguish anonymous speakers within this voice session and show likely sound classes. "
            "Models process the audio itself. Short or noisy samples can remain uncertain; this is not identity verification."
        )
        perception_note.setObjectName("muted")
        perception_note.setWordWrap(True)
        perception.addRow(perception_note)
        self.speaker_model_path = QLineEdit()
        self.speaker_model_path.setPlaceholderText(
            "Optional local speaker-embedding .onnx model"
        )
        perception.addRow("Speaker model", self._model_picker(self.speaker_model_path))
        self.speaker_download = QPushButton("Download speaker-embedding model")
        self.speaker_download.clicked.connect(
            lambda: self.download_requested.emit("speaker-embedding")
        )
        perception.addRow("", self.speaker_download)
        self.speaker_threshold = QDoubleSpinBox()
        self.speaker_threshold.setRange(0.5, 0.99)
        self.speaker_threshold.setSingleStep(0.01)
        self.speaker_threshold.setDecimals(2)
        self.speaker_threshold.setToolTip(
            "Minimum cosine similarity to match a session speaker; higher is stricter. This is not an identity probability."
        )
        perception.addRow("Speaker match threshold", self.speaker_threshold)
        self.sound_model_path = QLineEdit()
        self.sound_model_path.setPlaceholderText(
            "Optional local audio-event .onnx model"
        )
        perception.addRow("Sound model", self._model_picker(self.sound_model_path))
        self.sound_labels_path = QLineEdit()
        self.sound_labels_path.setPlaceholderText("Matching class_labels_indices.csv")
        perception.addRow(
            "Sound labels",
            self._model_picker(
                self.sound_labels_path, "Class labels (*.csv);;All files (*)"
            ),
        )
        self.sound_download = QPushButton("Download audio-event model + labels")
        self.sound_download.clicked.connect(
            lambda: self.download_requested.emit("audio-events")
        )
        perception.addRow("", self.sound_download)
        self.sound_threshold = QDoubleSpinBox()
        self.sound_threshold.setRange(0.05, 0.95)
        self.sound_threshold.setSingleStep(0.05)
        self.sound_threshold.setDecimals(2)
        self.sound_threshold.setToolTip(
            "Minimum model score to display a sound class; weaker detections remain unknown"
        )
        perception.addRow("Sound detection threshold", self.sound_threshold)
        self.sound_events_enabled = QCheckBox(
            "Also sample ambient sounds between speech"
        )
        self.sound_events_enabled.setToolTip(
            "Adds local audio analysis while the microphone is listening; playback remains excluded"
        )
        perception.addRow(self.sound_events_enabled)
        self.speaker_observation = QLabel("No speaker observation yet")
        self.sound_observation = QLabel("No sound observation yet")
        for label in (self.speaker_observation, self.sound_observation):
            label.setWordWrap(True)
            label.setTextFormat(Qt.PlainText)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        perception.addRow("Latest speaker", self.speaker_observation)
        perception.addRow("Latest sounds", self.sound_observation)
        uncertainty_note = QLabel(
            "Speaker similarity is not a calibrated confidence score. “Unknown sound” means no supported sound class "
            "passed the model threshold. Recent observations may accompany your next message as probabilistic sensor data."
        )
        uncertainty_note.setObjectName("muted")
        uncertainty_note.setWordWrap(True)
        perception.addRow(uncertainty_note)
        perception.setSizeConstraint(QLayout.SetMinimumSize)
        self.perception_scroll = QScrollArea()
        self.perception_scroll.setWidgetResizable(True)
        self.perception_scroll.setFrameShape(QFrame.NoFrame)
        self.perception_scroll.setWidget(self.perception_page)
        self.tabs.addTab(self.perception_scroll, "Perception")
        controls = QHBoxLayout()
        self.mute_button = QPushButton("Mute mic")
        self.mute_button.setCheckable(True)
        self.mute_button.toggled.connect(self._mute)
        controls.addWidget(self.mute_button)
        self.interrupt_button = QPushButton("Interrupt")
        self.interrupt_button.setToolTip(
            "Stop spoken output and resume microphone input"
        )
        self.interrupt_button.clicked.connect(self.interrupt_requested)
        controls.addWidget(self.interrupt_button)
        controls.addStretch()
        self.start_button = QPushButton("Start Voice")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self._toggle)
        controls.addWidget(self.start_button)
        outer.addLayout(controls)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setTextFormat(Qt.PlainText)
        self.error_label.setStyleSheet("color: #e58a6b;")
        self.error_label.hide()
        outer.addWidget(self.error_label)
        self.populate(settings or {})
        self._populating = False
        for widget in (
            self.language,
            self.stt_device,
            self.stt_compute_type,
            self.input_device,
            self.output_device,
            self.english_engine,
            self.input_mode,
            self.duplex_mode,
        ):
            widget.currentIndexChanged.connect(self._changed)
        self.stt_model.currentTextChanged.connect(self._changed)
        self.kokoro_voice.currentTextChanged.connect(self._changed)
        self.english_engine.currentIndexChanged.connect(self._update_engine)
        self._update_engine()
        self.input_mode.currentIndexChanged.connect(self._update_input_mode)
        self.duplex_mode.currentIndexChanged.connect(self._update_input_mode)
        self.transcript.submitted.connect(self._editor_submitted)
        for widget in (
            self.tts_fa_model,
            self.tts_en_model,
            self.kokoro_model_path,
            self.kokoro_voices_path,
            self.speaker_model_path,
            self.sound_model_path,
            self.sound_labels_path,
        ):
            widget.textChanged.connect(self._changed)
        for widget in (
            self.auto_send,
            self.speak_progress,
            self.perception_enabled,
            self.sound_events_enabled,
        ):
            widget.toggled.connect(self._changed)
        self.silence_ms.valueChanged.connect(self._changed)
        self.max_utterance_seconds.valueChanged.connect(self._changed)
        self.speech_speed.valueChanged.connect(self._changed)
        self.stt_cpu_threads.valueChanged.connect(self._changed)
        self.stt_beam_size.valueChanged.connect(self._changed)
        self.kokoro_cpu_threads.valueChanged.connect(self._changed)
        self.speaker_threshold.valueChanged.connect(self._changed)
        self.sound_threshold.valueChanged.connect(self._changed)
        self.transcript.textChanged.connect(self._review_ready)
        self.set_active(False)
        self._review_ready()

    def _editor_submitted(self):
        if self.input_mode.currentData() == "text":
            self._speak_text()
        else:
            self._send_transcript()

    def _speak_text(self):
        text = self.transcript.toPlainText().strip()
        if text:
            self.speak_requested.emit(text)

    def _update_input_mode(self, *args):
        text_only = self.input_mode.currentData() == "text"
        self.transcript_title.setText(
            "Text to speak or send" if text_only else "Recognized speech"
        )
        self.transcript.setPlaceholderText(
            "Type text · Enter speaks it · Shift+Enter adds a line. Send to Daena asks the Agent and reads its reply."
            if text_only
            else "Review recognized speech · Enter sends it to Daena · Shift+Enter adds a line."
        )
        if text_only:
            note = "Text only: Whisper and microphone are off. Speak text reads your text directly; Send to Daena reads the Agent’s answer. Local voices must already be installed."
        elif self.duplex_mode.currentData() == "headphones":
            note = "Wear headphones: the microphone stays open while Daena speaks, so speaking can interrupt the reply. This mode does not cancel speaker echo."
        elif self.duplex_mode.currentData() == "off":
            note = "Manual turn-taking: microphone input pauses during replies. Use Stop reply to interrupt and resume listening."
        else:
            note = "Automatic turn-taking uses local echo cancellation when available so you can interrupt by speaking. If unavailable, microphone input pauses during replies; Voice shows the active mode. Headphones mode is another option."
        self.input_note.setText(note)
        self.duplex_status.setVisible(not text_only)
        self.meter.setVisible(not text_only)
        if not self._active:
            self.start_button.setText(
                "Start voice output" if text_only else "Start Voice"
            )
        self.perception_summary.setVisible(not text_only)
        self.auto_send.setEnabled(not text_only)
        self.auto_send.setVisible(not text_only)
        self.mute_button.setVisible(not text_only)
        self.finish_button.setVisible(not text_only)
        self.finish_button.setEnabled(self._active and not text_only)
        self.mute_button.setEnabled(self._active and not text_only)
        for widget in (
            self.stt_model,
            self.stt_download,
            self.stt_device,
            self.stt_compute_type,
            self.stt_cpu_threads,
            self.stt_beam_size,
            self.input_device,
            self.duplex_mode,
            self.silence_ms,
            self.max_utterance_seconds,
        ):
            widget.setEnabled(not text_only)
        self.perception_page.setEnabled(not text_only)

    def configuration_widget(self):
        """Move the canonical controls into Settings; do not duplicate state."""
        if self._configuration is not None:
            return self._configuration
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        hint = QLabel(
            "Voice settings are separate from the Agent profile. Apply model/device changes to save and restart Voice only; the Agent keeps working."
        )
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        layout.addWidget(hint)
        layout.addWidget(self.status)
        self.tabs.removeTab(0)
        preferences = QWidget()
        form = QVBoxLayout(preferences)
        form.addWidget(QLabel("Voice input"))
        form.addWidget(self.input_mode)
        form.addWidget(self.auto_send)
        form.addWidget(self.speak_progress)
        form.addWidget(QLabel("Test a voice directly without sending to the Agent"))
        form.addWidget(self.transcript)
        form.addWidget(self.speak_button)
        form.addWidget(self.input_note)
        form.addStretch()
        self.tabs.insertTab(0, preferences, "Conversation")
        self.tabs.setCurrentIndex(1)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.metrics)
        layout.addWidget(self.error_label)
        row = QHBoxLayout()
        self.output_test_button = QPushButton("Test speaker")
        self.output_test_button.clicked.connect(self.test_output_requested)
        row.addWidget(self.output_test_button)
        refresh = QPushButton("Refresh devices")
        refresh.clicked.connect(self.refresh_requested)
        row.addWidget(refresh)
        row.addStretch()
        back = QPushButton("Open Voice")
        back.clicked.connect(self.open_requested)
        row.addWidget(back)
        self.apply_button = QPushButton("Apply voice settings")
        self.apply_button.setObjectName("primary")
        self.apply_button.clicked.connect(
            lambda: self.apply_requested.emit(self.values())
        )
        row.addWidget(self.apply_button)
        layout.addLayout(row)
        self._configuration = host
        return host

    def _model_picker(self, field, file_filter="ONNX models (*.onnx);;All files (*)"):
        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(field, 1)
        browse = QPushButton("Browse…")

        def choose():
            name, _ = QFileDialog.getOpenFileName(
                self._configuration or self,
                "Select a local voice model file",
                field.text(),
                file_filter,
            )
            if name:
                field.setText(name)

        browse.clicked.connect(choose)
        row.addWidget(browse)
        return host

    @staticmethod
    def _select(combo, value):
        index = combo.findData(value)
        if index < 0:
            index = combo.findText(str(value))
        if index >= 0:
            combo.setCurrentIndex(index)
        elif combo.isEditable():
            combo.setEditText(str(value))

    def populate(self, settings):
        previous, self._populating = self._populating, True
        self._select(self.language, settings.get("language", "auto"))
        self._select(self.input_mode, settings.get("input_mode", "microphone"))
        self._select(self.duplex_mode, settings.get("duplex_mode", "auto"))
        self._select(self.stt_model, settings.get("stt_model", "large-v3-turbo"))
        self._select(self.stt_device, settings.get("stt_device", "cpu"))
        self._select(self.stt_compute_type, settings.get("stt_compute_type", "int8"))
        self.stt_cpu_threads.setValue(int(settings.get("stt_cpu_threads", 4)))
        self.stt_beam_size.setValue(int(settings.get("stt_beam_size", 3)))
        self.tts_fa_model.setText(settings.get("tts_fa_model", ""))
        self.tts_en_model.setText(settings.get("tts_en_model", ""))
        self._select(self.english_engine, settings.get("english_engine", "piper"))
        self.kokoro_model_path.setText(settings.get("kokoro_model_path", ""))
        self.kokoro_voices_path.setText(settings.get("kokoro_voices_path", ""))
        self._select(self.kokoro_voice, settings.get("kokoro_voice", "af_heart"))
        self.kokoro_cpu_threads.setValue(int(settings.get("kokoro_cpu_threads", 2)))
        self.speech_speed.setValue(float(settings.get("speech_speed", 1.0)))
        self.perception_enabled.setChecked(
            bool(settings.get("perception_enabled", False))
        )
        self.speaker_model_path.setText(settings.get("speaker_model_path", ""))
        self.sound_model_path.setText(settings.get("sound_model_path", ""))
        self.sound_labels_path.setText(settings.get("sound_labels_path", ""))
        self.speaker_threshold.setValue(float(settings.get("speaker_threshold", 0.88)))
        self.sound_threshold.setValue(float(settings.get("sound_threshold", 0.25)))
        self.sound_events_enabled.setChecked(
            bool(settings.get("sound_events_enabled", False))
        )
        self._pending_device_ids = {
            self.input_device: settings.get("input_device_id", ""),
            self.output_device: settings.get("output_device_id", ""),
        }
        self._select(self.input_device, settings.get("input_device_id", ""))
        self._select(self.output_device, settings.get("output_device_id", ""))
        self.auto_send.setChecked(bool(settings.get("auto_send", True)))
        self.speak_progress.setChecked(bool(settings.get("speak_progress", False)))
        self.silence_ms.setValue(int(settings.get("silence_ms", 550)))
        self.max_utterance_seconds.setValue(
            int(settings.get("max_utterance_seconds", 30))
        )
        self._populating = previous
        self._update_engine()
        self._update_input_mode()

    def _update_engine(self, *args):
        kokoro = self.english_engine.currentData() == "kokoro"
        self.kokoro_page.setVisible(kokoro)
        self.tts_en_model.setEnabled(not kokoro)
        self.en_download.setEnabled(not kokoro)

    def _speaking_voice(self):
        index = self.kokoro_voice.currentIndex()
        if index >= 0 and self.kokoro_voice.currentText() == self.kokoro_voice.itemText(
            index
        ):
            return self.kokoro_voice.itemData(index)
        return self.kokoro_voice.currentText().strip()

    def _speech_model(self):
        index = self.stt_model.currentIndex()
        if index >= 0 and self.stt_model.currentText() == self.stt_model.itemText(
            index
        ):
            return self.stt_model.itemData(index)
        return self.stt_model.currentText().strip()

    def values(self):
        return {
            "input_mode": self.input_mode.currentData(),
            "duplex_mode": self.duplex_mode.currentData(),
            "language": self.language.currentData(),
            "stt_model": self._speech_model(),
            "stt_device": self.stt_device.currentText(),
            "stt_compute_type": self.stt_compute_type.currentText(),
            "stt_cpu_threads": self.stt_cpu_threads.value(),
            "stt_beam_size": self.stt_beam_size.value(),
            "tts_fa_model": self.tts_fa_model.text().strip(),
            "tts_en_model": self.tts_en_model.text().strip(),
            "english_engine": self.english_engine.currentData(),
            "kokoro_model_path": self.kokoro_model_path.text().strip(),
            "kokoro_voices_path": self.kokoro_voices_path.text().strip(),
            "kokoro_voice": self._speaking_voice(),
            "kokoro_cpu_threads": self.kokoro_cpu_threads.value(),
            "speech_speed": self.speech_speed.value(),
            "perception_enabled": self.perception_enabled.isChecked(),
            "speaker_model_path": self.speaker_model_path.text().strip(),
            "sound_model_path": self.sound_model_path.text().strip(),
            "sound_labels_path": self.sound_labels_path.text().strip(),
            "speaker_threshold": self.speaker_threshold.value(),
            "sound_threshold": self.sound_threshold.value(),
            "sound_events_enabled": self.sound_events_enabled.isChecked(),
            "input_device_id": self.input_device.currentData() or "",
            "output_device_id": self.output_device.currentData() or "",
            "auto_send": self.auto_send.isChecked(),
            "speak_progress": self.speak_progress.isChecked(),
            "silence_ms": self.silence_ms.value(),
            "max_utterance_seconds": self.max_utterance_seconds.value(),
        }

    def _changed(self, *args):
        if not self._populating:
            self.settings_changed.emit(self.values())

    def _toggle(self):
        self.error_label.hide()
        self.stop_requested.emit() if self._active else self.start_requested.emit(
            self.values()
        )

    def _mute(self, muted):
        self.mute_button.setText("Unmute mic" if muted else "Mute mic")
        self.mute_changed.emit(muted)

    def _download_speech(self):
        model = self._speech_model()
        key = {"large-v3-turbo": "whisper-turbo", "small": "whisper-small"}.get(model)
        if key:
            self.download_requested.emit(key)
        else:
            self.show_error(
                "Choose a listed speech model to download, or use a model that is already available locally."
            )

    def _send_transcript(self):
        text = self.transcript.toPlainText().strip()
        if text:
            self.send_requested.emit(text)

    def _review_ready(self):
        ready = bool(self.transcript.toPlainText().strip())
        self.send_button.setEnabled(ready)
        self.speak_button.setEnabled(ready)

    def set_devices(self, inputs, outputs=None):
        if isinstance(inputs, dict):
            inputs, outputs = inputs.get("inputs", []), inputs.get("outputs", [])
        for combo, rows, fallback in (
            (self.input_device, inputs, "System default microphone"),
            (self.output_device, outputs or [], "System default speakers"),
        ):
            selected = self._pending_device_ids.pop(combo, None)
            if selected is None:
                selected = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(fallback, "")
            for row in rows:
                combo.addItem(
                    row["name"] + (" · default" if row.get("default") else ""),
                    row["id"],
                )
            if selected and combo.findData(selected) < 0:
                combo.addItem("Unavailable device · choose another", selected)
            self._select(combo, selected)
            combo.blockSignals(False)

    def set_metrics(self, values):
        self.metrics_changed.emit(dict(values))
        parts = []
        if values.get("output_device"):
            parts.append("Output: " + str(values["output_device"]))
        if values.get("stt_ms") is not None:
            parts.append(f"Recognition {float(values['stt_ms']):.0f} ms")
        if values.get("first_pcm_ms") is not None:
            parts.append(f"First PCM {float(values['first_pcm_ms']):.0f} ms")
        if values.get("agent_ms") is not None:
            parts.append(f"Agent response {float(values['agent_ms']):.0f} ms")
        if values.get("first_playback_ms") is not None:
            parts.append(
                f"Playback dispatch {float(values['first_playback_ms']):.0f} ms"
            )
        if values.get("queued_seconds") is not None:
            parts.append(f"Queued audio {float(values['queued_seconds']):.1f} s")
        if parts:
            self.metrics.setText(" · ".join(parts))

    def set_duplex_status(self, status):
        """Show the applied audio backend separately from editable preferences."""
        message = str(status.get("message") or "").strip()
        if not message:
            message = (
                "Listening during replies"
                if status.get("active")
                else "Microphone pauses during replies; use Stop reply to interrupt"
            )
        backend = str(status.get("backend") or "").strip()
        self.duplex_status.setText(
            "Active: " + message[:500] + (" · " + backend[:80] if backend else "")
        )

    def set_state(self, text):
        self.status.setText(str(text))
        self.state_changed.emit(str(text))

    def set_perception(self, evidence):
        """Render bounded model evidence; anonymous IDs never imply a real identity."""
        speaker = evidence.get("speaker") or {}
        sounds = evidence.get("sounds") or {}
        speaker_id = speaker.get("id")
        status = str(speaker.get("status", "disabled"))
        if speaker_id:
            label = str(speaker_id).replace("speaker_", "Speaker ")[:80]
            similarity = speaker.get("similarity")
            if isinstance(similarity, (int, float)):
                label += f" · cosine similarity {similarity:.2f}"
            if status == "new":
                label += " · new in this session"
            self.speaker_observation.setText(label)
        else:
            labels = {
                "uncertain": "Uncertain speaker · sample did not match reliably",
                "too_short": "Speaker unknown · sample is too short",
                "not_speech": "No speech sample for speaker comparison",
                "silence": "No speech detected",
                "disabled": "Speaker model is not enabled",
            }
            self.speaker_observation.setText(labels.get(status, "Speaker unknown"))
        events = sounds.get("events", []) if isinstance(sounds, dict) else []
        names = []
        for event in events[:3]:
            if not isinstance(event, dict):
                continue
            name = str(event.get("label", "Unknown"))[:100]
            score = event.get("score")
            if isinstance(score, (int, float)):
                name += f" ({min(1.0, max(0.0, score)):.0%})"
            names.append(name)
        sound_status = sounds.get("status") if isinstance(sounds, dict) else "unknown"
        if sound_status == "known" and names:
            sound_label = ", ".join(names)
        else:
            sound_label = {
                "unknown": "Unknown sound · no class passed the threshold",
                "silence": "Silence or signal below the model input threshold",
                "disabled": "Sound model is not enabled",
            }.get(sound_status, "No sound observation yet")
        self.sound_observation.setText(sound_label)
        summary = []
        if speaker_id:
            summary.append(str(speaker_id).replace("speaker_", "Speaker ")[:80])
        elif status in {"uncertain", "too_short"}:
            summary.append("Speaker uncertain")
        if sound_status == "known" and names:
            summary.append("Sounds: " + ", ".join(names))
        elif sound_status == "unknown":
            summary.append("Unknown sound")
        self.perception_summary.setText(
            " · ".join(summary)
            or "Audio perception is optional · configure it in Perception"
        )

    def set_transcript(self, text):
        self.transcript.setPlainText(str(text))
        self._review_ready()
        self.transcript_changed.emit(str(text))

    def set_level(self, value):
        self.meter.set_level(value)
        self.level_changed.emit(float(value))

    def set_active(self, active):
        self._active = bool(active)
        self.input_mode.setEnabled(True)
        self.start_button.setText("End Voice" if active else "Start Voice")
        self.mute_button.setEnabled(active)
        self.interrupt_button.setEnabled(active)
        self.finish_button.setEnabled(active)
        self.model_page.setEnabled(True)
        self.audio_page.setEnabled(True)
        self.perception_page.setEnabled(True)
        if not active:
            self.mute_button.blockSignals(True)
            self.mute_button.setChecked(False)
            self.mute_button.blockSignals(False)
            self.meter.set_level(0)
        self._update_input_mode()
        self.active_changed.emit(bool(active))

    def show_error(self, message):
        self.error_label.setText(str(message))
        self.error_label.show()
        self.error_raised.emit(str(message))

    def reject(self):
        self.stop_requested.emit()
        super().reject()

    def closeEvent(self, event):
        self.stop_requested.emit()
        super().closeEvent(event)
