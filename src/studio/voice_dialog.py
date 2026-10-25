"""Local voice controls. No audio device, model or network is opened by this UI."""

from __future__ import annotations

from math import ceil

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
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


class VoiceDialog(QDialog):
    start_requested = Signal(dict)
    stop_requested = Signal()
    mute_changed = Signal(bool)
    interrupt_requested = Signal()
    end_utterance_requested = Signal()
    send_requested = Signal(str)
    download_requested = Signal(str)
    settings_changed = Signal(dict)

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Daena Voice")
        self.resize(710, 690)
        self.setMinimumSize(550, 560)
        self._active = False
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
        transcript_title = QLabel("Recognized speech")
        transcript_title.setObjectName("sectionTitle")
        chat.addWidget(transcript_title)
        self.transcript = QPlainTextEdit()
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
        review_row.addStretch()
        self.send_button = QPushButton("Send to Daena")
        self.send_button.setObjectName("primary")
        self.send_button.setStyleSheet(
            "QPushButton:disabled {background: transparent; color: #8f8c84; border: 1px solid #6b6860;}"
        )
        self.send_button.clicked.connect(self._send_transcript)
        review_row.addWidget(self.send_button)
        chat.addLayout(review_row)
        self.auto_send = QCheckBox("Send recognized speech automatically")
        self.auto_send.setToolTip(
            "Hands-free mode sends recognized text to the same Agent and tool permissions as typed chat"
        )
        chat.addWidget(self.auto_send)
        self.speak_progress = QCheckBox("Read progress updates aloud")
        self.speak_progress.setToolTip(
            "Only committed assistant messages; model reasoning is never spoken"
        )
        chat.addWidget(self.speak_progress)
        note = QLabel(
            "Microphone input pauses during playback to avoid hearing Daena’s own voice. Use Interrupt to stop speech and resume listening. This is not acoustic echo cancellation."
        )
        note.setObjectName("muted")
        note.setWordWrap(True)
        chat.addWidget(note)
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
        self.tts_fa_model = QLineEdit()
        self.tts_fa_model.setPlaceholderText("Local Persian .onnx model")
        form.addRow("Persian voice", self._model_picker(self.tts_fa_model))
        self.fa_download = QPushButton("Download Persian voice")
        self.fa_download.clicked.connect(
            lambda: self.download_requested.emit("piper-fa")
        )
        form.addRow("", self.fa_download)
        self.tts_en_model = QLineEdit()
        self.tts_en_model.setPlaceholderText("Local English .onnx model")
        form.addRow("English voice", self._model_picker(self.tts_en_model))
        self.en_download = QPushButton("Download English voice")
        self.en_download.clicked.connect(
            lambda: self.download_requested.emit("piper-en")
        )
        form.addRow("", self.en_download)
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
        ):
            widget.currentIndexChanged.connect(self._changed)
        self.stt_model.currentTextChanged.connect(self._changed)
        for widget in (self.tts_fa_model, self.tts_en_model):
            widget.textChanged.connect(self._changed)
        for widget in (self.auto_send, self.speak_progress):
            widget.toggled.connect(self._changed)
        self.silence_ms.valueChanged.connect(self._changed)
        self.max_utterance_seconds.valueChanged.connect(self._changed)
        self.transcript.textChanged.connect(self._review_ready)
        self.set_active(False)
        self._review_ready()

    def _model_picker(self, field):
        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(field, 1)
        browse = QPushButton("Browse…")

        def choose():
            name, _ = QFileDialog.getOpenFileName(
                self, "Select a local Piper model", field.text(), "Piper voice (*.onnx)"
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
        self._select(self.stt_model, settings.get("stt_model", "large-v3-turbo"))
        self._select(self.stt_device, settings.get("stt_device", "cpu"))
        self._select(self.stt_compute_type, settings.get("stt_compute_type", "int8"))
        self.tts_fa_model.setText(settings.get("tts_fa_model", ""))
        self.tts_en_model.setText(settings.get("tts_en_model", ""))
        self._pending_device_ids = {
            self.input_device: settings.get("input_device_id", ""),
            self.output_device: settings.get("output_device_id", ""),
        }
        self._select(self.input_device, settings.get("input_device_id", ""))
        self._select(self.output_device, settings.get("output_device_id", ""))
        self.auto_send.setChecked(bool(settings.get("auto_send", False)))
        self.speak_progress.setChecked(bool(settings.get("speak_progress", False)))
        self.silence_ms.setValue(int(settings.get("silence_ms", 550)))
        self.max_utterance_seconds.setValue(
            int(settings.get("max_utterance_seconds", 30))
        )
        self._populating = previous

    def _speech_model(self):
        index = self.stt_model.currentIndex()
        if index >= 0 and self.stt_model.currentText() == self.stt_model.itemText(
            index
        ):
            return self.stt_model.itemData(index)
        return self.stt_model.currentText().strip()

    def values(self):
        return {
            "language": self.language.currentData(),
            "stt_model": self._speech_model(),
            "stt_device": self.stt_device.currentText(),
            "stt_compute_type": self.stt_compute_type.currentText(),
            "tts_fa_model": self.tts_fa_model.text().strip(),
            "tts_en_model": self.tts_en_model.text().strip(),
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
        self.send_button.setEnabled(bool(self.transcript.toPlainText().strip()))

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
        parts = []
        if values.get("stt_ms") is not None:
            parts.append(f"Recognition {float(values['stt_ms']):.0f} ms")
        if values.get("first_pcm_ms") is not None:
            parts.append(f"First PCM {float(values['first_pcm_ms']):.0f} ms")
        if values.get("queued_seconds") is not None:
            parts.append(f"Queued audio {float(values['queued_seconds']):.1f} s")
        if parts:
            self.metrics.setText(" · ".join(parts))

    def set_state(self, text):
        self.status.setText(str(text))

    def set_transcript(self, text):
        self.transcript.setPlainText(str(text))
        self._review_ready()

    def set_level(self, value):
        self.meter.set_level(value)

    def set_active(self, active):
        self._active = bool(active)
        self.start_button.setText("End Voice" if active else "Start Voice")
        self.mute_button.setEnabled(active)
        self.interrupt_button.setEnabled(active)
        self.finish_button.setEnabled(active)
        self.model_page.setEnabled(not active)
        self.audio_page.setEnabled(not active)
        if not active:
            self.mute_button.setChecked(False)
            self.meter.set_level(0)

    def show_error(self, message):
        self.error_label.setText(str(message))
        self.error_label.show()

    def reject(self):
        self.stop_requested.emit()
        super().reject()

    def closeEvent(self, event):
        self.stop_requested.emit()
        super().closeEvent(event)
