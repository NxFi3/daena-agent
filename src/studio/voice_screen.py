"""Focused Voice conversation view; settings and audio work live elsewhere."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import (
    QAbstractButton,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QSizePolicy,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.studio.theme import palette_for_theme
from src.studio.widgets import Composer


class VoiceOrb(QAbstractButton):
    """Decorative breathing orb; microphone amplitude is supplied by the host."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._phase = 0.0
        self._level = 0.0
        self._active = self._busy = self._speaking = False
        self._mode = "microphone"
        self._theme = "dark"
        self.setMinimumSize(210, 210)
        self.setMaximumSize(310, 310)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("background:transparent;border:0;")
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._update_accessibility()

    def sizeHint(self):
        return QSize(280, 280)

    def _update_accessibility(self):
        label = "Use microphone" if self._mode == "text" else "Type instead of speaking"
        self.setAccessibleName(label)
        self.setToolTip(label)

    def set_theme(self, theme):
        self._theme = theme
        self.update()

    def set_activity(self, *, active=None, busy=None, speaking=None, mode=None):
        if active is not None:
            self._active = bool(active)
        if busy is not None:
            self._busy = bool(busy)
        if speaking is not None:
            self._speaking = bool(speaking)
        if mode is not None:
            self._mode = mode
            self._update_accessibility()
        self._sync_timer()
        self.update()

    def _sync_timer(self):
        if self.isVisible() and (self._active or self._busy):
            self._timer.start()
        else:
            self._timer.stop()

    def set_level(self, value):
        self._level = min(1.0, max(0.0, float(value)))
        self.update()

    def _tick(self):
        self._phase = (self._phase + 0.045) % (math.tau * 10)
        self.update()

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_timer()

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        colors = palette_for_theme(self._theme)
        center = QPointF(self.width() / 2, self.height() / 2)
        side = min(self.width(), self.height())
        breath = math.sin(self._phase) if self._active or self._busy else 0
        amplitude = self._level if self._mode == "microphone" and not self._busy else 0
        radius = side * (0.305 + 0.008 * breath + 0.035 * amplitude)

        glow = QRadialGradient(center, side * 0.49)
        accent = QColor(colors["accent"])
        glow.setColorAt(0, QColor(accent.red(), accent.green(), accent.blue(), 75))
        glow.setColorAt(0.62, QColor(accent.red(), accent.green(), accent.blue(), 28))
        glow.setColorAt(1, QColor(accent.red(), accent.green(), accent.blue(), 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(center, side * 0.49, side * 0.49)

        for index in range(3):
            shape = QPainterPath()
            for step in range(97):
                angle = step / 96 * math.tau
                ripple = math.sin(angle * 3 + self._phase + index * 1.6)
                ripple += math.sin(angle * 2 - self._phase * 0.7) * 0.4
                distance = radius * (1 + ripple * (0.025 + 0.04 * amplitude))
                point = center + QPointF(
                    math.cos(angle) * distance, math.sin(angle) * distance
                )
                if not step:
                    shape.moveTo(point)
                else:
                    shape.lineTo(point)
            shape.closeSubpath()
            gradient = QRadialGradient(
                center + QPointF(-radius * 0.45, -radius * 0.5), radius * 2.1
            )
            gradient.setColorAt(0, QColor("#f5c8a1"))
            gradient.setColorAt(0.42, QColor("#df9270"))
            gradient.setColorAt(1, QColor("#9d5347" if self._busy else "#b5644e"))
            painter.setOpacity(0.38 if index < 2 else 0.95)
            painter.setBrush(gradient)
            painter.drawPath(shape)
        painter.setOpacity(1)

        if self.hasFocus():
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(colors["text"]), 1.5, Qt.DashLine))
            painter.drawEllipse(center, radius + 12, radius + 12)
        painter.translate(center)
        painter.setPen(
            QPen(QColor("#fff8f0"), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        )
        painter.setBrush(Qt.NoBrush)
        if self._mode == "text":
            painter.drawRoundedRect(QRectF(-23, -14, 46, 29), 5, 5)
            for row in range(2):
                for column in range(5):
                    painter.drawPoint(QPointF(-15 + column * 7.5, -6 + row * 7))
            painter.drawLine(QPointF(-9, 8), QPointF(9, 8))
        else:
            painter.drawRoundedRect(QRectF(-8, -19, 16, 28), 8, 8)
            painter.drawArc(QRectF(-14, -12, 28, 30), 0, -180 * 16)
            painter.drawLine(QPointF(0, 18), QPointF(0, 24))
            painter.drawLine(QPointF(-7, 24), QPointF(7, 24))


class VoiceScreen(QDialog):
    """A small modeless conversation surface, independent from model settings."""

    send_requested = Signal(str)
    stop_requested = Signal()
    interrupt_requested = Signal()
    mute_changed = Signal(bool)
    input_mode_requested = Signal(str)
    settings_requested = Signal()
    start_requested = Signal()

    def __init__(self, parent=None, *, input_mode="microphone", theme="dark"):
        super().__init__(parent)
        self.setWindowTitle("Daena · Voice")
        self.resize(600, 700)
        self.setMinimumSize(500, 560)
        self.setModal(False)
        self._active = self._busy = False
        self._mode = "microphone"
        self._last_program_text = self._submitted_text = ""
        self._closing_sent = False
        self._output_ready = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 22, 28, 24)
        outer.setSpacing(12)
        header = QHBoxLayout()
        brand = QLabel("DAENA")
        brand.setObjectName("brand")
        header.addWidget(brand)
        header.addStretch()
        self.settings_button = QToolButton()
        self.settings_button.setText("Voice settings")
        self.settings_button.setObjectName("workspaceButton")
        self.settings_button.clicked.connect(self.settings_requested)
        header.addWidget(self.settings_button)
        outer.addLayout(header)

        center_widget = QWidget()
        center = QVBoxLayout(center_widget)
        center.setContentsMargins(0, 0, 0, 0)
        center.setSpacing(10)
        center.setSizeConstraint(QLayout.SetMinimumSize)
        self.center_scroll = QScrollArea()
        self.center_scroll.setWidgetResizable(True)
        self.center_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.center_scroll.setFrameShape(QFrame.NoFrame)
        self.center_scroll.setWidget(center_widget)
        outer.addWidget(self.center_scroll, 1)
        center.addStretch(1)
        self.orb = VoiceOrb(self)
        self.orb.set_theme(theme)
        self.orb.clicked.connect(self.toggle_input)
        center.addWidget(self.orb, 3, Qt.AlignHCenter)
        self.heading = QLabel("Ready when you are")
        self.heading.setAlignment(Qt.AlignCenter)
        self.heading.setObjectName("pageTitle")
        center.addWidget(self.heading)
        self.status = QLabel("Start Voice to begin a conversation")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setTextFormat(Qt.PlainText)
        self.status.setWordWrap(True)
        self.status.setObjectName("muted")
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        center.addWidget(self.status)
        self.input_hint = QLabel()
        self.input_hint.setAlignment(Qt.AlignCenter)
        self.input_hint.setWordWrap(True)
        self.input_hint.setObjectName("muted")
        self.input_hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        center.addWidget(self.input_hint)

        self.reply = QLabel()
        self.reply.setAlignment(Qt.AlignCenter)
        self.reply.setWordWrap(True)
        self.reply.setTextFormat(Qt.PlainText)
        self.reply.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.reply.setMaximumHeight(72)
        self.reply.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.reply.hide()
        center.addWidget(self.reply)

        self.composer_frame = QFrame()
        entry = QVBoxLayout(self.composer_frame)
        entry.setContentsMargins(0, 8, 0, 0)
        entry.setSpacing(7)
        self.composer = Composer()
        self.transcript = self.composer
        self.composer.setPlaceholderText("Message Daena · Enter to send")
        self.composer.submitted.connect(self._send)
        self.composer.textChanged.connect(self._update_send)
        entry.addWidget(self.composer)
        send_row = QHBoxLayout()
        entry_note = QLabel("Shift+Enter for a new line")
        entry_note.setObjectName("muted")
        send_row.addWidget(entry_note)
        send_row.addStretch()
        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("primary")
        self.send_button.clicked.connect(self._send)
        send_row.addWidget(self.send_button)
        entry.addLayout(send_row)
        center.addWidget(self.composer_frame)

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setTextFormat(Qt.PlainText)
        self.error_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.error_label.setStyleSheet("color:#e58a6b;")
        self.error_label.hide()
        center.addWidget(self.error_label)
        center.addStretch(1)
        controls = QHBoxLayout()
        self.input_button = QPushButton()
        self.input_button.clicked.connect(self.toggle_input)
        controls.addWidget(self.input_button)
        self.mute_button = QToolButton()
        self.mute_button.setText("Mute")
        self.mute_button.setCheckable(True)
        self.mute_button.setAccessibleName("Mute microphone")
        self.mute_button.toggled.connect(self._mute)
        controls.addWidget(self.mute_button)
        controls.addStretch()
        self.interrupt_button = QPushButton("Stop reply")
        self.interrupt_button.setToolTip("Interrupt speech and the active Agent turn")
        self.interrupt_button.clicked.connect(self.interrupt_requested)
        controls.addWidget(self.interrupt_button)
        self.end_button = QPushButton("Start Voice")
        self.end_button.clicked.connect(self._start_or_stop)
        controls.addWidget(self.end_button)
        outer.addLayout(controls)
        self.footer = QLabel()
        self.footer.setAlignment(Qt.AlignCenter)
        self.footer.setObjectName("muted")
        outer.addWidget(self.footer)
        self.set_input_mode(input_mode)
        self.set_active(False)
        self._update_send()

    def toggle_input(self):
        mode = "text" if self._mode == "microphone" else "microphone"
        self.set_input_mode(mode)
        self.input_mode_requested.emit(mode)
        if mode == "text":
            self.composer.setFocus(Qt.OtherFocusReason)
            QTimer.singleShot(
                0, lambda: self.center_scroll.ensureWidgetVisible(self.composer)
            )

    def set_input_mode(self, mode):
        if mode not in {"text", "microphone"}:
            raise ValueError("Voice input mode must be text or microphone")
        self._mode = mode
        typed = mode == "text"
        if typed and self.mute_button.isChecked():
            self.mute_button.blockSignals(True)
            self.mute_button.setChecked(False)
            self.mute_button.blockSignals(False)
            self.mute_button.setText("Mute")
        self.orb.setMinimumSize(140 if typed else 210, 140 if typed else 210)
        self.orb.setMaximumSize(210 if typed else 310, 210 if typed else 310)
        self.composer_frame.setVisible(typed)
        self.mute_button.setVisible(not typed)
        self.mute_button.setEnabled(self._active and not typed)
        self.input_button.setText("Use microphone" if typed else "Type instead")
        self.input_hint.setText(
            "Type your message. Daena will speak its answer."
            if typed
            else "Tap the circle to type instead of speaking"
        )
        self.footer.setText(
            "Microphone off · typed conversation"
            if typed
            else "Microphone muted"
            if self.mute_button.isChecked()
            else "Microphone active"
            if self._active
            else "Microphone off"
        )
        self.orb.set_activity(mode=mode)

    def _send(self):
        text = self.composer.toPlainText().strip()
        if text:
            self._submitted_text = text
            self.error_label.hide()
            self.send_requested.emit(text)

    def _update_send(self):
        self.send_button.setEnabled(bool(self.composer.toPlainText().strip()))

    def set_transcript(self, text):
        text = str(text)
        current = self.composer.toPlainText()
        if not text:
            if current.strip() in {
                self._submitted_text.strip(),
                self._last_program_text.strip(),
            }:
                self.composer.clear()
            self._submitted_text = self._last_program_text = ""
        elif not current or current == self._last_program_text:
            self.composer.setPlainText(text)
            self._last_program_text = text
        if self._mode == "microphone":
            review = bool(self.composer.toPlainText().strip())
            self.composer_frame.setVisible(review)
            self.input_hint.setText(
                "Review your words and Send, or tap the circle to stop microphone input."
                if review
                else "Tap the circle to type instead of speaking"
            )
        # Keep a user's edited draft intact if recognition or old callbacks arrive.

    def acknowledge_submission(self, text=None):
        if text is not None:
            self._submitted_text = str(text)
        self.set_transcript("")

    def set_reply(self, text):
        text = " ".join(str(text).split())
        self.reply.setText(text[:260] + ("…" if len(text) > 260 else ""))
        self.reply.setVisible(bool(text))

    def set_state(self, text):
        text = str(text)
        self.status.setText(text)
        lowered = text.casefold()
        speaking = "speaking" in lowered or "playing" in lowered
        self.orb.set_activity(speaking=speaking)
        self.heading.setText(
            "Daena is speaking"
            if speaking
            else "Daena is thinking"
            if self._busy
            else "Listening"
            if self._active and self._mode == "microphone"
            else "Talk with Daena"
            if self._active
            else "Ready when you are"
        )

    def set_active(self, active):
        self._active = bool(active)
        self.orb.set_activity(active=active)
        self.end_button.setText("End Voice" if active else "Start Voice")
        self.interrupt_button.setEnabled(self._active or self._busy)
        self.mute_button.setEnabled(self._active and self._mode == "microphone")
        if not active:
            self.mute_button.blockSignals(True)
            self.mute_button.setChecked(False)
            self.mute_button.blockSignals(False)
            self.orb.set_level(0)
        self.set_input_mode(self._mode)
        self.set_state(self.status.text())

    def set_busy(self, busy):
        self._busy = bool(busy)
        self.orb.set_activity(busy=busy)
        self.interrupt_button.setEnabled(self._active or self._busy)
        self.set_state(self.status.text())

    def set_level(self, level):
        self.orb.set_level(level)

    def set_metrics(self, values):
        # Detailed timings belong in Settings; keep the conversation calm.
        if values.get("first_pcm_ms") is not None:
            self.orb.setToolTip(
                f"First audio: {float(values['first_pcm_ms']):.0f} ms · {self.orb.accessibleName()}"
            )

    def set_output_ready(self, ready):
        self._output_ready = bool(ready)

    def show_error(self, text):
        self.error_label.setText(str(text))
        self.error_label.show()

    def _mute(self, muted):
        self.mute_button.setText("Unmute" if muted else "Mute")
        self.footer.setText(
            "Microphone muted"
            if muted
            else "Microphone active"
            if self._active
            else "Microphone off"
        )
        self.mute_changed.emit(bool(muted))

    def _start_or_stop(self):
        if self._active:
            self.stop_requested.emit()
        else:
            self.start_requested.emit()

    def _emit_close(self):
        if not self._closing_sent:
            self._closing_sent = True
            self.stop_requested.emit()

    def showEvent(self, event):
        self._closing_sent = False
        super().showEvent(event)

    def reject(self):
        self._emit_close()
        super().reject()

    def closeEvent(self, event):
        self._emit_close()
        super().closeEvent(event)
