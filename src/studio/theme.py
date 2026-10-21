"""A quiet, high-contrast desktop theme shared by the Studio surfaces."""
STYLE = """
QMainWindow,QDialog {background:#0b1019;}
QWidget {background:#0b1019;color:#dce4f2;font-family:'Inter','Segoe UI',sans-serif;font-size:13px;}
QLabel {background:transparent;}
QLabel#brand {font-size:22px;font-weight:800;color:#d1c4ff;letter-spacing:2px;}
QLabel#pageTitle {font-size:25px;font-weight:700;color:#f4f6fb;margin:2px 0 6px 0;}
QLabel#sectionTitle {font-size:20px;font-weight:700;color:#edf1f9;margin-bottom:8px;}
QLabel#muted,QLabel#quiet,QLabel#controlLabel {color:#8e9bb1;}
QLabel#controlLabel {font-size:11px;font-weight:600;}
QFrame#header {border-bottom:1px solid #242d40;background:#0f1623;}
QFrame#sidebar {background:#0f1623;border:1px solid #242d40;border-radius:14px;}
QPushButton,QToolButton {background:#1b2639;border:1px solid #303d54;border-radius:8px;padding:9px 13px;font-weight:600;}
QPushButton:hover,QToolButton:hover {background:#283650;border-color:#8b7cd2;}
QPushButton:pressed,QToolButton:pressed {background:#3a3655;}
QPushButton:disabled,QToolButton:disabled {color:#5f6d83;border-color:#253043;}
QPushButton#primary {background:#a78bfa;color:#141020;border-color:#b5a0fb;}
QPushButton#primary:hover {background:#bba5ff;}
QPushButton#danger {background:#2c1d2b;color:#ff9ba9;border-color:#593044;}
QLineEdit,QPlainTextEdit,QTextBrowser,QListWidget,QComboBox,QSpinBox,QDoubleSpinBox {
background:#111b2a;border:1px solid #2b3850;border-radius:8px;padding:9px;selection-background-color:#62508e;
}
QLineEdit:focus,QPlainTextEdit:focus,QComboBox:focus {border-color:#a18cdd;}
QComboBox {padding-right:25px;min-height:20px;}
QComboBox::drop-down {border:0;width:24px;}
QComboBox QAbstractItemView {background:#182338;border:1px solid #3f4b67;selection-background-color:#443965;padding:6px;}
QSpinBox,QDoubleSpinBox {min-height:22px;}
QListWidget::item {padding:10px 8px;border-radius:7px;margin:2px;}
QListWidget::item:selected {background:#342d50;color:#e8dfff;}
QListWidget::item:hover {background:#1d2a3f;}
QTabWidget::pane {border:1px solid #29364c;border-radius:10px;}
QTabBar::tab {padding:9px 14px;background:#111a29;color:#94a1b8;}
QTabBar::tab:selected {color:#d0c1ff;border-bottom:2px solid #a78bfa;}
QSplitter::handle {background:#1c2638;width:3px;height:3px;}
QScrollBar:vertical {background:transparent;width:8px;margin:2px;}
QScrollBar::handle:vertical {background:#34425b;border-radius:4px;min-height:30px;}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical {height:0;}
QScrollBar:horizontal {background:transparent;height:8px;}
QScrollBar::handle:horizontal {background:#34425b;border-radius:4px;min-width:30px;}
QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal {width:0;}
QProgressBar {border:0;border-radius:4px;background:#1c2638;height:7px;text-align:center;}
QProgressBar::chunk {background:#a78bfa;border-radius:4px;}
QFrame#composerPanel {background:#131d2c;border:1px solid #39445d;border-radius:15px;}
QFrame#composerPanel QWidget {background:transparent;}
QPlainTextEdit#composer {background:transparent;border:0;padding:9px;font-size:15px;}
QScrollArea#conversation,QWidget#conversationCanvas {background:#0b1019;border:0;}
QFrame#assistantMessage {background:#141d2b;border:1px solid #283449;border-radius:14px;}
QFrame#userMessage {background:#1d2438;border:1px solid #3a4260;border-radius:14px;}
QFrame#assistantMessage QTextBrowser {background:#141d2b;border:0;padding:0;font-size:14px;}
QFrame#userMessage QTextBrowser {background:#1d2438;border:0;padding:0;font-size:14px;}
QLabel#messageSpeaker {font-size:13px;font-weight:700;color:#eaf0fa;}
QLabel#agentAvatar {background:#3a2f59;border-radius:9px;color:#d9caff;font-size:19px;}
QLabel#userAvatar {background:#30435b;border-radius:9px;color:#bcd6f5;font-size:12px;font-weight:700;}
QLabel#messageStatus {color:#a6b9cd;font-size:11px;}
QFrame#assistantMessage QToolButton,QFrame#userMessage QToolButton {background:transparent;border:0;color:#a9b4c9;padding:4px 6px;font-size:11px;}
QCheckBox {spacing:9px;padding:4px;background:transparent;}
QCheckBox::indicator {width:17px;height:17px;border:1px solid #5c6c87;border-radius:4px;background:#111a2b;}
QCheckBox::indicator:checked {background:#a78bfa;border-color:#bda9ff;}
QToolTip {background:#24324a;color:#f1f5ff;border:1px solid #8a75cf;padding:8px;}
"""
