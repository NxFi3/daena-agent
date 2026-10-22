"""Quiet, conversation-first styling for Daena Studio.

The chat surface uses a neutral palette and minimal chrome. Advanced tools,
settings, graphs and debugging retain conventional Qt controls.
"""

from string import Template


_PALETTES = {
    "dark": {
        "bg": "#212121",
        "sidebar": "#171717",
        "text": "#f4f4f4",
        "muted": "#b0b0b0",
        "dim": "#888888",
        "border": "#414141",
        "hover": "#303030",
        "surface": "#2f2f2f",
        "field": "#292929",
        "selected": "#343434",
        "user": "#303030",
        "accent": "#f7f7f7",
        "accent_text": "#1c1c1c",
        "scroll": "#545454",
        "code": "#181818",
    },
    "light": {
        "bg": "#ffffff",
        "sidebar": "#f9f9f9",
        "text": "#202020",
        "muted": "#666666",
        "dim": "#858585",
        "border": "#dedede",
        "hover": "#eeeeee",
        "surface": "#f2f2f2",
        "field": "#ffffff",
        "selected": "#eaeaea",
        "user": "#f4f4f4",
        "accent": "#242424",
        "accent_text": "#ffffff",
        "scroll": "#c4c4c4",
        "code": "#eeeeee",
    },
}


_STYLES = Template("""
QMainWindow,QDialog {background:$bg;}
QWidget {background:$bg;color:$text;font-family:'Inter','Segoe UI',sans-serif;font-size:13px;}
QLabel {background:transparent;}
QLabel#brand,QLabel#sidebarBrand {font-size:21px;font-weight:750;color:$text;}
QLabel#sidebarBrand {font-size:18px;padding:9px 10px 15px;}
QLabel#pageTitle {font-size:23px;font-weight:650;color:$text;}
QLabel#sectionTitle {font-size:19px;font-weight:650;color:$text;margin-bottom:8px;}
QLabel#muted,QLabel#quiet,QLabel#controlLabel {color:$muted;}
QLabel#sidebarHeading {color:$dim;font-size:12px;font-weight:600;margin:17px 10px 3px;}
QLabel#modelHint {color:$dim;font-size:12px;padding:4px 9px;}
QLabel#composerHint {color:$muted;font-size:12px;}
QLabel#composerNote {color:$dim;font-size:11px;padding-top:4px;}
QLabel#runtimeStatus {color:$muted;font-size:11px;}
QFrame#header {background:$bg;border:0;}
QFrame#sidebar {background:$sidebar;border:0;}
QFrame#statusBar {background:$bg;border-top:1px solid $border;}
QPushButton,QToolButton {background:$surface;border:1px solid $border;border-radius:10px;padding:8px 12px;font-weight:550;color:$text;}
QPushButton:hover,QToolButton:hover {background:$hover;}
QPushButton:pressed,QToolButton:pressed {background:$selected;}
QPushButton:disabled,QToolButton:disabled {color:$dim;background:$surface;border-color:$border;}
QPushButton#newChat {background:transparent;border:0;padding:12px 11px;text-align:left;font-size:14px;font-weight:650;}
QPushButton#newChat:hover {background:$hover;}
QPushButton#primary {background:$accent;color:$accent_text;border:0;border-radius:10px;padding:10px 18px;font-weight:700;}
QPushButton#danger {background:$surface;color:$text;border:0;border-radius:17px;font-weight:700;}
QPushButton#danger:hover {background:$hover;}
QPushButton#sidebarFooter {background:transparent;border:0;text-align:left;color:$muted;font-weight:500;padding:12px;}
QPushButton#sidebarFooter:hover {background:$hover;color:$text;}
QPushButton#workspaceButton,QToolButton#sidebarToggle {background:transparent;border:0;color:$muted;}
QPushButton#workspaceButton:hover,QToolButton#sidebarToggle:hover {background:$hover;color:$text;}
QToolButton#sidebarToggle {font-size:20px;padding:7px 10px;}
QPushButton#modelBadge {background:transparent;border:0;padding:8px 12px;font-size:18px;font-weight:600;color:$text;}
QPushButton#modelBadge:hover {background:$hover;}
QToolButton#composerAccessory {background:transparent;border:0;border-radius:16px;font-size:19px;padding:5px 9px;}
QToolButton#composerAccessory:hover {background:$hover;}
QSlider#reasoningSlider::groove:horizontal {height:4px;background:$border;border-radius:2px;}
QSlider#reasoningSlider::sub-page:horizontal {background:$accent;border-radius:2px;}
QSlider#reasoningSlider::handle:horizontal {background:$accent;width:13px;height:13px;border-radius:6px;margin:-5px 0;}
QPushButton#sendMessage {background:$accent;color:$accent_text;border:0;border-radius:17px;font-size:20px;font-weight:800;padding:0;}
QPushButton#sendMessage:hover {background:$accent;}
QLineEdit,QPlainTextEdit,QTextBrowser,QListWidget,QComboBox,QSpinBox,QDoubleSpinBox {
background:$field;border:1px solid $border;border-radius:9px;padding:9px;selection-background-color:$selected;
}
QLineEdit:focus,QPlainTextEdit:focus,QComboBox:focus {border-color:$muted;}
QLineEdit#workspacePath {background:transparent;border:0;color:$muted;padding:3px;font-size:11px;}
QLineEdit#historySearch {background:transparent;border:0;padding:11px;color:$text;}
QLineEdit#historySearch:hover,QLineEdit#historySearch:focus {background:$hover;}
QComboBox {padding-right:25px;min-height:19px;}
QComboBox::drop-down {border:0;width:23px;}
QComboBox QAbstractItemView {background:$surface;border:1px solid $border;selection-background-color:$selected;padding:5px;}
QSpinBox,QDoubleSpinBox {min-height:22px;}
QListWidget::item {padding:10px 9px;border-radius:9px;margin:2px;}
QListWidget::item:selected {background:$selected;color:$text;}
QListWidget::item:hover {background:$hover;}
QListWidget#navigation,QListWidget#historyList {background:transparent;border:0;padding:0;outline:none;}
QListWidget#navigation::item {font-size:13px;padding:8px 11px;margin:1px 0;}
QListWidget#historyList::item {color:$muted;padding:11px 9px;margin:1px 0;}
QTabWidget::pane {border:1px solid $border;border-radius:9px;}
QTabBar::tab {padding:10px 13px;background:$surface;color:$muted;}
QTabBar::tab:selected {color:$text;border-bottom:2px solid $text;}
QSplitter::handle {background:$border;width:1px;height:1px;}
QScrollBar:vertical {background:transparent;width:8px;margin:2px;}
QScrollBar::handle:vertical {background:$scroll;border-radius:4px;min-height:28px;}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical {height:0;}
QScrollBar:horizontal {background:transparent;height:8px;}
QScrollBar::handle:horizontal {background:$scroll;border-radius:4px;min-width:28px;}
QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal {width:0;}
QProgressBar {border:0;border-radius:3px;background:$surface;height:6px;text-align:center;}
QProgressBar::chunk {background:$muted;border-radius:3px;}
QWidget#chatColumn {background:$bg;}
QFrame#composerPanel {background:$surface;border:1px solid $border;border-radius:24px;}
QPlainTextEdit#composer {background:transparent;border:0;padding:5px 8px;font-size:15px;selection-background-color:$selected;}
QScrollArea#conversation,QWidget#conversationCanvas,QWidget#messageRow,QWidget#welcomeHost {background:$bg;border:0;}
QFrame#assistantMessage {background:transparent;border:0;border-radius:0;}
QFrame#userMessage {background:$user;border:0;border-radius:20px;}
QFrame#assistantMessage QTextBrowser,QFrame#userMessage QTextBrowser {background:transparent;border:0;padding:0;font-size:15px;}
QLabel#messageStatus {color:$muted;font-size:12px;}
QToolButton#messageCopy {background:transparent;border:0;color:$dim;padding:4px 7px;font-size:11px;}
QToolButton#messageCopy:hover {color:$text;background:$hover;}
QFrame#toolActivity {background:transparent;border:0;}
QPlainTextEdit#toolReceipts {background:$code;color:$text;border:1px solid $border;border-radius:8px;font-family:monospace;font-size:11px;}
QLabel#welcomeSymbol {font-size:38px;font-weight:600;color:$text;}
QLabel#welcomeTitle {font-size:29px;font-weight:600;color:$text;}
QPushButton#suggestion {text-align:left;background:transparent;border:1px solid $border;border-radius:18px;padding:12px 16px;color:$muted;font-size:13px;font-weight:500;}
QPushButton#suggestion:hover {background:$hover;color:$text;}
QCheckBox {spacing:9px;padding:4px;background:transparent;}
QCheckBox::indicator {width:17px;height:17px;border:1px solid $muted;border-radius:4px;background:$field;}
QCheckBox::indicator:checked {background:$muted;border-color:$text;}
QToolTip {background:$surface;color:$text;border:1px solid $border;padding:8px;}
QMenu {background:$surface;color:$text;border:1px solid $border;border-radius:10px;padding:6px;}
QMenu::item {padding:9px 22px;border-radius:7px;}
QMenu::item:selected {background:$hover;}
""")


def style_for_theme(theme):
    return _STYLES.substitute(_PALETTES.get(theme, _PALETTES["dark"]))
