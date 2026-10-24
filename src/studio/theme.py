"""Modern, conversation-first styling for Daena Studio.

The look follows current chat products: a warm neutral palette, one accent
colour, generous spacing, rounded surfaces and almost no borders. Advanced
tools, settings, graphs and debugging keep conventional Qt controls that pick
up the same tokens.
"""

from string import Template


_PALETTES = {
    "dark": {
        "bg": "#212121",
        "sidebar": "#181818",
        "text": "#ececec",
        "muted": "#a8a49b",
        "dim": "#7d7a72",
        "border": "#3a3936",
        "hover": "#302f2c",
        "surface": "#30302e",
        "field": "#2b2a28",
        "selected": "#383734",
        "user": "#303030",
        "accent": "#d97757",
        "accent_hover": "#e58a6b",
        "accent_text": "#ffffff",
        "scroll": "#4d4b46",
        "code": "#1c1b1a",
        "code_text": "#e9e6df",
        "link": "#7fb2e8",
        "quote": "#4a4843",
        "danger": "#e0675a",
    },
    "light": {
        "bg": "#faf9f5",
        "sidebar": "#f2f0e9",
        "text": "#262522",
        "muted": "#6b6860",
        "dim": "#8f8c84",
        "border": "#e4e1d8",
        "hover": "#ebe8df",
        "surface": "#ffffff",
        "field": "#ffffff",
        "selected": "#e7e4da",
        "user": "#eeebe2",
        "accent": "#c6613f",
        "accent_hover": "#b25434",
        "accent_text": "#ffffff",
        "scroll": "#cfcbc0",
        "code": "#f0ede4",
        "code_text": "#2b2a27",
        "link": "#2a6cb8",
        "quote": "#cfcbc0",
        "danger": "#c4453a",
    },
}


_STYLES = Template("""
QMainWindow,QDialog {background:$bg;}
QWidget {background:$bg;color:$text;font-family:'Inter','Segoe UI','Helvetica Neue',sans-serif;font-size:13px;}
QLabel {background:transparent;}
QLabel#brand,QLabel#sidebarBrand {font-size:19px;font-weight:700;color:$text;}
QLabel#sidebarBrand {padding:8px 10px 14px;}
QLabel#pageTitle {font-size:23px;font-weight:650;color:$text;}
QLabel#sectionTitle {font-size:19px;font-weight:650;color:$text;margin-bottom:8px;}
QLabel#muted,QLabel#quiet,QLabel#controlLabel {color:$muted;}
QLabel#sidebarHeading {color:$dim;font-size:11px;font-weight:600;margin:18px 10px 4px;}
QLabel#modelHint {color:$dim;font-size:12px;padding:4px 9px;}
QLabel#composerHint {color:$muted;font-size:12px;}
QLabel#composerNote {color:$dim;font-size:11px;padding-top:6px;}
QLabel#runtimeStatus {color:$muted;font-size:11px;}
QFrame#header {background:$bg;border:0;border-bottom:1px solid $border;}
QFrame#sidebar {background:$sidebar;border:0;}
QFrame#statusBar {background:$bg;border-top:1px solid $border;}
QPushButton,QToolButton {background:$surface;border:1px solid $border;border-radius:10px;padding:8px 12px;font-weight:550;color:$text;}
QPushButton:hover,QToolButton:hover {background:$hover;}
QPushButton:pressed,QToolButton:pressed {background:$selected;}
QPushButton:disabled,QToolButton:disabled {color:$dim;background:$surface;border-color:$border;}
QPushButton#newChat {background:transparent;border:0;border-radius:10px;padding:10px 12px;text-align:left;font-size:14px;font-weight:600;color:$text;}
QPushButton#newChat:hover {background:$hover;}
QPushButton#primary {background:$accent;color:$accent_text;border:0;border-radius:10px;padding:10px 18px;font-weight:700;}
QPushButton#primary:hover {background:$accent_hover;}
QPushButton#danger {background:$surface;color:$text;border:0;border-radius:17px;font-weight:700;}
QPushButton#danger:hover {background:$hover;}
QPushButton#sidebarFooter {background:transparent;border:0;text-align:left;color:$muted;font-weight:500;padding:10px 12px;}
QPushButton#sidebarFooter:hover {background:$hover;color:$text;}
QPushButton#workspaceButton,QToolButton#sidebarToggle {background:transparent;border:0;color:$muted;}
QPushButton#workspaceButton:hover,QToolButton#sidebarToggle:hover {background:$hover;color:$text;}
QToolButton#sidebarToggle {font-size:18px;padding:6px 10px;}
QPushButton#modelBadge {background:transparent;border:0;border-radius:10px;padding:7px 12px;font-size:16px;font-weight:600;color:$text;}
QPushButton#modelBadge:hover {background:$hover;}
QPushButton#modelBadge::menu-indicator {image:none;width:0;}
QToolButton#composerAccessory {background:transparent;border:0;border-radius:16px;font-size:19px;color:$muted;padding:4px 9px;}
QToolButton#composerAccessory:hover {background:$hover;color:$text;}
QSlider#reasoningSlider::groove:horizontal {height:4px;background:$border;border-radius:2px;}
QSlider#reasoningSlider::sub-page:horizontal {background:$accent;border-radius:2px;}
QSlider#reasoningSlider::handle:horizontal {background:$accent;width:13px;height:13px;border-radius:6px;margin:-5px 0;}
QPushButton#sendMessage {background:$accent;color:$accent_text;border:0;border-radius:17px;font-size:18px;font-weight:800;padding:0;}
QPushButton#sendMessage:hover {background:$accent_hover;}
QPushButton#sendMessage:disabled {background:$border;color:$dim;}
QLineEdit,QPlainTextEdit,QTextBrowser,QListWidget,QComboBox,QSpinBox,QDoubleSpinBox {
background:$field;border:1px solid $border;border-radius:10px;padding:9px;selection-background-color:$selected;
}
QLineEdit:focus,QPlainTextEdit:focus,QComboBox:focus {border-color:$muted;}
QLineEdit#workspacePath {background:transparent;border:0;color:$muted;padding:3px;font-size:11px;}
QLineEdit#historySearch {background:$bg;border:1px solid $border;border-radius:10px;padding:9px 11px;color:$text;}
QLineEdit#historySearch:hover,QLineEdit#historySearch:focus {border-color:$muted;}
QComboBox {padding-right:25px;min-height:19px;}
QComboBox::drop-down {border:0;width:23px;}
QComboBox QAbstractItemView {background:$surface;border:1px solid $border;selection-background-color:$selected;padding:5px;}
QSpinBox,QDoubleSpinBox {min-height:22px;}
QListWidget::item {padding:10px 9px;border-radius:9px;margin:2px;}
QListWidget::item:selected {background:$selected;color:$text;}
QListWidget::item:hover {background:$hover;}
QListWidget#navigation,QListWidget#historyList {background:transparent;border:0;padding:0;outline:none;}
QListWidget#navigation::item {font-size:13px;padding:8px 11px;margin:1px 0;color:$muted;}
QListWidget#navigation::item:selected {color:$text;}
QListWidget#historyList::item {color:$muted;padding:10px 10px;margin:1px 0;}
QListWidget#historyList::item:selected {color:$text;}
QTabWidget::pane {border:1px solid $border;border-radius:10px;}
QTabBar::tab {padding:10px 13px;background:$surface;color:$muted;}
QTabBar::tab:selected {color:$text;border-bottom:2px solid $accent;}
QSplitter::handle {background:$border;width:1px;height:1px;}
QScrollBar:vertical {background:transparent;width:10px;margin:2px;}
QScrollBar::handle:vertical {background:$scroll;border-radius:4px;min-height:32px;}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical {height:0;}
QScrollBar:horizontal {background:transparent;height:10px;}
QScrollBar::handle:horizontal {background:$scroll;border-radius:4px;min-width:32px;}
QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal {width:0;}
QProgressBar {border:0;border-radius:3px;background:$surface;height:6px;text-align:center;}
QProgressBar::chunk {background:$accent;border-radius:3px;}
QWidget#chatColumn {background:$bg;}
QFrame#composerPanel {background:$surface;border:1px solid $border;border-radius:26px;}
QPlainTextEdit#composer {background:transparent;border:0;padding:5px 8px;font-size:15px;selection-background-color:$selected;}
QScrollArea#conversation,QWidget#conversationCanvas,QWidget#messageRow,QWidget#welcomeHost {background:$bg;border:0;}
QFrame#assistantMessage {background:transparent;border:0;border-radius:0;}
QFrame#userMessage {background:$user;border:0;border-radius:20px;}
QFrame#assistantMessage QTextBrowser,QFrame#userMessage QTextBrowser {background:transparent;border:0;padding:0;font-size:15px;selection-background-color:$selected;}
QLabel#messageStatus {color:$muted;font-size:12px;padding:2px 0 4px;}
QToolButton#messageCopy {background:transparent;border:0;color:$dim;padding:4px 7px;font-size:11px;}
QToolButton#messageCopy:hover {color:$text;background:$hover;}
QFrame#workTimeline {background:transparent;border:0;border-left:1px solid $border;}
QLabel#thinkingDots {color:$muted;font-size:12px;padding:3px 0;}
QFrame#toolActivity {background:transparent;border:0;margin:0;padding:2px 0 6px;}
QLabel#toolPreview {color:$muted;font-size:11px;padding:3px 0 4px;}
QLabel#activityLink {color:$muted;font-size:12px;}
QPlainTextEdit#toolReceipts {background:$code;color:$text;border:1px solid $border;border-radius:8px;font-family:monospace;font-size:11px;}
QLabel#welcomeSymbol {font-size:38px;font-weight:600;color:$text;}
QLabel#welcomeTitle {font-size:30px;font-weight:600;color:$text;padding:4px 0 10px;}
QPushButton#suggestion {text-align:left;background:transparent;border:1px solid $border;border-radius:18px;padding:12px 16px;color:$muted;font-size:13px;font-weight:500;}
QPushButton#suggestion:hover {background:$hover;color:$text;}
QCheckBox {spacing:9px;padding:4px;background:transparent;}
QCheckBox::indicator {width:17px;height:17px;border:1px solid $muted;border-radius:4px;background:$field;}
QCheckBox::indicator:checked {background:$accent;border-color:$accent;}
QToolTip {background:$surface;color:$text;border:1px solid $border;padding:8px;}
QMenu {background:$surface;color:$text;border:1px solid $border;border-radius:12px;padding:6px;}
QMenu::item {padding:9px 22px;border-radius:8px;}
QMenu::item:selected {background:$hover;}
QMenu::separator {height:1px;background:$border;margin:5px 8px;}
""")


def palette_for_theme(theme):
    """Return the colour tokens for a theme name (unknown names use dark)."""
    return dict(_PALETTES.get(theme, _PALETTES["dark"]))


def markdown_css(theme):
    """Stylesheet for rendered Markdown, kept in sync with the window theme."""
    p = palette_for_theme(theme)
    return (
        "body {line-height:1.6;}"
        "p {margin-top:4px;margin-bottom:12px;}"
        "h1,h2,h3,h4 {margin-top:16px;margin-bottom:8px;font-weight:650;}"
        "ul,ol {margin-top:2px;margin-bottom:12px;}"
        "li {margin-bottom:4px;}"
        f"pre {{background:{p['code']};color:{p['code_text']};padding:14px;"
        "white-space:pre-wrap;font-family:monospace;}"
        f"code {{font-family:monospace;color:{p['code_text']};}}"
        f"a {{color:{p['link']};text-decoration:none;}}"
        f"blockquote {{color:{p['muted']};}}"
        f"table {{border:1px solid {p['border']};}}"
    )


def style_for_theme(theme):
    return _STYLES.substitute(_PALETTES.get(theme, _PALETTES["dark"]))
