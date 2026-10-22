"""Daena Studio's graphite-and-violet design system.

Object names deliberately scope the conversation treatments: complex editors,
graph views, and settings panels still use conventional Qt controls.
"""

STYLE = """
QMainWindow,QDialog {background:#101116;}
QWidget {background:#101116;color:#e9e8ef;font-family:'Inter','Segoe UI',sans-serif;font-size:13px;}
QLabel {background:transparent;}
QLabel#brand {font-size:20px;font-weight:800;color:#f5f2ff;letter-spacing:.3px;}
QLabel#pageTitle {font-size:23px;font-weight:700;color:#f5f3f9;}
QLabel#sectionTitle {font-size:19px;font-weight:700;color:#f4f2f8;margin-bottom:8px;}
QLabel#muted,QLabel#quiet,QLabel#controlLabel {color:#898b9a;}
QLabel#controlLabel {font-size:11px;font-weight:600;}
QLabel#sidebarHeading {color:#777988;font-size:10px;font-weight:800;letter-spacing:1.5px;margin:14px 10px 2px;}
QLabel#modelBadge {color:#ccc2ff;background:#252233;border:1px solid #393348;border-radius:13px;padding:6px 12px;font-size:11px;font-weight:600;}
QLabel#runtimeStatus {color:#aeb4c3;font-size:11px;}
QFrame#header {background:#14151b;border-bottom:1px solid #24262e;}
QFrame#sidebar {background:#16171d;border:0;border-right:1px solid #292a34;border-radius:0;}
QFrame#statusBar {background:#14151b;border-top:1px solid #25262d;}
QPushButton,QToolButton {background:#242630;border:1px solid #343640;border-radius:9px;padding:8px 12px;font-weight:600;color:#e1e2ee;}
QPushButton:hover,QToolButton:hover {background:#30323e;border-color:#666277;}
QPushButton:pressed,QToolButton:pressed {background:#3d394f;}
QPushButton:disabled,QToolButton:disabled {color:#606370;background:#202127;border-color:#292a32;}
QPushButton#newChat {background:#e7e0ff;color:#171322;border:1px solid #e7e0ff;padding:12px;text-align:left;font-weight:800;}
QPushButton#newChat:hover {background:#f3eeff;}
QPushButton#primary {background:#b8a2ff;color:#1a142a;border:1px solid #cbbaff;border-radius:10px;padding:10px 20px;font-weight:800;}
QPushButton#primary:hover {background:#d2c4ff;}
QPushButton#danger {background:#38252e;color:#ffb5bf;border-color:#62414c;}
QPushButton#danger:hover {background:#58313e;}
QPushButton#sidebarFooter {background:transparent;border:1px solid #30313b;text-align:left;color:#a9a8b6;font-weight:500;}
QPushButton#workspaceButton,QToolButton#sidebarToggle,QToolButton#inspectorToggle {background:transparent;border:1px solid transparent;color:#abaebc;}
QPushButton#workspaceButton:hover,QToolButton#sidebarToggle:hover,QToolButton#inspectorToggle:hover {background:#2a2b35;color:#f1edfb;}
QToolButton#sidebarToggle {font-size:18px;padding:7px 10px;}
QLineEdit,QPlainTextEdit,QTextBrowser,QListWidget,QComboBox,QSpinBox,QDoubleSpinBox {
background:#1a1c24;border:1px solid #333640;border-radius:9px;padding:9px;selection-background-color:#62548a;
}
QLineEdit:focus,QPlainTextEdit:focus,QComboBox:focus {border-color:#9b86d6;}
QLineEdit#workspacePath {background:transparent;border:0;color:#a9a5b8;padding:3px;font-size:11px;}
QLineEdit#historySearch {background:#1e2028;border:1px solid #2d303a;padding:10px;}
QComboBox {padding-right:25px;min-height:19px;}
QComboBox::drop-down {border:0;width:23px;}
QComboBox QAbstractItemView {background:#252630;border:1px solid #424450;selection-background-color:#4b4266;padding:5px;}
QSpinBox,QDoubleSpinBox {min-height:22px;}
QListWidget::item {padding:10px 9px;border-radius:8px;margin:2px;}
QListWidget::item:selected {background:#302b40;color:#e8deff;}
QListWidget::item:hover {background:#272932;}
QListWidget#navigation,QListWidget#historyList {background:transparent;border:0;padding:0;outline:none;}
QListWidget#navigation::item {font-size:13px;padding:12px 11px;margin:2px 0;}
QListWidget#historyList::item {color:#b8b9c5;padding:11px 8px;margin:1px 0;}
QTabWidget::pane {border:1px solid #30313c;border-radius:9px;}
QTabBar::tab {padding:10px 13px;background:#1b1d25;color:#9294a4;}
QTabBar::tab:selected {color:#dbcfff;border-bottom:2px solid #b6a0ff;}
QSplitter::handle {background:#262831;width:2px;height:2px;}
QScrollBar:vertical {background:transparent;width:8px;margin:2px;}
QScrollBar::handle:vertical {background:#3d3e4b;border-radius:4px;min-height:28px;}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical {height:0;}
QScrollBar:horizontal {background:transparent;height:8px;}
QScrollBar::handle:horizontal {background:#3d3e4b;border-radius:4px;min-width:28px;}
QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal {width:0;}
QProgressBar {border:0;border-radius:3px;background:#2b2b35;height:6px;text-align:center;}
QProgressBar::chunk {background:#baa6ff;border-radius:3px;}
QFrame#composerPanel {background:#22232c;border:1px solid #4b465c;border-radius:18px;}
QPlainTextEdit#composer {background:transparent;border:0;padding:6px;font-size:15px;selection-background-color:#60537f;}
QScrollArea#conversation,QWidget#conversationCanvas,QWidget#messageRow,QWidget#welcomeHost {background:#101116;border:0;}
QFrame#assistantMessage {background:transparent;border:0;border-radius:0;}
QFrame#userMessage {background:#292a34;border:1px solid #363742;border-radius:17px;}
QFrame#assistantMessage QTextBrowser {background:transparent;border:0;padding:0;font-size:14px;}
QFrame#userMessage QTextBrowser {background:transparent;border:0;padding:0;font-size:14px;}
QLabel#messageSpeaker {font-size:13px;font-weight:800;color:#eeeaf9;}
QLabel#agentAvatar {background:#b8a2ff;border-radius:11px;color:#231a3a;font-size:19px;font-weight:800;}
QLabel#userAvatar {background:#3c4050;border-radius:11px;color:#d7d9ec;font-size:12px;font-weight:700;}
QLabel#messageStatus {color:#9aa0af;font-size:11px;}
QFrame#assistantMessage QToolButton,QFrame#userMessage QToolButton {background:transparent;border:0;color:#aaabb9;padding:4px 7px;font-size:11px;}
QFrame#assistantMessage QToolButton:hover,QFrame#userMessage QToolButton:hover {color:#eee4ff;background:#282934;}
QFrame#toolActivity {background:transparent;border-top:1px solid #302f3a;margin-top:5px;}
QPlainTextEdit#toolReceipts {background:#1c1d26;color:#bab8ca;border:1px solid #353440;border-radius:8px;font-family:monospace;font-size:11px;}
QLabel#welcomeSymbol {font-size:42px;font-weight:700;color:#b8a2ff;}
QLabel#welcomeTitle {font-size:31px;font-weight:700;color:#f4f1fb;}
QLabel#welcomeSubtitle {font-size:14px;color:#a1a1ae;line-height:1.4;}
QPushButton#suggestion {text-align:left;background:#1a1b23;border:1px solid #31323d;border-radius:12px;padding:15px 19px;color:#d5d5e2;font-size:13px;font-weight:600;}
QPushButton#suggestion:hover {background:#292735;border-color:#73658c;color:#f6f1ff;}
QCheckBox {spacing:9px;padding:4px;background:transparent;}
QCheckBox::indicator {width:17px;height:17px;border:1px solid #646578;border-radius:4px;background:#1a1c25;}
QCheckBox::indicator:checked {background:#b8a2ff;border-color:#d0c0ff;}
QToolTip {background:#302c3c;color:#faf8ff;border:1px solid #77678f;padding:8px;}
"""

# The light palette is derived from the same scoped styles so both appearances
# keep identical widget dimensions and interaction hit targets.
_LIGHT_COLORS = {
    "#101116": "#faf9fc",
    "#14151b": "#ffffff",
    "#16171d": "#f2f0f5",
    "#1a1b23": "#ffffff",
    "#1a1c24": "#ffffff",
    "#1a1c25": "#ffffff",
    "#1b1d25": "#f0eef4",
    "#1c1d26": "#f4f2f8",
    "#1e2028": "#ffffff",
    "#202127": "#f0edf4",
    "#22232c": "#ffffff",
    "#242630": "#eeecf2",
    "#252233": "#eee9fa",
    "#25262d": "#ded9e4",
    "#262831": "#e3dfea",
    "#272932": "#e7e3ed",
    "#282934": "#eeeaf3",
    "#292735": "#eeeafa",
    "#292a34": "#eeebf2",
    "#292a32": "#dad6e0",
    "#2a2b35": "#eeeaf3",
    "#2b2b35": "#e0dce5",
    "#2d303a": "#dedbe4",
    "#302b40": "#e1daf0",
    "#30313b": "#ded9e3",
    "#30323e": "#e5e0eb",
    "#302c3c": "#fffaff",
    "#31323d": "#ded9e4",
    "#333640": "#d9d5e1",
    "#343640": "#d5d0de",
    "#353440": "#ddd8e5",
    "#363742": "#dad6e1",
    "#393348": "#d8cbea",
    "#3c4050": "#dad6e5",
    "#3d3e4b": "#c9c3d2",
    "#424450": "#cec8d7",
    "#4b465c": "#cbc0dc",
    "#2f303a": "#e3dee9",
    "#302f3a": "#e4deea",
    "#e9e8ef": "#2b2932",
    "#f5f2ff": "#26212f",
    "#f5f3f9": "#27232e",
    "#f4f2f8": "#2b2933",
    "#f4f1fb": "#282330",
    "#f6f1ff": "#282330",
    "#faf8ff": "#302735",
    "#eeeaf9": "#302a3b",
    "#e1e2ee": "#302d3a",
    "#d5d5e2": "#454052",
    "#d7d9ec": "#393448",
    "#e8deff": "#443765",
    "#aeb4c3": "#686575",
    "#898b9a": "#72707d",
    "#777988": "#7a7582",
    "#abaebc": "#666275",
    "#a9a5b8": "#746b84",
    "#a9a8b6": "#646071",
    "#b8b9c5": "#55515f",
    "#9294a4": "#77717e",
    "#9aa0af": "#6b6776",
    "#aaabb9": "#746d81",
    "#a1a1ae": "#706a79",
    "#bab8ca": "#504a60",
    "#cbbaff": "#8d77cf",
    "#b8a2ff": "#8f78d6",
    "#baa6ff": "#9175d1",
    "#d2c4ff": "#ae9be6",
    "#e7e0ff": "#ded3ff",
    "#f3eeff": "#ebe4ff",
    "#171322": "#282033",
    "#1a142a": "#251c33",
    "#231a3a": "#ffffff",
    "#ccc2ff": "#604992",
    "#dbcfff": "#594184",
    "#b6a0ff": "#8f73c9",
    "#646578": "#a69daf",
    "#606370": "#a8a2b0",
    "#81729d": "#a28db9",
}


def style_for_theme(theme):
    if theme != "light":
        return STYLE
    import re

    return re.sub(
        r"#[0-9a-fA-F]{6}",
        lambda match: _LIGHT_COLORS.get(match.group().lower(), match.group()),
        STYLE,
    )
