"""CI-only UI previews with labelled sample data; does not call any model."""

import faulthandler
import os
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
faulthandler.dump_traceback_later(25, exit=True)
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from src.studio.window import StudioWindow

app = QApplication([])
output = Path(".studio-previews")
output.mkdir(exist_ok=True)


def capture(window, name):
    app.processEvents()
    QTest.qWait(120)
    pixmap = window.grab()
    pixmap.save(str(output / (name + ".png")))


with TemporaryDirectory() as directory:
    os.environ["XDG_DATA_HOME"] = str(Path(directory) / "user-data")
    print("PREVIEW_PHASE: construct window", flush=True)
    window = StudioWindow(directory)
    print("PREVIEW_PHASE: window constructed", flush=True)
    window.resize(1440, 940)
    window.show()
    app.processEvents()
    capture(window, "welcome")
    window.resize(1000, 700)
    capture(window, "welcome-compact")
    window.resize(1440, 940)
    app.processEvents()
    print("PREVIEW_PHASE: populate conversation", flush=True)
    window.transcript.clear()
    window.add_message(
        "You",
        "Build a polished dashboard for my software, and use two agents to review the result.",
    )
    window.transcript.begin_turn()
    window.transcript.append_stream(
        "I’ll inspect the interface, then verify the changes with a focused test run."
    )
    window.transcript.live.add_tool_call(
        "read_file", {"file_path": "src/studio/window.py"}
    )
    window.transcript.live.add_tool_result(
        "read_file", True, "Inspected the current interface"
    )
    window.transcript.append_stream(
        "The layout is in place. I’m checking the interactions and message order now."
    )
    window.transcript.live.add_tool_call("command_exec", {"command": ["pytest", "-q"]})
    window.transcript.live.add_tool_result(
        "command_exec", True, "All targeted UI tests passed"
    )
    window.transcript.append_reasoning(
        "Sample reasoning preview. Real provider output appears here when supplied."
    )
    window.transcript.add_detail("Preview data · no model calls were made.")
    window.transcript.add_detail("✓ read_file · inspected the application structure")
    window.transcript.finish(
        "### The dashboard is ready for review\n\nThe workspace now has a responsive layout, a consistent design system, and a clear product overview.\n\n- **Interface agent** reviewed spacing and accessibility.\n- **Verification agent** checked the important interactions.\n\nRun the project locally:\n\n"
        + chr(96) * 3
        + "bash\nnpm run dev\n"
        + chr(96) * 3,
        "UI preview · sample content",
    )
    card = window.transcript.live
    window.transcript.end_turn()
    window.input.setPlainText("Make the pricing cards a little more compact.")
    capture(window, "conversation")
    card._toggle_activity()
    window.transcript._follow = False
    window.transcript.verticalScrollBar().setValue(0)
    capture(window, "conversation-expanded")
    card._toggle_activity()
    window.transcript._follow = True
    window.toggle_theme()
    capture(window, "conversation-light")
    window.toggle_theme()
    window.graph.upsert(
        "parent", "Daena", "Coordinator · preview data", "running", x=0, y=150
    )
    for ident, name, mode, y, status in [
        ("interface", "Interface review", "read_only", 0, "completed"),
        ("verification", "Verification", "worker", 300, "running"),
    ]:
        window.on_child_event(
            {
                "agent_id": ident,
                "name": name,
                "mode": mode,
                "event": {
                    "type": "spawned",
                    "task": "Review the dashboard and report concrete findings. Preview data only.",
                },
            }
        )
        window.graph.nodes[ident].setPos(400, y)
        if status == "completed":
            window.on_child_event(
                {
                    "agent_id": ident,
                    "name": name,
                    "mode": mode,
                    "event": {
                        "type": "finished",
                        "status": status,
                        "integration": "none",
                        "response": "Sample report: layout and controls were reviewed. This screenshot uses preview data, not live inference.",
                    },
                }
            )
    window.nav.setCurrentRow(1)
    app.processEvents()
    window.graph.fit_graph()
    window.inspect_child("interface")
    capture(window, "graph")
    window.toggle_theme()
    capture(window, "graph-light")
    window.toggle_theme()
    window.nav.setCurrentRow(2)
    window.automation.add_node()
    window.automation.title.setText("Research")
    window.automation.prompt.setPlainText(
        "Investigate the design and capture concrete requirements."
    )
    window.automation.add_node()
    window.automation.title.setText("Implementation")
    window.automation.prompt.setPlainText(
        "Build the feature using the research report."
    )
    first, second = window.automation.graph.nodes
    window.automation.wire(first.id, second.id)
    window.automation.canvas.fit_graph()
    capture(window, "automation")
    window.toggle_theme()
    capture(window, "automation-light")
    window.toggle_theme()
    window.nav.setCurrentRow(4)
    window.settings.show_section(0)
    capture(window, "settings")
    window.close()
    app.processEvents()


faulthandler.cancel_dump_traceback_later()
