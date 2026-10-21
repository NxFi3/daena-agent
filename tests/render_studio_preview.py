"""CI-only UI previews with labelled sample data; does not call any model."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
# Return a bounded source-format snapshot for review when no local executor is available.
subprocess.run([sys.executable,"-m","ruff","format","src/studio","tests/test_studio.py","tests/test_studio_interactions.py","tests/render_studio_preview.py"],check=True)
changed=subprocess.check_output(["git","diff","--name-only"]).decode().splitlines()
snapshot={name:Path(name).read_text() for name in changed if name.startswith("src/studio/") or name in {"tests/test_studio.py","tests/test_studio_interactions.py","tests/render_studio_preview.py"}}
print("STUDIO_FORMAT_SNAPSHOT:"+base64.b64encode(json.dumps(snapshot).encode()).decode(),flush=True)

import faulthandler
faulthandler.dump_traceback_later(25,exit=True)
from PySide6.QtCore import QByteArray,QBuffer,QIODevice
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from src.studio.window import StudioWindow

app=QApplication([])
output=Path(".studio-previews");output.mkdir(exist_ok=True)

def capture(window,name):
    app.processEvents();QTest.qWait(120)
    pixmap=window.grab();pixmap.save(str(output/(name+".png")))
    data=QByteArray();buffer=QBuffer(data);buffer.open(QIODevice.WriteOnly);pixmap.save(buffer,"PNG")
    print("STUDIO_PREVIEW_"+name.upper()+":"+base64.b64encode(bytes(data)).decode(),flush=True)

with TemporaryDirectory() as directory:
    print("PREVIEW_PHASE: construct window",flush=True)
    window=StudioWindow(directory);print("PREVIEW_PHASE: window constructed",flush=True);window.resize(1440,940);window.show();app.processEvents()
    print("PREVIEW_PHASE: populate conversation",flush=True)
    window.transcript.clear()
    window.add_message("You","Build a polished dashboard for my software, and use two agents to review the result.")
    window.transcript.begin_turn()
    window.transcript.append_reasoning("Sample reasoning preview. Real provider output appears here when supplied.")
    window.transcript.add_detail("Preview data · no model calls were made.")
    window.transcript.add_detail("✓ read_file · inspected the application structure")
    window.transcript.finish("### The dashboard is ready for review\n\nThe workspace now has a responsive layout, a consistent design system, and a clear product overview.\n\n- **Interface agent** reviewed spacing and accessibility.\n- **Verification agent** checked the important interactions.\n\nRun the project locally:\n\n"+chr(96)*3+"bash\nnpm run dev\n"+chr(96)*3,"UI preview · sample content")
    window.transcript.end_turn()
    window.input.setPlainText("Make the pricing cards a little more compact.")
    capture(window,"conversation")
    window.graph.upsert("parent","Daena","Coordinator · preview data","running",x=0,y=150)
    for ident,name,mode,y,status in [("interface","Interface review","read_only",0,"completed"),("verification","Verification","worker",300,"running")]:
        window.on_child_event({"agent_id":ident,"name":name,"mode":mode,"event":{"type":"spawned","task":"Review the dashboard and report concrete findings. Preview data only."}})
        window.graph.nodes[ident].setPos(400,y)
        if status=="completed":
            window.on_child_event({"agent_id":ident,"name":name,"mode":mode,"event":{"type":"finished","status":status,"integration":"none","response":"Sample report: layout and controls were reviewed. This screenshot uses preview data, not live inference."}})
    window.nav.setCurrentRow(1);app.processEvents();window.graph.fit_graph();window.inspect_child("interface")
    capture(window,"graph")
    window.nav.setCurrentRow(4);window.settings.show_section(0);capture(window,"settings")
    window.close();app.processEvents()


faulthandler.cancel_dump_traceback_later()
