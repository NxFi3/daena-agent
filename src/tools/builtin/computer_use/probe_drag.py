"""Drive Daena's Computer Use tool against drag_probe_app.py WITHOUT any LLM.

1) In one terminal:  python drag_probe_app.py
2) In the repo root: python tools/probe_drag.py
It prints what Daena reported AND what the app really received (ground truth).
"""
import os
import sys

sys.path.insert(0, os.getcwd())
from src.tools.builtin.computer_use.computer_use import ComputerUseTool  # noqa: E402

LOG = os.path.expanduser("~/.cache/daena_drag_probe.log")


def events():
    try:
        return [line.split() for line in open(LOG).read().splitlines() if line.strip()]
    except OSError:
        return []


tool = ComputerUseTool()
tool.set_workspace(os.getcwd())
tool.set_context_store(None, lambda: "drag-probe")

cap = tool.execute(action="capture", mode="vision", window_title="DragProbe")
print("CAPTURE success:", cap.success)
print(str(cap.summary)[:600])
if not cap.success:
    sys.exit("capture failed - fix this first (driver/session), drag cannot be tested")

for mode in ("background", "foreground"):
    before = len(events())
    res = tool.execute(action="drag", from_coordinate=[150, 150], to_coordinate=[450, 300],
                       duration_ms=800, delivery_mode=mode)
    got = events()[before:]
    kinds = [e[1] for e in got]
    print(f"\n=== delivery_mode={mode}")
    print("tool success      :", res.success)
    print("tool summary      :", str(res.summary)[:300])
    print("recovery_hint     :", res.metadata.get("recovery_hint"))
    print(f"app really got    : press={kinds.count('press')} motion={kinds.count('motion')} release={kinds.count('release')}")
    if got:
        print("first/last event  :", got[0][1:], got[-1][1:])
