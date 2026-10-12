"""Print the RAW accessibility JSON of toolbar items so we can see how Cua Driver
really reports the checked/selected state.   python tools/dump_ax_states.py Paint Rectangle Pencil
"""
import json
import subprocess
import sys

title = sys.argv[1] if len(sys.argv) > 1 else "Paint"
labels = {x.casefold() for x in (sys.argv[2:] or ["Rectangle", "Pencil"])}


def call(name, args, timeout=20):
    done = subprocess.run(["cua-driver", "call", name, json.dumps(args)], capture_output=True,
                          text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    return done.stdout


wins = json.loads(call("list_windows", {"on_screen_only": True}))
wins = wins.get("windows", wins) if isinstance(wins, dict) else wins
win = next((w for w in wins if title.casefold() in f"{w.get('title','')} {w.get('app_name','')}".casefold()), None)
if not win:
    sys.exit(f"no window matching {title!r}")
state = json.loads(call("get_window_state", {
    "pid": win["pid"], "window_id": win["window_id"], "include_accessibility_tree": True,
    "include_screenshot": False, "max_elements": 5000, "max_depth": 20, "timeout_ms": 9000,
}, timeout=30))
print("top-level keys:", sorted(state))
for item in state.get("elements", []):
    if str(item.get("label", "")).casefold() in labels:
        print(json.dumps(item, ensure_ascii=False, indent=2))
