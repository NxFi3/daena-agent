"""Disposable test window: logs every mouse event it REALLY receives.

Run:  python drag_probe_app.py          (leave it open)
It writes ~/.cache/daena_drag_probe.log  (one line per press/motion/release).
Pure Tk, no extra packages. Close the window yourself when done.
"""
import os
import time
import tkinter as tk

LOG = os.path.expanduser("~/.cache/daena_drag_probe.log")
os.makedirs(os.path.dirname(LOG), exist_ok=True)
open(LOG, "w").close()

root = tk.Tk()
root.title("DragProbe")
root.geometry("700x500+200+120")
status = tk.Label(root, text="no events yet", anchor="w")
status.pack(fill="x")
canvas = tk.Canvas(root, bg="white", highlightthickness=0)
canvas.pack(fill="both", expand=True)
state = {"x": None, "y": None, "n": 0}


def log(kind, e):
    with open(LOG, "a") as f:
        f.write(f"{time.time():.3f} {kind} {e.x} {e.y}\n")


def press(e):
    state.update(x=e.x, y=e.y, n=0)
    log("press", e)


def motion(e):
    state["n"] += 1
    if state["x"] is not None:
        canvas.create_line(state["x"], state["y"], e.x, e.y, width=3, fill="blue")
        state.update(x=e.x, y=e.y)
    status.config(text=f"drag moves={state['n']} at ({e.x},{e.y})")
    log("motion", e)


def release(e):
    log("release", e)
    status.config(text=f"released after {state['n']} moves at ({e.x},{e.y})")
    state["x"] = None


canvas.bind("<ButtonPress-1>", press)
canvas.bind("<B1-Motion>", motion)
canvas.bind("<ButtonRelease-1>", release)
root.mainloop()
