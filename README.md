# Daena Computer Use 1.0

A replacement for the `computer_use` package supplied in `comp.zip`. The public
`ComputerUseTool` name stays; the observation/action contract is redesigned.
This is a desktop-control engine, **not a trained vision model**.

[راهنمای فارسی](README.fa.md) · [Protocol](docs/protocol.md) ·
[Integration](docs/integration.md) · [Validation](VALIDATION.md)

## Changes

- Unique snapshot-bound element IDs; explicit `snapshot_id` on input.
- A desktop lease across threads, processes, sessions and workspaces. Another
  participating agent's input invalidates an older observation.
- No click-count lockouts, website-specific rankings, task classifiers, guessed
  titlebar offsets, or automatic replay of failed input.
- Exact label/role matching; ambiguous matches stop before input.
- Immutable raw screenshots, separate numbered overlays, native dimensions,
  coordinate frames, visual deltas and zoom-to-native mappings.
- Delivery is separate from postcondition verification. An unchanged screen can
  be a valid click; a changed screen does not prove task completion.
- Bounded/cancellable driver calls and durable idempotency receipts. Long keyboard
  text reports partial delivery; native element replacement text is not chunked.
- Cua Driver is primary. Optional Linux/X11 foreground control is tested against
  a real Qt app, including Unicode, click, scroll and drag.
- A PySide6 **Observatory** provides a read-only timeline and zoomable screenshots.

## Install

Python 3.11+:

```bash
python -m pip install -e '.[viewer]'
daena-computer doctor
```

The primary backend requires the **external `cua-driver` executable** used by the
original package. It is not bundled. Run `cua-driver doctor` in the logged-in
desktop session. For the optional X11 backend, install `.[x11,viewer]` and the
system package `xdotool`, then use `--backend x11`. X11 requires DISPLAY and explicit
`delivery_mode="foreground"`; it has no accessibility tree and cannot control
native Wayland-only windows. Window pixels can be affected by occlusion.

## Install into Daena

Close Daena, install the dependencies above, then run:

```bash
python install_into_daena.py --repo ~/Desktop/daena-0.6.0 --wire-runtime --enable
```

The installer copies all modules, backs up replaced files, adds the tool to
repository configs and optionally adds workspace/cancellation/image-forwarding
hooks tested against Daena 0.6.0. Unknown host layouts fail before writing. Existing
workspace profiles keep their own tool selections; enable `computer_use` there.

If your newer Daena already binds workspace/cancellation and forwards `image_ref`
to the model, omit `--wire-runtime` to retain its existing visual pipeline:

```bash
python install_into_daena.py --repo /path/to/daena --enable
```

Restart Daena. A vision-capable model and screenshot forwarding are required for
pixel reasoning; AX-only operations can use accessible labels without images.
The selected/default model is not changed.

## Use

```python
from computer_use import ComputerUseTool

tool = ComputerUseTool(workspace="/path/to/project")
observation = tool.execute("capture", app="Firefox", mode="som")
if not observation.success:
    raise RuntimeError(observation.summary)
result = tool.execute(
    "click", snapshot_id=observation.content["snapshot_id"],
    target={"label": "Enabled", "role": "check box"},
    expect={"kind": "checked", "target": "Enabled", "value": True},
    seconds=3, request_id="enable-option-1",
)
```

Use a label from the actual observation; the example does not assume that this
control exists on a particular site. CLI requests can come from stdin:

```bash
daena-computer --workspace /path/to/project capture
daena-computer --workspace /path/to/project call < request.json
daena-computer --workspace /path/to/project serve
daena-computer-viewer /path/to/project
```

`serve` is a persistent JSONL request/response bridge. The viewer sends no input
and runs no model: wheel zooms, dragging pans, Fit frames the image and 100%
restores native scale.

## Storage and cancellation

Artifacts are in `<workspace>/.daena/computer_use/<session>/` in private directories.
Retention is 150 PNGs per session; the event log rotates at 2 MB. Request receipts
remain until explicitly removed, so old request IDs cannot become executable
again. Action logs omit typed text; screenshots/accessible field values can still
contain on-screen information. Password-role values are omitted from AX records.

Call `tool.cancel()` from another thread, or inject `set_cancellation_event(event)`.
Use `reset_cancel()` before a new task when using the tool's own event. The Daena
hook observes its parent stop token too. Stopping the client cannot undo native
input already accepted by a daemon: uncertainty is reported, never replayed.

## Development

```bash
python -m pip install -e '.[dev,x11,viewer]'
python -m pytest -q
python -m ruff check computer_use tests examples install_into_daena.py
python -m build
# Requires system xvfb, xdotool and xauth; use an isolated display:
DAENA_TEST_DESKTOP=1 xvfb-run -a python -m pytest -q
```

The desktop fixture asserts the app's received input events. Tests do not establish
live model reasoning quality or validate the user's actual Wayland/Cua installation.
