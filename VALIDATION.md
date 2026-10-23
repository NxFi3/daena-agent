# Validation report

Performed on 2026-10-23 in a Linux cloud workspace, Python 3.12.

## Executed checks

- **66 package tests passed**, with PySide6 and the real X11 desktop test enabled.
- **230 existing Daena 0.6 tests passed** after installing the package and optional
  runtime hooks. No baseline tests were weakened or edited.
- Host smoke passed through an actual Daena Agent loop with a scripted model:
  registry discovery, workspace binding, one-shot screenshot forwarding, and
  Ollama/OpenRouter/Gemini serialization were exercised.
- Ruff checks, formatting and Python compilation passed.
- Source/wheel build and an isolated installed-wheel smoke are included in the
  release packaging checks; the ZIP contains the complete source and wheel.

## What the tests establish

The X11 test runs a real PySide6 application on Xvfb and sends native input through
xdotool. The application records received events independently of tool receipts.
Assertions cover `Daena سلام` text entry, checkbox state, one Save click, a visible
drag stroke, button release, and scroll events. No model or website is involved.
Screenshots in `docs/screenshots/` show this actual fixture and its trace viewer.

Contract tests cover stale/consumed/cross-session snapshots, ambiguous
target handling, coordinate bounds and finite numbers, resized windows, exact
postconditions, incomplete accessibility trees, partial typing, timeouts, no replay,
idempotency conflicts/crash receipts, cancellation, lock contention, text budgets,
pixel-to-zoom mappings, protected field values, and post-dispatch logging errors.

Cua adapter tests exercise complete JSON replies, error envelopes, exact window
identity, combined screenshot/AX capture, drag-context preparation, process-group
cleanup, closed stdin and cancellation of blocked helpers.

An existing Daena command-process timing test was transiently slow during one
parallel verification run. It passed alone (0.85 s), and the subsequent full
baseline suite passed all 230 tests. No code outside the optional integration
hooks was changed to accommodate that timing result.

## Limits

- The user's real Cua binary and native Wayland desktop were unavailable. Their
  end-to-end behavior is **not** represented by the X11 fixture test.
- No live Ollama/hosted vision model was invoked. These tests do not benchmark
  reasoning quality or establish parity with Astra.
- X11 fallback is foreground-only, has no AX tree, and may be occlusion-sensitive.
- The provided CI workflow targets Python 3.11/3.12/3.13, but only Python 3.12 was
  executed in this session. The workflow has not been published/run remotely here.
- Snapshot guards serialize participating agents and detect window geometry/
  revision changes; they do not observe every external human/OS interaction.

## Reproduce

```bash
python -m pip install -e '.[dev,x11,viewer]' 'google-genai>=2,<3'
DAENA_TEST_DESKTOP=1 xvfb-run -a python -m pytest -q
python -m ruff check computer_use tests examples install_into_daena.py
python -m ruff format --check computer_use tests examples install_into_daena.py
python -m compileall -q computer_use tests examples install_into_daena.py
python -m build
```

For the host smoke, install into a disposable Daena 0.6 checkout with
`--wire-runtime --enable`, install that host's dependencies, then run:

```bash
PYTHONPATH=/path/to/daena python tests/host_smoke.py
```
