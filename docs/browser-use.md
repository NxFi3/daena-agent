# Browser control

`browser_use` controls real Chromium through Playwright. `open(url)` starts a
visible browser, navigates and returns the actual page text and controls in one
call. It does not require Firefox to be open, desktop focus, a Cua driver, or a
vision model. Native applications remain the responsibility of `computer_use`.

## Install and enable

Close Daena before updating your Git checkout:

```bash
git switch feat/studio-conversation-redesign
git pull --ff-only origin feat/studio-conversation-redesign
python -m pip install -e '.[desktop,browser,ollama,openrouter,gemini,web]'
python -m playwright install chromium
daena-studio
```

With an empty `browser.executable_path`, Daena uses the Playwright browser when
available, otherwise an installed Chrome/Chromium/Edge executable. It does not
download a browser automatically. An explicitly configured missing executable is
reported instead of silently selecting another browser. `browser_use(action="doctor")`
reports the selected path and source. If the Playwright download is unavailable
in your location but Chrome is installed, leave the path empty and retry `open`.

On Linux, if Chromium reports missing system libraries, install them with
`python -m playwright install --with-deps chromium` (system-package installation
may require administrator access). Alternatively choose an installed Chromium
executable in Settings → Tool permissions → Browser. Firefox is not a Chromium
executable and cannot be attached through CDP.

New default configurations enable `browser_use` and `tool_search`. Existing
saved profiles retain their permissions: enable both tools and network tools in
Settings → Tool permissions. For the CLI, add those names to the existing
`security.allowed_tools` array in the active config. Do not replace the array
with just these two names if you want to keep other capabilities.

Browser settings are host configuration, not model tool arguments:

```json
"browser": {
  "headless": false,
  "executable_path": "",
  "cdp_url": ""
}
```

The optional `browser-control` builtin skill describes the general workflow. No
site-specific selectors, login rules, guessed targets or click-count lockouts
are built into the engine. The model still chooses actions.

## Interaction contract

```json
{"action":"open","url":"https://example.org"}
{"action":"fill","target":{"label":"Email"},"text":"user@example.org"}
{"action":"click","target":{"role":"button","name":"Continue"}}
{"action":"wait","text":"Account overview"}
```

The labels above are illustrative; use the elements actually observed on the
target page. Each observation contains tab IDs, frame IDs and opaque element
refs, text, visibility/focus/disabled state and field values. Password field
values are redacted in DOM observations. Other page text, tool arguments and
screenshots may contain private information; the usual conversation/history
storage and configured model provider still apply.

Use the latest `ref` or an exact semantic target (`role` + `name`, `label`,
`placeholder`, `text`). CSS is available for controls without good accessibility
labels. Multiple matches return an error with candidates; no first match is
silently clicked. Semantic locators use Playwright's accessible-name matching.
Observed display labels are best-effort DOM descriptions, not a full accessibility
tree. References point to the original DOM node, and detached/replaced nodes are
not retargeted to a different element.

`fill` replaces text without a prior click; `press` sends keys. `select`, `check`,
`hover`, `scroll`, `drag` and `upload` support normal form interaction. Click can
specify `button`, `click_count` and a target-relative `position` for canvas work.
Playwright waits for actionability; no forced hidden clicks or sleeps are used.
Use `wait` for an explicit expected text/element/load condition. Action delivery
and goal verification are separate fields; no pixel change is treated as success.

For iframes use an observed `frame_id` with a semantic target, or a returned ref.
Open shadow roots are traversed. Closed shadow roots and native browser UI need
visual/native interaction. `observe(query,offset,limit)` pages through controls;
`read(offset,limit)` reads longer text (`limit` is in hundreds of characters).
Observations collect up to 1,200 controls, show 40 by default and report truncation.
Exact locators can address elements outside that list. Current body text is bounded;
use scoped `read` for full text, including frame content.

`tabs`, `select_tab`, `close_tab`, `back`, `forward`, and `reload` manage navigation.
Popups receive IDs and are not automatically selected. JavaScript dialogs remain
pending until an explicit `dialog(accept=...)` call. Download links produce
`download_id`s; `downloads` lists them and `save_download` writes one to a **new**
workspace path. Upload/download paths cannot escape the workspace or overwrite
an existing destination.

## Sessions and context

A managed browser and its cookies remain alive between messages in one Agent
conversation. For an authorized login the user can supply a one-time code in the
next message or complete a manual challenge in the visible browser. Normal task
cleanup releases child agents and managed shell processes without closing this
conversation resource. Closing the Agent closes its managed browser.

Profiles are stored under `.daena/browser_use/<session-hash>/profile`, separated by
workspace and conversation. A new conversation cannot read the previous one's
observation or refs. Returning to a saved conversation can reuse its cookies when
the browser is opened again; tab restoration across process restarts is not
guaranteed. Browser settings changes reconnect the browser on its next use.

Only the current compact DOM observation is pinned before context budgeting.
Historical receipts keep action, URL, tab, outcome and errors, without duplicating
all old controls. Screenshots are opt-in via `screenshot`, supported by the Ollama,
OpenRouter and Gemini image adapters. Request a vision-capable model for screenshots.
They are budgeted before the provider call and replaced by the next observation;
base64 image data is never persisted in conversation history. Screenshot pixels
are CSS viewport pixels, while click positions are relative to the target element.

Timeouts/cancellation never automatically replay input. Cancelling in-flight
mutating input closes the managed automation context to abort outstanding driver
work; use `open` again to reconnect. Cancelling a read/wait keeps the browser.
External CDP browsers are disconnected, not closed. An input already delivered
before cancellation cannot be undone. Receipts explicitly retain uncertainty.

## Attach to an existing Chromium browser

Set `browser.cdp_url` to an explicitly enabled Chrome DevTools endpoint, usually
`http://localhost:9222`. This is optional; managed mode works without it. The
endpoint grants control of that browser, so keep it on a trusted local interface.
Use a dedicated Chrome/Chromium profile when enabling remote debugging.

`tabs` connects and lists existing pages. Select a tab explicitly to reuse it;
otherwise `open` creates a new tab instead of navigating unrelated user work.
The tool's `close` disconnects, preserving the externally owned browser and tabs.

## Verification and limits

Tests use a real Chromium process against a local HTTP application: a multi-turn
login, Persian typing, ambiguous/replaced elements, frames, open shadow DOM,
checkboxes/selects, popups, dialogs, canvas clicks, drag/scroll, upload/download,
cancellation, screenshots and CDP ownership. A scripted model drives the actual
Agent loop through tool discovery, one-time-code handoff and completion. These
are host/runtime tests, not evidence of a specific model's reasoning quality.

The user's actual website, anti-bot systems, Firefox/Cua session and live Qwen
inference are not tested here. No automation tool can guarantee every site works;
manual authentication challenges, closed native surfaces and model decisions can
still require intervention. The tool does not bypass those systems.

```bash
python -m pip install -e '.[dev,browser]'
python -m playwright install chromium
DAENA_TEST_BROWSER=1 python -m pytest -q tests/browser_use
```

To test an existing installation, set `DAENA_TEST_CHROMIUM=/usr/bin/chromium`.
CI installs Playwright's bundled Chromium and runs the real-browser tests in a
separate CI check alongside the Python and native-desktop suites.

## Embedded forms and overlays

Observations interleave controls from visible frames before pagination, with a
bounded collection budget per frame. Frame text and control counts are included
even when the main URL does not change. `in_viewport` and `occluded` describe
viewport intersection and a sampled hit test; they are evidence, not guarantees
that an action will succeed. Playwright still performs its actionability checks.
Use an observed ref to target an embedded control, or include its frame_id in a
semantic target. No site-specific login selectors or forced overlay clicks are
used.
