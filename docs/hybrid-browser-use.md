# Hybrid Browser Use: DOM and visual interactions

Daena Browser Use is a **local Playwright-driven browser**, not the TinyFish
hosted Web Agent. Its Agent Loop takes a natural-language task and decides how
to observe, act, and verify; Browser Use provides browser actions and evidence.

Use an inexpensive DOM observation for ordinary forms, buttons, labels, tabs,
iframes and open shadow roots. Prefer exact element refs and semantic selectors.
For canvas, unlabeled controls, or layouts that cannot be reliably selected,
use `browser_use(action="screenshot")`. The returned observation includes
`image_ref`, `snapshot_id`, and `screenshot_size` in CSS pixels.

Visual input examples:

```json
{"action":"click_at","snapshot_id":"web_...","coordinate":[240,150]}
```

```json
{"action":"move_at","snapshot_id":"web_...","coordinate":[300,210]}
```

```json
{"action":"drag_at","snapshot_id":"web_...","from_coordinate":[100,120],"to_coordinate":[420,380]}
```

A visual input must refer to the latest screenshot of the active tab. An
intervening page observation or input invalidates its snapshot. A change of
URL or viewport size is also rejected. Out-of-bounds points never dispatch.
These guards reduce but do not eliminate UI-change races between screenshots
and pointer actions.

Each successful action returns a fresh observation. It is not proof that a
workflow objective has been met. Verify a visible postcondition (DOM text,
selected value, dialog, or fresh screenshot). Do not blindly replay actions
after a timeout.

TinyFish offers a separate hosted natural-language Web Agent, with remote
Chromium/CDP, profiles, optional structured output, streaming run progress and
recordings. Daena intentionally does not claim anti-bot bypass, proxying,
hosted browser isolation, or identical success rates. Relevant design ideas
are goal-driven orchestration, hybrid semantic/visual input, and explicit
observable completion criteria. See https://www.tinyfish.ai/browser and
https://playwright.dev/mcp/snapshots for general references.

Run `DAENA_TEST_BROWSER=1 python -m pytest -q tests/browser_use` with
Playwright Chromium installed. Inspect CI before deploying to schools or
other production environments.
