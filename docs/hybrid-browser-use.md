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

## Persistent sign-in profiles

Managed Chromium uses a **persistent workspace profile** by default.
Sessions stored in the same Daena workspace share browser login cookies
and local storage even when Studio opens a new conversation or restarts.
The browser does not reuse the user's personal Chrome profile.

`browser.profile_scope` can be set to `"conversation"` to isolate each
conversation and restore the old behaviour. The default is `"workspace"`.
The profile is stored under `.daena/browser_use/<workspace-derived-id>/profile`;
keep that directory private. Never commit it, copy it into public repos,
or share it with strangers: the Chrome profile can contain login credentials
and authentication cookies. Use a separate workspace for sites/accounts that
should not share sign-in state.

**Migration:** the new workspace profile is separate from older
conversation-specific profiles. You may need to sign in once after upgrading.
Subsequent conversations in the same workspace reuse the new profile.
Signing out from a site, expired cookies, server-side session revocation,
explicit browser-data deletion, or opening an unrelated workspace can
still require authentication again. Connected CDP sessions use the
external browser's context and are not affected by this profile setting.

Do not run two managed Daena Chromium instances with the same workspace
profile simultaneously: Chromium enforces a single active process per
profile. Stop the previous run/browser first.
