# Observation/action contract and migration

## Capture and targeting

`capture` selects an exact PID/window_id or an unambiguous app/title match. Without
a selector it retains the last live target, otherwise uses a uniquely active or
sole visible window. Desktop capture requires explicit `scope="desktop"`.
Modes are `som` (pixels+controls), `vision` (pixels) and `ax` (controls).

Responses include `snapshot_id`, `window`, `coordinate_frame`, native dimensions,
`elements`, `image_ref`, `raw_image_ref`, `pixel_hash`, counts, truncation and warnings.
Opaque element IDs are preferred; numeric IDs require the matching snapshot too.
Exact label/role matching never guesses among multiple controls. Use `query` and
`max_results` for compact observations. Unknown coordinate provenance/scale means
AX bounds are omitted; native tokens remain usable. SELECTED never implies CHECKED.

Every input consumes its snapshot. The default freshness limit is 180 seconds
(`max_snapshot_age` constructor setting). Capturing again replaces the snapshot.
Another participating agent's input invalidates an earlier desktop revision.
Human/OS changes are not fully tracked by this counter; review fresh observations.

## Inputs

`click`, `double_click`, `right_click`, `middle_click`, `move` and `scroll` require
one `element`, `target` or native `coordinate`. Scroll requires direction and an
explicit anchor. `type` accepts text with an optional target; otherwise it uses the
focused field. Native token editing follows the driver's set-value/insert semantics
and is never split into replacement chunks. Keyboard typing uses 400-character
chunks and stops at the first failure, reporting the acknowledged prefix.

`key` accepts a single key or keys chord. `drag` accepts both coordinate endpoints,
button and duration. Cua drag preparation refreshes pixels only (its drag contract
has no capture_id field), compares native pixels and dispatches one gesture.
`move` is available on X11; Cua reports unsupported because the original documented
driver contract has no move action. No backend silently escalates input delivery.

`zoom` takes `region=[x,y,width,height]` and scale. It returns crop origin and exact
scale factors; input coordinates remain in the original image frame. It does not
consume the snapshot. `capture_after=false` skips observation, so capture explicitly
before another input. Resized/disappearing windows reject old coordinates.

## Verification

`dispatch_status`: `not_dispatched`, `dispatched`, `partial` or `unknown`.
`success` means the requested tool operation completed, not the user's overall task.
Visual deltas describe pixel change without claiming semantic success.

Use an explicit postcondition:

```json
{"kind":"checked","target":{"label":"Enabled","role":"check box"},"value":true}
```

Kinds: `exists`, `absent`, `checked`, `selected`, `focused`, `value`, `window_title`.
Boolean states need boolean values; value/title use exact strings. The tool polls
observations for `seconds`, never repeats input. Missing state is `unavailable`;
multiple matches are `ambiguous`. Absence cannot be verified from an incomplete
tree. `wait` observes the existing target, optionally with a postcondition.

## Sequences, navigation and retries

`sequence` accepts 1–6 steps. Click/type/scroll require labels, freshly resolved
after each preceding observation. All step schemas are validated before input.
It stops on failure or an unmet explicit condition. Use per-step `expect` for
dependencies: the tool does not infer which semantic outcomes later steps need.

`navigate` operates on an already-selected browser using Ctrl+L, text and Return.
It does not guess a browser or infer navigation from a short user message. Delivery
is not a page-load guarantee; inspect the result or use `expect`. `launch` takes
one executable, never shell syntax. List/capture its actual window afterwards.

Same `request_id` + same arguments returns the original receipt without replay.
Different arguments yield a conflict. A started request without a finished receipt
reports unknown outcome and never re-executes. Receipts are not silently evicted.

`timeout` bounds the call including lock waits, readiness, input and observation.
Cancellation terminates local helper processes but cannot undo native input already
accepted by a remote daemon. Both uncertain input and partial typing require a fresh
observation before choosing a next action.

## Migration

Copy the entire package. `ComputerUseTool` remains public; old internal helper tool
classes are replaced by backend-neutral modules. Supply `snapshot_id`, read fresh
post-action state from `content.observation`, and use `verification.status` rather
than screenshot hashes to judge explicit conditions. The old heuristic repetition
and no-op tests are intentionally replaced by contract tests.
