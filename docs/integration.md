# Integration

The package exports only `ComputerUseTool`; helpers are not independently exposed
to the model. Internal imports are relative. `compat.py` uses Daena's actual Tool
and ToolResult when present, otherwise standalone dataclasses.

Bind workspace, session and stop token using `set_workspace`, `set_context_store`
and `set_cancellation_event` or `set_process_manager`. Forward `metadata.image_ref`
as an actual image to a vision-capable model, not as a filename. Keep the control
list in `result.evidence`; a truncated summary alone is insufficient.

`integration.visual_message(result, provider, workspace)` encodes a PNG for Ollama
(`images`), OpenRouter/OpenAI-compatible (`image_url`), or the supplied Gemini hook
(`computer_image` → SDK `Part.from_bytes`). No API calls or credentials are involved.
Paths are restricted to the workspace's computer-use artifacts.

## Optional stock Daena 0.6 hooks

`install_into_daena.py --wire-runtime` adds four marked, idempotent hooks:

1. Bind workspace, process manager/stop token and a per-manager session ID.
2. Remember the most recent `computer_use` result.
3. Forward at most one pending image before the next model call, then clear it.
4. Convert Gemini image annotations into typed SDK parts.

Base64 is not written into canonical conversation history. Ollama's serializer
preserves images; OpenRouter preserves multimodal content. Stock 0.6 text-context
estimates do not accurately count provider image tokens. Leave sufficient context
budget for images; this package does not retrofit every host's token estimator.

The installer validates all planned edits before writing. Existing markers avoid
duplicate hooks. Unknown layouts abort. For newer hosts with an existing image
pipeline, omit `--wire-runtime`. `--enable` adds the tool to repository configs;
workspace profiles retain their existing selections. Desktop input is not confined
by a filesystem workspace boundary.

## Backups

Backups live at `.daena-computer-use-backups/<timestamp>/` in the host checkout.
The manifest lists replaced files and whether an old package existed. Restore
those files/package to undo installation, taking care to preserve later edits.
The installer changes no global packages or system configuration.

## Backend contract

Cua uses `cua-driver call <name> <json>` with start_session, list_windows,
get_window_state, get_desktop_state and the original click/drag/type/key/scroll
actions. Window capture must attest PID/window_id and capture_id and write the
requested PNG. Structured AX elements use token, role, label and optional bounds,
states/actions. Undeclared coordinate frames are not guessed. Error JSON/nonzero
exits fail; oversized responses are rejected rather than truncated into bad JSON.

Linux/X11 was tested with actual input. Cua was tested through protocol fixtures
and subprocess tests; the user's actual Cua daemon/Wayland compositor and a live
vision model were unavailable for end-to-end validation.
