# Upgrade to Daena Studio 0.6.0

This update focuses on desktop usability. The default model remains
qwen3.6:35b-a3b-q4_K_M; automation execution semantics are unchanged.

## Install

Use Python 3.11+ and a running Ollama server for the default provider.

```bash
conda activate AI
cd ~/Desktop
git clone --branch release-0.6.0 --single-branch https://github.com/NxFi3/daena-agent.git daena-0.6.0
cd daena-0.6.0
python -m pip install -e '.[desktop,ollama,openrouter,gemini,web]'
daena-studio --cwd ~/Desktop/test
```

If you already have a Git checkout, commit or stash your own changes before
switching branches. Existing workspace profiles and conversation archives remain
usable. Source folders extracted from older ZIP files can be kept separately.

## Conversation

Enter sends the message. Shift+Enter inserts a newline. Ctrl+Enter is also
supported. Enter does not send while an input method has active preedit text.
Ctrl+L focuses the composer.

Responses stream inside one message card and the final response replaces the
draft, avoiding duplicate copies. Reasoning and execution details expand on
demand. Copy buttons copy plain message text. Run details provides activity,
context estimates and a raw stream for troubleshooting.

The sidebar groups saved turns by conversation and supports text search. Opening
a conversation restores its available archived turns, then continues its stored
session. The run archive format and SQLite history are unchanged.

## Model and tools

The conversation toolbar contains provider, editable model, reasoning and tools.
Changes apply between turns and persist in the workspace profile. Switching
providers restores independent provider settings instead of carrying a local
Ollama endpoint or options into a cloud request.

Reasoning supports Automatic, Off, On, Low, Medium, High and custom native
settings. Model support varies: Ollama receives think, OpenRouter receives
reasoning effort, and Gemini effort uses thinking budgets (1024/4096/8192 tokens;
Off uses zero). Gemini models that require thinking_level can use native
thinking_config in advanced provider options. Explicit native thinking_config
takes precedence over the common setting.

The searchable tool selector uses actual runtime tool names. Settings also exposes
web access, managed processes and file boundaries. Empty selections remain empty;
children cannot exceed the host's tool scope.

Settings is organized into Model & reasoning, Context & memory, Delegation &
skills, Tool permissions and Advanced profile. Ordinary parameters use form
controls; raw JSON is optional.

## Agent graph

- Wheel: zoom at the pointer, without requiring Ctrl.
- Drag the background: pan the canvas.
- Drag a node: move it and update its connected edges.
- Middle-button drag or Space+drag: pan even over a node.
- Plus/minus: zoom; 0: reset to 100%; F: fit the graph; Escape: clear selection.
- Visible minus, plus, 100% and Fit graph controls provide the same operations.

Canvas scrollbars are hidden; navigation is handled directly by the canvas.
Automation uses the same improved canvas without adding new execution features.
