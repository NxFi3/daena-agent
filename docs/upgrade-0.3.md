# Upgrade from Daena 0.2.0 to 0.3.0

Extract the new source archive into a new directory. Do not launch `python -m cli`
from the old `daena_agent-0.2.0` folder: Python would import that old source even
after installing a newer wheel.

For the Linux workspace shown in the reported issue, download the source archive
into `~/Downloads`, then run these commands in the existing Python environment:

```bash
cd ~/Desktop
tar -xzf ~/Downloads/daena_agent-0.3.0.tar.gz
cd daena_agent-0.3.0
python -m pip install -e '.[ollama,web]'
python -m cli --config config.json --cwd /home/itsnxfi3/Desktop/test --model qwen3.6:35b-a3b-q4_K_M
```

Ollama must be running and that model must already be available (`ollama list`).
The new configuration enables isolated workers. Do not overwrite it wholesale
with the old configuration, whose `subagents.allow_writes` is false. Copy specific
provider settings or skill paths as needed. `/agents` shows enabled modes.

The workspace's existing generated files are preserved. Start a new conversation
and ask Daena to inspect and continue the existing project. Old memory records
remain readable, but cannot recover facts never recorded by the old runtime.

The old command `mkdir .../{css,js,images,fonts}` used literal argv syntax, so it
may have created a directory named `{css,js,images,fonts}`. Inspect the actual
workspace before cleaning it up; the upgrade does not remove project files.

## Changes

The runtime no longer imposes duplicate-action, semantic-recovery or guessed
verification gates. Canonical tool exchanges remain chronological and historical
summaries are incremental. Large historical tool arguments are bounded while
remaining valid JSON. Actual disk inventory and later receipts supersede summary
claims. Cancellation now closes pending Ollama HTTP requests, including requests
without a UI callback, and interrupts managed process waits.

Workers use separate snapshots and run concurrently. Use `diff` to review,
`merge` to integrate, or `discard` to release changes. A conflicting merge leaves
parent files unchanged. Host permissions and resource limits still apply.

These changes address the demonstrated runtime defects. They cannot guarantee
that every model will always choose useful actions or produce a correct project.
