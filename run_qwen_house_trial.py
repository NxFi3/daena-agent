import json
import time
import traceback
from pathlib import Path
from src.agent.agent import Agent
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType

ROOT = Path("/home/itsnxfi3/Desktop/daena-agent")
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
MODEL = CONFIG["llm"]["provider_config"]["model_name"]
CONFIG["max_agent_iterations"] = 18
CONFIG["computer_use_max_iterations"] = 18
CONFIG["max_auto_continuations"] = 0
LOG_PATH = Path("/tmp/daena_qwen_house_trial.jsonl")
LOG_PATH.write_text("", encoding="utf-8")
AGENT = Agent(CONFIG)
AGENT.set_workingdirectory(str(ROOT))
EVENTS = []
STARTED = time.time()
TASK = (
    "Use Computer Use to draw a simple, recognizable house in the native Paint application "
    "already open on the desktop. First capture the app explicitly with app='Paint' and inspect "
    "the returned screenshot; do not select a browser tab just because its title contains the word Paint. "
    "Draw a square/rectangular house body, a triangular roof, one door, and two windows using actual GUI "
    "tools and mouse gestures only. If a toolbar control is missing from accessibility, use the labeled "
    "coordinate grid on the newest screenshot rather than retrying its label. After every interaction, "
    "inspect the new screenshot and change strategy if no visible progress occurs. Do not use shell, code, "
    "image APIs, or direct canvas manipulation to draw. Verify the finished drawing in a final screenshot "
    "before claiming success; if the iteration budget runs out, report what is and is not visibly complete."
)

def sink(event):
    if not isinstance(event, dict):
        return
    item = {key: event.get(key) for key in (
        "type", "iteration", "total_iterations", "segment", "name", "tool", "action",
        "decision", "reason", "next_action", "text", "completed", "stop_reason",
        "estimated_tokens", "messages", "success", "summary", "arguments"
    ) if key in event}
    # The house-only live regression logs safe, compact GUI evidence for debugging;
    # never dump arbitrary tool content or environment secrets into this artifact.
    payload = event.get("content")
    if isinstance(payload, dict):
        safe_keys = (
            "action", "mode", "window", "width", "height", "element_count", "elements_text",
            "image_ref", "coordinate_frame", "error", "recovery_hint", "input_dispatched",
            "target_label", "success", "complete", "completed_steps", "verification",
        )
        item["result_details"] = {key: payload[key] for key in safe_keys if key in payload}
    EVENTS.append(item)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    if item.get("type") in {
        "run_start", "iteration_start", "tool_call", "tool_result", "completion_review",
        "final_response", "run_end", "run_stopped", "context", "tool_search"
    }:
        print("EVENT", json.dumps(item, ensure_ascii=False, default=str)[:1400], flush=True)

try:
    print("LIVE_TRIAL_MODEL=", MODEL, flush=True)
    print("LIVE_TRIAL_TASK=", TASK, flush=True)
    result = AGENT.act(
        ContextEvent(role=ContextRole.USER, type=ContextType.MESSAGE, content=TASK),
        on_event=sink,
    )
    print("LIVE_TRIAL_ANSWER=", str(getattr(result, "response", "") or "")[:4000], flush=True)
    print("LIVE_TRIAL_METRICS=", json.dumps(AGENT.last_run_metrics, ensure_ascii=False, default=str), flush=True)
    print("LIVE_TRIAL_SECONDS=", round(time.time() - STARTED, 2), flush=True)
    print("LIVE_TRIAL_EVENT_COUNT=", len(EVENTS), flush=True)
except BaseException:
    traceback.print_exc()
finally:
    AGENT.close()
    print("LIVE_TRIAL_LOG=", str(LOG_PATH), flush=True)
