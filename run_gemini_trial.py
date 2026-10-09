from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/itsnxfi3/Desktop/daena-agent")
WORKSPACE = ROOT / "workspaces" / "house-price-dataset"
MODEL = os.getenv("DAENA_GEMINI_MODEL", "gemini-3.8-flash")
MAX_ITERATIONS = max(1, int(os.getenv("DAENA_GEMINI_MAX_ITERATIONS", "12")))
MAX_CONTINUATIONS = max(0, int(os.getenv("DAENA_GEMINI_MAX_CONTINUATIONS", "0")))
os.environ.setdefault("DAENA_CLI", "1")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

from src.agent.agent import Agent
from src.models.ContextEvent import ContextEvent, ContextPriority, ContextRole, ContextType
from cli.render import StreamRenderer

def main() -> int:
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY is not configured (value not printed).", flush=True)
        return 2

    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    # In-memory provider override only. The user's config.json is left unchanged.
    config["llm"] = {
        "provider": "gemini",
        "provider_config": {
            "model_name": MODEL,
            "generation_config": {"temperature": 0.2, "think": "medium"},
        },
    }
    # Bounded evaluation so free-tier API quota cannot be exhausted by an open-ended run.
    config["max_agent_iterations"] = MAX_ITERATIONS
    config["max_auto_continuations"] = MAX_CONTINUATIONS
    config["max_no_progress_segments"] = 2

    task_path = Path(os.getenv("DAENA_GEMINI_TASK_FILE", str(WORKSPACE / "TASK_PROMPT.md"))).expanduser().resolve()
    task_text = task_path.read_text(encoding="utf-8")
    agent = Agent(config)
    agent.set_workingdirectory(str(WORKSPACE))
    renderer = StreamRenderer(
        model=MODEL,
        workspace=str(WORKSPACE),
        session_id=str(agent.session_id),
        think_enabled="medium",
    )
    task = ContextEvent(
        role=ContextRole.USER,
        type=ContextType.MESSAGE,
        content=task_text,
        priority=ContextPriority.NORMAL,
        step=0,
        metadata={"source": "gemini-evaluation", "workspace": str(WORKSPACE)},
    )
    error = None
    result = None
    started = datetime.now(timezone.utc).isoformat()
    print(f"GEMINI_TRIAL_START={started}", flush=True)
    print(f"MODEL={MODEL}; MAX_ITERATIONS={MAX_ITERATIONS}; MAX_AUTO_CONTINUATIONS={MAX_CONTINUATIONS}; TASK_FILE={task_path.name}", flush=True)
    try:
        def emit(event: dict) -> None:
            renderer.handle(event)
            kind = str(event.get("type") or "")
            if kind in {"run_start", "iteration_start", "continuation", "run_end", "error"}:
                print(f"EVENT {kind}: " + json.dumps(event, ensure_ascii=False, default=str)[:1200], flush=True)
        result = agent.act(task, on_event=emit)
    except BaseException as exc:
        error = exc
        print(f"RUN_EXCEPTION={type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
    finally:
        renderer.finish(result=result, error=error)
        try:
            metrics = agent.last_run_metrics
        except Exception:
            metrics = {}
        out = {
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "provider": "gemini",
            "model": MODEL,
            "workspace": str(WORKSPACE),
            "max_iterations_per_segment": MAX_ITERATIONS,
            "max_auto_continuations": MAX_CONTINUATIONS,
            "error": None if error is None else f"{type(error).__name__}: {error}",
            "metrics": metrics,
            "result_type": type(result).__name__ if result is not None else None,
            "result_response_present": bool(getattr(result, "response", None)) if result is not None else False,
        }
        (ROOT / "run_logs").mkdir(parents=True, exist_ok=True)
        (ROOT / "run_logs" / "gemini_divar_trial_summary.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print("TRIAL_SUMMARY=" + json.dumps(out, ensure_ascii=False, default=str), flush=True)
        agent.close()
    return 1 if error is not None else 0

if __name__ == "__main__":
    raise SystemExit(main())
