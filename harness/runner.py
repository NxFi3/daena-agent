from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.agent.agent import Agent
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from harness.cases import CASES, BenchmarkCase


def load_config(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_case(
    case: BenchmarkCase,
    config: dict[str, Any],
    repetitions: int,
) -> list[dict[str, Any]]:
    results = []

    for repetition in range(1, repetitions + 1):
        with tempfile.TemporaryDirectory(prefix="daena-bench-") as raw:
            workspace = Path(raw)
            if case.setup is not None:
                case.setup(workspace)

            agent = Agent(config)
            agent.set_workingdirectory(str(workspace))
            event = ContextEvent(
                role=ContextRole.USER,
                type=ContextType.MESSAGE,
                content=case.prompt,
            )

            started = time.perf_counter()
            try:
                response = agent.act(event)
                elapsed_ms = round(
                    (time.perf_counter() - started) * 1000.0,
                    2,
                )
                verified, verification_note = case.verifier(workspace)
                metrics = agent.last_run_metrics
            except Exception as exc:
                elapsed_ms = round(
                    (time.perf_counter() - started) * 1000.0,
                    2,
                )
                response = None
                verified = False
                verification_note = f"harness exception: {type(exc).__name__}: {exc}"
                metrics = agent.last_run_metrics

            agent.close()

            results.append(
                {
                    "case": case.name,
                    "repetition": repetition,
                    "verified": verified,
                    "verification_note": verification_note,
                    "response": (
                        getattr(response, "response", "") if response is not None else ""
                    ),
                    "elapsed_ms": elapsed_ms,
                    "metrics": metrics,
                    "tags": list(case.tags),
                }
            )

    return results


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    completed = sum(bool(r["metrics"].get("completed")) for r in records)
    verified = sum(bool(r["verified"]) for r in records)

    def avg(key: str) -> float:
        values = [float(r["metrics"].get(key, 0) or 0) for r in records]
        return round(sum(values) / len(values), 2) if values else 0.0

    return {
        "runs": total,
        "completed_runs": completed,
        "verified_runs": verified,
        "completion_rate": round(completed / total, 4) if total else 0.0,
        "verification_rate": round(verified / total, 4) if total else 0.0,
        "average_iterations": avg("iterations"),
        "average_llm_calls": avg("llm_calls"),
        "average_tokens": avg("tokens"),
        "average_tool_calls": avg("tool_call_attempts"),
        "average_tool_failures": avg("tool_failures"),
        "average_tool_blocks": avg("tool_blocks"),
        "average_duration_ms": round(
            sum(float(r["elapsed_ms"]) for r in records) / total, 2
        ) if total else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Daena baseline benchmark.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output", default="baseline-results.json")
    parser.add_argument(
        "--case",
        action="append",
        dest="case_names",
        help="Run only selected case names.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    selected = [
        case for case in CASES
        if not args.case_names or case.name in args.case_names
    ]

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for case in selected:
        records.extend(run_case(case, config, max(1, args.repetitions)))

    report = {
        "harness": "daena-baseline",
        "version": 1,
        "config": args.config,
        "duration_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "summary": summarize(records),
        "records": records,
    }

    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
