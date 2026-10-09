from __future__ import annotations

import argparse
import copy
import json
import tempfile
import time
from pathlib import Path
from typing import Any

PROVIDER_DEFAULT_MODELS = {
    "ollama": "gpt-oss:20b",
    "gemini": "gemini-3.8-flash",
    "openrouter": "deepseek/deepseek-v4.1-flash",
}


def configure_run(config: dict[str, Any], provider: str | None, model: str | None) -> dict[str, Any]:
    """Return an isolated, provider-compatible config for one benchmark run."""
    run_config = copy.deepcopy(config)
    llm = run_config.setdefault("llm", {})
    active_provider = provider or llm.get("provider", "ollama")
    provider_changed = active_provider != llm.get("provider", "ollama")
    llm["provider"] = active_provider

    provider_config = llm.setdefault("provider_config", {})
    if model:
        provider_config["model_name"] = model
    elif provider_changed or not provider_config.get("model_name"):
        provider_config["model_name"] = PROVIDER_DEFAULT_MODELS.get(
            active_provider, provider_config.get("model_name", "")
        )

    generation_config = dict(provider_config.get("generation_config") or {})
    # The old fixed 1024 output cap is intentionally not part of the benchmark.
    generation_config.pop("num_predict", None)
    if active_provider != "ollama":
        generation_config.pop("num_thread", None)
        generation_config.pop("num_threads", None)
        generation_config.pop("num_ctx", None)
    provider_config["generation_config"] = generation_config
    return run_config

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
            root = Path(raw)
            workspace = root / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            if case.setup is not None:
                case.setup(workspace)

            run_config = copy.deepcopy(config)
            if "security" in case.tags:
                security_config = run_config.setdefault("security", {})
                security_config["workspace_only"] = True
                security_config["allow_network_tools"] = False
                security_config["force_approve"] = False

            agent = Agent(run_config)
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

            try:
                agent.close()
            finally:
                if case.cleanup is not None:
                    case.cleanup(workspace)

            metrics = metrics if isinstance(metrics, dict) else {}
            grouped_failures = metrics.get("tool_failures_by_error_type", {})
            if not isinstance(grouped_failures, dict):
                grouped_failures = {}
            run_metrics = {
                "iterations": int(metrics.get("iterations", 0) or 0),
                "tool_failures_by_error_type": dict(grouped_failures),
                "duplicate_cache_hits": int(
                    metrics.get("duplicate_observation_cache_hits", 0) or 0
                ),
                "tokens": int(metrics.get("tokens", 0) or 0),
                "wall_time_ms": elapsed_ms,
                "completed": bool(metrics.get("completed", False)),
            }
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
                    "run_metrics": run_metrics,
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

    failure_totals: dict[str, int] = {}
    for record in records:
        run_metrics = record.get("run_metrics") or {}
        grouped = run_metrics.get("tool_failures_by_error_type") or {}
        if isinstance(grouped, dict):
            for error_type, count in grouped.items():
                key = str(error_type or "unknown_error")
                failure_totals[key] = failure_totals.get(key, 0) + int(count or 0)

    duplicate_values = [
        int((record.get("run_metrics") or {}).get("duplicate_cache_hits", 0) or 0)
        for record in records
    ]
    average_duplicate_cache_hits = (
        round(sum(duplicate_values) / len(duplicate_values), 2)
        if duplicate_values else 0.0
    )

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
        "tool_failures_by_error_type": dict(sorted(failure_totals.items())),
        "average_duplicate_cache_hits": average_duplicate_cache_hits,
        "average_tool_blocks": avg("tool_blocks"),
        "average_duration_ms": round(
            sum(float(r["elapsed_ms"]) for r in records) / total, 2
        ) if total else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Daena baseline benchmark.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--provider", choices=("ollama", "gemini", "openrouter"))
    parser.add_argument("--model", help="Override model ID for the selected provider.")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output", default="baseline-results.json")
    parser.add_argument(
        "--case",
        action="append",
        dest="case_names",
        help="Run only selected case names.",
    )
    args = parser.parse_args()

    config = configure_run(
        load_config(args.config),
        provider=args.provider,
        model=args.model,
    )
    selected = [
        case for case in CASES
        if not args.case_names or case.name in args.case_names
    ]

    provider_name = (config.get("llm") or {}).get("provider", "ollama")
    provider_config = (config.get("llm") or {}).get("provider_config") or {}
    model_name = provider_config.get("model_name", "")

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for case in selected:
        records.extend(run_case(case, config, max(1, args.repetitions)))

    report = {
        "harness": "daena-baseline",
        "version": 2,
        "config": args.config,
        "provider": provider_name,
        "model": model_name,
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
