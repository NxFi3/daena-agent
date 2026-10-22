"""Versioned local profiles and run archives; no Qt dependency."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from src.security.Policy import SecurityPolicy


def default_config():
    return json.loads(
        (Path(__file__).resolve().parents[1] / "default_config.json").read_text()
    )


def validate_config(value):
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a JSON object")
    for section in (
        "llm",
        "context",
        "security",
        "skills",
        "subagents",
        "runtime",
        "studio",
    ):
        if section in value and not isinstance(value[section], dict):
            raise ValueError(f"{section} must be an object")
    llm = value.get("llm", {})
    if llm.get("provider", "ollama") not in {"ollama", "openrouter", "gemini"}:
        raise ValueError("Choose ollama, openrouter, or gemini")
    if not isinstance(llm.get("provider_config", {}), dict):
        raise ValueError("provider_config must be an object")
    provider = llm.get("provider_config", {})
    if not isinstance(provider.get("generation_config", {}), dict):
        raise ValueError("generation_config must be an object")
    if not isinstance(provider.get("model_name", ""), str):
        raise ValueError("model_name must be a string")
    from src.agent.subagents import SubagentManager
    from src.skills.registry import SkillRegistry

    SecurityPolicy.from_config(value.get("security"))
    SubagentManager(value)
    SkillRegistry(value)
    context = value.get("context", {})
    for key in ("max_prompt_tokens", "recent_event_limit", "compaction_target_tokens"):
        minimum = 4 if key == "recent_event_limit" else 128
        if key in context and (type(context[key]) is not int or context[key] < minimum):
            raise ValueError(f"context.{key} is too small or not an integer")
    protocol = value.get("runtime", {}).get("completion_protocol", "implicit")
    if protocol not in {"implicit", "explicit"}:
        raise ValueError("Invalid completion_protocol")
    if (
        protocol == "explicit"
        and "task_finish"
        not in SecurityPolicy.from_config(value.get("security")).allowed_tools
    ):
        raise ValueError("Explicit completion requires task_finish")
    if (
        type(value.get("max_agent_iterations", 100)) is not int
        or not 1 <= value.get("max_agent_iterations", 100) <= 1000
    ):
        raise ValueError("max_agent_iterations must be 1–1000")
    return deepcopy(value)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()


class StudioStore:
    def __init__(self, workspace):
        self.root = Path(workspace).expanduser().resolve() / ".daena" / "studio"
        self.root.mkdir(parents=True, exist_ok=True)
        data_home = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
        self.chat_root = data_home.expanduser() / "daena" / "studio"
        self.chat_root.mkdir(parents=True, exist_ok=True)
        self._register_workspace()

    def _register_workspace(self):
        registry = self.chat_root / "workspaces.json"
        try:
            locations = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else []
            if not isinstance(locations, list):
                locations = []
            directory = str(self.root.parents[1])
            if directory not in locations:
                write_json(registry, locations + [directory])
        except (OSError, ValueError):
            pass

    @property
    def profile(self):
        return self.root / "profile.json"

    def load_config(self, fallback=None):
        config = (
            json.loads(self.profile.read_text())
            if self.profile.exists()
            else (fallback if fallback is not None else default_config())
        )
        return validate_config(config)

    def save_config(self, config):
        write_json(self.profile, validate_config(config))

    def runtime_config(self, config):
        result = deepcopy(config)
        result["workspace"] = {"directory": str(self.root.parents[1])}
        result["memory"] = {"database": str(self.root / "history.sqlite3")}
        return result

    def save_run(self, session, title, events, metrics):
        ident = uuid4().hex
        write_json(
            self.chat_root / "runs" / (ident + ".json"),
            {
                "schema_version": 1,
                "id": ident,
                "session_id": str(session),
                "workspace": str(self.root.parents[1]),
                "title": title[:160],
                "events": events,
                "metrics": metrics,
            },
        )
        return ident

    def save_automation_run(self, title, blueprint, events, metrics):
        """Automation receipts never become chat history entries."""
        ident = uuid4().hex
        write_json(
            self.root / "automation_runs" / (ident + ".json"),
            {
                "schema_version": 1,
                "id": ident,
                "title": title[:160],
                "blueprint": blueprint,
                "events": events,
                "metrics": metrics,
            },
        )
        return ident

    def delete_session(self, session_id):
        """Delete every archived turn for one conversation session."""
        target = str(session_id)
        deleted = 0
        for path, run in list(self.runs()):
            if str(run.get("session_id")) != target:
                continue
            try:
                path.unlink()
                deleted += 1
            except FileNotFoundError:
                continue
        return deleted

    def runs(self):
        # New archives are global; older workspace-local archives remain readable.
        roots = [self.chat_root / "runs", self.root / "runs"]
        registry = self.chat_root / "workspaces.json"
        try:
            known = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else []
            if isinstance(known, list):
                for location in known:
                    if isinstance(location, str):
                        roots.append(Path(location) / ".daena" / "studio" / "runs")
        except (OSError, ValueError):
            pass
        paths = {}
        for root in roots:
            for path in root.glob("*.json"):
                paths[str(path)] = path
        entries = []
        for path in sorted(paths.values(), key=lambda p: p.stat().st_mtime, reverse=True)[:500]:
            try:
                run = json.loads(path.read_text(encoding="utf-8"))
                if "session_id" not in run:
                    continue
                if "workspace" not in run and path.parent != self.chat_root / "runs":
                    run["workspace"] = str(path.parent.parent.parent.parent)
                entries.append((path, run))
            except (OSError, ValueError):
                continue
        return entries
