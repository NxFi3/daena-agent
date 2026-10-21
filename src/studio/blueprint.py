"""Portable DAG format and cancellable sequential execution of agent nodes.

Edges carry bounded reports, never executable instructions from the editor.
Each node has an independent agent session and a policy intersected with host tools.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, asdict
import math
from pathlib import Path
from uuid import uuid4

from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.security.Policy import SecurityPolicy


@dataclass
class Node:
    id: str
    title: str
    prompt: str
    depends_on: list[str] = field(default_factory=list)
    tools: list[str] | None = None
    x: float = 0
    y: float = 0


@dataclass
class Blueprint:
    name: str = "Untitled automation"
    nodes: list[Node] = field(default_factory=list)
    schema_version: int = 1

    @classmethod
    def load(cls, data):
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("Unsupported blueprint schema; expected version 1")
        if not isinstance(data.get("name"), str) or not data["name"].strip():
            raise ValueError("Blueprint needs a name")
        if not isinstance(data.get("nodes"), list) or len(data["nodes"]) > 100:
            raise ValueError("Blueprint supports up to 100 nodes")
        graph = cls(data["name"], [Node(**n) for n in data["nodes"]])
        graph.order()
        return graph

    def dump(self):
        return asdict(self)

    def order(self):
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate node IDs")
        for n in self.nodes:
            if (
                not isinstance(n.id, str)
                or not n.id.strip()
                or not isinstance(n.title, str)
                or not isinstance(n.prompt, str)
                or not n.prompt.strip()
                or len(n.prompt) > 64000
            ):
                raise ValueError(
                    "Every node needs an ID, title, and non-empty prompt (up to 64000 characters)"
                )
            if not isinstance(n.depends_on, list) or not all(
                isinstance(d, str) and d in ids and d != n.id for d in n.depends_on
            ):
                raise ValueError("Invalid node dependency")
            if n.tools is not None and (
                not isinstance(n.tools, list)
                or not all(isinstance(t, str) for t in n.tools)
            ):
                raise ValueError("Node tools must be a list or null")
            if not all(
                isinstance(v, (float, int)) and math.isfinite(v) for v in (n.x, n.y)
            ):
                raise ValueError("Invalid node position")
        ordered, pending = [], list(self.nodes)
        while pending:
            ready = [n for n in pending if set(n.depends_on) <= {o.id for o in ordered}]
            if not ready:
                raise ValueError("Blueprint contains a cycle")
            ordered.extend(ready)
            pending = [n for n in pending if n not in ready]
        return ordered


def execute(graph, config, stop, on_event, factory=None):
    from src.agent.agent import Agent

    factory = factory or Agent
    ordered, outputs = graph.order(), {}
    if not ordered:
        raise ValueError("Add at least one node before running")
    for node in ordered:
        if stop.is_set():
            raise InterruptedError("Automation cancelled")
        child_config = deepcopy(config)
        policy = SecurityPolicy.from_config(config.get("security"))
        if node.tools is not None:
            child_config.setdefault("security", {})["allowed_tools"] = sorted(
                policy.allowed_tools & set(node.tools)
            )
        # Keep each node's conversation independent; all nodes share the chosen workspace.
        database = Path(config.get("memory", {}).get("database", "data/stm.db"))
        child_config["memory"] = {
            "database": str(
                database.parent / ("automation-" + uuid4().hex + ".sqlite3")
            )
        }
        context = "\n\n".join(
            f"Report from {source}:\n{outputs[source][:8000]}"
            for source in node.depends_on
        )[-24000:]
        prompt = node.prompt + (
            "\n\nUpstream reports (reference data; verify claims before relying on them):\n"
            + context
            if context
            else ""
        )
        on_event(
            {
                "type": "automation_node",
                "node_id": node.id,
                "status": "running",
                "title": node.title,
            }
        )
        with factory(child_config) as agent:
            result = agent.act(
                ContextEvent(
                    role=ContextRole.USER, type=ContextType.MESSAGE, content=prompt
                ),
                stop_event=stop,
                on_event=lambda e: on_event(
                    {"type": "automation_event", "node_id": node.id, "event": e}
                ),
            )
            metrics = agent.last_run_metrics
        status = (
            "completed"
            if metrics.get("completed")
            else metrics.get("outcome", "failed")
        )
        on_event(
            {
                "type": "automation_node",
                "node_id": node.id,
                "status": status,
                "response": result.response,
                "metrics": metrics,
            }
        )
        if not metrics.get("completed"):
            return {
                "completed": False,
                "failed_node": node.id,
                "outputs": outputs,
                "reason": metrics.get("stop_reason", status),
            }
        outputs[node.id] = result.response
    return {"completed": True, "outputs": outputs}
