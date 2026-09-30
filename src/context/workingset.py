# src.context/workingset.py
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from src.models.ToolCall import ToolCall
from src.models.ToolResult import ToolResult


class WorkingSet:
    """
    Compact model-facing execution state.

    Memory keeps the complete runtime history.
    WorkingSet keeps durable facts and evidence needed for the
    next model decision.
    """

    MUTATING_ACTIONS = {
        "create",
        "add",
        "modify",
        "update",
        "delete",
        "remove",
        "write",
        "rename",
        "move",
    }

    MAX_ARTIFACTS = 8
    MAX_ARTIFACT_PREVIEW_CHARS = 3000

    MAX_FACTS = 10
    MAX_UNRESOLVED = 6
    MAX_RECENT_ACTIONS = 10
    MAX_OBSERVATIONS = 3
    MAX_PROCESSES = 6
    MAX_VERIFICATION_EXCERPT_CHARS = 3200

    # Common command flags that change presentation rather than
    # the underlying verification target.
    NORMALIZED_COMMAND_FLAGS = {
        "-q",
        "-qq",
        "-v",
        "-vv",
        "-vvv",
        "--quiet",
        "--verbose",
        "-s",
    }

    def __init__(self, workspace_root: str | None = None) -> None:
        self.workspace_root: Path | None = None
        if workspace_root:
            self.set_workspace(workspace_root)
        self.reset()

    def set_workspace(self, directory: str) -> None:
        self.workspace_root = Path(directory).expanduser().resolve()

    def reset(self) -> None:
        self.artifacts: OrderedDict[
            str,
            dict[str, Any],
        ] = OrderedDict()

        self.facts: list[str] = []

        self.unresolved: list[str] = []

        self._unresolved_keys: dict[str, str] = {}

        self.verification: dict[str, Any] = {}

        # Preserve the latest failed verification independently from the
        # currently running or unrelated command. A later search, lsof, or
        # cleanup command must not erase the diagnostic needed for recovery.
        self.last_failed_verification: dict[str, Any] = {}

        self.recent_actions: list[dict[str, Any]] = []

        self.observations: list[dict[str, Any]] = []

        # Compact process receipts keep active and recently finished managed
        # processes visible to the next model turn without exposing internals.
        self.processes: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def update(
        self,
        tool_call: ToolCall,
        result: ToolResult,
        iteration: int,
    ) -> bool:
        """
        Update the working set from one tool execution.

        Returns True when the result represents a workspace mutation.
        """

        if not isinstance(result, ToolResult):
            return False

        tool_name = str(getattr(tool_call, "name", result.name)).strip()

        action = str(getattr(tool_call, "action", "execute")).strip()

        target = str(getattr(tool_call, "target", "")).strip()

        self._record_recent_action(
            tool_name=tool_name,
            action=action,
            target=target,
            result=result,
            iteration=iteration,
        )

        self._record_observation(result)
        self._update_process_state(result=result, iteration=iteration)

        for effect in result.effects:
            self._apply_effect(
                effect=effect,
                result=result,
                iteration=iteration,
            )

        self._update_verification(
            tool_call=tool_call,
            result=result,
        )

        self._update_facts(
            tool_call=tool_call,
            result=result,
        )

        if result.success:
            self._clear_related_failure(
                tool_call=tool_call,
                result=result,
            )
        else:
            self._record_failure(
                tool_call=tool_call,
                result=result,
            )

        return self._result_changes_workspace(
            tool_call=tool_call,
            result=result,
        )

    def _record_recent_action(
        self,
        tool_name: str,
        action: str,
        target: str,
        result: ToolResult,
        iteration: int,
    ) -> None:

        metadata = result.metadata if isinstance(result.metadata, dict) else {}

        is_duplicate = bool(metadata.get("duplicate_action", False))
        is_read_only_disabled = bool(metadata.get("read_only_disabled", False))

        item: dict[str, Any] = {
            "iteration": iteration,
            "tool": tool_name,
            "action": action,
            "target": target,
            "summary": result.summary,
        }

        if is_duplicate:
            item["outcome"] = "duplicate_blocked"
        elif is_read_only_disabled:
            item["outcome"] = "read_only_disabled"
        elif result.success:
            item["outcome"] = "success"
        else:
            item["outcome"] = "failure"

        self.recent_actions.append(item)

        if len(self.recent_actions) > self.MAX_RECENT_ACTIONS:
            del self.recent_actions[
                : len(self.recent_actions) - self.MAX_RECENT_ACTIONS
            ]

    def _record_observation(
        self,
        result: ToolResult,
    ) -> None:

        observation = result.to_observation()

        new_path = observation.get("evidence", {}).get("path")

        if new_path:
            for i, existing in enumerate(self.observations):
                existing_path = existing.get("evidence", {}).get("path")
                if existing_path == new_path:
                    self.observations.pop(i)
                    break

        self.observations.append(observation)

        if len(self.observations) > self.MAX_OBSERVATIONS:
            del self.observations[: len(self.observations) - self.MAX_OBSERVATIONS]

    def _update_process_state(
        self,
        result: ToolResult,
        iteration: int,
    ) -> None:
        content = result.content if isinstance(result.content, dict) else {}
        process_id = str(content.get("process_id", "") or "").strip()
        if not process_id:
            return

        status = str(content.get("status", "") or "").strip().lower()
        receipt = dict(self.processes.get(process_id, {}))
        receipt.update(
            {
                "process_id": process_id,
                "status": status or receipt.get("status", ""),
                "exit_code": content.get("exit_code", receipt.get("exit_code")),
                "pid": content.get("pid", receipt.get("pid")),
                "iteration": iteration,
                "tool": result.name,
                "success": result.success,
            }
        )

        for key in ("command", "workdir"):
            if content.get(key) is not None:
                receipt[key] = content[key]
        for key in ("stdout", "stderr"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                receipt[key] = self._truncate(value, 1200)

        self.processes.pop(process_id, None)
        self.processes[process_id] = receipt
        while len(self.processes) > self.MAX_PROCESSES:
            self.processes.popitem(last=False)

    def _apply_effect(
        self,
        effect: dict[str, str],
        result: ToolResult,
        iteration: int,
    ) -> None:

        action = str(effect.get("action", "")).strip().lower()

        raw_target = str(effect.get("target", "")).strip()

        if not raw_target:
            return

        target = self._canonical_path(raw_target)

        if target in self.artifacts:
            artifact = dict(self.artifacts[target])
        else:
            artifact = {
                "status": "unknown",
                "known": False,
            }

        artifact["last_operation"] = action

        artifact["last_iteration"] = iteration

        preview = self._extract_preview(
            result=result,
            target=target,
            raw_target=raw_target,
        )

        if action in {
            "inspect",
            "read",
        }:
            artifact["status"] = "known"
            artifact["known"] = True

            if preview:
                artifact["preview"] = preview

                artifact["preview_truncated"] = (
                    len(preview) >= self.MAX_ARTIFACT_PREVIEW_CHARS
                )

        elif action in {
            "create",
            "add",
        }:
            artifact["status"] = "created"
            artifact["known"] = bool(preview)

            if preview:
                artifact["preview"] = preview

                artifact["preview_truncated"] = (
                    len(preview) >= self.MAX_ARTIFACT_PREVIEW_CHARS
                )

        elif action in {
            "modify",
            "update",
            "write",
        }:
            artifact["status"] = "modified"
            artifact["known"] = bool(preview)

            if preview:
                artifact["preview"] = preview

                artifact["preview_truncated"] = (
                    len(preview) >= self.MAX_ARTIFACT_PREVIEW_CHARS
                )
            else:
                artifact.pop("preview", None)
                artifact.pop("preview_truncated", None)

        elif action in {
            "delete",
            "remove",
        }:
            artifact["status"] = "deleted"
            artifact["known"] = False

            artifact.pop("preview", None)
            artifact.pop("preview_truncated", None)

        else:
            artifact["status"] = action

        self._touch_artifact(
            target=target,
            artifact=artifact,
        )

    def _touch_artifact(
        self,
        target: str,
        artifact: dict[str, Any],
    ) -> None:

        target = self._canonical_path(target)

        if target in self.artifacts:
            del self.artifacts[target]

        self.artifacts[target] = artifact

        while len(self.artifacts) > self.MAX_ARTIFACTS:
            self.artifacts.popitem(last=False)

    def _extract_preview(
        self,
        result: ToolResult,
        target: str,
        raw_target: str,
    ) -> str:

        evidence = result.evidence

        if not isinstance(evidence, dict):
            return ""

        evidence_path = str(evidence.get("path", "")).strip()

        if evidence_path:

            normalized_evidence_path = self._canonical_path(evidence_path)

            normalized_target = self._canonical_path(target)

            if normalized_evidence_path == normalized_target and isinstance(
                evidence.get("content"),
                str,
            ):
                return self._truncate(
                    evidence["content"],
                    self.MAX_ARTIFACT_PREVIEW_CHARS,
                )

        files = result.content.get("files")

        if not isinstance(files, list):
            return ""

        for item in files:

            if not isinstance(item, dict):
                continue

            item_target = str(item.get("path") or item.get("target") or "").strip()

            if not item_target:
                continue

            if self._canonical_path(item_target) != self._canonical_path(target):
                continue

            content = item.get("content")

            if isinstance(content, str) and content:
                return self._truncate(
                    content,
                    self.MAX_ARTIFACT_PREVIEW_CHARS,
                )

            preview = item.get("content_preview")

            if isinstance(preview, str) and preview:
                return self._truncate(
                    preview,
                    self.MAX_ARTIFACT_PREVIEW_CHARS,
                )

        return ""

    def _update_verification(
        self,
        tool_call: ToolCall,
        result: ToolResult,
    ) -> None:

        content = result.content

        if not isinstance(content, dict):
            return

        if "exit_code" not in content:
            return

        stdout = content.get("stdout", "")
        stderr = content.get("stderr", "")

        output_parts: list[str] = []

        if isinstance(stdout, str) and stdout.strip():
            output_parts.append(stdout.strip())

        if isinstance(stderr, str) and stderr.strip():
            output_parts.append(stderr.strip())

        output = "\n".join(output_parts)
        status = str(content.get("status", "") or "").strip().lower()
        terminal = (
            status in {"exited", "terminated"}
            and content.get("exit_code") is not None
        )

        verification = {
            "tool": str(getattr(tool_call, "name", result.name)),
            "command": content.get("command"),
            "workdir": content.get("workdir"),
            "success": result.success,
            "exit_code": content.get("exit_code"),
            "exit_code_hint": content.get("exit_code_hint", ""),
            "status": status,
            "timed_out": content.get("timed_out", False),
            "duration_ms": content.get("duration_ms"),
            "output_excerpt": self._truncate(
                output,
                self.MAX_VERIFICATION_EXCERPT_CHARS,
            ),
        }

        self.verification = verification

        # A failed verification is durable state. Keep it until the same
        # verification scope succeeds, even when unrelated commands run.
        if terminal and isinstance(content.get("command"), list):
            scope_key = self._execution_scope_key(
                tool_call=tool_call,
                result=result,
            )

            if result.success:
                if self.last_failed_verification.get("scope_key") == scope_key:
                    self.last_failed_verification = {}
            else:
                failed = dict(verification)
                failed["scope_key"] = scope_key
                self.last_failed_verification = failed

    def _update_facts(
        self,
        tool_call: ToolCall,
        result: ToolResult,
    ) -> None:

        if not result.summary:
            return

        fact = str(result.summary).strip()

        if not fact:
            return

        if fact in self.facts:
            self.facts.remove(fact)

        self.facts.append(fact)

        if len(self.facts) > self.MAX_FACTS:
            del self.facts[: len(self.facts) - self.MAX_FACTS]

    def _record_failure(
        self,
        tool_call: ToolCall,
        result: ToolResult,
    ) -> None:

        message = result.summary.strip()

        if not message:
            return

        source_key = self._execution_scope_key(
            tool_call=tool_call,
            result=result,
        )

        target = str(getattr(tool_call, "target", "")).strip()

        if target:
            message = f"{target}: {message}"

        existing_messages = [
            item
            for item in self.unresolved
            if self._unresolved_keys.get(item) == source_key
        ]

        for item in existing_messages:
            self.unresolved.remove(item)
            self._unresolved_keys.pop(item, None)

        self.unresolved.append(message)
        self._unresolved_keys[message] = source_key

        if len(self.unresolved) > self.MAX_UNRESOLVED:

            removed = self.unresolved[: len(self.unresolved) - self.MAX_UNRESOLVED]

            del self.unresolved[: len(self.unresolved) - self.MAX_UNRESOLVED]

            for item in removed:
                self._unresolved_keys.pop(item, None)

    def _clear_related_failure(
        self,
        tool_call: ToolCall,
        result: ToolResult,
    ) -> None:

        source_key = self._execution_scope_key(
            tool_call=tool_call,
            result=result,
        )

        validation_source_key = "tool:" + json.dumps(
            {
                "tool": str(getattr(tool_call, "name", result.name)).strip().lower(),
                "action": str(getattr(tool_call, "action", "")).strip().lower(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

        remaining: list[str] = []

        for item in self.unresolved:

            item_key = self._unresolved_keys.get(item)
            if item_key == source_key or item_key == validation_source_key:
                self._unresolved_keys.pop(item, None)
                continue

            remaining.append(item)

        self.unresolved = remaining

    def _execution_scope_key(
        self,
        tool_call: ToolCall,
        result: ToolResult,
    ) -> str:
        """
        Produce a stable identity for an execution scope.

        This lets:
            pytest -q
        and:
            pytest -q -vv

        refer to the same verification scope.

        It does not make unrelated commands equivalent.
        """

        tool_name = str(getattr(tool_call, "name", result.name)).strip().lower()

        content = result.content

        if isinstance(content, dict) and "command" in content:
            command = content.get("command")

            if isinstance(command, list):
                normalized_command: list[str] = []

                for part in command:
                    part = str(part)
                    if part in self.NORMALIZED_COMMAND_FLAGS:
                        continue
                    normalized_command.append(part)

                workdir = str(content.get("workdir", "") or "").strip()

                return "command:" + json.dumps(
                    {
                        "tool": tool_name,
                        "command": normalized_command,
                        "workdir": (self._canonical_path(workdir) if workdir else ""),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )

        target = str(getattr(tool_call, "target", "")).strip()

        if not target:
            return "tool:" + json.dumps(
                {
                    "tool": tool_name,
                    "action": str(getattr(tool_call, "action", "")).strip().lower(),
                },
                ensure_ascii=False,
                sort_keys=True,
            )

        return "target:" + json.dumps(
            {
                "tool": tool_name,
                "target": self._canonical_path(target) if target else "",
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _result_changes_workspace(
        self,
        tool_call: ToolCall,
        result: ToolResult,
    ) -> bool:

        if not result.success:
            return False

        call_action = str(getattr(tool_call, "action", "")).strip().lower()

        if call_action in self.MUTATING_ACTIONS:
            return True

        for effect in result.effects:
            action = str(effect.get("action", "")).strip().lower()
            if action in self.MUTATING_ACTIONS:
                return True

        return bool(result.metadata.get("workspace_changed", False))

    def context(self) -> dict[str, Any]:
        return {
            "artifacts": dict(self.artifacts),
            "verification": dict(self.verification),
            "last_failed_verification": dict(self.last_failed_verification),
            "processes": dict(self.processes),
            "facts": list(self.facts),
            "unresolved": list(self.unresolved),
        }

    def process_context(self) -> dict[str, Any]:
        return {"items": list(self.processes.values())}

    def observation_context(self) -> dict[str, Any]:
        return {
            "items": list(self.observations),
        }

    def recent_actions_context(self) -> dict[str, Any]:
        return {
            "items": list(self.recent_actions),
        }

    def _canonical_path(self, value: str) -> str:

        value = str(value or "").strip()

        if not value:
            return ""

        if " " in value and not value.startswith("/") and not value.startswith("."):
            return value

        try:
            path = Path(value).expanduser()
            if not path.is_absolute() and self.workspace_root is not None:
                path = self.workspace_root / path
            return str(path.resolve(strict=False))

        except (OSError, RuntimeError):
            return value

    @staticmethod
    def _truncate(value: str, limit: int) -> str:

        value = str(value)

        if len(value) <= limit:
            return value

        return (
            value[: limit - 40].rstrip()
            + "\n"
            + f"... {len(value) - (limit - 40)} chars omitted ..."
        )
