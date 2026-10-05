from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from src.models.ToolCall import ToolCall
from src.models.ToolResult import ToolResult


@dataclass(frozen=True)
class GuardDecision:
    action: str = "allow"  # allow | warn | block
    code: str = "allow"
    message: str = ""
    count: int = 0

    @property
    def should_block(self) -> bool:
        return self.action == "block"


class ToolLoopGuard:
    """
    Runtime guard against repeated tool behavior that does not produce progress.

    The guard deliberately uses cheap deterministic signals only:
      - exact call + identical result repetition
      - repeating multi-call cycles
      - repeated mutations after a failed verification

    It does not try to judge whether arbitrary successful code changes are
    semantically correct. That remains a model responsibility.
    """

    IDENTICAL_WARN_AFTER = 2
    IDENTICAL_BLOCK_AFTER = 4

    CYCLE_WARN_AFTER = 2
    CYCLE_BLOCK_AFTER = 4
    MAX_CYCLE_PERIOD = 4
    HISTORY_LIMIT = 32

    MUTATION_WARN_AFTER = 4
    MUTATION_BLOCK_AFTER = 7
    REPEATABLE_TOOLS = frozenset({"process_poll"})

    _MUTATING_TOOLS = frozenset({
        "apply_patch",
        "process_write",
        "command_exec",
    })

    _VERIFICATION_MARKERS = frozenset({
        "test",
        "tests",
        "pytest",
        "jest",
        "vitest",
        "mocha",
        "check",
        "lint",
        "build",
        "typecheck",
        "type-check",
        "verify",
    })

    _VOLATILE_RESULT_KEYS = frozenset({
        "duration_ms",
        "pid",
    })

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last_signature: str | None = None
        self._last_result_hash: str = ""
        self._identical_count = 0

        self._history: deque[tuple[str, str]] = deque(maxlen=self.HISTORY_LIMIT)

        self._verification_failed = False
        self._verification_processes: set[str] = set()
        self._mutation_attempts: dict[str, int] = defaultdict(int)

    @staticmethod
    def _canonical_args(call: ToolCall) -> dict[str, Any]:
        args = getattr(call, "args", {}) or {}
        return args if isinstance(args, dict) else {}

    @classmethod
    def _signature(cls, call: ToolCall) -> str:
        payload = {
            "name": str(getattr(call, "name", "")).strip().lower(),
            "args": cls._canonical_args(call),
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )

    @classmethod
    def _result_hash(cls, result: ToolResult) -> str:
        content = result.content if isinstance(result.content, dict) else {}
        if isinstance(content, dict):
            content = {
                key: value
                for key, value in content.items()
                if key not in cls._VOLATILE_RESULT_KEYS
            }

        payload = {
            "success": bool(result.success),
            "content": content,
            "summary": str(result.summary or ""),
        }

        raw = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @classmethod
    def _mutation_targets(cls, call: ToolCall) -> list[str]:
        name = str(getattr(call, "name", "")).strip().lower()
        args = cls._canonical_args(call)

        if name == "apply_patch":
            patch = args.get("patch", "")
            if isinstance(patch, str):
                paths = re.findall(
                    r"^\*\*\*\s+(?:Add|Update|Delete) File:\s*(.+?)\s*$",
                    patch,
                    flags=re.MULTILINE,
                )
                if paths:
                    return sorted({path.strip() for path in paths if path.strip()})

        if name == "process_write":
            process_id = str(args.get("process_id", "")).strip()
            return [f"process:{process_id}"] if process_id else []

        if name == "command_exec":
            workdir = str(args.get("workdir", "") or ".").strip()
            command = args.get("command")
            command_text = (
                " ".join(str(item) for item in command)
                if isinstance(command, list)
                else str(command or "")
            )
            return [f"command:{workdir}:{command_text}"]

        target = str(getattr(call, "target", "") or "").strip()
        return [target] if target else []

    @classmethod
    def _is_verification_command(cls, command: Any) -> bool:
        if not isinstance(command, list):
            return False

        tokens = [str(item).strip().lower() for item in command if str(item).strip()]
        if not tokens:
            return False

        basenames = {token.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for token in tokens}
        return any(
            marker in basenames
            or any(marker == part for part in re.split(r"[^a-z0-9_-]+", token) if part)
            for token in tokens
            for marker in cls._VERIFICATION_MARKERS
        )

    def before_call(self, call: ToolCall) -> GuardDecision:
        signature = self._signature(call)
        name = str(getattr(call, "name", "")).strip().lower()


        if name not in self.REPEATABLE_TOOLS:
            cycle = self.cycle_decision()
            if cycle.should_block:
                return cycle

        if (
            name not in self.REPEATABLE_TOOLS
            and signature == self._last_signature
            and self._identical_count >= self.IDENTICAL_BLOCK_AFTER
        ):
            return GuardDecision(
                action="block",
                code="identical_call_loop",
                count=self._identical_count,
                message=(
                    f"Blocked {call.name}: the same tool call produced the same "
                    f"result {self._identical_count} times. Inspect the existing "
                    "result and change strategy instead of repeating it."
                ),
            )

        targets = self._mutation_targets(call)
        if (
            self._verification_failed
            and targets
            and any(
                self._mutation_attempts[target] >= self.MUTATION_BLOCK_AFTER
                for target in targets
            )
        ):
            return GuardDecision(
                action="block",
                code="mutation_no_progress",
                count=max(self._mutation_attempts[target] for target in targets),
                message=(
                    f"Blocked {call.name}: this workspace target has been modified "
                    f"{max(self._mutation_attempts[target] for target in targets)} "
                    "times since the last verification failure. Run the failing "
                    "verification again and inspect its concrete output before "
                    "making another edit."
                ),
            )

        return GuardDecision()

    def after_call(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        workspace_changed: bool,
    ) -> GuardDecision:
        signature = self._signature(call)
        result_hash = self._result_hash(result)

        if signature == self._last_signature and result_hash == self._last_result_hash:
            self._identical_count += 1
        else:
            self._identical_count = 1

        self._last_signature = signature
        self._last_result_hash = result_hash

        self._history.append((signature, result_hash))

        name = str(getattr(call, "name", "")).strip().lower()
        content = result.content if isinstance(result.content, dict) else {}

        # Track verification processes so their later poll can close the
        # verification attempt.
        process_id = str(content.get("process_id", "")).strip()
        if name == "command_exec" and self._is_verification_command(content.get("command")):
            if content.get("status") == "running" and process_id:
                self._verification_processes.add(process_id)
            elif content.get("status") in {"exited", "terminated"}:
                self._verification_processes.discard(process_id)

        verification = False
        verification_success = False

        if name == "command_exec":
            verification = self._is_verification_command(content.get("command"))
            if verification and content.get("status") in {"exited", "terminated"}:
                verification_success = result.success
        elif name == "process_poll" and process_id in self._verification_processes:
            status = str(content.get("status", "")).lower()
            if status in {"exited", "terminated"}:
                verification = True
                verification_success = result.success
                self._verification_processes.discard(process_id)

        if verification:
            if verification_success:
                self._verification_failed = False
                self._mutation_attempts.clear()
            else:
                self._verification_failed = True
                self._mutation_attempts.clear()

        if (
            not verification
            and name in self._MUTATING_TOOLS
            and workspace_changed
        ):
            for target in self._mutation_targets(call):
                self._mutation_attempts[target] += 1

        if (
            name not in self.REPEATABLE_TOOLS
            and signature == self._last_signature
            and self._identical_count >= self.IDENTICAL_BLOCK_AFTER
        ):
            return GuardDecision(
                action="block",
                code="identical_call_loop",
                count=self._identical_count,
                message=(
                    f"Blocked {call.name}: the same tool call produced the same "
                    f"result {self._identical_count} times. Change arguments, "
                    "use a different tool, or act on the evidence already collected."
                ),
            )

        if (
            name not in self.REPEATABLE_TOOLS
            and self._identical_count >= self.IDENTICAL_WARN_AFTER
        ):
            return GuardDecision(
                action="warn",
                code="identical_call_repeat",
                count=self._identical_count,
                message=(
                    f"{call.name} returned the same result {self._identical_count} "
                    "times with identical arguments. Do not repeat it unchanged; "
                    "use the result already available or change strategy."
                ),
            )

        if name in self._MUTATING_TOOLS and workspace_changed and self._verification_failed:
            targets = self._mutation_targets(call)
            count = max((self._mutation_attempts[target] for target in targets), default=0)
            if count >= self.MUTATION_WARN_AFTER:
                return GuardDecision(
                    action="warn",
                    code="mutation_no_progress",
                    count=count,
                    message=(
                        f"{call.name} has modified the same target repeatedly after a "
                        "failed verification (count={count}). Run the failing "
                        "verification again before making more edits."
                    ).format(count=count),
                )

        return GuardDecision()

    def cycle_decision(self) -> GuardDecision:
        history = list(self._history)
        for period in range(2, self.MAX_CYCLE_PERIOD + 1):
            if len(history) < period * 2:
                continue

            pattern = history[-period:]
            laps = 1
            while len(history) >= (laps + 1) * period:
                previous = history[-(laps + 1) * period : -laps * period]
                if previous != pattern:
                    break
                laps += 1

            if laps >= self.CYCLE_BLOCK_AFTER:
                return GuardDecision(
                    action="block",
                    code="repeating_cycle",
                    count=laps,
                    message=(
                        f"Blocked tool cycle: the same {period}-call sequence "
                        f"repeated {laps} times with identical results. Change "
                        "strategy instead of replaying the cycle."
                    ),
                )

            if laps >= self.CYCLE_WARN_AFTER:
                return GuardDecision(
                    action="warn",
                    code="repeating_cycle",
                    count=laps,
                    message=(
                        f"The same {period}-call tool cycle repeated {laps} times "
                        "with identical results. Change strategy instead of replaying it."
                    ),
                )

        return GuardDecision()
