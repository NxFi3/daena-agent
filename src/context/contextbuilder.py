from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.context.contextwindow import ContextWindow
from src.context.tokenbudget import TokenBudget
from src.engine.LlmProviderManager import LlmProvider
from src.models.ContextEvent import ContextEvent

DEFAULT_INSTRUCTION = (
    "You are Daena, an autonomous assistant and software engineering agent."
)

SYSTEM_INSTRUCTION_PATH = Path("AgentInstruction/systeminstruction.md")
EXPERIENCE_PATH = Path("AgentInstruction/experience.md")


def SystemInstructionReader() -> str:
    """Read once at ContextBuilder construction (rarely changes at runtime)."""

    try:
        text = SYSTEM_INSTRUCTION_PATH.read_text(encoding="utf-8").strip()

    except (FileNotFoundError, OSError):
        return DEFAULT_INSTRUCTION

    return text or DEFAULT_INSTRUCTION


def ExperienceReader() -> str:
    """
    Hand-curated lessons, edited directly by the user (or later by a
    self-evolving process). Re-read on every build_context() call, unlike
    the system instruction, so edits apply on the very next turn with no
    restart needed.
    """

    try:
        return EXPERIENCE_PATH.read_text(encoding="utf-8").strip()

    except (FileNotFoundError, OSError):
        return ""


class ContextBuilder:
    """
    Converts retrieved ContextEvents and runtime state into
    provider-visible messages.

    Responsibilities:
        - build the system message (instruction + experience + runtime)
        - reconstruct conversation messages from ContextEvents
        - preserve native assistant tool calls and tool_call_id
        - bound large tool outputs
        - enforce the token budget

    This class does NOT:
        - retrieve memory
        - search STM
        - perform embeddings / reranking
        - call an LLM
        - create long-term memory

    Design note (why agent_state / progress / working_set / observation /
    recent_actions are accepted but not rendered):

        Tool calls and tool results are replayed through the native
        tool-calling protocol as real chat messages (see
        `_build_conversation`). That means everything those blobs used to
        restate in the system prompt is ALREADY visible to the model in
        the conversation itself — same file contents, same errors, same
        outcomes. Rendering it a second time as JSON in the system message
        only duplicates tokens on every single iteration (and, because it
        changes every iteration, it also defeats prompt-prefix caching).
        The parameters are still accepted so ContextService/Loop do not
        need to change, but this builder no longer stuffs them into the
        prompt.
    """

    MAX_TOOL_CHARS = 8000
    OLD_TOOL_CHARS = 300
    FULL_TOOL_RESULTS = 6
    MAX_THINKING_CHARS = 2000

    MAX_EXPERIENCE_CHARS = 4000
    MAX_LEARNED_EXPERIENCE_CHARS = 3000

    _TEXT_FIELDS = (
        "content",
        "stdout",
        "stderr",
    )

    def __init__(
        self,
        config: dict[str, Any],
        llm_provider: LlmProvider,
    ) -> None:
        self.config = config
        self.llm = llm_provider

        self.window = ContextWindow()

        self.tokenbudget = TokenBudget(
            config,
            self.llm,
        )

        self.system_instruction = SystemInstructionReader()

    # ============================================================
    # JSON helpers
    # ============================================================

    @staticmethod
    def _safe_json(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, default=str)

        except Exception:
            return str(value)

    @staticmethod
    def _parse_json(value: Any) -> Any:
        if not isinstance(value, str):
            return value

        try:
            return json.loads(value)

        except (json.JSONDecodeError, TypeError, ValueError):
            return value

    # ============================================================
    # Utility
    # ============================================================

    @staticmethod
    def _last_index(
        messages: list[dict[str, Any]],
        role: str,
    ) -> int:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == role:
                return index

        return -1

    @staticmethod
    def _truncate(
        value: str,
        limit: int,
    ) -> str:
        value = str(value or "")

        if len(value) <= limit:
            return value

        if limit <= 64:
            return value[:limit]

        omitted = len(value) - limit

        return (
            value[: limit - 64].rstrip()
            + "\n\n"
            + f"... {omitted} characters omitted ..."
        )

    # ============================================================
    # Conversation reconstruction
    # ============================================================

    def _build_conversation(
        self,
        events: list[ContextEvent],
        task: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:

        messages: list[dict[str, Any]] = []

        seen_ids: set[str] = set()

        for event in events:

            if not isinstance(event, ContextEvent):
                continue

            event_id = str(event.id)

            if event_id in seen_ids:
                continue

            seen_ids.add(event_id)

            role = event.role.value
            event_type = event.type.value
            content = event.content
            metadata = event.metadata

            if not isinstance(metadata, dict):
                metadata = {}

            # --------------------------------------------------------
            # User message
            # --------------------------------------------------------

            if role == "user" and event_type == "message":
                text = str(content or "").strip()

                if text:
                    messages.append(
                        {
                            "role": "user",
                            "content": text,
                        }
                    )

            # --------------------------------------------------------
            # Assistant message
            # --------------------------------------------------------

            elif role == "assistant" and event_type == "message":
                message = self._assistant_message(
                    content=content,
                    metadata=metadata,
                )

                if message is not None:
                    messages.append(message)

            # --------------------------------------------------------
            # Assistant tool call
            #
            # Native tool_calls already live inside the assistant
            # message metadata, so we do NOT create another
            # provider-visible message here.
            # --------------------------------------------------------

            elif role == "assistant" and event_type == "tool_call":
                continue

            # --------------------------------------------------------
            # Tool result
            # --------------------------------------------------------

            elif role == "tool" and event_type == "tool_result":
                tool_payload = self._parse_json(content)

                if not isinstance(tool_payload, dict):
                    tool_payload = {
                        "content": str(tool_payload),
                        "success": False,
                    }

                tool_message: dict[str, Any] = {
                    "role": "tool",
                    "content": self._tool_payload(tool_payload),
                }

                tool_call_id = tool_payload.get("tool_call_id")

                if tool_call_id:
                    tool_message["tool_call_id"] = str(tool_call_id)

                tool_name = tool_payload.get("name")

                if tool_name:
                    tool_message["tool_name"] = str(tool_name)

                messages.append(tool_message)

            # --------------------------------------------------------
            # System event
            # --------------------------------------------------------

            elif role == "system":
                text = str(content or "").strip()

                if text:
                    messages.append(
                        {
                            "role": "system",
                            "content": text,
                        }
                    )

            # --------------------------------------------------------
            # Generic runtime event: persisted in STM but not
            # automatically injected.
            # --------------------------------------------------------

            elif event_type == "event":
                continue

        # Safety net: Loop persists the task to STM at the very start of
        # run(), so it is almost always already inside `events`. This only
        # fires if it somehow fell outside the retrieved/recent window and
        # search() also missed it (it shouldn't, since search is queried
        # with the task's own content).
        if isinstance(task, dict):

            task_id = task.get("id")
            task_text = str(task.get("content") or "").strip()

            if task_text and task_id is not None and str(task_id) not in seen_ids:
                messages.append(
                    {
                        "role": "user",
                        "content": task_text,
                    }
                )

        self._shrink_old_tool_results(messages)
        self._drop_old_thinking(messages)

        return messages

    # ============================================================
    # Assistant message
    # ============================================================

    def _assistant_message(
        self,
        content: Any,
        metadata: dict[str, Any],
    ) -> dict[str, Any] | None:

        raw = metadata.get("llm_message")

        if not isinstance(raw, dict):
            raw = {}

        message: dict[str, Any] = {
            "role": "assistant",
            "content": (raw.get("content") or content or ""),
        }

        tool_calls = raw.get("tool_calls")

        if tool_calls:
            message["tool_calls"] = tool_calls

        reasoning_details = raw.get("reasoning_details")

        if reasoning_details:
            message["reasoning_details"] = reasoning_details

        reasoning = raw.get("reasoning")

        if reasoning:
            message["reasoning"] = reasoning

        for key in ("refusal", "annotations", "audio"):
            value = raw.get(key)

            if value is not None:
                message[key] = value

        thinking = raw.get("thinking") or metadata.get("thinking")

        if thinking:
            message["thinking"] = self._truncate(
                str(thinking),
                self.MAX_THINKING_CHARS,
            )

        if (
            not str(message.get("content", "") or "").strip()
            and "tool_calls" not in message
        ):
            return None

        return message

    # ============================================================
    # Tool payload
    # ============================================================

    def _tool_payload(
        self,
        event_content: dict[str, Any],
    ) -> str:

        inner = event_content.get("content")

        if isinstance(inner, dict):
            payload = dict(inner)
        else:
            payload = {"message": str(inner or "")}

        payload["success"] = bool(event_content.get("success", False))

        blocks: list[str] = []

        for key in self._TEXT_FIELDS:

            value = payload.get(key)

            if isinstance(value, str):
                payload.pop(key, None)

                if value.strip():
                    blocks.append(f"[{key}]\n{value}")

        files = payload.get("files")

        if isinstance(files, list):
            slim_files: list[dict[str, Any]] = []

            for item in files:

                if not isinstance(item, dict):
                    continue

                slim_files.append(
                    {
                        key: value
                        for key, value in item.items()
                        if key not in ("content", "content_preview")
                    }
                )

                text = item.get("content")

                if isinstance(text, str) and text.strip():
                    blocks.append(f"[file: {item.get('path', '')}]\n{text}")

            payload["files"] = slim_files

        text = json.dumps(payload, ensure_ascii=False, default=str)

        if blocks:
            text += "\n\n" + "\n\n".join(blocks)

        if len(text) > self.MAX_TOOL_CHARS:
            text = text[: self.MAX_TOOL_CHARS] + "\n...[truncated]"

        return text

    # ============================================================
    # Tool result shrinking
    # ============================================================

    def _shrink_old_tool_results(
        self,
        messages: list[dict[str, Any]],
    ) -> None:

        tool_positions = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "tool"
        ]

        if len(tool_positions) <= self.FULL_TOOL_RESULTS:
            return

        for index in tool_positions[: -self.FULL_TOOL_RESULTS]:

            content = messages[index].get("content", "")

            if isinstance(content, str) and len(content) > self.OLD_TOOL_CHARS:
                messages[index]["content"] = (
                    content[: self.OLD_TOOL_CHARS] + " ...[old result truncated]"
                )

    # ============================================================
    # Thinking cleanup
    # ============================================================

    def _drop_old_thinking(
        self,
        messages: list[dict[str, Any]],
    ) -> None:

        last_user = self._last_index(messages, "user")

        for index, message in enumerate(messages):
            if index < last_user:
                message.pop("thinking", None)

    # ============================================================
    # Window population
    # ============================================================

    def _populate_window(
        self,
        events: list[ContextEvent],
        task: dict[str, Any] | None,
        workspace: str | None,
        learned_experience: str | None,
    ) -> None:

        self.window.set_system(self.system_instruction)

        experience = self._truncate(
            ExperienceReader(),
            self.MAX_EXPERIENCE_CHARS,
        )
        self.window.set_experience(experience)

        learned = self._truncate(
            str(learned_experience or ""),
            self.MAX_LEARNED_EXPERIENCE_CHARS,
        )
        self.window.set_learned_experience(learned)

        self.window.set_runtime(workspace)

        self.window.set_conversation(
            self._build_conversation(
                events=events,
                task=task,
            )
        )

    def _fit_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        system = [message for message in messages if message.get("role") == "system"][
            :1
        ]

        rest = [message for message in messages if message.get("role") != "system"]

        start = len(rest)

        for index in range(len(rest) - 1, -1, -1):

            candidate = system + rest[index:]

            if not self.tokenbudget.fits(candidate):
                break

            start = index

        while start < len(rest) and rest[start].get("role") == "tool":
            start += 1

        return system + rest[start:]

    def _minimal_messages(self) -> list[dict[str, Any]]:

        system = {
            "role": "system",
            "content": self.window.build_system_content(),
        }

        for message in reversed(self.window.conversation):
            if message.get("role") == "user":
                return [system, message]

        return [system]

    def build_context(
        self,
        events: list[ContextEvent],
        task: dict[str, Any] | None = None,
        agent_state: dict[str, Any] | None = None,
        progress: dict[str, Any] | None = None,
        working_set: dict[str, Any] | None = None,
        observation: dict[str, Any] | None = None,
        recent_actions: dict[str, Any] | None = None,
        workspace: str | None = None,
        learned_experience: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        `agent_state`, `progress`, `working_set`, `observation` and
        `recent_actions` are accepted for compatibility with
        ContextService/Loop but intentionally NOT rendered (see the class
        docstring for why). `learned_experience` is the hook point for a
        future self-evolving memory system: pass retrieved/learned text
        there and it renders in its own <learned_experience> section,
        separate from the hand-curated AgentInstruction/experience.md.
        """

        # Silence "unused parameter" concerns explicitly: these exist only
        # for call-site compatibility.
        del agent_state, progress, working_set, observation, recent_actions

        self._populate_window(
            events=events,
            task=task,
            workspace=workspace,
            learned_experience=learned_experience,
        )

        messages = self.window.get_prompt()

        if self.tokenbudget.fits(messages):
            return messages

        messages = self._fit_messages(messages)

        if self.tokenbudget.fits(messages):
            return messages

        return self._minimal_messages()
