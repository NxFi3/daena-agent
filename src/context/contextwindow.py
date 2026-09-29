from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

Message = dict[str, Any]


class ContextWindow:
    """
    Builds the final model-visible context.

    Structure of the system message:

        1. static instruction      (AgentInstruction/systeminstruction.md)
        2. hand-curated experience (AgentInstruction/experience.md)
        3. learned experience      (reserved for a future self-evolving
                                     memory system; empty until wired up)
        4. runtime info (os / cwd / workspace / date)

    Everything that changes turn to turn (agent state, progress, working
    set, tool results) lives in `conversation` as real chat messages, not
    in the system message. Stuffing it into the system prompt duplicates
    what the tool-call/tool-result messages already say and is what made
    the context blow up.

    ContextWindow holds no retrieval or file-reading logic; ContextBuilder
    populates it.
    """

    def __init__(self) -> None:

        self.os_name = platform.system()

        self.system_instruction: str = ""
        self.experience: str = ""
        self.learned_experience: str = ""
        self.plan: str = ""
        self.runtime: dict[str, Any] = self._base_runtime()

        self.conversation: list[Message] = []

    def _base_runtime(self) -> dict[str, Any]:

        return {
            "os": self.os_name,
            "cwd": str(Path.cwd()),
            "date": datetime.now().strftime("%Y-%m-%d (%A)"),
        }

    def set_system(
        self,
        instruction: str,
    ) -> None:
        self.system_instruction = str(instruction or "").strip()

    def set_experience(
        self,
        text: str,
    ) -> None:
        """
        Hand-curated, file-backed experience (AgentInstruction/experience.md).
        Edit that file directly; ContextBuilder re-reads it on every turn,
        so changes apply on the next model call with no restart needed.
        """
        self.experience = str(text or "").strip()

    def set_plan(
        self,
        text: str,
    ) -> None:
        """
        Sets the current file-backed execution plan (plan.md).

        ContextBuilder loads the plan and refreshes it on each turn,
        so changes made by the planning tool are reflected in the
        next model call without requiring a restart.
        """

        self.plan = str(text or "").strip()

    def set_learned_experience(
        self,
        text: str | None,
    ) -> None:
        """
        Reserved slot for a future self-evolving memory system: lessons the
        agent (or a consolidation process) derives on its own, kept separate
        from the hand-curated file above so the two never overwrite each
        other. Currently always empty unless something passes
        `learned_experience=` into ContextBuilder.build_context().
        """
        self.learned_experience = str(text or "").strip()

    def set_runtime(
        self,
        workspace: str | None = None,
    ) -> None:
        self.runtime = self._base_runtime()

        if workspace:
            self.runtime["workspace"] = str(Path(workspace).expanduser().resolve())

    def set_conversation(
        self,
        content: list[Message],
    ) -> None:
        self.conversation = list(content or [])

    @staticmethod
    def _serialize(
        content: Any,
    ) -> str:
        if isinstance(content, str):
            return content

        return json.dumps(
            content,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    @classmethod
    def _section(
        cls,
        name: str,
        content: Any,
    ) -> str:
        return f"<{name}>\n{cls._serialize(content)}\n</{name}>"

    def build_system_content(self) -> str:

        sections: list[str] = []

        if self.system_instruction:
            sections.append(self.system_instruction)
        if self.plan:
            sections.append(self._section("plan", self.plan))
        if self.experience:
            sections.append(self._section("experience", self.experience))

        if self.learned_experience:
            sections.append(
                self._section("learned_experience", self.learned_experience)
            )

        sections.append(self._section("runtime", self.runtime))

        return "\n\n".join(sections)

    def get_prompt(self) -> list[Message]:

        messages: list[Message] = []

        system_content = self.build_system_content()

        if system_content:
            messages.append(
                {
                    "role": "system",
                    "content": system_content,
                }
            )

        messages.extend(self.conversation)

        return messages
