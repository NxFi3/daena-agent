# src.context/compactor.py
from src.context.compactorprompt import BuildCompactorPrompt
from src.engine.LlmProviderManager import LlmProvider
from src.utils.logger import get_logger


class Compactor:

    def __init__(
        self,
        llm_provider: LlmProvider,
    ) -> None:
        self.llm_provider = llm_provider
        self.logger = get_logger("[COMPACTOR]")

    def compact(
        self,
        context: str,
        max_length: int,
    ) -> str:

        if not context or not context.strip():
            return ""

        try:
            prompt = BuildCompactorPrompt(
                context,
                max_length,
            )

            # Compaction is a summarization job, not an agent-thinking turn.
            # Reasoning models can otherwise spend the whole turn thinking
            # and return an empty response, which used to make compaction fail.
            result = self.llm_provider.generate(
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    }
                ],
                options={"think": False},
            )

            if result is None:
                self.logger.error("Compactor received no LLM result.")
                return ""

            compacted = str(result.response or "").strip()

            if not compacted:
                self.logger.error("Compactor returned empty response.")
                return ""

            return compacted

        except Exception as exc:
            self.logger.error(f"Compaction error: {exc}")
            return ""
