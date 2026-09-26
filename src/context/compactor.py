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

            result = self.llm_provider.generate(
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    }
                ]
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
