# src.context/tokenbudget.py
from typing import Any

from src.engine.LlmProviderManager import LlmProvider
from src.models.LLMResult import LLMResult
from src.utils.logger import get_logger

logger = get_logger("[TOKENBUDGET]")


class TokenBudget:

    def __init__(
        self,
        config: dict[str, Any],
        llm_provider: LlmProvider,
    ) -> None:

        self.config = config.get("context") or {}

        self.llm_provider = llm_provider

        provider_defaults = getattr(
            getattr(self.llm_provider, "model", None),
            "defaultConfig",
            {},
        ) or {}

        llm_config = config.get("llm") or {}
        provider_config = llm_config.get("provider_config") or {}
        generation_config = provider_config.get("generation_config") or {}

        context_length = (
            generation_config.get("num_ctx")
            or provider_config.get("num_ctx")
            or provider_defaults.get("num_ctx")
            or 120000
        )

        try:
            self.context_length = max(128, int(context_length))
        except (TypeError, ValueError):
            self.context_length = 120000

        self.safe_margin = int(
            self.config.get(
                "safe_margin",
                -1,
            )
        )

        if self.safe_margin < 0:
            self.safe_margin = int(self.context_length * 0.15)

        self.budget = max(
            1,
            self.context_length - self.safe_margin,
        )

        self.compaction_target_tokens = min(
            int(
                self.config.get(
                    "compaction_target_tokens",
                    min(
                        16384,
                        self.budget,
                    ),
                )
            ),
            self.budget,
        )

        self.chars_per_token = float(
            self.config.get(
                "initial_chars_per_token",
                4.0,
            )
        )

        if self.chars_per_token <= 0:
            self.chars_per_token = 4.0

        self.calibration_alpha = float(
            self.config.get(
                "calibration_alpha",
                0.2,
            )
        )

        self.min_chars_per_token = float(
            self.config.get(
                "min_chars_per_token",
                1.5,
            )
        )

        self.max_chars_per_token = float(
            self.config.get(
                "max_chars_per_token",
                8.0,
            )
        )

        logger.info(
            f"Context length={self.context_length}, "
            f"budget={self.budget}, "
            f"safe_margin={self.safe_margin}, "
            f"compaction_target={self.compaction_target_tokens}, "
            f"chars_per_token={self.chars_per_token}"
        )

    def _message_character_count(
        self,
        message: dict[str, Any],
    ) -> int:

        total = 0

        content = message.get("content")

        if content:
            total += len(str(content))

        tool_calls = message.get("tool_calls")

        if tool_calls:
            total += len(str(tool_calls))

        for key in (
            "name",
            "tool_name",
            "tool_call_id",
        ):
            value = message.get(key)

            if value:
                total += len(str(value))

        return total

    def estimate_messages_tokens(
        self,
        messages: list[dict[str, Any]],
    ) -> int:

        if not messages:
            return 0

        total_chars = sum(
            self._message_character_count(message) for message in messages
        )

        if total_chars <= 0:
            return 0

        return max(
            1,
            int(total_chars / self.chars_per_token),
        )

    def fits(
        self,
        messages: list[dict[str, Any]],
    ) -> bool:

        return self.estimate_messages_tokens(messages) <= self.budget

    def remaining_tokens(
        self,
        messages: list[dict[str, Any]],
    ) -> int:

        return self.budget - self.estimate_messages_tokens(messages)

    def calibrate_from_response(
        self,
        messages: list[dict[str, Any]],
        model_response: LLMResult,
    ) -> None:

        if not messages:
            return

        if model_response is None:
            return

        raw = model_response.raw

        if raw is None:
            return

        actual_prompt_tokens = (
            getattr(
                raw,
                "prompt_eval_count",
                0,
            )
            or 0
        )

        actual_prompt_tokens = int(actual_prompt_tokens)

        if actual_prompt_tokens <= 0:
            return

        measured_chars = sum(
            self._message_character_count(message) for message in messages
        )

        if measured_chars <= 0:
            return

        observed_ratio = measured_chars / actual_prompt_tokens

        observed_ratio = max(
            self.min_chars_per_token,
            min(
                self.max_chars_per_token,
                observed_ratio,
            ),
        )

        alpha = max(
            0.01,
            min(
                1.0,
                self.calibration_alpha,
            ),
        )

        old_ratio = self.chars_per_token

        self.chars_per_token = (1.0 - alpha) * old_ratio + alpha * observed_ratio

    def used_tokens(
        self,
        model_response: LLMResult,
    ) -> int:

        if model_response is None:
            return 0

        raw = model_response.raw

        if raw is not None:
            prompt_tokens = (
                getattr(
                    raw,
                    "prompt_eval_count",
                    0,
                )
                or 0
            )

            if prompt_tokens:
                return int(prompt_tokens)

        return int(model_response.usage)

    def remaining_budget(
        self,
        model_response: LLMResult,
    ) -> int:

        return self.budget - self.used_tokens(model_response)

    def is_within_budget(
        self,
        model_response: LLMResult,
    ) -> bool:

        return self.remaining_budget(model_response) > 0
