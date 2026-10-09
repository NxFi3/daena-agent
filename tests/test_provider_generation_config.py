from pathlib import Path
import json

from harness.runner import configure_run
from src.engine.providers.builtin.gemini.geminiprovider import GeminiProvider
from src.engine.providers.builtin.openrouter.openrouter import OpenRouterProvider


ROOT = Path(__file__).resolve().parents[1]


def test_config_has_no_fixed_output_token_cap():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    generation = config["llm"]["provider_config"]["generation_config"]
    assert "num_predict" not in generation


def test_gemini_uses_model_default_output_limit_and_drops_ollama_options():
    provider = GeminiProvider()
    assert "max_output_tokens" not in provider.defaultConfig
    normalized = provider._normalize_generation_options({
        "temperature": 0.3,
        "num_predict": 1024,
        "num_thread": 16,
        "num_ctx": 120000,
    })
    assert normalized == {"temperature": 0.3}


def test_openrouter_drops_legacy_ollama_options():
    normalized = OpenRouterProvider._normalize_generation_options({
        "temperature": 0.3,
        "num_predict": 1024,
        "num_thread": 16,
        "num_ctx": 120000,
        "think": "low",
    })
    assert normalized == {"temperature": 0.3, "think": "low"}


def test_openrouter_benchmark_config_selects_coding_agent_model():
    original = {
        "llm": {
            "provider": "ollama",
            "provider_config": {
                "model_name": "gpt-oss:20b",
                "generation_config": {
                    "temperature": 0.3,
                    "num_thread": 16,
                    "think": "low",
                    "num_predict": 1024,
                },
            },
        }
    }
    result = configure_run(original, "openrouter", None)
    provider_config = result["llm"]["provider_config"]
    generation = provider_config["generation_config"]
    assert result["llm"]["provider"] == "openrouter"
    assert provider_config["model_name"] == "deepseek/deepseek-v4.1-flash"
    assert "num_predict" not in generation
    assert "num_thread" not in generation
    assert original["llm"]["provider_config"]["generation_config"]["num_predict"] == 1024
