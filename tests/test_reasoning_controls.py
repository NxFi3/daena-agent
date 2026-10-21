import pytest
from src.models.LLMInput import LLMInput
from src.engine.providers.builtin.gemini.geminiprovider import GeminiProvider, types
from src.studio.profiles import set_reasoning,reasoning_value,switch_profile
from tests.test_remote_providers import server


@pytest.mark.parametrize("mode,budget",[("off",0),("low",1024),("medium",4096),("high",8192)])
def test_gemini_reasoning_control_reaches_provider_wire(server,monkeypatch,mode,budget):
    if types is None:pytest.skip("google-genai not installed")
    monkeypatch.setenv("GEMINI_API_KEY","test-only")
    state,url=server
    state["events"]=[{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}]
    options=set_reasoning({"base_url":url},mode)
    GeminiProvider().generate(LLMInput("test",[{"role":"user","content":"hello"}],options=options))
    assert state["requests"][0]["generationConfig"]["thinkingConfig"].get("thinkingBudget")==budget, state["requests"][0]


def test_reasoning_auto_clears_native_overrides_and_custom_preserves_them():
    options={"thinking_config":{"thinking_level":"LOW"},"temperature":.4}
    assert set_reasoning(options,"custom")==options
    assert reasoning_value(options)=="custom"
    assert set_reasoning(options,"auto")=={"temperature":.4}
