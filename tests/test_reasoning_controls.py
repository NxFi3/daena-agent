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
    thinking=types.ThinkingConfig.model_validate(state["requests"][0]["generationConfig"]["thinkingConfig"])
    assert thinking.thinking_budget==budget


def test_reasoning_auto_clears_native_overrides_and_custom_preserves_them():
    options={"thinking_config":{"thinking_level":"LOW"},"temperature":.4}
    assert set_reasoning(options,"custom")==options
    assert reasoning_value(options)=="custom"
    assert set_reasoning(options,"auto")=={"temperature":.4}


def test_gemini_tool_config_uses_native_wire_field_names(server,monkeypatch):
    if types is None:pytest.skip("google-genai not installed")
    monkeypatch.setenv("GEMINI_API_KEY","test-only")
    state,url=server
    state["events"]=[{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}]
    tool={"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}}
    GeminiProvider().generate(LLMInput("test",[{"role":"user","content":"inspect"}],tools=[tool],options={"base_url":url,"tool_choice":"required"}))
    request=state["requests"][0]
    assert request["toolConfig"]["functionCallingConfig"]["mode"]=="ANY"
    assert request["tools"][0]["functionDeclarations"][0]["name"]=="read_file"


def test_custom_gemini_thinking_level_overrides_common_effort(server,monkeypatch):
    if types is None:pytest.skip("google-genai not installed")
    monkeypatch.setenv("GEMINI_API_KEY","test-only")
    state,url=server
    state["events"]=[{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}]
    GeminiProvider().generate(LLMInput("test",[{"role":"user","content":"hello"}],options={"base_url":url,"think":"high","thinking_config":{"thinking_level":"LOW"}}))
    thinking=state["requests"][0]["generationConfig"]["thinkingConfig"]
    parsed=types.ThinkingConfig.model_validate(thinking)
    assert parsed.thinking_level==types.ThinkingLevel.LOW and parsed.thinking_budget is None
