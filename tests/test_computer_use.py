from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from src.context.contextbuilder import ContextBuilder
from src.models.ContextEvent import ContextEvent, ContextPriority, ContextRole, ContextType
from src.models.ToolResult import ToolResult
from src.tools.ToolManager import ToolManager
from src.tools.builtin.computer_use import act as act_module
from src.tools.builtin.computer_use import inspect as inspect_module
from src.tools.builtin.computer_use import observe as observe_module
from src.tools.builtin.computer_use import computer_use as computer_use_module
from src.tools.builtin.computer_use.act import ComputerActTool
from src.tools.builtin.computer_use.inspect import ComputerInspectTool
from src.tools.builtin.computer_use import driver as driver_module
from src.tools.builtin.computer_use.driver import session_label_from_id
from src.tools.builtin.computer_use.observe import ComputerObserveTool
from src.tools.builtin.computer_use.computer_use import ComputerUseTool


PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + (640).to_bytes(4, "big") + (480).to_bytes(4, "big") + b"test"


@pytest.fixture(autouse=True)
def _skip_live_cua_startup_for_unit_tests(monkeypatch):
    # These tests mock the driver actions; keep them hermetic and independent of
    # whether a real desktop daemon happens to be installed on the test runner.
    monkeypatch.setattr(computer_use_module, "ensure_driver_running", lambda _session=None: None)


def _event(role, event_type, content, step, metadata=None):
    return ContextEvent(
        role=role,
        type=event_type,
        content=content,
        step=step,
        metadata=metadata or {},
    )


def test_computer_tool_is_registered_but_discoverable_by_default(tmp_path):
    manager = ToolManager({
        "guard_level": "strict",
        "security": {"workspace_only": True, "force_approve": True},
    })
    manager.set_workspace(str(tmp_path))
    try:
        model_names = {item["function"]["name"] for item in manager.get_model_tools()}
        catalog_names = {item["function"]["name"] for item in manager.get_tools()}
        assert "computer_use" in catalog_names
        assert "computer_use" not in model_names
        outcome = manager._search_and_activate_tools(
            "control the desktop, click visible buttons, and inspect screenshots", top_k=5
        )
        assert "computer_use" in {item["name"] for item in outcome["matches"]}
        exposed_after_search = {item["function"]["name"] for item in manager.get_model_tools()}
        assert "computer_use" in exposed_after_search
        assert not ({"computer_observe", "computer_inspect", "computer_act"} & exposed_after_search)
    finally:
        manager.close()


def test_tool_model_exposure_can_be_overridden_in_config(tmp_path):
    manager = ToolManager({
        "guard_level": "strict",
        "tools": {"exposure": {"computer_use": "always"}},
        "security": {"workspace_only": True, "force_approve": True},
    })
    manager.set_workspace(str(tmp_path))
    try:
        names = {item["function"]["name"] for item in manager.get_model_tools()}
        assert "computer_use" in names
    finally:
        manager.close()


def test_computer_use_checks_driver_readiness_before_first_driver_call(tmp_path, monkeypatch):
    tool = ComputerUseTool()
    tool.set_workspace(tmp_path)
    order = []

    def ensure_ready(session=None):
        order.append(("ensure_ready", session))

    def fake_call(name, arguments=None, **kwargs):
        order.append(f"driver:{name}")
        assert name == "list_windows"
        return '{"windows": []}'

    monkeypatch.setattr(computer_use_module, "ensure_driver_running", ensure_ready)
    monkeypatch.setattr(computer_use_module, "call_driver", fake_call)

    result = tool.execute(action="list_windows")

    assert result.success is True
    assert order == [("ensure_ready", tool._observe._session_label()), "driver:list_windows"]


def test_observe_saves_screenshot_and_returns_stm_reference(tmp_path, monkeypatch):
    def fake_call(name, arguments=None, *, timeout=30.0):
        if name == "get_desktop_state":
            Path(arguments["screenshot_out_file"]).write_bytes(PNG_HEADER)
            return '{"ok":true,"capture_id":"capture_test_123"}'
        if name == "list_windows":
            return json.dumps({"windows": [{"app_name": "Firefox", "title": "Test window", "pid": 4321, "window_id": 1234, "is_on_screen": True, "x": 0, "y": 0, "width": 640, "height": 480}]})
        if name == "get_cursor_position":
            return json.dumps({"source": "synthetic", "x": 320, "y": 240})
        raise AssertionError(name)

    monkeypatch.setattr(observe_module, "call_driver", fake_call)
    tool = ComputerObserveTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "unit-test-session")

    result = tool.execute()

    assert result.success is True
    image_ref = result.metadata["image_ref"]
    assert image_ref.endswith(".png")
    assert Path(image_ref).is_file()
    assert ".daena" in Path(image_ref).parts
    assert "screenshots" in Path(image_ref).parts
    assert result.metadata["kind"] == "screen_observation"
    assert result.metadata["width"] == 640
    assert result.metadata["height"] == 480
    assert result.content["cursor_position"] == {"x": 320, "y": 240, "source": "synthetic"}
    assert "source=synthetic" in result.content["cursor_hint"]
    assert len(result.content["screenshot_hash"]) == 64
    compact_windows = json.loads(result.content["windows"])
    assert compact_windows[0]["app"] == "Firefox"
    assert compact_windows[0]["title"] == "Test window"
    assert compact_windows[0]["window_id"] == 1234
    from src.tools.builtin.computer_use.driver import pointer_state_path
    state = json.loads(pointer_state_path(tmp_path, tool._session_label()).read_text(encoding="utf-8"))
    assert state["latest_capture_id"] == "capture_test_123"


def test_computer_inspect_returns_actionable_tokens_and_persists_target(tmp_path, monkeypatch):
    payload = {
        "app_name": "Firefox", "window_title": "Aparat — Mozilla Firefox",
        "window_id": 1234, "element_count": 2, "elements_complete": True,
        "elements": [
            {"element_token": "s12345678:4", "role": "text field", "label": "Search videos", "value": "", "enabled": True, "actions": ["setValue", "click"], "frame": {"x": 300, "y": 100, "w": 400, "h": 32}},
            {"element_token": "s12345678:5", "role": "push button", "label": "Search", "enabled": True, "actions": ["press"], "frame": {"x": 710, "y": 100, "w": 40, "h": 32}},
        ],
    }
    calls = []
    def fake_inspect_call(name, arguments=None, **kwargs):
        calls.append((name, arguments))
        if name == "list_windows":
            return json.dumps({"windows": [{"app_name": "Firefox", "title": "Aparat — Mozilla Firefox", "pid": 4321, "window_id": 1234, "is_on_screen": True, "width": 1280, "height": 720}]})
        if name == "get_window_state":
            return json.dumps(payload)
        raise AssertionError(name)
    monkeypatch.setattr(inspect_module, "call_driver", fake_inspect_call)
    tool = ComputerInspectTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "inspect-unit-session")

    result = tool.execute(pid=4321, window_id=1234, query="search")

    assert result.success is True
    assert len(result.content["elements"]) == 2
    assert result.content["elements"][0]["element_token"] == "s12345678:4"
    tree_call = next((args for name, args in calls if name == "get_window_state"), None)
    assert tree_call is not None
    assert tree_call["include_screenshot"] is False
    from src.tools.builtin.computer_use.driver import pointer_state_path
    state = json.loads(pointer_state_path(tmp_path, tool._session_label()).read_text(encoding="utf-8"))
    assert state["latest_accessibility_pid"] == 4321
    assert state["latest_accessibility_elements"]["s12345678:4"]["label"] == "Search videos"


def test_inspect_recovers_from_scientific_window_id_pointing_to_file_dialog(tmp_path, monkeypatch):
    payload = {
        "app_name": "Firefox", "window_title": "Aparat — Mozilla Firefox",
        "pid": 7883, "window_id": 516620288, "element_count": 1,
        "elements": [{
            "element_index": 3, "parent_index": 2, "element_token": "sabcdef01:3",
            "role": "landmark", "label": "جست\u200cو\u200cجو در آپارات",
            "actions": ["activate"], "enabled": True, "in_web_content": True,
            "frame": {"x": 1135, "y": 94, "w": 414, "h": 33},
        }],
    }
    calls = []
    def fake_call(name, arguments=None, **kwargs):
        calls.append((name, arguments))
        if name == "list_windows":
            return json.dumps({"windows": [
                {"app_name": "org.gnome.Nautilus", "title": "Open Folder", "pid": 6521, "window_id": 427360257, "is_on_screen": True, "width": 890, "height": 550},
                {"app_name": "Firefox", "title": "Aparat — Mozilla Firefox", "pid": 7883, "window_id": 516620288, "is_on_screen": True, "width": 2560, "height": 1409},
            ]})
        if name == "get_window_state":
            return json.dumps(payload)
        raise AssertionError(name)
    monkeypatch.setattr(inspect_module, "call_driver", fake_call)
    tool = ComputerInspectTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "wrong-window-id-test")

    result = tool.execute(pid="6521", window_id="4.27360257e+08", query="جست\u200cو\u200cجو در آپارات")

    assert result.success is True
    assert result.content["best_match"]["element_token"] == "sabcdef01:3"
    assert result.content["window"]["pid"] == 7883
    tree_call = next(args for name, args in calls if name == "get_window_state")
    assert tree_call["pid"] == 7883
    assert tree_call["window_id"] == 516620288
    assert result.content["match_count"] == 1


def test_inspect_ranks_target_window_search_landmark_ahead_of_browser_chrome():
    payload = {
        "window_title": "Aparat — Mozilla Firefox",
        "elements": [
            {"element_index": 1, "role": "frame", "label": "Aparat — Mozilla Firefox", "parent_index": None, "element_token": "saaaaaaaa:1"},
            {"element_index": 2, "role": "combo box", "label": "Search with Google or enter address", "parent_index": 1, "element_token": "saaaaaaaa:2", "actions": ["activate"], "enabled": True, "in_web_content": False, "frame": {"x": 300, "y": 47, "w": 1000, "h": 29}},
            {"element_index": 3, "role": "landmark", "label": "جست\u200cو\u200cجو در آپارات", "parent_index": 1, "element_token": "saaaaaaaa:3", "actions": ["activate"], "enabled": True, "in_web_content": True, "frame": {"x": 1135, "y": 94, "w": 414, "h": 33}},
            {"element_index": 10, "role": "frame", "label": "GitHub — Mozilla Firefox", "parent_index": None, "element_token": "saaaaaaaa:10"},
            {"element_index": 11, "role": "text field", "label": "Search code", "parent_index": 10, "element_token": "saaaaaaaa:11", "actions": ["activate"], "enabled": True, "in_web_content": True, "frame": {"x": 100, "y": 100, "w": 400, "h": 30}},
        ],
    }

    ranked = ComputerInspectTool._rank_elements(payload, "")

    assert ranked[0]["element_token"] == "saaaaaaaa:3"
    assert all(item["element_token"] != "saaaaaaaa:11" for item in ranked)
    assert ranked[0]["frame"] == {"x": 1135, "y": 94, "w": 414, "h": 33}


def test_inspect_resolves_browser_address_bar_synonyms():
    payload = {
        "window_title": "Mozilla Firefox",
        "elements": [
            {"element_index": 1, "role": "frame", "label": "Mozilla Firefox", "parent_index": None, "element_token": "sbbbbbbbb:1"},
            {"element_index": 2, "role": "combo box", "label": "Search with Google or enter address", "parent_index": 1, "element_token": "sbbbbbbbb:2", "actions": ["activate"], "enabled": True, "in_web_content": False, "frame": {"x": 300, "y": 47, "w": 1000, "h": 29}},
            {"element_index": 3, "role": "search box", "label": "Search videos", "parent_index": 1, "element_token": "sbbbbbbbb:3", "actions": ["setValue"], "enabled": True, "in_web_content": True, "frame": {"x": 400, "y": 100, "w": 400, "h": 32}},
        ],
    }
    ranked = ComputerInspectTool._rank_elements(payload, "address bar or URL input field")
    assert len(ranked) == 1
    assert ranked[0]["element_token"] == "sbbbbbbbb:2"


def test_inspect_does_not_treat_chat_transcript_as_a_live_search_control():
    query = "جست\u200cو\u200cجو در آپارات"
    payload = {
        "window_title": "ChatGPT — Mozilla Firefox",
        "elements": [
            {"element_index": 1, "role": "frame", "label": "ChatGPT — Mozilla Firefox", "parent_index": None, "element_token": "sabcdefff:1"},
            {"element_index": 2, "role": "section", "label": "Earlier instructions: call computer_inspect query=" + query + " then click the search field. " * 20, "parent_index": 1, "element_token": "sabcdefff:2", "actions": [], "enabled": True, "in_web_content": True, "frame": {"x": 1200, "y": 300, "w": 500, "h": 900}},
        ],
    }
    assert ComputerInspectTool._rank_elements(payload, query) == []


def test_observe_returns_recoverable_error_when_driver_is_unavailable(tmp_path, monkeypatch):
    def fail_call(name, arguments=None, *, timeout=30.0):
        raise observe_module.CuaDriverError("Cua Driver daemon is not running.")

    monkeypatch.setattr(observe_module, "call_driver", fail_call)
    tool = ComputerObserveTool()
    tool.set_workspace(tmp_path)
    result = tool.execute()

    assert result.success is False
    assert result.content["error"]["type"] == "computer_use_unavailable"
    assert "do not keep retrying" in result.metadata["recovery_hint"].lower()


def test_computer_act_only_dispatches_allowlisted_single_action(tmp_path, monkeypatch):
    calls = []

    def fake_call(name, arguments=None, *, timeout=30.0):
        calls.append((name, arguments, timeout))
        return '{"ok":true}'

    monkeypatch.setattr(act_module, "call_driver", fake_call)
    tool = ComputerActTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "unit-test-session")
    from src.tools.builtin.computer_use.driver import pointer_state_path
    state_path = pointer_state_path(tmp_path, tool._session_label())
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"latest_capture_id": "capture_test_click"}), encoding="utf-8")
    result = tool.execute(action="click", x=100, y=200)

    assert result.success is True
    assert calls[0][0] == "click"
    assert calls[0][1]["x"] == 100
    assert calls[0][1]["y"] == 200
    assert calls[0][1]["target"] == {"kind": "desktop", "display_id": "primary"}
    assert calls[0][1]["coordinate_frame"] == "desktop"
    assert calls[0][1]["capture_id"] == "capture_test_click"
    assert "scope" not in calls[0][1]
    # Background is the unified Computer Use default; foreground must be explicit.
    assert calls[0][1]["delivery_mode"] == "background"
    assert calls[0][1]["session"] == session_label_from_id("unit-test-session")


def test_computer_act_prefers_current_accessibility_element_token(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(act_module, "call_driver", lambda name, arguments=None, **kwargs: calls.append((name, arguments)) or '{"route":"accessibility","effect":"clicked"}')
    tool = ComputerActTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "element-click-test")
    from src.tools.builtin.computer_use.driver import pointer_state_path
    state_path = pointer_state_path(tmp_path, tool._session_label())
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({
        "latest_accessibility_pid": 4321,
        "latest_accessibility_elements": {
            "s12345678:4": {"role": "text field", "label": "Search videos", "frame": {"x": 300, "y": 100, "w": 400, "h": 32}}
        },
        "recent_pixel_clicks": [{"x": 470, "y": 95, "action": "click"}],
        "pixel_clicks_since_progress": 3,
    }), encoding="utf-8")

    result = tool.execute(action="click", element_token="s12345678:4")

    assert result.success is True
    assert calls[0][0] == "click"
    assert calls[0][1]["pid"] == 4321
    assert calls[0][1]["element_token"] == "s12345678:4"
    assert "x" not in calls[0][1] and "y" not in calls[0][1]
    assert "target" not in calls[0][1] and "coordinate_frame" not in calls[0][1]
    updated = json.loads(state_path.read_text(encoding="utf-8"))
    assert updated["recent_pixel_clicks"] == []  # stale persistent guards expire between tasks
    assert updated["pixel_clicks_since_progress"] == 1
    assert updated["recent_element_actions"][0]["label"] == "search videos"
    assert updated["x"] == 500 and updated["y"] == 116


def test_accessibility_click_allows_same_control_with_a_fresh_token(tmp_path, monkeypatch):
    from src.tools.builtin.computer_use.driver import pointer_state_path

    tool = ComputerActTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "semantic-repeat-test")
    state_path = pointer_state_path(tmp_path, tool._session_label())
    state_path.parent.mkdir(parents=True)
    import time
    state_path.write_text(json.dumps({
        "latest_accessibility_pid": 4321,
        "latest_accessibility_elements": {
            "s12345679:9": {"role": "push button", "label": "Search", "frame": {"x": 100, "y": 100, "w": 40, "h": 30}}
        },
        "last_click_timestamp": time.time(),
        "recent_element_actions": [{"token": "s12345678:8", "action": "click", "role": "push button", "label": "search", "timestamp": time.time()}],
    }), encoding="utf-8")
    calls = []
    monkeypatch.setattr(act_module, "call_driver", lambda name, arguments=None, **kwargs: calls.append((name, arguments)) or '{"route":"accessibility","effect":"clicked"}')

    result = tool.execute(action="click", element_token="s12345679:9")
    assert result.success is True
    assert calls[0][0] == "click"
    assert calls[0][1]["element_token"] == "s12345679:9"


def test_accessibility_click_blocks_reusing_the_exact_same_token_immediately(tmp_path, monkeypatch):
    from src.tools.builtin.computer_use.driver import pointer_state_path
    import time

    tool = ComputerActTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "same-token-repeat-test")
    state_path = pointer_state_path(tmp_path, tool._session_label())
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({
        "latest_accessibility_pid": 4321,
        "latest_accessibility_elements": {
            "s12345679:9": {"role": "push button", "label": "Search", "frame": {"x": 100, "y": 100, "w": 40, "h": 30}}
        },
        "last_click_timestamp": time.time(),
        "recent_element_actions": [{"token": "s12345679:9", "action": "click", "role": "push button", "label": "search", "timestamp": time.time()}],
    }), encoding="utf-8")
    monkeypatch.setattr(act_module, "call_driver", lambda *args, **kwargs: pytest.fail("repeated current token must be blocked"))
    result = tool.execute(action="click", element_token="s12345679:9")
    assert result.success is False
    assert "same accessibility token" in result.content["error"]["message"]


def test_repeated_click_is_blocked_for_the_same_recent_pixel_target(tmp_path, monkeypatch):
    from src.tools.builtin.computer_use.driver import pointer_state_path
    import time

    tool = ComputerActTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "repeat-click-test")
    state_path = pointer_state_path(tmp_path, tool._session_label())
    state_path.parent.mkdir(parents=True)
    now = time.time()
    state_path.write_text(json.dumps({
        "x": 100, "y": 200, "action": "click",
        "latest_capture_id": "capture_repeat_test",
        "screenshot_hash": "same-screen", "latest_screenshot_hash": "different-dynamic-screen",
        "last_click_timestamp": now,
        "pixel_clicks_since_progress": 1,
        "recent_pixel_clicks": [{"x": 100, "y": 200, "action": "click", "timestamp": now}],
    }), encoding="utf-8")
    monkeypatch.setattr(
        act_module, "call_driver",
        lambda *args, **kwargs: pytest.fail("unchanged repeated click must be blocked"),
    )

    result = tool.execute(action="click", x=100, y=200)
    assert result.success is False
    assert "Blocked repeated pixel click" in result.content["error"]["message"]


def test_call_driver_rejects_structured_refusal_even_when_cli_exits_zero(monkeypatch):
    import subprocess

    monkeypatch.setattr(driver_module, "driver_executable", lambda: "/usr/bin/cua-driver")
    monkeypatch.setattr(
        driver_module.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout='{"code":"invalid_action_target"}\n', stderr=""
        ),
    )

    with pytest.raises(driver_module.CuaDriverError, match="invalid_action_target"):
        driver_module.call_driver(
            "click",
            {"target": {"kind": "desktop", "display_id": "primary"}, "x": 10, "y": 10},
        )


def test_multimodal_model_mismatch_is_non_retryable():
    from src.agent.agentloop import Loop

    assert Loop._is_non_retryable_generation_error(
        "Ollama streaming failed: Multimodal data provided, but model does not support multimodal requests."
    )
    assert not Loop._is_non_retryable_generation_error("Ollama streaming failed: timeout")


def test_computer_act_rejects_missing_coordinates_without_driver_call(monkeypatch):
    monkeypatch.setattr(
        act_module,
        "call_driver",
        lambda *args, **kwargs: pytest.fail("driver should not be called for invalid input"),
    )
    result = ComputerActTool().execute(action="click", x=22)
    assert result.success is False
    assert "requires numeric x and y" in result.content["error"]["message"]


def test_screen_observation_becomes_real_image_ref_message_in_context(tmp_path):
    image_path = tmp_path / ".daena" / "screenshots" / "session" / "screen.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(PNG_HEADER)

    task = _event(ContextRole.USER, ContextType.MESSAGE, "inspect the desktop", 1)
    assistant = _event(ContextRole.ASSISTANT, ContextType.MESSAGE, "", 2, metadata={
        "llm_message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "observe-1",
                "type": "function",
                "function": {"name": "computer_use", "arguments": "{\"action\":\"observe\"}"},
            }],
        }
    })
    result_payload = {
        "name": "computer_use",
        "tool_call_id": "observe-1",
        "success": True,
        "summary": "Captured desktop",
        "evidence": {"image_ref": str(image_path)},
        "effects": [],
        "content": {"action": "observe", "image_ref": str(image_path), "image_mime_type": "image/png"},
        "metadata": {"kind": "screen_observation", "action": "observe", "image_ref": str(image_path), "image_mime_type": "image/png"},
    }
    tool_event = _event(
        ContextRole.TOOL, ContextType.TOOL_RESULT,
        json.dumps(result_payload), 3,
    )

    builder = object.__new__(ContextBuilder)
    builder.preserve_thinking = False
    messages = builder._build_conversation(
        [task, assistant, tool_event],
        task={"id": str(task.id), "content": task.content, "step": task.step},
    )

    image_messages = [message for message in messages if message.get("image_ref")]
    assert len(image_messages) == 1
    assert image_messages[0]["role"] == "user"
    assert image_messages[0]["image_ref"] == str(image_path)
    assert image_messages[0]["image_mime_type"] == "image/png"
    assert messages.index(image_messages[0]) > next(
        i for i, message in enumerate(messages)
        if message.get("role") == "tool" and message.get("tool_name") == "computer_use"
    )


def _make_saved_png(tmp_path: Path) -> Path:
    path = tmp_path / ".daena" / "screenshots" / "unit" / "screen.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG_HEADER)
    return path


def test_ollama_adapter_attaches_bytes_and_strips_private_image_ref(tmp_path):
    pytest.importorskip("ollama")
    from src.engine.providers.builtin.ollama.ollama import OllamaProvider

    path = _make_saved_png(tmp_path)
    messages = OllamaProvider._prepare_messages([{
        "role": "user", "content": "Describe the current screen.",
        "image_ref": str(path), "image_mime_type": "image/png",
    }])
    assert messages[0]["images"] == [PNG_HEADER]
    assert "image_ref" not in messages[0]
    assert "image_mime_type" not in messages[0]


def test_openrouter_adapter_emits_openai_compatible_image_data_url(tmp_path):
    pytest.importorskip("openrouter")
    from src.engine.providers.builtin.openrouter.openrouter import OpenRouterProvider

    path = _make_saved_png(tmp_path)
    messages = OpenRouterProvider._prepare_messages([{
        "role": "user", "content": "Describe the current screen.",
        "image_ref": str(path), "image_mime_type": "image/png",
    }])
    blocks = messages[0]["content"]
    assert blocks[0] == {"type": "text", "text": "Describe the current screen."}
    assert blocks[1]["type"] == "image_url"
    assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "image_ref" not in messages[0]


def test_gemini_adapter_emits_inline_image_part(tmp_path):
    from src.engine.providers.builtin.gemini import geminiprovider

    if geminiprovider.types is None:
        pytest.skip("google-genai optional dependency is not installed")
    path = _make_saved_png(tmp_path)
    messages, _system = geminiprovider.GeminiProvider._convert_messages([{
        "role": "user", "content": "Describe the current screen.",
        "image_ref": str(path), "image_mime_type": "image/png",
    }])
    parts = messages[0].parts
    assert any(getattr(part, "text", None) == "Describe the current screen." for part in parts)
    inline_parts = [part for part in parts if getattr(part, "inline_data", None) is not None]
    assert len(inline_parts) == 1
    assert bytes(inline_parts[0].inline_data.data) == PNG_HEADER


def test_image_observation_does_not_replace_original_user_task_anchor():
    messages = [
        {"role": "user", "content": "Open the browser and search for a page."},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "Current desktop screenshot.", "image_ref": "/tmp/.daena/screenshots/screen.png"},
    ]
    assert ContextBuilder._last_index(messages, "user") == 0


def test_only_latest_screenshot_stays_attached_but_both_observations_stay_in_history(tmp_path):
    path1 = tmp_path / ".daena" / "screenshots" / "session" / "screen-1.png"
    path2 = tmp_path / ".daena" / "screenshots" / "session" / "screen-2.png"
    path1.parent.mkdir(parents=True)
    path1.write_bytes(PNG_HEADER)
    path2.write_bytes(PNG_HEADER)
    task = _event(ContextRole.USER, ContextType.MESSAGE, "Inspect the desktop and click the browser.", 1)

    def observation_pair(suffix: str, image_path: Path, assistant_step: int, result_step: int):
        assistant = _event(ContextRole.ASSISTANT, ContextType.MESSAGE, "", assistant_step, metadata={
            "llm_message": {
                "role": "assistant", "content": "", "tool_calls": [{
                    "id": f"observe-{suffix}", "type": "function",
                    "function": {"name": "computer_use", "arguments": "{\"action\":\"capture\"}"},
                }],
            }
        })
        payload = {
            "name": "computer_use", "tool_call_id": f"observe-{suffix}",
            "success": True, "summary": "Captured desktop", "evidence": {}, "effects": [],
            "content": {"action": "capture", "image_ref": str(image_path), "image_mime_type": "image/png"},
            "metadata": {"kind": "computer_use_capture", "action": "capture", "image_ref": str(image_path), "image_mime_type": "image/png"},
        }
        result = _event(ContextRole.TOOL, ContextType.TOOL_RESULT, json.dumps(payload), result_step)
        return assistant, result

    assistant1, result1 = observation_pair("1", path1, 2, 3)
    assistant2, result2 = observation_pair("2", path2, 4, 5)
    builder = object.__new__(ContextBuilder)
    builder.preserve_thinking = False
    messages = builder._build_conversation(
        [task, assistant1, result1, assistant2, result2],
        task={"id": str(task.id), "content": task.content, "step": task.step},
    )

    attached = [message for message in messages if message.get("image_ref")]
    assert len(attached) == 1
    assert attached[0]["image_ref"] == str(path2)
    assert any("Earlier screenshot is stored in STM" in str(message.get("content", "")) for message in messages)
    assert sum(1 for message in messages if message.get("tool_name") == "computer_use") == 2


def test_token_budget_reserves_visual_tokens_for_image_reference():
    from src.context.tokenbudget import TokenBudget

    budget = object.__new__(TokenBudget)
    budget.chars_per_token = 4.0
    text_only = {"role": "user", "content": "Current screenshot"}
    with_image = {**text_only, "image_ref": "/workspace/.daena/screenshots/screen.png"}
    assert budget._message_character_count(with_image) - budget._message_character_count(text_only) == 4 * 1536


def test_image_reference_rejects_paths_outside_daena_screenshot_store(tmp_path):
    from src.utils.image_reference import read_image_reference

    untrusted = tmp_path / "image.png"
    untrusted.write_bytes(PNG_HEADER)
    with pytest.raises(ValueError, match="outside Daena"):
        read_image_reference(str(untrusted), "image/png")



def test_screenshot_marks_daenas_last_click_when_cursor_is_not_in_capture(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    from src.tools.builtin.computer_use.driver import pointer_state_path

    tool = ComputerObserveTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "pointer-overlay-test")
    image_path = tmp_path / "screen.png"
    image_module.new("RGB", (100, 100), (0, 0, 0)).save(image_path)
    state_path = pointer_state_path(tmp_path, tool._session_label())
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({"x": 50, "y": 50, "action": "click"}), encoding="utf-8")

    assert tool._overlay_last_agent_pointer(image_path) is True
    with image_module.open(image_path) as marked:
        assert marked.size == (100, 100)
        assert marked.getpixel((62, 50)) == (235, 45, 65)


def test_wayland_desktop_environment_can_be_reconstructed(tmp_path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / "bus").touch()
    (runtime_dir / "wayland-0").touch()
    (runtime_dir / ".mutter-Xwaylandauth.test").touch()
    x11_dir = tmp_path / "X11-unix"
    x11_dir.mkdir()
    (x11_dir / "X0").touch()

    env = driver_module._compose_desktop_environment(
        {},
        [{"XDG_SESSION_TYPE": "wayland", "XDG_CURRENT_DESKTOP": "ubuntu:GNOME"}],
        runtime_dir=runtime_dir,
        home=tmp_path,
        x11_socket_dir=x11_dir,
    )

    assert env["WAYLAND_DISPLAY"] == "wayland-0"
    assert env["DISPLAY"] == ":0"
    assert env["XDG_RUNTIME_DIR"] == str(runtime_dir)
    assert env["DBUS_SESSION_BUS_ADDRESS"] == f"unix:path={runtime_dir / 'bus'}"
    assert env["XAUTHORITY"] == str(runtime_dir / ".mutter-Xwaylandauth.test")
    assert env["CUA_DRIVER_RS_ENABLE_WAYLAND"] == "1"


def test_missing_daemon_is_started_with_native_wayland_environment(tmp_path, monkeypatch):
    statuses = iter([(False, "daemon is not running"), (True, "daemon is running")])
    launches = []

    monkeypatch.setattr(driver_module, "_daemon_status", lambda _executable: next(statuses))
    monkeypatch.setattr(driver_module, "_desktop_session_environment", lambda: {
        "WAYLAND_DISPLAY": "wayland-0",
        "XDG_RUNTIME_DIR": str(tmp_path / "runtime"),
        "XDG_SESSION_TYPE": "wayland",
        "CUA_DRIVER_RS_ENABLE_WAYLAND": "1",
    })
    monkeypatch.setattr(driver_module.subprocess, "Popen", lambda command, **kwargs: launches.append((command, kwargs)))
    monkeypatch.setattr(driver_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    driver_module._ensure_daemon_running("/usr/bin/cua-driver")

    assert len(launches) == 1
    command, kwargs = launches[0]
    assert command == [
        "/usr/bin/cua-driver", "serve", "--socket",
        str(tmp_path / ".cache" / "cua-driver" / "cua-driver.sock"),
    ]
    assert kwargs["env"]["CUA_DRIVER_RS_ENABLE_WAYLAND"] == "1"
    assert kwargs["start_new_session"] is True



def test_ensure_driver_running_recreates_named_session(monkeypatch):
    calls = []
    monkeypatch.setattr(driver_module, "driver_executable", lambda: "/usr/bin/cua-driver")
    monkeypatch.setattr(driver_module, "_ensure_daemon_running", lambda executable: calls.append(("daemon", executable)))
    monkeypatch.setattr(
        driver_module, "call_driver",
        lambda name, arguments=None, **kwargs: calls.append((name, dict(arguments or {}))) or '{"ok":true}',
    )

    driver_module.ensure_driver_running("daena-test-session")

    assert calls == [
        ("daemon", "/usr/bin/cua-driver"),
        ("start_session", {"session": "daena-test-session"}),
    ]


def test_observe_explains_rootless_xwayland_capture_failure(tmp_path, monkeypatch):
    def fail_capture(name, arguments=None, *, timeout=30.0):
        assert name == "get_desktop_state"
        raise observe_module.CuaDriverError(
            'Cua Driver action failed: X11 error X11Error { request_name: Some("GetImage"), error_kind: Match, BadMatch rootless XWayland }'
        )

    monkeypatch.setattr(observe_module, "call_driver", fail_capture)
    tool = ComputerObserveTool()
    tool.set_workspace(tmp_path)
    result = tool.execute(include_windows=False)

    assert result.success is False
    assert result.content["error"]["type"] == "desktop_capture_backend_unavailable"
    assert "CUA_DRIVER_RS_ENABLE_WAYLAND=1" in result.metadata["recovery_hint"]


def test_capture_retries_unfiltered_controls_when_query_is_missing_or_unreadable(tmp_path):
    tool = ComputerUseTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "query-fallback-test")
    tool._observe.execute = lambda **kwargs: ToolResult(
        success=True,
        name="computer_observe",
        content={"image_ref": "/tmp/screen.png", "image_mime_type": "image/png",
                 "width": 1280, "height": 720, "windows": "[]"},
        metadata={"image_ref": "/tmp/screen.png", "image_mime_type": "image/png",
                  "width": 1280, "height": 720},
        summary="Captured desktop.",
    )
    calls = []

    def fake_inspect(**kwargs):
        calls.append(kwargs)
        if kwargs.get("query"):
            return ToolResult(
                success=False,
                name="computer_inspect",
                content={"error": {"message": "query-specific read failed"}},
                summary="Query-specific accessibility inspection failed.",
            )
        return ToolResult(
            success=True,
            name="computer_inspect",
            content={
                "window": {"app": "Firefox", "title": "Aparat — Mozilla Firefox",
                           "pid": 4321, "window_id": 1234},
                "elements": [{
                    "element_token": "sphone:1", "role": "text field",
                    "label": "شماره تلفن همراه", "actions": ["setValue", "click"],
                    "enabled": True, "in_web_content": True,
                    "frame": {"x": 900, "y": 650, "w": 300, "h": 40},
                }],
            },
            summary="Found one accessible field.",
        )

    tool._inspect.execute = fake_inspect
    tool._ensure_window_front = lambda target: True
    result = tool._capture(mode="som", app="Firefox", query="phone number input")

    assert result.success is True
    assert [c.get("query", "") for c in calls] == ["", "phone number input", ""]
    assert result.content["query_fallback"] is True
    assert result.content["query_requested"] == "phone number input"
    assert result.content["element_count"] == 1
    assert "شماره تلفن همراه" in result.content["elements_text"]
    assert "unfiltered accessible controls" in result.content["selection_warning"]


def test_scroll_driver_arguments_include_screenshot_scoped_anchor():
    args = ComputerActTool._driver_args(
        "scroll", {"direction": "down", "amount": 4, "x": 120, "y": 240}, "test-session"
    )
    assert args["x"] == 120 and args["y"] == 240
    assert args["direction"] == "down" and args["amount"] == 4
    assert args["coordinate_frame"] == "desktop"


def test_scroll_anchor_uses_visible_accessible_content_region():
    state = {"latest_accessibility_elements": {
        "web": {"role": "document web", "in_web_content": True,
                "frame": {"x": 80, "y": 120, "w": 900, "h": 600}},
        "button": {"role": "push button", "in_web_content": True,
                   "frame": {"x": 100, "y": 140, "w": 100, "h": 32}},
    }}
    assert ComputerUseTool._scroll_anchor_from_state(state) == [530.0, 420.0]


def test_scroll_anchor_falls_back_to_small_visible_web_control():
    state = {"latest_accessibility_elements": {
        "link": {"role": "link", "label": "Learn more", "in_web_content": True,
                 "frame": {"x": 1264, "y": 968, "w": 84, "h": 19}},
    }}
    assert ComputerUseTool._scroll_anchor_from_state(state) == [1306.0, 977.5]


def test_navigate_opens_url_through_browser_cli_not_global_keyboard(tmp_path, monkeypatch):
    tool = ComputerUseTool()
    tool.set_workspace(tmp_path)
    tool._navigation_browser = lambda requested_app="": ("firefox", "/usr/bin/firefox")
    captured = {}

    def fake_capture(**kwargs):
        captured.update(kwargs)
        return ToolResult(
            success=True,
            name="computer_use",
            content={"window": {"title": "Aparat — Mozilla Firefox"},
                     "elements": [], "element_count": 0, "mode": "som"},
            metadata={},
            summary="Captured Aparat.",
        )

    tool._capture = fake_capture
    window_rows = [
        {"app_name": "Aparat — Mozilla Firefox", "title": "Aparat — Mozilla Firefox",
         "pid": 4321, "window_id": 1234, "is_on_screen": True,
         "width": 1280, "height": 720, "z_index": 3}
    ]
    window_calls = {"count": 0}

    def windows_during_navigation():
        window_calls["count"] += 1
        # The browser window is not immediately visible after Popen; the navigation
        # code should poll rather than fail after its first post-launch snapshot.
        return [] if window_calls["count"] <= 2 else window_rows

    tool._inspect._windows = windows_during_navigation
    tool._ensure_window_front = lambda target: True
    monkeypatch.setattr(computer_use_module, "_desktop_session_environment",
                        lambda: {"DISPLAY": ":0", "XDG_RUNTIME_DIR": "/run/user/1000"})
    from unittest.mock import patch
    with patch.object(computer_use_module.subprocess, "Popen") as popen, \
         patch.object(computer_use_module.time, "sleep", lambda *_: None):
        result = tool._navigate("https://aparat.com", 0, "Firefox")

    assert result.success is True
    assert window_calls["count"] >= 3
    assert popen.call_args.args[0] == ["/usr/bin/firefox", "--new-window", "https://aparat.com"]
    assert captured["pid"] == 4321
    assert captured["window_id"] == 1234
    assert result.content["browser"] == "firefox"
    assert result.content["url_requested"] == "https://aparat.com"


def test_navigate_reuses_existing_browser_window_and_reports_actual_address(tmp_path, monkeypatch):
    tool = ComputerUseTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "reuse-browser-window-test")
    tool._navigation_browser = lambda requested_app="": ("firefox", "/usr/bin/firefox")
    initial = {
        "app_name": "Old page — Mozilla Firefox", "title": "Old page — Mozilla Firefox",
        "pid": 4321, "window_id": 1234, "is_on_screen": True,
        "width": 1280, "height": 720, "z_index": 2,
    }
    navigated = {**initial, "title": "Hugging Face — Mozilla Firefox", "app_name": "Hugging Face — Mozilla Firefox"}
    inventory_calls = {"count": 0}

    def windows():
        inventory_calls["count"] += 1
        return [initial] if inventory_calls["count"] == 1 else [navigated]

    tool._inspect._windows = windows
    tool._ensure_window_front = lambda target: True
    observed_capture = {
        "window": {"app": navigated["app_name"], "title": navigated["title"], "pid": 4321, "window_id": 1234},
        "mode": "som",
        "elements": [
            {"id": 1, "role": "combo box", "label": "Search with Google or enter address",
             "value": "https://huggingface.co/settings/hardware", "enabled": True},
        ],
        "element_count": 1,
    }
    tool._capture = lambda **kwargs: ToolResult(
        success=True, name="computer_use", content=dict(observed_capture),
        metadata={"image_ref": "/tmp/.daena/screenshots/current.png", "image_mime_type": "image/png"},
        summary="Captured current browser page.",
    )
    driver_calls = []
    monkeypatch.setattr(
        computer_use_module, "call_driver",
        lambda name, arguments=None, **kwargs: driver_calls.append((name, dict(arguments or {}))) or '{"ok":true}',
    )
    from unittest.mock import patch
    with patch.object(computer_use_module.subprocess, "Popen") as popen, \
         patch.object(computer_use_module.time, "sleep", lambda *_: None):
        result = tool._navigate("https://huggingface.co/settings/hardware", 0, "Firefox")

    assert result.success is True
    popen.assert_not_called()
    assert [name for name, _ in driver_calls] == ["hotkey", "type_text", "press_key"]
    assert result.content["navigation_reused_existing_browser_window"] is True
    assert result.content["observed_url"] == "https://huggingface.co/settings/hardware"
    assert result.content["navigation_status"] == "address_matches_request"
    assert result.content["destination_verified"] is True


def test_navigation_address_status_marks_redirect_as_unverified():
    status, verified = ComputerUseTool._navigation_address_status(
        "https://blubank.com/", "https://blubank.com/hi"
    )
    assert status == "redirected_within_same_host"
    assert verified is False


def test_navigation_address_status_does_not_infer_https_when_browser_hides_scheme():
    status, verified = ComputerUseTool._navigation_address_status(
        "https://huggingface.co/settings/hardware", "huggingface.co/settings/hardware"
    )
    assert status == "address_host_path_matches_scheme_unobservable"
    assert verified is False


def test_navigation_address_status_rejects_scheme_downgrade():
    status, verified = ComputerUseTool._navigation_address_status(
        "https://blubank.com/", "http://blubank.com/"
    )
    assert status == "scheme_changed"
    assert verified is False


def test_focus_ignores_auxiliary_same_process_rows_when_checking_occlusion(tmp_path, monkeypatch):
    tool = ComputerUseTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "same-pid-focus-test")
    windows = [
        {"pid": 100, "window_id": 7, "title": "Firefox", "app_name": "Firefox",
         "is_on_screen": True, "width": 2560, "height": 1400, "z_index": 2},
        # Driver can report an auxiliary row from the same app/process above the
        # main row even though the accessibility target is PID-scoped.
        {"pid": 100, "window_id": 24, "title": "", "app_name": "",
         "is_on_screen": True, "width": 279, "height": 211, "z_index": 3},
        {"pid": 200, "window_id": 3, "title": "Terminal", "app_name": "Terminal",
         "is_on_screen": True, "width": 1344, "height": 937, "z_index": 1},
    ]
    calls = []
    tool._inspect._windows = lambda: windows
    monkeypatch.setattr(computer_use_module, "call_driver", lambda *args, **kwargs: calls.append(args))

    assert tool._ensure_window_front(windows[0]) is True
    assert calls == []


def test_focus_reresolves_window_id_that_changes_during_browser_launch(tmp_path):
    tool = ComputerUseTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "stale-window-id-focus-test")
    target = {"pid": 100, "window_id": 7, "title": "Hugging Face — Mozilla Firefox", "app": "Hugging Face"}
    tool._inspect._windows = lambda: [
        {"pid": 100, "window_id": 31, "title": "Hugging Face — Mozilla Firefox",
         "app_name": "Hugging Face — Mozilla Firefox", "is_on_screen": True,
         "width": 2560, "height": 1400, "z_index": 2},
        {"pid": 200, "window_id": 3, "title": "Terminal", "app_name": "Terminal",
         "is_on_screen": True, "width": 1344, "height": 937, "z_index": 1},
    ]

    assert tool._ensure_window_front(target) is True
    assert target["window_id"] == 31


def test_actions_capture_after_by_default_unless_explicitly_disabled(tmp_path):
    tool = ComputerUseTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "auto-capture-default-test")
    from src.tools.builtin.computer_use.driver import pointer_state_path
    state_path = pointer_state_path(tmp_path, tool._observe._session_label())
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"latest_screenshot_hash": "unchanged-screen"}),
                          encoding="utf-8")
    tool._act_once = lambda action, **kwargs: ToolResult(
        success=True, name="computer_use", content={"action": action},
        metadata={}, summary="Action dispatched.", evidence={}, effects=[],
    )
    captures = []

    def fake_capture(**kwargs):
        captures.append(kwargs)
        return ToolResult(
            success=True, name="computer_use",
            content={"action": "capture", "window": {}, "elements": [],
                     "element_count": 0, "mode": "som"},
            metadata={}, summary="Captured the desktop.", evidence={},
        )

    tool._capture = fake_capture
    result = tool.execute(action="key", keys="ENTER")
    assert result.success is True
    assert len(captures) == 1
    assert result.content["capture_after"]["success"] is True
    assert result.content["capture_after"]["screen_hash_changed"] is False
    assert "do not repeat the same click" in result.content["capture_after"]["no_visible_change_warning"]
    assert "No screenshot pixel change detected" in result.summary

    captures.clear()
    result = tool.execute(action="key", keys="ENTER", capture_after=False)
    assert result.success is True
    assert captures == []


def test_desktop_action_auto_escalates_only_on_driver_recommended_foreground(tmp_path, monkeypatch):
    from src.tools.builtin.computer_use.driver import pointer_state_path

    tool = ComputerActTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "auto-foreground-test")
    state_path = pointer_state_path(tmp_path, tool._session_label())
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"latest_capture_id": "capture-auto-foreground"}),
                          encoding="utf-8")
    calls = []

    def fake_call(name, arguments=None, **kwargs):
        calls.append((name, dict(arguments or {})))
        if len(calls) == 1:
            raise act_module.CuaDriverError(
                'Cua Driver action "click" refused: {"code":"background_unavailable",'
                '"escalation":{"recommended":"foreground"}}'
            )
        return '{"ok":true}'

    monkeypatch.setattr(act_module, "call_driver", fake_call)
    result = tool.execute(action="click", x=100, y=200)

    assert result.success is True
    assert len(calls) == 2
    assert calls[0][1]["delivery_mode"] == "background"
    assert calls[1][1]["delivery_mode"] == "foreground"
    assert result.metadata["delivery_mode_escalated"] is True
    assert result.metadata["delivery_mode"] == "foreground"



def test_navigation_browser_detection_prefers_visible_firefox(tmp_path, monkeypatch):
    tool = ComputerUseTool()
    tool.set_workspace(tmp_path)
    tool._inspect._windows = lambda: [
        {"app_name": "terminal", "title": "python -m cli", "pid": 100,
         "window_id": 1, "is_on_screen": True, "width": 1200, "height": 800, "z_index": 0},
        {"app_name": "Firefox", "title": "Aparat — Mozilla Firefox", "pid": 200,
         "window_id": 2, "is_on_screen": True, "width": 2560, "height": 1400, "z_index": 1},
    ]
    monkeypatch.setattr(
        computer_use_module.shutil, "which",
        lambda candidate: "/usr/bin/firefox" if candidate == "firefox" else None,
    )

    assert tool._navigation_browser() == ("firefox", "/usr/bin/firefox")


def test_direct_element_typing_clears_previous_click_guard(tmp_path):
    tool = ComputerUseTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "direct-type-progress-test")
    state_path = driver_module.pointer_state_path(tmp_path, tool._observe._session_label())
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "latest_element_ids": {"1": "sphone:1"},
        "latest_accessibility_elements": {
            "sphone:1": {"role": "text field", "label": "شماره تلفن همراه", "actions": ["setValue"]}
        },
        "last_element_click": "push button|ورود به حساب آپارات",
    }), encoding="utf-8")
    calls = []

    def fake_act(action, **kwargs):
        calls.append((action, kwargs))
        return ToolResult(success=True, name="computer_act", content={"action": action},
                          metadata={}, summary="typed", evidence={}, effects=[])

    tool._act.execute = fake_act
    result = tool._act_once("type", element=1, text="09000000000")

    assert result.success is True
    assert calls[0][0] == "type_text"
    assert calls[0][1]["element_token"] == "sphone:1"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "last_element_click" not in state


def test_screenshot_history_can_keep_three_chronological_computer_use_frames(tmp_path):
    paths = []
    events = []
    task = _event(ContextRole.USER, ContextType.MESSAGE, "Open Aparat and enter the phone number.", 1)
    events.append(task)
    for number in range(1, 6):
        image_path = tmp_path / ".daena" / "screenshots" / "session" / f"screen-{number}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(PNG_HEADER)
        paths.append(image_path)
        call_id = f"capture-{number}"
        assistant = _event(
            ContextRole.ASSISTANT, ContextType.MESSAGE, "", number * 2,
            metadata={"llm_message": {
                "role": "assistant", "content": "", "tool_calls": [{
                    "id": call_id, "type": "function",
                    "function": {"name": "computer_use", "arguments": '{"action":"capture"}'},
                }],
            }},
        )
        payload = {
            "name": "computer_use", "tool_call_id": call_id,
            "success": True, "summary": "Captured desktop", "evidence": {}, "effects": [],
            "content": {
                "action": "capture", "image_ref": str(image_path),
                "image_mime_type": "image/png",
                "screenshot_hash": f"screen-state-{1 if number in (1, 2) else number}",
            },
            "metadata": {
                "kind": "computer_use_capture", "action": "capture",
                "image_ref": str(image_path), "image_mime_type": "image/png",
                "screenshot_hash": f"screen-state-{1 if number in (1, 2) else number}",
            },
        }
        result = _event(ContextRole.TOOL, ContextType.TOOL_RESULT, json.dumps(payload), number * 2 + 1)
        events.extend([assistant, result])

    builder = object.__new__(ContextBuilder)
    builder.preserve_thinking = False
    builder.screenshot_history_frames = 3
    messages = builder._build_conversation(
        events,
        task={"id": str(task.id), "content": task.content, "step": task.step},
    )

    attached = [m for m in messages if m.get("image_ref")]
    assert [m["image_ref"] for m in attached] == [str(paths[2]), str(paths[3]), str(paths[4])]
    assert "frame 1/3 (oldest to newest)" in attached[0]["content"]
    assert "frame 2/3 (oldest to newest)" in attached[1]["content"]
    assert "frame 3/3 (oldest to newest)" in attached[2]["content"]
    assert sum("Earlier screenshot is stored in STM" in str(m.get("content", "")) for m in messages) == 2


def test_screenshot_history_setting_is_clamped_to_one_through_five_frames():
    builder = object.__new__(ContextBuilder)
    builder.screenshot_history_frames = 3
    assert max(1, min(5, int(getattr(builder, "screenshot_history_frames", 1)))) == 3
    builder.screenshot_history_frames = 100
    assert max(1, min(5, int(getattr(builder, "screenshot_history_frames", 1)))) == 5
    builder.screenshot_history_frames = 0
    assert max(1, min(5, int(getattr(builder, "screenshot_history_frames", 1)))) == 1


def test_observe_draws_distinct_labelled_cua_cursor_marker(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    tool = ComputerObserveTool()
    tool.set_workspace(tmp_path)
    image_path = tmp_path / "cursor.png"
    image_module.new("RGB", (100, 100), (0, 0, 0)).save(image_path)

    assert tool._overlay_cursor_position(
        image_path, {"x": 50, "y": 50, "source": "synthetic"}
    ) is True
    with image_module.open(image_path) as marked:
        assert marked.size == (100, 100)
        assert marked.getpixel((58, 50)) == (30, 220, 235)


def test_cursor_position_is_allowlisted_for_computer_observation():
    assert "get_cursor_position" in driver_module.DRIVER_TOOLS


def test_type_on_activate_only_input_focuses_element_then_uses_keyboard(tmp_path):
    tool = ComputerUseTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "type-activate-fallback-test")
    state_path = driver_module.pointer_state_path(tmp_path, tool._observe._session_label())
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "latest_element_ids": {"1": "sentry:1"},
        "latest_accessibility_elements": {
            "sentry:1": {
                "role": "entry", "label": "موبایل یا ایمیل خود را وارد کنید:",
                "actions": ["activate"], "frame": {"x": 10, "y": 10, "w": 200, "h": 32},
            }
        },
    }), encoding="utf-8")
    calls = []

    def fake_act(**kwargs):
        calls.append(kwargs)
        return ToolResult(
            success=True, name="computer_act", content={"action": kwargs["action"]},
            metadata={}, summary="mocked success", evidence={}, effects=[],
        )

    tool._act.execute = fake_act
    result = tool._act_once("type", element=1, text="sample test text")

    assert result.success is True
    assert len(calls) == 2
    assert calls[0]["action"] == "click"
    assert calls[0]["x"] == 110
    assert calls[0]["y"] == 26
    assert calls[0]["delivery_mode"] == "foreground"
    assert calls[1]["action"] == "type_text"
    assert calls[1]["text"] == "sample test text"
    assert "element_token" not in calls[1]
    assert result.metadata["target_mode"] == "visible_field_coordinates_then_keyboard"


def test_type_on_input_with_setvalue_keeps_direct_element_typing(tmp_path):
    tool = ComputerUseTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "type-setvalue-direct-test")
    state_path = driver_module.pointer_state_path(tmp_path, tool._observe._session_label())
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "latest_element_ids": {"1": "sentry:2"},
        "latest_accessibility_elements": {
            "sentry:2": {
                "role": "text field", "label": "Search",
                "actions": ["setValue", "click"],
            }
        },
    }), encoding="utf-8")
    calls = []
    tool._act.execute = lambda **kwargs: calls.append(kwargs) or ToolResult(
        success=True, name="computer_act", content={"action": kwargs["action"]},
        metadata={}, summary="mocked success", evidence={}, effects=[],
    )

    result = tool._act_once("type", element=1, text="hello")

    assert result.success is True
    assert len(calls) == 1
    assert calls[0]["action"] == "type_text"
    assert calls[0]["element_token"] == "sentry:2"
    assert calls[0]["text"] == "hello"


def test_inspection_excludes_controls_outside_visible_window_bounds():
    title = "Aparat — Mozilla Firefox"
    payload = {
        "window_title": title,
        "window_bounds": {"x": 0, "y": 31, "width": 1000, "height": 600},
        "elements": [
            {"element_index": 0, "parent_index": None, "role": "frame", "label": title,
             "frame": {"x": 0, "y": 31, "w": 1000, "h": 600}, "element_token": "root",
             "actions": ["click"], "enabled": True},
            {"element_index": 1, "parent_index": 0, "role": "push button", "label": "Visible login",
             "frame": {"x": 50, "y": 150, "w": 120, "h": 40}, "element_token": "visible",
             "actions": ["press"], "enabled": True, "in_web_content": True},
            {"element_index": 2, "parent_index": 0, "role": "push button", "label": "Old background-tab login",
             "frame": {"x": 50, "y": -3400, "w": 120, "h": 40}, "element_token": "off-top",
             "actions": ["press"], "enabled": True, "in_web_content": True},
            {"element_index": 3, "parent_index": 0, "role": "push button", "label": "Off-right control",
             "frame": {"x": 1200, "y": 150, "w": 120, "h": 40}, "element_token": "off-right",
             "actions": ["press"], "enabled": True, "in_web_content": True},
        ],
    }

    ranked = ComputerInspectTool._rank_elements(payload, "", limit=30)
    labels = [item["label"] for item in ranked]

    assert "Visible login" in labels
    assert "Old background-tab login" not in labels
    assert "Off-right control" not in labels



def test_stale_accessibility_handle_is_re_resolved_and_retried_once(tmp_path):
    tool = ComputerUseTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "stale-handle-recovery-test")
    state_path = driver_module.pointer_state_path(tmp_path, tool._observe._session_label())
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "latest_element_ids": {"1": "old-token"},
        "latest_accessibility_elements": {
            "old-token": {"role": "push button", "label": "Continue", "actions": ["click"]}
        },
        "latest_computer_use_target": {"pid": 123, "window_id": 456},
    }), encoding="utf-8")

    calls = []

    def fake_act(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return ToolResult(
                success=False, name="computer_act",
                content={"error": {"message": 'stale_element_token {"code":"stale_element_token","effect":"none"}'}},
                metadata={}, summary="stale handle", evidence={}, effects=[],
            )
        return ToolResult(
            success=True, name="computer_act", content={"action": "click"},
            metadata={}, summary="clicked", evidence={}, effects=[],
        )

    def fake_capture(**kwargs):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["latest_element_ids"] = {"1": "fresh-token"}
        state["latest_accessibility_elements"] = {
            "fresh-token": {"role": "push button", "label": "Continue", "actions": ["click"]}
        }
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return ToolResult(
            success=True, name="computer_use",
            content={"elements": [{"id": 1, "role": "push button", "label": "Continue"}]},
            metadata={}, summary="Fresh accessibility snapshot", evidence={},
        )

    tool._act.execute = fake_act
    tool._capture = fake_capture
    result = tool._act_once("click", element=1)

    assert result.success is True
    assert result.metadata["stale_handle_recovered"] is True
    assert len(calls) == 2
    assert calls[1]["element_token"] == "fresh-token"


def test_computer_use_resolves_exact_target_label_without_a_query_capture(tmp_path):
    tool = ComputerUseTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "semantic-target-test")
    state_path = driver_module.pointer_state_path(tmp_path, tool._observe._session_label())
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "latest_element_ids": {"1": "continue-token"},
        "latest_accessibility_elements": {
            "continue-token": {
                "role": "push button", "label": "Continue", "enabled": True,
                "actions": ["press"],
            }
        },
        "latest_computer_use_target": {"pid": 123, "window_id": 456},
    }), encoding="utf-8")
    calls = []

    def fake_act(action, **kwargs):
        calls.append((action, kwargs))
        return ToolResult(success=True, name="computer_act", content={"action": action},
                          metadata={}, summary="mock dispatched", evidence={}, effects=[])

    tool._act.execute = fake_act
    captures = []
    tool._capture = lambda **kwargs: captures.append(kwargs) or ToolResult(
        success=True, name="computer_use", content={"elements": []},
        metadata={}, summary="fresh accessibility snapshot", evidence={},
    )
    result = tool.execute(action="click", target="Continue", capture_after=False)

    assert result.success is True
    assert len(calls) == 1
    assert captures == []
    assert calls[0][1]["element_token"] == "continue-token"


def test_computer_use_refuses_ambiguous_semantic_target_without_dispatch(tmp_path):
    tool = ComputerUseTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "semantic-target-ambiguous-test")
    state_path = driver_module.pointer_state_path(tmp_path, tool._observe._session_label())
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "latest_element_ids": {"1": "continue-a", "2": "continue-b"},
        "latest_accessibility_elements": {
            "continue-a": {"role": "push button", "label": "Continue", "enabled": True, "actions": ["press"]},
            "continue-b": {"role": "push button", "label": "Continue", "enabled": True, "actions": ["press"]},
        },
    }), encoding="utf-8")
    calls = []
    tool._act.execute = lambda action, **kwargs: calls.append((action, kwargs)) or ToolResult(
        success=True, name="computer_act", content={}, metadata={}, summary="should not run", evidence={}, effects=[]
    )
    tool._capture = lambda **kwargs: ToolResult(
        success=True, name="computer_use", content={"elements": []}, metadata={},
        summary="fresh accessibility snapshot", evidence={},
    )

    result = tool.execute(action="click", target="Continue", capture_after=False)

    assert result.success is False
    assert calls == []
    assert "multiple current controls" in result.summary


def test_computer_use_sequence_batches_actions_and_stops_at_fresh_state(tmp_path):
    tool = ComputerUseTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "sequence-batch-test")
    state_path = driver_module.pointer_state_path(tmp_path, tool._observe._session_label())
    state_path.parent.mkdir(parents=True, exist_ok=True)
    phase = {"value": "login"}
    typed_text = {"value": ""}
    dispatched = []

    def controls_for_phase():
        if phase["value"] == "continued":
            rows = [{"id": 1, "token": "code-field", "role": "entry", "label": "Verification code", "actions": ["setValue"]}]
        else:
            rows = [
                {"id": 1, "token": "phone-field", "role": "text field", "label": "Phone number", "actions": ["setValue"],
                 "value": typed_text["value"] if phase["value"] == "typed" else ""},
                {"id": 2, "token": "continue-button", "role": "push button", "label": "Continue", "actions": ["press"]},
            ]
        return rows

    def fake_capture(*, mode="som", **kwargs):
        rows = controls_for_phase()
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        state["latest_element_ids"] = {str(row["id"]): row["token"] for row in rows}
        state["latest_accessibility_elements"] = {
            row["token"]: {
                "role": row["role"], "label": row["label"], "actions": row["actions"],
                "value": row.get("value"), "enabled": True,
            }
            for row in rows
        }
        state["latest_computer_use_target"] = {"pid": 123, "window_id": 456, "title": "Mock"}
        state_path.write_text(json.dumps(state), encoding="utf-8")
        elements = [{key: value for key, value in row.items() if key != "token"} for row in rows]
        elements_text = "\n".join(f"#{row['id']} {row['role']} {row['label']}" for row in rows)
        content = {
            "action": "capture", "mode": mode,
            "window": {"pid": 123, "window_id": 456, "title": "Mock"},
            "element_count": len(rows), "elements": elements, "elements_text": elements_text,
        }
        if mode != "ax":
            content.update({"image_ref": None, "image_mime_type": "image/png", "screenshot_hash": phase["value"]})
            state["latest_screenshot_hash"] = phase["value"]
            state_path.write_text(json.dumps(state), encoding="utf-8")
        return ToolResult(success=True, name="computer_use", content=content, metadata={},
                          summary="Mock UI state\nControls:\n" + elements_text, evidence={})

    def fake_act(action, **kwargs):
        dispatched.append((action, kwargs.get("element_token"), kwargs.get("text")))
        if action == "type_text":
            phase["value"] = "typed"
            typed_text["value"] = kwargs.get("text", "")
        elif action == "click":
            phase["value"] = "continued"
        return ToolResult(success=True, name="computer_act", content={"action": action},
                          metadata={}, summary=f"{action} dispatched", evidence={}, effects=[])

    tool._capture = fake_capture
    tool._act.execute = fake_act
    # Seed a visual screenshot so the first state-changing step has a comparable baseline.
    fake_capture(mode="som")

    result = tool.execute(action="sequence", steps=[
        {"action": "type", "target": "Phone number", "text": "test value"},
        {"action": "click", "target": "Continue"},
    ])

    assert result.success is True
    assert result.content["complete"] is True
    assert result.content["completed_steps"] == 2
    assert [item[0] for item in dispatched] == ["type_text", "click"]
    assert dispatched[0][1] == "phone-field"
    assert dispatched[1][1] == "continue-button"
    assert "Verification code" in result.content["elements_text"]



def test_inspect_filters_inactive_tabs_and_other_windows_but_keeps_nested_modal_controls():
    title = "آپارات - سرویس اشتراک ویدیو — Mozilla Firefox"
    payload = {
        "window_title": title,
        "window_bounds": {"x": 0, "y": 31, "width": 2560, "height": 1409},
        "elements": [
            {"element_index": 0, "role": "frame", "label": title, "parent_index": None, "element_token": "sactive:0", "frame": {"x": 0, "y": 31, "w": 2560, "h": 1409}},
            {"element_index": 1, "role": "tool bar", "label": "Browser tabs", "parent_index": 0, "element_token": "sactive:1", "in_web_content": False},
            {"element_index": 2, "role": "page tab list", "label": "Browser tabs", "parent_index": 1, "element_token": "sactive:2", "in_web_content": False},
            {"element_index": 3, "role": "page tab", "label": "آپارات - سرویس اشتراک ویدیو", "parent_index": 2, "selected": True, "element_token": "sactive:3", "enabled": True, "in_web_content": False, "frame": {"x": 100, "y": 0, "w": 210, "h": 40}},
            {"element_index": 4, "role": "page tab", "label": "Diagnose Browser Login Issues", "parent_index": 2, "selected": False, "element_token": "sactive:4", "enabled": True, "in_web_content": False, "frame": {"x": 310, "y": 0, "w": 210, "h": 40}},
            {"element_index": 5, "role": "combo box", "label": "Search with Google or enter address", "parent_index": 0, "element_token": "sactive:5", "actions": ["activate"], "enabled": True, "in_web_content": False, "frame": {"x": 300, "y": 47, "w": 1000, "h": 29}},
            {"element_index": 6, "role": "panel", "label": "Browser content", "parent_index": 0, "element_token": "sactive:6", "in_web_content": False},
            {"element_index": 7, "role": "scroll pane", "label": "Aparat page", "parent_index": 6, "element_token": "sactive:7", "in_web_content": True, "frame": {"x": 52, "y": 82, "w": 2508, "h": 1327}},
            {"element_index": 8, "role": "internal frame", "label": "Aparat", "parent_index": 7, "element_token": "sactive:8", "in_web_content": True},
            {"element_index": 10, "role": "document web", "label": "آپارات - سرویس اشتراک ویدیو", "parent_index": 8, "element_token": "sactive:10", "in_web_content": True, "frame": {"x": 52, "y": 82, "w": 2508, "h": 1327}},
            {"element_index": 11, "role": "push button", "label": "ورود به حساب آپارات", "parent_index": 10, "element_token": "sactive:11", "actions": ["press"], "enabled": True, "in_web_content": True, "frame": {"x": 75, "y": 122, "w": 127, "h": 39}},
            {"element_index": 12, "role": "entry", "label": "Search videos", "parent_index": 10, "element_token": "sactive:12", "actions": ["setValue"], "enabled": True, "in_web_content": True, "frame": {"x": 400, "y": 100, "w": 300, "h": 32}},
            {"element_index": 14, "role": "dialog", "label": "Sign-in dialog", "parent_index": 10, "element_token": "sactive:14", "in_web_content": True, "frame": {"x": 1000, "y": 450, "w": 500, "h": 400}},
            {"element_index": 15, "role": "section", "label": "Dialog content", "parent_index": 14, "element_token": "sactive:15", "in_web_content": True},
            {"element_index": 16, "role": "internal frame", "label": "Auth Modal", "parent_index": 15, "element_token": "sactive:16", "in_web_content": True},
            {"element_index": 17, "role": "document web", "label": "Auth Modal", "parent_index": 16, "element_token": "sactive:17", "in_web_content": True, "frame": {"x": 1079, "y": 486, "w": 455, "h": 581}},
            {"element_index": 18, "role": "entry", "label": "موبایل یا ایمیل خود را وارد کنید", "parent_index": 17, "element_token": "sactive:18", "actions": ["activate"], "enabled": True, "in_web_content": True, "frame": {"x": 1123, "y": 634, "w": 366, "h": 46}},
            {"element_index": 20, "role": "panel", "label": "Other browser content", "parent_index": 0, "element_token": "sactive:20", "in_web_content": False},
            {"element_index": 21, "role": "document web", "label": "Diagnose Browser Login Issues", "parent_index": 20, "element_token": "sactive:21", "in_web_content": True, "frame": {"x": 52, "y": 82, "w": 2508, "h": 1327}},
            {"element_index": 22, "role": "push button", "label": "Search", "parent_index": 21, "element_token": "sactive:22", "actions": ["press"], "enabled": True, "in_web_content": True, "frame": {"x": 300, "y": 120, "w": 34, "h": 35}},
            {"element_index": 23, "role": "toggle button", "label": "Read aloud", "parent_index": 21, "element_token": "sactive:23", "actions": ["press"], "enabled": True, "in_web_content": True, "frame": {"x": 1200, "y": 600, "w": 30, "h": 30}},
            {"element_index": 30, "role": "frame", "label": "Diagnose Browser Login Issues — Mozilla Firefox", "parent_index": None, "element_token": "sactive:30", "frame": {"x": 0, "y": 31, "w": 2560, "h": 1409}},
            {"element_index": 31, "role": "tool bar", "label": "Browser tabs", "parent_index": 30, "element_token": "sactive:31", "in_web_content": False},
            {"element_index": 32, "role": "page tab", "label": "Diagnose Browser Login Issues", "parent_index": 31, "selected": True, "element_token": "sactive:32", "in_web_content": False, "frame": {"x": 300, "y": 0, "w": 210, "h": 40}},
            {"element_index": 33, "role": "document web", "label": "Diagnose Browser Login Issues", "parent_index": 30, "element_token": "sactive:33", "in_web_content": True, "frame": {"x": 52, "y": 82, "w": 2508, "h": 1327}},
            {"element_index": 34, "role": "push button", "label": "Copy message", "parent_index": 33, "element_token": "sactive:34", "actions": ["press"], "enabled": True, "in_web_content": True, "frame": {"x": 900, "y": 800, "w": 30, "h": 30}},
        ],
    }

    ranked = ComputerInspectTool._rank_elements(payload, "", limit=30)
    labels = [item["label"] for item in ranked]
    assert "ورود به حساب آپارات" in labels
    assert "موبایل یا ایمیل خود را وارد کنید" in labels
    assert "Search videos" in labels
    assert "Search with Google or enter address" in labels
    assert "Search" not in labels
    assert "Read aloud" not in labels
    assert "Copy message" not in labels


def test_inspect_window_selection_prefers_frontmost_equal_relevance_window():
    tool = ComputerInspectTool()
    tool._windows = lambda: [
        {"app_name": "Firefox", "title": "Aparat", "pid": 1, "window_id": 11,
         "is_on_screen": True, "width": 2560, "height": 1440, "z_index": 0},
        {"app_name": "Firefox", "title": "Chat page", "pid": 1, "window_id": 12,
         "is_on_screen": True, "width": 1280, "height": 720, "z_index": 3},
    ]

    pid, window_id, title, _app = tool._resolve_target(None, None, "", "")

    assert (pid, window_id, title) == (1, 12, "Chat page")


def test_computer_use_drag_is_exposed_for_general_pointer_gestures():
    action_schema = ComputerUseTool.parameters["properties"]["action"]
    assert "drag" in action_schema["enum"]
    assert "from_coordinate" in ComputerUseTool.parameters["properties"]
    assert "to_coordinate" in ComputerUseTool.parameters["properties"]
    sequence_step = ComputerUseTool.parameters["properties"]["steps"]["items"]
    assert "drag" in sequence_step["properties"]["action"]["enum"]
    assert "duration_ms" in sequence_step["properties"]


def test_computer_use_maps_drag_coordinates_without_element_ids(tmp_path):
    tool = ComputerUseTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "drag-mapping-test")
    calls = []

    def fake_act(action, **kwargs):
        calls.append((action, kwargs))
        return ToolResult(success=True, name="computer_act", content={"action": action},
                          metadata={}, summary="drag dispatched", evidence={}, effects=[])

    def fake_capture(**kwargs):
        return ToolResult(success=True, name="computer_use", content={
            "action": "capture", "mode": kwargs.get("mode", "som"),
            "window": {"title": "Paint", "pid": 7, "window_id": 8},
            "elements": [], "element_count": 0,
            "image_ref": None, "screenshot_hash": "same",
        }, metadata={}, summary="Captured Paint", evidence={})

    tool._act.execute = fake_act
    tool._capture = fake_capture
    result = tool.execute(action="drag", from_coordinate=[10, 20], to_coordinate=[80, 90],
                          duration_ms=350, capture_after=False)

    assert result.success is True
    assert calls == [("drag", {
        "from_x": 10.0, "from_y": 20.0, "to_x": 80.0, "to_y": 90.0,
        "duration_ms": 350,
    })]


def test_computer_act_dispatches_real_drag_with_desktop_coordinates(tmp_path, monkeypatch):
    calls = []

    def fake_call(name, arguments=None, *, timeout=30.0):
        calls.append((name, arguments, timeout))
        return '{"ok":true,"effect":"dispatched"}'

    monkeypatch.setattr(act_module, "call_driver", fake_call)
    tool = ComputerActTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "act-drag-dispatch-test")
    from src.tools.builtin.computer_use.driver import pointer_state_path
    state_path = pointer_state_path(tmp_path, tool._session_label())
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"latest_capture_id": "capture-drag-test"}), encoding="utf-8")

    result = tool.execute(action="drag", from_x=10, from_y=20, to_x=80, to_y=90,
                          duration_ms=350, button="left")

    assert result.success is True
    assert calls[0][0] == "drag"
    assert calls[0][1]["coordinate_frame"] == "desktop"
    assert calls[0][1]["target"] == {"kind": "desktop", "display_id": "primary"}
    assert calls[0][1]["from_x"] == 10
    assert calls[0][1]["from_y"] == 20
    assert calls[0][1]["to_x"] == 80
    assert calls[0][1]["to_y"] == 90
    assert calls[0][1]["duration_ms"] == 350
    assert calls[0][1]["button"] == "left"


def test_inspect_generic_window_selection_does_not_prefer_background_browser_over_front_app():
    tool = ComputerInspectTool()
    tool._windows = lambda: [
        {"app_name": "Firefox", "title": "Aparat", "pid": 1, "window_id": 11,
         "is_on_screen": True, "width": 2560, "height": 1440, "z_index": 0},
        {"app_name": "Paint", "title": "Untitled - Paint", "pid": 2, "window_id": 22,
         "is_on_screen": True, "width": 1280, "height": 800, "z_index": 4},
    ]

    pid, window_id, title, app = tool._resolve_target(None, None, "", "")

    assert (pid, window_id, title, app) == (2, 22, "Untitled - Paint", "Paint")


def test_computer_act_uses_desktop_coordinate_frame_for_targeted_text_input():
    args = ComputerActTool._driver_args(
        "type_text",
        {"text": "typing test", "x": 450, "y": 360, "delivery_mode": "background"},
        "type-coordinate-session",
    )
    assert args["coordinate_frame"] == "desktop"
    assert args["target"] == {"kind": "desktop", "display_id": "primary"}
    assert (args["x"], args["y"], args["text"]) == (450, 360, "typing test")


def test_computer_use_types_into_activate_only_field_via_current_visible_bounds(tmp_path):
    from src.tools.builtin.computer_use.driver import pointer_state_path

    tool = ComputerUseTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "activate-only-text-coordinate-test")
    state_path = pointer_state_path(tmp_path, tool._observe._session_label())
    state_path.parent.mkdir(parents=True, exist_ok=True)
    token = "sabcdef12:7"
    state_path.write_text(json.dumps({
        "latest_element_ids": {"1": token},
        "latest_accessibility_elements": {
            token: {
                "role": "entry",
                "label": "Demo text field",
                "actions": ["activate"],
                "frame": {"x": 100, "y": 200, "w": 200, "h": 40},
                "in_web_content": True,
            }
        },
        "latest_accessibility_pid": 123,
        "latest_accessibility_window_id": 456,
    }), encoding="utf-8")
    calls = []

    def fake_act(action, **kwargs):
        calls.append((action, kwargs))
        return ToolResult(success=True, name="computer_act", content={
            "action": action, "driver_output": "typed"},
            metadata={"delivery_mode": kwargs.get("delivery_mode", "background")},
            summary="typed", evidence={}, effects=[])

    tool._act.execute = fake_act
    result = tool._act_once("type", element=1, text="hello canvas")

    assert result.success is True
    assert len(calls) == 2
    action, args = calls[0]
    assert action == "click"
    assert args["x"] == 200
    assert args["y"] == 220
    assert args["delivery_mode"] == "foreground"
    action, args = calls[1]
    assert action == "type_text"
    assert args["text"] == "hello canvas"
    assert "element_token" not in args
    assert result.content["target_mode"] == "visible_field_coordinates_then_keyboard"


def test_coordinate_type_focuses_point_before_sending_keyboard_text(tmp_path):
    tool = ComputerUseTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "coordinate-type-focus-test")
    calls = []

    def fake_act(action, **kwargs):
        calls.append((action, kwargs))
        return ToolResult(success=True, name="computer_act", content={
            "action": action, "driver_output": "mocked"},
            metadata={}, summary="mocked success", evidence={}, effects=[])

    tool._act.execute = fake_act
    result = tool._act_once("type", coordinate=[400, 300], text="visible input test")

    assert result.success is True
    assert [action for action, _kwargs in calls] == ["click", "type_text"]
    assert calls[0][1] == {"x": 400, "y": 300, "delivery_mode": "foreground"}
    assert calls[1][1] == {"text": "visible input test", "delivery_mode": "background"}
    assert result.content["target_mode"] == "visible_coordinate_then_keyboard"


def test_computer_use_sequence_pauses_before_submit_when_typed_value_is_unreadable(tmp_path):
    tool = ComputerUseTool()
    tool.set_workspace(tmp_path)
    tool.set_context_store(None, lambda: "sequence-unverified-type-test")
    state_path = driver_module.pointer_state_path(tmp_path, tool._observe._session_label())
    state_path.parent.mkdir(parents=True, exist_ok=True)
    phase = {"value": "login"}
    dispatched = []

    def fake_capture(*, mode="som", **kwargs):
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        rows = [
            {"id": 1, "token": "phone-field", "role": "entry", "label": "Phone number",
             "actions": ["activate"], "value": None, "enabled": True,
             "frame": {"x": 100, "y": 200, "w": 200, "h": 40}},
            {"id": 2, "token": "continue-button", "role": "push button", "label": "Continue",
             "actions": ["press"], "enabled": True},
        ]
        state["latest_element_ids"] = {str(row["id"]): row["token"] for row in rows}
        state["latest_accessibility_elements"] = {
            row["token"]: {key: value for key, value in row.items() if key not in {"id", "token"}}
            for row in rows
        }
        state["latest_computer_use_target"] = {"pid": 123, "window_id": 456, "title": "Mock"}
        if mode != "ax":
            state["latest_screenshot_hash"] = phase["value"]
        state_path.write_text(json.dumps(state), encoding="utf-8")
        content = {
            "action": "capture", "mode": mode,
            "window": {"pid": 123, "window_id": 456, "title": "Mock"},
            "element_count": len(rows), "elements": rows,
            "elements_text": "#1 entry Phone number\n#2 push button Continue",
            "screenshot_hash": phase["value"] if mode != "ax" else None,
            "image_ref": "/tmp/mock.png" if mode != "ax" else None,
            "image_mime_type": "image/png" if mode != "ax" else None,
        }
        return ToolResult(success=True, name="computer_use", content=content, metadata={},
                          summary="Mock UI state", evidence={})

    def fake_act(action, **kwargs):
        dispatched.append((action, kwargs.get("element_token"), kwargs.get("text")))
        phase["value"] = "after-type"
        return ToolResult(success=True, name="computer_act", content={"action": action},
                          metadata={}, summary=f"{action} dispatched", evidence={}, effects=[])

    tool._capture = fake_capture
    tool._act.execute = fake_act
    fake_capture(mode="som")
    result = tool.execute(action="sequence", steps=[
        {"action": "type", "target": "Phone number", "text": "sample value"},
        {"action": "click", "target": "Continue"},
    ])

    assert result.success is False
    assert result.content["complete"] is False
    assert result.content["completed_steps"] == 1
    assert [item[0] for item in dispatched] == ["click", "type_text"]
    assert result.content["step_results"][0]["verification"] == "text_value_not_read_back"
    assert "not run" in result.summary


def test_browser_target_selection_uses_generic_page_title_and_intent_not_site_rules():
    browser_rows = [
        {"app_name": "Firefox", "title": "Project Wiki — Mozilla Firefox",
         "pid": 7, "window_id": 70, "is_on_screen": True,
         "width": 1600, "height": 900, "z_index": 2},
        {"app_name": "Firefox", "title": "Mail — Mozilla Firefox",
         "pid": 7, "window_id": 71, "is_on_screen": True,
         "width": 1200, "height": 800, "z_index": 1},
    ]

    named_page = ComputerInspectTool._matching_browser_windows(browser_rows, "open Project Wiki login")
    generic_web = ComputerInspectTool._matching_browser_windows(browser_rows, "search the web for a topic")

    assert [row["window_id"] for row in named_page] == [70]
    assert [row["window_id"] for row in generic_web] == [70, 71]


def test_navigate_reuses_existing_same_site_root_instead_of_abandoning_form(tmp_path, monkeypatch):
    tool = ComputerUseTool()
    tool.set_workspace(tmp_path)
    tool._navigation_browser = lambda requested_app="": ("firefox", "/usr/bin/firefox")
    window = {
        "app_name": "Notion — Mozilla Firefox", "title": "Notion — Mozilla Firefox",
        "pid": 4321, "window_id": 1234, "is_on_screen": True,
        "width": 1280, "height": 720, "z_index": 3,
    }
    tool._inspect._windows = lambda: [window]
    activated = []
    tool._ensure_window_front = lambda target: activated.append(target) or True
    captured = {}

    def fake_capture(**kwargs):
        captured.update(kwargs)
        return ToolResult(
            success=True, name="computer_use",
            content={"window": {"title": window["title"], "pid": 4321, "window_id": 1234},
                     "elements": [], "element_count": 0, "mode": "som"},
            metadata={"image_ref": "/tmp/current-form.png"},
            summary="Current sign-in form is still open.",
        )

    tool._capture = fake_capture
    from unittest.mock import patch
    with patch.object(computer_use_module.subprocess, "Popen") as popen, \
         patch.object(computer_use_module.time, "sleep", lambda *_: None):
        result = tool._navigate("https://notion.so/", 0, "Firefox")

    assert result.success is True
    popen.assert_not_called()
    assert activated == [window]
    assert captured["pid"] == 4321
    assert captured["window_id"] == 1234
    assert result.content["navigation_reused_existing_site_window"] is True
    assert "no duplicate tab was opened" in result.summary
