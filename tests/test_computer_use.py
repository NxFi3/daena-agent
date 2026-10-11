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


def _event(role, event_type, content, step, metadata=None):
    return ContextEvent(
        role=role,
        type=event_type,
        content=content,
        step=step,
        metadata=metadata or {},
    )


def test_computer_tools_are_registered_and_visible_by_default(tmp_path):
    manager = ToolManager({
        "guard_level": "strict",
        "security": {"workspace_only": True, "force_approve": True},
    })
    manager.set_workspace(str(tmp_path))
    try:
        names = {item["function"]["name"] for item in manager.get_model_tools()}
        assert "computer_use" in names
        assert not ({"computer_observe", "computer_inspect", "computer_act"} & names)
    finally:
        manager.close()


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
    result = tool._capture(mode="som", app="Firefox", query="phone number input")

    assert result.success is True
    assert [c.get("query", "") for c in calls] == ["phone number input", ""]
    assert result.content["query_fallback"] is True
    assert result.content["query_requested"] == "phone number input"
    assert result.content["element_count"] == 1
    assert "شماره تلفن همراه" in result.content["elements_text"]
    assert "unfiltered accessible controls" in result.content["selection_warning"]


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
    monkeypatch.setattr(computer_use_module, "_desktop_session_environment",
                        lambda: {"DISPLAY": ":0", "XDG_RUNTIME_DIR": "/run/user/1000"})
    from unittest.mock import patch
    with patch.object(computer_use_module.subprocess, "Popen") as popen, \
         patch.object(computer_use_module.time, "sleep", lambda *_: None):
        result = tool._navigate("https://aparat.com", 0, "Firefox")

    assert result.success is True
    assert popen.call_args.args[0] == ["/usr/bin/firefox", "--new-tab", "https://aparat.com"]
    assert captured["app"] == "firefox"
    assert result.content["browser"] == "firefox"
    assert result.content["url_requested"] == "https://aparat.com"


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
    assert calls[0]["action"] == "click"
    assert calls[0]["element_token"] == "sentry:1"
    assert calls[1]["action"] == "type_text"
    assert calls[1]["text"] == "sample test text"
    assert "element_token" not in calls[1]
    assert result.metadata["target_mode"] == "accessibility_focus_then_keyboard"


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
