import base64
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from computer_use.integration import append_pending_visual, gemini_image_parts, visual_message
from computer_use.runtime import ComputerError


@pytest.mark.parametrize("provider", ["ollama", "openrouter", "gemini"])
def test_screenshot_encoded_for_provider_without_text_history(engine, provider):
    tool, _ = engine
    result = tool.execute("capture")
    message = visual_message(result, provider, tool.workspace_root)
    assert message["role"] == "user"
    if provider == "ollama":
        data = message["images"][0]
    elif provider == "openrouter":
        data = message["content"][1]["image_url"]["url"].split(",", 1)[1]
    else:
        data = message["computer_image"]
    assert base64.b64decode(data).startswith(b"\x89PNG")


def test_vision_is_forwarded_once_and_only_after_computer_result(engine):
    tool, _ = engine
    result = tool.execute("capture")
    loop = SimpleNamespace(
        _computer_pending_visual=result,
        config={"llm": {"provider": "ollama"}},
        tool=SimpleNamespace(
            security=SimpleNamespace(sandbox=SimpleNamespace(root=tool.workspace_root))
        ),
    )
    messages = [{"role": "user", "content": "test task"}]
    forwarded = append_pending_visual(loop, messages)
    assert len(forwarded) == 2 and len(messages) == 1
    assert append_pending_visual(loop, messages) == messages


def test_image_path_cannot_escape_workspace(engine, tmp_path):
    tool, _ = engine
    result = tool.execute("capture")
    result.metadata["image_ref"] = str(tmp_path.parent / "outside.png")
    with pytest.raises(ComputerError, match="outside"):
        visual_message(result, "ollama", tool.workspace_root)


def test_gemini_native_parts_roundtrip(engine):
    types = pytest.importorskip("google.genai.types")
    tool, _ = engine
    message = visual_message(tool.execute("capture"), "gemini", tool.workspace_root)
    parts = gemini_image_parts(message, types)
    assert parts[1].inline_data.mime_type == "image/png"
    assert parts[1].inline_data.data.startswith(b"\x89PNG")


def test_host_installer_validates_before_writing(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "installer", Path(__file__).parents[1] / "install_into_daena.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "repo"
    (root / "src/tools").mkdir(parents=True)
    (root / "src/tools/Tool.py").write_text("# dummy base")
    (root / "src/tools/ToolManager.py").write_text("class ToolManager: pass")
    with pytest.raises(ValueError):
        module.install(root, wire_runtime=True)
    assert not (root / "src/tools/builtin/computer_use").exists()
    assert not (root / ".daena-computer-use-backups").exists()
