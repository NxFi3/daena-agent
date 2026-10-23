"""Run manually after install_into_daena.py against a real Daena checkout.

PYTHONPATH=/path/to/daena python tests/host_smoke.py
"""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))
from conftest import FakeBackend
from src.agent.agent import Agent
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult
from src.engine.providers.builtin.ollama.ollama import OllamaProvider
from src.engine.providers.builtin.gemini.geminiprovider import GeminiProvider
from src.engine.transport import canonical_messages
from src.tools.builtin.computer_use.integration import visual_message


class Model:
    model = None

    def __init__(self):
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append(deepcopy(messages))
        if len(self.calls) == 1:
            call = {
                "id": "capture-1",
                "type": "function",
                "function": {"name": "computer_use", "arguments": '{"action":"capture"}'},
            }
            return LLMResult(
                "", {"role": "assistant", "content": "", "tool_calls": [call]}, [call], None, 1
            )
        return LLMResult(
            "Captured the fixture",
            {"role": "assistant", "content": "Captured the fixture"},
            [],
            None,
            1,
        )


with TemporaryDirectory() as directory:
    root = Path(directory)
    config = {
        "llm": {"provider": "ollama", "provider_config": {"generation_config": {"num_ctx": 32768}}},
        "workspace": {"directory": directory},
        "memory": {"database": str(root / "memory.sqlite")},
        "security": {"allowed_tools": ["computer_use"]},
        "max_agent_iterations": 3,
        "context": {"max_prompt_tokens": 24000},
        "experience": {"enabled": False},
    }
    model = Model()
    agent = Agent(config, llm=model)
    tool = agent.loop.tool.get_tool("computer_use")
    tool.backend = FakeBackend()
    tool.backend.rows.extend(
        {"element_token": f"extra-{i}", "label": "Long accessible label " * 30, "role": "entry"}
        for i in range(1013)
    )
    assert tool.workspace_root == root
    agent.act(
        ContextEvent(
            role=ContextRole.USER,
            type=ContextType.MESSAGE,
            content="Capture this test window once.",
        )
    )
    assert len(model.calls) == 2 and agent.last_run_metrics["completed"]
    images = [m for m in model.calls[1] if m.get("images")]
    assert len(images) == 1
    receipts = [json.loads(m["content"]) for m in model.calls[1] if m.get("role") == "tool"]
    assert receipts and receipts[-1]["evidence"]["snapshot_id"]
    assert receipts[-1]["evidence"]["next_offset"] is not None
    assert OllamaProvider._prepare_messages(images)[0]["images"] == images[0]["images"]
    result = tool.execute("capture")
    msg = visual_message(result, "openrouter", root)
    assert canonical_messages([msg])[0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    msg = visual_message(result, "gemini", root)
    native, _ = GeminiProvider._convert_messages([msg])
    assert native[0].parts[1].inline_data.data.startswith(b"\x89PNG")
    agent.close()
print(
    "DAENA_HOST_VISION_SMOKE_OK: runtime binding, one-shot screenshot, Ollama/OpenRouter/Gemini serialization"
)
