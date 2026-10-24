import json
import os

import pytest

from src.agent.agent import Agent
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult

pytestmark = pytest.mark.browser


def test_actual_agent_loop_keeps_login_across_user_turns_and_bounds_history(
    tmp_path, website
):
    if os.environ.get("DAENA_TEST_BROWSER") != "1":
        pytest.skip("Set DAENA_TEST_BROWSER=1")
    pytest.importorskip("playwright")

    class Model:
        model = None

        def __init__(self):
            self.index = 0
            self.observations = []

        def generate(self, messages, **kwargs):
            self.index += 1
            index = self.index
            pinned = [
                m
                for m in messages
                if isinstance(m.get("content"), str)
                and m["content"].startswith("Current browser observation")
            ]
            assert not any(m.get("images") for m in messages)
            if index > 2:
                assert len(pinned) == 1
                state = json.loads(pinned[0]["content"].split("\n", 1)[1])
                self.observations.append(state)
            for m in messages:
                if m.get("role") == "tool":
                    receipt = json.loads(m["content"])
                    assert "elements" not in receipt.get("evidence", {}).get(
                        "observation", {}
                    )
            actions = {
                1: (
                    "tool_search",
                    {"query": "browser website forms login", "top_k": 1},
                ),
                2: ("browser_use", {"action": "open", "url": website}),
                3: (
                    "browser_use",
                    {
                        "action": "fill",
                        "target": {"label": "Phone"},
                        "text": "09120000000",
                    },
                ),
                4: (
                    "browser_use",
                    {
                        "action": "click",
                        "target": {"role": "button", "name": "Send code"},
                    },
                ),
                6: (
                    "browser_use",
                    {
                        "action": "fill",
                        "target": {"label": "One-time code"},
                        "text": "123456",
                    },
                ),
                7: (
                    "browser_use",
                    {
                        "action": "click",
                        "target": {"role": "button", "name": "Sign in"},
                    },
                ),
            }
            if index in {5, 8}:
                assert ("Code sent" if index == 5 else "Signed in") in state["text"]
                return LLMResult(
                    "Please provide the code."
                    if index == 5
                    else "Signed in, verified on the page.",
                    {},
                    [],
                    None,
                    1,
                )
            name, args = actions[index]
            call = {
                "id": f"web-{index}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            return LLMResult(
                "",
                {"role": "assistant", "content": "", "tool_calls": [call]},
                [call],
                None,
                1,
            )

    model = Model()
    config = {
        "llm": {
            "provider": "ollama",
            "provider_config": {"generation_config": {"num_ctx": 32768}},
        },
        "workspace": {"directory": str(tmp_path)},
        "memory": {"database": str(tmp_path / "memory.db")},
        "context": {"compaction_enabled": False},
        "security": {"allowed_tools": ["tool_search", "browser_use"]},
        "browser": {
            "headless": True,
            "executable_path": os.environ.get("DAENA_TEST_CHROMIUM", ""),
        },
        "max_agent_iterations": 8,
    }
    with Agent(config, llm=model) as agent:
        agent.act(
            ContextEvent(
                role=ContextRole.USER,
                type=ContextType.MESSAGE,
                content="Sign in on the test website with the provided number.",
            )
        )
        assert agent.last_run_metrics["completed"]
        browser = agent.loop.tool.get_tool("browser_use")
        session = browser._session
        assert session.context and browser.model_exposure == "always"
        agent.act(
            ContextEvent(
                role=ContextRole.USER,
                type=ContextType.MESSAGE,
                content="The code is 123456.",
            )
        )
        assert agent.last_run_metrics["completed"]
        assert browser._session is session
        assert agent.last_run_metrics["context_image_tokens"] == 0
        assert model.index == 8
    assert browser._thread is None
