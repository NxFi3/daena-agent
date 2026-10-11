from __future__ import annotations

from src.tools.ToolManager import ToolManager


def _call_search(manager: ToolManager, query: str, top_k: int = 5):
    calls = manager.dispatcher.dispatch({
        "name": "tool_search",
        "arguments": {"query": query, "top_k": top_k},
    })
    return manager.execute(calls)["results"][0]


def test_model_receives_search_entrypoint_before_full_tool_schemas(tmp_path):
    manager = ToolManager({
        "guard_level": "strict",
        "security": {"workspace_only": True, "force_approve": True},
    })
    manager.set_workspace(str(tmp_path))
    try:
        names = [item["function"]["name"] for item in manager.get_model_tools()]
        assert names[0] == "tool_search"
        assert {
            "command_exec", "read_file", "write_file", "edit_file", "apply_patch"
        }.issubset(names)
        assert "web_search" not in names
        assert "context_search" not in names
        assert len(names) < len(manager.get_tools())
    finally:
        manager.close()


def test_search_activates_only_relevant_tool_schemas_for_next_turn(tmp_path):
    manager = ToolManager({
        "guard_level": "strict",
        "security": {"workspace_only": True, "force_approve": True},
    })
    manager.set_workspace(str(tmp_path))
    try:
        result = _call_search(
            manager,
            "write a Python file, run tests, read output and verify the result",
            top_k=5,
        )
        assert result.success is True
        matches = [item["name"] for item in result.content["matches"]]
        assert "write_file" in matches
        assert "command_exec" in matches
        assert "read_file" in matches

        exposed = {item["function"]["name"] for item in manager.get_model_tools()}
        assert "tool_search" in exposed
        assert set(matches).issubset(exposed)
        assert len(exposed) <= 1 + len(manager.CORE_TOOL_NAMES) + 5
        assert "web_fetch" not in exposed
        assert {"command_exec", "read_file", "write_file"}.issubset(exposed)
    finally:
        manager.close()


def test_search_never_activates_disabled_tools(tmp_path):
    manager = ToolManager({
        "guard_level": "strict",
        "tools": {"disabled": ["command_exec"]},
        "security": {"workspace_only": True, "force_approve": True},
    })
    manager.set_workspace(str(tmp_path))
    try:
        result = _call_search(manager, "run a local command and execute Python tests", top_k=8)
        names = {item["name"] for item in result.content["matches"]}
        exposed = {item["function"]["name"] for item in manager.get_model_tools()}
        assert "command_exec" not in names
        assert "command_exec" not in exposed
    finally:
        manager.close()


def test_no_match_does_not_discard_currently_selected_tools(tmp_path):
    manager = ToolManager({
        "guard_level": "strict",
        "security": {"workspace_only": True, "force_approve": True},
    })
    manager.set_workspace(str(tmp_path))
    try:
        first = _call_search(manager, "inspect Python files and find source code", top_k=3)
        assert first.success is True
        previous = {item["function"]["name"] for item in manager.get_model_tools()}

        second = _call_search(manager, "banana zorb quux capability", top_k=3)
        assert second.success is True
        assert second.content["matches"] == []
        current = {item["function"]["name"] for item in manager.get_model_tools()}
        assert current == previous
    finally:
        manager.close()


def test_unexposed_tool_calls_are_blocked_until_the_model_searches_for_them(tmp_path):
    from src.agent.agentloop import Loop
    from src.models.ToolCall import ToolCall

    class TinyLLM:
        model = type("Model", (), {"defaultConfig": {"num_ctx": 4096}})()

        def generate(self, messages, tools=None, **kwargs):
            raise AssertionError("generation is not needed for this guard test")

    loop = Loop({
        "llm": {"provider_config": {"generation_config": {"num_ctx": 4096}}},
        "context": {"safe_margin": 0, "compaction_enabled": False},
        "security": {"workspace_only": True, "force_approve": True},
        "experience": {"enabled": False},
    }, TinyLLM())
    try:
        loop.set_workspace(str(tmp_path))
        loop._turn_tool_names = {"tool_search"}
        call = ToolCall(
            name="web_fetch",
            valid=True,
            args={"url": "https://example.invalid/"},
        )
        allowed, blocked = loop._classify_calls([call])
        assert allowed == []
        assert blocked[0].success is False
        assert blocked[0].content["error"]["type"] == "tool_not_exposed"
        assert "tool_search" in blocked[0].metadata["recovery_hint"]
    finally:
        loop.close()


def test_natural_language_search_handles_common_synonyms(tmp_path):
    manager = ToolManager({
        "guard_level": "strict",
        "security": {"workspace_only": True, "force_approve": True},
    })
    manager.set_workspace(str(tmp_path))
    try:
        create = manager._search_and_activate_tools(
            "create result.txt and verify it by reading the file back", top_k=5
        )
        create_names = {item["name"] for item in create["matches"]}
        assert {"write_file", "read_file"}.issubset(create_names)

        coding = manager._search_and_activate_tools(
            "run Python tests and fix errors", top_k=5
        )
        coding_names = {item["name"] for item in coding["matches"]}
        assert "command_exec" in coding_names
        assert coding_names.intersection({"edit_file", "apply_patch", "write_file"})

        web = manager._search_and_activate_tools(
            "search current public web pages and retrieve a known URL", top_k=5
        )
        web_names = {item["name"] for item in web["matches"]}
        assert {"web_search", "web_fetch"}.issubset(web_names)
    finally:
        manager.close()


def test_compile_cpp_query_finds_a_command_execution_capability(tmp_path):
    manager = ToolManager({
        "guard_level": "strict",
        "security": {"workspace_only": True, "force_approve": True},
    })
    manager.set_workspace(str(tmp_path))
    try:
        outcome = manager._search_and_activate_tools("compile C++ program", top_k=1)
        assert outcome["matches"]
        assert outcome["matches"][0]["name"] == "command_exec"
        exposed = {item["function"]["name"] for item in manager.get_model_tools()}
        assert "command_exec" in exposed
    finally:
        manager.close()


def test_computer_use_query_retrieves_computer_tools_without_changing_core_visibility(tmp_path):
    manager = ToolManager({
        "guard_level": "strict",
        "security": {"workspace_only": True, "force_approve": True},
    })
    manager.set_workspace(str(tmp_path))
    try:
        first = _call_search(manager, "read a known file and inspect its contents", top_k=3)
        assert first.content["matches"]
        previous = {item["function"]["name"] for item in manager.get_model_tools()}

        result = manager._search_and_activate_tools(
            "click buttons and inspect screenshots to control the computer", top_k=5
        )
        matched = {item["name"] for item in result["matches"]}
        # Desktop automation is exposed to the model as one Hermes-style tool,
        # not the former separate observe/act helpers.
        assert "computer_use" in matched
        assert not {"computer_observe", "computer_act"}.intersection(matched)
        current = {item["function"]["name"] for item in manager.get_model_tools()}
        # Computer Use is now an always-available core capability. Tool Search
        # may replace optional schemas as relevance changes, but must preserve core.
        assert manager.CORE_TOOL_NAMES.issubset(current)
        assert manager.CORE_TOOL_NAMES.issubset(previous)
    finally:
        manager.close()
