from pathlib import Path

import pytest

from src.agent.agent import Agent
from src.models.ContextEvent import ContextEvent, ContextRole, ContextType
from src.models.LLMResult import LLMResult
from src.models.ToolCall import ToolCall
from src.skills import SkillRegistry
from src.tools.ToolManager import ToolManager


def write_skill(
    root, name="sample", body="Unique instructions", description="A useful skill"
):
    path = root / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}\n")
    return path


def registry(tmp_path):
    result = SkillRegistry({"skills": {"include_builtin": False}})
    result.set_workspace(str(tmp_path))
    return result


def test_progressive_loading_and_resources(tmp_path):
    path = write_skill(tmp_path / ".daena/skills")
    (path.parent / "reference.md").write_text("Reference content")
    skills = registry(tmp_path)
    assert [item["name"] for item in skills.list()] == ["sample"]
    assert "Unique instructions" not in skills.context()
    result = skills.read("sample")
    assert result["base_directory"] == str(path.parent)
    assert "Unique instructions" in skills.context()
    assert skills.read("sample", "reference.md")["content"] == "Reference content"


@pytest.mark.parametrize("resource", ["../secret", "/etc/passwd", ""])
def test_rejects_resource_escape(tmp_path, resource):
    write_skill(tmp_path / ".daena/skills")
    with pytest.raises(ValueError):
        registry(tmp_path).read("sample", resource)


def test_rejects_symlinks_and_malformed_metadata(tmp_path):
    path = write_skill(tmp_path / ".daena/skills")
    secret = tmp_path / "secret.txt"
    secret.write_text("private")
    (path.parent / "linked.md").symlink_to(secret)
    bad = write_skill(tmp_path / ".daena/skills", "invalid", description="[not, text]")
    skills = registry(tmp_path)
    assert "invalid" not in skills.skills
    assert any(str(bad) in warning for warning in skills.warnings)
    with pytest.raises(ValueError):
        skills.read("sample", "linked.md")
    path.unlink()
    path.symlink_to(secret)
    skills.refresh()
    assert skills.skills == {}


def test_skill_size_and_active_budget(tmp_path):
    root = tmp_path / ".daena/skills"
    for index in range(5):
        write_skill(root, f"skill-{index}")
    write_skill(root, "oversize", "x" * 65537)
    skills = registry(tmp_path)
    assert "oversize" not in skills.skills
    for index in range(4):
        skills.read(f"skill-{index}")
    with pytest.raises(ValueError, match="budget"):
        skills.read("skill-4")


def test_override_and_workspace_reset(tmp_path):
    write_skill(tmp_path / ".daena/skills", body="local")
    extra = tmp_path / "extra"
    write_skill(extra, body="configured")
    skills = SkillRegistry(
        {"skills": {"include_builtin": False, "directories": ["extra"]}}
    )
    skills.set_workspace(str(tmp_path))
    assert skills.read("sample")["content"] == "configured"
    assert "overridden" in skills.warnings[0]
    skills.set_workspace(str(tmp_path / "other"))
    assert not skills.active and not skills.skills


def test_disabled_skills_cannot_be_called_directly(tmp_path):
    manager = ToolManager({"skills": {"enabled": False}})
    try:
        manager.set_workspace(str(tmp_path))
        names = {item["function"]["name"] for item in manager.get_tools()}
        assert "skill_read" not in names
        result = manager.execute([ToolCall(name="skill_list", args={}, valid=True)])[
            "results"
        ][0]
        assert not result.success
        assert result.metadata["security_rule"] == "feature_disabled"
    finally:
        manager.close()


def test_builtin_skills_are_valid(tmp_path):
    skills = SkillRegistry()
    skills.set_workspace(str(tmp_path))
    assert len(skills.list()) == 8
    assert not skills.warnings


def test_loaded_skill_reaches_model_and_resets_between_tasks(tmp_path):
    write_skill(tmp_path / ".daena/skills")

    class ScriptedModel:
        model = None
        calls = 0

        def generate(self, messages, tools=None, **kwargs):
            self.calls += 1
            system = messages[0]["content"]
            if self.calls == 1:
                assert "Unique instructions" not in system
                calls = [
                    {
                        "id": "skill-1",
                        "function": {
                            "name": "skill_read",
                            "arguments": {"name": "sample"},
                        },
                    }
                ]
                return LLMResult("", {}, calls, None, 10)
            if self.calls == 2:
                assert "Unique instructions" in system
            else:
                assert "Unique instructions" not in system
            return LLMResult(
                "done", {"role": "assistant", "content": "done"}, [], None, 10
            )

    config = {
        "workspace": {"directory": str(tmp_path)},
        "memory": {"database": str(tmp_path / "memory.db")},
        "max_agent_iterations": 5,
        "context": {"compaction_enabled": False},
    }
    with Agent(config, llm=ScriptedModel()) as agent:
        result = agent.act(
            ContextEvent(
                role=ContextRole.USER,
                type=ContextType.MESSAGE,
                content="Use sample skill",
            )
        )
        assert result.response == "done"
        assert agent.last_run_metrics["completed"]
        agent.act(
            ContextEvent(
                role=ContextRole.USER,
                type=ContextType.MESSAGE,
                content="New unrelated task",
            )
        )
