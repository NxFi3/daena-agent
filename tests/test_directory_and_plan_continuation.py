from src.tools.ToolManager import ToolManager
from src.tools.builtin.plan.tool import Plan


def invoke(manager, name, **arguments):
    calls = manager.dispatcher.dispatch(
        [{"function": {"name": name, "arguments": arguments}}]
    )
    return manager.execute(calls)["results"][0]


def test_directory_tool_creates_individual_nested_paths_and_is_idempotent(tmp_path):
    manager = ToolManager({})
    manager.set_workspace(str(tmp_path))
    try:
        paths = ["store/public/css", "store/public/js", "store/routes", "store/data"]
        result = invoke(manager, "make_directory", paths=paths)
        assert result.success
        assert all((tmp_path / p).is_dir() for p in paths)
        assert len(result.content["created"]) == 4
        second = invoke(manager, "make_directory", paths=paths)
        assert second.success and not second.content["created"]
    finally:
        manager.close()


def test_directory_tool_validates_all_paths_before_any_write(tmp_path):
    manager = ToolManager({})
    manager.set_workspace(str(tmp_path))
    try:
        result = invoke(manager, "make_directory", paths=["allowed", "../escape"])
        assert not result.success
        assert not (tmp_path / "allowed").exists()
        outside = tmp_path.parent / (tmp_path.name + "-outside")
        outside.mkdir()
        (tmp_path / "link").symlink_to(outside, target_is_directory=True)
        result = invoke(manager, "make_directory", paths=["allowed", "link/escape"])
        assert not result.success
        assert not (outside / "escape").exists()
    finally:
        manager.close()


def test_plan_can_be_read_replaced_and_archived(tmp_path):
    tool = Plan()
    tool.set_workspace(tmp_path)
    assert tool.execute(action="read").content["exists"] is False
    assert tool.execute(action="create", goal="Old task", steps=["Step one"]).success
    original = tool.execute(action="read").content["plan"]
    blocked = tool.execute(action="create", goal="New task", steps=["Step two"])
    assert not blocked.success and blocked.evidence["plan"] == original
    assert tool.execute(action="replace", goal="New task", steps=["Step two"]).success
    assert "New task" in tool.execute(action="read").content["plan"]
    archive = list((tmp_path / ".daena" / "plan-history").glob("*.md"))
    assert len(archive) == 1 and archive[0].read_text() == original


def test_invalid_replacement_keeps_existing_plan(tmp_path):
    tool = Plan()
    tool.set_workspace(tmp_path)
    tool.execute(action="create", goal="Keep", steps=["Work"])
    before = tool.execute(action="read").content["plan"]
    assert not tool.execute(action="replace", goal="", steps=[]).success
    assert tool.execute(action="read").content["plan"] == before


def test_plan_rejects_workspace_escape_through_runtime_symlink(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (tmp_path / ".daena").symlink_to(outside, target_is_directory=True)
    tool = Plan()
    tool.set_workspace(tmp_path)
    result = tool.execute(action="create", goal="Attempt", steps=["Work"])
    assert not result.success
    assert "escapes" in result.content["error"]["message"]
    assert not (outside / "plan.md").exists()


def test_plan_history_cannot_be_redirected_outside_runtime_directory(tmp_path):
    tool = Plan()
    tool.set_workspace(tmp_path)
    tool.execute(action="create", goal="Keep", steps=["Work"])
    outside = tmp_path.parent / (tmp_path.name + "-history")
    outside.mkdir()
    (tmp_path / ".daena" / "plan-history").symlink_to(outside, target_is_directory=True)
    result = tool.execute(action="replace", goal="New", steps=["New work"])
    assert not result.success
    assert not list(outside.iterdir())
    assert "Keep" in tool.execute(action="read").content["plan"]
