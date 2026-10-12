from types import SimpleNamespace

from src.agent.agentloop import Loop
from src.models.ToolResult import ToolResult


def test_unverified_computer_use_dispatch_is_treated_as_failure():
    call = SimpleNamespace(
        name="computer_use",
        args={"action": "click", "target": "Rectangle"},
    )
    result = ToolResult(
        success=True,
        name="computer_use",
        content={
            "message": (
                "WARNING: Rectangle activation was not verified from a real "
                "checked-state transition. The action was dispatched. This does "
                "not prove the intended state changed."
            )
        },
    )

    normalized = Loop._normalize_unverified_gui_result(call, result)

    assert normalized is not result
    assert normalized.success is False
    assert normalized.content["error"]["type"] == "gui_action_unverified"
    assert normalized.metadata["recovery_required"] is True


def test_capture_result_does_not_count_as_a_state_changing_gui_action():
    call = SimpleNamespace(name="computer_use", args={"action": "capture", "mode": "som"})
    result = ToolResult(
        success=True,
        name="computer_use",
        content={"message": "Captured screenshot."},
    )

    assert Loop._gui_action_key(call) is None
    assert Loop._normalize_unverified_gui_result(call, result) is result


def test_failed_gui_action_stays_blocked_after_workspace_revision_changes():
    call = SimpleNamespace(
        name="computer_use",
        args={"action": "click", "coordinate": [230, 91]},
    )
    key = Loop._gui_action_key(call)

    # Use a lightweight instance because the recovery gate itself does not
    # need a configured LLM, database, or GUI backend for this check.
    loop = object.__new__(Loop)
    loop._failed_gui_call_keys = {key}
    loop._active_process_ids = set()
    loop.workspace_revision = 42
    loop._failed_call_keys = {}
    loop._semantic_failure_counts = {}
    loop._workspace_root = None
    loop.tool = SimpleNamespace(get_tool=lambda _name: None)
    loop._predict_failure_signature = lambda _call: None

    blocked = loop._runtime_recovery_gate(call)

    assert blocked is not None
    assert blocked.success is False
    assert blocked.content["error"]["type"] == "gui_action_repeat"


def test_different_gui_target_is_not_blocked_by_previous_failure():
    failed_call = SimpleNamespace(
        name="computer_use",
        args={"action": "click", "target": "Rectangle"},
    )
    new_call = SimpleNamespace(
        name="computer_use",
        args={"action": "click", "target": "Line"},
    )

    loop = object.__new__(Loop)
    loop._failed_gui_call_keys = {Loop._gui_action_key(failed_call)}
    loop._active_process_ids = set()
    loop.workspace_revision = 0
    loop._failed_call_keys = {}
    loop._semantic_failure_counts = {}
    loop._workspace_root = None
    loop.tool = SimpleNamespace(get_tool=lambda _name: None)
    loop._predict_failure_signature = lambda _call: None

    assert Loop._runtime_recovery_gate(loop, new_call) is None
