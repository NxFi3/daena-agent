from src.agent.toolguard import ToolLoopGuard
from src.models.ToolCall import ToolCall
from src.models.ToolResult import ToolResult


def _call(name: str, args: dict) -> ToolCall:
    return ToolCall(name=name, id=f"{name}-1", valid=True, args=args)


def _result(name: str, *, success: bool = True, content: dict | None = None) -> ToolResult:
    return ToolResult(
        success=success,
        name=name,
        content=content or {"success": success},
    )


def test_identical_successful_call_warns_then_blocks():
    guard = ToolLoopGuard()
    call = _call("apply_patch", {"patch": "same"})

    assert guard.before_call(call).action == "allow"
    first = guard.after_call(call, _result("apply_patch"), workspace_changed=False)
    assert first.action == "allow"

    second = guard.after_call(call, _result("apply_patch"), workspace_changed=False)
    assert second.action == "warn"

    third = guard.after_call(call, _result("apply_patch"), workspace_changed=False)
    assert third.action == "warn"

    decision = guard.before_call(call)
    assert decision.action == "block"
    assert decision.code == "identical_call_loop"


def test_failed_verification_then_repeated_edits_get_bounded():
    guard = ToolLoopGuard()

    verify = _call(
        "command_exec",
        {"command": ["npm", "test"], "workdir": "."},
    )
    failed = _result(
        "command_exec",
        success=False,
        content={
            "success": False,
            "command": ["npm", "test"],
            "workdir": ".",
            "status": "exited",
            "exit_code": 1,
            "stdout": "FAIL one test",
            "stderr": "",
        },
    )
    guard.after_call(verify, failed, workspace_changed=False)

    patch = _call(
        "apply_patch",
        {"patch": "*** Begin Patch\n*** Update File: backend/routes/notes.js\n@@\n-old\n+new\n*** End Patch"},
    )

    for _ in range(3):
        assert guard.before_call(patch).action == "allow"
        guard.after_call(
            patch,
            _result(
                "apply_patch",
                content={
                    "success": True,
                    "files": [{"operation": "update", "path": "backend/routes/notes.js"}],
                },
            ),
            workspace_changed=True,
        )

    warning = guard.after_call(
        patch,
        _result(
            "apply_patch",
            content={
                "success": True,
                "files": [{"operation": "update", "path": "backend/routes/notes.js"}],
            },
        ),
        workspace_changed=True,
    )
    assert warning.action == "warn"
    assert warning.code == "mutation_no_progress"

    for _ in range(2):
        guard.after_call(
            patch,
            _result(
                "apply_patch",
                content={
                    "success": True,
                    "files": [{"operation": "update", "path": "backend/routes/notes.js"}],
                },
            ),
            workspace_changed=True,
        )

    blocked = guard.before_call(patch)
    assert blocked.action == "block"
    assert blocked.code == "mutation_no_progress"


def test_successful_verification_resets_mutation_burst():
    guard = ToolLoopGuard()

    failed_verify = _call("command_exec", {"command": ["npm", "test"]})
    guard.after_call(
        failed_verify,
        _result(
            "command_exec",
            success=False,
            content={
                "success": False,
                "command": ["npm", "test"],
                "status": "exited",
                "exit_code": 1,
            },
        ),
        workspace_changed=False,
    )

    patch = _call(
        "apply_patch",
        {"patch": "*** Begin Patch\n*** Add File: a.txt\n+hello\n*** End Patch"},
    )
    for _ in range(4):
        guard.after_call(
            patch,
            _result(
                "apply_patch",
                content={
                    "success": True,
                    "files": [{"operation": "add", "path": "a.txt"}],
                },
            ),
            workspace_changed=True,
        )

    successful_verify = _call("command_exec", {"command": ["npm", "test"]})
    guard.after_call(
        successful_verify,
        _result(
            "command_exec",
            success=True,
            content={
                "success": True,
                "command": ["npm", "test"],
                "status": "exited",
                "exit_code": 0,
            },
        ),
        workspace_changed=False,
    )

    assert guard.before_call(patch).action == "allow"


def test_process_poll_is_repeatable():
    guard = ToolLoopGuard()
    call = _call("process_poll", {"process_id": "proc-1"})

    for _ in range(6):
        decision = guard.before_call(call)
        assert decision.action == "allow"
        guard.after_call(
            call,
            _result(
                "process_poll",
                content={
                    "success": True,
                    "process_id": "proc-1",
                    "status": "running",
                    "stdout": "",
                    "stderr": "",
                },
            ),
            workspace_changed=False,
        )
