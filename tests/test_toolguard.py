from src.agent.toolguard import ToolLoopGuard
from src.models.ToolCall import ToolCall
from src.models.ToolResult import ToolResult


def call(name: str, args: dict) -> ToolCall:
    return ToolCall(name=name, id=f"{name}-1", valid=True, args=args)


def result(name: str, *, success: bool = True, content: dict | None = None) -> ToolResult:
    return ToolResult(
        success=success,
        name=name,
        content=content or {"success": success},
    )


def patch_call(path: str = "backend/routes/notes.js") -> ToolCall:
    return call(
        "apply_patch",
        {
            "patch": (
                "*** Begin Patch\n"
                f"*** Update File: {path}\n"
                "@@\n"
                "-old\n"
                "+new\n"
                "*** End Patch"
            )
        },
    )


def patch_result(path: str = "backend/routes/notes.js") -> ToolResult:
    return result(
        "apply_patch",
        content={
            "success": True,
            "files": [{"operation": "update", "path": path}],
        },
    )


def test_identical_successful_call_warns_then_blocks():
    guard = ToolLoopGuard()
    tool_call = call("search", {"query": "notes"})

    guard.after_call(
        tool_call,
        result("search", content={"success": True, "matches": []}),
        workspace_changed=False,
    )
    assert guard.before_call(tool_call).action == "allow"

    second = guard.after_call(
        tool_call,
        result("search", content={"success": True, "matches": []}),
        workspace_changed=False,
    )
    assert second.action == "warn"

    third = guard.after_call(
        tool_call,
        result("search", content={"success": True, "matches": []}),
        workspace_changed=False,
    )
    assert third.action == "warn"

    fourth = guard.after_call(
        tool_call,
        result("search", content={"success": True, "matches": []}),
        workspace_changed=False,
    )
    assert fourth.action == "warn"

    blocked = guard.before_call(tool_call)
    assert blocked.action == "block"
    assert blocked.code == "identical_call_loop"


def test_failed_verification_then_repeated_edits_get_bounded():
    guard = ToolLoopGuard()

    verify = call(
        "command_exec",
        {"command": ["npm", "test"], "workdir": "."},
    )
    guard.after_call(
        verify,
        result(
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
        ),
        workspace_changed=False,
    )

    tool_call = patch_call()
    for _ in range(3):
        assert guard.before_call(tool_call).action == "allow"
        guard.after_call(tool_call, patch_result(), workspace_changed=True)

    warning = guard.after_call(
        tool_call,
        patch_result(),
        workspace_changed=True,
    )
    assert warning.action == "warn"
    assert warning.code == "mutation_no_progress"

    blocked = guard.before_call(tool_call)
    assert blocked.action == "allow"

    for _ in range(3):
        guard.after_call(tool_call, patch_result(), workspace_changed=True)

    blocked = guard.before_call(tool_call)
    assert blocked.action == "block"
    assert blocked.code == "mutation_no_progress"


def test_successful_verification_resets_mutation_burst():
    guard = ToolLoopGuard()

    failed_verify = call("command_exec", {"command": ["npm", "test"]})
    guard.after_call(
        failed_verify,
        result(
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

    tool_call = patch_call("a.txt")
    for _ in range(4):
        guard.after_call(tool_call, patch_result("a.txt"), workspace_changed=True)

    successful_verify = call("command_exec", {"command": ["npm", "test"]})
    guard.after_call(
        successful_verify,
        result(
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

    assert guard.before_call(tool_call).action == "allow"


def test_process_poll_is_repeatable():
    guard = ToolLoopGuard()
    tool_call = call("process_poll", {"process_id": "proc-1"})

    for _ in range(4):
        assert guard.before_call(tool_call).action == "allow"
        guard.after_call(
            tool_call,
            result(
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


def test_repeating_two_call_cycle_is_detected():
    guard = ToolLoopGuard()

    read = call("read_file", {"file_path": "a.txt"})
    search = call("search", {"query": "hello"})

    for _ in range(8):
        guard.after_call(
            read,
            result("read_file", content={"success": True, "path": "a.txt", "content": "same"}),
            workspace_changed=False,
        )
        guard.after_call(
            search,
            result("search", content={"success": True, "matches": ["a.txt"]}),
            workspace_changed=False,
        )

    decision = guard.cycle_decision()
    assert decision.action == "block"
    assert decision.code == "repeating_cycle"
