from pathlib import Path

from src.models.ToolCall import ToolCall
from src.security.Policy import SecurityPolicy
from src.security.Sandbox import WorkspaceSandbox
from src.security.securityService import SecurityService


def make_service(tmp_path):
    service = SecurityService({})
    service.set_workspace(str(tmp_path))
    return service


def test_read_file_cannot_escape_workspace(tmp_path):
    service = make_service(tmp_path)
    call = ToolCall(
        name="read_file",
        id="c1",
        valid=True,
        args={"file_path": "../secret.txt"},
    )

    checked = service.check(call)

    assert checked.approved is False
    assert checked.security_rule == "workspace_boundary"


def test_patch_cannot_escape_workspace(tmp_path):
    service = make_service(tmp_path)
    call = ToolCall(
        name="apply_patch",
        id="c2",
        valid=True,
        args={
            "patch": "*** Begin Patch\n"
            "*** Add File: ../outside.txt\n"
            "+nope\n"
            "*** End Patch"
        },
    )

    checked = service.check(call)

    assert checked.approved is False
    assert checked.security_rule == "workspace_boundary"


def test_safe_command_is_approved(tmp_path):
    service = make_service(tmp_path)
    call = ToolCall(
        name="command_exec",
        id="c3",
        valid=True,
        args={
            "command": ["python", "-m", "pytest", "-q"],
            "workdir": str(tmp_path),
        },
    )

    checked = service.check(call)

    assert checked.approved is True
    assert checked.security_rule == "workspace_command"


def test_dangerous_executable_is_blocked(tmp_path):
    service = make_service(tmp_path)
    call = ToolCall(
        name="command_exec",
        id="c4",
        valid=True,
        args={"command": ["rm", "-rf", str(tmp_path)]},
    )

    checked = service.check(call)

    assert checked.approved is False
    assert checked.security_rule == "blocked_executable"


def test_background_execution_is_blocked_by_default(tmp_path):
    service = make_service(tmp_path)
    call = ToolCall(
        name="command_exec",
        id="c5",
        valid=True,
        args={
            "command": ["python", "-m", "http.server", "8000"],
            "workdir": str(tmp_path),
            "background": True,
        },
    )

    checked = service.check(call)

    assert checked.approved is False
    assert checked.security_rule == "background_disabled"


def test_sandbox_resolves_relative_paths(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    assert sandbox.resolve("src/main.py") == (
        Path(tmp_path).resolve() / "src/main.py"
    )
    assert sandbox.contains(Path(tmp_path).resolve())


def test_policy_blocks_unknown_tools():
    policy = SecurityPolicy.from_config({})
    sandbox = WorkspaceSandbox(Path.cwd())
    call = ToolCall(
        name="unknown_tool",
        id="c6",
        valid=True,
        args={},
    )

    decision = policy.evaluate(call, sandbox)

    assert decision.allowed is False
    assert decision.rule == "tool_allowlist"


def test_inline_python_evaluation_is_blocked(tmp_path):
    service = make_service(tmp_path)
    call = ToolCall(
        name="command_exec",
        id="c7",
        valid=True,
        args={
            "command": ["python", "-c", "print('outside')"],
            "workdir": str(tmp_path),
        },
    )

    checked = service.check(call)

    assert checked.approved is False
    assert checked.security_rule == "inline_eval_blocked"



def test_background_command_can_be_approved_for_session(tmp_path):
    service = make_service(tmp_path)
    command = ["python", "-m", "http.server", "8000"]

    service.approve_background_command(
        command,
        workdir=str(tmp_path),
    )

    call = ToolCall(
        name="command_exec",
        id="c8",
        valid=True,
        args={
            "command": command,
            "workdir": str(tmp_path),
            "background": True,
        },
    )

    checked = service.check(call)

    assert checked.approved is True
    assert checked.security_rule == "background_session_approval"


def test_background_approval_is_exact(tmp_path):
    service = make_service(tmp_path)
    service.approve_background_command(
        ["python", "-m", "http.server", "8000"],
        workdir=str(tmp_path),
    )

    different_command = ToolCall(
        name="command_exec",
        id="c9",
        valid=True,
        args={
            "command": ["python", "-m", "http.server", "9000"],
            "workdir": str(tmp_path),
            "background": True,
        },
    )

    checked = service.check(different_command)

    assert checked.approved is False
    assert checked.security_rule == "background_disabled"


def test_background_approval_does_not_bypass_workspace_boundary(tmp_path):
    service = make_service(tmp_path)
    service.approve_background_command(
        ["python", "-m", "http.server", "8000"],
        workdir=str(tmp_path),
    )

    escaping = ToolCall(
        name="command_exec",
        id="c10",
        valid=True,
        args={
            "command": ["python", "-m", "http.server", "8000"],
            "workdir": str(tmp_path / ".."),
            "background": True,
        },
    )

    checked = service.check(escaping)

    assert checked.approved is False
    assert checked.security_rule == "workspace_boundary"


def test_background_approval_does_not_bypass_blocked_executable(tmp_path):
    service = make_service(tmp_path)
    command = ["rm", "-rf", str(tmp_path)]

    service.approve_background_command(
        command,
        workdir=str(tmp_path),
    )

    call = ToolCall(
        name="command_exec",
        id="c11",
        valid=True,
        args={
            "command": command,
            "workdir": str(tmp_path),
            "background": True,
        },
    )

    checked = service.check(call)

    assert checked.approved is False
    assert checked.security_rule == "blocked_executable"


def test_clear_background_approvals(tmp_path):
    service = make_service(tmp_path)
    command = ["python", "-m", "http.server", "8000"]

    service.approve_background_command(
        command,
        workdir=str(tmp_path),
    )
    service.clear_background_approvals()

    call = ToolCall(
        name="command_exec",
        id="c12",
        valid=True,
        args={
            "command": command,
            "workdir": str(tmp_path),
            "background": True,
        },
    )

    checked = service.check(call)

    assert checked.approved is False
    assert checked.security_rule == "background_disabled"

def test_workspace_search_cannot_escape_workspace(tmp_path):
    service = make_service(tmp_path)
    call = ToolCall(
        name="search",
        id="c13",
        valid=True,
        args={"query": "secret", "path": "../"},
    )

    checked = service.check(call)

    assert checked.approved is False
    assert checked.security_rule == "workspace_boundary"
