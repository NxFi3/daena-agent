from __future__ import annotations

from src.tools.builtin.command_exec.tool import CommandExec


RECOVERY = (
    "command is executed as argv without a shell, so heredocs, pipes and redirection "
    "are not interpreted. Write the code to a .py file with write_file, then run "
    "['python3', 'file.py']."
)


def test_python_heredoc_returns_actionable_error():
    result = CommandExec().execute(
        command=["python3", "- <<'PY'\\nprint(1)\\nPY"],
    )
    assert not result.success
    assert result.content["error"]["type"] == "shell_syntax_not_supported"
    assert result.content["error"]["message"] == RECOVERY


def test_interpreter_stdin_dash_returns_actionable_error():
    result = CommandExec().execute(command=["python3", "-"])
    assert not result.success
    assert result.content["error"]["type"] == "shell_syntax_not_supported"


def test_shell_operator_token_returns_actionable_error():
    result = CommandExec().execute(command=["bash", "script.sh", "&&", "echo", "x"])
    assert not result.success
    assert result.content["error"]["type"] == "shell_syntax_not_supported"


def test_failure_diagnostic_keeps_first_and_last_stderr_lines():
    diagnostic = CommandExec._failure_diagnostic(
        stdout="",
        stderr="usage: python [options]\\nTry python -h for more information.\\n",
    )
    assert "usage: python [options]" in diagnostic
    assert "Try python -h for more information." in diagnostic


def test_policy_inline_python_error_has_recovery_guidance():
    from src.security.Policy import SecurityPolicy
    from src.models.ToolCall import ToolCall

    decision = SecurityPolicy().evaluate(
        ToolCall(name="command_exec", args={"command": ["python3", "-c", "print(1)"]}),
        sandbox=type("Sandbox", (), {"resolve": lambda self, path: path})(),
    )
    assert not decision.allowed
    assert "write_file" in decision.reason
    assert "file.py" in decision.reason
