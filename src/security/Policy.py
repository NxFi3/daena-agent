from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_BLOCKED_COMMANDS = {
    "sudo",
    "su",
    "doas",
    "rm",
    "rmdir",
    "del",
    "format",
    "mkfs",
    "fdisk",
    "parted",
    "mount",
    "umount",
    "shutdown",
    "reboot",
    "poweroff",
    "halt",
    "iptables",
    "nft",
    "useradd",
    "userdel",
    "passwd",
    "killall",
}

DEFAULT_ALLOWED_TOOLS = {
    "plan",
    "read_file",
    "context_search",
    "grep",
    "glob",
    "list_dir",
    "explore",
    "apply_patch",
    "write_file",
    "edit_file",
    "command_exec",
    "process_poll",
    "process_write",
    "process_stop",
    "web_search",
    "web_fetch",
}


@dataclass(frozen=True)
class SecurityDecision:
    allowed: bool
    reason: str
    rule: str = ""


@dataclass
class SecurityPolicy:
    """Fail-closed runtime policy for agent tool execution.

    This is a defense-in-depth policy layer, not an operating-system sandbox.
    It prevents accidental workspace escape and obviously destructive command
    execution while keeping normal software-engineering tools usable.
    """

    allowed_tools: set[str] = field(default_factory=lambda: set(DEFAULT_ALLOWED_TOOLS))
    blocked_commands: set[str] = field(
        default_factory=lambda: set(DEFAULT_BLOCKED_COMMANDS)
    )
    allowed_commands: set[str] | None = None
    workspace_only: bool = True
    allow_background: bool = False
    allow_network_tools: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "SecurityPolicy":
        config = config or {}
        allowed_tools = config.get("allowed_tools")
        blocked_commands = config.get("blocked_commands")
        allowed_commands = config.get("allowed_commands")

        return cls(
            allowed_tools={
                str(item).strip().lower()
                for item in (allowed_tools or DEFAULT_ALLOWED_TOOLS)
                if str(item).strip()
            },
            blocked_commands={
                str(item).strip().lower()
                for item in (blocked_commands or DEFAULT_BLOCKED_COMMANDS)
                if str(item).strip()
            },
            allowed_commands=(
                {
                    str(item).strip().lower()
                    for item in allowed_commands
                    if str(item).strip()
                }
                if allowed_commands
                else None
            ),
            workspace_only=bool(config.get("workspace_only", True)),
            allow_background=bool(config.get("allow_background", False)),
            allow_network_tools=bool(config.get("allow_network_tools", True)),
        )

    def evaluate(self, toolcall, sandbox) -> SecurityDecision:
        name = str(getattr(toolcall, "name", "")).strip().lower()
        if not name:
            return SecurityDecision(False, "Tool name is empty.", "tool_name")

        if name not in self.allowed_tools:
            return SecurityDecision(
                False,
                f"Tool '{name}' is not allowed by the security policy.",
                "tool_allowlist",
            )

        args = getattr(toolcall, "args", {}) or {}
        if not isinstance(args, dict):
            return SecurityDecision(False, "Tool arguments must be an object.", "args")

        if name == "read_file":
            return self._check_read_file(args, sandbox)

        if name == "apply_patch":
            return self._check_apply_patch(args, sandbox)

        if name == "write_file":
            return self._check_write_file(args, sandbox)

        if name == "edit_file":
            return self._check_edit_file(args, sandbox)

        if name == "command_exec":
            return self._check_command(args, sandbox)

        if name in {"web_search", "web_fetch"}:
            if not self.allow_network_tools:
                return SecurityDecision(
                    False,
                    f"Network tool '{name}' is disabled by policy.",
                    "network_disabled",
                )
            if args.get("save_to") is not None:
                return self._check_web_save(args, sandbox)

        return SecurityDecision(True, "Allowed by policy.", "default_allow")

    def _check_read_file(self, args: dict[str, Any], sandbox) -> SecurityDecision:
        path = args.get("file_path")
        if not isinstance(path, str) or not path.strip():
            return SecurityDecision(False, "file_path is required.", "path_required")

        if self.workspace_only:
            try:
                sandbox.resolve(path)
            except (PermissionError, ValueError) as exc:
                return SecurityDecision(False, str(exc), "workspace_boundary")

        return SecurityDecision(True, "Read target is inside the workspace.", "workspace_read")

    def _check_apply_patch(self, args: dict[str, Any], sandbox) -> SecurityDecision:
        patch = args.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            return SecurityDecision(False, "patch is required.", "patch_required")

        if self.workspace_only:
            try:
                sandbox.validate_patch_paths(patch)
            except (PermissionError, ValueError, TypeError) as exc:
                return SecurityDecision(False, str(exc), "workspace_boundary")

        return SecurityDecision(True, "Patch targets are inside the workspace.", "workspace_patch")

    def _check_write_file(self, args: dict[str, Any], sandbox) -> SecurityDecision:
        path = args.get("file_path")
        content = args.get("content")
        overwrite = args.get("overwrite", False)

        if not isinstance(path, str) or not path.strip():
            return SecurityDecision(False, "file_path is required.", "path_required")
        if not isinstance(content, str):
            return SecurityDecision(False, "content must be a string.", "content_type")
        if type(overwrite) is not bool:
            return SecurityDecision(False, "overwrite must be a boolean.", "overwrite_type")

        if self.workspace_only:
            try:
                sandbox.resolve(path)
            except (PermissionError, ValueError) as exc:
                return SecurityDecision(False, str(exc), "workspace_boundary")

        return SecurityDecision(True, "Write target is permitted by policy.", "workspace_write")

    def _check_edit_file(self, args: dict[str, Any], sandbox) -> SecurityDecision:
        path = args.get("file_path")
        old_string = args.get("old_string")
        new_string = args.get("new_string")
        replace_all = args.get("replace_all", False)
        if not isinstance(path, str) or not path.strip():
            return SecurityDecision(False, "file_path is required.", "path_required")
        if not isinstance(old_string, str) or not old_string:
            return SecurityDecision(False, "old_string must be a non-empty string.", "old_string_required")
        if not isinstance(new_string, str):
            return SecurityDecision(False, "new_string must be a string.", "new_string_type")
        if old_string == new_string:
            return SecurityDecision(False, "old_string and new_string are identical.", "no_change")
        if type(replace_all) is not bool:
            return SecurityDecision(False, "replace_all must be a boolean.", "replace_all_type")
        try:
            sandbox.resolve(path)
        except (PermissionError, ValueError) as exc:
            return SecurityDecision(False, str(exc), "workspace_boundary")
        return SecurityDecision(True, "Edit target is inside the workspace.", "workspace_edit")

    def _check_web_save(self, args: dict[str, Any], sandbox) -> SecurityDecision:
        path = args.get("save_to")
        save_format = args.get("save_format", "text")
        if not isinstance(path, str) or not path.strip():
            return SecurityDecision(False, "save_to must be a non-empty workspace-relative path.", "path_required")
        if Path(path).is_absolute():
            return SecurityDecision(False, "save_to must be workspace-relative.", "workspace_boundary")
        if save_format not in {"text", "json"}:
            return SecurityDecision(False, "save_format must be 'text' or 'json'.", "save_format")
        try:
            sandbox.resolve(path)
        except (PermissionError, ValueError) as exc:
            return SecurityDecision(False, str(exc), "workspace_boundary")
        return SecurityDecision(True, "Web output path is inside the workspace.", "workspace_write")

    def _check_command(self, args: dict[str, Any], sandbox) -> SecurityDecision:
        command = args.get("command")
        if not isinstance(command, list) or not command:
            return SecurityDecision(False, "command must be a non-empty argument list.", "command_required")

        if not all(isinstance(item, str) for item in command):
            return SecurityDecision(False, "command arguments must all be strings.", "command_args")

        executable = command[0].strip()
        executable_name = executable.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if executable_name.endswith(".exe"):
            executable_name = executable_name[:-4]

        if executable_name in self.blocked_commands:
            return SecurityDecision(
                False,
                f"Executable '{executable_name}' is blocked by the security policy.",
                "blocked_executable",
            )

        if self.allowed_commands is not None and executable_name not in self.allowed_commands:
            return SecurityDecision(
                False,
                f"Executable '{executable_name}' is not in the configured command allowlist.",
                "command_allowlist",
            )

        workdir = args.get("workdir")
        if workdir is not None and not isinstance(workdir, str):
            return SecurityDecision(False, "workdir must be a string.", "workdir_type")

        if self.workspace_only:
            try:
                sandbox.resolve(workdir or ".")
            except (PermissionError, ValueError) as exc:
                return SecurityDecision(False, str(exc), "workspace_boundary")

        # Avoid the most obvious shell/interpreter escape hatch while keeping
        # normal script execution available through files in the workspace.
        lowered = [item.strip().lower() for item in command[1:]]
        if executable_name in {"sh", "bash", "zsh", "dash"} and "-c" in lowered:
            return SecurityDecision(
                False,
                "command is executed as argv without a shell, so heredocs, pipes and redirection are not interpreted. Write the code to a .py file with write_file, then run ['python3', 'file.py'].",
                "inline_shell_blocked",
            )
        if executable_name in {"node"} and any(flag in lowered for flag in {"-e", "--eval"}):
            return SecurityDecision(
                False,
                "command is executed as argv without a shell, so heredocs, pipes and redirection are not interpreted. Write the code to a .py file with write_file, then run ['python3', 'file.py'].",
                "inline_eval_blocked",
            )
        if executable_name in {"python", "python3", "pypy", "pypy3"} and any(
            flag in lowered for flag in {"-c", "--command"}
        ):
            return SecurityDecision(
                False,
                "command is executed as argv without a shell, so heredocs, pipes and redirection are not interpreted. Write the code to a .py file with write_file, then run ['python3', 'file.py'].",
                "inline_eval_blocked",
            )

        if bool(args.get("background", False)) and not self.allow_background:
            return SecurityDecision(
                False,
                "Background processes require explicit approval.",
                "background_disabled",
            )

        return SecurityDecision(True, "Command is allowed inside the workspace.", "workspace_command")
