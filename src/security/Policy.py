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
    "apply_patch",
    "command_exec",
    "process_poll",
    "process_write",
    "process_stop",
    "web_search",
    "web_fetch",
    "skill_list",
    "skill_read",
    "subagent",
    "task_finish",
    "make_directory",
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
        config = {} if config is None else config
        if not isinstance(config, dict):
            raise ValueError("security must be an object")
        for key in ("workspace_only", "allow_background", "allow_network_tools", "force_approve"):
            if key in config and type(config[key]) is not bool:
                raise ValueError(f"security.{key} must be a boolean")
        for key in ("allowed_tools", "blocked_commands", "allowed_commands"):
            value = config.get(key)
            if value is not None and (
                not isinstance(value, (list, set, tuple))
                or not all(isinstance(item, str) and item.strip() for item in value)
            ):
                raise ValueError(f"security.{key} must be a collection of non-empty strings")
        allowed_tools = config.get("allowed_tools")
        blocked_commands = config.get("blocked_commands")
        allowed_commands = config.get("allowed_commands")

        return cls(
            allowed_tools={
                str(item).strip().lower()
                for item in (DEFAULT_ALLOWED_TOOLS if allowed_tools is None else allowed_tools)
                if str(item).strip()
            },
            blocked_commands={
                str(item).strip().lower()
                for item in (DEFAULT_BLOCKED_COMMANDS if blocked_commands is None else blocked_commands)
                if str(item).strip()
            },
            allowed_commands=(
                {
                    str(item).strip().lower()
                    for item in allowed_commands
                    if str(item).strip()
                }
                if allowed_commands is not None
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

        if name == "make_directory":
            paths = args.get("paths")
            if not isinstance(paths, list) or not paths or not all(isinstance(p, str) and p.strip() for p in paths):
                return SecurityDecision(False, "paths must be a nonempty list of paths.", "path_required")
            if self.workspace_only:
                try:
                    for path in paths:
                        sandbox.resolve(path)
                except (PermissionError, ValueError) as exc:
                    return SecurityDecision(False, str(exc), "workspace_boundary")
            return SecurityDecision(True, "Directory paths allowed.", "workspace_directory")

        if name == "read_file":
            return self._check_read_file(args, sandbox)

        if name == "apply_patch":
            return self._check_apply_patch(args, sandbox)

        if name == "command_exec":
            return self._check_command(args, sandbox)

        if name in {"web_search", "web_fetch"}:
            if not self.allow_network_tools:
                return SecurityDecision(
                    False,
                    f"Network tool '{name}' is disabled by policy.",
                    "network_disabled",
                )

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
                "Inline shell execution with -c is blocked; use argv-style commands.",
                "inline_shell_blocked",
            )
        if executable_name in {"node"} and any(flag in lowered for flag in {"-e", "--eval"}):
            return SecurityDecision(
                False,
                "Inline Node evaluation is blocked.",
                "inline_eval_blocked",
            )
        if executable_name in {"python", "python3", "pypy", "pypy3"} and any(
            flag in lowered for flag in {"-c", "--command"}
        ):
            return SecurityDecision(
                False,
                "Inline Python evaluation is blocked; execute a workspace file instead.",
                "inline_eval_blocked",
            )

        if bool(args.get("background", False)) and not self.allow_background:
            return SecurityDecision(
                False,
                "Background processes require explicit approval.",
                "background_disabled",
            )

        return SecurityDecision(True, "Command is allowed inside the workspace.", "workspace_command")
