"""Deterministic permission checking and safety tiers for M.A.D.D.Y. Agent Core.
(Modular Autonomous Digital Daemon for You)

Fulfills CLAUDE.md Engineering Rule 7:
"Every tool call must pass deterministic permission checks."
"""

from __future__ import annotations

import enum
import os
import re
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from core.errors import PermissionDeniedError

# Regex patterns matching shell metacharacters, subshells, process substitution, and redirection
_METAPATTERN_BLOCKS = re.compile(
    r"(\$\(|\`|>\(|<\(|;\s*|&&|\|\||\||>>|>|<|&|\n|\r|\btee\b)"
)


@dataclass
class PermissionCheckResult:
    """Structured result of a non-raising permission verification."""
    is_allowed: bool
    reason: str


class PermissionTier(str, enum.Enum):
    """Deterministic safety tiers for tool dispatch."""
    SAFE = "safe"                          # Read-only, telemetry, observation. Auto-allowed.
    SYSTEM_IMPACTING = "system_impacting"  # Mutating, file write/delete, shell execution. Requires authorization.


class PermissionGate:
    """Deterministic gate evaluating tool authorizations prior to invocation."""

    def __init__(
        self,
        auto_approve_tier2: bool = False,
        allowed_write_directories: Optional[List[str]] = None,
        allowed_commands: Optional[List[str]] = None,
        allowed_interactive_tools: Optional[Set[str]] = None,
        approval_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    ):
        self.auto_approve_tier2 = auto_approve_tier2
        self.approval_callback = approval_callback

        # Standard interactive desktop tools authorized for operator UI use
        default_interactive = {
            "app_control", "web_browser", "web_search", "media_control", "system_control", "timer_tool",
        }
        self.allowed_interactive_tools: Set[str] = (
            set(allowed_interactive_tools) if allowed_interactive_tools is not None else default_interactive
        )

        # Normalize allowed command allowlist with cross-platform defaults
        default_commands = [
            # Linux & POSIX diagnostic commands
            "uname", "date", "echo", "pwd", "whoami", "hostname", "uptime", "nvidia-smi",
            # Windows diagnostic commands
            "ver", "systeminfo", "cd", "dir",
        ]
        self.allowed_commands: Set[str] = set(allowed_commands or default_commands)

        # Normalize allowed write sandboxes across platforms (dynamic temp dir)
        default_sandboxes = [tempfile.gettempdir(), "/tmp", str(Path.home() / "sandbox")]
        all_dirs = (allowed_write_directories or []) + default_sandboxes
        self.allowed_write_directories: Set[Path] = set()
        for d in all_dirs:
            try:
                p = Path(os.path.expanduser(d)).resolve()
                self.allowed_write_directories.add(p)
            except Exception:
                pass

    def is_path_in_allowed_sandbox(self, target_path: str | Path) -> bool:
        """Verify if a filesystem path resides strictly within an authorized sandbox."""
        try:
            resolved = Path(os.path.expanduser(str(target_path))).resolve()
            for sandbox in self.allowed_write_directories:
                try:
                    resolved.relative_to(sandbox)
                    return True
                except ValueError:
                    continue
        except Exception:
            return False
        return False

    def is_safe_allowlisted_command(self, cmd: str) -> bool:
        """Verify command is a single safe invocation from allowed_commands without subshells or metacharacters.

        Blocks:
        - Subshell expansion: $(...), `...`
        - Process substitution: <(...), >(...)
        - Chaining & logic: &&, ||, ;, &
        - Redirections & pipelines: |, >, >>, <
        - Unquoted newlines and carriage returns
        - Obfuscation or invalid quoting
        """
        if not cmd or not isinstance(cmd, str):
            return False

        clean_cmd = cmd.strip()
        if not clean_cmd:
            return False

        # Fast rejection of shell metacharacters
        if _METAPATTERN_BLOCKS.search(clean_cmd):
            return False

        try:
            # Tokenize strictly matching POSIX / Windows shell rules
            tokens = shlex.split(clean_cmd, posix=(os.name != "nt"))
        except Exception:
            return False

        if not tokens:
            return False

        # Verify root executable is in allowlist and NOT a path with directory traversal or custom path
        raw_root = tokens[0]
        if "/" in raw_root or "\\" in raw_root:
            return False

        cmd_root = raw_root.lower()
        if cmd_root not in self.allowed_commands:
            return False

        # Additional safety checks for diagnostic commands with dangerous flags
        if cmd_root == "date" and any(t in ("-s", "--set") for t in tokens[1:]):
            return False
        if cmd_root == "hostname" and len(tokens) > 1 and not tokens[1].startswith("-"):
            return False
        if cmd_root == "nvidia-smi" and any(t in ("-f", "--filename") for t in tokens[1:]):
            return False

        # Verify no token contains raw expansion or backticks
        for token in tokens[1:]:
            if any(char in token for char in ("$", "`", "\n", "\r", ";", "&", "|", "<", ">")):
                return False

        return True

    def check_permission(
        self,
        tool_name: str,
        tier: PermissionTier,
        arguments: Dict[str, Any],
    ) -> None:
        """Deterministic permission preflight check.

        Raises:
            PermissionDeniedError: If the operation is not permitted under current policy.
        """
        # Tier 1 (Safe) is always auto-approved
        if tier == PermissionTier.SAFE:
            return

        # Tier 2 (System Impacting): Check if globally auto-approved
        if self.auto_approve_tier2:
            return

        # Check allowed interactive desktop tools (app launching, web navigation, media control)
        if tool_name in self.allowed_interactive_tools:
            # Critical actions like shutdown and reboot require operator approval
            if tool_name == "system_control":
                action = str(arguments.get("action", "")).lower()
                if action in ("shutdown_workstation", "reboot_workstation", "shutdown", "reboot"):
                    if self.approval_callback and self.approval_callback(tool_name, arguments):
                        return
                    raise PermissionDeniedError(
                        message=f"Critical system action '{action}' requires explicit operator authorization.",
                        tool_name=tool_name,
                        tier=tier.value,
                        arguments=arguments,
                    )
            return

        # Check path-based exemption for file write operations
        if tool_name in ("write_file", "filesystem.write", "create_file"):
            file_path = arguments.get("file_path") or arguments.get("path")
            if file_path and self.is_path_in_allowed_sandbox(file_path):
                return  # Sandboxed write permitted

        # Check command allowlist exemption for shell execution
        if tool_name in ("execute_command", "terminal", "bash", "execute_shell", "run_command"):
            cmd = (arguments.get("command") or "").strip()
            if self.is_safe_allowlisted_command(cmd):
                return  # Allowed read-only diagnostic command with zero subshells or chaining

        # Interactive operator confirmation callback
        if self.approval_callback is not None:
            approved = self.approval_callback(tool_name, arguments)
            if approved:
                return

        # Fail closed
        raise PermissionDeniedError(
            message=f"Deterministic permission denied for system-impacting tool '{tool_name}'. Explicit operator authorization required.",
            tool_name=tool_name,
            tier=tier.value,
            arguments=arguments,
        )

    def check_tool_execution(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tier: Optional[PermissionTier] = None,
    ) -> PermissionCheckResult:
        """Deterministic permission preflight returning a structured status instead of raising.

        Used by AutonomyPolicyGate and external verification layers to evaluate
        sandbox boundaries and authorization states cleanly.
        """
        if tier is None:
            tier = PermissionTier.SYSTEM_IMPACTING

        try:
            self.check_permission(tool_name, tier, arguments)
            return PermissionCheckResult(
                is_allowed=True,
                reason=f"Operation for tool '{tool_name}' authorized under deterministic permission policy.",
            )
        except PermissionDeniedError as pde:
            return PermissionCheckResult(
                is_allowed=False,
                reason=pde.message,
            )
        except Exception as exc:
            return PermissionCheckResult(
                is_allowed=False,
                reason=f"Permission check error: {str(exc)}",
            )
