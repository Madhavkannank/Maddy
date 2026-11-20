"""
Command and Process Sandboxing Engine for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Analyzes shell commands (PowerShell, cmd.exe, bash, sh) for dangerous patterns,
command composition, privilege escalation, destructive filesystem effects,
obfuscation, and data exfiltration.
Also provides environment variable sanitization to prevent leaking secrets to child processes.
"""
import enum
import os
import re
import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from tools.base import RiskLevel


class CommandRiskCategory(enum.Enum):
    SAFE = "safe"
    DESTRUCTIVE = "destructive"
    SYSTEM_INTERRUPTION = "system_interruption"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    REMOTE_EXECUTION = "remote_execution"
    OBFUSCATED_EXECUTION = "obfuscated_execution"
    NETWORK_EXFILTRATION = "network_exfiltration"


class CommandDecision(enum.Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"
    RESTRICTED = "restricted"


@dataclass
class CommandAssessment:
    raw_command: str
    shell_type: str
    categories: List[CommandRiskCategory]
    detected_patterns: List[str]
    has_chaining: bool
    has_redirection: bool
    has_pipeline: bool
    recommended_action: CommandDecision
    effective_risk_level: RiskLevel
    reason: str

    def to_dict(self) -> dict:
        return {
            "raw_command": self.raw_command,
            "shell_type": self.shell_type,
            "categories": [c.value for c in self.categories],
            "detected_patterns": self.detected_patterns,
            "has_chaining": self.has_chaining,
            "has_redirection": self.has_redirection,
            "has_pipeline": self.has_pipeline,
            "recommended_action": self.recommended_action.value,
            "effective_risk_level": self.effective_risk_level.value,
            "reason": self.reason,
        }


class CommandSandbox:
    """
    Lexical and structural command analyzer.
    """

    # Secret environment key prefixes and substrings to strip when launching untrusted processes
    DEFAULT_SECRET_KEYS: Set[str] = {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AZURE_OPENAI_API_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "SLACK_BOT_TOKEN",
        "SPOTIPY_CLIENT_SECRET",
        "MADDY_AUTH_TOKEN",
        "DATABASE_URL",
        "SECRET_KEY",
    }

    # Destructive commands across PowerShell, cmd, bash
    DESTRUCTIVE_PATTERNS = [
        (r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r|\-\-recursive\s+\-\-force)\b", "Recursive forced file deletion (rm -rf)"),
        (r"\brm\s+.*(-r|-R|--recursive)\s+.*(-f|--force)\b", "Recursive forced file deletion (rm -r -f)"),
        (r"\brm\s+.*(-f|--force)\s+.*(-r|-R|--recursive)\b", "Recursive forced file deletion (rm -f -r)"),
        (r"\bdel(\.exe)?\s+.*(/f|/s|/q)", "Forced recursive deletion (del /f /s)"),
        (r"\brmdir(\.exe)?\s+.*(/s|/q)", "Silent recursive folder removal (rmdir /s /q)"),
        (r"\bformat(\.exe)?\s+[a-zA-Z]:", "Disk formatting command (format X:)"),
        (r"\bdiskpart(\.exe)?\b", "Disk partitioning utility (diskpart)"),
        (r"\bmkfs(\.[a-zA-Z0-9]+)?\b", "Filesystem creation utility (mkfs)"),
        (r"\bfdisk\b", "Partition table manipulator (fdisk)"),
        (r"\bdd\s+if=.*of=/dev/", "Raw disk block overwrite (dd of=/dev/...)"),
        (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;", "Bash fork bomb pattern"),
    ]

    # System interruption commands
    INTERRUPTION_PATTERNS = [
        (r"\bshutdown(\.exe)?\b", "System shutdown invocation"),
        (r"\breboot\b", "System reboot command"),
        (r"\bpoweroff\b", "System poweroff command"),
        (r"\binit\s+[06]\b", "System runlevel transition to halt/reboot"),
        (r"\btaskkill(\.exe)?\s+.*(/f|/im)\s+(explorer\.exe|csrss\.exe|lsass\.exe|svchost\.exe)", "Critical system process termination"),
        (r"\bpkill\s+-9\s+(systemd|init|Xorg|gnome-shell)", "Core OS process kill"),
    ]

    # Privilege escalation commands
    ESCALATION_PATTERNS = [
        (r"\bsudo\b", "Sudo superuser invocation"),
        (r"\brunas(\.exe)?\b", "RunAs elevated command execution"),
        (r"\bSet-ExecutionPolicy\s+(Unrestricted|Bypass)\b", "PowerShell execution policy bypass"),
        (r"\bchmod\s+(-R\s+)?(777|a\+rwx|u\+s)\b", "Overly permissive or SUID permission change"),
        (r"\bchown\s+(-R\s+)?root\b", "Ownership transfer to root"),
        (r"\bnet\s+user\s+.*\/add\b", "Local user account creation"),
    ]

    # Obfuscated execution patterns
    OBFUSCATION_PATTERNS = [
        (r"\bpowershell(\.exe)?\s+.*(-e|-enc|-encodedcommand)\b", "Base64 encoded PowerShell execution"),
        (r"\bInvoke-Expression\b|\biex\b", "Dynamic string execution via Invoke-Expression/iex"),
        (r"\[Convert\]::FromBase64String", "Dynamic Base64 string decoding"),
        (r"\b(python|python3|perl|ruby)\s+-c\s+['\"].*eval\(", "Dynamic in-line code evaluation"),
    ]

    # Remote execution & exfiltration patterns
    REMOTE_PATTERNS = [
        (r"\b(curl|wget)\s+.*\|\s*(sh|bash|powershell|cmd)\b", "Piping downloaded web content directly into shell"),
        (r"\bInvoke-WebRequest\s+.*\|\s*iex\b", "PowerShell download and immediate execute"),
        (r"\birm\s+.*\|\s*iex\b", "PowerShell irm piped to iex"),
        (r"\bnc(\.exe)?\s+.*-e\s+(/bin/sh|/bin/bash|cmd\.exe|powershell\.exe)", "Netcat reverse shell pattern"),
        (r"/dev/tcp/\d+\.\d+\.\d+\.\d+/\d+", "Direct bash /dev/tcp socket connection"),
    ]

    # Known safe query / status commands
    SAFE_COMMAND_PREFIXES = [
        "git status",
        "git log",
        "git diff",
        "git branch",
        "dir",
        "ls",
        "echo",
        "pwd",
        "cd",
        "type",
        "cat",
        "whoami",
        "hostname",
        "date",
        "time",
    ]

    def __init__(self, default_shell: str = "powershell" if os.name == "nt" else "bash"):
        self.default_shell = default_shell.lower()

    def identify_shell(self, command: str) -> str:
        """
        Infer shell interpreter from command prefix or environment.
        """
        cmd_lower = command.strip().lower()
        if cmd_lower.startswith("powershell") or cmd_lower.startswith("pwsh"):
            return "powershell"
        if cmd_lower.startswith("cmd") or cmd_lower.startswith("cmd.exe"):
            return "cmd"
        if cmd_lower.startswith("bash"):
            return "bash"
        if cmd_lower.startswith("sh"):
            return "sh"
        return self.default_shell

    def analyze_command(self, command: str, shell_type: Optional[str] = None) -> CommandAssessment:
        """
        Perform deep structural and pattern analysis on a shell command string.
        """
        if not command or not isinstance(command, str):
            return CommandAssessment(
                raw_command="",
                shell_type=self.default_shell,
                categories=[CommandRiskCategory.DESTRUCTIVE],
                detected_patterns=["Empty or invalid command string"],
                has_chaining=False,
                has_redirection=False,
                has_pipeline=False,
                recommended_action=CommandDecision.DENY,
                effective_risk_level=RiskLevel.CRITICAL,
                reason="Command validation failed: command string is empty or invalid",
            )

        cmd = command.strip()
        shell = (shell_type or self.identify_shell(cmd)).lower()

        # 1. Structural composition analysis
        has_chaining = bool(re.search(r"(&&|\|\||;)", cmd))
        has_redirection = bool(re.search(r"(>|>>|<|2>&1)", cmd))
        has_pipeline = bool(re.search(r"(?<!\|)\|(?!\|)", cmd))

        categories: Set[CommandRiskCategory] = set()
        detected_patterns: List[str] = []

        # 2. Check destructive patterns
        for pattern, desc in self.DESTRUCTIVE_PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                categories.add(CommandRiskCategory.DESTRUCTIVE)
                detected_patterns.append(desc)

        # 3. Check system interruption
        for pattern, desc in self.INTERRUPTION_PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                categories.add(CommandRiskCategory.SYSTEM_INTERRUPTION)
                detected_patterns.append(desc)

        # 4. Check privilege escalation
        for pattern, desc in self.ESCALATION_PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                categories.add(CommandRiskCategory.PRIVILEGE_ESCALATION)
                detected_patterns.append(desc)

        # 5. Check obfuscation
        for pattern, desc in self.OBFUSCATION_PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                categories.add(CommandRiskCategory.OBFUSCATED_EXECUTION)
                detected_patterns.append(desc)

        # 6. Check remote execution / exfiltration
        for pattern, desc in self.REMOTE_PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                categories.add(CommandRiskCategory.REMOTE_EXECUTION)
                detected_patterns.append(desc)

        # 7. Check if command matches known safe read/status prefixes without dangerous composition
        is_known_safe = False
        cmd_lower = cmd.lower()
        if not categories and not has_chaining and not has_redirection and not has_pipeline:
            for safe_prefix in self.SAFE_COMMAND_PREFIXES:
                if cmd_lower == safe_prefix or cmd_lower.startswith(safe_prefix + " "):
                    is_known_safe = True
                    break

        # 8. Determine risk level and decision
        if CommandRiskCategory.DESTRUCTIVE in categories or CommandRiskCategory.REMOTE_EXECUTION in categories:
            recommended_action = CommandDecision.DENY
            effective_risk = RiskLevel.CRITICAL
            reason = f"Command blocked: contains dangerous patterns ({', '.join(detected_patterns)})"
        elif CommandRiskCategory.SYSTEM_INTERRUPTION in categories or CommandRiskCategory.PRIVILEGE_ESCALATION in categories or CommandRiskCategory.OBFUSCATED_EXECUTION in categories:
            recommended_action = CommandDecision.CONFIRM
            effective_risk = RiskLevel.HIGH
            reason = f"Command requires confirmation: ({', '.join(detected_patterns)})"
        elif has_chaining or has_pipeline or has_redirection:
            recommended_action = CommandDecision.CONFIRM
            effective_risk = RiskLevel.MEDIUM
            reason = "Command contains composition constructs (chaining, pipeline, or redirection)"
            categories.add(CommandRiskCategory.SAFE)
        elif is_known_safe:
            recommended_action = CommandDecision.ALLOW
            effective_risk = RiskLevel.LOW
            reason = "Command recognized as safe read-only operation"
            categories.add(CommandRiskCategory.SAFE)
        else:
            # Uncategorized standalone command
            recommended_action = CommandDecision.CONFIRM
            effective_risk = RiskLevel.MEDIUM
            reason = "General command execution requires standard policy check"
            categories.add(CommandRiskCategory.SAFE)

        return CommandAssessment(
            raw_command=cmd,
            shell_type=shell,
            categories=list(categories),
            detected_patterns=detected_patterns,
            has_chaining=has_chaining,
            has_redirection=has_redirection,
            has_pipeline=has_pipeline,
            recommended_action=recommended_action,
            effective_risk_level=effective_risk,
            reason=reason,
        )

    @classmethod
    def sanitize_environment(
        cls,
        env: Optional[Dict[str, str]] = None,
        extra_secret_keys: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Produce a sanitized copy of environment variables, stripping all API keys,
        passwords, and session secrets to prevent leakage to child subprocesses.
        """
        source_env = dict(env if env is not None else os.environ)
        sanitized = {}

        secret_keys = set(cls.DEFAULT_SECRET_KEYS)
        if extra_secret_keys:
            secret_keys.update(extra_secret_keys)

        secret_key_substrings = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "AUTH", "CREDENTIAL")

        for k, v in source_env.items():
            k_upper = k.upper()
            if k in secret_keys or k_upper in secret_keys:
                continue
            if any(sub in k_upper for sub in secret_key_substrings):
                continue
            sanitized[k] = v

        return sanitized
