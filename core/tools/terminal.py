"""Terminal and shell execution tool for M.A.D.D.Y. Agent Core (Modular Autonomous Digital Daemon for You).

Provides safe command execution with timeout bounding, workdir validation, and classified output.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from core.permissions import PermissionTier
from core.redact import redact_text
from core.tools.base import BaseTool, ToolResult

_WORKDIR_SAFE_CHARS = frozenset("/\\:_-.~ +@=,")
DEFAULT_MAX_OUTPUT_CHARS = 25_000
HEAD_RATIO = 0.4


def truncate_head_tail(text: str, max_chars: int = DEFAULT_MAX_OUTPUT_CHARS, label: str = "OUTPUT") -> str:
    """Head/tail truncation for oversized tool output.

    Preserves 40% head (command initialization/early output) and 60% tail
    (latest lines/stack traces) with a prominent omission banner.
    """
    if not text or len(text) <= max_chars:
        return text
    head_len = int(max_chars * HEAD_RATIO)
    tail_len = max_chars - head_len
    omitted = len(text) - head_len - tail_len
    notice = f"\n\n... [{label} TRUNCATED - {omitted:,} chars omitted out of {len(text):,} total] ...\n\n"
    return text[:head_len] + notice + text[-tail_len:]


def validate_workdir(workdir: str) -> Optional[str]:
    """Validate that working directory contains no shell metacharacters."""
    for ch in workdir:
        if not (ch.isalnum() or ch in _WORKDIR_SAFE_CHARS):
            return f"Blocked: workdir contains disallowed character {ch!r}."
    return None


class ExecuteCommandTool(BaseTool):
    """Executes a shell command on the host with timeout bounding and output capture."""

    name = "execute_command"
    description = (
        "Execute a shell command on the host system. Bounded by a strict timeout (default 10s). "
        "System-impacting operation requiring authorization."
    )
    tier = PermissionTier.SYSTEM_IMPACTING

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash/shell command string to execute.",
                },
                "workdir": {
                    "type": "string",
                    "description": "Optional working directory in which to execute the command.",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Execution timeout limit in seconds (default: 10.0, max: 30.0).",
                    "default": 10.0,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        }

    def execute(
        self,
        command: str,
        workdir: Optional[str] = None,
        timeout_seconds: float = 10.0,
        **kwargs: Any,
    ) -> ToolResult:
        timeout = min(max(1.0, float(timeout_seconds)), 30.0)

        cwd = Path(os.getcwd())
        if workdir:
            validation_err = validate_workdir(workdir)
            if validation_err:
                return ToolResult(success=False, output="", error=validation_err)
            cwd_candidate = Path(os.path.expanduser(workdir)).resolve()
            if not cwd_candidate.is_dir():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Specified workdir does not exist or is not a directory: {cwd_candidate}",
                )
            cwd = cwd_candidate

        is_posix = (os.name != "nt")
        popen_kwargs: Dict[str, Any] = {
            "shell": True,
            "cwd": str(cwd),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
        }
        if is_posix:
            popen_kwargs["start_new_session"] = True
        else:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            proc = subprocess.Popen(command, **popen_kwargs)
            try:
                raw_stdout, raw_stderr = proc.communicate(timeout=timeout)
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                # Terminate entire process tree to prevent runaway zombie processes
                if is_posix:
                    try:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=0.25)
                        except subprocess.TimeoutExpired:
                            os.killpg(pgid, signal.SIGKILL)
                    except Exception:
                        with contextlib.suppress(Exception):
                            proc.kill()
                else:
                    try:
                        subprocess.run(
                            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                            capture_output=True,
                            timeout=5,
                        )
                    except Exception:
                        with contextlib.suppress(Exception):
                            proc.kill()
                with contextlib.suppress(Exception):
                    proc.communicate(timeout=1.0)
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Command timed out after {timeout} seconds.",
                    metadata={"command": redact_text(command), "timeout": timeout},
                )

            stdout = redact_text(raw_stdout or "")
            stderr = redact_text(raw_stderr or "")

            output = stdout
            if stderr:
                output = f"{stdout}\n[STDERR]\n{stderr}".strip() if stdout else f"[STDERR]\n{stderr}".strip()

            # Bound output size with head/tail truncation to protect prompt context
            bounded_output = truncate_head_tail(output, max_chars=DEFAULT_MAX_OUTPUT_CHARS)

            return ToolResult(
                success=(exit_code == 0),
                output=bounded_output if bounded_output else f"Command exited with code {exit_code} (no output).",
                error=f"Exit code {exit_code}" if exit_code != 0 else None,
                metadata={
                    "exit_code": exit_code,
                    "command": redact_text(command),
                    "workdir": str(cwd),
                    "truncated": len(output) > DEFAULT_MAX_OUTPUT_CHARS,
                },
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to execute command: {str(exc)}",
            )
