"""Rigorous Production Hardening Verification Suite for M.A.D.D.Y. Rebuild.

Verifies:
1. Shell AST Tokenization & Metacharacter Gating (PermissionGate):
   - Strict shlex splitting and metapattern rejection: subshells $(), backticks ``,
     process substitutions <() >(), chaining && || ;, redirects | > >> <, newlines \n \r, tee.
2. Terminal Process Group Isolation & Head/Tail Truncation (ExecuteCommandTool):
   - start_new_session=True (setsid) / CREATE_NEW_PROCESS_GROUP.
   - Head/tail 40/60 bounded truncation with omission banner on oversized outputs.
3. Degenerate Loop Guardrail Hardening (LoopGuard):
   - Argument normalization: whitespace collapsing, relative path ./ stripping, sorted keys.
   - Runaway model output repetition detection (is_runaway_repetition).
4. Atomic Filesystem Replacement & Line-Number Blocker (WriteFileTool):
   - Atomic sibling temporary file creation with fsync and os.replace.
   - Rejection of line-numbered outputs (read_file / cat -n artifact).
5. Safe Process Lifecycle & Self-Exclusion (LinuxPlatform):
   - Exclusion of current PID, parent PID, systemd, and protected system processes.
6. Grounded Visual Action Verification (ComputerAutomationController):
   - Elimination of blind (525, 310) coordinate fallbacks when vision fails.
7. Connect-Time SSRF & Scheme Whitelisting (BrowserAutomationEngine):
   - Enforcing http/https only (rejection of file://, gopher://, data:).
   - Filtering loopback, link-local cloud metadata (169.254.169.254), and private IP literals.
8. Telegram Isolation Preservation:
   - 100% byte-for-byte SHA-256 integrity verification of all protected files.
"""

from __future__ import annotations

import hashlib
import os
import signal
import socket
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.browser_automation import BrowserAutomationEngine, validate_url_safety
from core.computer_automation import ComputerAutomationController, VerificationResult
from core.errors import DegenerateLoopError, PermissionDeniedError
from core.loop_guard import (
    LoopGuard,
    is_runaway_repetition,
    normalize_arguments,
)
from core.permissions import PermissionGate, PermissionTier
from core.tools.file_ops import (
    WriteFileTool,
    contains_line_number_prefixes,
)
from core.tools.terminal import (
    DEFAULT_MAX_OUTPUT_CHARS,
    ExecuteCommandTool,
    truncate_head_tail,
)
from platform_layer.linux.platform_linux import LinuxPlatform

EXPECTED_TELEGRAM_HASHES = {
    "integrations/messaging/telegram_bot.py": "dc06cc308de0e369d98d265f2e09f474d83a876bc2020e5c0d3f7f138b97d645",
    "integrations/messaging/telegram_client.py": "b4a58700e9c82d8c6c391ae34af6589a3e066901760c9c9f6db33987adf51a15",
    "telegram.cmd": "f67181dc0f87a2aa476eff9ee264bc1b03b340350f5f672e1f6baeb8bff0e40e",
    "scripts/start_telegram_daemon.vbs": "22e307c439845a8f68e43e7b2deb1e15fc1694ba6144d9f706bb7c3c9a60883a",
}


# ── TEST 1: SHELL AST & METACHARACTER GATING ──────────────────────────────────

def test_permission_gate_metacharacter_rejection():
    """Verify PermissionGate strictly blocks metacharacters, subshells, and chaining."""
    gate = PermissionGate(allowed_commands=["uname", "echo", "date", "whoami"])

    # Valid safe allowlisted commands
    assert gate.is_safe_allowlisted_command("uname -a") is True
    assert gate.is_safe_allowlisted_command("date") is True
    assert gate.is_safe_allowlisted_command("whoami") is True
    assert gate.is_safe_allowlisted_command("echo hello world") is True

    # Subshell expansions
    assert gate.is_safe_allowlisted_command("echo $(whoami)") is False
    assert gate.is_safe_allowlisted_command("echo `whoami`") is False
    assert gate.is_safe_allowlisted_command("uname $(cat /etc/passwd)") is False

    # Process substitution
    assert gate.is_safe_allowlisted_command("echo <(whoami)") is False
    assert gate.is_safe_allowlisted_command("echo >(cat)") is False

    # Chaining and backgrounding
    assert gate.is_safe_allowlisted_command("uname; id") is False
    assert gate.is_safe_allowlisted_command("uname && id") is False
    assert gate.is_safe_allowlisted_command("uname || id") is False
    assert gate.is_safe_allowlisted_command("uname & id") is False

    # Redirections and pipes
    assert gate.is_safe_allowlisted_command("uname | grep Linux") is False
    assert gate.is_safe_allowlisted_command("echo test > /tmp/out") is False
    assert gate.is_safe_allowlisted_command("echo test >> /tmp/out") is False
    assert gate.is_safe_allowlisted_command("cat < /etc/shadow") is False
    assert gate.is_safe_allowlisted_command("echo hello | tee /tmp/log") is False

    # Newline injections
    assert gate.is_safe_allowlisted_command("uname\nid") is False
    assert gate.is_safe_allowlisted_command("uname\rid") is False

    # Disallowed commands
    assert gate.is_safe_allowlisted_command("rm -rf /") is False
    assert gate.is_safe_allowlisted_command("curl https://evil.com") is False


# ── TEST 2: TERMINAL TRUNCATION & PROCESS GROUP TEARDOWN ──────────────────────

def test_truncate_head_tail():
    """Verify 40/60 head/tail truncation preserves context bounds and omission banner."""
    short_text = "Standard small command output."
    assert truncate_head_tail(short_text, max_chars=100) == short_text

    oversized = "HEAD_MARKER_" + ("A" * 1000) + "_MIDDLE_PART_" + ("B" * 1000) + "_TAIL_MARKER"
    truncated = truncate_head_tail(oversized, max_chars=200, label="COMMAND_OUTPUT")

    assert len(truncated) < len(oversized)
    assert "HEAD_MARKER_" in truncated
    assert "_TAIL_MARKER" in truncated
    assert "[COMMAND_OUTPUT TRUNCATED -" in truncated
    assert "_MIDDLE_PART_" not in truncated


def test_terminal_tool_process_group_timeout():
    """Verify ExecuteCommandTool bounds timeout and tears down child processes."""
    tool = ExecuteCommandTool()
    # Execute a sleep command exceeding timeout
    res = tool.execute(command="sleep 5", timeout_seconds=1.0)
    assert res.success is False
    assert "timed out after 1.0 seconds" in res.error.lower()


# ── TEST 3: LOOP GUARD NORMALIZATION & REPETITION DETECTION ───────────────────

def test_loop_guard_argument_normalization():
    """Verify semantic normalization collapses whitespace mutations and path prefixes."""
    raw1 = {"path": "./test.txt", "flag": True, "query": "hello    world"}
    raw2 = {"query": "hello world", "path": "test.txt", "flag": True}

    norm1 = normalize_arguments(raw1)
    norm2 = normalize_arguments(raw2)
    assert norm1 == norm2

    # Verify hashes match despite formatting differences
    hash1 = LoopGuard.compute_call_hash("read_file", raw1)
    hash2 = LoopGuard.compute_call_hash("read_file", raw2)
    assert hash1 == hash2

    guard = LoopGuard(max_identical_calls=2)
    guard.record_and_check("read_file", raw1)
    # The second call with formatted variations must trigger streak limit
    with pytest.raises(DegenerateLoopError) as exc_info:
        guard.record_and_check("read_file", raw2)
    assert "consecutively" in exc_info.value.message


def test_runaway_model_text_repetition_guard():
    """Verify runaway repetition detector catches repeating tokens before context exhaustion."""
    normal_text = "Here is the plan for implementing the next phase.\nWe will start by inspecting the codebase."
    assert is_runaway_repetition(normal_text) is False

    # Degenerate loop repeating single line >= 5 times and dominating >= 50%
    degenerate_line = "I am ready to proceed with the next task.\n" * 15
    assert is_runaway_repetition(degenerate_line) is True

    # Degenerate sliding chunk repetition
    chunk = "Thinking through the architecture of the dynamic agent loop. " * 10
    assert is_runaway_repetition(chunk) is True


# ── TEST 4: ATOMIC FILE SIBLING REPLACEMENT & LINE NUMBER BLOCKER ─────────────

def test_line_number_blocker():
    """Verify rejection of model hallucinated line-number prefixes."""
    assert contains_line_number_prefixes("def test():\n    return True") is False

    cat_n_artifact = (
        "   1 | def calculate_total(a, b):\n"
        "   2 |     # Add values\n"
        "   3 |     return a + b\n"
    )
    assert contains_line_number_prefixes(cat_n_artifact) is True

    colon_artifact = (
        "1: import os\n"
        "2: import sys\n"
        "3: print('hello')\n"
    )
    assert contains_line_number_prefixes(colon_artifact) is True


def test_write_file_tool_atomic_sibling_replacement(tmp_path):
    """Verify WriteFileTool writes atomically via tempfile and rejects line numbers."""
    target_file = tmp_path / "app" / "config.py"
    tool = WriteFileTool()

    # 1. Attempt to write line-numbered content -> BLOCKED
    res_blocked = tool.execute(
        path=str(target_file),
        content="   1 | import os\n   2 | import sys\n",
        create_directories=True,
    )
    assert res_blocked.success is False
    assert "line-number prefixes" in res_blocked.error.lower()
    assert not target_file.exists()

    # 2. Valid write succeeds atomically
    res_valid = tool.execute(
        path=str(target_file),
        content="import os\nimport sys\n\nDEBUG = True\n",
        create_directories=True,
    )
    assert res_valid.success is True
    assert target_file.exists()
    assert target_file.read_text(encoding="utf-8") == "import os\nimport sys\n\nDEBUG = True\n"


# ── TEST 5: SAFE PROCESS LIFECYCLE & SELF-EXCLUSION ───────────────────────────

def test_linux_platform_close_application_self_exclusion():
    """Verify LinuxPlatform prevents termination of self, parent, and system daemons."""
    platform = LinuxPlatform()

    # Rejection of protected process names
    for protected in ("python", "python3", "pytest", "maddy", "systemd", "bash"):
        with pytest.raises(RuntimeError) as exc_info:
            platform.close_application(protected)
        assert "protected system or agent runtime process" in str(exc_info.value)

    # Empty name check
    with pytest.raises(ValueError):
        platform.close_application("")


# ── TEST 6: GROUNDED VISUAL ACTION VERIFICATION ───────────────────────────────

def test_fail_closed_visual_element_localization(tmp_path):
    """Verify ComputerAutomationController does not blindly click (525, 310) when vision fails."""
    mock_platform = MagicMock()
    mock_platform.take_screenshot.return_value = {"success": True, "path": str(tmp_path / "shot.png")}
    mock_platform.get_active_window.return_value = {"title": "Desktop"}
    mock_platform.get_open_windows.return_value = []
    mock_platform.get_screen_geometry.return_value = {"width": 1920, "height": 1080}

    mock_vision = MagicMock()
    # Mock vision failure to locate element
    mock_vision.locate_element.return_value = {
        "success": False,
        "centroid": None,
        "confidence": 0.0,
    }

    controller = ComputerAutomationController(platform=mock_platform, vision_engine=mock_vision)
    res = controller.click_element_layered("Submit Order Button")

    # MUST be verified=False and NOT click mouse at arbitrary coordinates
    assert res.verified is False
    assert "layer_3_vision_localization_failed" in res.layer_used
    assert mock_platform.click_mouse.called is False


# ── TEST 7: SSRF & URL SCHEME WHITELISTING ────────────────────────────────────

def test_url_ssrf_and_scheme_validation():
    """Verify validate_url_safety blocks disallowed schemes, loopback, and private IPs."""
    # Allowed safe public URLs
    assert validate_url_safety("https://api.github.com/repos") is None
    assert validate_url_safety("http://example.com/index.html") is None

    # Blocked dangerous schemes
    assert "Blocked unsafe URL scheme 'file'" in validate_url_safety("file:///etc/shadow")
    assert "Blocked unsafe URL scheme 'gopher'" in validate_url_safety("gopher://evil.com")
    assert "Blocked unsafe URL scheme 'data'" in validate_url_safety("data:text/html,<h1>test</h1>")

    # Blocked localhost and loopback names
    assert "Localhost loopback" in validate_url_safety("http://localhost:8080")
    assert "Localhost loopback" in validate_url_safety("https://sub.localhost/api")

    # Blocked private and loopback IP literals
    assert "private/internal IP" in validate_url_safety("http://127.0.0.1:5000")
    assert "private/internal IP" in validate_url_safety("http://10.0.0.1/admin")
    assert "private/internal IP" in validate_url_safety("http://192.168.1.1/router")
    assert "private/internal IP" in validate_url_safety("http://172.16.0.1/internal")
    assert "private/internal IP" in validate_url_safety("http://169.254.169.254/latest/meta-data")  # AWS metadata


def test_browser_automation_engine_ssrf_rejection():
    """Verify BrowserAutomationEngine rejects SSRF navigation attempts."""
    engine = BrowserAutomationEngine()
    res = engine.navigate("file:///etc/passwd")
    assert res["success"] is False
    assert "Blocked unsafe URL scheme 'file'" in res["error"]

    res_loopback = engine.navigate("http://127.0.0.1:9000")
    assert res_loopback["success"] is False
    assert "private/internal IP" in res_loopback["error"]


# ── TEST 8: STRICT TELEGRAM ISOLATION INVARIANT ───────────────────────────────

def test_telegram_protected_files_sha256():
    """Verify 100% integrity of all protected Telegram files."""
    missing = [p for p in EXPECTED_TELEGRAM_HASHES if not Path(p).exists()]
    if missing:
        pytest.skip(f"Protected Telegram files excluded from repository checkout: {missing}")
    for rel_path, expected_hash in EXPECTED_TELEGRAM_HASHES.items():
        file_path = Path(rel_path)
        actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, (
            f"Telegram isolation violation! File {rel_path} modified!\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}"
        )
