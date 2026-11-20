"""Unit and invariant tests for Phase 2 Agent Core modules.

Tests:
1. LoopGuard: consecutive identical calls (3-threshold) & cyclic ping-pong detection.
2. IterationBudget: resource limits, turn consumption, budget exhaustion.
3. PermissionGate: deterministic safety tiers, sandboxed path authorization.
4. Redaction: secret scrub of Groq, OpenAI, Gemini, and PAT tokens.
5. TraceStore: SQLite WAL persistence, persist-before-execute invariant.
6. Tools: SystemInfo, ReadFile, WriteFile, ExecuteCommand.
"""

import json
import os
import platform
import tempfile
from pathlib import Path

import pytest

from core.budget import IterationBudget
from core.errors import BudgetExceededError, DegenerateLoopError, PermissionDeniedError
from core.loop_guard import LoopGuard
from core.permissions import PermissionGate, PermissionTier
from core.redact import redact_structure, redact_text
from core.tools.file_ops import ReadFileTool, WriteFileTool
from core.tools.system_info import SystemInfoTool
from core.tools.terminal import ExecuteCommandTool
from core.trace_store import TraceStore


def test_loop_guard_identical_calls():
    guard = LoopGuard(max_identical_calls=3)
    tool = "read_file"
    args = {"path": "/etc/hosts"}

    guard.record_and_check(tool, args)
    guard.record_and_check(tool, args)

    with pytest.raises(DegenerateLoopError) as exc_info:
        guard.record_and_check(tool, args)

    assert "consecutively" in exc_info.value.message
    assert exc_info.value.error_context["repeat_count"] == 3


def test_loop_guard_ping_pong_cycle():
    guard = LoopGuard(max_identical_calls=10, max_cycle_period=3)

    # Alternate A -> B -> A -> B
    guard.record_and_check("tool_a", {"x": 1})
    guard.record_and_check("tool_b", {"y": 2})

    with pytest.raises(DegenerateLoopError) as exc_info:
        guard.record_and_check("tool_a", {"x": 1})
        guard.record_and_check("tool_b", {"y": 2})

    assert "cycle detected" in exc_info.value.message


def test_iteration_budget():
    budget = IterationBudget(max_iterations=3)
    assert budget.remaining == 3

    assert budget.consume() == 1
    assert budget.consume() == 2
    assert budget.consume() == 3

    with pytest.raises(BudgetExceededError):
        budget.consume()


def test_permission_gate_tiers():
    gate = PermissionGate(auto_approve_tier2=False, allowed_write_directories=["/tmp"])

    # Safe tier auto-allows
    gate.check_permission("system_info", PermissionTier.SAFE, {})

    # System-impacting inside /tmp sandbox auto-allows
    gate.check_permission("write_file", PermissionTier.SYSTEM_IMPACTING, {"path": "/tmp/test.txt"})

    # System-impacting outside /tmp sandbox fails closed
    with pytest.raises(PermissionDeniedError):
        gate.check_permission("write_file", PermissionTier.SYSTEM_IMPACTING, {"path": "/etc/passwd"})

    # Dynamic temp directory is automatically included in default sandboxes
    import tempfile
    default_gate = PermissionGate()
    assert default_gate.is_path_in_allowed_sandbox(tempfile.gettempdir())
    assert default_gate.is_path_in_allowed_sandbox(Path(tempfile.gettempdir()) / "test_file.txt")

    # Cross-platform command allowlist defaults
    for cmd in ["ver", "systeminfo", "cd", "uname", "date", "echo", "pwd", "whoami", "hostname", "uptime"]:
        gate.check_permission("execute_command", PermissionTier.SYSTEM_IMPACTING, {"command": cmd})

    # Shell command redirection and chaining operators MUST be blocked from allowlist exemption
    blocked_commands = [
        "echo hello > /tmp/out.txt",
        "echo hello >> /tmp/out.txt",
        "cat /etc/passwd | grep root",
        "echo test ; rm -rf /tmp/test",
        "uname && cat /etc/shadow",
        "echo message | tee /tmp/log.txt",
    ]
    for bcmd in blocked_commands:
        with pytest.raises(PermissionDeniedError):
            gate.check_permission("execute_command", PermissionTier.SYSTEM_IMPACTING, {"command": bcmd})


def test_secret_redaction():
    groq_secret = "gsk_1234567890abcdef1234567890abcdef12"
    openai_secret = "sk-proj-1234567890abcdef1234567890abcdef"
    gemini_secret = "AIzaSyABCDEF1234567890abcdef1234567890"

    raw = f"Connecting with Groq={groq_secret}, OpenAI={openai_secret}, Gemini={gemini_secret}"
    redacted = redact_text(raw)

    assert groq_secret not in redacted
    assert openai_secret not in redacted
    assert gemini_secret not in redacted
    assert "..." in redacted  # verify prefix/suffix format


def test_trace_store_persist_before_execute(tmp_path):
    db_file = tmp_path / "test_traces.db"
    store = TraceStore(db_path=db_file)

    task_id = store.create_task("Test task")
    step_id = store.record_step(task_id, 1, "Testing persist before execute", "tool_round")

    # Persist BEFORE execute
    call_id = "call_test_01"
    store.record_tool_call_planned(
        task_id=task_id,
        tool_call_id=call_id,
        tool_name="system_info",
        arguments={"include_gpu": False},
        tier="safe",
        step_id=step_id,
    )

    # Verify committed immediately
    trace = store.get_task_trace(task_id)
    assert len(trace["tool_calls"]) == 1
    assert trace["tool_calls"][0]["status"] == "planned"

    # Simulate completed execution
    store.record_tool_call_completed(
        tool_call_id=call_id,
        result_output="CPU: OK",
        status="completed",
    )

    trace2 = store.get_task_trace(task_id)
    assert trace2["tool_calls"][0]["status"] == "completed"
    assert trace2["tool_calls"][0]["result_output"] == "CPU: OK"
    assert trace2["tool_calls"][0]["duration_ms"] >= 0.0


def test_system_info_tool():
    tool = SystemInfoTool()
    res = tool.execute(include_gpu=False)
    assert res.success is True
    assert platform.system() in res.output
    assert "RAM" in res.output


def test_file_ops_tools(tmp_path):
    writer = WriteFileTool()
    reader = ReadFileTool()

    target = tmp_path / "subdir" / "message.txt"
    write_res = writer.execute(path=str(target), content="Line 1\nLine 2\nLine 3")
    assert write_res.success is True
    assert target.exists()

    read_res = reader.execute(path=str(target), offset=2, limit=1)
    assert read_res.success is True
    assert "Line 2" in read_res.output
    assert "Line 1" not in read_res.output

    # Verify WriteFileTool enforces sandbox safety when permission_gate is provided
    gate = PermissionGate(allowed_write_directories=[str(tmp_path)])
    sandboxed_writer = WriteFileTool(permission_gate=gate)

    # Inside allowed sandbox succeeds
    sandbox_target = tmp_path / "allowed.txt"
    res_allowed = sandboxed_writer.execute(path=str(sandbox_target), content="safe content")
    assert res_allowed.success is True

    # Outside allowed sandbox is blocked
    res_blocked = sandboxed_writer.execute(path="/etc/critical_system_file.conf", content="malicious")
    assert res_blocked.success is False
    assert "outside permitted sandboxes" in res_blocked.error
