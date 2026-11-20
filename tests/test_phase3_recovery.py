"""Unit and invariant tests for Phase 3 Checkpointing, Recovery, and Resumption.

Tests:
1. TraceStore: Checkpoint persistence, chronological ordering, and JSON serialization.
2. Checkpoint Creation: run_turn automatically saves checkpoints after each tool round.
3. Resumption: resume_turn restores message history and budget, continuing execution
   without repeating completed tool calls.
4. Error Handling: Attempting to resume a nonexistent or uncheckpointed task raises MaddyError.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.agent_loop import MaddyAgentCore
from core.errors import MaddyError
from core.llm_client import CloudLLMClient, ModelResponse, ToolCallEnvelope
from core.permissions import PermissionGate
from core.tools.base import ToolRegistry
from core.tools.calculator import CalculatorTool
from core.tools.system_info import SystemInfoTool
from core.trace_store import TraceStore


def test_trace_store_checkpoint_crud(tmp_path):
    db_file = tmp_path / "test_recovery.db"
    store = TraceStore(db_path=db_file)
    task_id = store.create_task("Test checkpoint task")

    # Initially no checkpoints
    assert store.get_latest_checkpoint(task_id) is None
    assert store.list_checkpoints(task_id) == []

    # Save checkpoint 1
    chk1_id = store.save_checkpoint(
        task_id=task_id,
        step_index=1,
        state_snapshot={"step": 1, "data": "alpha", "budget": 3},
    )
    assert chk1_id.startswith(f"chk_{task_id}_1_")

    # Save checkpoint 2
    chk2_id = store.save_checkpoint(
        task_id=task_id,
        step_index=2,
        state_snapshot={"step": 2, "data": "beta", "budget": 2},
    )

    # Verify latest checkpoint is step 2
    latest = store.get_latest_checkpoint(task_id)
    assert latest is not None
    assert latest["checkpoint_id"] == chk2_id
    assert latest["step_index"] == 2
    assert latest["state_snapshot"]["data"] == "beta"

    # Verify list checkpoints contains both in chronological order
    all_chks = store.list_checkpoints(task_id)
    assert len(all_chks) == 2
    assert all_chks[0]["step_index"] == 1
    assert all_chks[1]["step_index"] == 2

    # Verify direct lookup by checkpoint_id
    single = store.get_checkpoint(chk1_id)
    assert single is not None
    assert single["state_snapshot"]["data"] == "alpha"


def test_agent_turn_creates_checkpoints(tmp_path):
    db_file = tmp_path / "test_agent_checkpoints.db"
    store = TraceStore(db_path=db_file)
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    # Mock LLM to perform 1 tool call then final answer
    mock_llm = MagicMock(spec=CloudLLMClient)
    mock_llm.chat_completion.side_effect = [
        # Round 1: Model requests calculation
        ModelResponse(
            content="Let me compute that.",
            tool_calls=[ToolCallEnvelope(id="tc_calc_01", name="calculator", arguments={"expression": "12 * 12"})],
        ),
        # Round 2: Model provides final answer
        ModelResponse(
            content="12 * 12 is 144.",
            tool_calls=[],
        ),
    ]

    agent = MaddyAgentCore(
        llm_client=mock_llm,
        tool_registry=registry,
        trace_store=store,
    )

    result = agent.run_turn("Calculate 12 * 12")
    assert result["completed"] is True
    assert "144" in result["response"]

    # Verify a checkpoint was automatically saved after tool execution
    checkpoints = store.list_checkpoints(result["task_id"])
    assert len(checkpoints) >= 1
    latest = checkpoints[-1]
    assert latest["step_index"] == 1
    assert len(latest["state_snapshot"]["messages"]) >= 3  # system, user, assistant_tool, tool_result


def test_agent_resume_turn(tmp_path):
    db_file = tmp_path / "test_resumption.db"
    store = TraceStore(db_path=db_file)
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    task_id = store.create_task("Interrupted calculation")

    # Simulate an interrupted task by manually creating a checkpoint after step 1
    messages = [
        {"role": "system", "content": "You are J.A.R.V.I.S."},
        {"role": "user", "content": "Compute 50 * 50"},
        {
            "role": "assistant",
            "content": "Computing...",
            "tool_calls": [{"id": "tc_01", "name": "calculator", "arguments": {"expression": "50 * 50"}}],
        },
        {"role": "tool", "tool_call_id": "tc_01", "name": "calculator", "content": "2500"},
    ]
    # Mark the tool call as completed in the trace store
    store.record_tool_call_planned(task_id, "tc_01", "calculator", {"expression": "50 * 50"}, "safe")
    store.record_tool_call_completed("tc_01", "2500")

    chk_id = store.save_checkpoint(
        task_id=task_id,
        step_index=1,
        state_snapshot={
            "task_id": task_id,
            "step_index": 1,
            "messages": messages,
            "budget_remaining": 3,
        },
    )

    # Setup mock LLM for resumption: it receives the existing messages and responds with final answer
    mock_llm = MagicMock(spec=CloudLLMClient)
    mock_llm.chat_completion.return_value = ModelResponse(
        content="The calculation yields exactly 2,500, sir.",
        tool_calls=[],
    )

    agent = MaddyAgentCore(
        llm_client=mock_llm,
        tool_registry=registry,
        trace_store=store,
    )

    # Resume the task
    resume_result = agent.resume_turn(task_id)

    assert resume_result["completed"] is True
    assert resume_result["resumed"] is True
    assert "2,500" in resume_result["response"]

    # Verify no new tool calls were executed
    trace = store.get_task_trace(task_id)
    assert len(trace["tool_calls"]) == 1  # Only the original tool call exists
    assert trace["task"]["status"] == "completed"

    # Verify task_resumed event was recorded
    events = [e for e in trace["events"] if e["event_type"] == "task_resumed"]
    assert len(events) == 1
    assert json.loads(events[0]["payload"])["resumed_from_step"] == 1


def test_resume_nonexistent_task(tmp_path):
    store = TraceStore(db_path=tmp_path / "empty.db")
    agent = MaddyAgentCore(trace_store=store)

    with pytest.raises(MaddyError) as exc_info:
        agent.resume_turn("task_nonexistent_999")

    assert "no valid checkpoint discovered" in exc_info.value.message


def test_loop_guard_serialization_and_recovery():
    from core.errors import DegenerateLoopError
    from core.loop_guard import LoopGuard, ToolCallSignature

    sig = ToolCallSignature(tool_name="calculator", arguments={"expression": "2 + 2"})
    sig_dict = sig.to_dict()
    assert sig_dict["tool_name"] == "calculator"
    assert sig_dict["arguments"] == {"expression": "2 + 2"}
    assert "call_hash" in sig_dict

    restored_sig = ToolCallSignature.from_dict(sig_dict)
    assert restored_sig.tool_name == "calculator"
    assert restored_sig.arguments == {"expression": "2 + 2"}
    assert restored_sig.call_hash == sig_dict["call_hash"]

    # Test LoopGuard state serialization
    lg = LoopGuard(max_identical_calls=3)
    lg.record_and_check("calculator", {"expression": "1 + 1"})
    lg.record_and_check("calculator", {"expression": "1 + 1"})
    assert lg._identical_streak == 2
    assert lg.streak_count == 2

    lg_dict = lg.to_dict()
    assert lg_dict["_identical_streak"] == 2
    assert len(lg_dict["_call_history"]) == 2

    # Deserialize
    restored_lg = LoopGuard.from_dict(lg_dict)
    assert restored_lg.max_identical_calls == 3
    assert restored_lg._identical_streak == 2
    assert restored_lg.streak_count == 2
    assert list(restored_lg._call_history) == list(lg._call_history)

    # Calling the identical tool for the 3rd time on the restored guard should halt
    with pytest.raises(DegenerateLoopError):
        restored_lg.record_and_check("calculator", {"expression": "1 + 1"})


def test_resume_turn_restores_loop_guard_and_prevents_looping(tmp_path):
    """Verify that resuming a task restores loop_guard state and prevents identical call looping."""
    db_file = tmp_path / "test_lg_resumption.db"
    store = TraceStore(db_path=db_file)
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    task_id = store.create_task("Interrupted loop task")

    from core.loop_guard import LoopGuard
    lg = LoopGuard(max_identical_calls=3)
    lg.record_and_check("calculator", {"expression": "100 / 2"})
    lg.record_and_check("calculator", {"expression": "100 / 2"})
    assert lg._identical_streak == 2

    messages = [
        {"role": "system", "content": "You are J.A.R.V.I.S."},
        {"role": "user", "content": "Keep calculating"},
        {"role": "assistant", "content": "Computing...", "tool_calls": [{"id": "tc_1", "name": "calculator", "arguments": {"expression": "100 / 2"}}]},
        {"role": "tool", "tool_call_id": "tc_1", "name": "calculator", "content": "50"},
        {"role": "assistant", "content": "Computing again...", "tool_calls": [{"id": "tc_2", "name": "calculator", "arguments": {"expression": "100 / 2"}}]},
        {"role": "tool", "tool_call_id": "tc_2", "name": "calculator", "content": "50"},
    ]

    store.save_checkpoint(
        task_id=task_id,
        step_index=2,
        state_snapshot={
            "task_id": task_id,
            "step_index": 2,
            "messages": messages,
            "budget_remaining": 3,
            "loop_guard": lg.to_dict(),
        },
    )

    mock_llm = MagicMock(spec=CloudLLMClient)
    mock_llm.chat_completion.side_effect = [
        ModelResponse(
            content="Let me calculate again.",
            tool_calls=[ToolCallEnvelope(id="tc_3", name="calculator", arguments={"expression": "100 / 2"})],
        ),
        ModelResponse(
            content="Loop was blocked, ending turn.",
            tool_calls=[],
        ),
    ]

    agent = MaddyAgentCore(
        llm_client=mock_llm,
        tool_registry=registry,
        trace_store=store,
    )

    res = agent.resume_turn(task_id)
    assert res["resumed"] is True

    # Check trace: tc_3 was planned and blocked by Guardrail Blocked (DegenerateLoopError)
    trace = store.get_task_trace(task_id)
    blocked_calls = [tc for tc in trace["tool_calls"] if tc["tool_call_id"] == "tc_3"]
    assert len(blocked_calls) == 1
    assert blocked_calls[0]["status"] == "blocked"
    assert "Guardrail Blocked" in blocked_calls[0]["result_output"]
    assert "identical arguments 3 times" in blocked_calls[0]["error_info"]

