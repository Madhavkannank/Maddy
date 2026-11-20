"""Unit tests for M.A.D.D.Y. Phase 4 Memory System.

Fulfills CLAUDE.md Phase 4 Requirements:
1. SQLite-based memory storage with WAL mode.
2. FTS5 retrieval with BM25 score ranking.
3. Triad synchronization triggers (ai, ad, au).
4. Robust FTS5 query syntax sanitization (injection & syntax error defense).
5. Bi-temporal invalidation (created_at, valid_at, invalid_at) for superseding facts.
6. Soft deletion (forget) and relevance exclusion.
7. Task history FTS5 retrieval.
8. Context budgeting and token/character pruning.
9. Native memory tools (remember, recall, forget) integration with ToolRegistry.
10. Dynamic memory injection in MaddyAgentCore.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.errors import MaddyError
from core.memory import (
    MemoryCategory,
    MemoryRecord,
    MemoryStore,
    build_fts_query,
    sanitize_fts_query,
)
from core.permissions import PermissionTier
from core.tools.base import ToolRegistry
from core.tools.memory_tools import ForgetTool, RecallTool, RememberTool


@pytest.fixture
def temp_db_path(tmp_path):
    return tmp_path / "test_maddy_memory.db"


@pytest.fixture
def memory_store(temp_db_path):
    store = MemoryStore(db_path=temp_db_path)
    return store


# ── 1. SCHEMA & INITIALIZATION ───────────────────────────────────────────────

def test_memory_store_schema_initialization(memory_store, temp_db_path):
    """Verify that tables, FTS5 virtual tables, and triggers are created in SQLite WAL mode."""
    with memory_store._get_connection() as conn:
        # Check journal mode
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert journal_mode.lower() == "wal"

        # Check memories table exists
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
        assert "memories" in tables
        assert "memories_fts" in tables
        assert "tasks" in tables
        assert "tasks_fts" in tables

        # Check triggers exist
        triggers = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger';").fetchall()]
        assert "memories_ai" in triggers
        assert "memories_ad" in triggers
        assert "memories_au" in triggers
        assert "tasks_ai" in triggers
        assert "tasks_ad" in triggers
        assert "tasks_au" in triggers


# ── 2. FTS5 QUERY SANITIZATION & SECURITY ───────────────────────────────────

def test_fts_query_sanitization_adversarial_inputs():
    """Verify that raw user strings containing syntax-breaking characters do not crash FTS5."""
    assert sanitize_fts_query("") == ""
    assert sanitize_fts_query("   ") == ""

    # Unbalanced quotes, colons, wildcards, operators
    res = sanitize_fts_query('python: "hello" * (world)?')
    assert '"python"' in res
    assert '"hello"' in res
    assert '"world"*' in res

    # Pure punctuation / symbols
    assert sanitize_fts_query("!@#$%^&*()_+=-~`{}[]|;:,.<>?/") == ""

    # Reserved operators alone
    reserved_res = sanitize_fts_query("AND OR NOT NEAR")
    assert reserved_res != ""  # fallback prevents crash

    # Safe term building for search
    built = build_fts_query("What OS do I prefer for development?")
    assert "prefer" in built or "development" in built


# ── 3. REMEMBER & FTS5 BM25 SEARCH ──────────────────────────────────────────

def test_remember_and_fts_bm25_search(memory_store):
    """Verify adding memories and retrieving them with FTS5 BM25 ranking."""
    m1 = memory_store.remember(
        content="Operator strongly prefers Fedora Linux for backend Python engineering.",
        category=MemoryCategory.USER_PREFERENCE,
        tags=["os", "linux", "python"],
        importance=1.5,
    )
    m2 = memory_store.remember(
        content="Host workstation is equipped with an NVIDIA RTX 3050 Laptop GPU with 6GB VRAM.",
        category=MemoryCategory.SYSTEM_FACT,
        tags=["gpu", "nvidia", "hardware"],
        importance=2.0,
    )
    m3 = memory_store.remember(
        content="Operator occasionally drinks espresso in the early morning.",
        category=MemoryCategory.USER_PREFERENCE,
        tags=["coffee", "habits"],
        importance=0.8,
    )

    assert m1.startswith("mem_")
    assert m2.startswith("mem_")

    # Search for Linux OS preference
    results = memory_store.recall("Fedora Linux python")
    assert len(results) >= 1
    assert results[0]["id"] == m1
    assert "Fedora Linux" in results[0]["content"]
    assert "rank" in results[0]

    # Search for GPU hardware
    gpu_results = memory_store.recall("RTX 3050 GPU")
    assert len(gpu_results) >= 1
    assert gpu_results[0]["id"] == m2
    assert "RTX 3050" in gpu_results[0]["content"]


# ── 4. BI-TEMPORAL INVALIDATION & SUPERSEDING ────────────────────────────────

def test_bi_temporal_invalidation(memory_store):
    """Verify that updated/contradicted facts set invalid_at and are excluded from active search."""
    m_old = memory_store.remember(
        content="Operator primary browser is Google Chrome.",
        category=MemoryCategory.USER_PREFERENCE,
        tags=["browser"],
    )

    # Search finds old browser
    res1 = memory_store.recall("primary browser")
    assert len(res1) == 1
    assert res1[0]["id"] == m_old

    # Operator changes preference to Firefox: supersede old memory
    time.sleep(0.01)
    m_new = memory_store.supersede(
        old_memory_id=m_old,
        new_content="Operator switched primary browser to Mozilla Firefox for privacy.",
        category=MemoryCategory.USER_PREFERENCE,
        tags=["browser", "firefox"],
    )

    # Active search should now return only the new memory
    res2 = memory_store.recall("primary browser")
    assert len(res2) == 1
    assert res2[0]["id"] == m_new
    assert "Firefox" in res2[0]["content"]

    # With include_invalid=True, both should be visible
    all_res = memory_store.recall("primary browser", include_invalid=True)
    ids = [r["id"] for r in all_res]
    assert m_old in ids
    assert m_new in ids

    # Check old record has invalid_at set
    old_rec = memory_store.get_memory(m_old)
    assert old_rec is not None
    assert old_rec["invalid_at"] is not None


# ── 5. SOFT DELETION (FORGET) ────────────────────────────────────────────────

def test_forget_soft_deletion(memory_store):
    """Verify forget soft-deletes memory and cleans it from active FTS5 recall."""
    m_id = memory_store.remember(
        content="Temporary secret project codename is Project Chimera.",
        category=MemoryCategory.PROJECT,
        tags=["secret"],
    )

    # Verify searchable
    assert len(memory_store.recall("Project Chimera")) == 1

    # Forget memory
    success = memory_store.forget(m_id)
    assert success is True

    # No longer returned in active search
    assert len(memory_store.recall("Project Chimera")) == 0

    # Record still exists with is_deleted = 1
    raw = memory_store.get_memory(m_id)
    assert raw is not None
    assert raw["is_deleted"] == 1


# ── 6. TASK HISTORY FTS5 RETRIEVAL ───────────────────────────────────────────

def test_task_history_fts_retrieval(memory_store):
    """Verify completed tasks are synchronized to tasks_fts and retrievable via BM25."""
    with memory_store._get_connection() as conn:
        conn.execute(
            """INSERT INTO tasks (task_id, description, status, created_at, completed_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("task_101", "Inspect NVIDIA RTX 3050 driver state and CUDA telemetry", "completed", time.time() - 100, time.time() - 90, "{}"),
        )
        conn.execute(
            """INSERT INTO tasks (task_id, description, status, created_at, completed_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("task_102", "Compile Linux kernel Btrfs snapshot diagnostic report", "completed", time.time() - 50, time.time() - 40, "{}"),
        )
        conn.commit()

    # Search task history
    cuda_tasks = memory_store.search_task_history("CUDA telemetry NVIDIA")
    assert len(cuda_tasks) >= 1
    assert cuda_tasks[0]["task_id"] == "task_101"
    assert "CUDA telemetry" in cuda_tasks[0]["description"]

    btrfs_tasks = memory_store.search_task_history("Btrfs snapshot")
    assert len(btrfs_tasks) >= 1
    assert btrfs_tasks[0]["task_id"] == "task_102"


# ── 7. CONTEXT PRUNING & INJECTION BUDGETING ──────────────────────────────────

def test_retrieve_relevant_context_budgeting(memory_store):
    """Verify context injection formats markdown cleanly and strictly respects character limits."""
    # Seed memories
    memory_store.remember(
        content="Operator works on M.A.D.D.Y. AI assistant architecture.",
        category=MemoryCategory.PROJECT,
    )
    memory_store.remember(
        content="Operator workstation OS is Fedora 44 Linux with dual-boot Windows 11.",
        category=MemoryCategory.SYSTEM_FACT,
    )

    # Query with relevant query
    context = memory_store.retrieve_relevant_context("What workstation OS is M.A.D.D.Y. running on?", max_chars=400)
    assert "<recalled_context>" in context
    assert "</recalled_context>" in context
    assert "Fedora 44 Linux" in context
    assert len(context) <= 400 + 100  # including structural tags

    # Query with completely unrelated keywords should return empty string (graceful fallback)
    empty_context = memory_store.retrieve_relevant_context("quarks and gluons astrophysics")
    assert empty_context == ""


# ── 8. NATIVE MEMORY TOOLS VIA TOOL REGISTRY ─────────────────────────────────

def test_memory_tools_execution(memory_store):
    """Verify remember, recall, and forget tools execute cleanly via ToolRegistry."""
    registry = ToolRegistry()
    remember_tool = RememberTool(memory_store=memory_store)
    recall_tool = RecallTool(memory_store=memory_store)
    forget_tool = ForgetTool(memory_store=memory_store)

    registry.register_tool(remember_tool)
    registry.register_tool(recall_tool)
    registry.register_tool(forget_tool)

    assert remember_tool.tier == PermissionTier.SAFE
    assert recall_tool.tier == PermissionTier.SAFE
    assert forget_tool.tier == PermissionTier.SAFE

    # 1. Execute remember
    res1 = registry.execute_tool(
        "remember",
        {"content": "Preferred terminal font is JetBrains Mono.", "category": "user_preference", "tags": ["terminal", "fonts"]},
    )
    assert res1.success is True
    assert "Memory preserved" in res1.output
    # Extract ID
    mem_id = res1.output.split("[ID: ")[1].split("]")[0]

    # 2. Execute recall
    res2 = registry.execute_tool("recall", {"query": "terminal font"})
    assert res2.success is True
    assert "JetBrains Mono" in res2.output

    # 3. Execute forget
    res3 = registry.execute_tool("forget", {"memory_id": mem_id})
    assert res3.success is True
    assert "Memory forgotten" in res3.output

    # 4. Verify no longer recalled
    res4 = registry.execute_tool("recall", {"query": "terminal font"})
    assert res4.success is True
    assert "No relevant memories" in res4.output


# ── 9. AGENT LOOP RELEVANT CONTEXT INJECTION ────────────────────────────────

def test_agent_loop_context_injection_integration(temp_db_path):
    """Verify that MaddyAgentCore integrates MemoryStore and injects relevant context into the prompt."""
    from core.agent_loop import MaddyAgentCore
    from core.llm_client import CloudLLMClient, ModelResponse

    store = MemoryStore(db_path=temp_db_path)
    store.remember(
        content="Operator default code formatting indentation is strictly 4 spaces.",
        category=MemoryCategory.USER_PREFERENCE,
        tags=["formatting", "python"],
    )

    mock_llm = MagicMock(spec=CloudLLMClient)
    # Return mock response
    mock_llm.chat_completion.return_value = ModelResponse(
        content="Understood. I will strictly apply 4-space indentation per your stored preference.",
        tool_calls=[],
        finish_reason="stop",
        prompt_tokens=40,
        completion_tokens=20,
    )

    agent = MaddyAgentCore(
        llm_client=mock_llm,
        memory_store=store,
    )

    turn_res = agent.run_turn(user_prompt="What indentation style should we use for this Python file?")
    assert turn_res["completed"] is True

    # Inspect the messages passed to chat_completion
    call_args = mock_llm.chat_completion.call_args[1]
    passed_messages = call_args["messages"]

    # Verify that either the system prompt or user context contains the recalled memory
    all_content = " ".join(m.get("content", "") for m in passed_messages)
    assert "4 spaces" in all_content or "4-space" in all_content
    assert "<recalled_context>" in all_content
