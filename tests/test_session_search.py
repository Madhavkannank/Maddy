"""Tests for SQLite FTS5 Session Search & Historical Conversation Retrieval in M.A.D.D.Y.

Verifies:
1. FTS5 full-text indexing and BM25 discovery ranking with snippets.
2. Chronological session transcript reading with content length bounding.
3. Context window scrolling around an anchor message ID.
4. Recent sessions browsing without arguments.
5. Relative and ISO time bound filtering ('24h', '7d').
6. ToolRegistry and ToolExecutor integration.
7. MaddyAgent multi-turn recall via session_search tool.
"""
import shutil
import tempfile
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from agents.base import AgentStatus
from agents.maddy_agent import MaddyAgent
from memory.session_search_store import SessionSearchStore
from models.base import ModelProvider, ModelResponse
from tools.base import Tool
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry
from tools.session_search_tool import (
    create_session_search_tool,
    session_search_tool,
    set_session_search_store,
)


class MockProvider(ModelProvider):
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, prompt, **kwargs):
        if self.responses:
            return ModelResponse(content=self.responses.pop(0))
        return ModelResponse(content="Done.")

    def stream(self, prompt, **kwargs):
        yield self.generate(prompt).content

    def structured(self, prompt, schema, **kwargs):
        return {}


@pytest.fixture
def temp_db():
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_sessions.db"
    store = SessionSearchStore(db_path=db_path)
    set_session_search_store(store)
    yield store
    set_session_search_store(None)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_fts5_discovery_and_snippets(temp_db):
    """Verify FTS5 full-text indexing finds matching sessions with snippets."""
    # Seed 3 distinct sessions
    temp_db.log_message("sess_auth", "user", "How do we refactor the JWT auth tokens?", title="Auth Refactor")
    temp_db.log_message("sess_auth", "assistant", "We can use asymmetric RSA keys with RS256 algorithm.")

    temp_db.log_message("sess_vision", "user", "Inspect the bridge pillar for concrete spalling.", title="Defect Inspection")
    temp_db.log_message("sess_vision", "assistant", "Identified 3 crack lines and severe spalling on the cap.")

    temp_db.log_message("sess_music", "user", "Play some rock music on Spotify.", title="Spotify Playlist")
    temp_db.log_message("sess_music", "assistant", "Queuing up classic rock favorites.")

    # 1. Search for 'spalling'
    res = temp_db.discover("spalling")
    assert res["success"] is True
    assert res["total_matches"] == 1
    hit = res["results"][0]
    assert hit["session_id"] == "sess_vision"
    assert "spalling" in hit["snippet"].lower()

    # 2. Search for 'auth tokens'
    res2 = temp_db.discover("auth tokens")
    assert res2["success"] is True
    assert res2["total_matches"] == 1
    assert res2["results"][0]["session_id"] == "sess_auth"


def test_read_session_and_truncation(temp_db):
    """Verify read mode returns sequential messages and bounds long outputs."""
    temp_db.log_message("sess_read", "user", "Step 1: check files.")
    temp_db.log_message("sess_read", "assistant", "Found files A, B, C.")
    long_content = "X" * 3500
    temp_db.log_message("sess_read", "tool", long_content)

    res = temp_db.read("sess_read")
    assert res["success"] is True
    assert res["total_messages"] == 3
    msgs = res["messages"]
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    # Long content must be truncated to protect context window
    assert "[Truncated:" in msgs[2]["content"]
    assert len(msgs[2]["content"]) < 2500


def test_scroll_window_around_anchor(temp_db):
    """Verify scroll mode returns centered window around anchor message ID."""
    m_ids = []
    for i in range(1, 11):
        m_id = temp_db.log_message("sess_scroll", "user" if i % 2 == 1 else "assistant", f"Message number {i}")
        m_ids.append(m_id)

    anchor_id = m_ids[5]  # Message 6
    res = temp_db.scroll("sess_scroll", around_message_id=anchor_id, window=2)
    assert res["success"] is True
    messages = res["messages"]
    # Window of 2 means 2 before + anchor + 2 after = 5 messages (4, 5, 6, 7, 8)
    assert len(messages) == 5
    assert any(m["message_id"] == anchor_id and m["is_anchor"] is True for m in messages)
    assert messages[0]["content"] == "Message number 4"
    assert messages[-1]["content"] == "Message number 8"


def test_browse_recent_sessions(temp_db):
    """Verify browse returns sessions sorted by recent activity."""
    temp_db.log_message("sess_1", "user", "First session", title="Alpha")
    time.sleep(0.01)
    temp_db.log_message("sess_2", "user", "Second session", title="Beta")

    res = temp_db.browse(limit=10)
    assert res["success"] is True
    assert res["total_sessions"] == 2
    # sess_2 was logged after sess_1, so it must be first
    assert res["sessions"][0]["session_id"] == "sess_2"
    assert res["sessions"][1]["session_id"] == "sess_1"


def test_time_bounds_and_exclusions(temp_db):
    """Verify time filters and exclude_session_ids filtering."""
    now = time.time()
    # Log an old message (10 days ago) and a fresh message (now)
    temp_db.log_message("sess_old", "user", "Old database migration guide", timestamp=now - (10 * 86400))
    temp_db.log_message("sess_new", "user", "Fresh database migration script", timestamp=now)

    # Search with after='7d' (only fresh should appear)
    res = temp_db.discover("database migration", after="7d")
    assert res["total_matches"] == 1
    assert res["results"][0]["session_id"] == "sess_new"

    # Search with exclude_session_ids
    res_exc = temp_db.discover("database migration", exclude_session_ids=["sess_new"])
    assert res_exc["total_matches"] == 1
    assert res_exc["results"][0]["session_id"] == "sess_old"


def test_maddy_agent_recalls_past_session_via_tool(temp_db):
    """Verify MaddyAgent autonomously uses session_search tool to answer a recall question."""
    # Seed historical conversation
    temp_db.log_message(
        "sess_historical",
        "user",
        "What was the port we decided to use for the FastAPI server?",
        title="Server Architecture",
    )
    temp_db.log_message(
        "sess_historical",
        "assistant",
        "We agreed to bind FastAPI to port 8090 on localhost.",
    )

    registry = ToolRegistry()
    registry.register(create_session_search_tool(temp_db))
    executor = ToolExecutor(registry=registry)

    # Step 1: Model calls session_search
    # Step 2: Model answers using retrieved port
    responses = [
        '<thought>The user is asking about a past discussion. Let me search past sessions.</thought>\n<tool_call>{"name": "session_search", "arguments": {"query": "FastAPI server port"}}</tool_call>',
        '<thought>I see we agreed on port 8090.</thought>\nWe decided to bind the FastAPI server to port 8090, boss.',
    ]
    model = MockProvider(responses)

    agent = MaddyAgent(
        model_provider=model,
        executor=executor,
        registry=registry,
        max_steps=5,
        address_title="boss",
    )

    result = agent.run("What port did we choose for FastAPI last time?")
    assert result.status == AgentStatus.SUCCESS
    assert "8090" in result.final_answer
    assert "boss" in result.final_answer
