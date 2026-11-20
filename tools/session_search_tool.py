"""
Session Search Tool for M.A.D.D.Y.
Exposes the single unified `session_search` tool for full-text searching past conversations,
reading past session transcripts, and scrolling contextual message windows using SQLite FTS5.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from memory.session_search_store import SessionSearchStore
from tools.base import RiskLevel, Tool

logger = logging.getLogger("tools.session_search_tool")

_GLOBAL_SESSION_STORE: Optional[SessionSearchStore] = None


def get_session_search_store() -> SessionSearchStore:
    """Returns or creates the process-wide SessionSearchStore singleton."""
    global _GLOBAL_SESSION_STORE
    if _GLOBAL_SESSION_STORE is None:
        _GLOBAL_SESSION_STORE = SessionSearchStore()
    return _GLOBAL_SESSION_STORE


def set_session_search_store(store: Optional[SessionSearchStore]) -> None:
    """Overrides the global session search store (used for test isolation)."""
    global _GLOBAL_SESSION_STORE
    _GLOBAL_SESSION_STORE = store


SESSION_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Search term or keywords to discover matching past conversations via FTS5. "
                "Returns matching sessions, relevance scores, and highlighted snippets."
            ),
        },
        "session_id": {
            "type": "string",
            "description": (
                "Specific session ID to inspect. When passed alone, reads the session transcript. "
                "When paired with around_message_id, scrolls window around that message."
            ),
        },
        "around_message_id": {
            "type": "integer",
            "description": "Message ID to center context window around (scroll mode). Requires session_id.",
        },
        "window": {
            "type": "integer",
            "description": "Number of messages before and after around_message_id to include (default 5).",
        },
        "limit": {
            "type": "integer",
            "description": "Max results to return (default 5 for search, 50 for read, 10 for browse).",
        },
        "after": {
            "type": "string",
            "description": "Time filter lower bound (ISO date '2026-10-01' or relative '7d', '24h', '2w').",
        },
        "before": {
            "type": "string",
            "description": "Time filter upper bound (ISO date or relative).",
        },
        "exclude_session_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Session IDs to exclude from search results.",
        },
    },
    "required": [],
}


def session_search_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified router for session search operations:
    1. SCROLL: session_id + around_message_id
    2. READ: session_id alone
    3. DISCOVER: query provided
    4. BROWSE: no arguments provided
    """
    store = get_session_search_store()

    query = params.get("query")
    session_id = params.get("session_id")
    around_msg_id = params.get("around_message_id")
    window = params.get("window", 5)
    limit = params.get("limit")
    after = params.get("after")
    before = params.get("before")
    exclude_ids = params.get("exclude_session_ids")

    # Shape 1: SCROLL
    if session_id and around_msg_id is not None:
        return store.scroll(session_id=session_id, around_message_id=int(around_msg_id), window=int(window))

    # Shape 2: READ
    if session_id:
        read_limit = int(limit) if limit else 50
        return store.read(session_id=session_id, limit=read_limit)

    # Shape 3: DISCOVERY
    if query and query.strip():
        search_limit = int(limit) if limit else 5
        return store.discover(
            query=query.strip(),
            limit=search_limit,
            after=after,
            before=before,
            exclude_session_ids=exclude_ids,
        )

    # Shape 4: BROWSE
    browse_limit = int(limit) if limit else 10
    return store.browse(limit=browse_limit)


session_search_tool = Tool(
    name="session_search",
    description=(
        "Search past conversations stored in the local SQLite database using FTS5, "
        "or inspect a specific past session. Four operational modes:\n"
        "1. DISCOVER: Pass 'query' (with optional 'after'/'before' time bounds) to find past discussions.\n"
        "2. READ: Pass 'session_id' alone to read the chronological messages of a conversation.\n"
        "3. SCROLL: Pass 'session_id' and 'around_message_id' to zoom into context around a specific turn.\n"
        "4. BROWSE: Call with no arguments to see recent conversation sessions and dates."
    ),
    parameter_schema=SESSION_SEARCH_SCHEMA,
    handler=session_search_handler,
    risk_level=RiskLevel.LOW,
)


def create_session_search_tool(store: SessionSearchStore) -> Tool:
    """Convenience factory returning an isolated session_search Tool instance."""
    return Tool(
        name="session_search",
        description=session_search_tool.description,
        parameter_schema=SESSION_SEARCH_SCHEMA,
        handler=lambda params: _handle_with_store(store, params),
        risk_level=RiskLevel.LOW,
    )


def _handle_with_store(store: SessionSearchStore, params: Dict[str, Any]) -> Dict[str, Any]:
    query = params.get("query")
    session_id = params.get("session_id")
    around_msg_id = params.get("around_message_id")
    window = params.get("window", 5)
    limit = params.get("limit")
    after = params.get("after")
    before = params.get("before")
    exclude_ids = params.get("exclude_session_ids")

    if session_id and around_msg_id is not None:
        return store.scroll(session_id=session_id, around_message_id=int(around_msg_id), window=int(window))
    if session_id:
        return store.read(session_id=session_id, limit=int(limit) if limit else 50)
    if query and query.strip():
        return store.discover(
            query=query.strip(),
            limit=int(limit) if limit else 5,
            after=after,
            before=before,
            exclude_session_ids=exclude_ids,
        )
    return store.browse(limit=int(limit) if limit else 10)
