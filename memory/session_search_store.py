"""
SQLite FTS5-backed Session Search & Historical Conversation Store for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).

Features:
- Full-text search over all past conversational turns with BM25 ranking.
- 4 operational query modes:
  1. Discovery: Search across sessions with query, snippet generation, and time bounds.
  2. Read: Retrieve sequential messages of a session with per-message content bounding.
  3. Scroll: Zoom in on context window around a specific anchor message ID.
  4. Browse: List recent conversational sessions with summaries and timestamps.
- Zero LLM inference overhead (pure deterministic SQLite indexing).
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import get_memory_dir

logger = logging.getLogger("memory.session_search_store")

_RELATIVE_BOUND_RE = re.compile(r"^(\d+)\s*(h|d|w|m)$", re.IGNORECASE)
_RELATIVE_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800}

_MAX_MESSAGE_CONTENT_READ = 2000
_MAX_SNIPPET_LENGTH = 200


def parse_time_bound(val: Optional[str]) -> Optional[float]:
    """Parses ISO timestamp or relative bound like '24h', '7d', '2w' into epoch timestamp."""
    if not val:
        return None
    val = val.strip()
    match = _RELATIVE_BOUND_RE.match(val)
    if match:
        amount, unit = int(match.group(1)), match.group(2).lower()
        sec = amount * _RELATIVE_UNIT_SECONDS[unit]
        return time.time() - sec

    # Try ISO formats
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(val, fmt).timestamp()
        except ValueError:
            continue
    return None


class SessionSearchStore:
    """
    Durable, thread-safe SQLite FTS5 database storing and indexing conversation history.
    """

    def __init__(self, db_path: Optional[Path | str] = None):
        if db_path:
            self.db_path = Path(db_path)
        else:
            self.db_path = Path(get_memory_dir()) / "sessions.db"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at REAL,
                    updated_at REAL,
                    message_count INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'interactive',
                    summary TEXT DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    content,
                    session_id UNINDEXED,
                    role UNINDEXED,
                    message_id UNINDEXED,
                    tokenize='porter unicode61'
                )
                """
            )
            conn.commit()

    def log_message(
        self,
        session_id: str,
        role: str,
        content: str,
        title: Optional[str] = None,
        source: str = "interactive",
        timestamp: Optional[float] = None,
    ) -> int:
        """Logs a single message and indexes it into FTS5."""
        if not content or not content.strip():
            return -1

        ts = timestamp or time.time()
        clean_content = content.strip()

        with self._lock, self._get_connection() as conn:
            # Check or create session row
            row = conn.execute(
                "SELECT message_count, title FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            if row:
                next_index = row["message_count"] + 1
                curr_title = row["title"] or title or f"Session {session_id[:8]}"
                conn.execute(
                    """
                    UPDATE sessions
                    SET updated_at = ?, message_count = ?, title = ?
                    WHERE session_id = ?
                    """,
                    (ts, next_index, curr_title, session_id),
                )
            else:
                next_index = 1
                sess_title = title or (clean_content[:60] + ("..." if len(clean_content) > 60 else ""))
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, title, created_at, updated_at, message_count, source)
                    VALUES (?, ?, ?, ?, 1, ?)
                    """,
                    (session_id, sess_title, ts, ts, source),
                )

            # Insert message
            cur = conn.execute(
                """
                INSERT INTO messages (session_id, message_index, role, content, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, next_index, role, clean_content, ts),
            )
            msg_id = cur.lastrowid

            # Index into FTS5
            conn.execute(
                """
                INSERT INTO messages_fts (content, session_id, role, message_id)
                VALUES (?, ?, ?, ?)
                """,
                (clean_content, session_id, role, msg_id),
            )
            conn.commit()
            return msg_id

    def discover(
        self,
        query: str,
        limit: int = 5,
        after: Optional[str] = None,
        before: Optional[str] = None,
        exclude_session_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        FTS5 full-text discovery over past sessions.
        Returns matching sessions ranked by BM25 relevance with snippets.
        """
        if not query or not query.strip():
            return self.browse(limit=limit)

        clean_query = query.strip()
        # Sanitize query for FTS5 (strip operators if syntax invalid)
        safe_query = re.sub(r'["\*\+\-\:\(\)]', " ", clean_query)
        safe_query = " ".join(safe_query.split())
        if not safe_query:
            return {"success": True, "results": [], "query": query, "total_matches": 0}

        after_ts = parse_time_bound(after)
        before_ts = parse_time_bound(before)
        excluded = set(exclude_session_ids or [])

        sql = """
            SELECT
                m.id as message_id,
                m.session_id,
                m.role,
                m.timestamp,
                s.title as session_title,
                s.source as session_source,
                s.created_at as session_created_at,
                snippet(messages_fts, 0, '<b>', '</b>', '...', 20) as snippet,
                rank
            FROM messages_fts f
            JOIN messages m ON f.message_id = m.id
            JOIN sessions s ON m.session_id = s.session_id
            WHERE messages_fts MATCH ?
        """
        params: List[Any] = [safe_query]

        if after_ts:
            sql += " AND m.timestamp >= ?"
            params.append(after_ts)
        if before_ts:
            sql += " AND m.timestamp <= ?"
            params.append(before_ts)

        sql += " ORDER BY rank LIMIT 100"

        with self._lock, self._get_connection() as conn:
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                # Fallback to phrase search
                rows = conn.execute(
                    sql.replace("messages_fts MATCH ?", "messages_fts MATCH ?"),
                    [f'"{safe_query}"'] + params[1:],
                ).fetchall()

        # Group and rank by session, avoiding duplicate hits per session
        seen_sessions: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            sess_id = r["session_id"]
            if sess_id in excluded:
                continue

            if sess_id not in seen_sessions:
                seen_sessions[sess_id] = {
                    "session_id": sess_id,
                    "session_title": r["session_title"],
                    "session_source": r["session_source"],
                    "session_date": datetime.fromtimestamp(r["session_created_at"]).strftime("%Y-%m-%d %H:%M"),
                    "top_match_message_id": r["message_id"],
                    "top_match_role": r["role"],
                    "snippet": r["snippet"],
                    "score": round(float(r["rank"]), 3),
                }

            if len(seen_sessions) >= limit:
                break

        results = list(seen_sessions.values())
        return {
            "success": True,
            "query": query,
            "total_matches": len(results),
            "results": results,
        }

    def read(self, session_id: str, limit: int = 50) -> Dict[str, Any]:
        """
        Reads all messages in a session in chronological order,
        bounding content length per message to protect token context.
        """
        with self._lock, self._get_connection() as conn:
            sess = conn.execute(
                "SELECT session_id, title, created_at, source, summary FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            if not sess:
                return {"success": False, "error": f"Session '{session_id}' not found."}

            rows = conn.execute(
                """
                SELECT id, message_index, role, content, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY message_index ASC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        messages = []
        for r in rows:
            raw_text = r["content"]
            if len(raw_text) > _MAX_MESSAGE_CONTENT_READ:
                text = raw_text[:_MAX_MESSAGE_CONTENT_READ] + f"\n... [Truncated: {len(raw_text):,} chars total]"
            else:
                text = raw_text

            messages.append(
                {
                    "message_id": r["id"],
                    "index": r["message_index"],
                    "role": r["role"],
                    "content": text,
                    "timestamp": datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        return {
            "success": True,
            "session_id": session_id,
            "title": sess["title"],
            "created_at": datetime.fromtimestamp(sess["created_at"]).strftime("%Y-%m-%d %H:%M:%S"),
            "total_messages": len(messages),
            "messages": messages,
        }

    def scroll(self, session_id: str, around_message_id: int, window: int = 5) -> Dict[str, Any]:
        """
        Scrolls context window centered on anchor message ID in a specific session.
        """
        with self._lock, self._get_connection() as conn:
            anchor = conn.execute(
                "SELECT message_index FROM messages WHERE id = ? AND session_id = ?",
                (around_message_id, session_id),
            ).fetchone()

            if not anchor:
                return {
                    "success": False,
                    "error": f"Message {around_message_id} not found in session '{session_id}'.",
                }

            anchor_idx = anchor["message_index"]
            start_idx = max(1, anchor_idx - window)
            end_idx = anchor_idx + window

            rows = conn.execute(
                """
                SELECT id, message_index, role, content, timestamp
                FROM messages
                WHERE session_id = ? AND message_index BETWEEN ? AND ?
                ORDER BY message_index ASC
                """,
                (session_id, start_idx, end_idx),
            ).fetchall()

        messages = []
        for r in rows:
            messages.append(
                {
                    "message_id": r["id"],
                    "index": r["message_index"],
                    "role": r["role"],
                    "is_anchor": r["id"] == around_message_id,
                    "content": r["content"][:_MAX_MESSAGE_CONTENT_READ],
                    "timestamp": datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        return {
            "success": True,
            "session_id": session_id,
            "anchor_message_id": around_message_id,
            "window": window,
            "messages": messages,
        }

    def browse(self, limit: int = 10) -> Dict[str, Any]:
        """Lists recent conversational sessions sorted by latest activity."""
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT session_id, title, created_at, updated_at, message_count, source
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        sessions = []
        for r in rows:
            sessions.append(
                {
                    "session_id": r["session_id"],
                    "title": r["title"],
                    "message_count": r["message_count"],
                    "source": r["source"],
                    "updated_at": datetime.fromtimestamp(r["updated_at"]).strftime("%Y-%m-%d %H:%M"),
                }
            )

        return {
            "success": True,
            "total_sessions": len(sessions),
            "sessions": sessions,
        }
