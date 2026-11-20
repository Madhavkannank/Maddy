"""SQLite WAL-mode execution trace store for M.A.D.D.Y. Agent Core.
(Modular Autonomous Digital Daemon for You)

Fulfills CLAUDE.md Engineering Rule 5:
"Persist execution traces in SQLite: tasks, steps, tool calls, events, and checkpoints."
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.redact import redact_structure, redact_text


SCHEMA_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    created_at REAL NOT NULL,
    completed_at REAL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS steps (
    step_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    thought TEXT,
    action_type TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('planned', 'executing', 'completed', 'failed')),
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_calls (
    tool_call_id TEXT PRIMARY KEY,
    step_id TEXT REFERENCES steps(step_id) ON DELETE CASCADE,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    tool_name TEXT NOT NULL,
    arguments TEXT NOT NULL,
    tier TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('planned', 'running', 'completed', 'failed', 'blocked')),
    start_time REAL NOT NULL,
    end_time REAL,
    duration_ms REAL,
    result_output TEXT,
    error_info TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    task_id TEXT REFERENCES tasks(task_id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    state_snapshot TEXT NOT NULL,
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    timestamp REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_steps_task ON steps(task_id, step_index);
CREATE INDEX IF NOT EXISTS idx_tool_calls_task ON tool_calls(task_id);
CREATE INDEX IF NOT EXISTS idx_messages_task ON messages(task_id);
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "logs" / "maddy_traces.db"


class TraceStore:
    """Manages transactional durability and execution tracing in SQLite with WAL mode."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(SCHEMA_DDL)
            conn.commit()

    def create_task(self, description: str, metadata: Optional[Dict[str, Any]] = None, task_id: Optional[str] = None) -> str:
        tid = task_id or f"task_{uuid.uuid4().hex[:12]}"
        now = time.time()
        meta_json = json.dumps(redact_structure(metadata or {}))
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO tasks (task_id, description, status, created_at, metadata) VALUES (?, ?, ?, ?, ?)",
                (tid, redact_text(description), "running", now, meta_json),
            )
            conn.commit()
        return tid

    def update_task_status(self, task_id: str, status: str) -> None:
        now = time.time() if status in ("completed", "failed") else None
        with self._get_connection() as conn:
            if status == "running":
                conn.execute(
                    "UPDATE tasks SET status = ?, completed_at = NULL WHERE task_id = ?",
                    (status, task_id),
                )
            else:
                conn.execute(
                    "UPDATE tasks SET status = ?, completed_at = ? WHERE task_id = ?",
                    (status, now, task_id),
                )
            conn.commit()

    def record_step(self, task_id: str, step_index: int, thought: str, action_type: str, status: str = "completed") -> str:
        step_id = f"step_{task_id}_{step_index}_{uuid.uuid4().hex[:6]}"
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO steps (step_id, task_id, step_index, thought, action_type, status, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (step_id, task_id, step_index, redact_text(thought), action_type, status, now),
            )
            conn.commit()
        return step_id

    # ── PERSIST-BEFORE-EXECUTE INVARIANT ──────────────────────────────────────
    def record_tool_call_planned(
        self,
        task_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        tier: str,
        step_id: Optional[str] = None,
    ) -> None:
        """Persist tool call BEFORE invocation commences to guarantee audit durability."""
        now = time.time()
        sanitized_args = json.dumps(redact_structure(arguments))
        with self._get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO tool_calls
                   (tool_call_id, step_id, task_id, tool_name, arguments, tier, status, start_time, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'planned', ?, ?)""",
                (tool_call_id, step_id, task_id, tool_name, sanitized_args, tier, now, now),
            )
            conn.commit()

    def record_tool_call_completed(
        self,
        tool_call_id: str,
        result_output: str,
        error_info: Optional[str] = None,
        status: str = "completed",
    ) -> None:
        """Update tool call record with execution result, duration, and final status."""
        now = time.time()
        with self._get_connection() as conn:
            row = conn.execute("SELECT start_time FROM tool_calls WHERE tool_call_id = ?", (tool_call_id,)).fetchone()
            start_time = row["start_time"] if row else now
            duration_ms = (now - start_time) * 1000.0

            conn.execute(
                """UPDATE tool_calls
                   SET status = ?, end_time = ?, duration_ms = ?, result_output = ?, error_info = ?
                   WHERE tool_call_id = ?""",
                (
                    status,
                    now,
                    duration_ms,
                    redact_text(result_output[:10000] if result_output else ""),
                    redact_text(error_info) if error_info else None,
                    tool_call_id,
                ),
            )
            conn.commit()

    def record_event(self, task_id: str, event_type: str, payload: Dict[str, Any]) -> str:
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO events (event_id, task_id, event_type, payload, timestamp) VALUES (?, ?, ?, ?, ?)",
                (event_id, task_id, event_type, json.dumps(redact_structure(payload)), now),
            )
            conn.commit()
        return event_id

    def save_checkpoint(self, task_id: str, step_index: int, state_snapshot: Dict[str, Any]) -> str:
        checkpoint_id = f"chk_{task_id}_{step_index}_{uuid.uuid4().hex[:6]}"
        now = time.time()
        if isinstance(state_snapshot, str):
            snapshot_json = redact_text(state_snapshot)
        else:
            snapshot_json = json.dumps(redact_structure(state_snapshot))
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO checkpoints (checkpoint_id, task_id, step_index, state_snapshot, timestamp) VALUES (?, ?, ?, ?, ?)",
                (checkpoint_id, task_id, step_index, snapshot_json, now),
            )
            conn.commit()
        return checkpoint_id

    def get_latest_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the most recent checkpoint for a task."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE task_id = ? ORDER BY step_index DESC, timestamp DESC LIMIT 1",
                (task_id,),
            ).fetchone()
            if not row:
                return None
            res = dict(row)
            if isinstance(res.get("state_snapshot"), str):
                try:
                    res["state_snapshot"] = json.loads(res["state_snapshot"])
                except Exception:
                    pass
            return res

    def list_checkpoints(self, task_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a task ordered chronologically by step index."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM checkpoints WHERE task_id = ? ORDER BY step_index ASC, timestamp ASC",
                (task_id,),
            ).fetchall()
            checkpoints = []
            for r in rows:
                item = dict(r)
                if isinstance(item.get("state_snapshot"), str):
                    try:
                        item["state_snapshot"] = json.loads(item["state_snapshot"])
                    except Exception:
                        pass
                checkpoints.append(item)
            return checkpoints

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific checkpoint by checkpoint_id."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,),
            ).fetchone()
            if not row:
                return None
            res = dict(row)
            if isinstance(res.get("state_snapshot"), str):
                try:
                    res["state_snapshot"] = json.loads(res["state_snapshot"])
                except Exception:
                    pass
            return res

    def append_message(
        self,
        task_id: str,
        role: str,
        content: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        now = time.time()
        tc_json = json.dumps(redact_structure(tool_calls)) if tool_calls else None
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO messages (task_id, role, content, tool_call_id, tool_calls, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, role, redact_text(content) if content else None, tool_call_id, tc_json, now),
            )
            conn.commit()

    def get_task_trace(self, task_id: str) -> Dict[str, Any]:
        """Fetch full diagnostic execution trace for a task."""
        with self._get_connection() as conn:
            task = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            if not task:
                return {}
            steps = [dict(r) for r in conn.execute("SELECT * FROM steps WHERE task_id = ? ORDER BY step_index", (task_id,)).fetchall()]
            tool_calls = [dict(r) for r in conn.execute("SELECT * FROM tool_calls WHERE task_id = ? ORDER BY start_time", (task_id,)).fetchall()]
            events = [dict(r) for r in conn.execute("SELECT * FROM events WHERE task_id = ? ORDER BY timestamp", (task_id,)).fetchall()]
            messages = [dict(r) for r in conn.execute("SELECT * FROM messages WHERE task_id = ? ORDER BY id", (task_id,)).fetchall()]
            checkpoints = self.list_checkpoints(task_id)

            return {
                "task": dict(task),
                "steps": steps,
                "tool_calls": tool_calls,
                "events": events,
                "messages": messages,
                "checkpoints": checkpoints,
            }

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Fetch single task record by task_id."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            return dict(row) if row else None

    def get_task_tool_calls(self, task_id: str) -> List[Dict[str, Any]]:
        """Fetch all tool call records for a task ordered by start time."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM tool_calls WHERE task_id = ? ORDER BY start_time ASC",
                (task_id,),
            ).fetchall()
            return [dict(r) for r in rows]
