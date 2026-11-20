"""SQLite-Backed Persistent Repository for M.A.D.D.Y. Scheduled Jobs and Autonomy Traces.

Thread-safe, WAL-enabled storage for job specifications and historical execution traces.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.autonomy.models import JobExecutionRecord, JobStatus, JobTriggerType, ScheduledJob

logger = logging.getLogger(__name__)

DEFAULT_AUTONOMY_DB_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "autonomy.db"


class JobRepository:
    """Thread-safe SQLite storage for scheduled jobs and historical execution audit trails."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_AUTONOMY_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or initialize thread-local SQLite connection with WAL mode."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            # Enable WAL mode and foreign keys for high-concurrency reliability
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA busy_timeout=5000;")
                conn.execute("PRAGMA foreign_keys=ON;")
            except sqlite3.OperationalError as exc:
                logger.warning("Failed to set WAL pragma on %s: %s", self.db_path, exc)
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        conn = self._get_connection()
        with conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS scheduled_jobs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    interval_seconds REAL,
                    time_of_day TEXT,
                    target_epoch REAL,
                    cron_expression TEXT,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    last_run_at REAL,
                    next_run_at REAL,
                    run_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'IDLE',
                    last_status TEXT NOT NULL DEFAULT 'IDLE',
                    last_result TEXT,
                    last_error TEXT,
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    failure_threshold INTEGER NOT NULL DEFAULT 3,
                    backoff_multiplier REAL NOT NULL DEFAULT 2.0,
                    max_retries INTEGER NOT NULL DEFAULT 3,
                    timeout_seconds REAL NOT NULL DEFAULT 120.0,
                    deliver_voice INTEGER NOT NULL DEFAULT 1,
                    context TEXT,
                    required_tools TEXT,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS job_executions (
                    execution_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    completed_at REAL,
                    duration_seconds REAL NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL DEFAULT 'RUNNING',
                    result_summary TEXT,
                    error TEXT,
                    total_steps INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT,
                    FOREIGN KEY (job_id) REFERENCES scheduled_jobs(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_status ON scheduled_jobs(status);
                CREATE INDEX IF NOT EXISTS idx_jobs_next_run ON scheduled_jobs(next_run_at);
                CREATE INDEX IF NOT EXISTS idx_executions_job_id ON job_executions(job_id);
                CREATE INDEX IF NOT EXISTS idx_executions_started ON job_executions(started_at);
                """
            )

    def save_job(self, job: ScheduledJob) -> None:
        """Insert or upsert a ScheduledJob in SQLite without triggering cascade deletions."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO scheduled_jobs (
                    id, name, goal, trigger_type, interval_seconds, time_of_day,
                    target_epoch, cron_expression, enabled, created_at, last_run_at,
                    next_run_at, run_count, status, last_status, last_result,
                    last_error, consecutive_failures, failure_threshold,
                    backoff_multiplier, max_retries, timeout_seconds, deliver_voice,
                    context, required_tools, metadata
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    goal = excluded.goal,
                    trigger_type = excluded.trigger_type,
                    interval_seconds = excluded.interval_seconds,
                    time_of_day = excluded.time_of_day,
                    target_epoch = excluded.target_epoch,
                    cron_expression = excluded.cron_expression,
                    enabled = excluded.enabled,
                    created_at = excluded.created_at,
                    last_run_at = excluded.last_run_at,
                    next_run_at = excluded.next_run_at,
                    run_count = excluded.run_count,
                    status = excluded.status,
                    last_status = excluded.last_status,
                    last_result = excluded.last_result,
                    last_error = excluded.last_error,
                    consecutive_failures = excluded.consecutive_failures,
                    failure_threshold = excluded.failure_threshold,
                    backoff_multiplier = excluded.backoff_multiplier,
                    max_retries = excluded.max_retries,
                    timeout_seconds = excluded.timeout_seconds,
                    deliver_voice = excluded.deliver_voice,
                    context = excluded.context,
                    required_tools = excluded.required_tools,
                    metadata = excluded.metadata;
                """,
                (
                    job.id,
                    job.name,
                    job.goal,
                    job.trigger_type.value,
                    job.interval_seconds,
                    job.time_of_day,
                    job.target_epoch,
                    job.cron_expression,
                    1 if job.enabled else 0,
                    job.created_at,
                    job.last_run_at,
                    job.next_run_at,
                    job.run_count,
                    job.status.value,
                    job.last_status,
                    job.last_result,
                    job.last_error,
                    job.consecutive_failures,
                    job.failure_threshold,
                    job.backoff_multiplier,
                    job.max_retries,
                    job.timeout_seconds,
                    1 if job.deliver_voice else 0,
                    job.context,
                    json.dumps(job.required_tools) if job.required_tools else None,
                    json.dumps(job.metadata) if job.metadata else "{}",
                ),
            )

    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Retrieve a ScheduledJob by its ID."""
        conn = self._get_connection()
        cur = conn.execute("SELECT * FROM scheduled_jobs WHERE id = ?;", (job_id,))
        row = cur.fetchone()
        if not row:
            return None
        return self._row_to_job(row)

    def list_jobs(self, enabled_only: bool = False) -> List[ScheduledJob]:
        """List all registered jobs, optionally filtering by enabled status."""
        conn = self._get_connection()
        if enabled_only:
            cur = conn.execute("SELECT * FROM scheduled_jobs WHERE enabled = 1 ORDER BY next_run_at ASC;")
        else:
            cur = conn.execute("SELECT * FROM scheduled_jobs ORDER BY created_at ASC;")
        return [self._row_to_job(r) for r in cur.fetchall()]

    def delete_job(self, job_id: str) -> bool:
        """Delete a job by ID."""
        conn = self._get_connection()
        with conn:
            cur = conn.execute("DELETE FROM scheduled_jobs WHERE id = ?;", (job_id,))
            return cur.rowcount > 0

    def get_stale_running_jobs(self) -> List[ScheduledJob]:
        """Identify jobs that were left in RUNNING status during unexpected process shutdown."""
        conn = self._get_connection()
        cur = conn.execute("SELECT * FROM scheduled_jobs WHERE status = 'RUNNING';")
        return [self._row_to_job(r) for r in cur.fetchall()]

    def record_execution_start(
        self,
        execution_id: str,
        job_id: str,
        started_at: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> JobExecutionRecord:
        """Record the start of a job execution."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO job_executions (
                    execution_id, job_id, started_at, status, metadata
                ) VALUES (?, ?, ?, 'RUNNING', ?);
                """,
                (
                    execution_id,
                    job_id,
                    started_at,
                    json.dumps(metadata or {}),
                ),
            )
        return JobExecutionRecord(
            execution_id=execution_id,
            job_id=job_id,
            started_at=started_at,
            status="RUNNING",
            metadata=metadata or {},
        )

    def record_execution_finish(
        self,
        execution_id: str,
        completed_at: float,
        status: str,
        result_summary: Optional[str] = None,
        error: Optional[str] = None,
        total_steps: int = 0,
    ) -> None:
        """Finalize an execution record with duration, status, and outcome summary."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                UPDATE job_executions
                SET completed_at = ?,
                    duration_seconds = max(0.0, ? - started_at),
                    status = ?,
                    result_summary = ?,
                    error = ?,
                    total_steps = ?
                WHERE execution_id = ?;
                """,
                (
                    completed_at,
                    completed_at,
                    status,
                    result_summary,
                    error,
                    total_steps,
                    execution_id,
                ),
            )

    def get_executions_for_job(self, job_id: str, limit: int = 10) -> List[JobExecutionRecord]:
        """Retrieve recent execution history for a specific job."""
        conn = self._get_connection()
        cur = conn.execute(
            """
            SELECT * FROM job_executions
            WHERE job_id = ?
            ORDER BY started_at DESC
            LIMIT ?;
            """,
            (job_id, limit),
        )
        return [self._row_to_execution(r) for r in cur.fetchall()]

    def _row_to_job(self, row: sqlite3.Row) -> ScheduledJob:
        """Convert a SQLite row into a typed ScheduledJob."""
        req_tools = json.loads(row["required_tools"]) if row["required_tools"] else None
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        return ScheduledJob(
            id=row["id"],
            name=row["name"],
            goal=row["goal"],
            trigger_type=JobTriggerType(row["trigger_type"]),
            interval_seconds=row["interval_seconds"],
            time_of_day=row["time_of_day"],
            target_epoch=row["target_epoch"],
            cron_expression=row["cron_expression"],
            enabled=bool(row["enabled"]),
            created_at=row["created_at"],
            last_run_at=row["last_run_at"],
            next_run_at=row["next_run_at"],
            run_count=row["run_count"],
            status=JobStatus(row["status"]),
            last_status=row["last_status"],
            last_result=row["last_result"],
            last_error=row["last_error"],
            consecutive_failures=row["consecutive_failures"],
            failure_threshold=row["failure_threshold"],
            backoff_multiplier=row["backoff_multiplier"],
            max_retries=row["max_retries"],
            timeout_seconds=row["timeout_seconds"],
            deliver_voice=bool(row["deliver_voice"]),
            context=row["context"],
            required_tools=req_tools,
            metadata=meta,
        )

    def _row_to_execution(self, row: sqlite3.Row) -> JobExecutionRecord:
        """Convert a SQLite row into a typed JobExecutionRecord."""
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        return JobExecutionRecord(
            execution_id=row["execution_id"],
            job_id=row["job_id"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            duration_seconds=row["duration_seconds"],
            status=row["status"],
            result_summary=row["result_summary"],
            error=row["error"],
            total_steps=row["total_steps"],
            metadata=meta,
        )
