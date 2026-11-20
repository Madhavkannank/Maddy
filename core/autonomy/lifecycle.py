"""Lifecycle Guard and Crash Recovery Manager for M.A.D.D.Y. Background Autonomy.

Responsible for zombie / orphan task reconciliation upon daemon restarts,
heartbeat tracking, and state transition auditing.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from core.autonomy.models import JobExecutionRecord, JobStatus, ScheduledJob
from core.autonomy.repository import JobRepository

logger = logging.getLogger(__name__)


class LifecycleGuard:
    """Monitors job lifecycle states and performs crash reconciliation on startup."""

    def __init__(self, repository: JobRepository):
        self.repository = repository
        self._active_heartbeats: Dict[str, float] = {}

    def reconcile_orphans(self) -> List[ScheduledJob]:
        """Detect and reconcile jobs stranded in RUNNING state due to daemon crashes or unexpected reboots.

        Returns:
            List of recovered jobs updated in repository.
        """
        stale_jobs = self.repository.get_stale_running_jobs()
        recovered: List[ScheduledJob] = []

        now = time.time()
        for job in stale_jobs:
            logger.warning(
                "Orphaned background task detected on startup: job_id='%s' (%s). Reconciling to RECOVERED.",
                job.id,
                job.name,
            )
            # 1. Finalize any open execution records
            executions = self.repository.get_executions_for_job(job.id, limit=1)
            if executions and executions[0].status == "RUNNING":
                self.repository.record_execution_finish(
                    execution_id=executions[0].execution_id,
                    completed_at=now,
                    status="CRASHED_RECOVERED",
                    error="Daemon process terminated abruptly during active execution.",
                )

            # 2. Reset job status and reschedule
            job.last_status = "CRASHED_RECOVERED"
            job.status = JobStatus.RECOVERED
            job.last_error = "Previous execution interrupted by process shutdown."
            job.next_run_at = job.calculate_next_run(from_time=now)

            self.repository.save_job(job)
            recovered.append(job)

        if recovered:
            logger.info("Successfully reconciled %d orphaned background job(s).", len(recovered))
        return recovered

    def start_heartbeat(self, execution_id: str) -> None:
        """Register the start of an active execution heartbeat."""
        self._active_heartbeats[execution_id] = time.time()

    def update_heartbeat(self, execution_id: str) -> None:
        """Update last seen heartbeat timestamp for an active execution."""
        self._active_heartbeats[execution_id] = time.time()

    def clear_heartbeat(self, execution_id: str) -> None:
        """Remove heartbeat upon execution completion."""
        self._active_heartbeats.pop(execution_id, None)

    def is_stalled(self, execution_id: str, timeout_seconds: float) -> bool:
        """Determine if an execution has stalled past its allowed timeout."""
        last_hb = self._active_heartbeats.get(execution_id)
        if not last_hb:
            return False
        return (time.time() - last_hb) > timeout_seconds
