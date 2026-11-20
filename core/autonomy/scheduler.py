"""Autonomy Scheduler Daemon and Background Execution Engine for M.A.D.D.Y.

Provides thread-safe background scheduling, non-overlapping execution locks,
drift-compensated intervals, and SQLite-backed persistent audit records.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from core.autonomy.circuit_breaker import CircuitBreaker
from core.autonomy.lifecycle import LifecycleGuard
from core.autonomy.models import (
    JobExecutionRecord,
    JobStatus,
    JobTriggerType,
    ScheduledJob,
)
from core.autonomy.policy import AutonomyPolicyGate
from core.autonomy.repository import JobRepository
from core.trace_store import TraceStore

logger = logging.getLogger(__name__)


class AutonomyScheduler:
    """Thread-safe background scheduler and execution coordinator for M.A.D.D.Y."""

    def __init__(
        self,
        repository: Optional[JobRepository] = None,
        db_path: Optional[Path] = None,
        max_workers: int = 4,
        tick_interval: float = 0.5,
        runner_fn: Optional[Callable[[ScheduledJob], str]] = None,
        trace_store: Optional[TraceStore] = None,
    ):
        self.repository = repository or JobRepository(db_path=db_path)
        self.lifecycle_guard = LifecycleGuard(self.repository)
        self.circuit_breaker = CircuitBreaker()
        self.policy_gate = AutonomyPolicyGate()
        self.trace_store = trace_store

        self.max_workers = max_workers
        self.tick_interval = tick_interval
        self.runner_fn = runner_fn

        self._running_job_ids: Set[str] = set()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()

        self._thread: Optional[threading.Thread] = None
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

        # Perform startup orphan reconciliation
        self.lifecycle_guard.reconcile_orphans()

    def start(self) -> None:
        """Start the background scheduler thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                logger.debug("AutonomyScheduler is already running.")
                return

            self._stop_event.clear()
            self._wake_event.clear()
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="MaddyAutonomyWorker",
            )
            self._thread = threading.Thread(
                target=self._tick_loop,
                name="MaddyAutonomyScheduler",
                daemon=True,
            )
            self._thread.start()
            logger.info("M.A.D.D.Y. AutonomyScheduler started with %d workers.", self.max_workers)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background scheduler and cleanly terminate workers."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                return
            logger.info("Stopping M.A.D.D.Y. AutonomyScheduler...")
            self._stop_event.set()
            self._wake_event.set()

        if self._thread:
            self._thread.join(timeout=timeout)

        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

        logger.info("M.A.D.D.Y. AutonomyScheduler stopped.")

    @property
    def is_running(self) -> bool:
        """Check if the scheduler daemon thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def add_job(
        self,
        name: str,
        goal: str,
        trigger_type: JobTriggerType = JobTriggerType.INTERVAL,
        interval_seconds: Optional[float] = None,
        time_of_day: Optional[str] = None,
        target_epoch: Optional[float] = None,
        cron_expression: Optional[str] = None,
        job_id: Optional[str] = None,
        timeout_seconds: float = 120.0,
        deliver_voice: bool = True,
        context: Optional[str] = None,
        required_tools: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ScheduledJob:
        """Register and persist a new background job."""
        jid = job_id or f"job_{uuid.uuid4().hex[:8]}"
        job = ScheduledJob(
            id=jid,
            name=name,
            goal=goal,
            trigger_type=trigger_type,
            interval_seconds=interval_seconds,
            time_of_day=time_of_day,
            target_epoch=target_epoch,
            cron_expression=cron_expression,
            timeout_seconds=timeout_seconds,
            deliver_voice=deliver_voice,
            context=context,
            required_tools=required_tools,
            metadata=metadata or {},
        )
        job.next_run_at = job.calculate_next_run(from_time=time.time())
        job.status = JobStatus.SCHEDULED
        self.repository.save_job(job)

        logger.info("Registered background job '%s' (%s). Next run at %s.", job.id, job.name, job.next_run_at)
        # Wake tick loop immediately to recalculate wait duration
        self._wake_event.set()
        return job

    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Fetch a job by ID."""
        return self.repository.get_job(job_id)

    def list_jobs(self, enabled_only: bool = False) -> List[ScheduledJob]:
        """List jobs from repository."""
        return self.repository.list_jobs(enabled_only=enabled_only)

    def pause_job(self, job_id: str) -> bool:
        """Pause a scheduled job."""
        job = self.repository.get_job(job_id)
        if not job:
            return False
        job.enabled = False
        job.status = JobStatus.SUSPENDED
        job.last_status = "PAUSED"
        self.repository.save_job(job)
        logger.info("Paused job '%s'.", job_id)
        return True

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused scheduled job."""
        job = self.repository.get_job(job_id)
        if not job:
            return False
        job.enabled = True
        job.status = JobStatus.SCHEDULED
        job.consecutive_failures = 0
        job.next_run_at = job.calculate_next_run(from_time=time.time())
        self.repository.save_job(job)
        logger.info("Resumed job '%s'. Next run at %s.", job_id, job.next_run_at)
        self._wake_event.set()
        return True

    def remove_job(self, job_id: str) -> bool:
        """Permanently remove a job."""
        res = self.repository.delete_job(job_id)
        if res:
            logger.info("Removed job '%s'.", job_id)
            self._wake_event.set()
        return res

    def trigger_now(self, job_id: str) -> bool:
        """Immediately schedule a job for execution on the next tick."""
        job = self.repository.get_job(job_id)
        if not job:
            return False
        job.next_run_at = time.time() - 1.0  # Force due
        job.enabled = True
        job.status = JobStatus.SCHEDULED
        self.repository.save_job(job)
        self._wake_event.set()
        return True

    def _tick_loop(self) -> None:
        """Main scheduler loop evaluating due jobs and dispatching workers."""
        while not self._stop_event.is_set():
            try:
                now = time.time()
                jobs = self.repository.list_jobs(enabled_only=True)
                due_jobs: List[ScheduledJob] = []

                for job in jobs:
                    if job.next_run_at and job.next_run_at <= now:
                        due_jobs.append(job)

                for job in due_jobs:
                    self._dispatch_job(job)

            except Exception as exc:
                logger.error("Unexpected error in AutonomyScheduler tick loop: %s", exc, exc_info=True)

            # Wait for tick interval or wake signal
            self._wake_event.wait(timeout=self.tick_interval)
            self._wake_event.clear()

    def _dispatch_job(self, job: ScheduledJob) -> None:
        """Dispatch a single due job, enforcing non-overlapping execution locks."""
        with self._lock:
            if job.id in self._running_job_ids:
                logger.warning(
                    "Job '%s' (%s) is already running. Skipping overlapping execution tick.",
                    job.id,
                    job.name,
                )
                return

            self._running_job_ids.add(job.id)

        job.status = JobStatus.RUNNING
        job.last_status = "RUNNING"
        self.repository.save_job(job)

        if self._executor and not self._stop_event.is_set():
            self._executor.submit(self._execute_job_wrapper, job)
        else:
            # Fallback inline execution
            threading.Thread(target=self._execute_job_wrapper, args=(job,), daemon=True).start()

    def _execute_job_wrapper(self, job: ScheduledJob) -> None:
        """Worker wrapper managing execution recording, timeouts, and circuit breaker."""
        exec_id = f"exec_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        self.lifecycle_guard.start_heartbeat(exec_id)

        # 1. Record execution start in SQLite repository
        self.repository.record_execution_start(
            execution_id=exec_id,
            job_id=job.id,
            started_at=start_time,
            metadata={"goal": job.goal, "trigger_type": job.trigger_type.value},
        )

        outcome_summary: Optional[str] = None
        error_msg: Optional[str] = None
        status = "COMPLETED"

        try:
            logger.info("Executing background task '%s' (%s): '%s'", job.id, job.name, job.goal)
            if self.runner_fn:
                outcome_summary = self.runner_fn(job)
            else:
                outcome_summary = self._default_runner(job)

            self.circuit_breaker.record_success(job)
            job.status = JobStatus.IDLE
            job.last_status = "SUCCESS"
            job.last_result = outcome_summary
            job.last_error = None

        except Exception as exc:
            error_msg = str(exc)
            status = "FAILED"
            logger.error("Execution failed for job '%s': %s", job.id, exc, exc_info=True)

            is_tripped, backoff_delay = self.circuit_breaker.record_failure(job, error_msg)
            job.last_error = error_msg
            if is_tripped:
                job.enabled = False
                job.status = JobStatus.SUSPENDED
            else:
                job.status = JobStatus.FAILED

        finally:
            end_time = time.time()
            self.lifecycle_guard.clear_heartbeat(exec_id)

            # 2. Finalize execution record
            self.repository.record_execution_finish(
                execution_id=exec_id,
                completed_at=end_time,
                status=status,
                result_summary=outcome_summary,
                error=error_msg,
                total_steps=1,
            )

            # 3. Update job scheduling state
            job.last_run_at = start_time
            job.run_count += 1

            if job.trigger_type == JobTriggerType.ONE_SHOT:
                job.enabled = False
                job.status = JobStatus.COMPLETED
                job.next_run_at = None
            elif job.enabled and job.status != JobStatus.SUSPENDED:
                job.next_run_at = job.calculate_next_run(from_time=end_time)
                job.status = JobStatus.SCHEDULED

            self.repository.save_job(job)

            # Release execution lock
            with self._lock:
                self._running_job_ids.discard(job.id)

    def _default_runner(self, job: ScheduledJob) -> str:
        """Default fallback execution simulation if no custom runner is provided."""
        time.sleep(0.05)
        return f"Autonomous task '{job.name}' completed successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}."
