"""Circuit Breaker and Quota Protection for M.A.D.D.Y. Autonomous Background Tasks.

Prevents runaway failure loops and infinite token burn by quarantining tasks
that exceed consecutive error thresholds.
"""

from __future__ import annotations

import logging
import time
from typing import Tuple

from core.autonomy.models import JobStatus, ScheduledJob

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Manages failure thresholds and exponential backoff quarantine for scheduled jobs."""

    def __init__(self, base_backoff_seconds: float = 60.0, max_backoff_seconds: float = 86400.0):
        self.base_backoff_seconds = base_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds

    def record_success(self, job: ScheduledJob) -> None:
        """Reset failure counters and restore normal scheduling upon successful execution."""
        if job.consecutive_failures > 0:
            logger.info(
                "Job '%s' recovered after %d consecutive failure(s). Resetting circuit breaker.",
                job.id,
                job.consecutive_failures,
            )
        job.consecutive_failures = 0
        if job.status == JobStatus.SUSPENDED:
            job.status = JobStatus.IDLE

    def record_failure(self, job: ScheduledJob, error: str) -> Tuple[bool, float]:
        """Record an execution failure, calculate backoff delay, and trip breaker if threshold exceeded.

        Returns:
            Tuple of (is_tripped: bool, backoff_seconds: float)
        """
        job.consecutive_failures += 1
        job.last_error = error
        threshold = job.failure_threshold or 3

        # Compute exponential backoff delay
        backoff_multiplier = job.backoff_multiplier or 2.0
        delay = min(
            self.max_backoff_seconds,
            self.base_backoff_seconds * (backoff_multiplier ** (job.consecutive_failures - 1)),
        )

        if job.consecutive_failures >= threshold:
            logger.error(
                "Circuit breaker TRIPPED for job '%s' (%s): %d consecutive failures reached (threshold=%d). "
                "Suspending job to prevent runaway failure loop. Error: %s",
                job.id,
                job.name,
                job.consecutive_failures,
                threshold,
                error,
            )
            job.status = JobStatus.SUSPENDED
            job.last_status = "CIRCUIT_BROKEN"
            job.metadata["circuit_breaker_tripped_at"] = time.time()
            job.metadata["circuit_breaker_reason"] = error
            return True, delay
        else:
            logger.warning(
                "Job '%s' failed (%d/%d consecutive failures). Error: %s. Applying backoff delay of %.1fs.",
                job.id,
                job.consecutive_failures,
                threshold,
                error,
                delay,
            )
            job.status = JobStatus.FAILED
            job.last_status = "FAILED"
            return False, delay
