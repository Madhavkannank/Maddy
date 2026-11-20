"""Background Autonomy and Scheduling Architecture for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).

Provides persistent background job scheduling, crash recovery reconciliation,
consecutive failure circuit breakers, and policy-governed execution.
"""

from core.autonomy.circuit_breaker import CircuitBreaker
from core.autonomy.lifecycle import LifecycleGuard
from core.autonomy.models import (
    JobExecutionRecord,
    JobStatus,
    JobTriggerType,
    ScheduledJob,
)
from core.autonomy.policy import AutonomyDecision, AutonomyPolicyGate
from core.autonomy.repository import JobRepository
from core.autonomy.scheduler import AutonomyScheduler

__all__ = [
    "AutonomyDecision",
    "AutonomyPolicyGate",
    "AutonomyScheduler",
    "CircuitBreaker",
    "JobExecutionRecord",
    "JobRepository",
    "JobStatus",
    "JobTriggerType",
    "LifecycleGuard",
    "ScheduledJob",
]
