"""Typed Data Models for Background Autonomy and Scheduled Execution in M.A.D.D.Y.

Data definitions for scheduled jobs, execution records, trigger mechanisms, and lifecycle states.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class JobTriggerType(str, Enum):
    """Trigger mechanism for scheduled background jobs."""
    INTERVAL = "interval"
    DAILY_TIME = "daily_time"
    ONE_SHOT = "one_shot"
    CRON_EXPRESSION = "cron_expression"


class JobStatus(str, Enum):
    """Lifecycle status of a scheduled background job."""
    IDLE = "IDLE"
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SUSPENDED = "SUSPENDED"
    RECOVERED = "RECOVERED"


@dataclass
class ScheduledJob:
    """Represents a persistent background task or cron job in M.A.D.D.Y."""
    id: str
    name: str
    goal: str
    trigger_type: JobTriggerType = JobTriggerType.INTERVAL
    interval_seconds: Optional[float] = None
    time_of_day: Optional[str] = None  # Format "HH:MM" (24h)
    target_epoch: Optional[float] = None  # Exact timestamp for one-shot
    cron_expression: Optional[str] = None  # e.g. "*/15 * * * *"
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    last_run_at: Optional[float] = None
    next_run_at: Optional[float] = None
    run_count: int = 0
    status: JobStatus = JobStatus.IDLE
    last_status: str = "IDLE"
    last_result: Optional[str] = None
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    failure_threshold: int = 3
    backoff_multiplier: float = 2.0
    max_retries: int = 3
    timeout_seconds: float = 120.0
    deliver_voice: bool = True
    context: Optional[str] = None
    required_tools: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_next_run(self, from_time: Optional[float] = None) -> float:
        """Calculates next epoch timestamp based on trigger_type."""
        now = from_time if from_time is not None else time.time()

        if self.trigger_type == JobTriggerType.ONE_SHOT:
            if self.target_epoch is not None:
                return self.target_epoch
            return now

        if self.trigger_type == JobTriggerType.INTERVAL:
            interval = self.interval_seconds or 3600.0
            if self.last_run_at and self.last_run_at > 0:
                # Anchor calculation to avoid cumulative clock drift
                elapsed = now - self.last_run_at
                intervals_passed = int(elapsed // interval) + 1
                return self.last_run_at + (intervals_passed * interval)
            return now + interval

        if self.trigger_type == JobTriggerType.DAILY_TIME and self.time_of_day:
            try:
                parts = self.time_of_day.strip().split(":")
                hour, minute = int(parts[0]), int(parts[1])
                dt_now = datetime.fromtimestamp(now)
                candidate = dt_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if candidate.timestamp() <= now:
                    # Move to next calendar day
                    days_to_add = 1
                    try:
                        candidate = candidate.replace(day=candidate.day + 1)
                    except ValueError:
                        # Month wrap-around handled via timestamp offset
                        candidate = datetime.fromtimestamp(candidate.timestamp() + 86400.0)
                return candidate.timestamp()
            except Exception as exc:
                logger.warning("Invalid time_of_day format '%s': %s", self.time_of_day, exc)
                return now + 3600.0

        if self.trigger_type == JobTriggerType.CRON_EXPRESSION and self.cron_expression:
            # Standard 5-part cron: minute hour day month day_of_week
            return self._calculate_cron_next(now)

        return now + 3600.0

    def _calculate_cron_next(self, now: float) -> float:
        """Parse standard 5-part cron expressions (minute hour day_of_month month day_of_week)."""
        if not self.cron_expression:
            return now + 3600.0
        parts = self.cron_expression.strip().split()
        if len(parts) != 5:
            logger.warning("Invalid cron expression '%s', defaulting to 1h", self.cron_expression)
            return now + 3600.0

        minute_part, hour_part = parts[0], parts[1]
        # Common pattern: */N in minute
        if minute_part.startswith("*/"):
            try:
                step = int(minute_part.split("/")[1])
                interval_secs = max(1, step) * 60.0
                return now + interval_secs
            except Exception:
                pass

        # Hourly at specific minute: "15 * * * *"
        if minute_part.isdigit() and hour_part == "*":
            target_min = int(minute_part)
            dt = datetime.fromtimestamp(now)
            candidate = dt.replace(minute=target_min, second=0, microsecond=0)
            if candidate.timestamp() <= now:
                candidate = datetime.fromtimestamp(candidate.timestamp() + 3600.0)
            return candidate.timestamp()

        # Fixed hour and minute: "30 8 * * *" -> equivalent to daily_time
        if minute_part.isdigit() and hour_part.isdigit():
            self.time_of_day = f"{int(hour_part):02d}:{int(minute_part):02d}"
            self.trigger_type = JobTriggerType.DAILY_TIME
            return self.calculate_next_run(now)

        return now + 3600.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to serialized dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "goal": self.goal,
            "trigger_type": self.trigger_type.value,
            "interval_seconds": self.interval_seconds,
            "time_of_day": self.time_of_day,
            "target_epoch": self.target_epoch,
            "cron_expression": self.cron_expression,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "last_run_at": self.last_run_at,
            "next_run_at": self.next_run_at,
            "run_count": self.run_count,
            "status": self.status.value,
            "last_status": self.last_status,
            "last_result": self.last_result,
            "last_error": self.last_error,
            "consecutive_failures": self.consecutive_failures,
            "failure_threshold": self.failure_threshold,
            "backoff_multiplier": self.backoff_multiplier,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "deliver_voice": self.deliver_voice,
            "context": self.context,
            "required_tools": self.required_tools,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScheduledJob:
        """Construct ScheduledJob from dictionary."""
        d = dict(data)
        if "trigger_type" in d and isinstance(d["trigger_type"], str):
            try:
                d["trigger_type"] = JobTriggerType(d["trigger_type"])
            except ValueError:
                d["trigger_type"] = JobTriggerType.INTERVAL
        if "status" in d and isinstance(d["status"], str):
            try:
                d["status"] = JobStatus(d["status"])
            except ValueError:
                d["status"] = JobStatus.IDLE

        # Retain only recognized dataclass fields
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class JobExecutionRecord:
    """Historical audit record of a single job execution."""
    execution_id: str
    job_id: str
    started_at: float
    completed_at: Optional[float] = None
    duration_seconds: float = 0.0
    status: str = "RUNNING"
    result_summary: Optional[str] = None
    error: Optional[str] = None
    total_steps: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JobExecutionRecord:
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)
