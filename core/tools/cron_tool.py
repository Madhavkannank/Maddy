"""Typed Native Tool for Background Scheduling and Autonomy Management in M.A.D.D.Y.

Enables MaddyAgentCore to schedule periodic tasks, daily routines, delayed one-shots,
and manage active background job lifecycles.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from core.autonomy.models import JobTriggerType, ScheduledJob
from core.autonomy.scheduler import AutonomyScheduler
from core.permissions import PermissionTier
from core.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_GLOBAL_SCHEDULER: Optional[AutonomyScheduler] = None


def get_global_scheduler() -> AutonomyScheduler:
    """Get or initialize the process-wide AutonomyScheduler singleton."""
    global _GLOBAL_SCHEDULER
    if _GLOBAL_SCHEDULER is None:
        _GLOBAL_SCHEDULER = AutonomyScheduler()
        _GLOBAL_SCHEDULER.start()
    return _GLOBAL_SCHEDULER


def set_global_scheduler(scheduler: Optional[AutonomyScheduler]) -> None:
    """Override process-wide scheduler (used for test isolation)."""
    global _GLOBAL_SCHEDULER
    _GLOBAL_SCHEDULER = scheduler


class CronJobManageTool(BaseTool):
    """Native typed model tool for scheduling and managing background tasks."""

    name = "cronjob_manage"
    description = (
        "Schedule and control background tasks in M.A.D.D.Y. that execute periodically, "
        "at daily times, or as one-shot delayed goals.\n"
        "Supported actions:\n"
        "- 'add': Schedule a task (requires 'goal' and trigger parameters like 'interval_seconds' or 'time_of_day').\n"
        "- 'list': View all active, paused, and completed scheduled jobs.\n"
        "- 'pause': Temporarily suspend a job from executing by 'job_id'.\n"
        "- 'resume': Re-enable a paused or backoff-suspended job by 'job_id'.\n"
        "- 'remove': Permanently delete a scheduled job by 'job_id'.\n"
        "- 'run': Immediately trigger an execution of a job by 'job_id'.\n"
        "- 'history': View recent execution audit records for a job by 'job_id'."
    )
    tier = PermissionTier.SAFE

    def __init__(self, scheduler: Optional[AutonomyScheduler] = None):
        self._scheduler = scheduler

    @property
    def scheduler(self) -> AutonomyScheduler:
        if self._scheduler is not None:
            return self._scheduler
        return get_global_scheduler()

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "pause", "resume", "remove", "run", "history"],
                    "description": "Cron management action to perform.",
                },
                "job_id": {
                    "type": "string",
                    "description": "Unique identifier of the job (required for pause, resume, remove, run, history).",
                },
                "name": {
                    "type": "string",
                    "description": "Short human-readable label for the job (e.g. 'Morning Briefing', 'GPU Check').",
                },
                "goal": {
                    "type": "string",
                    "description": "Autonomous instruction or goal M.A.D.D.Y. will execute when fired.",
                },
                "trigger_type": {
                    "type": "string",
                    "enum": ["interval", "daily_time", "one_shot", "cron_expression"],
                    "description": "Trigger strategy: 'interval', 'daily_time', 'one_shot', or 'cron_expression'.",
                },
                "interval_seconds": {
                    "type": "number",
                    "description": "Cadence in seconds (e.g. 300 for 5m, 3600 for 1h, 86400 for 1d).",
                },
                "time_of_day": {
                    "type": "string",
                    "description": "Daily 24-hour trigger time formatted as 'HH:MM' (e.g. '08:00', '18:30').",
                },
                "target_epoch": {
                    "type": "number",
                    "description": "Absolute timestamp in epoch seconds for one-shot scheduling.",
                },
                "cron_expression": {
                    "type": "string",
                    "description": "Standard 5-part cron syntax (e.g. '*/15 * * * *').",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Max execution time in seconds for the background task (default 120.0).",
                },
                "deliver_voice": {
                    "type": "boolean",
                    "description": "Whether to speak completion outcome aloud upon finishing (default True).",
                },
            },
            "required": ["action"],
        }

    def execute(
        self,
        action: str,
        job_id: Optional[str] = None,
        name: Optional[str] = None,
        goal: Optional[str] = None,
        trigger_type: Optional[str] = None,
        interval_seconds: Optional[float] = None,
        time_of_day: Optional[str] = None,
        target_epoch: Optional[float] = None,
        cron_expression: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        deliver_voice: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute scheduled task management action."""
        act = action.strip().lower()

        # 1. Action: LIST
        if act == "list":
            jobs = self.scheduler.list_jobs(enabled_only=False)
            if not jobs:
                return ToolResult(
                    success=True,
                    output="No background tasks are currently scheduled.",
                    metadata={"total_jobs": 0, "jobs": []},
                )
            lines = [f"Registered Scheduled Tasks ({len(jobs)} total):"]
            for j in jobs:
                status_icon = "ENABLED" if j.enabled else "PAUSED"
                next_str = (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(j.next_run_at))
                    if j.next_run_at
                    else "NONE"
                )
                lines.append(
                    f"- [{j.id}] '{j.name}' [{status_icon} / {j.status.value}] | "
                    f"Trigger: {j.trigger_type.value} | Next: {next_str} | Runs: {j.run_count} | Goal: {j.goal}"
                )
            return ToolResult(
                success=True,
                output="\n".join(lines),
                metadata={"total_jobs": len(jobs), "jobs": [j.to_dict() for j in jobs]},
            )

        # 2. Action: ADD
        if act == "add":
            if not goal or not goal.strip():
                return ToolResult(
                    success=False,
                    output="Error: 'goal' is required to schedule a background task.",
                    error="Missing required 'goal' parameter",
                )

            # Determine trigger type
            ttype = JobTriggerType.INTERVAL
            if trigger_type:
                try:
                    ttype = JobTriggerType(trigger_type)
                except ValueError:
                    ttype = JobTriggerType.INTERVAL
            elif time_of_day:
                ttype = JobTriggerType.DAILY_TIME
            elif target_epoch:
                ttype = JobTriggerType.ONE_SHOT
            elif cron_expression:
                ttype = JobTriggerType.CRON_EXPRESSION

            created_job = self.scheduler.add_job(
                name=name or "Scheduled Task",
                goal=goal.strip(),
                trigger_type=ttype,
                interval_seconds=interval_seconds,
                time_of_day=time_of_day,
                target_epoch=target_epoch,
                cron_expression=cron_expression,
                timeout_seconds=float(timeout_seconds) if timeout_seconds else 120.0,
                deliver_voice=deliver_voice,
            )
            next_str = (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_job.next_run_at))
                if created_job.next_run_at
                else "NONE"
            )
            return ToolResult(
                success=True,
                output=(
                    f"Successfully scheduled task '{created_job.name}' [ID: {created_job.id}]:\n"
                    f"- Trigger: {created_job.trigger_type.value}\n"
                    f"- Next Run: {next_str}\n"
                    f"- Goal: {created_job.goal}"
                ),
                metadata=created_job.to_dict(),
            )

        # 3. Action: PAUSE
        if act == "pause":
            if not job_id:
                return ToolResult(success=False, output="Error: 'job_id' is required to pause a task.", error="Missing job_id")
            success = self.scheduler.pause_job(job_id)
            if not success:
                return ToolResult(success=False, output=f"Task '{job_id}' not found.", error=f"Job {job_id} not found")
            return ToolResult(success=True, output=f"Scheduled task '{job_id}' has been paused.")

        # 4. Action: RESUME
        if act == "resume":
            if not job_id:
                return ToolResult(success=False, output="Error: 'job_id' is required to resume a task.", error="Missing job_id")
            success = self.scheduler.resume_job(job_id)
            if not success:
                return ToolResult(success=False, output=f"Task '{job_id}' not found.", error=f"Job {job_id} not found")
            return ToolResult(success=True, output=f"Scheduled task '{job_id}' has been resumed.")

        # 5. Action: REMOVE
        if act == "remove":
            if not job_id:
                return ToolResult(success=False, output="Error: 'job_id' is required to remove a task.", error="Missing job_id")
            success = self.scheduler.remove_job(job_id)
            if not success:
                return ToolResult(success=False, output=f"Task '{job_id}' not found.", error=f"Job {job_id} not found")
            return ToolResult(success=True, output=f"Scheduled task '{job_id}' has been permanently deleted.")

        # 6. Action: RUN (immediate trigger)
        if act == "run":
            if not job_id:
                return ToolResult(success=False, output="Error: 'job_id' is required to run a task.", error="Missing job_id")
            success = self.scheduler.trigger_now(job_id)
            if not success:
                return ToolResult(success=False, output=f"Task '{job_id}' not found.", error=f"Job {job_id} not found")
            return ToolResult(success=True, output=f"Scheduled task '{job_id}' has been triggered for immediate execution.")

        # 7. Action: HISTORY
        if act == "history":
            if not job_id:
                return ToolResult(success=False, output="Error: 'job_id' is required to view execution history.", error="Missing job_id")
            records = self.scheduler.repository.get_executions_for_job(job_id, limit=5)
            if not records:
                return ToolResult(
                    success=True,
                    output=f"No execution history found for task '{job_id}'.",
                    metadata={"executions": []},
                )
            lines = [f"Execution History for Task '{job_id}':"]
            for r in records:
                start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.started_at))
                lines.append(
                    f"- [{start_str}] Status: {r.status} ({r.duration_seconds:.2f}s) | "
                    f"Result: {r.result_summary or r.error or 'N/A'}"
                )
            return ToolResult(
                success=True,
                output="\n".join(lines),
                metadata={"executions": [r.to_dict() for r in records]},
            )

        return ToolResult(
            success=False,
            output=f"Error: Unknown action '{action}'. Supported actions: add, list, pause, resume, remove, run, history.",
            error="Unknown action",
        )
