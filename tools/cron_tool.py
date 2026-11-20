"""Background Cronjob Manage Tool for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Exposes the single compressed `cronjob_manage` tool for scheduling background tasks,
listing active jobs, pausing/resuming schedules, and running jobs manually.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from tasks.cron_engine import CronEngine
from tools.base import RiskLevel, Tool

logger = logging.getLogger("tools.cron_tool")

_GLOBAL_CRON_ENGINE: Optional[CronEngine] = None


def get_cron_engine() -> CronEngine:
    """Returns or creates the process-wide CronEngine singleton."""
    global _GLOBAL_CRON_ENGINE
    if _GLOBAL_CRON_ENGINE is None:
        _GLOBAL_CRON_ENGINE = CronEngine()
    return _GLOBAL_CRON_ENGINE


def set_cron_engine(engine: Optional[CronEngine]) -> None:
    """Overrides the global cron engine (used for test isolation)."""
    global _GLOBAL_CRON_ENGINE
    _GLOBAL_CRON_ENGINE = engine


CRONJOB_MANAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["add", "list", "remove", "pause", "resume", "run"],
            "description": "Cron management action to perform.",
        },
        "job_id": {
            "type": "string",
            "description": "Unique identifier of the job (required for remove, pause, resume, run).",
        },
        "name": {
            "type": "string",
            "description": "Short human-readable label for the job (e.g. 'Morning Briefing', 'GPU Check').",
        },
        "goal": {
            "type": "string",
            "description": "Natural language command or task goal M.A.D.D.Y. will execute when fired.",
        },
        "interval_seconds": {
            "type": "number",
            "description": "Interval cadence in seconds (e.g. 300 for 5m, 3600 for 1h, 86400 for 1d).",
        },
        "time_of_day": {
            "type": "string",
            "description": "Daily 24-hour trigger time formatted as 'HH:MM' (e.g. '08:00', '18:30').",
        },
        "deliver_voice": {
            "type": "boolean",
            "description": "Whether to speak the completion outcome aloud upon finishing (default True).",
        },
    },
    "required": ["action"],
}


def cronjob_manage_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for the cronjob_manage Tool."""
    engine = get_cron_engine()
    action = params.get("action", "list").lower().strip()

    if action == "list":
        jobs = engine.list_jobs()
        return {"success": True, "total_jobs": len(jobs), "jobs": jobs}

    elif action == "add":
        return engine.add_job(
            name=params.get("name", "Scheduled Task"),
            goal=params.get("goal", ""),
            interval_seconds=params.get("interval_seconds"),
            time_of_day=params.get("time_of_day"),
            deliver_voice=params.get("deliver_voice", True),
        )

    elif action == "remove":
        job_id = params.get("job_id", "").strip()
        if not job_id:
            return {"success": False, "error": "job_id is required for remove."}
        return engine.remove_job(job_id)

    elif action == "pause":
        job_id = params.get("job_id", "").strip()
        if not job_id:
            return {"success": False, "error": "job_id is required for pause."}
        return engine.pause_job(job_id)

    elif action == "resume":
        job_id = params.get("job_id", "").strip()
        if not job_id:
            return {"success": False, "error": "job_id is required for resume."}
        return engine.resume_job(job_id)

    elif action == "run":
        job_id = params.get("job_id", "").strip()
        if not job_id:
            return {"success": False, "error": "job_id is required for run."}
        return engine.trigger_job(job_id)

    return {
        "success": False,
        "error": f"Unknown action '{action}'. Supported actions: add, list, remove, pause, resume, run.",
    }


cronjob_manage_tool = Tool(
    name="cronjob_manage",
    description=(
        "Schedule and control background tasks in M.A.D.D.Y. that fire periodically or at specific times.\n"
        "ACTIONS:\n"
        "- 'add': Schedule a task (requires 'goal' and either 'interval_seconds' or 'time_of_day').\n"
        "- 'list': View all active and scheduled jobs with run counts and next fire times.\n"
        "- 'remove': Delete a job by 'job_id'.\n"
        "- 'pause': Temporarily stop a job from firing by 'job_id'.\n"
        "- 'resume': Re-enable a paused job by 'job_id'.\n"
        "- 'run': Manually trigger an immediate run of a job by 'job_id'."
    ),
    parameter_schema=CRONJOB_MANAGE_SCHEMA,
    handler=cronjob_manage_handler,
    risk_level=RiskLevel.MEDIUM,
)


def create_cronjob_tool(engine: CronEngine) -> Tool:
    """Convenience factory returning an isolated cronjob_manage Tool instance."""
    def _handler(params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "list").lower().strip()
        if action == "list":
            jobs = engine.list_jobs()
            return {"success": True, "total_jobs": len(jobs), "jobs": jobs}
        elif action == "add":
            return engine.add_job(
                name=params.get("name", "Scheduled Task"),
                goal=params.get("goal", ""),
                interval_seconds=params.get("interval_seconds"),
                time_of_day=params.get("time_of_day"),
                deliver_voice=params.get("deliver_voice", True),
            )
        elif action == "remove":
            return engine.remove_job(params.get("job_id", ""))
        elif action == "pause":
            return engine.pause_job(params.get("job_id", ""))
        elif action == "resume":
            return engine.resume_job(params.get("job_id", ""))
        elif action == "run":
            return engine.trigger_job(params.get("job_id", ""))
        return {"success": False, "error": f"Unknown action '{action}'."}

    return Tool(
        name="cronjob_manage",
        description=cronjob_manage_tool.description,
        parameter_schema=CRONJOB_MANAGE_SCHEMA,
        handler=_handler,
        risk_level=RiskLevel.MEDIUM,
    )
