"""
Lightweight, thread-safe Cron and Scheduled Task Engine for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).

Features:
- Periodic interval jobs (e.g. every N seconds/minutes/hours) and daily time-of-day jobs.
- Bounded background execution with non-overlapping execution locks.
- JSON persistence at <memory_dir>/cron_jobs.json.
- Voice notification delivery integration.
- Full CRUD lifecycle: add, list, pause, resume, remove, trigger.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config.settings import get_memory_dir

logger = logging.getLogger("tasks.cron_engine")


@dataclass
class CronJob:
    id: str
    name: str
    goal: str
    interval_seconds: Optional[float] = None
    time_of_day: Optional[str] = None  # Format "HH:MM" (24h)
    enabled: bool = True
    deliver_voice: bool = True
    created_at: float = field(default_factory=time.time)
    last_run_at: Optional[float] = None
    next_run_at: Optional[float] = None
    run_count: int = 0
    last_status: str = "IDLE"
    last_result: Optional[str] = None

    def calculate_next_run(self, from_time: Optional[float] = None) -> float:
        """Calculates next epoch timestamp based on interval or time_of_day."""
        now = from_time or time.time()
        if self.interval_seconds and self.interval_seconds > 0:
            return now + self.interval_seconds

        if self.time_of_day:
            try:
                parts = self.time_of_day.strip().split(":")
                hour, minute = int(parts[0]), int(parts[1])
                dt = datetime.fromtimestamp(now)
                target = dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if target.timestamp() <= now:
                    # Tomorrow
                    target = target.replace(day=target.day + 1)
                return target.timestamp()
            except Exception as e:
                logger.warning(f"[CronEngine] Invalid time_of_day '{self.time_of_day}': {e}")
                return now + 3600.0

        return now + 3600.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CronJob:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CronEngine:
    """
    Background scheduler thread executing periodic and timed M.A.D.D.Y. tasks.
    """

    def __init__(
        self,
        storage_path: Optional[Path | str] = None,
        job_executor: Optional[Callable[[str], Any]] = None,
        voice_notifier: Optional[Callable[[str], None]] = None,
        check_interval: float = 1.0,
    ):
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path(get_memory_dir()) / "cron_jobs.json"

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.job_executor = job_executor
        self.voice_notifier = voice_notifier
        self.check_interval = check_interval

        self._jobs: Dict[str, CronJob] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._load_jobs()

    def start(self) -> None:
        """Starts the background cron scheduler thread."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._scheduler_loop,
                name="MaddyCronScheduler",
                daemon=True,
            )
            self._thread.start()
            logger.info("[CronEngine] Scheduler thread started.")

    def stop(self) -> None:
        """Signals the background scheduler thread to stop."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            logger.info("[CronEngine] Scheduler thread stopped.")

    def _load_jobs(self) -> None:
        with self._lock:
            if not self.storage_path.exists():
                self._jobs = {}
                return
            try:
                raw = self.storage_path.read_text(encoding="utf-8")
                data = json.loads(raw)
                self._jobs = {j["id"]: CronJob.from_dict(j) for j in data}
            except Exception as e:
                logger.error(f"[CronEngine] Failed to load jobs from {self.storage_path}: {e}")
                self._jobs = {}

    def _save_jobs(self) -> None:
        try:
            items = [j.to_dict() for j in self._jobs.values()]
            tmp_path = self.storage_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
            tmp_path.replace(self.storage_path)
        except Exception as e:
            logger.error(f"[CronEngine] Failed to save jobs to {self.storage_path}: {e}")

    def add_job(
        self,
        name: str,
        goal: str,
        interval_seconds: Optional[float] = None,
        time_of_day: Optional[str] = None,
        deliver_voice: bool = True,
    ) -> Dict[str, Any]:
        """Adds a new scheduled job."""
        if not goal or not goal.strip():
            return {"success": False, "error": "Goal cannot be empty."}

        job_id = f"job_{int(time.time())}_{len(self._jobs) + 1}"
        job = CronJob(
            id=job_id,
            name=name.strip() or f"Task {job_id}",
            goal=goal.strip(),
            interval_seconds=interval_seconds,
            time_of_day=time_of_day,
            deliver_voice=deliver_voice,
        )
        job.next_run_at = job.calculate_next_run()

        with self._lock:
            self._jobs[job_id] = job
            self._save_jobs()

        return {"success": True, "job_id": job_id, "job": job.to_dict()}

    def remove_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                self._save_jobs()
                return {"success": True, "removed_id": job_id}
        return {"success": False, "error": f"Job '{job_id}' not found."}

    def pause_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].enabled = False
                self._save_jobs()
                return {"success": True, "job_id": job_id, "status": "PAUSED"}
        return {"success": False, "error": f"Job '{job_id}' not found."}

    def resume_job(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].enabled = True
                self._jobs[job_id].next_run_at = self._jobs[job_id].calculate_next_run()
                self._save_jobs()
                return {"success": True, "job_id": job_id, "status": "ACTIVE"}
        return {"success": False, "error": f"Job '{job_id}' not found."}

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]

    def trigger_job(self, job_id: str) -> Dict[str, Any]:
        """Manually trigger immediate execution of a job in a worker thread."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return {"success": False, "error": f"Job '{job_id}' not found."}

        threading.Thread(target=self._run_single_job, args=(job,), daemon=True).start()
        return {"success": True, "triggered_job": job_id}

    def _scheduler_loop(self) -> None:
        """Main loop periodically checking for due jobs."""
        while not self._stop_event.is_set():
            now = time.time()
            due_jobs: List[CronJob] = []

            with self._lock:
                for j in self._jobs.values():
                    if j.enabled and j.next_run_at and now >= j.next_run_at:
                        due_jobs.append(j)

            for job in due_jobs:
                threading.Thread(target=self._run_single_job, args=(job,), daemon=True).start()

            self._stop_event.wait(timeout=self.check_interval)

    def _run_single_job(self, job: CronJob) -> None:
        """Executes a single job, records outcome, schedules next run, and dispatches voice notification."""
        logger.info(f"[CronEngine] Running job '{job.name}' ({job.id}): '{job.goal}'")
        start_t = time.time()
        result_text = ""
        success = True

        try:
            with self._lock:
                job.last_status = "RUNNING"
                job.last_run_at = start_t

            if self.job_executor:
                out = self.job_executor(job.goal)
                if isinstance(out, dict):
                    result_text = out.get("response") or out.get("final_answer") or str(out)
                else:
                    result_text = str(out)
            else:
                result_text = f"Executed task: {job.goal}"

        except Exception as err:
            logger.error(f"[CronEngine] Error running job {job.id}: {err}")
            result_text = f"Error: {err}"
            success = False

        with self._lock:
            job.run_count += 1
            job.last_status = "SUCCESS" if success else "FAILED"
            job.last_result = result_text[:500]
            job.next_run_at = job.calculate_next_run(from_time=time.time())
            self._save_jobs()

        # Proactive voice announcement if enabled
        if job.deliver_voice and self.voice_notifier and result_text:
            try:
                alert_msg = f"Scheduled update for {job.name}, boss: {result_text[:200]}"
                self.voice_notifier(alert_msg)
            except Exception as e:
                logger.debug(f"[CronEngine] Voice alert delivery failed: {e}")
