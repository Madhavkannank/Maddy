"""
Task Memory implementation for external task state disk persistence.
Supports atomic checkpointing, versioning, backup recovery, and task graph rehydration.
Structure:
  ~/.maddy/memory/tasks/<task_id>/
    ├── goal.json
    ├── plan.json
    ├── progress.json
    ├── observations.json
    ├── errors.json
    ├── artifacts/
    ├── checkpoint.json
    ├── checkpoint.json.bak
    └── result.json
"""
import json
import logging
import os
import shutil
import threading
import time
from typing import Any, Dict, List, Optional
from tasks.graph import TaskGraph
from tasks.task import TaskState

logger = logging.getLogger(__name__)

CURRENT_CHECKPOINT_VERSION = 1

class TaskMemory:
    """
    Manages persistent external task directories on disk for long-running task auditability,
    atomic checkpointing, corruption recovery, and crash resumption.
    """

    def __init__(self, task_id: str, base_dir: Optional[str] = None):
        self.task_id = task_id
        if base_dir:
            self.task_dir = os.path.join(base_dir, task_id)
            self._base_dir = base_dir
        else:
            from config.settings import get_memory_dir
            self._base_dir = os.path.join(get_memory_dir(), "tasks")
            self.task_dir = os.path.join(self._base_dir, task_id)

        self.artifacts_dir = os.path.join(self.task_dir, "artifacts")
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        os.makedirs(self.task_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)

    _dir_locks: Dict[str, threading.RLock] = {}
    _global_lock: threading.RLock = threading.RLock()

    def _get_lock(self) -> threading.RLock:
        with self._global_lock:
            if self.task_dir not in self._dir_locks:
                self._dir_locks[self.task_dir] = threading.RLock()
            return self._dir_locks[self.task_dir]

    # ------------------------------------------------------------------
    # Atomic file persistence helpers
    # ------------------------------------------------------------------
    def _write_json_atomic(self, filename: str, data: Any) -> str:
        """
        Write JSON data atomically using temporary file + os.replace.
        Prevents corrupted/partially written files during unexpected termination.
        Includes thread locking and Windows file lock retry handling.
        """
        with self._get_lock():
            target_path = os.path.join(self.task_dir, filename)
            tmp_path = f"{target_path}.tmp.{os.getpid()}_{threading.get_ident()}_{int(time.time() * 1000)}"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                # Windows retry loop for transient file locking
                for attempt in range(5):
                    try:
                        os.replace(tmp_path, target_path)
                        return target_path
                    except PermissionError:
                        if attempt < 4:
                            time.sleep(0.03 * (attempt + 1))
                        else:
                            raise
                return target_path
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    def _read_json(self, filename: str, default: Any) -> Any:
        path = os.path.join(self.task_dir, filename)
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as err:
            logger.warning(f"Error reading JSON from {path}: {err}")
            return default

    # ------------------------------------------------------------------
    # Checkpoint Management & Corruption Recovery
    # ------------------------------------------------------------------
    def save_checkpoint(
        self,
        graph: TaskGraph,
        step_metadata: Optional[Dict[str, Any]] = None,
        journal_entries: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Atomically save a full execution checkpoint.
        Maintains a .bak copy of the previous valid checkpoint for corruption recovery.
        """
        with self._get_lock():
            checkpoint_path = os.path.join(self.task_dir, "checkpoint.json")
            backup_path = os.path.join(self.task_dir, "checkpoint.json.bak")

            # Rotate current valid checkpoint to backup before writing new one
            if os.path.exists(checkpoint_path):
                try:
                    with open(checkpoint_path, "r", encoding="utf-8") as f:
                        json.load(f)
                    shutil.copy2(checkpoint_path, backup_path)
                except Exception:
                    pass

            payload = {
                "checkpoint_version": CURRENT_CHECKPOINT_VERSION,
                "task_id": self.task_id,
                "graph_id": graph.graph_id,
                "timestamp": time.time(),
                "step_metadata": step_metadata or {},
                "journal": journal_entries or [],
                "graph": graph.to_dict(),
            }

            return self._write_json_atomic("checkpoint.json", payload)

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint data with schema versioning and automatic corruption recovery.
        If checkpoint.json is corrupted or unparseable, attempts recovery from checkpoint.json.bak.
        """
        checkpoint_path = os.path.join(self.task_dir, "checkpoint.json")
        backup_path = os.path.join(self.task_dir, "checkpoint.json.bak")

        data = None
        loaded_from_backup = False

        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Verify required structure
                if not isinstance(data, dict) or "graph" not in data:
                    raise ValueError("Malformed checkpoint: missing 'graph' payload")
            except Exception as e:
                logger.warning(f"Primary checkpoint damaged for task '{self.task_id}': {e}. Attempting backup recovery.")
                data = None

        # Fallback to backup if primary failed or absent
        if data is None and os.path.exists(backup_path):
            try:
                with open(backup_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "graph" in data:
                    loaded_from_backup = True
                    logger.info(f"Successfully recovered checkpoint from backup for task '{self.task_id}'")
                    # Restore primary from backup
                    try:
                        shutil.copy2(backup_path, checkpoint_path)
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"Backup checkpoint also damaged for task '{self.task_id}': {e}")
                data = None

        if not data:
            return None

        # Schema migration check
        version = data.get("checkpoint_version", 1)
        if version != CURRENT_CHECKPOINT_VERSION:
            data = self._migrate_checkpoint_schema(data, from_version=version, to_version=CURRENT_CHECKPOINT_VERSION)

        if loaded_from_backup:
            data["_recovered_from_backup"] = True

        return data

    def _migrate_checkpoint_schema(self, data: Dict[str, Any], from_version: int, to_version: int) -> Dict[str, Any]:
        """Handles forward-compatibility schema migrations."""
        logger.info(f"Migrating checkpoint schema from v{from_version} to v{to_version}")
        migrated = dict(data)
        migrated["checkpoint_version"] = to_version
        return migrated

    def load_graph_checkpoint(self) -> Optional[TaskGraph]:
        """
        Rehydrate a TaskGraph directly from the authoritative checkpoint.
        Returns None if no valid checkpoint exists.
        """
        checkpoint_data = self.load_checkpoint()
        if not checkpoint_data or "graph" not in checkpoint_data:
            return None

        graph_dict = checkpoint_data["graph"]
        return TaskGraph.from_dict(graph_dict)

    # ------------------------------------------------------------------
    # Discovery API for Resumable Tasks
    # ------------------------------------------------------------------
    @classmethod
    def list_resumable_tasks(cls, base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Scan task storage directories to identify incomplete tasks with valid checkpoint states.
        Distinguishes active, paused, interrupted, and timed-out states.
        Completed tasks are strictly excluded.
        """
        if base_dir:
            tasks_root = base_dir
        else:
            from config.settings import get_memory_dir
            tasks_root = os.path.join(get_memory_dir(), "tasks")

        if not os.path.exists(tasks_root):
            return []

        resumable: List[Dict[str, Any]] = []

        try:
            entries = os.listdir(tasks_root)
        except OSError:
            return []

        for task_id in entries:
            task_path = os.path.join(tasks_root, task_id)
            if not os.path.isdir(task_path):
                continue

            checkpoint_file = os.path.join(task_path, "checkpoint.json")
            backup_file = os.path.join(task_path, "checkpoint.json.bak")
            if not os.path.exists(checkpoint_file) and not os.path.exists(backup_file):
                continue

            mem = cls(task_id=task_id, base_dir=tasks_root if base_dir else None)
            checkpoint = mem.load_checkpoint()
            if not checkpoint:
                continue

            graph_data = checkpoint.get("graph", {})
            is_complete = graph_data.get("is_complete", False)
            if is_complete:
                continue

            tasks_map = graph_data.get("tasks", {})
            total_tasks = len(tasks_map)
            completed_tasks = sum(1 for t in tasks_map.values() if t.get("status") == TaskState.SUCCESS.value)

            # If all tasks succeeded, it is complete
            if total_tasks > 0 and completed_tasks == total_tasks:
                continue

            # Determine dominant execution status
            statuses = [t.get("status") for t in tasks_map.values()]
            if TaskState.INTERRUPTED.value in statuses:
                overall_status = "interrupted"
            elif TaskState.PAUSED.value in statuses:
                overall_status = "paused"
            elif TaskState.TIMED_OUT.value in statuses:
                overall_status = "timed_out"
            elif TaskState.RUNNING.value in statuses:
                # If it's saved as RUNNING on disk, on reboot it is considered interrupted
                overall_status = "interrupted"
            elif TaskState.READY.value in statuses:
                overall_status = "ready"
            elif TaskState.FAILED.value in statuses:
                overall_status = "failed"
            else:
                overall_status = "pending"

            goal_data = mem.load_goal()
            goal_text = goal_data.get("goal", "")

            resumable.append({
                "task_id": task_id,
                "goal": goal_text,
                "graph_id": graph_data.get("graph_id", ""),
                "status": overall_status,
                "updated_at": checkpoint.get("timestamp", 0.0),
                "completed_tasks": completed_tasks,
                "total_tasks": total_tasks,
            })

        # Sort by updated_at descending
        resumable.sort(key=lambda r: r["updated_at"], reverse=True)
        return resumable

    # ------------------------------------------------------------------
    # Standard Task Memory Persistence (Updated to Atomic Writes)
    # ------------------------------------------------------------------
    def save_goal(self, goal: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save task goal statement and metadata."""
        data = {"task_id": self.task_id, "goal": goal, "timestamp": time.time(), "metadata": metadata or {}}
        self._write_json_atomic("goal.json", data)

    def save_plan(self, plan: List[Dict[str, Any]]) -> None:
        """Save decomposed plan tasks."""
        data = {"task_id": self.task_id, "plan": plan, "timestamp": time.time()}
        self._write_json_atomic("plan.json", data)

    def record_progress(self, current_step: int, status: str, description: str) -> None:
        """Record progress checkpoint."""
        history = self.load_progress()
        history.append({
            "step": current_step,
            "status": status,
            "description": description,
            "timestamp": time.time(),
        })
        self._write_json_atomic("progress.json", history)

    def record_observation(self, step_number: int, tool_name: str, observation_data: Any) -> None:
        """Record tool observation artifact."""
        observations = self.load_observations()
        observations.append({
            "step": step_number,
            "tool": tool_name,
            "data": observation_data,
            "timestamp": time.time(),
        })
        self._write_json_atomic("observations.json", observations)

    def record_error(self, step_number: int, error_type: str, error_message: str) -> None:
        """Record execution error."""
        errors = self.load_errors()
        errors.append({
            "step": step_number,
            "type": error_type,
            "message": error_message,
            "timestamp": time.time(),
        })
        self._write_json_atomic("errors.json", errors)

    def save_artifact(self, filename: str, content: str) -> str:
        """Save a generated task artifact to disk."""
        target_path = os.path.join(self.artifacts_dir, filename)
        tmp_path = f"{target_path}.tmp.{os.getpid()}_{int(time.time() * 1000)}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target_path)
        return target_path

    def save_result(self, status: str, final_result: Any) -> None:
        """Save final task result outcome."""
        data = {"task_id": self.task_id, "status": status, "result": final_result, "timestamp": time.time()}
        self._write_json_atomic("result.json", data)

    def load_goal(self) -> Dict[str, Any]:
        return self._read_json("goal.json", {})

    def load_plan(self) -> List[Dict[str, Any]]:
        return self._read_json("plan.json", {}).get("plan", [])

    def load_progress(self) -> List[Dict[str, Any]]:
        return self._read_json("progress.json", [])

    def load_observations(self) -> List[Dict[str, Any]]:
        return self._read_json("observations.json", [])

    def load_errors(self) -> List[Dict[str, Any]]:
        return self._read_json("errors.json", [])

    def load_result(self) -> Dict[str, Any]:
        return self._read_json("result.json", {})
