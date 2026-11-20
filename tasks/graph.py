"""
Task Graph Engine for managing task dependencies, state machine transitions,
DAG integrity, graph serialization, stuck-task detection, and control operations.
"""
import time
import uuid
from typing import Any, Dict, List, Optional, Set
from tasks.task import Task, TaskState, TERMINAL_STATES

class TaskGraphError(ValueError):
    """Exception raised for TaskGraph errors and dependency violations."""
    pass

class TaskGraph:
    """
    DAG state engine for managing task decomposition, dependency resolution,
    lifecycle control, status propagation, and crash-resilient rehydration.
    """

    def __init__(
        self,
        graph_id: Optional[str] = None,
        created_at: Optional[float] = None,
        updated_at: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.graph_id = graph_id or f"graph_{uuid.uuid4().hex[:10]}"
        now = time.time()
        self.created_at = created_at if created_at is not None else now
        self.updated_at = updated_at if updated_at is not None else now
        self.metadata = dict(metadata) if metadata is not None else {}
        self._tasks: Dict[str, Task] = {}

    def add_task(self, task: Task) -> None:
        """
        Add a task to the graph. Performs validation for duplicate IDs, invalid dependencies, and cycles.
        """
        if not task or not isinstance(task, Task):
            raise TaskGraphError("Invalid Task object provided")

        if not task.id or not isinstance(task.id, str):
            raise TaskGraphError("Task id must be a non-empty string")

        if task.id in self._tasks:
            raise TaskGraphError(f"Task with id '{task.id}' already exists in TaskGraph")

        # Validate that referenced dependencies exist or will exist without self-loops
        for dep_id in task.dependencies:
            if dep_id == task.id:
                raise TaskGraphError(f"Task '{task.id}' cannot depend on itself")

        self._tasks[task.id] = task
        self.updated_at = time.time()

        # Perform cycle detection
        if self.detect_cycles():
            del self._tasks[task.id]
            raise TaskGraphError(f"Adding task '{task.id}' creates a cyclic dependency in TaskGraph")

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by ID."""
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[Task]:
        """Return all tasks in the graph."""
        return list(self._tasks.values())

    def get_ready_tasks(self) -> List[Task]:
        """
        Return all tasks that are currently READY for execution.
        A task is READY if its status is PENDING/READY and all its dependency tasks are in SUCCESS status.
        """
        ready: List[Task] = []
        for task in self._tasks.values():
            if task.status in (TaskState.PENDING, TaskState.READY):
                # Check dependencies
                deps_met = True
                for dep_id in task.dependencies:
                    dep_task = self.get_task(dep_id)
                    if not dep_task or dep_task.status != TaskState.SUCCESS:
                        deps_met = False
                        break

                if deps_met:
                    task.transition_to(TaskState.READY, force=True)
                    ready.append(task)

        # Sort ready tasks by priority (descending)
        ready.sort(key=lambda t: t.priority, reverse=True)
        return ready

    def update_status(
        self,
        task_id: str,
        new_status: TaskState,
        result: Optional[Any] = None,
        error: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """
        Update the status of a task with transition validation and propagate consequences to dependent tasks.
        """
        task = self.get_task(task_id)
        if not task:
            raise TaskGraphError(f"Task '{task_id}' not found in TaskGraph")

        task.transition_to(new_status, force=force)
        self.updated_at = time.time()

        if result is not None:
            task.result = result
        if error is not None:
            task.error = error

        # Failure propagation: block dependent tasks if this task reached a failed/cancelled terminal state
        if new_status in (TaskState.FAILED, TaskState.BLOCKED, TaskState.CANCELLED):
            self._propagate_blocked(task_id)

    def retry_task(self, task_id: str) -> bool:
        """
        Attempt to set a failed/retrying task back to READY if retries remain.
        Returns True if task was reset for retry, False if max_retries exceeded.
        """
        task = self.get_task(task_id)
        if not task:
            return False

        if task.retry_count < task.max_retries:
            task.retry_count += 1
            task.transition_to(TaskState.READY, force=True)
            self.updated_at = time.time()
            return True
        else:
            task.transition_to(TaskState.FAILED, force=True)
            self.updated_at = time.time()
            self._propagate_blocked(task_id)
            return False

    def cancel_task(self, task_id: str, reason: str = "Cancelled by user/supervisor") -> None:
        """
        Cancel a single task and block all downstream dependent tasks.
        """
        task = self.get_task(task_id)
        if task:
            task.error = reason
            task.transition_to(TaskState.CANCELLED, force=True)
            self.updated_at = time.time()
            self._propagate_blocked(task_id)

    def cancel_all(self, reason: str = "Cancelled by supervisor/user") -> None:
        """
        Cancel all non-terminal tasks in the graph and mark errors.
        """
        for task in self._tasks.values():
            if not task.is_terminal():
                task.error = reason
                task.transition_to(TaskState.CANCELLED, force=True)
        self.updated_at = time.time()

    def pause(self) -> None:
        """
        Pause all currently RUNNING, READY, or PENDING tasks cooperatively.
        """
        for task in self._tasks.values():
            if task.status in (TaskState.RUNNING, TaskState.READY, TaskState.VERIFYING):
                task.transition_to(TaskState.PAUSED, force=True)
        self.updated_at = time.time()

    def resume(self) -> None:
        """
        Resume all PAUSED tasks in the graph.
        Tasks whose dependencies are satisfied become READY; otherwise PENDING.
        """
        for task in self._tasks.values():
            if task.status == TaskState.PAUSED:
                deps_met = True
                for dep_id in task.dependencies:
                    dep_task = self.get_task(dep_id)
                    if not dep_task or dep_task.status != TaskState.SUCCESS:
                        deps_met = False
                        break
                target = TaskState.READY if deps_met else TaskState.PENDING
                task.transition_to(target, force=True)
        self.updated_at = time.time()

    def is_complete(self) -> bool:
        """
        Returns True if all tasks in the graph have reached terminal states.
        """
        if not self._tasks:
            return True
        return all(task.is_terminal() for task in self._tasks.values())

    def find_stuck_tasks(
        self,
        default_timeout_seconds: float = 120.0,
        current_time: Optional[float] = None,
    ) -> List[Task]:
        """
        Scan and return all tasks in RUNNING or VERIFYING state that have exceeded
        their configured deadline or the default timeout without progress.
        """
        stuck: List[Task] = []
        now = current_time if current_time is not None else time.time()
        for task in self._tasks.values():
            if task.status in (TaskState.RUNNING, TaskState.VERIFYING):
                if task.is_timed_out(current_time=now, default_timeout=default_timeout_seconds):
                    stuck.append(task)
        return stuck

    def _propagate_blocked(self, failed_task_id: str) -> None:
        """
        Recursively mark all downstream tasks depending on failed_task_id as BLOCKED.
        """
        for task in self._tasks.values():
            if failed_task_id in task.dependencies and not task.is_terminal():
                task.error = f"Blocked due to prerequisite task '{failed_task_id}' failure or cancellation"
                task.transition_to(TaskState.BLOCKED, force=True)
                # Recursive propagation down the DAG
                self._propagate_blocked(task.id)

    def detect_cycles(self) -> bool:
        """
        Return True if there is a cycle in the task graph, False otherwise.
        Uses DFS 3-color traversal (0=WHITE, 1=GRAY, 2=BLACK).
        """
        visited: Dict[str, int] = {t_id: 0 for t_id in self._tasks}

        def dfs(node_id: str) -> bool:
            visited[node_id] = 1  # GRAY (visiting)
            task = self._tasks.get(node_id)
            if task:
                for dep_id in task.dependencies:
                    if dep_id in self._tasks:
                        if visited[dep_id] == 1:
                            return True  # Cycle detected
                        if visited[dep_id] == 0 and dfs(dep_id):
                            return True
            visited[node_id] = 2  # BLACK (visited)
            return False

        for node_id in list(self._tasks.keys()):
            if visited[node_id] == 0:
                if dfs(node_id):
                    return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize entire TaskGraph and its DAG state to dictionary.
        """
        return {
            "graph_id": self.graph_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
            "is_complete": self.is_complete(),
            "total_tasks": len(self._tasks),
            "completed_tasks": sum(1 for t in self._tasks.values() if t.status == TaskState.SUCCESS),
            "tasks": {task_id: task.to_dict() for task_id, task in self._tasks.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskGraph":
        """
        Reconstruct a complete TaskGraph instance from serialized dictionary data.
        Restores all tasks, dependency edges, execution state, and validates integrity.
        """
        graph = cls(
            graph_id=data.get("graph_id"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            metadata=data.get("metadata", {}),
        )

        tasks_data = data.get("tasks", {})
        # Load tasks
        for task_id, task_dict in tasks_data.items():
            task = Task.from_dict(task_dict)
            graph._tasks[task.id] = task

        # Validate DAG cycle free
        if graph.detect_cycles():
            raise TaskGraphError("Deserialized TaskGraph contains a cyclic dependency violation")

        return graph
