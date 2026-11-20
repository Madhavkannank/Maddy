"""
Unit tests for TaskGraph engine.
"""
import unittest
from tasks import Task, TaskGraph, TaskGraphError, TaskState

class TestTaskGraph(unittest.TestCase):

    def setUp(self):
        self.graph = TaskGraph()

    def test_add_and_get_task(self):
        task = Task(id="T1", description="First task")
        self.graph.add_task(task)
        retrieved = self.graph.get_task("T1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.description, "First task")

    def test_duplicate_task_id_raises_error(self):
        t1 = Task(id="T1", description="First task")
        t2 = Task(id="T1", description="Duplicate task")
        self.graph.add_task(t1)
        with self.assertRaises(TaskGraphError):
            self.graph.add_task(t2)

    def test_self_dependency_raises_error(self):
        t1 = Task(id="T1", description="Self dep", dependencies=["T1"])
        with self.assertRaises(TaskGraphError):
            self.graph.add_task(t1)

    def test_cycle_detection_raises_error(self):
        t1 = Task(id="T1", description="Task 1", dependencies=["T2"])
        t2 = Task(id="T2", description="Task 2", dependencies=["T1"])
        self.graph.add_task(t1)
        with self.assertRaises(TaskGraphError):
            self.graph.add_task(t2)

    def test_ready_tasks_detection(self):
        t1 = Task(id="T1", description="Task 1", dependencies=[])
        t2 = Task(id="T2", description="Task 2", dependencies=["T1"])
        self.graph.add_task(t1)
        self.graph.add_task(t2)

        ready = self.graph.get_ready_tasks()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].id, "T1")

        # Complete T1
        self.graph.update_status("T1", TaskState.SUCCESS, result="T1 done")
        ready_next = self.graph.get_ready_tasks()
        self.assertEqual(len(ready_next), 1)
        self.assertEqual(ready_next[0].id, "T2")

    def test_failure_propagation_blocks_dependents(self):
        t1 = Task(id="T1", description="Task 1", dependencies=[], max_retries=0)
        t2 = Task(id="T2", description="Task 2", dependencies=["T1"])
        self.graph.add_task(t1)
        self.graph.add_task(t2)

        # Fail T1 without retries
        self.graph.update_status("T1", TaskState.FAILED, error="T1 error")
        self.assertEqual(self.graph.get_task("T2").status, TaskState.BLOCKED)

    def test_task_cancellation_blocks_dependents(self):
        t1 = Task(id="T1", description="Task 1")
        t2 = Task(id="T2", description="Task 2", dependencies=["T1"])
        self.graph.add_task(t1)
        self.graph.add_task(t2)

        self.graph.cancel_task("T1", reason="User cancelled T1")
        self.assertEqual(self.graph.get_task("T1").status, TaskState.CANCELLED)
        self.assertEqual(self.graph.get_task("T2").status, TaskState.BLOCKED)

    def test_bounded_retry(self):
        t1 = Task(id="T1", description="Task 1", max_retries=1)
        self.graph.add_task(t1)

        # 1st retry succeeds
        retry1 = self.graph.retry_task("T1")
        self.assertTrue(retry1)
        self.assertEqual(t1.status, TaskState.READY)
        self.assertEqual(t1.retry_count, 1)

        # 2nd retry fails (exceeds max_retries=1)
        retry2 = self.graph.retry_task("T1")
        self.assertFalse(retry2)
        self.assertEqual(t1.status, TaskState.FAILED)

if __name__ == "__main__":
    unittest.main()
