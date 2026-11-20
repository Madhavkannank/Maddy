"""
Bounded delivery queue for dispatching session capture artifacts to Telegram.
Enforces hard bounds on pending item counts and aggregate payload sizes to prevent
runaway resource consumption during extended network or Telegram API outages.
"""
import collections
import threading
from typing import Any, Dict, List, Optional

from session_capture.artifacts import CaptureArtifact, DeliveryState


class BoundedDeliveryQueue:
    """
    Thread-safe bounded queue for scheduling artifact delivery attempts.
    """

    def __init__(
        self,
        max_pending_artifacts: int = 50,
        max_pending_bytes: int = 50 * 1024 * 1024,
    ):
        self.max_pending_artifacts = max_pending_artifacts
        self.max_pending_bytes = max_pending_bytes
        self._queue: collections.deque[CaptureArtifact] = collections.deque()
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._current_pending_bytes = 0
        self._total_enqueued = 0
        self._total_delivered = 0
        self._total_dropped = 0
        self._total_failed = 0

    def enqueue(self, artifact: CaptureArtifact) -> bool:
        """
        Attempt to enqueue an artifact for delivery.
        Fails safely if queue capacity or byte limits are exceeded.
        """
        with self._lock:
            # Check item count capacity
            if len(self._queue) >= self.max_pending_artifacts:
                self._total_dropped += 1
                artifact.delivery_state = DeliveryState.SKIPPED
                artifact.delivery_error = (
                    f"Delivery queue item limit exceeded ({len(self._queue)} >= {self.max_pending_artifacts})"
                )
                return False

            # Check byte capacity
            projected = self._current_pending_bytes + artifact.file_size_bytes
            if projected > self.max_pending_bytes:
                self._total_dropped += 1
                artifact.delivery_state = DeliveryState.SKIPPED
                artifact.delivery_error = (
                    f"Delivery queue byte limit exceeded ({projected} > {self.max_pending_bytes})"
                )
                return False

            artifact.delivery_state = DeliveryState.PENDING
            self._queue.append(artifact)
            self._current_pending_bytes += artifact.file_size_bytes
            self._total_enqueued += 1
            self._not_empty.notify()
            return True

    def dequeue(self, timeout: Optional[float] = 0.5) -> Optional[CaptureArtifact]:
        """
        Pop the next pending artifact from the queue.
        Waits up to `timeout` seconds before returning None.
        """
        with self._lock:
            if not self._queue:
                if timeout and timeout > 0:
                    self._not_empty.wait(timeout)
            if self._queue:
                return self._queue.popleft()
            return None

    def requeue(self, artifact: CaptureArtifact, max_retries: int = 3) -> bool:
        """
        Requeue a failed artifact if retry limit has not been exhausted.
        """
        with self._lock:
            artifact.retry_count += 1
            if artifact.retry_count <= max_retries:
                # Reinsert at the end
                self._queue.append(artifact)
                self._not_empty.notify()
                return True
            else:
                artifact.delivery_state = DeliveryState.FAILED
                self._total_failed += 1
                self._current_pending_bytes = max(0, self._current_pending_bytes - artifact.file_size_bytes)
                return False

    def mark_delivered(self, artifact: CaptureArtifact) -> None:
        """Record successful delivery of an artifact."""
        with self._lock:
            artifact.delivery_state = DeliveryState.DELIVERED
            artifact.delivery_error = None
            self._total_delivered += 1
            self._current_pending_bytes = max(0, self._current_pending_bytes - artifact.file_size_bytes)

    def mark_failed(self, artifact: CaptureArtifact, error: str) -> None:
        """Record terminal failure of an artifact delivery."""
        with self._lock:
            artifact.delivery_state = DeliveryState.FAILED
            artifact.delivery_error = error
            self._total_failed += 1
            self._current_pending_bytes = max(0, self._current_pending_bytes - artifact.file_size_bytes)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "pending_count": len(self._queue),
                "pending_bytes": self._current_pending_bytes,
                "max_pending_artifacts": self.max_pending_artifacts,
                "max_pending_bytes": self.max_pending_bytes,
                "total_enqueued": self._total_enqueued,
                "total_delivered": self._total_delivered,
                "total_dropped": self._total_dropped,
                "total_failed": self._total_failed,
            }

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
            self._current_pending_bytes = 0
