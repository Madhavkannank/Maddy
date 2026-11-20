"""
Telegram delivery adapter for session capture artifacts.
Runs asynchronously and independently of capture workers so that network latency,
rate limits, or Telegram outages never stall display/camera capture.
"""
import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

try:
    from integrations.messaging.telegram_client import TelegramClient
except ImportError:
    from integrations.messaging import TelegramClient
from security.audit_ledger import AuditLedger
from security.policy import PermissionDecision
from session_capture.artifacts import CaptureArtifact, DeliveryState
from session_capture.queue import BoundedDeliveryQueue
from tools.base import RiskLevel

logger = logging.getLogger(__name__)


class TelegramDeliveryAdapter:
    """
    Asynchronous delivery worker reading from BoundedDeliveryQueue and dispatching
    artifacts via the official TelegramClient.
    """

    def __init__(
        self,
        queue: BoundedDeliveryQueue,
        telegram_client: Optional[TelegramClient] = None,
        audit_ledger: Optional[AuditLedger] = None,
        default_chat_id: Optional[str] = None,
        max_retries: int = 3,
        poll_interval: float = 0.5,
    ):
        self.queue = queue
        self.telegram_client = telegram_client or TelegramClient()
        self.audit_ledger = audit_ledger
        self.default_chat_id = default_chat_id or getattr(self.telegram_client, "default_chat_id", None)
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background delivery worker."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="SessionCapture-TelegramDeliveryWorker",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Signal worker to stop, allowing pending items to flush within timeout."""
        start_t = time.time()
        while self.queue is not None and not self.queue.is_empty() and (time.time() - start_t < timeout - 0.5):
            time.sleep(0.1)
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            remaining = max(0.5, timeout - (time.time() - start_t))
            self._thread.join(timeout=remaining)
            self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def send_artifact_sync(self, artifact: CaptureArtifact, chat_id: Optional[str] = None) -> Tuple[bool, str]:
        """
        Direct synchronous delivery attempt of a single artifact.
        Handles missing files, unconfigured client, and network errors safely.
        """
        target_chat = chat_id or self.default_chat_id

        if not os.path.exists(artifact.file_path):
            err_msg = f"Artifact file not found: {artifact.file_path}"
            self._record_audit(artifact, False, err_msg)
            return False, err_msg

        if not self.telegram_client.is_configured:
            err_msg = "TelegramClient is not configured with valid bot token or chat ID."
            self._record_audit(artifact, False, err_msg)
            return False, err_msg

        caption = (
            f"📸 *[M.A.D.D.Y. Capture Session]*\n"
            f"• Session: `{artifact.session_id}`\n"
            f"• Seq: `#{artifact.sequence_number}` ({artifact.artifact_type.value.upper()})\n"
            f"• SHA256: `{artifact.sha256[:16]}...`"
        )

        try:
            res = self.telegram_client.send_photo(
                photo_path=artifact.file_path,
                caption=caption,
                chat_id=target_chat,
                parse_mode="Markdown",
            )
            success = bool(res.get("success"))
            msg = res.get("message_id") if success else res.get("error", "Unknown delivery failure")
            self._record_audit(artifact, success, str(msg))
            return success, str(msg)
        except Exception as e:
            err_msg = f"Telegram delivery exception: {str(e)}"
            logger.warning(err_msg)
            self._record_audit(artifact, False, err_msg)
            return False, err_msg

    def _worker_loop(self) -> None:
        """Main queue-draining loop."""
        while not self._stop_event.is_set():
            artifact = self.queue.dequeue(timeout=self.poll_interval)
            if artifact is None:
                continue

            success, msg = self.send_artifact_sync(artifact)
            if success:
                self.queue.mark_delivered(artifact)
            else:
                requeued = self.queue.requeue(artifact, max_retries=self.max_retries)
                if not requeued:
                    self.queue.mark_failed(artifact, msg)
                # Exponential/backoff wait on failure
                time.sleep(1.0)

    def _record_audit(self, artifact: CaptureArtifact, success: bool, reason: str) -> None:
        """Record artifact delivery attempt in the tamper-evident AuditLedger."""
        if not self.audit_ledger:
            return

        decision = PermissionDecision(
            allowed=success,
            requires_confirmation=False,
            reason=f"Telegram delivery {'succeeded' if success else 'failed'}: {reason}",
            decision_code="TELEGRAM_DELIVERY_SUCCESS" if success else "TELEGRAM_DELIVERY_FAILED",
            risk_level=RiskLevel.LOW.value,
        )

        args = {
            "artifact_id": artifact.artifact_id,
            "session_id": artifact.session_id,
            "sequence_number": artifact.sequence_number,
            "artifact_type": artifact.artifact_type.value,
            "sha256": artifact.sha256,
            "retry_count": artifact.retry_count,
        }

        try:
            self.audit_ledger.record(
                tool_name="session.capture.delivery",
                arguments=args,
                risk_level=RiskLevel.LOW,
                decision=decision,
            )
        except Exception as e:
            logger.error(f"Failed to record delivery audit: {e}")
