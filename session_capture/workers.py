"""
Periodic display and camera capture workers for Session Capture.
Designed to run in background threads, enforcing quota checks, PathSandbox,
and SHA-256 integrity verification while isolating failures.
"""
from datetime import datetime
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

from computer.controller import ComputerController
from security.audit_ledger import AuditLedger
from security.policy import PermissionDecision
from security.quotas import ResourceQuotaManager
from security.sandbox import PathOperation, PathSandbox
from session_capture.artifacts import ArtifactType, CaptureArtifact, DeliveryState
from session_capture.queue import BoundedDeliveryQueue
from session_capture.session import CaptureSession
from tools.base import RiskLevel
from tools.camera import CameraEngine

logger = logging.getLogger(__name__)


class DisplayCaptureWorker(threading.Thread):
    """
    Background worker periodically capturing display screenshots using ComputerController.
    """

    def __init__(
        self,
        session: CaptureSession,
        computer_controller: Optional[ComputerController] = None,
        delivery_queue: Optional[BoundedDeliveryQueue] = None,
        quota_manager: Optional[ResourceQuotaManager] = None,
        path_sandbox: Optional[PathSandbox] = None,
        audit_ledger: Optional[AuditLedger] = None,
    ):
        super().__init__(name="SessionCapture-DisplayWorker", daemon=True)
        self.session = session
        self.computer_controller = computer_controller or ComputerController()
        self.delivery_queue = delivery_queue
        self.quota_manager = quota_manager
        self.path_sandbox = path_sandbox
        self.audit_ledger = audit_ledger
        self._stop_event = threading.Event()
        self._sequence = 0
        self.quota_exceeded = False

    def stop(self) -> None:
        """Signal worker to stop immediately."""
        self._stop_event.set()

    def run(self) -> None:
        """Periodic capture loop."""
        interval = self.session.config.display_interval

        # On Windows, ensure worker thread is attached to the interactive input desktop
        if os.name == "nt":
            try:
                import ctypes
                user32 = ctypes.windll.user32
                h_desk = user32.OpenInputDesktop(0, False, 0x01FF)
                if h_desk:
                    user32.SetThreadDesktop(h_desk)
            except Exception:
                pass

        # Initial capture shortly after startup
        time.sleep(0.05)

        while not self._stop_event.is_set():
            try:
                self._capture_once()
            except Exception as e:
                logger.error(f"[DisplayCaptureWorker] Error during capture: {e}", exc_info=True)
                self._record_audit(
                    "DISPLAY_CAPTURE_ERROR",
                    False,
                    f"Capture worker encountered exception: {str(e)}",
                )

            # Wait for next interval or stop event
            self._stop_event.wait(timeout=interval)

    def _capture_once(self) -> None:
        """Execute a single display capture cycle."""
        # 1. Quota & storage limits check
        estimated_bytes = 500 * 1024  # estimate ~500 KB per screenshot
        if self.session.get_total_bytes() + estimated_bytes > self.session.config.max_storage_bytes:
            if not self.quota_exceeded:
                self.quota_exceeded = True
                self._record_audit(
                    "QUOTA_EXCEEDED",
                    False,
                    f"Session storage limit ({self.session.config.max_storage_bytes} bytes) reached",
                )
            return

        if self.quota_manager is not None:
            q_res = self.quota_manager.check_file_write(
                estimated_bytes,
                task_id=self.session.authorized_task_id,
            )
            if not q_res.allowed:
                if not self.quota_exceeded:
                    self.quota_exceeded = True
                    self._record_audit("QUOTA_EXCEEDED", False, q_res.reason)
                return

        # 2. Sequence and filepath generation
        self._sequence += 1
        seq = self._sequence
        timestamp = time.time()
        ts_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S_%f")[:19]
        filename = f"disp_{seq:05d}_{ts_str}.png"
        output_path = os.path.join(self.session.display_dir, filename)

        # 3. PathSandbox validation
        if self.path_sandbox is not None:
            sb_res = self.path_sandbox.validate_path(output_path, PathOperation.WRITE)
            if not sb_res.allowed:
                self._record_audit("SANDBOX_VIOLATION", False, sb_res.reason)
                return

        # 4. Perform display capture
        capture_res = self.computer_controller.take_screenshot(output_path)
        actual_path = capture_res.get("path") if isinstance(capture_res, dict) else output_path

        if not os.path.exists(actual_path):
            return

        file_size = os.path.getsize(actual_path)
        if self.quota_manager is not None:
            self.quota_manager.record_file_write(file_size, task_id=self.session.authorized_task_id)

        # 5. Cryptographic SHA-256 integrity verification
        sha256 = CaptureArtifact.compute_file_sha256(actual_path)
        artifact_id = f"art_disp_{seq:05d}_{sha256[:8]}"

        artifact = CaptureArtifact(
            artifact_id=artifact_id,
            session_id=self.session.session_id,
            artifact_type=ArtifactType.DISPLAY,
            file_path=actual_path,
            timestamp=timestamp,
            sequence_number=seq,
            file_size_bytes=file_size,
            sha256=sha256,
            delivery_state=DeliveryState.PENDING if self.session.config.enable_telegram else DeliveryState.NOT_REQUESTED,
        )

        # 6. Record in session manifest
        self.session.add_artifact(artifact)

        # 7. Enqueue for Telegram delivery if enabled
        if self.session.config.enable_telegram and self.delivery_queue is not None:
            self.delivery_queue.enqueue(artifact)

        # 8. Record in audit ledger
        self._record_audit(
            "DISPLAY_CAPTURE_SUCCESS",
            True,
            f"Display capture sequence #{seq} saved ({file_size} bytes)",
            artifact=artifact,
        )

    def _record_audit(
        self,
        decision_code: str,
        allowed: bool,
        reason: str,
        artifact: Optional[CaptureArtifact] = None,
    ) -> None:
        if not self.audit_ledger:
            return
        decision = PermissionDecision(
            allowed=allowed,
            requires_confirmation=False,
            reason=reason,
            decision_code=decision_code,
            risk_level=RiskLevel.LOW.value,
        )
        args = {
            "session_id": self.session.session_id,
            "stream": "display",
            "sequence": self._sequence,
        }
        if artifact:
            args["artifact_id"] = artifact.artifact_id
            args["sha256"] = artifact.sha256
            args["file_size_bytes"] = artifact.file_size_bytes

        try:
            self.audit_ledger.record(
                tool_name="session.capture.display",
                arguments=args,
                risk_level=RiskLevel.LOW,
                decision=decision,
            )
        except Exception as e:
            logger.error(f"Audit record error in DisplayCaptureWorker: {e}")


class CameraCaptureWorker(threading.Thread):
    """
    Background worker periodically capturing webcam frames using CameraEngine.
    Isolates camera hardware failures without affecting display capture or crashing M.A.D.D.Y.
    """

    def __init__(
        self,
        session: CaptureSession,
        camera_engine: Optional[CameraEngine] = None,
        delivery_queue: Optional[BoundedDeliveryQueue] = None,
        quota_manager: Optional[ResourceQuotaManager] = None,
        path_sandbox: Optional[PathSandbox] = None,
        audit_ledger: Optional[AuditLedger] = None,
    ):
        super().__init__(name="SessionCapture-CameraWorker", daemon=True)
        self.session = session
        self.camera_engine = camera_engine or CameraEngine(save_dir=session.camera_dir)
        self.delivery_queue = delivery_queue
        self.quota_manager = quota_manager
        self.path_sandbox = path_sandbox
        self.audit_ledger = audit_ledger
        self._stop_event = threading.Event()
        self._sequence = 0
        self.camera_available = True
        self.quota_exceeded = False

    def stop(self) -> None:
        """Signal worker to stop immediately."""
        self._stop_event.set()

    def run(self) -> None:
        """Periodic camera capture loop."""
        interval = self.session.config.camera_interval

        # Initial capture shortly after startup
        time.sleep(0.1)

        while not self._stop_event.is_set():
            try:
                self._capture_once()
            except Exception as e:
                logger.error(f"[CameraCaptureWorker] Error during capture: {e}", exc_info=True)
                self.camera_available = False
                self._record_audit(
                    "CAMERA_CAPTURE_ERROR",
                    False,
                    f"Camera worker encountered exception: {str(e)}",
                )

            # Wait for next interval or stop event
            self._stop_event.wait(timeout=interval)

    def _capture_once(self) -> None:
        """Execute a single camera capture cycle."""
        # 1. Quota & storage limits check
        estimated_bytes = 300 * 1024  # estimate ~300 KB per photo
        if self.session.get_total_bytes() + estimated_bytes > self.session.config.max_storage_bytes:
            if not self.quota_exceeded:
                self.quota_exceeded = True
                self._record_audit(
                    "QUOTA_EXCEEDED",
                    False,
                    f"Session storage limit ({self.session.config.max_storage_bytes} bytes) reached",
                )
            return

        if self.quota_manager is not None:
            q_res = self.quota_manager.check_file_write(
                estimated_bytes,
                task_id=self.session.authorized_task_id,
            )
            if not q_res.allowed:
                if not self.quota_exceeded:
                    self.quota_exceeded = True
                    self._record_audit("QUOTA_EXCEEDED", False, q_res.reason)
                return

        # 2. Sequence and filename generation
        self._sequence += 1
        seq = self._sequence
        timestamp = time.time()
        ts_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S_%f")[:19]
        filename = f"cam_{seq:05d}_{ts_str}.jpg"
        target_path = os.path.join(self.session.camera_dir, filename)

        # 3. PathSandbox validation
        if self.path_sandbox is not None:
            sb_res = self.path_sandbox.validate_path(target_path, PathOperation.WRITE)
            if not sb_res.allowed:
                self._record_audit("SANDBOX_VIOLATION", False, sb_res.reason)
                return

        # 4. Perform camera capture via CameraEngine
        # Ensure save_dir points to camera_dir
        self.camera_engine.save_dir = self.session.camera_dir
        success, photo_path, msg = self.camera_engine.capture_photo(filename=filename)

        if not success or not os.path.exists(photo_path):
            self.camera_available = False
            self._record_audit("CAMERA_UNAVAILABLE", False, msg)
            return

        self.camera_available = True
        file_size = os.path.getsize(photo_path)
        if self.quota_manager is not None:
            self.quota_manager.record_file_write(file_size, task_id=self.session.authorized_task_id)

        # 5. Cryptographic SHA-256 integrity verification
        sha256 = CaptureArtifact.compute_file_sha256(photo_path)
        artifact_id = f"art_cam_{seq:05d}_{sha256[:8]}"

        artifact = CaptureArtifact(
            artifact_id=artifact_id,
            session_id=self.session.session_id,
            artifact_type=ArtifactType.CAMERA,
            file_path=photo_path,
            timestamp=timestamp,
            sequence_number=seq,
            file_size_bytes=file_size,
            sha256=sha256,
            delivery_state=DeliveryState.PENDING if self.session.config.enable_telegram else DeliveryState.NOT_REQUESTED,
        )

        # 6. Record in session manifest
        self.session.add_artifact(artifact)

        # 7. Enqueue for Telegram delivery if enabled
        if self.session.config.enable_telegram and self.delivery_queue is not None:
            self.delivery_queue.enqueue(artifact)

        # 8. Record in audit ledger
        self._record_audit(
            "CAMERA_CAPTURE_SUCCESS",
            True,
            f"Camera snapshot sequence #{seq} saved ({file_size} bytes)",
            artifact=artifact,
        )

    def _record_audit(
        self,
        decision_code: str,
        allowed: bool,
        reason: str,
        artifact: Optional[CaptureArtifact] = None,
    ) -> None:
        if not self.audit_ledger:
            return
        decision = PermissionDecision(
            allowed=allowed,
            requires_confirmation=False,
            reason=reason,
            decision_code=decision_code,
            risk_level=RiskLevel.LOW.value,
        )
        args = {
            "session_id": self.session.session_id,
            "stream": "camera",
            "sequence": self._sequence,
        }
        if artifact:
            args["artifact_id"] = artifact.artifact_id
            args["sha256"] = artifact.sha256
            args["file_size_bytes"] = artifact.file_size_bytes

        try:
            self.audit_ledger.record(
                tool_name="session.capture.camera",
                arguments=args,
                risk_level=RiskLevel.LOW,
                decision=decision,
            )
        except Exception as e:
            logger.error(f"Audit record error in CameraCaptureWorker: {e}")
