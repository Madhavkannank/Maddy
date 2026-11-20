"""
Session Capture Manager for M.A.D.D.Y.
Coordinates lifecycle, authorization, sandboxed storage, periodic capture workers,
and delivery queuing without global singleton lock-in.
"""
from datetime import datetime
import getpass
import json
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

from computer.controller import ComputerController
from config.settings import get_memory_dir
from security.audit_ledger import AuditLedger
from security.permission_engine import PermissionEngine
from security.policy import PermissionDecision
from security.quotas import ResourceQuotaManager
from security.sandbox import PathOperation, PathSandbox
from session_capture.config import SessionCaptureConfig
from session_capture.delivery import TelegramDeliveryAdapter
from session_capture.queue import BoundedDeliveryQueue
from session_capture.session import CaptureSession, SessionState
from session_capture.workers import CameraCaptureWorker, DisplayCaptureWorker
from tools.base import RiskLevel
from tools.camera import CameraEngine

logger = logging.getLogger(__name__)


class SessionCaptureManager:
    """
    Core orchestrator for temporary capture sessions.
    Provides explicit lifecycle controls, injectable dependencies, and integration
    with M.A.D.D.Y. security components.
    """

    def __init__(
        self,
        permission_engine: Optional[PermissionEngine] = None,
        path_sandbox: Optional[PathSandbox] = None,
        quota_manager: Optional[ResourceQuotaManager] = None,
        audit_ledger: Optional[AuditLedger] = None,
        computer_controller: Optional[ComputerController] = None,
        camera_engine: Optional[CameraEngine] = None,
        base_storage_dir: Optional[str] = None,
    ):
        self.permission_engine = permission_engine
        self.path_sandbox = path_sandbox or (permission_engine.path_sandbox if permission_engine else PathSandbox())
        self.quota_manager = quota_manager or (permission_engine.quota_manager if permission_engine else ResourceQuotaManager())
        self.audit_ledger = audit_ledger or (permission_engine.audit_ledger if permission_engine else AuditLedger())
        self.computer_controller = computer_controller or ComputerController()
        self.camera_engine = camera_engine or CameraEngine()

        if base_storage_dir:
            self.base_storage_dir = base_storage_dir
        else:
            # Default to <workspace_root>/memory/capture_sessions inside allowed workspace
            root = self.path_sandbox.workspace_roots[0] if (self.path_sandbox and self.path_sandbox.workspace_roots) else os.getcwd()
            self.base_storage_dir = os.path.join(root, "memory", "capture_sessions")

        self._lock = threading.RLock()
        self._current_session: Optional[CaptureSession] = None
        self._display_worker: Optional[DisplayCaptureWorker] = None
        self._camera_worker: Optional[CameraCaptureWorker] = None
        self._delivery_queue: Optional[BoundedDeliveryQueue] = None
        self._delivery_adapter: Optional[TelegramDeliveryAdapter] = None
        self._recover_orphaned_sessions()

    def _recover_orphaned_sessions(self) -> None:
        """Mark any previously active sessions on disk from older processes as interrupted."""
        if not os.path.exists(self.base_storage_dir):
            return
        try:
            for entry in os.scandir(self.base_storage_dir):
                if entry.is_dir() and entry.name.startswith("cap_"):
                    meta_path = os.path.join(entry.path, "metadata.json")
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            if data.get("state") in ("active", "starting"):
                                data["state"] = "stopped"
                                data["stopped_at"] = time.time()
                                with open(meta_path, "w", encoding="utf-8") as f:
                                    json.dump(data, f, indent=2)
                        except Exception:
                            pass
        except Exception:
            pass

    def has_active_session(self) -> bool:
        """Return True if a capture session is currently running locally or as a background daemon."""
        with self._lock:
            if self._current_session and self._current_session.state == SessionState.ACTIVE:
                return True
        # Check detached daemon
        try:
            from session_capture.cli import get_daemon_info, is_process_alive
            info = get_daemon_info()
            if info and is_process_alive(info.get("pid", 0)):
                return True
        except Exception:
            pass
        return False

    @property
    def current_session(self) -> Optional[CaptureSession]:
        with self._lock:
            return self._current_session

    def start_session(
        self,
        config: Optional[SessionCaptureConfig] = None,
        task_id: Optional[str] = None,
        caller_id: Optional[str] = None,
        detached: bool = False,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Start an owner-authorized capture session.
        Validates configuration against security bounds and initializes workers.
        If detached=True, spawns the independent background daemon so capture survives assistant shutdown.
        """
        # 1. Guard against concurrent active sessions (in-process or background daemon)
        if self.has_active_session():
            sess_id = self._current_session.session_id if self._current_session else "background"
            return (
                False,
                f"A capture session is already active (ID: '{sess_id}'). "
                "Stop the current session before starting a new one.",
                self.get_status(),
            )

        effective_config = config or SessionCaptureConfig()

        # 2. Validate configuration bounds and sandbox rules
        try:
            effective_config.validate(self.path_sandbox)
        except ValueError as ve:
            err = f"Invalid session capture configuration: {str(ve)}"
            self._record_audit("CONFIG_VALIDATION_FAILED", False, err)
            return False, err, {"error": err}

        # 3. Resolve session identity and authorization metadata
        caller = caller_id or getpass.getuser() or "owner"
        tid = task_id or f"task_cap_{int(time.time())}"
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"cap_{ts_str}_{os.urandom(3).hex()}"

        storage_root = effective_config.base_storage_dir or self.base_storage_dir
        session_dir = os.path.join(storage_root, session_id)

        # Handle detached daemon execution
        if detached:
            python_exe = sys.executable
            cmd = [
                python_exe,
                "-m",
                "session_capture.daemon",
                "--caller-id", str(caller),
                "--task-id", str(tid),
                "--session-id", str(session_id),
                "--display-interval", str(effective_config.display_interval),
                "--camera-interval", str(effective_config.camera_interval),
                "--storage-dir", str(storage_root),
            ]
            if effective_config.enable_display:
                cmd.append("--enable-display")
            else:
                cmd.append("--no-display")

            if effective_config.enable_camera:
                cmd.append("--enable-camera")
            else:
                cmd.append("--no-camera")

            if effective_config.enable_telegram:
                cmd.append("--enable-telegram")
                if effective_config.telegram_chat_id:
                    cmd.extend(["--telegram-chat-id", str(effective_config.telegram_chat_id)])
            else:
                cmd.append("--no-telegram")

            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

            cwd = self.path_sandbox.workspace_roots[0] if (self.path_sandbox and self.path_sandbox.workspace_roots) else os.getcwd()
            log_path = os.path.join(storage_root, "daemon.log")
            log_file = open(log_path, "a", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                creationflags=creationflags,
                close_fds=True,
                stdout=log_file,
                stderr=log_file,
            )

            # Wait briefly for daemon.pid to be written
            pid_file = os.path.join(storage_root, "daemon.pid")
            daemon_data = {"session_id": session_id, "pid": proc.pid, "detached": True}
            for _ in range(30):
                if os.path.exists(pid_file):
                    try:
                        with open(pid_file, "r", encoding="utf-8") as f:
                            d = json.load(f)
                            if d.get("pid"):
                                daemon_data.update(d)
                                break
                    except Exception:
                        pass
                time.sleep(0.1)

            self._record_audit(
                "SESSION_START_SUCCESS",
                True,
                f"Detached capture session '{session_id}' activated by '{caller}' (PID: {proc.pid})",
            )
            return True, f"Capture session '{session_id}' is now active.", daemon_data

        # In-process execution (for tests and attached mode)
        with self._lock:
            # 4. Construct and initialize session
            session = CaptureSession(
                session_id=session_id,
                config=effective_config,
                session_dir=session_dir,
                created_by=caller,
                authorized_task_id=tid,
                created_at=time.time(),
            )

            try:
                session.initialize_directories(self.path_sandbox)
            except Exception as e:
                err = f"Failed to initialize capture session directories: {str(e)}"
                self._record_audit("SESSION_INIT_FAILED", False, err)
                return False, err, {"error": err}

            session.state = SessionState.STARTING
            session.started_at = time.time()
            session.write_metadata()
            session.write_manifest()

            # 5. Initialize Telegram delivery queue & adapter if enabled
            if effective_config.enable_telegram:
                self._delivery_queue = BoundedDeliveryQueue(
                    max_pending_artifacts=effective_config.max_pending_artifacts,
                    max_pending_bytes=effective_config.max_pending_bytes,
                )
                self._delivery_adapter = TelegramDeliveryAdapter(
                    queue=self._delivery_queue,
                    audit_ledger=self.audit_ledger,
                    default_chat_id=effective_config.telegram_chat_id,
                    max_retries=effective_config.max_retry_attempts,
                )
                self._delivery_adapter.start()
            else:
                self._delivery_queue = None
                self._delivery_adapter = None

            # 6. Initialize and start capture workers
            if effective_config.enable_display:
                self._display_worker = DisplayCaptureWorker(
                    session=session,
                    computer_controller=self.computer_controller,
                    delivery_queue=self._delivery_queue,
                    quota_manager=self.quota_manager,
                    path_sandbox=self.path_sandbox,
                    audit_ledger=self.audit_ledger,
                )
                self._display_worker.start()

            if effective_config.enable_camera:
                self._camera_worker = CameraCaptureWorker(
                    session=session,
                    camera_engine=self.camera_engine,
                    delivery_queue=self._delivery_queue,
                    quota_manager=self.quota_manager,
                    path_sandbox=self.path_sandbox,
                    audit_ledger=self.audit_ledger,
                )
                self._camera_worker.start()

            session.state = SessionState.ACTIVE
            session.write_metadata()
            self._current_session = session

            # 7. Record in audit ledger
            self._record_audit(
                "SESSION_START_SUCCESS",
                True,
                f"Capture session '{session_id}' activated by '{caller}' (task: {tid})",
                session=session,
            )

            return True, f"Capture session '{session_id}' is now active.", session.to_dict()

    def stop_session(
        self,
        task_id: Optional[str] = None,
        caller_id: Optional[str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Stop the active capture session (in-process or background daemon),
        flush artifacts and manifests, and terminate all capture threads/processes cleanly.
        """
        # 1. First check if a detached daemon is running
        try:
            from session_capture.cli import get_daemon_info, is_process_alive, cmd_stop
            daemon_info = get_daemon_info()
            if daemon_info and is_process_alive(daemon_info.get("pid", 0)):
                sess_id = daemon_info.get("session_id", "detached")
                cmd_stop(passkey="stop")
                self._record_audit(
                    "SESSION_STOP_SUCCESS",
                    True,
                    f"Detached capture session '{sess_id}' stopped via stop signal.",
                )
                return True, f"Capture session '{sess_id}' stopped.", {
                    "session_id": sess_id,
                    "state": "stopped",
                    "detached": True,
                }
        except Exception as e:
            logger.error(f"Error checking or stopping detached daemon: {e}")

        # 2. Check and stop in-process session
        with self._lock:
            if not self._current_session or self._current_session.state in (
                SessionState.INACTIVE,
                SessionState.STOPPED,
            ):
                return False, "No capture session is currently active.", self.get_status()

            session = self._current_session

            # Check authorization if a specific task_id is enforced
            if task_id and session.authorized_task_id != task_id and caller_id != "owner":
                err = f"Unauthorized: task '{task_id}' cannot terminate session authorized for '{session.authorized_task_id}'"
                self._record_audit("UNAUTHORIZED_STOP_ATTEMPT", False, err)
                return False, err, self.get_status()

            session.state = SessionState.STOPPING

            # Stop capture workers cleanly
            if self._display_worker:
                self._display_worker.stop()
                self._display_worker.join(timeout=3.0)
                self._display_worker = None

            if self._camera_worker:
                self._camera_worker.stop()
                self._camera_worker.join(timeout=3.0)
                self._camera_worker = None

            # Stop delivery adapter cleanly
            if self._delivery_adapter:
                self._delivery_adapter.stop(timeout=2.0)
                self._delivery_adapter = None

            session.stopped_at = time.time()
            session.state = SessionState.STOPPED
            session.write_manifest()
            session.write_metadata()

            # Record in audit ledger
            self._record_audit(
                "SESSION_STOP_SUCCESS",
                True,
                f"Capture session '{session.session_id}' stopped. Artifacts captured: {len(session.artifacts)}",
                session=session,
            )

            res_dict = session.to_dict()
            return True, f"Capture session '{session.session_id}' stopped.", res_dict

    def get_status(self) -> Dict[str, Any]:
        """Return structured real-time status representation."""
        with self._lock:
            if self._current_session:
                s = self._current_session
                queue_status = self._delivery_queue.get_status() if self._delivery_queue else {}
                camera_avail = self._camera_worker.camera_available if self._camera_worker else True
                delivered_count = sum(1 for a in s.artifacts if getattr(a, "delivery_state", None) == "delivered")

                return {
                    "active": s.state == SessionState.ACTIVE,
                    "state": s.state.value,
                    "session_id": s.session_id,
                    "created_by": s.created_by,
                    "authorized_task_id": s.authorized_task_id,
                    "started_at": s.started_at,
                    "stopped_at": s.stopped_at,
                    "display_capture_enabled": s.config.enable_display,
                    "camera_capture_enabled": s.config.enable_camera,
                    "display_interval": s.config.display_interval,
                    "camera_interval": s.config.camera_interval,
                    "artifacts_created": len(s.artifacts),
                    "artifacts_delivered": delivered_count,
                    "pending_deliveries": queue_status.get("pending_count", 0),
                    "storage_used": s.get_total_bytes(),
                    "camera_available": camera_avail,
                }

        # Check detached daemon info
        try:
            from session_capture.cli import get_daemon_info, is_process_alive
            info = get_daemon_info()
            if info and is_process_alive(info.get("pid", 0)):
                return {
                    "active": True,
                    "state": "active",
                    "session_id": info.get("session_id"),
                    "created_by": info.get("caller_id", "owner"),
                    "authorized_task_id": info.get("task_id"),
                    "started_at": info.get("started_at"),
                    "stopped_at": None,
                    "display_capture_enabled": True,
                    "camera_capture_enabled": True,
                    "display_interval": info.get("display_interval"),
                    "camera_interval": info.get("camera_interval"),
                    "artifacts_created": 0,
                    "artifacts_delivered": 0,
                    "pending_deliveries": 0,
                    "storage_used": 0,
                    "camera_available": True,
                    "pid": info.get("pid"),
                    "detached": True,
                }
        except Exception:
            pass

        return {
            "active": False,
            "state": SessionState.INACTIVE.value,
            "session_id": None,
            "created_by": None,
            "authorized_task_id": None,
            "started_at": None,
            "stopped_at": None,
            "display_capture_enabled": False,
            "camera_capture_enabled": False,
            "display_interval": None,
            "camera_interval": None,
            "artifacts_created": 0,
            "artifacts_delivered": 0,
            "pending_deliveries": 0,
            "storage_used": 0,
            "camera_available": True,
        }

    def _record_audit(
        self,
        decision_code: str,
        allowed: bool,
        reason: str,
        session: Optional[CaptureSession] = None,
    ) -> None:
        if not self.audit_ledger:
            return
        decision = PermissionDecision(
            allowed=allowed,
            requires_confirmation=False,
            reason=reason,
            decision_code=decision_code,
            risk_level=RiskLevel.MEDIUM.value if allowed else RiskLevel.HIGH.value,
        )
        args = {}
        if session:
            args["session_id"] = session.session_id
            args["created_by"] = session.created_by
            args["authorized_task_id"] = session.authorized_task_id
            args["config_hash"] = session.config_hash
            args["artifact_count"] = len(session.artifacts)

        try:
            self.audit_ledger.record(
                tool_name="session.capture.lifecycle",
                arguments=args,
                risk_level=RiskLevel.MEDIUM,
                decision=decision,
            )
        except Exception as e:
            logger.error(f"Failed to record session lifecycle audit: {e}")


# Application-level factory helper (avoids hard singleton; can be reset/injected in tests)
_global_session_capture_manager: Optional[SessionCaptureManager] = None


def get_session_capture_manager(
    permission_engine: Optional[PermissionEngine] = None,
    reset: bool = False,
) -> SessionCaptureManager:
    """
    Factory function returning the application-level SessionCaptureManager instance.
    Supports resetting or supplying explicit dependencies.
    """
    global _global_session_capture_manager
    if reset or _global_session_capture_manager is None:
        _global_session_capture_manager = SessionCaptureManager(
            permission_engine=permission_engine,
        )
    return _global_session_capture_manager
