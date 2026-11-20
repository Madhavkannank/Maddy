"""
Standalone background daemon for Owner-Activated Session Capture.
Runs independently of interactive M.A.D.D.Y. processes so capture continues even
if the M.A.D.D.Y. console or voice assistant is closed.
Terminates only upon receipt of explicit stop signals, passkeys, or CLI stop commands.
"""
import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from security.audit_ledger import AuditLedger
from security.permission_engine import PermissionEngine
from security.policy import PermissionDecision
from security.quotas import ResourceQuotaManager
from security.sandbox import PathSandbox
from session_capture.config import SessionCaptureConfig
from session_capture.delivery import TelegramDeliveryAdapter
from session_capture.queue import BoundedDeliveryQueue
from session_capture.session import CaptureSession, SessionState
from session_capture.workers import CameraCaptureWorker, DisplayCaptureWorker
from tools.base import RiskLevel
from tools.camera import CameraEngine
from computer.controller import ComputerController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [SessionCaptureDaemon] %(message)s",
)
logger = logging.getLogger("session_capture.daemon")

PID_FILENAME = "daemon.pid"
STOP_SIGNAL_FILENAME = "stop.signal"
LOG_FILENAME = "daemon.log"


class SessionCaptureDaemon:
    """
    Manages background workers and lifecycle for a detached capture session.
    """

    def __init__(
        self,
        config: SessionCaptureConfig,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        caller_id: str = "owner",
        base_storage_dir: Optional[str] = None,
    ):
        self.config = config
        self.sandbox = PathSandbox(allowed_workspace_roots=[PROJECT_ROOT])
        self.quota_manager = ResourceQuotaManager()
        self.audit_ledger = AuditLedger()
        self.computer_controller = ComputerController()
        self.camera_engine = CameraEngine()

        if base_storage_dir:
            self.base_storage_dir = base_storage_dir
        else:
            self.base_storage_dir = os.path.join(PROJECT_ROOT, "memory", "capture_sessions")

        os.makedirs(self.base_storage_dir, exist_ok=True)
        self.pid_file = os.path.join(self.base_storage_dir, PID_FILENAME)
        self.stop_signal_file = os.path.join(self.base_storage_dir, STOP_SIGNAL_FILENAME)

        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = session_id or f"cap_{ts_str}_{os.urandom(3).hex()}"
        self.task_id = task_id or f"daemon_{int(time.time())}"
        self.caller_id = caller_id
        self.session_dir = os.path.join(self.base_storage_dir, self.session_id)

        self.session = CaptureSession(
            session_id=self.session_id,
            config=self.config,
            session_dir=self.session_dir,
            created_by=self.caller_id,
            authorized_task_id=self.task_id,
            created_at=time.time(),
        )

        self._display_worker: Optional[DisplayCaptureWorker] = None
        self._camera_worker: Optional[CameraCaptureWorker] = None
        self._delivery_queue: Optional[BoundedDeliveryQueue] = None
        self._delivery_adapter: Optional[TelegramDeliveryAdapter] = None
        self._is_running = False

    def _write_pid(self) -> None:
        """Write current PID and metadata for CLI and assistant status inspection."""
        data = {
            "pid": os.getpid(),
            "session_id": self.session_id,
            "task_id": self.task_id,
            "caller_id": self.caller_id,
            "started_at": self.session.started_at or time.time(),
            "display_interval": self.config.display_interval,
            "camera_interval": self.config.camera_interval,
            "enable_telegram": self.config.enable_telegram,
            "telegram_chat_id": self.config.telegram_chat_id,
        }
        with open(self.pid_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _cleanup_pid(self) -> None:
        """Remove PID and stop signal files on shutdown."""
        for path in (self.pid_file, self.stop_signal_file):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    def start(self) -> None:
        """Start workers and enter execution loop."""
        logger.info(f"Initializing daemon for session '{self.session_id}' (PID: {os.getpid()})")
        self.session.initialize_directories(self.sandbox)
        self.session.state = SessionState.STARTING
        self.session.started_at = time.time()
        self.session.write_metadata()
        self.session.write_manifest()

        # Clean any stale stop signal before starting
        if os.path.exists(self.stop_signal_file):
            try:
                os.remove(self.stop_signal_file)
            except Exception:
                pass

        self._write_pid()

        # 1. Initialize Telegram delivery if enabled
        if self.config.enable_telegram:
            self._delivery_queue = BoundedDeliveryQueue(
                max_pending_artifacts=self.config.max_pending_artifacts,
                max_pending_bytes=self.config.max_pending_bytes,
            )
            self._delivery_adapter = TelegramDeliveryAdapter(
                queue=self._delivery_queue,
                audit_ledger=self.audit_ledger,
                default_chat_id=self.config.telegram_chat_id,
                max_retries=self.config.max_retry_attempts,
            )
            self._delivery_adapter.start()
            logger.info("Telegram live delivery adapter started.")

        # 2. Start capture workers
        if self.config.enable_display:
            self._display_worker = DisplayCaptureWorker(
                session=self.session,
                computer_controller=self.computer_controller,
                delivery_queue=self._delivery_queue,
                quota_manager=self.quota_manager,
                path_sandbox=self.sandbox,
                audit_ledger=self.audit_ledger,
            )
            self._display_worker.start()
            logger.info(f"Display capture worker active (interval: {self.config.display_interval}s).")

        if self.config.enable_camera:
            self._camera_worker = CameraCaptureWorker(
                session=self.session,
                camera_engine=self.camera_engine,
                delivery_queue=self._delivery_queue,
                quota_manager=self.quota_manager,
                path_sandbox=self.sandbox,
                audit_ledger=self.audit_ledger,
            )
            self._camera_worker.start()
            logger.info(f"Camera capture worker active (interval: {self.config.camera_interval}s).")

        self.session.state = SessionState.ACTIVE
        self.session.write_metadata()
        self._is_running = True

        self._record_audit(
            "DAEMON_SESSION_START",
            True,
            f"Decoupled capture session '{self.session_id}' running as PID {os.getpid()}",
        )

        logger.info(f"Session '{self.session_id}' is now fully active in the background.")

    def run(self) -> None:
        """Run blocking main loop checking for termination signals or stop files."""
        self.start()

        # Handle standard OS shutdown signals
        def _signal_handler(signum, frame):
            logger.info(f"Received termination signal {signum}, stopping daemon...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        try:
            while self._is_running:
                # Check for file-based stop signal (from CLI or M.A.D.D.Y. voice dismiss)
                if os.path.exists(self.stop_signal_file):
                    logger.info("Detected stop signal file, initiating graceful shutdown...")
                    break
                time.sleep(0.5)
        except Exception as e:
            logger.error(f"Unexpected exception in daemon main loop: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully stop all workers, flush artifacts, and write final metadata."""
        if not self._is_running:
            return
        self._is_running = False
        logger.info(f"Stopping session capture daemon '{self.session_id}'...")

        self.session.state = SessionState.STOPPING

        if self._display_worker:
            self._display_worker.stop()
            self._display_worker.join(timeout=3.0)
            self._display_worker = None

        if self._camera_worker:
            self._camera_worker.stop()
            self._camera_worker.join(timeout=3.0)
            self._camera_worker = None

        if self._delivery_adapter:
            self._delivery_adapter.stop(timeout=3.0)
            self._delivery_adapter = None

        self.session.stopped_at = time.time()
        self.session.state = SessionState.STOPPED
        self.session.write_manifest()
        self.session.write_metadata()

        self._record_audit(
            "DAEMON_SESSION_STOP",
            True,
            f"Capture session '{self.session_id}' cleanly stopped. Total artifacts: {len(self.session.artifacts)}",
        )

        self._cleanup_pid()
        logger.info(f"Daemon stopped cleanly. Manifest written to {self.session_dir}")

    def _record_audit(self, decision_code: str, allowed: bool, reason: str) -> None:
        decision = PermissionDecision(
            allowed=allowed,
            requires_confirmation=False,
            reason=reason,
            decision_code=decision_code,
            risk_level=RiskLevel.LOW.value,
        )
        try:
            self.audit_ledger.record(
                tool_name="session.capture.daemon",
                arguments={"session_id": self.session_id, "pid": os.getpid()},
                risk_level=RiskLevel.LOW,
                decision=decision,
            )
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="M.A.D.D.Y. Session Capture Background Daemon")
    parser.add_argument("--session-id", type=str, default=None, help="Explicit session identifier")
    parser.add_argument("--task-id", type=str, default=None, help="Authorizing task ID")
    parser.add_argument("--caller-id", type=str, default="owner", help="Caller identity")
    parser.add_argument("--display-interval", type=float, default=30.0, help="Display capture interval (s)")
    parser.add_argument("--camera-interval", type=float, default=30.0, help="Camera capture interval (s)")
    parser.add_argument("--enable-display", action="store_true", default=True, help="Enable display capture")
    parser.add_argument("--no-display", dest="enable_display", action="store_false", help="Disable display capture")
    parser.add_argument("--enable-camera", action="store_true", default=True, help="Enable camera capture")
    parser.add_argument("--no-camera", dest="enable_camera", action="store_false", help="Disable camera capture")
    parser.add_argument("--enable-telegram", action="store_true", default=None, help="Enable live Telegram sync")
    parser.add_argument("--no-telegram", dest="enable_telegram", action="store_false", help="Disable Telegram sync")
    parser.add_argument("--telegram-chat-id", type=str, default=None, help="Explicit Telegram chat ID")
    parser.add_argument("--storage-dir", type=str, default=None, help="Base storage directory")
    args = parser.parse_args()

    # Resolve Telegram auto-enable from environment if not explicit
    enable_tg = args.enable_telegram
    if enable_tg is None:
        try:
            from integrations.messaging.telegram_client import TelegramClient
        except ImportError:
            from integrations.messaging import TelegramClient
        enable_tg = TelegramClient().is_configured

    chat_id = args.telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip() or None

    config = SessionCaptureConfig(
        display_interval=args.display_interval,
        camera_interval=args.camera_interval,
        enable_display=args.enable_display,
        enable_camera=args.enable_camera,
        enable_telegram=bool(enable_tg),
        telegram_chat_id=chat_id,
    )

    daemon = SessionCaptureDaemon(
        config=config,
        session_id=args.session_id,
        task_id=args.task_id,
        caller_id=args.caller_id,
        base_storage_dir=args.storage_dir,
    )
    daemon.run()


if __name__ == "__main__":
    main()
