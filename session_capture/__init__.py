"""
Session Capture subsystem for M.A.D.D.Y.
Provides owner-authorized, sandboxed, periodic display and camera capture
with cryptographic artifact integrity and independent bounded Telegram delivery.
"""
from session_capture.artifacts import (
    ArtifactType,
    CaptureArtifact,
    DeliveryState,
)
from session_capture.config import (
    DEFAULT_CAPTURE_INTERVAL,
    DEFAULT_PENDING_ARTIFACTS,
    DEFAULT_PENDING_BYTES,
    DEFAULT_STORAGE_BYTES,
    MAX_CAPTURE_INTERVAL,
    MAX_PENDING_ARTIFACTS,
    MAX_PENDING_BYTES,
    MAX_RETRY_ATTEMPTS,
    MAX_STORAGE_BYTES,
    MIN_CAPTURE_INTERVAL,
    MIN_PENDING_ARTIFACTS,
    MIN_PENDING_BYTES,
    MIN_STORAGE_BYTES,
    SessionCaptureConfig,
)
from session_capture.delivery import TelegramDeliveryAdapter
from session_capture.manager import (
    SessionCaptureManager,
    get_session_capture_manager,
)
from session_capture.queue import BoundedDeliveryQueue
from session_capture.session import CaptureSession, SessionState
from session_capture.workers import CameraCaptureWorker, DisplayCaptureWorker

__all__ = [
    "SessionState",
    "ArtifactType",
    "DeliveryState",
    "CaptureArtifact",
    "SessionCaptureConfig",
    "CaptureSession",
    "BoundedDeliveryQueue",
    "TelegramDeliveryAdapter",
    "DisplayCaptureWorker",
    "CameraCaptureWorker",
    "SessionCaptureManager",
    "get_session_capture_manager",
    "MIN_CAPTURE_INTERVAL",
    "MAX_CAPTURE_INTERVAL",
    "DEFAULT_CAPTURE_INTERVAL",
    "MIN_STORAGE_BYTES",
    "MAX_STORAGE_BYTES",
    "DEFAULT_STORAGE_BYTES",
    "MIN_PENDING_ARTIFACTS",
    "MAX_PENDING_ARTIFACTS",
    "DEFAULT_PENDING_ARTIFACTS",
    "MIN_PENDING_BYTES",
    "MAX_PENDING_BYTES",
    "DEFAULT_PENDING_BYTES",
    "MAX_RETRY_ATTEMPTS",
]
