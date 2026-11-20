"""
Configuration and validation for M.A.D.D.Y. Session Capture subsystem.
Enforces strict bounds on capture intervals, storage limits, and delivery queues
to prevent denial-of-service, runaway disk usage, and model-generated manipulation.
"""
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from security.sandbox import PathOperation, PathSandbox

MIN_CAPTURE_INTERVAL: float = 1.0        # 1 second minimum
MAX_CAPTURE_INTERVAL: float = 3600.0     # 1 hour maximum
DEFAULT_CAPTURE_INTERVAL: float = 30.0   # 30 seconds default

MIN_STORAGE_BYTES: int = 5 * 1024 * 1024           # 5 MB minimum
MAX_STORAGE_BYTES: int = 2000 * 1024 * 1024        # 2 GB maximum
DEFAULT_STORAGE_BYTES: int = 200 * 1024 * 1024     # 200 MB default

MIN_PENDING_ARTIFACTS: int = 1
MAX_PENDING_ARTIFACTS: int = 200
DEFAULT_PENDING_ARTIFACTS: int = 50

MIN_PENDING_BYTES: int = 1 * 1024 * 1024           # 1 MB
MAX_PENDING_BYTES: int = 200 * 1024 * 1024         # 200 MB
DEFAULT_PENDING_BYTES: int = 50 * 1024 * 1024      # 50 MB

MAX_RETRY_ATTEMPTS: int = 3


@dataclass
class SessionCaptureConfig:
    """
    Strongly typed, validated configuration for an authorized capture session.
    """
    display_interval: float = DEFAULT_CAPTURE_INTERVAL
    camera_interval: float = DEFAULT_CAPTURE_INTERVAL
    enable_display: bool = True
    enable_camera: bool = True
    enable_telegram: bool = False
    telegram_chat_id: Optional[str] = None
    base_storage_dir: Optional[str] = None
    max_storage_bytes: int = DEFAULT_STORAGE_BYTES
    max_pending_artifacts: int = DEFAULT_PENDING_ARTIFACTS
    max_pending_bytes: int = DEFAULT_PENDING_BYTES
    max_retry_attempts: int = MAX_RETRY_ATTEMPTS

    def __post_init__(self) -> None:
        import os
        if self.enable_telegram and not self.telegram_chat_id:
            chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
            if chat_id:
                self.telegram_chat_id = chat_id

    def validate(self, path_sandbox: Optional[PathSandbox] = None) -> None:
        """
        Validate all parameters against strict bounds.
        Raises ValueError if any parameter is invalid, out of bounds, or violates sandboxing.
        """
        # 1. Interval bounds
        if not isinstance(self.display_interval, (int, float)) or not (
            MIN_CAPTURE_INTERVAL <= self.display_interval <= MAX_CAPTURE_INTERVAL
        ):
            raise ValueError(
                f"display_interval must be between {MIN_CAPTURE_INTERVAL} and {MAX_CAPTURE_INTERVAL} seconds, "
                f"got {self.display_interval}"
            )

        if not isinstance(self.camera_interval, (int, float)) or not (
            MIN_CAPTURE_INTERVAL <= self.camera_interval <= MAX_CAPTURE_INTERVAL
        ):
            raise ValueError(
                f"camera_interval must be between {MIN_CAPTURE_INTERVAL} and {MAX_CAPTURE_INTERVAL} seconds, "
                f"got {self.camera_interval}"
            )

        # 2. Storage bounds
        if not isinstance(self.max_storage_bytes, int) or not (
            MIN_STORAGE_BYTES <= self.max_storage_bytes <= MAX_STORAGE_BYTES
        ):
            raise ValueError(
                f"max_storage_bytes must be between {MIN_STORAGE_BYTES} and {MAX_STORAGE_BYTES} bytes, "
                f"got {self.max_storage_bytes}"
            )

        # 3. Delivery queue bounds
        if not isinstance(self.max_pending_artifacts, int) or not (
            MIN_PENDING_ARTIFACTS <= self.max_pending_artifacts <= MAX_PENDING_ARTIFACTS
        ):
            raise ValueError(
                f"max_pending_artifacts must be between {MIN_PENDING_ARTIFACTS} and {MAX_PENDING_ARTIFACTS}, "
                f"got {self.max_pending_artifacts}"
            )

        if not isinstance(self.max_pending_bytes, int) or not (
            MIN_PENDING_BYTES <= self.max_pending_bytes <= MAX_PENDING_BYTES
        ):
            raise ValueError(
                f"max_pending_bytes must be between {MIN_PENDING_BYTES} and {MAX_PENDING_BYTES} bytes, "
                f"got {self.max_pending_bytes}"
            )

        if not isinstance(self.max_retry_attempts, int) or self.max_retry_attempts < 0 or self.max_retry_attempts > 10:
            raise ValueError(f"max_retry_attempts must be between 0 and 10, got {self.max_retry_attempts}")

        # 4. Telegram destination format
        if self.telegram_chat_id is not None:
            chat_str = str(self.telegram_chat_id).strip()
            if not chat_str or not re.match(r"^[-@\w]+$", chat_str):
                raise ValueError(f"Invalid telegram_chat_id format: '{self.telegram_chat_id}'")

        # 5. Base storage directory sandbox validation
        if self.base_storage_dir is not None:
            if not self.base_storage_dir.strip():
                raise ValueError("base_storage_dir cannot be empty string")
            if path_sandbox is not None:
                res = path_sandbox.validate_path(self.base_storage_dir, PathOperation.WRITE)
                if not res.allowed:
                    raise ValueError(f"base_storage_dir violates PathSandbox: {res.reason}")

        # 6. Ensure at least one capture channel is enabled
        if not self.enable_display and not self.enable_camera:
            raise ValueError("At least one capture stream (display or camera) must be enabled")

    def compute_hash(self) -> str:
        """Deterministically compute SHA-256 digest of configuration."""
        data = asdict(self)
        canonical = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
