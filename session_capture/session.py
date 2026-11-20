"""
Capture Session model, lifecycle states, and atomic manifest persistence.
"""
from dataclasses import asdict, dataclass
import enum
import json
import os
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

from security.sandbox import PathOperation, PathSandbox
from session_capture.artifacts import CaptureArtifact
from session_capture.config import SessionCaptureConfig


class SessionState(str, enum.Enum):
    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class CaptureSession:
    """
    Manages metadata, state transitions, filesystem directories, and atomic
    manifest records for a single owner-authorized capture session.
    """

    def __init__(
        self,
        session_id: str,
        config: SessionCaptureConfig,
        session_dir: str,
        created_by: str = "owner",
        authorized_task_id: str = "default",
        created_at: Optional[float] = None,
    ):
        self.session_id = session_id
        self.config = config
        self.session_dir = session_dir
        self.display_dir = os.path.join(session_dir, "display")
        self.camera_dir = os.path.join(session_dir, "camera")
        self.created_by = created_by
        self.authorized_task_id = authorized_task_id
        self.created_at = created_at or time.time()
        self.started_at: Optional[float] = None
        self.stopped_at: Optional[float] = None
        self.state: SessionState = SessionState.INACTIVE
        self.config_hash: str = config.compute_hash()
        self.artifacts: List[CaptureArtifact] = []
        self._lock = threading.RLock()

    def initialize_directories(self, path_sandbox: Optional[PathSandbox] = None) -> None:
        """
        Validate paths against PathSandbox and create directory structure.
        """
        with self._lock:
            if path_sandbox is not None:
                res = path_sandbox.validate_path(self.session_dir, PathOperation.WRITE)
                if not res.allowed:
                    raise PermissionError(f"Session directory violates PathSandbox: {res.reason}")

            os.makedirs(self.session_dir, exist_ok=True)
            os.makedirs(self.display_dir, exist_ok=True)
            os.makedirs(self.camera_dir, exist_ok=True)

    def add_artifact(self, artifact: CaptureArtifact) -> None:
        """Append an artifact record and flush manifest atomically."""
        with self._lock:
            self.artifacts.append(artifact)
            self.write_manifest()

    def get_total_bytes(self) -> int:
        """Sum of all artifact bytes captured in this session."""
        with self._lock:
            return sum(art.file_size_bytes for art in self.artifacts)

    def write_metadata(self) -> None:
        """
        Atomically write session metadata to metadata.json.
        """
        with self._lock:
            metadata_path = os.path.join(self.session_dir, "metadata.json")
            data = {
                "session_id": self.session_id,
                "created_by": self.created_by,
                "authorized_task_id": self.authorized_task_id,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "stopped_at": self.stopped_at,
                "state": self.state.value,
                "config_hash": self.config_hash,
                "configuration": self.config.to_dict(),
            }
            self._atomic_write_json(metadata_path, data)

    def write_manifest(self) -> None:
        """
        Atomically write manifest.json containing metadata and full artifact index with SHA-256 hashes.
        """
        with self._lock:
            manifest_path = os.path.join(self.session_dir, "manifest.json")
            data = {
                "session_id": self.session_id,
                "state": self.state.value,
                "artifact_count": len(self.artifacts),
                "total_bytes": self.get_total_bytes(),
                "artifacts": [art.to_dict() for art in self.artifacts],
            }
            self._atomic_write_json(manifest_path, data)

    def _atomic_write_json(self, target_path: str, data: Dict[str, Any]) -> None:
        """
        Write JSON to a temporary file in the same directory, then atomic rename.
        Prevents file corruption on unexpected process shutdown.
        """
        parent_dir = os.path.dirname(target_path)
        os.makedirs(parent_dir, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(dir=parent_dir, prefix=".tmp_", suffix=".json")
        try:
            with open(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(temp_path, target_path)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            raise

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "created_by": self.created_by,
                "authorized_task_id": self.authorized_task_id,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "stopped_at": self.stopped_at,
                "state": self.state.value,
                "session_dir": self.session_dir,
                "config_hash": self.config_hash,
                "artifact_count": len(self.artifacts),
                "total_bytes": self.get_total_bytes(),
                "configuration": self.config.to_dict(),
            }
