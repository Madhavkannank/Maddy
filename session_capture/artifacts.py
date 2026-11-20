"""
Artifact definitions, types, and cryptographic integrity verification for Session Capture.
"""
from dataclasses import asdict, dataclass
import enum
import hashlib
import os
from typing import Any, Dict, Optional


class ArtifactType(str, enum.Enum):
    DISPLAY = "display"
    CAMERA = "camera"


class DeliveryState(str, enum.Enum):
    NOT_REQUESTED = "not_requested"
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CaptureArtifact:
    """
    Metadata representation of an individual captured display or camera snapshot.
    Includes SHA-256 digest for tamper-detection and integrity verification.
    """
    artifact_id: str
    session_id: str
    artifact_type: ArtifactType
    file_path: str
    timestamp: float
    sequence_number: int
    file_size_bytes: int
    sha256: str
    delivery_state: DeliveryState = DeliveryState.NOT_REQUESTED
    delivery_error: Optional[str] = None
    retry_count: int = 0

    @classmethod
    def compute_file_sha256(cls, file_path: str) -> str:
        """
        Compute deterministic SHA-256 digest of an on-disk artifact file.
        Reads in chunks of 64 KB to remain memory-efficient on large frames.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Artifact file not found for integrity hashing: {file_path}")

        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "session_id": self.session_id,
            "artifact_type": self.artifact_type.value if isinstance(self.artifact_type, ArtifactType) else str(self.artifact_type),
            "file_path": self.file_path,
            "timestamp": self.timestamp,
            "sequence_number": self.sequence_number,
            "file_size_bytes": self.file_size_bytes,
            "sha256": self.sha256,
            "delivery_state": self.delivery_state.value if isinstance(self.delivery_state, DeliveryState) else str(self.delivery_state),
            "delivery_error": self.delivery_error,
            "retry_count": self.retry_count,
        }
