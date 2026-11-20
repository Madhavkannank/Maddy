"""
Scoped Permission Elevation and Capability Lease Engine for M.A.D.D.Y.
Provides temporary, bounded authorization for elevated operations without perpetual over-privilege.
Leases specify tool scope, resource/path patterns, time-to-live (TTL), and use caps.
"""
import fnmatch
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ElevationLease:
    """
    A temporary, capability-scoped authorization token.
    """
    lease_id: str
    tool_pattern: str
    path_pattern: Optional[str] = None
    granted_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 300.0)  # Default 5 min TTL
    max_uses: Optional[int] = 1  # Default single-use lease
    use_count: int = 0
    revoked: bool = False
    originating_authorization: str = "user_confirmed"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_active(self, now: Optional[float] = None) -> bool:
        """Check if lease is unrevoked, unexpired, and under max use cap."""
        current_time = now if now is not None else time.time()
        if self.revoked:
            return False
        if current_time >= self.expires_at:
            return False
        if self.max_uses is not None and self.use_count >= self.max_uses:
            return False
        return True

    def matches(self, tool_name: str, target_path: Optional[str] = None, now: Optional[float] = None) -> bool:
        """
        Evaluate whether this lease covers the given tool and resource path.
        """
        if not self.is_active(now=now):
            return False

        # Match tool name using glob pattern (e.g. 'filesystem.write' or 'filesystem.*')
        if not fnmatch.fnmatch(tool_name.lower(), self.tool_pattern.lower()):
            return False

        # Match path if path pattern is specified
        if self.path_pattern is not None:
            if not target_path:
                return False
            norm_target = os.path.normpath(target_path)
            norm_pattern = os.path.normpath(self.path_pattern)
            if sys.platform == "win32":
                norm_target = norm_target.lower()
                norm_pattern = norm_pattern.lower()

            if not fnmatch.fnmatch(norm_target, norm_pattern):
                # Also check if target is within the directory specified by pattern
                clean_pat = norm_pattern.rstrip("*").rstrip("\\").rstrip("/")
                if not norm_target.startswith(clean_pat):
                    return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "tool_pattern": self.tool_pattern,
            "path_pattern": self.path_pattern,
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
            "max_uses": self.max_uses,
            "use_count": self.use_count,
            "revoked": self.revoked,
            "originating_authorization": self.originating_authorization,
            "metadata": self.metadata,
        }


class ElevationManager:
    """
    Thread-safe manager for issuing, validating, consuming, and revoking capability leases.
    """

    def __init__(self):
        self._leases: Dict[str, ElevationLease] = {}
        self._lock = threading.RLock()

    def grant_lease(
        self,
        tool_pattern: str,
        path_pattern: Optional[str] = None,
        duration_seconds: float = 300.0,
        max_uses: Optional[int] = 1,
        originating_authorization: str = "user_confirmed",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ElevationLease:
        """
        Issue a new scoped elevation lease.
        """
        with self._lock:
            lease_id = str(uuid.uuid4())
            now = time.time()
            lease = ElevationLease(
                lease_id=lease_id,
                tool_pattern=tool_pattern,
                path_pattern=path_pattern,
                granted_at=now,
                expires_at=now + max(0.001, float(duration_seconds)),
                max_uses=max_uses,
                use_count=0,
                revoked=False,
                originating_authorization=originating_authorization,
                metadata=dict(metadata or {}),
            )
            self._leases[lease_id] = lease
            return lease

    def find_matching_lease(
        self,
        tool_name: str,
        target_path: Optional[str] = None,
    ) -> Optional[ElevationLease]:
        """
        Find an active lease authorizing the requested tool and path.
        """
        with self._lock:
            now = time.time()
            for lease in self._leases.values():
                if lease.matches(tool_name, target_path, now=now):
                    return lease
            return None

    def consume_lease(
        self,
        lease_id: str,
        tool_name: str,
        target_path: Optional[str] = None,
    ) -> bool:
        """
        Consume one use of the matching lease.
        Returns True if consumed successfully, False if lease is invalid, expired, or doesn't match.
        """
        with self._lock:
            lease = self._leases.get(lease_id)
            if not lease:
                return False

            now = time.time()
            if not lease.matches(tool_name, target_path, now=now):
                return False

            lease.use_count += 1
            return True

    def revoke_lease(self, lease_id: str) -> bool:
        """
        Explicitly revoke an active lease.
        """
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease:
                lease.revoked = True
                return True
            return False

    def revoke_all(self) -> int:
        """
        Revoke all active leases immediately.
        """
        with self._lock:
            count = 0
            for lease in self._leases.values():
                if not lease.revoked:
                    lease.revoked = True
                    count += 1
            return count

    def get_lease(self, lease_id: str) -> Optional[ElevationLease]:
        with self._lock:
            return self._leases.get(lease_id)

    def list_active_leases(self) -> List[ElevationLease]:
        with self._lock:
            now = time.time()
            return [l for l in self._leases.values() if l.is_active(now=now)]

    def cleanup_expired(self) -> int:
        """Prune expired and exhausted leases from memory."""
        with self._lock:
            now = time.time()
            expired_ids = [lid for lid, l in self._leases.items() if not l.is_active(now=now)]
            for lid in expired_ids:
                del self._leases[lid]
            return len(expired_ids)
