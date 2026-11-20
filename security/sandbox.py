"""
Path Sandboxing and Filesystem Boundary Enforcement Engine for M.A.D.D.Y.
Enforces strict filesystem security across Windows and Fedora Linux platforms.
Protects against path traversal, symlink/junction escapes, non-existent target tricks,
and sensitive credential access while differentiating READ, WRITE, DELETE, and EXECUTE operations.
"""
import enum
import os
import pathlib
import sys
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


class PathOperation(enum.Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    LIST = "list"


class SandboxViolationType(enum.Enum):
    TRAVERSAL_ATTEMPT = "traversal_attempt"
    OUTSIDE_WORKSPACE = "outside_workspace"
    FORBIDDEN_SYSTEM_PATH = "forbidden_system_path"
    SENSITIVE_CREDENTIAL_PATH = "sensitive_credential_path"
    RESERVED_DEVICE_NAME = "reserved_device_name"
    SYMLINK_ESCAPE = "symlink_escape"
    CANONICALIZATION_FAILURE = "canonicalization_failure"


@dataclass
class PathSandboxResult:
    allowed: bool
    canonical_path: str
    operation: PathOperation
    reason: str
    violation_type: Optional[SandboxViolationType] = None

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "canonical_path": self.canonical_path,
            "operation": self.operation.value,
            "reason": self.reason,
            "violation_type": self.violation_type.value if self.violation_type else None,
        }


class PathSandbox:
    """
    Filesystem sandbox enforcer providing deterministic, fail-closed path boundary validation.
    """

    WINDOWS_RESERVED_NAMES: Set[str] = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }

    # Sensitive credential targets that must never be read, modified, or deleted without elevation
    SENSITIVE_CREDENTIAL_PATTERNS: List[str] = [
        ".env",
        ".env.local",
        ".env.production",
        ".env.development",
        ".git/config",
        ".git/credentials",
        ".netrc",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
        "id_dsa",
    ]

    # Sensitive user directories
    SENSITIVE_USER_DIRS: List[str] = [
        os.path.expanduser("~/.ssh"),
        os.path.expanduser("~/.aws"),
        os.path.expanduser("~/.gnupg"),
        os.path.expanduser("~/.kube"),
        os.path.expanduser("~/.config/credentials"),
        os.path.expanduser("~/.docker"),
    ]

    def __init__(
        self,
        allowed_workspace_roots: Optional[List[str]] = None,
        temporary_roots: Optional[List[str]] = None,
        allow_system_read: bool = True,
        strict_mode: bool = True,
    ):
        """
        Initialize PathSandbox.
        :param allowed_workspace_roots: Directories within which read/write/delete are allowed.
        :param temporary_roots: Scratch and temp directories allowed for transient files.
        :param allow_system_read: If True, system libraries and docs are readable but not writable.
        :param strict_mode: If True, unknown path conditions fail-closed.
        """
        self.strict_mode = strict_mode
        self.allow_system_read = allow_system_read

        # 1. Setup workspace roots (canonicalized)
        raw_workspaces = list(allowed_workspace_roots) if allowed_workspace_roots is not None else [os.getcwd()]
        self.workspace_roots: List[str] = []
        for ws in raw_workspaces:
            if ws and os.path.exists(ws):
                self.workspace_roots.append(self._canonicalize_existing(ws))
            elif ws:
                # If workspace dir does not exist yet, normalize it
                self.workspace_roots.append(self._normalize_path(os.path.abspath(ws)))

        # 2. Setup temporary roots
        raw_temps = list(temporary_roots) if temporary_roots is not None else [
            tempfile.gettempdir(),
            os.path.join(os.getcwd(), "scratch"),
        ]
        self.temporary_roots: List[str] = []
        for t in raw_temps:
            if t and os.path.exists(t):
                self.temporary_roots.append(self._canonicalize_existing(t))
            elif t:
                self.temporary_roots.append(self._normalize_path(os.path.abspath(t)))

        # 3. System roots
        self.windows_system_roots: List[str] = [
            self._normalize_path("C:\\Windows"),
            self._normalize_path("C:\\Program Files"),
            self._normalize_path("C:\\Program Files (x86)"),
            self._normalize_path("C:\\ProgramData"),
        ]
        # Include system drive root if available
        sys_drive = os.environ.get("SystemDrive", "C:")
        if not sys_drive.endswith("\\"):
            sys_drive += "\\"
        self.windows_system_roots.append(self._normalize_path(os.path.join(sys_drive, "Windows")))

        self.linux_system_roots: List[str] = [
            "/etc", "/proc", "/sys", "/dev", "/boot", "/root", "/var", "/usr",
        ]

    def _normalize_path(self, path: str) -> str:
        """
        Normalize path slashes and case (case-insensitive on Windows).
        """
        norm = os.path.normpath(path)
        if sys.platform == "win32":
            norm = os.path.normcase(norm)
        return norm

    def _canonicalize_existing(self, path: str) -> str:
        """
        Resolve realpath including symlinks and junctions.
        """
        real = os.path.realpath(path)
        return self._normalize_path(real)

    def _is_windows_reserved_device(self, path: str) -> bool:
        """
        Detect Windows DOS reserved device names (CON, PRN, AUX, NUL, COM1, etc.).
        """
        if sys.platform != "win32":
            return False
        
        parts = pathlib.Path(path).parts
        for part in parts:
            stem = part.split(".")[0].upper()
            if stem in self.WINDOWS_RESERVED_NAMES:
                return True
        return False

    def _is_sensitive_credential_target(self, canonical_path: str, raw_path: str) -> Tuple[bool, str]:
        """
        Detect if path targets credentials, private keys, or sensitive configs.
        """
        raw_lower = raw_path.lower().replace("\\", "/")
        canon_lower = canonical_path.lower().replace("\\", "/")
        base_name = os.path.basename(canonical_path).lower()

        # Check sensitive file patterns
        for pat in self.SENSITIVE_CREDENTIAL_PATTERNS:
            pat_lower = pat.lower()
            if base_name == pat_lower or raw_lower.endswith(pat_lower) or canon_lower.endswith(pat_lower):
                return True, f"Target matches sensitive credential pattern '{pat}'"

        # Check sensitive user directories
        for sdir in self.SENSITIVE_USER_DIRS:
            norm_sdir = self._normalize_path(sdir).lower().replace("\\", "/")
            if canon_lower == norm_sdir or canon_lower.startswith(norm_sdir + "/"):
                return True, f"Target is within sensitive directory '{sdir}'"

        return False, ""

    def _is_system_path(self, canonical_path: str) -> Tuple[bool, str]:
        """
        Check if canonical path points into core OS system roots.
        """
        canon_norm = self._normalize_path(canonical_path)

        if sys.platform == "win32":
            for s_root in self.windows_system_roots:
                if canon_norm == s_root or canon_norm.startswith(s_root + "\\"):
                    return True, s_root
        else:
            for s_root in self.linux_system_roots:
                if canon_norm == s_root or canon_norm.startswith(s_root + "/"):
                    return True, s_root

        return False, ""

    def _is_within_roots(self, canonical_path: str, roots: List[str]) -> bool:
        """
        Check if canonical path is within any of the provided canonical root directories.
        """
        canon_norm = self._normalize_path(canonical_path)
        sep = "\\" if sys.platform == "win32" else "/"

        for r in roots:
            r_norm = self._normalize_path(r)
            if canon_norm == r_norm:
                return True
            if not r_norm.endswith(sep):
                r_norm_sep = r_norm + sep
            else:
                r_norm_sep = r_norm
            if canon_norm.startswith(r_norm_sep):
                return True
        return False

    def resolve_target_path(self, path: str) -> Tuple[str, bool]:
        """
        Safely canonicalize a path, correctly handling non-existent targets.
        For non-existent paths:
        1. Identify nearest existing parent directory.
        2. Canonicalize the existing parent (resolving any symlinks/junctions up to that point).
        3. Resolve the remainder of the target path relative to the canonical parent.
        Returns (canonical_path, exists_flag).
        """
        if not path or not isinstance(path, str):
            raise ValueError("Path must be a non-empty string")

        abs_path = os.path.abspath(path)

        if os.path.exists(abs_path):
            canonical = self._canonicalize_existing(abs_path)
            return canonical, True

        # Non-existent target: traverse up to find nearest existing parent
        curr = abs_path
        unresolved_tail: List[str] = []

        while curr and not os.path.exists(curr):
            parent = os.path.dirname(curr)
            if parent == curr:
                # Reached filesystem root and nothing exists (highly unusual)
                break
            unresolved_tail.insert(0, os.path.basename(curr))
            curr = parent

        if curr and os.path.exists(curr):
            canonical_parent = self._canonicalize_existing(curr)
            canonical = os.path.join(canonical_parent, *unresolved_tail)
            canonical = self._normalize_path(canonical)
            return canonical, False

        # Fallback if no parent exists
        canonical = self._normalize_path(abs_path)
        return canonical, False

    def validate_path(self, path: str, operation: PathOperation = PathOperation.READ) -> PathSandboxResult:
        """
        Validate whether a requested path operation is permitted under sandbox policies.
        Enforces fail-closed evaluation.
        """
        if not path or not isinstance(path, str):
            return PathSandboxResult(
                allowed=False,
                canonical_path="",
                operation=operation,
                reason="Invalid path: target path must be a non-empty string",
                violation_type=SandboxViolationType.CANONICALIZATION_FAILURE,
            )

        # 1. Reject Windows reserved DOS device names
        if self._is_windows_reserved_device(path):
            return PathSandboxResult(
                allowed=False,
                canonical_path=path,
                operation=operation,
                reason=f"Target path '{path}' contains Windows reserved device name",
                violation_type=SandboxViolationType.RESERVED_DEVICE_NAME,
            )

        # 2. Canonicalize path handling existing and non-existent targets
        try:
            canonical_path, exists = self.resolve_target_path(path)
        except Exception as e:
            return PathSandboxResult(
                allowed=False,
                canonical_path=path,
                operation=operation,
                reason=f"Failed to resolve and canonicalize path '{path}': {str(e)}",
                violation_type=SandboxViolationType.CANONICALIZATION_FAILURE,
            )

        # 3. Check sensitive credentials (strictly blocked for all operations)
        is_sensitive, sensitive_reason = self._is_sensitive_credential_target(canonical_path, path)
        if is_sensitive:
            return PathSandboxResult(
                allowed=False,
                canonical_path=canonical_path,
                operation=operation,
                reason=f"Access denied: {sensitive_reason}",
                violation_type=SandboxViolationType.SENSITIVE_CREDENTIAL_PATH,
            )

        # 4. Check system paths
        is_system, system_root = self._is_system_path(canonical_path)
        if is_system:
            # Operation differentiation: WRITE and DELETE are strictly prohibited on system paths
            if operation in (PathOperation.WRITE, PathOperation.DELETE, PathOperation.EXECUTE):
                return PathSandboxResult(
                    allowed=False,
                    canonical_path=canonical_path,
                    operation=operation,
                    reason=f"Modification or execution in system directory '{system_root}' is strictly prohibited",
                    violation_type=SandboxViolationType.FORBIDDEN_SYSTEM_PATH,
                )

            # READ / LIST on system paths: permitted only if allow_system_read is active
            if self.allow_system_read and operation in (PathOperation.READ, PathOperation.LIST):
                return PathSandboxResult(
                    allowed=True,
                    canonical_path=canonical_path,
                    operation=operation,
                    reason=f"Read-only access to system directory '{system_root}' permitted by policy",
                )
            else:
                return PathSandboxResult(
                    allowed=False,
                    canonical_path=canonical_path,
                    operation=operation,
                    reason=f"Read access to system directory '{system_root}' is disabled",
                    violation_type=SandboxViolationType.FORBIDDEN_SYSTEM_PATH,
                )

        # 5. Check if within allowed workspace roots or temporary roots
        all_allowed_roots = self.workspace_roots + self.temporary_roots
        if self._is_within_roots(canonical_path, all_allowed_roots):
            return PathSandboxResult(
                allowed=True,
                canonical_path=canonical_path,
                operation=operation,
                reason="Target path is within allowed workspace/temporary boundaries",
            )

        # 6. If target is outside allowed workspace/temp roots
        if operation in (PathOperation.WRITE, PathOperation.DELETE):
            return PathSandboxResult(
                allowed=False,
                canonical_path=canonical_path,
                operation=operation,
                reason=f"Write/delete target '{canonical_path}' is outside allowed workspace boundaries",
                violation_type=SandboxViolationType.OUTSIDE_WORKSPACE,
            )

        # For READ/LIST outside workspace: if strict_mode is True, deny unless elevated
        if self.strict_mode:
            return PathSandboxResult(
                allowed=False,
                canonical_path=canonical_path,
                operation=operation,
                reason=f"Read target '{canonical_path}' is outside allowed workspace boundaries (strict mode)",
                violation_type=SandboxViolationType.OUTSIDE_WORKSPACE,
            )

        # Fail-closed fallback
        return PathSandboxResult(
            allowed=False,
            canonical_path=canonical_path,
            operation=operation,
            reason="Path boundary validation failed (fail-closed fallback)",
            violation_type=SandboxViolationType.OUTSIDE_WORKSPACE,
        )

    def is_safe_path(self, path: str, operation: PathOperation = PathOperation.READ) -> bool:
        """Helper returning boolean safe status."""
        return self.validate_path(path, operation).allowed
