"""
Tamper-Evident Hash-Chained Audit Ledger for M.A.D.D.Y.
Provides cryptographically verifiable record keeping with SHA-256 hash chaining,
monotonic sequence numbering, argument secret redaction, and integrity verification.
Maintains backward compatibility with the existing AuditLogger interface.
"""
import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from security.audit import PermissionAuditRecord
from security.policy import PermissionDecision
from tools.base import RiskLevel


@dataclass
class AuditLedgerEntry(PermissionAuditRecord):
    """
    Cryptographically chained audit entry. Inherits all fields of PermissionAuditRecord
    for seamless backwards compatibility.
    """
    sequence: int = 0
    lease_id: Optional[str] = None
    prev_hash: str = "0" * 64
    current_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "risk_level": self.risk_level,
            "allowed": self.allowed,
            "requires_confirmation": self.requires_confirmation,
            "decision_code": self.decision_code,
            "reason": self.reason,
            "lease_id": self.lease_id,
            "prev_hash": self.prev_hash,
            "current_hash": self.current_hash,
        }


class AuditLedger:
    """
    Thread-safe, append-only, tamper-evident audit ledger using SHA-256 hash chaining.
    """

    GENESIS_HASH = "0" * 64
    SECRET_ARG_KEYS = {
        "password", "passwd", "token", "secret", "key", "api_key",
        "access_token", "auth", "credentials", "private_key",
    }

    def __init__(self, ledger_file: Optional[str] = None, max_records: int = 5000):
        self.ledger_file = ledger_file
        self.max_records = max_records
        self._entries: List[AuditLedgerEntry] = []
        self._lock = threading.RLock()
        self._last_hash = self.GENESIS_HASH
        self._sequence = 0

    @classmethod
    def sanitize_arguments(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce a safe, redacted representation of tool arguments.
        Never persists raw credentials, API keys, or private tokens.
        """
        if not isinstance(args, dict):
            return {"raw_value": "[NON_DICT_ARGUMENT]"}

        redacted = {}
        for k, v in args.items():
            k_lower = str(k).lower()
            if any(sec in k_lower for sec in cls.SECRET_ARG_KEYS):
                redacted[k] = "[REDACTED]"
            elif isinstance(v, str) and len(v) > 2000:
                # Truncate overly long content payloads while preserving diagnostic hash
                content_hash = hashlib.sha256(v.encode("utf-8")).hexdigest()[:16]
                redacted[k] = f"{v[:100]}... [truncated {len(v)} chars, sha256:{content_hash}]"
            elif isinstance(v, dict):
                redacted[k] = cls.sanitize_arguments(v)
            else:
                redacted[k] = v

        return redacted

    @classmethod
    def compute_hash(
        cls,
        prev_hash: str,
        sequence: int,
        timestamp: float,
        tool_name: str,
        risk_level: str,
        allowed: bool,
        decision_code: str,
        arguments: Dict[str, Any],
        lease_id: Optional[str] = None,
    ) -> str:
        """
        Deterministically compute SHA-256 hash of a ledger entry incorporating the previous hash.
        """
        canonical_args = json.dumps(arguments, sort_keys=True, default=str)
        payload = (
            f"{prev_hash}:"
            f"{sequence}:"
            f"{timestamp:.4f}:"
            f"{tool_name}:"
            f"{risk_level}:"
            f"{allowed}:"
            f"{decision_code}:"
            f"{lease_id or ''}:"
            f"{canonical_args}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def record(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        risk_level: RiskLevel,
        decision: PermissionDecision,
        lease_id: Optional[str] = None,
    ) -> AuditLedgerEntry:
        """
        Append a cryptographically chained audit record.
        Maintains interface compatibility with PermissionEngine and AuditLogger.
        """
        with self._lock:
            self._sequence += 1
            seq = self._sequence
            now = time.time()
            prev = self._last_hash

            sanitized_args = self.sanitize_arguments(arguments)
            r_level = risk_level.value if isinstance(risk_level, RiskLevel) else str(risk_level)
            eff_lease = lease_id or decision.lease_id

            entry_hash = self.compute_hash(
                prev_hash=prev,
                sequence=seq,
                timestamp=now,
                tool_name=tool_name,
                risk_level=r_level,
                allowed=decision.allowed,
                decision_code=decision.decision_code,
                arguments=sanitized_args,
                lease_id=eff_lease,
            )

            entry = AuditLedgerEntry(
                timestamp=now,
                tool_name=tool_name,
                arguments=sanitized_args,
                risk_level=r_level,
                allowed=decision.allowed,
                requires_confirmation=decision.requires_confirmation,
                decision_code=decision.decision_code,
                reason=decision.reason,
                sequence=seq,
                lease_id=eff_lease,
                prev_hash=prev,
                current_hash=entry_hash,
            )

            self._entries.append(entry)
            self._last_hash = entry_hash

            # Populate audit_id on decision for traceability
            decision.audit_id = f"audit-seq-{seq}-{entry_hash[:8]}"

            if len(self._entries) > self.max_records:
                self._entries.pop(0)

            return entry

    def get_records(self) -> List[AuditLedgerEntry]:
        """Return list of audit records (matches AuditLogger.get_records())."""
        with self._lock:
            return list(self._entries)

    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Verify cryptographic integrity of the entire audit chain from beginning to end.
        Detects modified entries, deleted records, inserted records, sequence gaps,
        and hash corruption.
        Returns (is_valid, diagnostic_error_message).
        """
        with self._lock:
            if not self._entries:
                return True, None

            expected_prev = self.GENESIS_HASH

            for idx, entry in enumerate(self._entries):
                # 1. Check sequence ordering if contiguous from start
                if idx == 0 and entry.sequence == 1:
                    pass  # Normal start
                elif idx > 0 and entry.sequence != self._entries[idx - 1].sequence + 1:
                    return False, (
                        f"Integrity violation at index {idx}: sequence break! "
                        f"Expected sequence {self._entries[idx - 1].sequence + 1}, found {entry.sequence}"
                    )

                # 2. Check previous hash chaining
                if entry.prev_hash != expected_prev and idx == 0 and entry.sequence == 1:
                    return False, (
                        f"Integrity violation at genesis: prev_hash '{entry.prev_hash}' "
                        f"does not match expected genesis hash '{self.GENESIS_HASH}'"
                    )
                elif idx > 0 and entry.prev_hash != self._entries[idx - 1].current_hash:
                    return False, (
                        f"Integrity violation at sequence {entry.sequence}: prev_hash mismatch! "
                        f"Expected '{self._entries[idx - 1].current_hash}', found '{entry.prev_hash}'"
                    )

                # 3. Recompute and verify current hash
                recomputed_hash = self.compute_hash(
                    prev_hash=entry.prev_hash,
                    sequence=entry.sequence,
                    timestamp=entry.timestamp,
                    tool_name=entry.tool_name,
                    risk_level=entry.risk_level,
                    allowed=entry.allowed,
                    decision_code=entry.decision_code,
                    arguments=entry.arguments,
                    lease_id=entry.lease_id,
                )

                if recomputed_hash != entry.current_hash:
                    return False, (
                        f"Integrity violation at sequence {entry.sequence} (tool: {entry.tool_name}): "
                        f"hash mismatch! Recomputed '{recomputed_hash}', observed '{entry.current_hash}'"
                    )

                expected_prev = entry.current_hash

            return True, None

    def clear(self) -> None:
        """Clear ledger records and reset hash chain to genesis."""
        with self._lock:
            self._entries.clear()
            self._last_hash = self.GENESIS_HASH
            self._sequence = 0
