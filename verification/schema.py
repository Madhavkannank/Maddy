"""
Verification Engine Schema Definitions for M.A.D.D.Y.
Defines assertion types, verification outcomes, structured assertions,
results, and comprehensive verification reports for environment grounding.
"""
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional

class AssertionType(Enum):
    # Filesystem assertions
    FILE_EXISTS = "FILE_EXISTS"
    FILE_NOT_EXISTS = "FILE_NOT_EXISTS"
    FILE_CONTAINS = "FILE_CONTAINS"
    FILE_SIZE_MIN = "FILE_SIZE_MIN"
    DIR_EXISTS = "DIR_EXISTS"

    # Process & Window assertions
    PROCESS_RUNNING = "PROCESS_RUNNING"
    PROCESS_NOT_RUNNING = "PROCESS_NOT_RUNNING"
    WINDOW_TITLE_EXISTS = "WINDOW_TITLE_EXISTS"

    # Visual & Screen assertions
    SCREEN_CHANGED = "SCREEN_CHANGED"
    VISUAL_ELEMENT_PRESENT = "VISUAL_ELEMENT_PRESENT"

    # Command & Execution assertions
    EXIT_CODE_ZERO = "EXIT_CODE_ZERO"
    STDOUT_CONTAINS = "STDOUT_CONTAINS"

    # Network assertions
    HTTP_STATUS_200 = "HTTP_STATUS_200"
    URL_REACHABLE = "URL_REACHABLE"

    # Custom assertion
    CUSTOM = "CUSTOM"

class VerificationOutcome(Enum):
    VERIFIED_SUCCESS = "VERIFIED_SUCCESS"
    VERIFIED_FAILURE = "VERIFIED_FAILURE"
    INDETERMINATE = "INDETERMINATE"
    UNVERIFIED = "UNVERIFIED"

@dataclass
class Assertion:
    """
    A concrete verifiable assertion about the operating environment.
    """
    assertion_type: AssertionType
    target: str
    expected_value: Any = None
    description: str = ""
    strict: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assertion_type": self.assertion_type.value,
            "target": self.target,
            "expected_value": self.expected_value,
            "description": self.description,
            "strict": self.strict,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Assertion":
        return cls(
            assertion_type=AssertionType(data["assertion_type"]),
            target=str(data.get("target", "")),
            expected_value=data.get("expected_value"),
            description=str(data.get("description", "")),
            strict=bool(data.get("strict", True)),
        )

@dataclass
class AssertionResult:
    """
    The outcome of evaluating an Assertion against real environment ground truth.
    """
    assertion: Assertion
    passed: bool
    actual_value: Any
    evidence: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assertion": self.assertion.to_dict(),
            "passed": self.passed,
            "actual_value": self.actual_value,
            "evidence": self.evidence,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssertionResult":
        return cls(
            assertion=Assertion.from_dict(data["assertion"]),
            passed=bool(data.get("passed", False)),
            actual_value=data.get("actual_value"),
            evidence=str(data.get("evidence", "")),
            timestamp=float(data.get("timestamp", time.time())),
        )

@dataclass
class VerificationReport:
    """
    Structured verification report summarizing all assertion outcomes,
    authoritative ground-truth evidence, and whether execution can safely proceed.
    """
    target_id: str
    outcome: VerificationOutcome
    results: List[AssertionResult] = field(default_factory=list)
    evidence_summary: str = ""
    can_proceed: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "outcome": self.outcome.value,
            "results": [r.to_dict() for r in self.results],
            "evidence_summary": self.evidence_summary,
            "can_proceed": self.can_proceed,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationReport":
        return cls(
            target_id=str(data.get("target_id", "")),
            outcome=VerificationOutcome(data.get("outcome", VerificationOutcome.UNVERIFIED.value)),
            results=[AssertionResult.from_dict(r) for r in data.get("results", [])],
            evidence_summary=str(data.get("evidence_summary", "")),
            can_proceed=bool(data.get("can_proceed", False)),
            timestamp=float(data.get("timestamp", time.time())),
        )
