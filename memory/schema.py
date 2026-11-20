"""
Memory schema definitions, enums, and dataclasses for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Standard-library dataclasses with pure Python typing.
"""
from dataclasses import dataclass, field
import enum
import time
from typing import Any, Dict, List, Optional


class MemoryType(str, enum.Enum):
    USER_PREFERENCE = "USER_PREFERENCE"
    SEMANTIC_FACT = "SEMANTIC_FACT"
    SYSTEM_STATE = "SYSTEM_STATE"
    EPISODIC_EVENT = "EPISODIC_EVENT"
    TASK_STATE = "TASK_STATE"


class MemoryScope(str, enum.Enum):
    GLOBAL = "GLOBAL"
    PROJECT = "PROJECT"
    TASK = "TASK"
    SESSION = "SESSION"


class MemoryStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    EXPIRED = "EXPIRED"
    DELETED = "DELETED"


class MemoryConfidence(float, enum.Enum):
    EXPLICIT_INSTRUCTION = 1.0
    ESTABLISHED_PREFERENCE = 0.85
    OBSERVED_BEHAVIOR = 0.60
    MODEL_INFERENCE = 0.40


class SourceType(str, enum.Enum):
    USER_MESSAGE = "USER_MESSAGE"
    TOOL_OBSERVATION = "TOOL_OBSERVATION"
    SYSTEM_INSPECTION = "SYSTEM_INSPECTION"
    TASK_RESULT = "TASK_RESULT"
    MODEL_INFERENCE = "MODEL_INFERENCE"


class CreatedBy(str, enum.Enum):
    USER = "USER"
    MADDY = "MADDY"
    SYSTEM = "SYSTEM"


class MemoryRelation(str, enum.Enum):
    NEW = "NEW"
    REINFORCES = "REINFORCES"
    DUPLICATE = "DUPLICATE"
    UPDATES = "UPDATES"
    CONTRADICTS = "CONTRADICTS"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class MemoryProvenance:
    """Provenance tracking to answer 'Why do you remember that?'"""
    source_type: SourceType
    source_reference: str  # Conversation turn, task_id, tool_call_id
    source_text: str       # Original snippet / user quote
    created_by: CreatedBy
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type.value if hasattr(self.source_type, "value") else str(self.source_type),
            "source_reference": self.source_reference,
            "source_text": self.source_text,
            "created_by": self.created_by.value if hasattr(self.created_by, "value") else str(self.created_by),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryProvenance":
        return cls(
            source_type=SourceType(data.get("source_type", SourceType.USER_MESSAGE)),
            source_reference=data.get("source_reference", ""),
            source_text=data.get("source_text", ""),
            created_by=CreatedBy(data.get("created_by", CreatedBy.USER)),
            created_at=data.get("created_at", time.time()),
        )


@dataclass
class MemoryMutationAudit:
    """Auditable log of state transitions (superseding, deprecation, deletion)."""
    mutation_type: str  # "CREATE", "REINFORCE", "UPDATE", "SUPERSEDE", "DELETE"
    timestamp: float = field(default_factory=time.time)
    reason: str = ""
    affected_memory_id: Optional[str] = None
    replacing_memory_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mutation_type": self.mutation_type,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "affected_memory_id": self.affected_memory_id,
            "replacing_memory_id": self.replacing_memory_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryMutationAudit":
        return cls(
            mutation_type=data.get("mutation_type", "UNKNOWN"),
            timestamp=data.get("timestamp", time.time()),
            reason=data.get("reason", ""),
            affected_memory_id=data.get("affected_memory_id"),
            replacing_memory_id=data.get("replacing_memory_id"),
        )


@dataclass
class MemoryRecord:
    """Canonical representation of a persistent memory unit."""
    memory_id: str
    memory_type: MemoryType
    scope: MemoryScope
    status: MemoryStatus
    content: str
    provenance: MemoryProvenance
    confidence: float = 1.0
    importance: float = 0.5
    entities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    expires_at: Optional[float] = None
    superseded_by: Optional[str] = None
    superseded_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    mutation_history: List[MemoryMutationAudit] = field(default_factory=list)

    def is_active(self, current_time: Optional[float] = None) -> bool:
        """Check if memory is active and not expired."""
        if self.status != MemoryStatus.ACTIVE:
            return False
        if self.expires_at is not None:
            now = current_time if current_time is not None else time.time()
            if now > self.expires_at:
                return False
        return True

    def touch(self) -> None:
        """Increment access count and update last_accessed timestamp."""
        self.last_accessed = time.time()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type.value if hasattr(self.memory_type, "value") else str(self.memory_type),
            "scope": self.scope.value if hasattr(self.scope, "value") else str(self.scope),
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            "content": self.content,
            "provenance": self.provenance.to_dict(),
            "confidence": self.confidence,
            "importance": self.importance,
            "entities": self.entities,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "expires_at": self.expires_at,
            "superseded_by": self.superseded_by,
            "superseded_at": self.superseded_at,
            "metadata": self.metadata,
            "mutation_history": [m.to_dict() for m in self.mutation_history],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
        return cls(
            memory_id=data["memory_id"],
            memory_type=MemoryType(data.get("memory_type", MemoryType.SEMANTIC_FACT)),
            scope=MemoryScope(data.get("scope", MemoryScope.GLOBAL)),
            status=MemoryStatus(data.get("status", MemoryStatus.ACTIVE)),
            content=data.get("content", ""),
            provenance=MemoryProvenance.from_dict(data.get("provenance", {})),
            confidence=float(data.get("confidence", 1.0)),
            importance=float(data.get("importance", 0.5)),
            entities=list(data.get("entities", [])),
            tags=list(data.get("tags", [])),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            last_accessed=float(data.get("last_accessed", time.time())),
            access_count=int(data.get("access_count", 0)),
            expires_at=float(data["expires_at"]) if data.get("expires_at") is not None else None,
            superseded_by=data.get("superseded_by"),
            superseded_at=float(data["superseded_at"]) if data.get("superseded_at") is not None else None,
            metadata=dict(data.get("metadata", {})),
            mutation_history=[MemoryMutationAudit.from_dict(m) for m in data.get("mutation_history", [])],
        )
