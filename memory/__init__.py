"""
Memory module package initializer for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Provides persistent and transient memory architectures across Windows and Fedora Linux dual-boot setup.
"""
from memory.working_memory import WorkingMemory, ScratchpadItem
from memory.task_memory import TaskMemory
from memory.episodic_memory import EpisodicMemory
from memory.user_preferences import UserPreferences
from memory.context_engine import ContextCompressor
from memory.schema import (
    MemoryRecord,
    MemoryProvenance,
    MemoryMutationAudit,
    MemoryType,
    MemoryScope,
    MemoryStatus,
    MemoryConfidence,
    SourceType,
    CreatedBy,
    MemoryRelation,
)
from memory.store import MemoryStore, JsonFileMemoryStore
from memory.relation import classify_memory_relation, apply_relation_mutation
from memory.manager import MemoryManager

__all__ = [
    "WorkingMemory",
    "ScratchpadItem",
    "TaskMemory",
    "EpisodicMemory",
    "UserPreferences",
    "ContextCompressor",
    "MemoryRecord",
    "MemoryProvenance",
    "MemoryMutationAudit",
    "MemoryType",
    "MemoryScope",
    "MemoryStatus",
    "MemoryConfidence",
    "SourceType",
    "CreatedBy",
    "MemoryRelation",
    "MemoryStore",
    "JsonFileMemoryStore",
    "classify_memory_relation",
    "apply_relation_mutation",
    "MemoryManager",
]
