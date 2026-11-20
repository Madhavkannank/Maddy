"""
Storage abstraction and thread-safe atomic JSON file store for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You) memory records.
"""
from abc import ABC, abstractmethod
import json
import logging
import os
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

from memory.schema import (
    CreatedBy,
    MemoryConfidence,
    MemoryMutationAudit,
    MemoryProvenance,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    SourceType,
)

logger = logging.getLogger(__name__)


class MemoryStore(ABC):
    """Abstract base class for long-term memory record storage."""

    @abstractmethod
    def save(self, record: MemoryRecord) -> None:
        """Persist a new memory record."""
        pass

    @abstractmethod
    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        """Fetch a record by memory_id."""
        pass

    @abstractmethod
    def update(self, record: MemoryRecord) -> None:
        """Update an existing memory record."""
        pass

    @abstractmethod
    def delete(self, memory_id: str, soft_delete: bool = True) -> bool:
        """Delete or mark as DELETED a memory record."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        scope: Optional[MemoryScope] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        include_inactive: bool = False,
    ) -> List[MemoryRecord]:
        """Search records by keyword relevance."""
        pass

    @abstractmethod
    def list_records(
        self,
        scope: Optional[MemoryScope] = None,
        memory_type: Optional[MemoryType] = None,
        status: Optional[MemoryStatus] = None,
        limit: int = 100,
    ) -> List[MemoryRecord]:
        """List records matching specified filters."""
        pass


class JsonFileMemoryStore(MemoryStore):
    """
    Thread-safe, atomic disk storage for M.A.D.D.Y. memory records.
    Uses .tmp write + os.replace to guarantee zero file corruption.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        if storage_dir:
            self.storage_dir = storage_dir
        else:
            from config.settings import get_memory_dir
            self.storage_dir = os.path.join(get_memory_dir(), "long_term")

        self.storage_file = os.path.join(self.storage_dir, "records.json")
        self._lock = threading.RLock()
        self._ensure_dir()
        self._records: Dict[str, MemoryRecord] = {}
        self._load_records()

    def _ensure_dir(self) -> None:
        os.makedirs(self.storage_dir, exist_ok=True)

    def _load_records(self) -> None:
        """Load records from disk atomically."""
        with self._lock:
            self._records.clear()
            if not os.path.exists(self.storage_file):
                return
            try:
                with open(self.storage_file, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    if isinstance(raw_data, list):
                        for item in raw_data:
                            try:
                                rec = MemoryRecord.from_dict(item)
                                self._records[rec.memory_id] = rec
                            except Exception as e:
                                logger.warning(f"Failed to parse memory record: {e}")
                    elif isinstance(raw_data, dict):
                        for mid, item in raw_data.items():
                            try:
                                rec = MemoryRecord.from_dict(item)
                                self._records[mid] = rec
                            except Exception as e:
                                logger.warning(f"Failed to parse memory record {mid}: {e}")
            except Exception as e:
                logger.error(f"Failed reading memory records file {self.storage_file}: {e}")

    def _save_to_disk(self) -> None:
        """Atomic write to disk using a temporary file."""
        with self._lock:
            data = [rec.to_dict() for rec in self._records.values()]
            temp_file = None
            try:
                dir_name = os.path.dirname(self.storage_file)
                fd, temp_file = tempfile.mkstemp(prefix="mem_", suffix=".tmp", dir=dir_name)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                os.replace(temp_file, self.storage_file)
            except Exception as e:
                logger.error(f"Atomic memory write failed for {self.storage_file}: {e}")
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except OSError:
                        pass
                raise

    def save(self, record: MemoryRecord) -> None:
        """Save a new record."""
        with self._lock:
            record.updated_at = time.time()
            self._records[record.memory_id] = record
            self._save_to_disk()

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        """Fetch record by ID."""
        with self._lock:
            return self._records.get(memory_id)

    def update(self, record: MemoryRecord) -> None:
        """Update existing record."""
        with self._lock:
            record.updated_at = time.time()
            self._records[record.memory_id] = record
            self._save_to_disk()

    def delete(self, memory_id: str, soft_delete: bool = True) -> bool:
        """Delete a record."""
        with self._lock:
            rec = self._records.get(memory_id)
            if not rec:
                return False

            if soft_delete:
                rec.status = MemoryStatus.DELETED
                rec.updated_at = time.time()
                rec.mutation_history.append(
                    MemoryMutationAudit(
                        mutation_type="DELETE",
                        timestamp=time.time(),
                        reason="Soft deleted by user or system",
                        affected_memory_id=memory_id,
                    )
                )
            else:
                del self._records[memory_id]

            self._save_to_disk()
            return True

    def list_records(
        self,
        scope: Optional[MemoryScope] = None,
        memory_type: Optional[MemoryType] = None,
        status: Optional[MemoryStatus] = None,
        limit: int = 100,
    ) -> List[MemoryRecord]:
        """List records matching query filters."""
        with self._lock:
            results = []
            now = time.time()
            for rec in self._records.values():
                # Check status
                if status is not None:
                    if rec.status != status:
                        continue
                else:
                    # Default: only active
                    if not rec.is_active(now):
                        continue

                # Check scope
                if scope is not None and rec.scope != scope:
                    continue

                # Check type
                if memory_type is not None and rec.memory_type != memory_type:
                    continue

                results.append(rec)
                if len(results) >= limit:
                    break

            return results

    def search(
        self,
        query: str,
        scope: Optional[MemoryScope] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        include_inactive: bool = False,
    ) -> List[MemoryRecord]:
        """
        Search memory records by query terms across content, entities, and tags.
        Calculates term frequency score and recency.
        """
        with self._lock:
            if not query or not query.strip():
                return self.list_records(scope=scope, memory_type=memory_type, limit=limit)

            terms = [t.lower() for t in query.split() if len(t) > 1]
            scored_records = []
            now = time.time()

            for rec in self._records.values():
                if not include_inactive and not rec.is_active(now):
                    continue

                if scope is not None and rec.scope != scope:
                    continue

                if memory_type is not None and rec.memory_type != memory_type:
                    continue

                # Searchable text
                content_lower = rec.content.lower()
                entities_lower = [e.lower() for e in rec.entities]
                tags_lower = [t.lower() for t in rec.tags]

                score = 0.0
                for term in terms:
                    if term in content_lower:
                        score += 2.0
                    if any(term == e or term in e for e in entities_lower):
                        score += 3.0
                    if any(term == t for t in tags_lower):
                        score += 1.5

                if score > 0:
                    # Weight by confidence and importance
                    final_score = score * rec.confidence * (0.5 + 0.5 * rec.importance)
                    scored_records.append((final_score, rec.updated_at, rec))

            scored_records.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return [item[2] for item in scored_records[:limit]]
