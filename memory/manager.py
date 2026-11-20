"""
Intelligent Long-Term Memory Manager for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Unifies storage, relation checks, deduplication, relevance retrieval ranking,
conflict resolution, and user privacy commands.
"""
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from memory.relation import apply_relation_mutation, classify_memory_relation
from memory.schema import (
    CreatedBy,
    MemoryConfidence,
    MemoryMutationAudit,
    MemoryProvenance,
    MemoryRecord,
    MemoryRelation,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    SourceType,
)
from memory.store import JsonFileMemoryStore, MemoryStore
from memory.user_preferences import UserPreferences
from memory.episodic_memory import EpisodicMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    High-level orchestrator for long-term memory operations in M.A.D.D.Y.
    """

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        user_preferences: Optional[UserPreferences] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        sync_legacy: bool = False,
    ):
        self.store = store or JsonFileMemoryStore()
        self.user_preferences = user_preferences or UserPreferences()
        self.episodic_memory = episodic_memory or EpisodicMemory()
        if sync_legacy:
            self._sync_legacy_preferences_if_empty()


    def _sync_legacy_preferences_if_empty(self) -> None:
        """Seed long-term memory from legacy user_preferences if long_term store is empty."""
        try:
            existing = self.store.list_records(limit=1)
            if existing:
                return  # Already has records

            legacy_prefs = self.user_preferences.get_all_preferences()
            for key, val in legacy_prefs.items():
                if isinstance(val, (str, int, float, bool)):
                    self.remember(
                        content=f"Preference: {key} is {val}",
                        memory_type=MemoryType.USER_PREFERENCE,
                        scope=MemoryScope.GLOBAL,
                        source_type=SourceType.USER_MESSAGE,
                        source_reference="legacy_preferences",
                        source_text=f"{key}={val}",
                        created_by=CreatedBy.USER,
                        confidence=MemoryConfidence.ESTABLISHED_PREFERENCE.value,
                        importance=0.8,
                        entities=[key],
                        tags=["preference", "legacy"],
                    )
        except Exception as e:
            logger.warning(f"Legacy preferences sync warning: {e}")

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC_FACT,
        scope: MemoryScope = MemoryScope.GLOBAL,
        source_type: SourceType = SourceType.USER_MESSAGE,
        source_reference: str = "",
        source_text: str = "",
        created_by: CreatedBy = CreatedBy.USER,
        confidence: float = 1.0,
        importance: float = 0.5,
        entities: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        expires_in_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        """
        Create, classify, and store a validated memory record with relation checking.
        """
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Memory content cannot be empty.")

        now = time.time()
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        extracted_entities = list(entities or [])
        if not extracted_entities:
            # Simple keyword entity extraction excluding stop entities
            stop_entities = {
                "User", "The", "A", "An", "Preference", "Sir", "Boss", "My", "I",
                "He", "She", "It", "They", "We", "Is", "Are", "In", "On", "At",
                "To", "For", "Of", "With", "By", "And", "Or", "Not"
            }
            words = re.findall(r"\b[A-Z][a-z0-9_\-]+\b", clean_content)
            extracted_entities = [w for w in set(words) if w not in stop_entities and len(w) > 2]

        provenance = MemoryProvenance(
            source_type=source_type,
            source_reference=source_reference or "direct_input",
            source_text=source_text or clean_content,
            created_by=created_by,
            created_at=now,
        )

        expires_at = (now + expires_in_seconds) if expires_in_seconds is not None else None

        candidate = MemoryRecord(
            memory_id=memory_id,
            memory_type=memory_type,
            scope=scope,
            status=MemoryStatus.ACTIVE,
            content=clean_content,
            provenance=provenance,
            confidence=max(0.0, min(1.0, confidence)),
            importance=max(0.0, min(1.0, importance)),
            entities=extracted_entities,
            tags=list(tags or []),
            created_at=now,
            updated_at=now,
            last_accessed=now,
            access_count=0,
            expires_at=expires_at,
            metadata=dict(metadata or {}),
        )

        # Relation classification against active records
        existing_records = self.store.list_records(scope=scope, status=MemoryStatus.ACTIVE, limit=200)
        relation, target_record, reason = classify_memory_relation(candidate, existing_records)

        # Deterministic state transition
        result = apply_relation_mutation(
            store=self.store,
            candidate=candidate,
            relation=relation,
            target_record=target_record,
            reason=reason,
        )

        # Synchronize user name preference if a name memory is stored or updated
        if hasattr(self, "user_preferences") and self.user_preferences:
            m_name = re.match(r"^User name is\s+(.+)$", result.content, re.IGNORECASE)
            if m_name:
                new_name = m_name.group(1).strip()
                self.user_preferences.set_preference("user.real_name", new_name)

        return result

    def retrieve_relevant(
        self,
        query: str,
        scope: Optional[MemoryScope] = None,
        max_records: int = 5,
        min_score: float = 0.35,
    ) -> List[MemoryRecord]:
        """
        Relevance-driven memory search across active records.
        Strictly excludes memories below `min_score` to prevent context leakage.
        """
        if not query or not query.strip():
            return []

        search_results = self.store.search(
            query=query,
            scope=scope,
            limit=max_records * 2,
            include_inactive=False,
        )

        relevant: List[MemoryRecord] = []
        for rec in search_results:
            # Score check
            rec_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]+\b", rec.content.lower()))
            query_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]+\b", query.lower()))
            overlap = len(rec_tokens.intersection(query_tokens))

            # Multi-factor score
            score = (overlap / max(1, len(query_tokens))) * 0.6 + (rec.confidence * 0.2) + (rec.importance * 0.2)
            if score >= min_score:
                rec.touch()
                self.store.update(rec)
                relevant.append(rec)
                if len(relevant) >= max_records:
                    break

        return relevant

    def what_do_you_remember(
        self,
        scope: Optional[MemoryScope] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> List[Dict[str, Any]]:
        """List active memories for user privacy inspection."""
        records = self.store.list_records(scope=scope, memory_type=memory_type, status=MemoryStatus.ACTIVE)
        return [
            {
                "id": rec.memory_id,
                "content": rec.content,
                "type": rec.memory_type.value,
                "scope": rec.scope.value,
                "confidence": rec.confidence,
                "created_at": rec.created_at,
                "access_count": rec.access_count,
            }
            for rec in records
        ]

    def why_do_you_remember(self, memory_id_or_topic: str) -> Optional[Dict[str, Any]]:
        """Explain the provenance of a specific memory."""
        target = memory_id_or_topic.strip()
        rec = self.store.get(target)

        if not rec:
            # Try searching by topic
            matches = self.store.search(query=target, limit=1)
            if matches:
                rec = matches[0]

        if not rec:
            return None

        return {
            "memory_id": rec.memory_id,
            "content": rec.content,
            "source_type": rec.provenance.source_type.value,
            "source_text": rec.provenance.source_text,
            "source_reference": rec.provenance.source_reference,
            "created_by": rec.provenance.created_by.value,
            "created_at": rec.provenance.created_at,
            "confidence": rec.confidence,
            "access_count": rec.access_count,
            "history": [m.to_dict() for m in rec.mutation_history],
        }

    def forget(self, target_query_or_id: str) -> bool:
        """User command to delete a memory by ID or topic."""
        target = target_query_or_id.strip()
        if not target:
            return False

        exact = self.store.get(target)
        if exact:
            return self.store.delete(exact.memory_id, soft_delete=True)

        matches = self.store.search(query=target, limit=50)
        if not matches:
            return False

        deleted_any = False
        for rec in matches:
            deleted_any = self.store.delete(rec.memory_id, soft_delete=True) or deleted_any
        return deleted_any

    # ----------------------------------------------------------------------
    # Native Markdown Memory Integration (USER.md & MEMORY.md)
    # ----------------------------------------------------------------------
    @staticmethod
    def _resolve_md_path(filename: str) -> str:
        """Resolve path to USER.md or MEMORY.md across dual-boot / workspace."""
        from config.settings import get_memory_dir
        mem_dir = get_memory_dir()
        candidate = os.path.join(mem_dir, filename)
        if os.path.exists(candidate):
            return candidate

        local_candidate = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(local_candidate):
            return local_candidate

        return candidate

    def get_user_md_content(self) -> str:
        """Read USER.md content for Volatile prompt tier."""
        path = self._resolve_md_path("USER.md")
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to read USER.md from {path}: {e}")
        return ""

    def get_memory_md_content(self) -> str:
        """Read MEMORY.md content for Volatile prompt tier."""
        path = self._resolve_md_path("MEMORY.md")
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to read MEMORY.md from {path}: {e}")
        return ""

    def append_memory_note(self, note: str) -> None:
        """Append a learned system fact or project note to MEMORY.md safely."""
        path = self._resolve_md_path("MEMORY.md")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"\n- {note.strip()}")
        except Exception as e:
            logger.error(f"Failed to append to MEMORY.md at {path}: {e}")

    def append_user_preference(self, pref: str) -> None:
        """Append an explicit user preference or rule to USER.md safely."""
        path = self._resolve_md_path("USER.md")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"\n- {pref.strip()}")
        except Exception as e:
            logger.error(f"Failed to append to USER.md at {path}: {e}")

