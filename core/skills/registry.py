"""Central Skill Registry for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).

Provides fast in-memory indexing, prompt catalog formatting, delta tracking,
and keyword search.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.skills.discovery import SkillDiscovery
from core.skills.models import SkillDefinition

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Thread-safe registry for discovered and runtime-registered skills."""

    def __init__(
        self,
        discovery: Optional[SkillDiscovery] = None,
        auto_discover: bool = True,
    ):
        self._lock = threading.RLock()
        self.discovery = discovery or SkillDiscovery()
        self._skills: Dict[str, SkillDefinition] = {}

        if auto_discover:
            self.reload()

    def reload(self) -> int:
        """Re-scan all search paths and update the active registry."""
        with self._lock:
            discovered = self.discovery.discover_all()
            self._skills = discovered
            logger.info("SkillRegistry discovered %d active skills.", len(self._skills))
            return len(self._skills)

    def register(self, skill_def: SkillDefinition) -> None:
        """Register or update a skill definition in the registry."""
        with self._lock:
            self._skills[skill_def.name] = skill_def

    def unregister(self, name: str) -> bool:
        """Remove a skill from the registry by name."""
        with self._lock:
            if name in self._skills:
                del self._skills[name]
                return True
            return False

    def get(self, name: str) -> Optional[SkillDefinition]:
        """Retrieve a skill definition by name."""
        with self._lock:
            return self._skills.get(name)

    def list_skills(self, category: Optional[str] = None) -> List[SkillDefinition]:
        """List all active skills, optionally filtered by category."""
        with self._lock:
            skills = list(self._skills.values())
            if category:
                cat_lower = category.lower().strip()
                skills = [s for s in skills if s.category.lower() == cat_lower]
            return sorted(skills, key=lambda s: s.name)

    def search(self, query: str) -> List[SkillDefinition]:
        """Search skills across name, description, tags, and category."""
        q = query.lower().strip()
        if not q:
            return self.list_skills()

        words = [w for w in re.findall(r"\w+", q) if len(w) >= 2]

        with self._lock:
            results: List[tuple[int, SkillDefinition]] = []
            for skill in self._skills.values():
                score = 0
                name_lower = skill.name.lower()
                desc_lower = skill.description.lower()
                cat_lower = skill.category.lower()
                tags_lower = [t.lower() for t in skill.tags]

                # Full query matches
                if q == name_lower:
                    score += 100
                elif q in name_lower:
                    score += 50

                if any(q == t for t in tags_lower):
                    score += 40
                elif any(q in t for t in tags_lower):
                    score += 20

                if q in cat_lower:
                    score += 30

                if q in desc_lower:
                    score += 25

                # Word-level matches for multi-word conversational queries
                for w in words:
                    if w == name_lower:
                        score += 30
                    elif w in name_lower:
                        score += 15
                    if any(w == t for t in tags_lower):
                        score += 20
                    elif any(w in t for t in tags_lower):
                        score += 10
                    if w in desc_lower:
                        score += 10
                    if w in cat_lower:
                        score += 15

                if score > 0:
                    results.append((score, skill))

            results.sort(key=lambda x: x[0], reverse=True)
            return [skill for _, skill in results]

    def get_content_hash(self) -> str:
        """Compute composite SHA-256 hash of all currently registered skills."""
        with self._lock:
            items = []
            for name in sorted(self._skills.keys()):
                s = self._skills[name]
                items.append(f"{name}:{s.version}:{s.content_hash}")
            summary = "\n".join(items)
            return hashlib.sha256(summary.encode("utf-8")).hexdigest()

    def get_available_skills_prompt_block(
        self,
        max_signatures: int = 15,
        user_query: Optional[str] = None,
    ) -> str:
        """Generate compact <available_skills> XML block for system prompt injection.

        - Preserves prompt-cache warmth by strictly sorting skills alphabetically by name.
        - Defends against prompt bloating: When registered skills exceed max_signatures,
          performs sparse relevance scoring against user_query to inject only the top
          relevant skills (default 5), advising the model to use `search_skills` for others.
        """
        with self._lock:
            if not self._skills:
                return ""

            all_sorted = sorted(self._skills.values(), key=lambda x: x.name)

            # If under threshold or no query provided, return deterministically sorted full catalog
            if len(all_sorted) <= max_signatures or not user_query:
                lines = ["<available_skills>"]
                for s in all_sorted:
                    tags_str = f" [{', '.join(s.tags[:4])}]" if s.tags else ""
                    lines.append(f"- {s.name} (v{s.version}): {s.description}{tags_str}")
                lines.append("</available_skills>")
                return "\n".join(lines)

            # Sparse Gating: Score skills against user_query
            scored = self.search(user_query)
            selected = scored[:5] if scored else all_sorted[:5]
            selected = sorted(selected, key=lambda x: x.name)

            lines = [f'<available_skills mode="sparse" total_available="{len(all_sorted)}">']
            for s in selected:
                tags_str = f" [{', '.join(s.tags[:4])}]" if s.tags else ""
                lines.append(f"- {s.name} (v{s.version}): {s.description}{tags_str}")
            remaining = len(all_sorted) - len(selected)
            if remaining > 0:
                lines.append(f"<!-- Note: {remaining} additional skills available via 'search_skills' or 'list_skills' -->")
            lines.append("</available_skills>")
            return "\n".join(lines)

    def compute_index_delta(self, previous_hash: str) -> Dict[str, Any]:
        """Calculate skill delta since previous_hash without rebuilding system prompt."""
        current_hash = self.get_content_hash()
        if current_hash == previous_hash:
            return {"has_changes": False, "added": [], "removed": []}

        with self._lock:
            return {
                "has_changes": True,
                "previous_hash": previous_hash,
                "current_hash": current_hash,
                "added": [s.name for s in self._skills.values()],
                "removed": [],
            }

    def count(self) -> int:
        """Return count of registered skills."""
        with self._lock:
            return len(self._skills)

    refresh = reload
    get_index_hash = get_content_hash
    generate_prompt_catalog = get_available_skills_prompt_block
