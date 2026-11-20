"""
Automatic Conversational Memory Extractor for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Mimics ChatGPT and Gemini's autonomous background memory extraction.
Identifies and persists user preferences, biographical facts, hardware/system specs,
project details, and persistent instructions from natural dialogue without requiring
explicit 'M.A.D.D.Y., remember that...' commands.
"""
import logging
import re
from typing import Any, Dict, List, Optional

from memory.manager import MemoryManager
from memory.schema import CreatedBy, MemoryConfidence, MemoryScope, MemoryType, SourceType

logger = logging.getLogger(__name__)

# Action words indicating transient commands that should NOT be memorized
TRANSIENT_ACTION_PATTERNS = [
    r"^(?:open|launch|start|run|close|terminate|exit)\b",
    r"^(?:search|find|look\s+up|google|show|display|get)\b",
    r"^(?:play|pause|resume|next|previous|shuffle)\b",
    r"^(?:set\s+volume|turn\s+up|turn\s+down|mute|unmute)\b",
    r"^(?:set\s+brightness|dim|increase\s+brightness)\b",
    r"^(?:calculate|compute|solve|simplify|evaluate|convert)\b",
    r"^(?:take\s+a\s+screenshot|click|type|scroll)\b",
    r"^(?:set\s+a?\s*timer|alarm|countdown)\b",
    r"^(?:what|who|where|when|why|how|is|are|can\s+you|could\s+you)\b",
]


class AutoMemoryExtractor:
    """
    Analyzes natural dialogue to detect and extract long-term memorable facts about the user.
    """

    def is_transient_command(self, text: str) -> bool:
        """Check if request is an ephemeral tool execution or query."""
        clean = text.strip().lower()
        return any(re.search(pat, clean) for pat in TRANSIENT_ACTION_PATTERNS)

    def extract_candidate_fact(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract meaningful user facts, preferences, project details, or constraints.
        Returns dict with:
          - 'content': Normalized memory fact string
          - 'type': MemoryType
          - 'importance': float
          - 'tags': List[str]
        or None if no memorable fact was detected.
        """
        clean = text.strip()
        lower = clean.lower()

        # Never memorize transient tool invocations
        if self.is_transient_command(clean):
            return None

        # 1. Project / Work Context: "I am participating in...", "I'm working on...", "My project is..."
        m_proj = re.search(r"\b(?:i\s+am\s+participating\s+in|i\'m\s+participating\s+in)\s+([a-zA-Z0-9_\-\. ]{4,80})\b", clean, re.I)
        if m_proj:
            val = m_proj.group(1).strip().rstrip(".?!")
            return {
                "content": f"User is participating in {val}",
                "type": MemoryType.SEMANTIC_FACT,
                "importance": 0.85,
                "tags": ["project", "event"],
            }

        m_work = re.search(r"\b(?:i\s+am\s+working\s+on|i\'m\s+working\s+on|i\s+work\s+on)\s+([a-zA-Z0-9_\-\. ]{4,80})\b", clean, re.I)
        if m_work:
            val = m_work.group(1).strip().rstrip(".?!")
            return {
                "content": f"User is working on {val}",
                "type": MemoryType.SEMANTIC_FACT,
                "importance": 0.8,
                "tags": ["project", "work"],
            }

        # 2. Hardware / Possessions: "My GPU is...", "My laptop has...", "My budget is..."
        m_poss = re.search(r"\bmy\s+([a-zA-Z0-9_\- ]{2,20})\s+(?:is|has)\s+([a-zA-Z0-9_\-\. ]{2,80})\b", clean, re.I)
        if m_poss:
            attr = m_poss.group(1).strip()
            val = m_poss.group(2).strip().rstrip(".?!")
            # Filter out non-memorable conversational fillers (e.g. "my question is", "my issue is")
            if attr.lower() not in ("question", "problem", "issue", "request", "point", "thought"):
                return {
                    "content": f"User {attr} is {val}",
                    "type": MemoryType.SEMANTIC_FACT,
                    "importance": 0.75,
                    "tags": ["hardware" if any(h in attr.lower() for h in ["gpu", "cpu", "ram", "laptop", "pc", "device"]) else "profile"],
                }

        # 3. User Identity & Background: "I am a...", "I'm a..."
        m_ident = re.search(r"\b(?:i\s+am|i\'m)\s+(?:a\s+|an\s+)?([^.?!\n,;]{3,60})\b", clean, re.I)
        if m_ident:
            val = m_ident.group(1).strip().rstrip(".?!")
            # Avoid conversational/health states like "I am ready", "I am tired", "I am having..."
            transient_states = [
                "ready", "fine", "good", "okay", "sorry", "tired", "back", "here",
                "sure", "looking", "trying", "wondering", "asking", "having", "feeling",
                "experiencing", "suffering", "dealing", "hurting", "getting"
            ]
            if not any(val.lower().startswith(ts) for ts in transient_states):
                if not re.search(r"\b(?:how|what|why|where|when|can|should|fix)\b", val, re.I):
                    return {
                        "content": f"User is {val}",
                        "type": MemoryType.SEMANTIC_FACT,
                        "importance": 0.8,
                        "tags": ["identity", "profile"],
                    }

        # 4. User Preferences: "I prefer...", "I like...", "I love...", "I always use..."
        m_pref = re.search(r"\b(?:i\s+prefer|i\s+like|i\s+love|i\s+always\s+use|i\s+usually\s+use)\s+([a-zA-Z0-9_\-\. ]{3,80})\b", clean, re.I)
        if m_pref:
            val = m_pref.group(1).strip().rstrip(".?!")
            return {
                "content": f"User prefers {val}",
                "type": MemoryType.USER_PREFERENCE,
                "importance": 0.85,
                "tags": ["preference"],
            }

        # 5. User Location / Residence: "I live in...", "I'm based in..."
        m_loc = re.search(r"\b(?:i\s+live\s+in|i\'m\s+based\s+in|i\s+am\s+based\s+in)\s+([a-zA-Z0-9_\-\. ]{3,50})\b", clean, re.I)
        if m_loc:
            val = m_loc.group(1).strip().rstrip(".?!")
            return {
                "content": f"User lives in {val}",
                "type": MemoryType.SEMANTIC_FACT,
                "importance": 0.8,
                "tags": ["location", "profile"],
            }

        # 6. Explicit home location preference: "I am from ..." / "I'm from ..."
        m_home = re.search(r"\b(?:i\s+am\s+from|i\'m\s+from)\s+([a-zA-Z0-9_\-\. ]{3,50})\b", clean, re.I)
        if m_home:
            val = m_home.group(1).strip().rstrip(".?!")
            return {
                "content": f"User lives in {val}",
                "type": MemoryType.SEMANTIC_FACT,
                "importance": 0.8,
                "tags": ["location", "profile"],
            }

        return None

    def process_and_remember(self, text: str, memory_manager: MemoryManager) -> Optional[str]:
        """
        Inspect user utterance and automatically store candidate memories in the background.
        Returns stored memory content if a memory was created/updated, else None.
        """
        if not text or not memory_manager:
            return None

        fact_data = self.extract_candidate_fact(text)
        if not fact_data:
            return None

        try:
            # Do not persist user location as a fixed preference; live geolocation should remain authoritative.
            location_prefix = "user lives in "
            if fact_data["content"].lower().startswith(location_prefix):
                return None

            rec = memory_manager.remember(
                content=fact_data["content"],
                memory_type=fact_data["type"],
                scope=MemoryScope.GLOBAL,
                source_type=SourceType.USER_MESSAGE,
                source_text=text,
                created_by=CreatedBy.SYSTEM,
                confidence=MemoryConfidence.ESTABLISHED_PREFERENCE.value,
                importance=fact_data.get("importance", 0.75),
                tags=fact_data.get("tags", ["auto_memory"]),
            )
            logger.info(f"AutoMemoryExtractor: memorized '{rec.content}'")
            return rec.content
        except Exception as err:
            logger.debug(f"AutoMemoryExtractor ignored duplicate or failed: {err}")
            return None
