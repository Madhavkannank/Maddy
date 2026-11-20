"""Autonomous Conversational Memory Extractor & Conflict Resolution Pipeline.

Provides fact and preference extraction from conversational turns, utility horizon
salience gating, token Jaccard deduplication, core memory canonical key mapping,
linguistic stance analysis, bi-temporal invalidation, and FTS5 query grammar sanitization.

Fulfills CLAUDE.md Engineering Rules 2, 4, 5, 6, 9, 11, 12, 13.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.llm_client import CloudLLMClient
from core.memory import (
    MemoryCategory,
    MemoryStore,
    calculate_token_jaccard,
    sanitize_fts_query,
)
from core.redact import redact_structure, redact_text

logger = logging.getLogger(__name__)

# ── NOISE AND SALIENCE REJECTION PATTERNS ─────────────────────────────────────
# Casual greetings, affirmations, and transient chatter that must never enter persistent storage
_CHITCHAT_PATTERNS = [
    r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening|night)|yo|howdy)(\s+(maddy|m\.a\.d\.d\.y\.?))?[\.!\?,]*$",
    r"^(thanks|thank\s+you(\s+(so|very)\s+much)?|thx|cheers|much\s+appreciated|great|cool|awesome|perfect|nice|ok|okay|got\s+it|understood)([\s,]+(thanks|thank\s+you|ok|okay|got\s+it|understood|cool|great|much\s+appreciated))*[\.!\?]*$",
    r"^(how\s+are\s+you|what\'?s\s+up|who\s+are\s+you|are\s+you\s+(awake|there|alive))[\.!\?]*$",
    r"^(i\'?m|i\s+am|feeling)\s+(tired|sleepy|hungry|thirsty|bored|busy|back|leaving|heading\s+out|grabbing\s+a\s+coffee)(\s+(today|now|right\s+now|for\s+today))?[\.!\?]*$",
    r"^(heading\s+out(\s+for\s+a\s+coffee)?\s*(brb)?|brb|ttyl|gtg|bye|goodbye|see\s+(ya|you)(\s+later)?|catch\s+you\s+later)[\.!\?]*$",
    r"^(yes|no|yep|nope|sure|definitely|absolutely|maybe)[\.!\?]*$",
]

# One-off transient task commands that should stay in Working/Task Memory only
_INCIDENTAL_TASK_PATTERNS = [
    r"^(open|launch|run|execute|check|inspect|read|cat|ls|cd|grep|find|show|test|calculate)\b",
    r"\bfor\s+(this|now|a\s+sec|a\s+moment|this\s+file|this\s+time|today)\b",
    r"\b(can\s+you|could\s+you|please|just)\s+(open|run|check|read|show|do)\b",
]

# High-confidence permanent stance patterns
_PERMANENT_STANCE_PATTERNS = [
    r"\b(always|never|default|preferred|prefer|favorite|favourite|my\s+rule|from\s+now\s+on)\b",
    r"\b(i\s+use|i\s+work\s+with|i\s+switched\s+to|i\s+have\s+switched\s+to|i\s+don\'?t\s+use\b.+\banymore)\b",
    r"\b(call\s+me|my\s+name\s+is|address\s+me\s+as)\b",
    r"\b(my\s+server|my\s+router|my\s+repo|my\s+editor|my\s+terminal|my\s+theme|my\s+os)\b",
]

# Canonical preference key mappings
_CANONICAL_PREFERENCE_KEYS: Dict[str, str] = {
    "editor": "editor.preferred",
    "ide": "editor.preferred",
    "text_editor": "editor.preferred",
    "terminal": "terminal.preferred",
    "shell": "terminal.shell",
    "browser": "browser.default",
    "theme": "ui.theme",
    "ui_theme": "ui.theme",
    "color_scheme": "ui.theme",
    "address": "user.preferred_address",
    "preferred_address": "user.preferred_address",
    "title": "user.preferred_address",
    "name": "user.preferred_name",
    "user_name": "user.preferred_name",
    "code_formatting": "code_style.formatting",
    "indentation": "code_style.indentation",
}


@dataclass
class ExtractedMemoryFact:
    """Atomic fact extracted from conversation."""

    fact: str
    category: str
    canonical_key: Optional[str] = None
    value: Any = None
    is_explicit: bool = True
    confidence: float = 1.0
    action: str = "ADD"  # ADD, UPDATE, DELETE
    salience_score: float = 1.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConversationMemoryExtractor:
    """Analyzes conversation turns, extracts atomic facts, and updates memory with conflict resolution."""

    def __init__(
        self,
        memory_store: MemoryStore,
        llm_client: Optional[CloudLLMClient] = None,
        min_salience_threshold: float = 0.70,
        min_confidence_threshold: float = 0.60,
    ) -> None:
        self.memory_store = memory_store
        self.llm_client = llm_client
        self.min_salience = min_salience_threshold
        self.min_confidence = min_confidence_threshold

    # ── 1. SALIENCE & STANCE GATING ──────────────────────────────────────────

    def is_noise_or_chitchat(self, text: str) -> bool:
        """Return True if text is trivial conversational chitchat with zero long-term utility."""
        clean = (text or "").strip().lower()
        if not clean or len(clean) < 3:
            return True
        for pat in _CHITCHAT_PATTERNS:
            if re.match(pat, clean):
                return True
        return False

    def is_incidental_task_directive(self, text: str) -> bool:
        """Return True if statement is a one-off operational command rather than a stable preference."""
        clean = (text or "").strip().lower()
        # If it explicitly contains permanent stance markers, it's not incidental
        for p_pat in _PERMANENT_STANCE_PATTERNS:
            if re.search(p_pat, clean):
                return False
        # Otherwise check if it matches incidental command patterns
        for pat in _INCIDENTAL_TASK_PATTERNS:
            if re.search(pat, clean):
                return True
        return False

    def normalize_canonical_key(self, raw_key: str) -> str:
        """Map raw preference key to standardized canonical key path."""
        clean = re.sub(r"[^a-zA-Z0-9_\.]", "", raw_key.strip().lower().replace(" ", "_").replace("-", "_"))
        return _CANONICAL_PREFERENCE_KEYS.get(clean, clean)

    # ── 2. DETERMINISTIC HEURISTIC EXTRACTION ────────────────────────────────

    def extract_deterministic_facts(self, user_prompt: str) -> List[ExtractedMemoryFact]:
        """Extract high-confidence explicit facts using linguistic stance patterns.

        Guarantees deterministic extraction for common preference and environment statements
        even without invoking external LLMs.
        """
        facts: List[ExtractedMemoryFact] = []
        text = user_prompt.strip()
        lower = text.lower()

        if self.is_noise_or_chitchat(text) or self.is_incidental_task_directive(text):
            return []

        # A. PREFERRED ADDRESS / NAME
        m_addr = re.search(
            r"\b(?:call\s+me|address\s+me\s+as)\s+([A-Za-z0-9_\-]{2,20})\b",
            text,
            re.IGNORECASE,
        )
        if m_addr:
            raw_val = m_addr.group(1).strip()
            val = raw_val.title() if (raw_val.islower() or raw_val.isupper()) else raw_val
            facts.append(
                ExtractedMemoryFact(
                    fact=f"Operator prefers to be addressed as {val}.",
                    category="user_preference",
                    canonical_key="user.preferred_address",
                    value=val,
                    is_explicit=True,
                    confidence=1.0,
                    action="UPDATE",
                    salience_score=1.0,
                    reason="Explicit operator address directive",
                )
            )

        m_name = re.search(r"\bmy\s+name\s+is\s+([A-Za-z]{2,25})\b", text, re.IGNORECASE)
        if m_name:
            raw_val = m_name.group(1).strip()
            val = raw_val.capitalize() if (raw_val.islower() or raw_val.isupper()) else raw_val
            facts.append(
                ExtractedMemoryFact(
                    fact=f"Operator name is {val}.",
                    category="user_preference",
                    canonical_key="user.preferred_name",
                    value=val,
                    is_explicit=True,
                    confidence=1.0,
                    action="UPDATE",
                    salience_score=1.0,
                    reason="Explicit operator name identification",
                )
            )

        # B. EDITOR / IDE (including switch detection: "switched from X to Y" or "switched to Y")
        m_switch_editor = re.search(
            r"\b(?:switched\s+(?:from\s+[a-zA-Z0-9_\-\s]+\s+)?to|use\s+now|now\s+use)\s+([a-zA-Z0-9_\-\s]{2,20})\b",
            text,
            re.IGNORECASE,
        )
        if m_switch_editor and any(ed in lower for ed in ["editor", "ide", "vscode", "vs code", "cursor", "vim", "neovim", "emacs", "pycharm"]):
            target_editor = m_switch_editor.group(1).strip()
            target_editor = re.split(r"\b(?:for|because|as|is|and)\b", target_editor, flags=re.IGNORECASE)[0].strip()
            facts.append(
                ExtractedMemoryFact(
                    fact=f"Operator prefers {target_editor} as their primary editor.",
                    category="user_preference",
                    canonical_key="editor.preferred",
                    value=target_editor,
                    is_explicit=True,
                    confidence=1.0,
                    action="UPDATE",
                    salience_score=0.95,
                    reason="Operator explicitly indicated editor switch",
                )
            )
        else:
            m_editor = re.search(
                r"\b(?:my\s+(?:preferred\s+|default\s+)?editor\s+is|i\s+(?:prefer|always\s+use)\s+editor)\s+([a-zA-Z0-9_\-\s]{2,20})\b",
                text,
                re.IGNORECASE,
            )
            if m_editor:
                val = m_editor.group(1).strip()
                val = re.split(r"\b(?:for|because|as|is|and)\b", val, flags=re.IGNORECASE)[0].strip()
                facts.append(
                    ExtractedMemoryFact(
                        fact=f"Operator prefers {val} as their primary editor.",
                        category="user_preference",
                        canonical_key="editor.preferred",
                        value=val,
                        is_explicit=True,
                        confidence=1.0,
                        action="UPDATE",
                        salience_score=0.95,
                        reason="Explicit operator editor preference",
                    )
                )

        # C. UI THEME
        m_theme = re.search(r"\b(?:i\s+prefer|always\s+use|my\s+theme\s+is|use)\s+(dark|light)\s+(?:mode|theme)\b", lower)
        if m_theme:
            val = m_theme.group(1).strip().lower()
            facts.append(
                ExtractedMemoryFact(
                    fact=f"Operator prefers {val} mode UI theme.",
                    category="user_preference",
                    canonical_key="ui.theme",
                    value=val,
                    is_explicit=True,
                    confidence=1.0,
                    action="UPDATE",
                    salience_score=0.90,
                    reason="Explicit UI theme preference",
                )
            )

        # D. TERMINAL / SHELL
        m_term = re.search(r"\b(?:my\s+(?:preferred\s+|default\s+)?terminal\s+is|i\s+use\s+terminal)\s+([a-zA-Z0-9_\-]{2,20})\b", lower)
        if m_term:
            val = m_term.group(1).strip().lower()
            facts.append(
                ExtractedMemoryFact(
                    fact=f"Operator preferred terminal is {val}.",
                    category="user_preference",
                    canonical_key="terminal.preferred",
                    value=val,
                    is_explicit=True,
                    confidence=1.0,
                    action="UPDATE",
                    salience_score=0.90,
                    reason="Explicit terminal preference",
                )
            )

        # E. NETWORK / SERVER HARDWARE FACTS
        m_ip = re.search(r"\bmy\s+(?:home\s+|local\s+)?(?:server|router|nas|gateway)\s+(?:ip\s+is|changed\s+to|is\s+at)\s+([0-9]{1,3}(?:\.[0-9]{1,3}){3})\b", lower)
        if m_ip:
            ip_val = m_ip.group(1).strip()
            facts.append(
                ExtractedMemoryFact(
                    fact=f"Operator local server IP address is {ip_val}.",
                    category="system_fact",
                    canonical_key="hardware.server_ip",
                    value=ip_val,
                    is_explicit=True,
                    confidence=1.0,
                    action="UPDATE",
                    salience_score=0.95,
                    reason="Operator network configuration statement",
                )
            )

        return facts

    # ── 3. LLM-assisted extraction ───────────────────────────────────────────

    def extract_llm_facts(self, user_prompt: str, assistant_response: str) -> List[ExtractedMemoryFact]:
        """Extract atomic facts using structured LLM classification when available."""
        if not self.llm_client:
            return []

        if self.is_noise_or_chitchat(user_prompt) or self.is_incidental_task_directive(user_prompt):
            return []

        extraction_system_prompt = """You are the Memory Extraction Engine for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Analyze the user utterance and extract only meaningful, durable facts, stable preferences, and configuration details.

STRICT INVARIANTS:
1. Ignore casual chitchat, transient commands, and temporary statuses (e.g. 'thanks', 'I am tired', 'open nano for this').
2. Classify each fact into: 'user_preference', 'system_fact', or 'project'.
3. Assign canonical_key (dot-notation, e.g. 'editor.preferred', 'ui.theme', 'terminal.preferred').
4. Determine action: 'ADD', 'UPDATE' (if it alters a known setting), or 'DELETE'.
5. Set confidence from 0.0 to 1.0 (1.0 for explicit user statements, <0.7 for inferences).
6. Assign salience_score (1.0 for permanent facts, <0.5 for temporary info).

Return a JSON array of objects with keys:
[
  {
    "fact": "Operator prefers Cursor as their code editor.",
    "category": "user_preference",
    "canonical_key": "editor.preferred",
    "value": "Cursor",
    "is_explicit": true,
    "confidence": 1.0,
    "action": "UPDATE",
    "salience_score": 1.0,
    "reason": "User explicitly stated they switched to Cursor"
  }
]
If no durable facts exist, return an empty array: []
Output ONLY valid JSON. No conversational preamble.
"""

        messages = [
            {"role": "system", "content": extraction_system_prompt},
            {"role": "user", "content": f"User prompt: {user_prompt}\nAssistant response: {assistant_response[:300]}"},
        ]

        try:
            resp = self.llm_client.chat_completion(messages=messages, temperature=0.0)
            raw = (resp.content or "").strip()
            if not raw or raw == "[]":
                return []
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?", "", raw)
                raw = re.sub(r"```$", "", raw).strip()
            data = json.loads(raw)
            if not isinstance(data, list):
                return []

            results = []
            for item in data:
                if not isinstance(item, dict) or not item.get("fact"):
                    continue
                results.append(
                    ExtractedMemoryFact(
                        fact=str(item["fact"]).strip(),
                        category=str(item.get("category", "user_preference")).strip().lower(),
                        canonical_key=self.normalize_canonical_key(str(item.get("canonical_key", ""))),
                        value=item.get("value"),
                        is_explicit=bool(item.get("is_explicit", True)),
                        confidence=float(item.get("confidence", 1.0)),
                        action=str(item.get("action", "ADD")).upper(),
                        salience_score=float(item.get("salience_score", 1.0)),
                        reason=str(item.get("reason", "LLM extraction")),
                    )
                )
            return results
        except Exception as exc:
            logger.debug("LLM-assisted extraction skipped: %s", exc)
            return []

    # ── 4. CONFLICT RESOLUTION & SYNCHRONIZATION ─────────────────────────────

    def extract_and_sync(
        self,
        user_prompt: str,
        assistant_response: str = "",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        provenance_source: str = "operator",
    ) -> List[Dict[str, Any]]:
        """Main entry point: extracts facts and synchronizes memory with conflict resolution."""
        # 1. Provenance Gate: Only operator utterances can write user preferences
        if provenance_source != "operator":
            logger.info("Ignoring memory extraction for non-operator source: %s", provenance_source)
            return []

        # 2. Extract facts via deterministic heuristics first
        candidates = self.extract_deterministic_facts(user_prompt)

        # 3. If heuristics found nothing, try LLM-assisted extraction
        if not candidates:
            candidates = self.extract_llm_facts(user_prompt, assistant_response)

        if not candidates:
            return []

        applied: List[Dict[str, Any]] = []
        now = time.time()

        for cand in candidates:
            # Check salience and confidence thresholds
            if cand.salience_score < self.min_salience or cand.confidence < self.min_confidence:
                logger.debug("Rejected candidate below threshold: %s", cand.fact)
                continue

            # ── A. USER PREFERENCE PROCESSING ─────────────────────────────────
            if cand.category == "user_preference" and cand.canonical_key:
                clean_key = self.normalize_canonical_key(cand.canonical_key)
                # Store in UserPreferenceMemory with deterministic precedence
                self.memory_store.preferences.set_preference(
                    key=clean_key,
                    value=cand.value if cand.value is not None else cand.fact,
                    confidence=cand.confidence,
                    is_explicit=cand.is_explicit,
                    source="conversational_extraction",
                )
                applied.append({
                    "action": cand.action,
                    "type": "preference",
                    "key": clean_key,
                    "value": cand.value,
                    "fact": cand.fact,
                })
                continue

            # ── B. ARCHIVAL NARRATIVE FACTS (SYSTEM_FACT, PROJECT, ETC.) ─────
            # Conflict resolution: query existing active memories with similar topic or high Jaccard
            cat_enum = MemoryCategory.SYSTEM_FACT if cand.category == "system_fact" else MemoryCategory.PROJECT
            active_existing = self.memory_store.recall(query=cand.fact, limit=3, category=cat_enum)
            supersedes_id: Optional[str] = None

            for existing in active_existing:
                sim = calculate_token_jaccard(cand.fact, existing.get("content", ""))
                # If high overlap or strong topical contradiction
                if sim >= 0.75:
                    if cand.is_explicit or cand.confidence >= existing.get("confidence", 1.0):
                        supersedes_id = existing.get("id")
                        break
                    else:
                        # Existing memory has higher confidence; skip duplicate insertion
                        supersedes_id = "SKIP"
                        break

            if supersedes_id == "SKIP":
                continue

            # Invalidate older superseded memory and insert fresh record
            mem_id = self.memory_store.remember(
                content=cand.fact,
                category=cat_enum,
                confidence=cand.confidence,
                is_explicit=cand.is_explicit,
                importance=1.2 if cand.is_explicit else 1.0,
                supersedes_id=supersedes_id if supersedes_id != "SKIP" else None,
                provenance={"source": provenance_source, "extracted_at": now},
                deduplicate=True,
            )
            applied.append({
                "action": "UPDATE" if supersedes_id else "ADD",
                "type": "archival",
                "memory_id": mem_id,
                "fact": cand.fact,
                "supersedes_id": supersedes_id,
            })

        return applied


__all__ = ["ConversationMemoryExtractor", "ExtractedMemoryFact"]
