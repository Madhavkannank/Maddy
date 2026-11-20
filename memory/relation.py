"""
Memory relation classifier and deterministic state mutation engine for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Evaluates relationship between memory candidates and active records across:
NEW, REINFORCES, DUPLICATE, UPDATES, CONTRADICTS, UNCERTAIN.
Enforces deterministic state transitions. Raw model output never directly mutates state.
"""
import re
import time
from typing import Any, List, Optional, Set, Tuple

from memory.schema import (
    MemoryConfidence,
    MemoryMutationAudit,
    MemoryRecord,
    MemoryRelation,
    MemoryScope,
    MemoryStatus,
)
from memory.store import MemoryStore


def _normalize_tokens(text: str) -> Set[str]:
    """Tokenize and filter stop words for comparison."""
    words = re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())
    stop_words = {
        "i", "a", "an", "the", "to", "for", "in", "on", "at", "by", "is", "am", "are",
        "was", "were", "be", "been", "my", "me", "you", "your", "it", "its", "that", "this",
        "user", "preference",
    }
    return {w for w in words if w not in stop_words and (len(w) > 1 or w.isdigit())}


def _calculate_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection) / float(union) if union > 0 else 0.0


def classify_memory_relation(
    candidate: MemoryRecord,
    existing_records: List[MemoryRecord],
) -> Tuple[MemoryRelation, Optional[MemoryRecord], str]:
    """
    Classify the relationship between a candidate memory and existing active records.
    Returns: (MemoryRelation, target_record, explanation)
    """
    cand_tokens = _normalize_tokens(candidate.content)
    generic_entities = {"user", "preference", "the", "sir", "boss"}
    cand_entities = {e.lower() for e in candidate.entities if e.lower() not in generic_entities}
    cand_lower = candidate.content.lower().strip()

    # Negation and deprecation indicators
    negation_patterns = [
        r"\b(?:don't|do not|no longer|stop|never|quit|switch(?:ed)? from|replaced?)\b",
        r"\b(?:hate|dislike|avoid)\b",
        r"\bnot\s+anymore\b",
    ]
    is_candidate_negating = any(re.search(pat, cand_lower) for pat in negation_patterns)

    best_match: Optional[MemoryRecord] = None
    highest_sim = 0.0

    for rec in existing_records:
        if not rec.is_active():
            continue

        rec_tokens = _normalize_tokens(rec.content)
        rec_entities = {e.lower() for e in rec.entities if e.lower() not in generic_entities}
        rec_lower = rec.content.lower().strip()

        # 1. Exact or near-exact duplicate content
        if cand_lower == rec_lower:
            return MemoryRelation.DUPLICATE, rec, "Content is identical to existing memory."

        # If candidate and record are about specific distinct attributes, do not cross-match
        cand_attrs = re.findall(r"\b(name|gpu|cpu|ram|editor|browser|budget|location|city|country|os|project|hackathon)\b", cand_lower)
        rec_attrs = re.findall(r"\b(name|gpu|cpu|ram|editor|browser|budget|location|city|country|os|project|hackathon)\b", rec_lower)
        if cand_attrs and rec_attrs and not set(cand_attrs).intersection(set(rec_attrs)):
            continue

        # If both explicitly specify non-generic entities and have zero overlap, do not match
        if cand_entities and rec_entities and not cand_entities.intersection(rec_entities):
            continue

        sim = _calculate_jaccard_similarity(cand_tokens, rec_tokens)
        entity_overlap = cand_entities.intersection(rec_entities)

        if entity_overlap:
            sim += 0.30  # Boost similarity if shared domain entity key

        if sim > highest_sim:
            highest_sim = sim
            best_match = rec

    if not best_match or highest_sim < 0.35:
        return MemoryRelation.NEW, None, "Novel memory with no significant overlap."

    best_lower = best_match.content.lower()

    # Attribute update check (e.g. user name is X vs user name is Y)
    cand_attrs = set(re.findall(r"\b(name|gpu|cpu|ram|editor|browser|budget|location|city|country|os|favorite|favourite|song|music)\b", cand_lower))
    best_attrs = set(re.findall(r"\b(name|gpu|cpu|ram|editor|browser|budget|location|city|country|os|favorite|favourite|song|music)\b", best_lower))
    if cand_attrs and best_attrs and cand_attrs.intersection(best_attrs):
        if cand_lower != best_lower:
            return MemoryRelation.UPDATES, best_match, f"Updates existing user attribute '{sorted(cand_attrs.intersection(best_attrs))[0]}'."

    # Same-domain overwrite signals: favorite song / user name / project preference etc.
    same_domain_update = (
        bool(re.search(r"\b(?:favorite|favourite)\s+(?:song|music)\b", cand_lower)) and
        bool(re.search(r"\b(?:favorite|favourite)\s+(?:song|music)\b", best_lower))
    ) or (
        bool(re.search(r"\b(?:my\s+name\s+is|user\s+name\s+is|i\s+am\s+called)\b", cand_lower)) and
        bool(re.search(r"\b(?:my\s+name\s+is|user\s+name\s+is|i\s+am\s+called)\b", best_lower))
    )
    update_indicators = ["actually", "instead", "changed", "switch", "updated", "new", "now", "different", "replaced", "set"]
    if same_domain_update and (any(word in cand_lower for word in update_indicators) or best_lower != cand_lower):
        return MemoryRelation.UPDATES, best_match, "User explicitly changed a previously remembered fact in the same domain."

    # 2. Check for DUPLICATE: very high token similarity without state change
    if highest_sim >= 0.85 and not is_candidate_negating:
        return MemoryRelation.DUPLICATE, best_match, f"High token overlap ({highest_sim:.2f}) without state change."

    # 3. Check for CONTRADICTION: user switches away or explicitly negates previous active memory
    if is_candidate_negating and (not best_match.entities or any(e in cand_lower for e in best_match.entities)):
        return MemoryRelation.CONTRADICTS, best_match, "Candidate explicitly negates or abandons previous memory."

    # If both describe mutually exclusive choices in the same domain
    exclusive_pairs = [
        ({"dark", "dark mode"}, {"light", "light mode"}),
        ({"fedora"}, {"ubuntu", "arch", "windows"}),
        ({"flask"}, {"fastapi", "django"}),
        ({"chrome"}, {"firefox", "edge"}),
    ]
    for group_a, group_b in exclusive_pairs:
        in_a_best = any(term in best_lower for term in group_a)
        in_b_best = any(term in best_lower for term in group_b)
        in_a_cand = any(term in cand_lower for term in group_a)
        in_b_cand = any(term in cand_lower for term in group_b)

        if (in_a_best and in_b_cand) or (in_b_best and in_a_cand):
            return MemoryRelation.CONTRADICTS, best_match, "Mutually exclusive preference detected in same domain."

    # 4. Check for explicit UPDATES
    update_indicators = [
        r"\b(?:update|change|prefer|now|switch(?:ed)? to|instead)\b",
    ]
    if any(re.search(pat, cand_lower) for pat in update_indicators) and not any(w in cand_lower for w in ["confirm", "still", "frequently", "heavily"]):
        return MemoryRelation.UPDATES, best_match, f"Refines or modifies existing memory (sim={highest_sim:.2f})."

    # 5. Check for REINFORCES: corroborates existing memory
    if highest_sim >= 0.50 and not is_candidate_negating:
        return MemoryRelation.REINFORCES, best_match, f"Corroborates existing memory with similarity {highest_sim:.2f}."

    return MemoryRelation.NEW, None, "Overlap not sufficient to modify existing memory."



def apply_relation_mutation(
    store: MemoryStore,
    candidate: MemoryRecord,
    relation: MemoryRelation,
    target_record: Optional[MemoryRecord],
    reason: str,
) -> MemoryRecord:
    """
    Execute deterministic state mutation on memory store based on relation classification.
    Raw model output never directly mutates state.
    """
    now = time.time()

    if relation == MemoryRelation.DUPLICATE and target_record is not None:
        target_record.touch()
        target_record.updated_at = now
        target_record.mutation_history.append(
            MemoryMutationAudit(
                mutation_type="DUPLICATE_TOUCH",
                timestamp=now,
                reason=reason,
                affected_memory_id=target_record.memory_id,
            )
        )
        store.update(target_record)
        return target_record

    elif relation == MemoryRelation.REINFORCES and target_record is not None:
        # Boost confidence safely capped at 1.0
        target_record.confidence = min(1.0, target_record.confidence + 0.10)
        target_record.touch()
        target_record.updated_at = now
        # Merge new entities or tags if present
        for ent in candidate.entities:
            if ent not in target_record.entities:
                target_record.entities.append(ent)
        for tag in candidate.tags:
            if tag not in target_record.tags:
                target_record.tags.append(tag)

        target_record.mutation_history.append(
            MemoryMutationAudit(
                mutation_type="REINFORCE",
                timestamp=now,
                reason=reason,
                affected_memory_id=target_record.memory_id,
                replacing_memory_id=candidate.memory_id,
            )
        )
        store.update(target_record)
        return target_record

    elif relation in (MemoryRelation.UPDATES, MemoryRelation.CONTRADICTS) and target_record is not None:
        # Supersede the older record
        target_record.status = MemoryStatus.SUPERSEDED
        target_record.superseded_by = candidate.memory_id
        target_record.superseded_at = now
        target_record.updated_at = now
        target_record.mutation_history.append(
            MemoryMutationAudit(
                mutation_type="SUPERSEDE",
                timestamp=now,
                reason=reason,
                affected_memory_id=target_record.memory_id,
                replacing_memory_id=candidate.memory_id,
            )
        )
        store.update(target_record)

        # Save new active record with audit link
        candidate.status = MemoryStatus.ACTIVE
        candidate.updated_at = now
        candidate.mutation_history.append(
            MemoryMutationAudit(
                mutation_type="CREATE_SUPERSEDING",
                timestamp=now,
                reason=f"Supersedes {target_record.memory_id}: {reason}",
                affected_memory_id=target_record.memory_id,
                replacing_memory_id=candidate.memory_id,
            )
        )
        store.save(candidate)
        return candidate

    elif relation == MemoryRelation.NEW:
        candidate.status = MemoryStatus.ACTIVE
        candidate.updated_at = now
        candidate.mutation_history.append(
            MemoryMutationAudit(
                mutation_type="CREATE",
                timestamp=now,
                reason=reason,
                replacing_memory_id=candidate.memory_id,
            )
        )
        store.save(candidate)
        return candidate

    elif relation == MemoryRelation.UNCERTAIN:
        # Uncertain memories are not committed permanently to disk
        return candidate

    else:
        candidate.status = MemoryStatus.ACTIVE
        store.save(candidate)
        return candidate
