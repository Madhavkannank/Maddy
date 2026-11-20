"""
Context Compression Engine for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Handles selective context retrieval, observation history compression, and bounded prompt generation.
"""
import json
from typing import Any, Dict, List, Optional
from memory.working_memory import WorkingMemory
from memory.user_preferences import UserPreferences
from memory.episodic_memory import EpisodicMemory

class ContextCompressor:
    """
    Ensures LLM context windows do not overflow by truncating or summarizing long execution histories.
    """

    def __init__(self, max_chars: int = 8000, max_recent_steps: int = 5):
        self.max_chars = max_chars
        self.max_recent_steps = max_recent_steps

    def compress_observations(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compress observation history.
        Preserves the most recent `max_recent_steps` in full detail.
        Summarizes older observations into compressed text snippets.
        """
        if len(observations) <= self.max_recent_steps:
            return observations

        older = observations[:-self.max_recent_steps]
        recent = observations[-self.max_recent_steps:]

        compressed_older_text = f"Summary of {len(older)} earlier step(s): " + "; ".join(
            [f"[Step {obs.get('step')}: {obs.get('tool')} -> {str(obs.get('data'))[:100]}...]" for obs in older]
        )

        compressed_summary_item = {
            "step": older[0].get("step", 0),
            "tool": "context_compressor",
            "data": compressed_older_text,
            "compressed": True,
        }

        return [compressed_summary_item] + recent

    def truncate_text(self, text: str, max_len: int = 1500) -> str:
        """Truncate long text content with an ellipsis indicator."""
        if len(text) <= max_len:
            return text
        return text[:max_len] + f"\n... [Truncated {len(text) - max_len} characters]"

    def build_bounded_context(
        self,
        goal: str,
        working_memory: Optional[WorkingMemory] = None,
        user_preferences: Optional[UserPreferences] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        memory_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Build a selective, bounded context dictionary containing relevant memories, user preferences,
        relevant past episode memories, and current working scratchpad state.
        Allocates context strictly by relevance rather than fixed percentage quotas.
        Current user instruction always takes precedence over background memory.
        """
        context: Dict[str, Any] = {
            "goal": goal,
            "relevant_memory": [],
            "user_preferences": {},
            "relevant_episodes": [],
            "scratchpad": {},
            "observations": [],
            "_memory_precedence_note": "Current user instruction takes absolute precedence over all past preferences and memories.",
        }

        # 1. Relevance-driven retrieval from MemoryManager
        if memory_manager:
            try:
                relevant_recs = memory_manager.retrieve_relevant(goal, max_records=5, min_score=0.35)
                context["relevant_memory"] = [
                    {
                        "content": rec.content,
                        "type": rec.memory_type.value,
                        "confidence": rec.confidence,
                        "importance": rec.importance,
                    }
                    for rec in relevant_recs
                ]
            except Exception:
                pass

        # 2. Preferences (legacy or fallback)
        if user_preferences:
            context["user_preferences"] = user_preferences.get_all_preferences()

        # 3. Episodic memory search
        if episodic_memory:
            episodes = episodic_memory.search_episodes(goal, limit=3)
            context["relevant_episodes"] = [
                {
                    "task_id": ep.get("task_id"),
                    "goal": ep.get("goal"),
                    "status": ep.get("status"),
                    "summary": ep.get("summary"),
                }
                for ep in episodes
            ]

        # 4. Working memory scratchpad and observations
        if working_memory:
            context["scratchpad"] = dict(working_memory._scratchpad)
            context["observations"] = self.compress_observations(working_memory.get_observations())

        # Enforce total serialized character length bound dynamically
        serialized = json.dumps(context)
        if len(serialized) > self.max_chars:
            # First trim observations
            if context["observations"]:
                context["observations"] = context["observations"][-2:]
            serialized = json.dumps(context)
            if len(serialized) > self.max_chars:
                context["relevant_episodes"] = context["relevant_episodes"][:1]
            serialized = json.dumps(context)
            if len(serialized) > self.max_chars:
                context["relevant_memory"] = context["relevant_memory"][:2]

        return context

