"""
Episodic Memory implementation for historical task execution indexing.
Stores summaries of past tasks, goals, outcomes, and keywords for retrieval across OS sessions.
Stored at: <memory_dir>/episodic/episodes.json
"""
import json
import os
import time
from typing import Any, Dict, List, Optional
from config.settings import get_memory_dir

class EpisodicMemory:
    """
    Historical log index of past task executions for search and recall.
    """

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir:
            self.base_dir = base_dir
        else:
            self.base_dir = os.path.join(get_memory_dir(), "episodic")

        self.storage_file = os.path.join(self.base_dir, "episodes.json")
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)

    def record_episode(
        self,
        task_id: str,
        goal: str,
        status: str,
        summary: str,
        keywords: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record a completed task episode summary.
        """
        episodes = self.get_all_episodes()
        episode_data = {
            "task_id": task_id,
            "goal": goal,
            "status": status,
            "summary": summary,
            "keywords": keywords or [],
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        
        # Replace if task_id already exists or append
        existing_idx = next((i for i, ep in enumerate(episodes) if ep.get("task_id") == task_id), None)
        if existing_idx is not None:
            episodes[existing_idx] = episode_data
        else:
            episodes.append(episode_data)

        self._write_episodes(episodes)
        return episode_data

    def get_all_episodes(self) -> List[Dict[str, Any]]:
        """Return list of all recorded episodes."""
        if not os.path.exists(self.storage_file):
            return []
        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def get_episode(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve episode by task_id."""
        episodes = self.get_all_episodes()
        for ep in episodes:
            if ep.get("task_id") == task_id:
                return ep
        return None

    def get_recent_episodes(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Return the N most recent episodes."""
        episodes = self.get_all_episodes()
        episodes.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return episodes[:limit]

    def search_episodes(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search episodes matching query keywords in task_id, goal, summary, or keywords list.
        """
        if not query:
            return self.get_recent_episodes(limit=limit)

        query_terms = [t.lower() for t in query.split() if len(t) > 1]
        episodes = self.get_all_episodes()
        scored_episodes = []

        for ep in episodes:
            score = 0
            searchable_text = f"{ep.get('task_id', '')} {ep.get('goal', '')} {ep.get('summary', '')} {' '.join(ep.get('keywords', []))}".lower()

            for term in query_terms:
                if term in searchable_text:
                    score += 1
                    # Bonus score if term matches keyword exactly
                    if any(term == kw.lower() for kw in ep.get("keywords", [])):
                        score += 2

            if score > 0:
                scored_episodes.append((score, ep.get("timestamp", 0), ep))

        scored_episodes.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [item[2] for item in scored_episodes[:limit]]

    def _write_episodes(self, episodes: List[Dict[str, Any]]) -> None:
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(episodes, f, indent=2)
