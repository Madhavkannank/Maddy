"""
Verbal Micro-Feedback Engine for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Provides real-time, non-blocking verbal cues during slow tool executions
(web searches, arXiv fetches, subagent delegations) so the user experiences zero dead silence.

Features:
- Tool latency classification (slow vs fast).
- Intelligent debouncing (speaks at most once per conversational turn).
- Asynchronous, fire-and-forget speech dispatch (never blocks tool execution).
- Personality preservation (concise, dry wit, natural address).
"""
from __future__ import annotations

import logging
import os
import random
import threading
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Tools that typically take >1.5s and warrant verbal grounding
SLOW_TOOL_CUES: Dict[str, List[str]] = {
    "browser_search": [
        "Searching the web, {addr}.",
        "Looking that up online, {addr}.",
        "Checking the web now, {addr}.",
    ],
    "browser_navigate": [
        "Opening that page, {addr}.",
        "Navigating there now, {addr}.",
    ],
    "browser_extract": [
        "Extracting content from the page, {addr}.",
        "Reading the page now, {addr}.",
    ],
    "arxiv_search": [
        "Searching arXiv papers, {addr}.",
        "Scanning the research archives, {addr}.",
    ],
    "delegate_task": [
        "Delegating that to a subagent now, {addr}.",
        "Spawning an isolated worker for that, {addr}.",
        "On it, delegating the workload, {addr}.",
    ],
    "filesystem_search": [
        "Scanning the filesystem, {addr}.",
        "Searching the workspace, {addr}.",
    ],
    "spotify_play": [
        "Connecting to Spotify, {addr}.",
        "Queuing that up, {addr}.",
    ],
    "spotify_play_playlist": [
        "Finding that playlist on Spotify, {addr}.",
        "Queuing your playlist, {addr}.",
    ],
}

DEFAULT_CUES: List[str] = [
    "On it, {addr}.",
    "One moment, {addr}.",
    "Working on that now, {addr}.",
]

# Fast tools that finish in <100ms and must remain silent
FAST_TOOLS: Set[str] = {
    "calculator",
    "filesystem_read",
    "filesystem_write",
    "memory_aci_tool",
    "memory",
    "skills_list",
    "skill_view",
    "skill_manage",
    "hardware_status",
    "application_open",
    "application_close",
}


class MicroFeedbackEngine:
    """
    Coordinates real-time verbal micro-feedback when long-running tools execute.
    """

    def __init__(
        self,
        speak_fn: Optional[Callable[[str], None]] = None,
        address_title: str = "boss",
        enabled: Optional[bool] = None,
    ):
        self.speak_fn = speak_fn
        self.address_title = address_title
        if enabled is None:
            self.enabled = os.getenv("VOICE_MICRO_FEEDBACK", "true").lower() in ("1", "true")
        else:
            self.enabled = enabled

        self._has_spoken_this_turn = False
        self._lock = threading.Lock()

    def reset_turn(self) -> None:
        """Call at the start of a user conversational turn to re-arm the trigger."""
        with self._lock:
            self._has_spoken_this_turn = False

    def get_cue(self, tool_name: str) -> Optional[str]:
        """Returns a formatted verbal cue if tool qualifies as slow, otherwise None."""
        if tool_name in FAST_TOOLS:
            return None

        candidates = SLOW_TOOL_CUES.get(tool_name)
        if not candidates:
            # If not explicitly marked fast or slow, check if it's a known slow prefix
            if any(tool_name.startswith(p) for p in ("browser_", "search_", "fetch_", "delegate")):
                candidates = DEFAULT_CUES
            else:
                return None

        phrase = random.choice(candidates)
        return phrase.format(addr=self.address_title)

    def on_tool_start(self, tool_name: str) -> Optional[str]:
        """
        Handler triggered when a tool execution begins.
        Dispatches fire-and-forget non-blocking speech if eligible.
        Returns the spoken cue text (or None).
        """
        if not self.enabled or not self.speak_fn:
            return None

        with self._lock:
            if self._has_spoken_this_turn:
                return None  # Debounce: already acknowledged this turn

            cue = self.get_cue(tool_name)
            if not cue:
                return None

            self._has_spoken_this_turn = True

        # Dispatch speech asynchronously so tool execution is never blocked
        def _speak_async():
            try:
                logger.info(f"[MicroFeedback] Speaking cue: '{cue}' for tool: '{tool_name}'")
                self.speak_fn(cue)
            except Exception as e:
                logger.debug(f"[MicroFeedback] Speech failed: {e}")

        threading.Thread(target=_speak_async, daemon=True, name="MicroFeedbackSpeech").start()
        return cue
