"""Personality module for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You)."""
from personality.engine import PersonalityEngine
from personality.policy import PersonalityPolicy
from personality.prompts import (
    MADDY_BASE_PERSONALITY_PROMPT,
    MADDY_CORE_SYSTEM_PROMPT,
    build_personality_system_prompt,
)

__all__ = [
    "PersonalityEngine",
    "PersonalityPolicy",
    "MADDY_BASE_PERSONALITY_PROMPT",
    "MADDY_CORE_SYSTEM_PROMPT",
    "build_personality_system_prompt",
]

