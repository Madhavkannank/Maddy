"""
M.A.D.D.Y. Personality Engine.
(Modular Autonomous Digital Daemon for You)
Integrates prompts, address policies, name permissions, and speech formatting.
Enforces consistent, natural conversational behavior, dry contextual humor,
natural user addressing ('boss'/'sir'), and standardized task failure responses.
"""
import re
from typing import Any, Dict, Optional, Tuple

from memory.user_preferences import UserPreferences
from personality.policy import PersonalityPolicy, UNWANTED_CLICHES
from personality.prompts import (
    MADDY_BASE_PERSONALITY_PROMPT,
    MADDY_CORE_SYSTEM_PROMPT,
    build_personality_system_prompt,
)


class PersonalityEngine:
    """
    Manages the persona, conversational nuances, and preferred address for M.A.D.D.Y.
    """

    def __init__(
        self,
        user_preferences: Optional[UserPreferences] = None,
        memory_manager: Optional[Any] = None,
    ):
        self.user_preferences = user_preferences or UserPreferences()
        self.memory_manager = memory_manager
        self.policy = PersonalityPolicy(user_preferences=self.user_preferences, memory_manager=self.memory_manager)
        self._address_counter = 0

    def get_address_title(self) -> str:
        """
        Get user's preferred address title or name.
        Defaults to naturally alternating 'boss' or 'sir' unless explicitly configured.
        """
        return self.policy.get_natural_address()

    def get_greeting_candidates(self, addr: Optional[str] = None) -> list:
        """
        Generate dynamic charismatic wake greeting options.
        Delegated directly to PersonalityPolicy.
        """
        return self.policy.get_greeting_candidates(addr=addr)

    def get_callout_acknowledgment(self, moniker: str = "buddy", addr: Optional[str] = None) -> str:
        """
        Dynamic acknowledgment for friendly monikers and starter phrases.
        Delegated directly to PersonalityPolicy.
        """
        return self.policy.get_callout_acknowledgment(moniker=moniker, addr=addr)


    def needs_address_onboarding(self) -> bool:
        """Check if user has not yet specified a custom name or address override."""
        return not self.user_preferences.has_configured_address()

    def check_address_setup_input(self, user_text: str) -> Optional[str]:
        """
        Check if user's input during onboarding setup is specifying their preferred name/address (e.g. 'Call me Alex' or 'Alex').
        Updates UserPreferences directly for explicit onboarding input.
        """
        clean = (user_text or "").strip()
        name = re.sub(r"^(?:please\s+)?(?:call me|address me as)\s+", "", clean, flags=re.IGNORECASE).strip().strip(".,!?:;\"'")
        if name and name.lower() not in ("hi", "hello", "hey", "yes", "no", "ok", "okay", "maddy"):
            self.user_preferences.set_preferred_address(name)
            self.user_preferences.set_preferred_name(name)
            return name
        return None

    def handle_conversational_intercept(self, user_text: str) -> Tuple[bool, Optional[str]]:
        """
        Check conversational intercept for name declaration or setup.
        Returns: (handled: bool, response_text: Optional[str])
        """
        clean = (user_text or "").strip()
        m = re.match(r"^(?:hello|hi|hey)?[,\s]*(?:my\s+name\s+is|my\s+name's)\s+([A-Za-z0-9_.\s]+)$", clean, re.IGNORECASE)
        if m:
            raw_name = m.group(1).strip().strip(".,!?:;\"'")
            if raw_name and raw_name.lower() not in ("boss", "sir", "here", "ready", "maddy"):
                self.user_preferences.set_preference("user.real_name", raw_name)
                self.user_preferences.set_preference("real_name", raw_name)
                addr = self.get_address_title()
                return True, f"Understood. Pleasure to meet you, {raw_name}. I will address you as {addr}."

        return self.policy.handle_name_query_or_setup(user_text)

    def get_llm_system_prompt(self, context: str = "general") -> str:
        """
        Build complete system prompt for Llama 70B reflecting the active address mode
        and contextual guidelines.
        """
        addr = self.get_address_title()
        return build_personality_system_prompt(address_title=addr, context=context)

    def format_task_failure(
        self,
        reason: Optional[str] = None,
        retry_allowed: bool = True,
    ) -> str:
        """
        Standardized failure format:
        'Task failed, boss. [Brief reason]. Want me to retry?'
        """
        addr = self.get_address_title()
        return self.policy.format_task_failure(
            reason=reason,
            address=addr,
            retry_allowed=retry_allowed,
        )

    def format_response(self, text: str, context: str = "general") -> str:
        """
        Refine the visual response to align with M.A.D.D.Y. personality:
        - Strip corporate cliches and fake emotions
        - Deduplicate contradictory address titles (e.g. 'sir, boss' -> 'boss')
        - Preserve structured markdown, lists, and links for visual screen display
        """
        if not text:
            return ""

        # 1. Clean visual response for screen display
        clean = self.policy.format_visual_response(text)

        # 2. Contextual dry tone: if technical diagnostic or error, keep strictly factual
        if context in ("investigation", "diagnostic", "error", "security"):
            return clean

        return clean

    def generate_first_run_prompt(self) -> str:
        """Prompt to ask user how they want to be addressed."""
        return "Before we get started, what would you like me to call you?"
