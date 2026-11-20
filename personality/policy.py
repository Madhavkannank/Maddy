"""
Policy definitions and behavioral rules for M.A.D.D.Y. Personality.
(Modular Autonomous Digital Daemon for You)
Manages address alternation, name-permission state machine, failure formatting, and speech sanitization.
"""
import enum
import os
import re
from typing import Any, Dict, Optional, Tuple
from memory.user_preferences import UserPreferences

UNWANTED_CLICHES = [
    r"as an ai language model",
    r"as an ai,? i",
    r"i'd be delighted to assist",
    r"i'm delighted to help",
    r"i would be more than happy to",
    r"i'd be happy to help",
    r"let's embark on this",
    r"let's dive into that",
    r"feel free to ask",
    r"is there anything else i can help you with today\??",
    r"certainly!?",
    r"absolutely!?",
    r"great question!",
    r"excellent question!",
    r"i hope this helps!?",
    r"hope this helps!?",
    r"here's what i found for you!?",
    r"i am here to assist you with whatever you need",
    r"i'm excited to help",
    r"sure thing!",
    r"no worries!",
    r"how can i help you today\??",
    r"how can i assist you today\??",
    r"how may i assist you today\??",
    r"how may i help you\??",
    r"how can i assist you\??",
]

class NamePermissionState(enum.Enum):
    IDLE = "IDLE"
    AWAITING_NAME_REVEAL_CONFIRMATION = "AWAITING_NAME_REVEAL_CONFIRMATION"

class PersonalityPolicy:
    """
    Enforces address policy, name permission gates, failure formatting, and response sanitization.
    """

    def __init__(
        self,
        user_preferences: Optional[UserPreferences] = None,
        memory_manager: Optional[Any] = None,
    ):
        self.user_preferences = user_preferences or UserPreferences()
        self.memory_manager = memory_manager
        self._address_counter = 0
        self._name_state = NamePermissionState.IDLE

    def get_natural_address(self) -> str:
        """
        Return the preferred address ('boss' or 'sir' naturally alternating)
        unless the user explicitly gave permission to be called by a specific name.
        """
        custom_name = self.user_preferences.get_preferred_name("")
        explicit_addr = self.user_preferences.get_preferred_address("")

        # If user explicitly instructed "Call me X"
        if explicit_addr and explicit_addr.lower() not in ("boss", "sir"):
            return explicit_addr
        if custom_name and explicit_addr == custom_name:
            return custom_name

        # Otherwise alternate naturally between "boss" and "sir"
        self._address_counter += 1
        return "boss" if (self._address_counter % 2 == 1) else "sir"

    def get_real_user_name(self) -> str:
        """Retrieve known user name from memory, preferences, or OS environment without volunteering it."""
        # 1. First check unified long-term memory for latest explicit user name
        if self.memory_manager:
            try:
                recs = self.memory_manager.retrieve_relevant("user name is", max_records=5, min_score=0.1)
                for r in recs:
                    m = re.match(r"^User name is\s+(.+)$", r.content, re.IGNORECASE)
                    if m:
                        mem_name = m.group(1).strip()
                        if mem_name:
                            self.user_preferences.set_preference("user.real_name", mem_name)
                            return mem_name
            except Exception as e:
                logger.debug(f"Memory lookup for user name failed: {e}")

        # 2. Check user preferences
        pref_name = self.user_preferences.get_preference("user.real_name", "") or self.user_preferences.get_preference("real_name", "")
        if pref_name:
            return pref_name

        # 3. Check OS environment
        os_user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
        if os_user:
            return os_user.capitalize()
        return "Operator"

    def handle_name_query_or_setup(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        No-op interceptor: Conversational requests, identity questions, loyalty queries,
        and preference updates ('Call me sir', 'My name is John') are NEVER intercepted
        by hardcoded regex patterns. All turns flow directly into MaddyAgentCore.run_turn()
        so the LLM can dynamically reason, call tools (update_preference, remember),
        and respond naturally without procedural regex interception.
        Returns: (handled: bool, response: Optional[str])
        """
        return False, None

    def format_task_failure(
        self,
        reason: Optional[str] = None,
        address: Optional[str] = None,
        retry_allowed: bool = True,
    ) -> str:
        """
        Standardized failure format per policy:
        'Task failed, boss. [Brief reason]. Want me to retry?'
        - Never outputs full user request.
        - Never dumps raw traceback stack traces.
        - Provides specific simple reason if known, or generic if unknown.
        """
        addr = address or self.get_natural_address()
        retry_suffix = " Want me to retry?" if retry_allowed else ""

        if not reason or not reason.strip():
            return f"Task failed, {addr}. I couldn't determine the exact cause.{retry_suffix}"

        clean_reason = reason.strip()

        # Simplify common technical exception strings into clear language
        if "No module named" in clean_reason:
            clean_reason = "a required Python module is not installed"
        elif "Permission denied" in clean_reason or "permission_denied" in clean_reason.lower():
            clean_reason = "permission was denied by system security"
        elif "timed out" in clean_reason.lower() or "timeout" in clean_reason.lower():
            clean_reason = "the operation timed out before responding"
        elif "connection refused" in clean_reason.lower():
            clean_reason = "the target service refused the connection"
        elif "not found" in clean_reason.lower():
            # e.g., "Application chrome not found"
            clean_reason = clean_reason.rstrip(".,;:")
        else:
            # Strip excessive traceback prefixes if any
            clean_reason = clean_reason.splitlines()[0]
            clean_reason = re.sub(r"^[A-Za-z0-9_.]+(?:Error|Exception):\s*", "", clean_reason)
            clean_reason = clean_reason.rstrip(".,;:")

        # Ensure correct sentence casing
        if clean_reason and clean_reason[0].islower():
            clean_reason = clean_reason[0].upper() + clean_reason[1:]

        return f"Task failed, {addr}. {clean_reason}.{retry_suffix}"

    def format_visual_response(self, text: str) -> str:
        """
        Formats visual screen response for terminal/chat display:
        - Extracts pure answer if raw JSON with thought/final_answer
        - Strips unwanted corporate cliches
        - Deduplicates conflicting address titles (e.g. 'sir, boss' -> 'boss')
        - Preserves clean markdown, line breaks, bullet points, and clickable links for display
        """
        if not text:
            return ""

        clean = text.strip()

        # Extract pure answer if raw JSON
        if "thought" in clean and ("final_answer" in clean or "finalanswer" in clean):
            try:
                import json
                j_text = re.sub(r"^```(?:json)?\s*", "", clean)
                j_text = re.sub(r"\s*```$", "", j_text).strip()
                data = json.loads(j_text)
                if isinstance(data, dict):
                    extracted = data.get("final_answer") or data.get("finalanswer") or data.get("answer") or data.get("response")
                    if extracted:
                        clean = str(extracted).strip()
            except Exception:
                m = re.search(r'"(?:final_answer|finalanswer|answer)":\s*"([^"]+)"', clean)
                if m:
                    clean = m.group(1).replace("\\n", "\n").replace('\\"', '"').strip()

        # Strip corporate cliches
        for pat in UNWANTED_CLICHES:
            clean = re.sub(pat, "", clean, flags=re.IGNORECASE).strip()

        # Deduplicate contradictory or repeated address titles
        clean = re.sub(r"\b(?:sir,\s*boss|boss,\s*sir)\b", "boss", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:boss,\s*boss)\b", "boss", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:sir,\s*sir)\b", "sir", clean, flags=re.IGNORECASE)

        return clean.strip()

    def sanitize_spoken_output(self, text: str) -> str:
        """
        Prepares text for natural speech via TTS:
        - Removes markdown formatting: bold (**), headers (#), backticks (`), bullet asterisks (*)
        - Strips unwanted corporate cliches
        - Normalizes whitespace and punctuation
        """
        clean = text.strip()

        # 0. If input is raw JSON with thought/final_answer, extract pure answer text
        if "thought" in clean and ("final_answer" in clean or "finalanswer" in clean):
            try:
                import json
                # Handle potential markdown code fences around JSON
                j_text = re.sub(r"^```(?:json)?\s*", "", clean)
                j_text = re.sub(r"\s*```$", "", j_text).strip()
                data = json.loads(j_text)
                if isinstance(data, dict):
                    extracted = data.get("final_answer") or data.get("finalanswer") or data.get("answer") or data.get("response")
                    if extracted:
                        clean = str(extracted).strip()
            except Exception:
                m = re.search(r'"(?:final_answer|finalanswer|answer)":\s*"([^"]+)"', clean)
                if m:
                    clean = m.group(1).replace("\\n", "\n").replace('\\"', '"').strip()

        # 1. Strip corporate cliches
        for pat in UNWANTED_CLICHES:
            clean = re.sub(pat, "", clean, flags=re.IGNORECASE).strip()

        # 2. Strip markdown headers (e.g., '### Heading' -> 'Heading')
        clean = re.sub(r"^#{1,6}\s+", "", clean, flags=re.MULTILINE)

        # 3. Strip markdown bold / italic formatting (e.g., '**Result:** 1200' -> 'Result: 1200')
        clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", clean)
        clean = re.sub(r"\*([^*]+)\*", r"\1", clean)
        clean = re.sub(r"__([^_]+)__", r"\1", clean)
        clean = re.sub(r"_([^_]+)_", r"\1", clean)

        # 4. Strip markdown code backticks
        clean = re.sub(r"```[a-zA-Z0-9]*\n(.*?)\n```", r"\1", clean, flags=re.DOTALL)
        clean = re.sub(r"`([^`]+)`", r"\1", clean)

        # 5. Sanitize mathematical and technical expressions for natural spoken audio
        try:
            from core.math_phonetics import MathPhoneticizer
            clean = MathPhoneticizer.sanitize_for_speech(clean)
        except Exception:
            pass

        # 5b. Deduplicate contradictory or repeated address titles (e.g. 'sir, boss' -> 'boss')
        clean = re.sub(r"\b(?:sir,\s*boss|boss,\s*sir)\b", "boss", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:boss,\s*boss)\b", "boss", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:sir,\s*sir)\b", "sir", clean, flags=re.IGNORECASE)

        # 6. Normalize multiple spaces, newlines, and dangling punctuation
        clean = re.sub(r"\s+", " ", clean).strip()
        clean = clean.lstrip(" ,:.-")

        return clean

    def get_greeting_candidates(self, addr: Optional[str] = None) -> list:
        """
        Generate charismatic, snappy, dynamic wake greeting options based on time-of-day
        and active operator address. Managed by Personality policy.
        Provides a diverse pool of 35+ polite, respectful acknowledgments.
        """
        from datetime import datetime
        active_addr = addr or self.get_natural_address()
        hour = datetime.now().hour

        # Time-of-day specific polite candidates (5 variations per time slot)
        if 5 <= hour < 12:
            tod_greetings = [
                f"Good morning {active_addr}, how may I assist you today?",
                f"Good morning {active_addr}, ready when you are.",
                f"Good morning {active_addr}, what's on our agenda today?",
                f"Good morning {active_addr}, I'm listening.",
                f"A very pleasant morning {active_addr}, how can I help?",
            ]
        elif 12 <= hour < 17:
            tod_greetings = [
                f"Good afternoon {active_addr}, how can I help you?",
                f"Good afternoon {active_addr}, I'm listening.",
                f"Good afternoon {active_addr}, what can I do for you?",
                f"Good afternoon {active_addr}, ready for your command.",
                f"Hope your day is going smoothly {active_addr}, how may I assist?",
            ]
        elif 17 <= hour < 22:
            tod_greetings = [
                f"Good evening {active_addr}, how may I help tonight?",
                f"Good evening {active_addr}, standing by for your command.",
                f"Good evening {active_addr}, right here whenever you need me.",
                f"Good evening {active_addr}, what can I do for you?",
                f"Good evening {active_addr}, I'm listening.",
            ]
        else:
            tod_greetings = [
                f"Still working {active_addr}? I'm right here.",
                f"Late night session {active_addr}? Standing by.",
                f"Right here with you {active_addr}, how can I assist?",
                f"Working late {active_addr}? How may I help?",
                f"Always online {active_addr}, what do you need?",
            ]

        # Rich pool of 30+ polite, dignified, respectful acknowledgments
        general_polite_pool = [
            f"Always a pleasure, {active_addr}.",
            f"At your service, {active_addr}.",
            f"Yes {active_addr}, how can I help you?",
            f"Right with you, {active_addr}.",
            f"Standing by, {active_addr}.",
            f"All systems ready, {active_addr}.",
            f"How may I assist you, {active_addr}?",
            f"I'm listening, {active_addr}.",
            f"Ready for your command, {active_addr}.",
            f"Here and ready to help, {active_addr}.",
            f"What can I do for you, {active_addr}?",
            f"Yes {active_addr}, I'm right here.",
            f"Ready when you are, {active_addr}.",
            f"How can I be of service, {active_addr}?",
            f"Awaiting your instructions, {active_addr}.",
            f"Right here, {active_addr}. What do you need?",
            f"At your command, {active_addr}.",
            f"Listening closely, {active_addr}.",
            f"Yes {active_addr}, how may I help?",
            f"Ready to assist, {active_addr}.",
            f"At your disposal, {active_addr}.",
            f"Speak freely {active_addr}, I'm listening.",
            f"Always ready, {active_addr}.",
            f"What are we working on, {active_addr}?",
            f"Good to hear from you, {active_addr}. How can I assist?",
            f"Online and listening, {active_addr}.",
            f"What's on your mind, {active_addr}?",
            f"How can I make your day easier, {active_addr}?",
            f"At your beck and call, {active_addr}.",
            f"Ready and listening, {active_addr}.",
            f"Go ahead {active_addr}, I'm with you.",
            f"I'm all ears, {active_addr}.",
        ]

        return tod_greetings + general_polite_pool

    def get_callout_acknowledgment(self, moniker: str = "buddy", addr: Optional[str] = None) -> str:
        """
        Dynamic acknowledgment when the user addresses M.A.D.D.Y. with a friendly moniker or starter phrase.
        Managed by Personality policy.
        """
        active_addr = addr or self.get_natural_address()
        clean = moniker.lower().strip()
        if "daddy" in clean:
            return f"Welcome back, {active_addr}. Ready when you are."
        elif any(k in clean for k in ("wake", "listen", "online", "ready", "there", "start", "begin")):
            return f"Right here, {active_addr}. How may I help?"
        return f"Standing by, {active_addr}."

