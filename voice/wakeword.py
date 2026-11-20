"""
Local Persistent Wake-Word Engine and State Machine for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Operates fully offline, managing transitions across:
IDLE -> WAKE_DETECTED -> LISTENING -> TRANSCRIBING -> PROCESSING -> RESPONDING -> IDLE.
Provides barge-in interruption detection and fault tolerance against silence and false triggers.
"""
import enum
import time
import re
import threading
from typing import Any, Callable, Dict, List, Optional

class WakeWordState(enum.Enum):
    IDLE = "IDLE"
    WAKE_DETECTED = "WAKE_DETECTED"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    PROCESSING = "PROCESSING"
    RESPONDING = "RESPONDING"
    ERROR = "ERROR"

class WakeWordError(RuntimeError):
    """Exception raised for wake word state machine errors."""
    pass

class WakeWordEngine:
    """
    Manages local offline wake-word detection, conversational state machine,
    barge-in interruption, and transition events.
    """

    WAKE_WORDS = [
        # Primary conversational wake phrases for Maddy
        "maddy",
        "hey maddy",
        "hi maddy",
        "hello maddy",
        "ok maddy",
        "okay maddy",
        "yo maddy",
        "wake up maddy",
        "maddy wake up",
        "listen maddy",
        "maddy listen",
        "maddy online",
        "are you there maddy",
        "you there maddy",
        "maddy are you there",

        # Custom callouts
        "daddy's home", "daddys home", "daddy is home",

        # Natural wake starters
        "wake up",
        "listen up",
    ]
    AMBIGUOUS_SHORT_WORDS = {
        "wake up",
        "listen up",
    }
    STOP_WORDS = [
        "stop",
        "cancel",
        "shut up",
        "pause",
        "be quiet",
        "stop speaking",
        "stop talking",
        "quiet",
        "silence",
        "enough",
        "hold on",
        "wait",
        "hush",
        "abort",
        "never mind",
        "nevermind",
    ]

    def __init__(
        self,
        wake_phrase: str = "hey maddy",
        listen_timeout_seconds: float = 8.0,
        silence_timeout_seconds: float = 4.0,
        on_state_change: Optional[Callable[[WakeWordState, WakeWordState], None]] = None,
        on_command_ready: Optional[Callable[[str], None]] = None,
    ):
        self.wake_phrase = wake_phrase.lower().strip()
        self.listen_timeout_seconds = listen_timeout_seconds
        self.silence_timeout_seconds = silence_timeout_seconds
        self.state = WakeWordState.IDLE
        self.on_state_change = on_state_change
        self.on_command_ready = on_command_ready
        self._lock = threading.RLock()
        self._is_running = False
        self._interrupted = False
        self._state_history: List[WakeWordState] = [WakeWordState.IDLE]

    def get_state(self) -> WakeWordState:
        with self._lock:
            return self.state

    def _transition_to(self, new_state: WakeWordState) -> None:
        with self._lock:
            old_state = self.state
            if old_state == new_state:
                return
            self.state = new_state
            self._state_history.append(new_state)
            if self.on_state_change:
                try:
                    self.on_state_change(old_state, new_state)
                except Exception:
                    pass

    @staticmethod
    def _normalize_wake_phrase(text: str) -> str:
        """Collapse punctuation, map phonetic homophones, and canonicalize Maddy variants."""
        normalized = text.lower().strip()
        normalized = normalized.replace("’", "'")
        # 1. Canonicalize acronym forms: m.a.d.d.y. / m-a-d-d-y / m a d d y -> maddy
        normalized = re.sub(r"\bm[\s\.\-_]*a[\s\.\-_]*d[\s\.\-_]*d[\s\.\-_]*y\b", "maddy", normalized, flags=re.IGNORECASE)
        # 2. Canonicalize common speech-to-text phonetic homophones of the spoken name "Maddy"
        normalized = re.sub(r"\b(maddie|madi|maddi|mady|matty|meddy|madhi|maadi)\b", "maddy", normalized, flags=re.IGNORECASE)
        # 3. Strip extraneous punctuation
        normalized = re.sub(r"[^a-z0-9\s']+", " ", normalized)
        normalized = normalized.replace("'", "")
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def detect_wake_word(self, audio_text: str) -> bool:
        """
        Check if provided phrase contains any configured wake word.
        Supports Latin and Unicode Indic scripts with word boundaries.
        Prevents false wake on arbitrary chatter.
        """
        clean = audio_text.strip().lower()
        if not clean:
            return False

        normalized_text = self._normalize_wake_phrase(clean)
        negative_prefixes = (
            "no maddy", "not maddy", "e maddy",
        )
        if any(normalized_text.startswith(prefix) or f" {prefix}" in f" {normalized_text} " for prefix in negative_prefixes):
            return False

        for ww in self.WAKE_WORDS:
            normalized_ww = self._normalize_wake_phrase(ww)
            if not normalized_ww:
                continue

            # Ambient noise / chatter defense:
            # Short starter phrases must either be the entire utterance or start the utterance
            # as an explicit wake callout. They must never trigger when buried inside ambient chatter.
            is_ambiguous = (ww in self.AMBIGUOUS_SHORT_WORDS or normalized_ww in self.AMBIGUOUS_SHORT_WORDS)
            if is_ambiguous:
                if normalized_text == normalized_ww or normalized_text.startswith(normalized_ww + " "):
                    return True
            else:
                if re.search(rf"\b{re.escape(normalized_ww)}\b", normalized_text):
                    return True
        return False

    def detect_barge_in(self, speech_text: str) -> bool:
        """
        Detect if user spoke an interruption / barge-in command while M.A.D.D.Y. was speaking.
        """
        clean = speech_text.strip().lower()
        if not clean:
            return False
        normalized = re.sub(r"[^a-z0-9\s']+", " ", clean)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        for sw in self.STOP_WORDS:
            if re.search(rf"\b{re.escape(sw)}\b", clean) or re.search(rf"\b{re.escape(sw)}\b", normalized):
                self._interrupted = True
                return True
        return False

    def process_audio_stream_event(self, speech_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a discrete speech/audio event through the state machine.
        Handles wake detection, commands, silence, timeouts, and barge-in.
        """
        with self._lock:
            event_type = speech_event.get("type", "speech")
            text = speech_event.get("text", "").strip()
            is_silent = speech_event.get("is_silent", False)
            timed_out = speech_event.get("timed_out", False)

            # Case A: Currently IDLE -> Scanning for wake word
            if self.state == WakeWordState.IDLE:
                if event_type == "mic_unavailable":
                    self._transition_to(WakeWordState.ERROR)
                    return {"state": self.state.value, "action": "mic_error"}

                if text and self.detect_wake_word(text):
                    self._transition_to(WakeWordState.WAKE_DETECTED)
                    # Automatically transition to LISTENING for following command
                    self._transition_to(WakeWordState.LISTENING)
                    # Extract any trailing command already in the phrase (e.g. 'Hey M.A.D.D.Y., open Spotify')
                    trailing_cmd = self._extract_trailing_command(text)
                    if trailing_cmd:
                        self._transition_to(WakeWordState.PROCESSING)
                        if self.on_command_ready:
                            self.on_command_ready(trailing_cmd)
                        return {
                            "state": self.state.value,
                            "action": "command_extracted",
                            "command": trailing_cmd,
                        }
                    return {"state": self.state.value, "action": "listening_started"}
                return {"state": self.state.value, "action": "continue_standby"}

            # Case B: Currently LISTENING -> Waiting for user command
            elif self.state == WakeWordState.LISTENING:
                if timed_out or (is_silent and speech_event.get("duration", 0) >= self.silence_timeout_seconds):
                    # Timed out without speech -> return to IDLE standby
                    self._transition_to(WakeWordState.IDLE)
                    return {"state": self.state.value, "action": "timeout_to_idle"}

                if text:
                    # User cancelled
                    if any(c in text.lower() for c in ["never mind", "cancel", "stop"]):
                        self._transition_to(WakeWordState.IDLE)
                        return {"state": self.state.value, "action": "cancelled_to_idle"}

                    # Transition through TRANSCRIBING to PROCESSING
                    self._transition_to(WakeWordState.TRANSCRIBING)
                    self._transition_to(WakeWordState.PROCESSING)
                    if self.on_command_ready:
                        self.on_command_ready(text)
                    return {
                        "state": self.state.value,
                        "action": "command_received",
                        "command": text,
                    }
                return {"state": self.state.value, "action": "listening"}

            # Case C: Currently RESPONDING -> Watching for Barge-In Interruption
            elif self.state == WakeWordState.RESPONDING:
                if text and self.detect_barge_in(text):
                    # User interrupted speech
                    self._transition_to(WakeWordState.IDLE)
                    return {"state": self.state.value, "action": "barge_in_interrupted"}
                return {"state": self.state.value, "action": "speaking"}

            # Default
            return {"state": self.state.value, "action": "noop"}

    def complete_response_and_standby(self) -> None:
        """Call when assistant has finished responding to return cleanly to IDLE standby."""
        with self._lock:
            self._interrupted = False
            self._transition_to(WakeWordState.IDLE)

    def mark_responding(self) -> None:
        """Call when assistant begins speaking/responding."""
        with self._lock:
            self._transition_to(WakeWordState.RESPONDING)

    def listen_from_microphone(self, voice_engine: Optional[Any] = None, duration_seconds: float = 3.0) -> Dict[str, Any]:
        """Capture real microphone audio frames using sounddevice and feed into wake-word state machine."""
        if voice_engine is None:
            from voice.engine import VoiceEngine
            voice_engine = VoiceEngine()
        audio_res = voice_engine.listen(duration_seconds=duration_seconds)
        text = audio_res.get("transcription", "")
        return self.process_audio_stream_event({
            "type": "speech",
            "text": text,
            "is_silent": len(text.strip()) == 0,
            "duration": duration_seconds,
            "captured_samples": audio_res.get("captured_samples", 0),
        })

    def _extract_trailing_command(self, text: str) -> Optional[str]:
        """Strip wake word and return trailing command if present."""
        clean = text.strip()
        # Canonicalize Maddy acronyms and homophones for matching
        norm_for_matching = re.sub(r"\bm[\s\.\-_]*a[\s\.\-_]*d[\s\.\-_]*d[\s\.\-_]*y\b", "maddy", clean, flags=re.IGNORECASE)
        norm_for_matching = re.sub(r"\b(maddie|madi|maddi|mady|matty|meddy|madhi|maadi)\b", "maddy", norm_for_matching, flags=re.IGNORECASE)

        # Sort by length descending so longer compound phrases match before shorter ones
        for ww in sorted(self.WAKE_WORDS, key=len, reverse=True):
            norm_ww = self._normalize_wake_phrase(ww)
            parts = [re.escape(p) for p in norm_ww.split()]
            ww_regex = r"\s+".join(parts)
            pattern = rf"^{ww_regex}[\s,:\!\.]*"
            if re.search(pattern, norm_for_matching, re.IGNORECASE):
                cmd = re.sub(pattern, "", norm_for_matching, count=1, flags=re.IGNORECASE).strip()
                if cmd:
                    return cmd

            # Also check direct pattern against original text
            direct_parts = [re.escape(p) for p in ww.split()]
            direct_regex = r"\s+".join(direct_parts)
            direct_pattern = rf"^{direct_regex}[\s,:\!\.]*"
            if re.search(direct_pattern, clean, re.IGNORECASE):
                cmd = re.sub(direct_pattern, "", clean, count=1, flags=re.IGNORECASE).strip()
                if cmd:
                    return cmd
        return None
