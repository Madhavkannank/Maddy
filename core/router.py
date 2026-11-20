"""
Capability Router for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Classifies user requests into distinct capability tiers:
CONVERSATION, MATH, BROWSER, APPLICATION, FILESYSTEM, SPOTIFY, SCREEN_VISION, VOICE, COMPOUND_TASK.
"""
import enum
import re
from typing import Any, Dict, List, Optional, Set

from core.math_engine import MathEngine

class CapabilityType(enum.Enum):
    CONVERSATION = "CONVERSATION"
    MATH = "MATH"
    BROWSER = "BROWSER"
    APPLICATION = "APPLICATION"
    FILESYSTEM = "FILESYSTEM"
    SPOTIFY = "SPOTIFY"
    SCREEN_VISION = "SCREEN_VISION"
    VOICE = "VOICE"
    HARDWARE = "HARDWARE"
    RESEARCH = "RESEARCH"
    MESSAGING = "MESSAGING"
    SESSION_CAPTURE = "SESSION_CAPTURE"
    COMPOUND_TASK = "COMPOUND_TASK"

# Screen-dependent keywords and phrases triggering automatic screenshot capture
SCREEN_REFERENTS = [
    r"this error",
    r"what('s| is) wrong (here|with this)",
    r"what am i looking at",
    r"why is this (error|window|page) showing",
    r"what button should i (press|click)",
    r"read this",
    r"what('s| is) on my screen",
    r"what does this (say|mean|graph mean)",
    r"why isn'?t this working",
    r"look at this",
    r"on my screen",
    r"take a screenshot",
    r"capture (the )?screen",
    r"visible on my screen",
    r"look at (?:my|the) screen",
    r"see (?:my|the) screen",
    r"check (?:my|the) screen",
    r"view (?:my|the) screen",
    r"read (?:my|the) screen",
    r"summarize (?:my|the|this) screen",
    r"describe (?:my|the) screen",
    r"inspect (?:my|the) screen",
    r"what is open right now",
    r"what window is (?:active|open)",
    r"active window",
    r"explain this (?:error|code|issue|warning|bug)",
]

class CapabilityRouter:
    """
    Analyzes intent and capability requirements to determine the optimal execution path.
    """

    def __init__(self, math_engine: Optional[MathEngine] = None):
        self.math_engine = math_engine or MathEngine()

    def is_screen_dependent(self, text: str) -> bool:
        """Determine if request specifically refers to content currently visible on the user's screen."""
        clean = text.lower()
        return any(re.search(pat, clean) for pat in SCREEN_REFERENTS)

    def route(self, user_request: str) -> Dict[str, Any]:
        """
        Classify user request and return matched capabilities and execution metadata.
        """
        clean = user_request.strip()
        lower = clean.lower()

        capabilities: Set[CapabilityType] = set()

        # 1. Check for compound task markers
        # Strip out timestamp durations like "2 minutes and 25 seconds" before checking compound conjunctions
        compound_check_text = re.sub(
            r"\b\d+\s*(?:hours?|hrs?|h|minutes?|mins?|m)\s+and\s+\d+\s*(?:minutes?|mins?|m|seconds?|secs?|s)\b",
            "",
            lower,
            flags=re.IGNORECASE,
        )
        is_search_intent = bool(re.search(r"\b(?:open|launch|go\s+to)\s+.+?\s+and\s+(?:search|look\s+up|find)\b", compound_check_text))
        is_compound = bool(re.search(r"\b(and then|and also|after that)\b", compound_check_text) or
                          (re.search(r"\band\b", compound_check_text) and len(compound_check_text.split()) > 7 and not is_search_intent))

        # 2. Check Math
        if self.math_engine.is_math_query(clean):
            capabilities.add(CapabilityType.MATH)

        # 3. Check Screen / Vision awareness
        if self.is_screen_dependent(clean):
            capabilities.add(CapabilityType.SCREEN_VISION)

        # 4. Check Spotify (ensure questions, games, and page/search navigation are never hijacked by Spotify)
        is_page_navigation = bool(re.search(r"\b(?:page|next\s+page|move\s+to\s+next|go\s+to\s+next|next\s+result|next\s+item)\b", lower))
        is_conversational_question = bool(
            re.search(
                r"^(?:do\s+(?:you|u)|can\s+(?:you|u)|could\s+(?:you|u)|would\s+(?:you|u)|will\s+(?:you|u)|are\s+(?:you|u)|have\s+(?:you|u)|did\s+(?:you|u)|how\s+do\s+(?:you|u)|how\s+to|why\s+do\s+(?:you|u)|what\s+(?:do|is|are|can)\s+(?:you|u))\b",
                lower,
            )
            or re.search(r"\b(?:chess|game|games|football|cricket|tennis|basketball|soccer|hockey|badminton|guitar|piano|violin|drums|flute|cards|poker|monopoly|board\s*game|video\s*game|with\s+me)\b", lower)
        )
        has_explicit_music_token = any(w in lower for w in [
            "playlist", "playback", "song", "songs", "track", "tracks", "music", "shuffle", "spotify", "album", "artist", "lofi", "melody", "melodies", "soundtrack"
        ])

        is_search_intent = bool(re.search(r"\b(?:search|look\s+up|find)\b", lower)) and not bool(re.search(r"\b(?:play|listen\s+to|stream)\b", lower))
        has_video_destination = any(w in lower for w in [
            "youtube", "netflix", "vimeo", "twitch", "prime video", "disney", "hotstar", "hulu"
        ]) and ("spotify" not in lower)
        has_spotify_playback = not is_page_navigation and not is_search_intent and not has_video_destination and not (is_conversational_question and not has_explicit_music_token) and (
            has_explicit_music_token or
            (re.search(r"\b(?:pause|resume|shuffle)\b", lower) and not any(w in lower for w in ["game", "video", "youtube", "page", "tab", "window", "search", "document", "sheet"])) or
            (re.search(r"^\s*(?:play|listen\s+to)\s+[a-zA-Z0-9_\- ']+", lower) and not is_conversational_question)
        )
        if (has_spotify_playback or ("spotify" in lower and not re.search(r"\b(?:search|look\s+up|find|close|kill|terminate|exit)\b", lower))) and not has_video_destination:
            capabilities.add(CapabilityType.SPOTIFY)

        # 5. Check Browser
        is_explicit_site = any(w in lower for w in [
            "browser", "firefox", "chrome", "edge", "website", "navigate to", "search the web", "url",
            "wikipedia", "youtube", "reddit", "github", "chatgpt", "netflix", "gmail", "twitter",
            "amazon", "flipkart", "ebay", "linkedin", "duckduckgo", "bing", "imdb", "pinterest", "pintrest", "twitch", "maps",
            "wallpaper", "wallpapers", "buy", "order", "price", "product", "deal", "cart", "wishlist",
            ".com", ".org", ".net", ".io", ".edu", ".gov"
        ])
        is_explicit_search = bool(
            re.search(r"\b(?:picture|pictures|photo|photos|image|images)\b", lower)
            or re.search(r"\b(?:search\s+(?:for\s+)?|look\s+up\s+on\s+|search\s+on\s+)\b", lower)
            or re.search(r"\b(?:under|below|less than|budget)\s+(?:rs\.?|inr|₹|\$)?\s*\d+", lower)
            or (re.search(r"\b(?:search|look\s+up|find)\b.+?\b(?:on|in|from)\s+[a-zA-Z0-9_\-\. ]+", lower))
        )
        # Avoid tagging pure math queries ("find x in ...") as browser
        if CapabilityType.MATH not in capabilities:
            if is_explicit_site or is_explicit_search or is_page_navigation or re.search(r"\b(?:search\s+for|search\s+the\s+web)\b", lower):
                capabilities.add(CapabilityType.BROWSER)
        else:
            if is_explicit_site and any(site in lower for site in ["youtube", "amazon", "flipkart", "wikipedia", "google"]):
                capabilities.add(CapabilityType.BROWSER)

        # 6. Check Application control
        is_session_cmd = bool(re.search(r"\b(?:session\s*capture|guest\s*audit|capture\s*session|guest\s*session)\b", lower))
        if re.search(r"\b(open|launch|start|close|terminate|exit|switch\s+to|bring\s+up)\s+[a-zA-Z0-9_\-\.]+", lower) and not is_session_cmd:
            if CapabilityType.BROWSER not in capabilities and CapabilityType.SPOTIFY not in capabilities:
                capabilities.add(CapabilityType.APPLICATION)

        # 7. Check Filesystem
        if any(w in lower for w in ["file", "directory", "folder", "read file", "write file", "save to file"]):
            capabilities.add(CapabilityType.FILESYSTEM)

        # 8. Check Voice / Speaking
        if any(w in lower for w in ["say out loud", "speak", "voice settings"]):
            capabilities.add(CapabilityType.VOICE)

        # 9. Check Hardware / GPU / System metrics
        if any(w in lower for w in ["gpu", "vram", "nvidia", "rtx 3050", "graphic card", "graphics card", "gpu temp", "gpu temperature", "system metrics", "hardware status", "vram usage", "how is my gpu", "gpu holding up"]):
            capabilities.add(CapabilityType.HARDWARE)

        # 10. Check Research / arXiv
        if any(w in lower for w in ["arxiv", "research paper", "research papers", "papers on", "paper on", "digest on"]):
            capabilities.add(CapabilityType.RESEARCH)

        # 11. Check Messaging (Telegram / WhatsApp)
        if any(w in lower for w in ["telegram", "whatsapp", "send a message to", "send message to", "notify my phone", "alert my phone"]):
            capabilities.add(CapabilityType.MESSAGING)

        # 11b. Check Session Capture & Guest Audit
        session_patterns = [
            r"\b(?:session\s*capture|guest\s*audit|capture\s*session|guest\s*session)\b",
            r"\b(?:dismiss|stop|end|cancel|terminate|abort|close|finish|kill|halt)\s+(?:the\s+)?(?:session|capture|audit|guest\s*audit)\b",
            r"\b(?:start|begin|activate|enable|launch|resume)\s+(?:the\s+)?(?:session|capture|audit|guest\s*audit)\b",
            r"\b(?:session|capture|guest\s*audit)\s+(?:status|state|info)\b",
        ]
        if any(re.search(pat, lower) for pat in session_patterns) or lower in (
            "dismiss session", "stop session", "end session", "cancel session", "close session", "terminate session",
            "stop stop", "code red"
        ):
            capabilities.add(CapabilityType.SESSION_CAPTURE)

        # 12. Handle Compound Task classification
        if is_compound and len(capabilities) == 1 and CapabilityType.SPOTIFY in capabilities:
            is_compound = False

        if len(capabilities) > 1 or is_compound:
            primary = CapabilityType.COMPOUND_TASK
        elif len(capabilities) == 1:
            primary = list(capabilities)[0]
        else:
            # Default to Conversation (greetings, chit-chat, general questions, direct advice)
            primary = CapabilityType.CONVERSATION
            capabilities.add(CapabilityType.CONVERSATION)

        return {
            "primary": primary,
            "all_capabilities": [c.value for c in capabilities],
            "is_screen_dependent": CapabilityType.SCREEN_VISION in capabilities,
            "fast_path_eligible": primary in (
                CapabilityType.CONVERSATION,
                CapabilityType.MATH,
                CapabilityType.HARDWARE,
                CapabilityType.RESEARCH,
            ),
        }
