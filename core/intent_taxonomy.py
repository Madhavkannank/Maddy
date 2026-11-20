"""
Granular Semantic Intent Taxonomy for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Defines formal classifications for semantic queries across Knowledge/QA, Search sub-types,
Mathematics & Logic, Code/Technical, Multimodal Perception, and Persona.
"""
from dataclasses import dataclass, field
import enum
from typing import Any, Dict, List, Optional


class SemanticIntent(str, enum.Enum):
    # 1. Knowledge & Direct Question-Answering (Spoken directly, NO browser window opened)
    QA_FACTOID = "QA_FACTOID"                   # Factual facts ("who invented telephone", "capital of Japan")
    QA_EXPLANATION = "QA_EXPLANATION"           # Concept explanations, definitions ("how does gravity work")
    QA_REALTIME_LOOKUP = "QA_REALTIME_LOOKUP"   # Realtime facts ("current weather in Mumbai", "who won the match")

    # 2. Targeted Web & Navigational Search
    SEARCH_NAVIGATIONAL = "SEARCH_NAVIGATIONAL" # Visit specific site or URL ("open github", "go to netflix.com")
    SEARCH_MEDIA_VIDEO = "SEARCH_MEDIA_VIDEO"   # Watch or stream music/video ("search youtube for coldplay live")
    SEARCH_COMMERCIAL = "SEARCH_COMMERCIAL"     # E-commerce, products, price checks ("best laptop under 60k on amazon")
    SEARCH_CONSENSUS = "SEARCH_CONSENSUS"       # Reviews, ratings, discussions ("what does reddit say about M3 macbook")
    SEARCH_LOCAL = "SEARCH_LOCAL"               # Directions, places, local businesses ("coffee shops near me")
    SEARCH_TECHNICAL = "SEARCH_TECHNICAL"       # Documentation, API references, research papers ("torch dataloader docs")

    # 3. Mathematics, Computation & Logic
    MATH_ALGEBRA = "MATH_ALGEBRA"               # Polynomials, symbolic equations ("find x in x^2 - 2x = 3", "solve 2x + 5 = 15")
    MATH_ARITHMETIC = "MATH_ARITHMETIC"         # Pure numerical math, percentages ("48 * 25", "15% of 850", "sqrt(144)")
    MATH_CONVERSION = "MATH_CONVERSION"         # Unit & currency conversions ("75 F to C", "10 miles to km")
    MATH_WORD_PROBLEM = "MATH_WORD_PROBLEM"     # Multi-step mathematical word problems and combinatorial logic

    # 4. Code & Technical Assistance
    CODE_GENERATION = "CODE_GENERATION"         # Generating code snippets, shell commands, regex, git syntax

    # 5. Screen & Multimodal Perception
    SCREEN_ANALYZE = "SCREEN_ANALYZE"           # Inspect active window, debug on-screen error, read graph/table
    SCREEN_AUTOMATE = "SCREEN_AUTOMATE"         # Coordinate clicks, typing into fields, UI automation

    # 6. Conversational, Creative & Persona
    CHITCHAT_BANTER = "CHITCHAT_BANTER"         # Greetings, jokes, identity, persona banter
    CREATIVE_DRAFT = "CREATIVE_DRAFT"           # Drafting emails, summaries, rewriting text

    # 7. OS Colloquial Fallback (when user phrases an OS task conversationally)
    OS_COLLOQUIAL = "OS_COLLOQUIAL"             # Casual OS controls ("it's too loud", "kill calculator app")

    # 8. Owner Session Capture & Guest Audit
    SESSION_CAPTURE = "SESSION_CAPTURE"         # Owner-activated temporary capture ("start session capture", "stop guest audit")

    # Unknown / General fallback
    UNKNOWN = "UNKNOWN"


@dataclass
class IntentPayload:
    """
    Structured payload produced by the Groq Semantic Dispatcher.
    """
    intent: SemanticIntent
    confidence: float = 1.0
    slots: Dict[str, Any] = field(default_factory=dict)
    direct_speech: Optional[str] = None
    target_site: Optional[str] = None
    search_query: Optional[str] = None
    equation: Optional[str] = None
    os_action: Optional[str] = None
    os_parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "slots": self.slots,
            "direct_speech": self.direct_speech,
            "target_site": self.target_site,
            "search_query": self.search_query,
            "equation": self.equation,
            "os_action": self.os_action,
            "os_parameters": self.os_parameters,
        }
