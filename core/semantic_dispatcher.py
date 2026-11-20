"""
High-Speed Semantic Dispatcher for M.A.D.D.Y.
(Modular Autonomous Digital Daemon for You)
Uses Groq LPU inference to classify user queries across the granular Semantic Intent Taxonomy
and extract relevant slots (equations, target sites, search queries, direct answers).
Includes zero-error fallback resilience.
"""
import json
import logging
import re
from typing import Any, Dict, Optional

from core.intent_taxonomy import IntentPayload, SemanticIntent
from models.base import ModelProvider

logger = logging.getLogger(__name__)

DISPATCHER_SYSTEM_PROMPT = """You are the M.A.D.D.Y. Semantic Intent Classifier.
(Modular Autonomous Digital Daemon for You)
Classify the user utterance into exactly ONE of the following granular intents and extract rich, structured slots.

INTENTS:
1. QA_FACTOID: Direct factual questions requiring a concise answer without web navigation.
   Examples: "who invented telephone", "what is the capital of Peru", "speed of light".
2. QA_EXPLANATION: In-depth explanations, definitions, concepts.
   Examples: "explain quantum entanglement", "how does a compiler work".
3. QA_REALTIME_LOOKUP: Live/current info.
   Examples: "what's the weather in Tokyo", "latest gold price".
4. SEARCH_NAVIGATIONAL: Intent to visit/open a specific website, app URL, or domain/section.
   Examples: "open github", "go to netflix", "take me to reddit.com", "open my amazon cart", "show flipkart orders".
5. SEARCH_MEDIA_VIDEO: Watch or stream video, audio, or podcasts on YouTube or media sites.
   Examples: "search youtube for lofi beats", "play coldplay live on youtube", "find mrbeast video", "open youtube and play songs".
6. SEARCH_COMMERCIAL: Shopping, product reviews, price comparisons, or buying products on Amazon, Flipkart, etc.
   Examples: "pen drive and filter out only sandisk and samsung", "best laptop under 60k on amazon", "buy wireless mouse on flipkart", "rtx 4070 price".
7. SEARCH_CONSENSUS: Review aggregation, Reddit opinions, IMDb ratings.
   Examples: "what does reddit think of M3 macbook", "imdb rating of interstellar", "is dune 2 worth watching".
8. SEARCH_LOCAL: Directions, places, local businesses, nearby spots.
   Examples: "coffee shops near me", "directions to nearest airport", "restaurants in indiranagar".
9. SEARCH_TECHNICAL: API documentation, GitHub repos, arXiv research papers, StackOverflow.
   Examples: "pytorch dataloader documentation", "look up attention is all you need on arxiv", "search github for autonomous agents".
10. MATH_ALGEBRA: Algebraic equations, solving for unknown variables, roots, polynomials.
    Examples: "find the value of x in x square minus 2 x to 3", "solve x^2 + 5x + 6 = 0", "what is y in 3y - 9 = 0".
11. MATH_ARITHMETIC: Direct numeric calculations, percentages, powers.
    Examples: "48 times 25", "what is 15% of 850", "sqrt(144)".
12. MATH_CONVERSION: Unit and currency conversions.
    Examples: "convert 75 fahrenheit to celsius", "10 miles to km", "5kg to pounds".
13. CODE_GENERATION: Writing code snippets, shell commands, regex, git.
    Examples: "write a python function to reverse a string", "how to revert a git commit".
14. SCREEN_ANALYZE: Inspect active window, read screen content, debug on-screen error.
    Examples: "look at my screen", "what is this error", "read the text on screen".
15. CHITCHAT_BANTER: Greetings, personal banter, humor, identity.
    Examples: "hello maddy", "who are you", "tell me a joke", "thank you".
16. OS_COLLOQUIAL: Colloquially phrased OS requests (volume, brightness, app close/launch).
    Examples: "it's too loud in here", "kill the calculator app", "dim the screen a bit", "வால்யூம் குறை", "ശബ്ദം കുറയ്ക്കൂ".

MULTILINGUAL SUPPORT:
- The user may speak in English, Tamil (தமிழ்), Malayalam (മലയാളം), or mixed (Tanglish / Manglish).
- Seamlessly understand the intent across all three languages.
- If the user asks a question in Tamil, provide 'direct_speech' in polite, natural Tamil.
- If the user asks in Malayalam, provide 'direct_speech' in polite, natural Malayalam.
- If the user asks an OS or search task in Tamil or Malayalam (e.g. "யூடியூப்ல பாட்டு போடு" / "യൂട്യൂബിൽ പാട്ട് വെക്കൂ"), extract the target_site ("youtube") and search_query accurately.

SLOT EXTRACTION GUIDELINES:
- For SEARCH_COMMERCIAL:
  - "product_category": The core product type (e.g. "pen drive", "mechanical keyboard", "running shoes", "smartphone", "laptop").
  - "brands": Array of specific brands requested or filtered for (e.g. ["SanDisk", "Samsung"], ["Apple"], ["Sony"]). Empty array if none.
  - "is_branded": Boolean true if user said "branded", "top brands", "best brands", etc., else false.
  - "min_price": Integer minimum price constraint in local currency, or null (e.g. 1000 for 1k).
  - "max_price": Integer maximum budget/price constraint, or null (e.g. 2000 for 2k, 60000 for 60k).
  - "min_rating": 4 if "good", "best", "top rated", "4 star" mentioned, or null.
  - "sort_by": "price_asc" (low to high), "price_desc" (high to low), "rating", "newest", "popularity", or null.
  - "specs": Array of technical specs or attributes (e.g. ["8gb ram", "128gb", "blue switches", "noise cancelling"]).
  - "fast_delivery": Boolean true if "fast delivery", "prime", "tomorrow" mentioned, else false.
  - "delivery_days": Integer days if mentioned (e.g. 3), else null.
  - "target_site": "amazon", "flipkart", "ebay", etc. (Default to "amazon" if shopping site not named).
  - "search_query": Pure item/product keyword without conversational filler, brands, prices, ratings, or sort instructions (e.g. "laptop", "running shoes", "pen drive").
- For SEARCH_MEDIA_VIDEO:
  - "target_site": "youtube"
  - "search_query": Pure video title, artist, or channel to search for (e.g. "coldplay live in sao paulo", "lofi beats").
- For SEARCH_NAVIGATIONAL:
  - "target_site": Website name (e.g. "amazon", "flipkart", "youtube", "github", "reddit", "netflix").
  - "destination_section": Specific section if requested (e.g. "cart", "orders", "wishlist", "history", "trending").
  - "direct_url": Direct URL if easily resolvable, else null.
- For SEARCH_TECHNICAL:
  - "target_site": "github", "arxiv", "stackoverflow", or "docs".
  - "search_query": Pure technical topic, library, or paper title.
- For SEARCH_LOCAL:
  - "target_site": "maps"
  - "search_query": Pure location, business type, or place.
- For OS_COLLOQUIAL:
  - "os_action": "set_volume", "mute", "unmute", "increase_volume", "decrease_volume", "set_brightness", "increase_brightness", "decrease_brightness", "launch_app", "close_app".
  - "os_target": Target device control or application name (e.g. "volume", "brightness", "calculator", "chrome").
  - "os_value": Target integer value (e.g. 80, 50, 20) or null.

RULES:
- Respond strictly in valid JSON matching this schema:
{
  "intent": "<ONE_OF_THE_ABOVE_INTENT_STRINGS>",
  "slots": {
    "target_site": "<e.g. amazon, flipkart, youtube, reddit, github, arxiv, maps, google>",
    "search_query": "<pure extracted query without conversational filler>",
    "product_category": "<core product name if commercial>",
    "brands": ["<brand1>", "<brand2>"],
    "is_branded": false,
    "min_price": null,
    "max_price": null,
    "min_rating": null,
    "sort_by": null,
    "specs": [],
    "fast_delivery": false,
    "delivery_days": null,
    "destination_section": null,
    "direct_url": null,
    "os_action": null,
    "os_target": null,
    "os_value": null,
    "equation": null,
    "variable": null,
    "entity": null
  },
  "direct_speech": "<concise direct spoken answer ONLY for QA_FACTOID, QA_EXPLANATION, or CHITCHAT_BANTER, else empty string>"
}
- Do NOT output markdown code fences or backticks. Only valid JSON.
"""


class SemanticDispatcher:
    """
    Dispatcher classifying user queries across the granular Semantic Intent Taxonomy using Groq.
    """

    def __init__(self, model_provider: ModelProvider):
        self.model_provider = model_provider

    def dispatch(self, user_request: str) -> IntentPayload:
        """
        Classify user request and return structured IntentPayload.
        Guaranteed to never raise an unhandled exception.
        """
        clean = user_request.strip()
        if not clean:
            return IntentPayload(intent=SemanticIntent.UNKNOWN)

        lower = clean.lower()

        # Session capture and guest audit commands are deterministic system controls
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
            action = "status"
            if lower == "code red" or any(w in lower for w in ["start", "begin", "activate", "turn on", "enable", "launch", "resume"]):
                action = "start"
            elif lower == "stop stop" or any(w in lower for w in ["stop", "end", "deactivate", "turn off", "disable", "terminate", "finish", "dismiss", "cancel", "abort", "close", "kill", "halt"]):
                action = "stop"
            return IntentPayload(intent=SemanticIntent.SESSION_CAPTURE, os_action=action, slots={"action": action})

        # Explicit Spotify playback queries must never be classified as YouTube video searches
        if "spotify" in lower:
            return IntentPayload(intent=SemanticIntent.UNKNOWN)

        # 1. Try Groq fast classification
        if hasattr(self.model_provider, "is_configured") and self.model_provider.is_configured():
            try:
                res = self.model_provider.generate(
                    prompt=f"User request: {clean}",
                    system_prompt=DISPATCHER_SYSTEM_PROMPT,
                )
                if res and res.content:
                    payload = self._parse_model_output(res.content, clean)
                    if payload:
                        return payload
            except Exception as e:
                logger.warning(f"Groq semantic dispatch error: {e}")

        # 2. Local heuristic fallback if Groq is unavailable or returns unparsable output
        return self._local_fallback_classify(clean)

    def _parse_model_output(self, raw_content: str, original_request: str) -> Optional[IntentPayload]:
        """Parse and validate JSON output from the model with regex extraction fallback."""
        text = raw_content.strip()

        # Remove markdown fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

        data = None
        try:
            data = json.loads(text)
        except Exception:
            # Try regex extraction of JSON object
            m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(1))
                except Exception:
                    pass

        if not isinstance(data, dict):
            return None

        raw_intent = data.get("intent", "").strip().upper()
        try:
            intent_enum = SemanticIntent(raw_intent)
        except Exception:
            # Map partial intent strings
            matched_enum = None
            for e in SemanticIntent:
                if e.value in raw_intent or raw_intent in e.value:
                    matched_enum = e
                    break
            intent_enum = matched_enum or SemanticIntent.UNKNOWN

        slots = data.get("slots") or {}
        direct_speech = data.get("direct_speech") or ""

        equation = slots.get("equation")
        target_site = slots.get("target_site")
        search_query = slots.get("search_query")
        os_action = slots.get("os_action") or data.get("action")

        return IntentPayload(
            intent=intent_enum,
            confidence=0.95,
            slots=slots,
            direct_speech=direct_speech if direct_speech else None,
            target_site=target_site,
            search_query=search_query,
            equation=equation,
            os_action=os_action,
            os_parameters={k: v for k, v in slots.items() if k.startswith("os_") or k in ("brands", "min_price", "max_price", "specs", "min_rating", "destination_section", "direct_url")},
        )

    def _local_fallback_classify(self, text: str) -> IntentPayload:
        """
        Deterministic local fallback classifier when cloud inference is offline.
        Populates rich slots locally so downstream pipelines can execute accurately.
        """
        lower = text.lower()
        slots: Dict[str, Any] = {}

        # Check session capture & guest audit commands
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
            action = "status"
            if lower == "code red" or any(w in lower for w in ["start", "begin", "activate", "turn on", "enable", "launch", "resume"]):
                action = "start"
            elif lower == "stop stop" or any(w in lower for w in ["stop", "end", "deactivate", "turn off", "disable", "terminate", "finish", "dismiss", "cancel", "abort", "close", "kill", "halt"]):
                action = "stop"
            slots["action"] = action
            return IntentPayload(intent=SemanticIntent.SESSION_CAPTURE, os_action=action, slots=slots)

        # Check multilingual Tamil/Malayalam volume controls
        if any(w in text for w in ["வால்யூம்", "சத்தம்", "சத்தத்தை", "ശബ്ദം"]):
            slots["os_action"] = "set_volume"
            return IntentPayload(intent=SemanticIntent.OS_COLLOQUIAL, os_action="volume", slots=slots)

        # Check multilingual Tamil/Malayalam media playback / YouTube
        if any(w in text for w in ["பாட்டு", "பாடல்", "வீடியோ", "பாடல்கள்", "പാട്ട്", "ഗാനം", "വീഡിയോ"]):
            q = re.sub(r"\b(?:யூடியூப்ல|யூடியூபில்|பாட்டு|போடு|பாடல்கள்|യൂട്യൂബിൽ|പാട്ട്|വെക്കൂ)\b", "", text).strip()
            slots["target_site"] = "youtube"
            slots["search_query"] = q or text
            return IntentPayload(intent=SemanticIntent.SEARCH_MEDIA_VIDEO, target_site="youtube", search_query=q or text, slots=slots)

        # Check algebra / math
        if re.search(r"\b(?:find|what('s| is)|solve)\b.*\b(?:value of\s+)?[a-zA-Z]\b.*\b(?:in|=|equal)", lower) or \
           re.search(r"\b[a-zA-Z]\s*(?:square|squared|\^2|²)\b", lower):
            slots["equation"] = text
            return IntentPayload(intent=SemanticIntent.MATH_ALGEBRA, equation=text, slots=slots)

        if re.search(r"(\d+\s*[\+\-\*\/×÷\^]\s*\d+)", lower) or re.search(r"\b(?:calculate|compute|what is)\b.*\d+", lower):
            return IntentPayload(intent=SemanticIntent.MATH_ARITHMETIC, slots=slots)

        if re.search(r"\b(?:convert)\s+\d+.*\b(?:to)\b", lower):
            return IntentPayload(intent=SemanticIntent.MATH_CONVERSION, slots=slots)

        # Check media / video
        if "youtube" in lower:
            q = re.sub(r"\b(?:open\s+youtube\s+and\s+search\s+(?:for\s+)?|search\s+youtube\s+for|search\s+for|on\s+youtube|play|watch|youtube)\b", "", lower, flags=re.IGNORECASE).strip()
            slots["target_site"] = "youtube"
            slots["search_query"] = q or text
            return IntentPayload(intent=SemanticIntent.SEARCH_MEDIA_VIDEO, target_site="youtube", search_query=q or text, slots=slots)

        # Check shopping / commercial
        from core.product_search import ProductSearchEngine
        prod_engine = ProductSearchEngine()
        if prod_engine.is_product_query(text) or any(w in lower for w in ["amazon", "flipkart", "buy", "price", "under rs", "budget", "sort by", "branded", "low to high", "high to low", "4 star", "5 star"]):
            parsed = prod_engine.parse_product_search(text)
            if parsed:
                slots["target_site"] = parsed.get("site", "Amazon").lower()
                slots["search_query"] = parsed.get("product")
                slots["product_category"] = parsed.get("product")
                slots["max_price"] = parsed.get("max_price")
                slots["min_price"] = parsed.get("min_price")
                slots["min_rating"] = 4 if parsed.get("has_rating_filter") else None
                slots["fast_delivery"] = parsed.get("has_fast_delivery", False)
                slots["delivery_days"] = parsed.get("delivery_days")
                slots["sort_by"] = parsed.get("sort_by")
                slots["is_branded"] = parsed.get("is_branded", False)
                slots["brands"] = parsed.get("brands", [])
                return IntentPayload(
                    intent=SemanticIntent.SEARCH_COMMERCIAL,
                    target_site=slots["target_site"],
                    search_query=slots["search_query"],
                    slots=slots,
                )
            return IntentPayload(intent=SemanticIntent.SEARCH_COMMERCIAL, search_query=text, slots=slots)

        # Check download software intent dynamically without hardcoding specific apps
        if re.search(r"\b(?:download|downloading|installer|install|setup|releases?|page\s+to\s+download|download\s+page)\b", lower):
            clean_entity = re.sub(r"\b(?:download|downloading|installer|install|setup|releases?|page\s+to\s+download|download\s+page|page|the|a|for)\b", "", lower).strip()
            slots["target_site"] = clean_entity or text
            slots["destination_section"] = "download"
            slots["search_query"] = text
            slots["entity"] = (clean_entity.title() + " Download") if clean_entity else "Download"
            return IntentPayload(intent=SemanticIntent.SEARCH_NAVIGATIONAL, target_site=clean_entity or text, search_query=text, slots=slots)

        # Check navigational site sections dynamically
        sec_match = re.search(r"\b(cart|orders|wishlist|history|account|trending|notifications|repos|issues)\b", lower)
        if sec_match:
            sec = sec_match.group(1)
            slots["destination_section"] = sec
            m_site = re.search(r"(?:on|in|at|from)\s+([a-zA-Z0-9_\-\.]+)", lower)
            if m_site:
                slots["target_site"] = m_site.group(1).strip()
            else:
                m_lead = re.match(r"^([a-zA-Z0-9_\-\.]+)\s+(?:cart|orders|wishlist|history|account|trending|notifications)", lower)
                if m_lead:
                    slots["target_site"] = m_lead.group(1).strip()
            if slots.get("target_site"):
                return IntentPayload(intent=SemanticIntent.SEARCH_NAVIGATIONAL, target_site=slots["target_site"], slots=slots)

        # Check consensus / opinions dynamically
        if any(w in lower for w in ["opinions", "thoughts", "reviews", "review", "is it worth", "rating of"]):
            m_target = re.search(r"\b(?:what\s+does\s+([a-zA-Z0-9_]+)\s+think\s+of|rating\s+of|reviews?\s+of)\s*(.*)", lower)
            if m_target and m_target.group(1):
                slots["target_site"] = m_target.group(1).strip()
                slots["search_query"] = (m_target.group(2) or text).strip()
            else:
                q = re.sub(r"\b(?:reviews?\s+of|review|is it worth|rating of)\b", "", lower).strip()
                slots["search_query"] = q or text
            return IntentPayload(intent=SemanticIntent.SEARCH_CONSENSUS, target_site=slots.get("target_site"), search_query=slots.get("search_query") or text, slots=slots)

        # Check screen
        if any(w in lower for w in ["on my screen", "this error", "look at this", "what is on screen"]):
            return IntentPayload(intent=SemanticIntent.SCREEN_ANALYZE, slots=slots)

        # Check OS application launch / switch / close
        if re.search(r"\b(?:open|launch|start|switch\s+to|bring\s+up|close|terminate|exit|kill)\s+([a-zA-Z0-9_\-\. ]+)", lower):
            m = re.search(r"\b(?:open|launch|start|switch\s+to|bring\s+up|close|terminate|exit|kill)\s+([a-zA-Z0-9_\-\. ]+)", lower)
            action = "close_app" if any(w in lower for w in ["close", "terminate", "exit", "kill"]) else "launch_app"
            slots["os_action"] = action
            slots["os_target"] = m.group(1).strip()
            return IntentPayload(intent=SemanticIntent.OS_COLLOQUIAL, os_action=action, slots=slots)

        # Default fallback to chitchat or factoid
        return IntentPayload(intent=SemanticIntent.CHITCHAT_BANTER, direct_speech=None, slots=slots)

