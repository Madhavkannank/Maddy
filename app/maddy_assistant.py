"""
Production M.A.D.D.Y. Assistant System Manager.
Unified entrypoint orchestrating Capability Router, Personality Engine,
Deterministic Math, Screen Awareness, Semantic Spotify, Permission Engine,
Supervisor, Memory Architecture, Vision, Voice, and Failure Recovery Engines.
"""
import datetime
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from config.settings import Settings, get_settings, get_memory_dir
from models.ollama_provider import OllamaProvider
from models.cloud_provider import CloudLLMProvider
from models.queue import ModelInferenceQueue
from security.permission_engine import PermissionEngine
from security.policy import PolicyMode
from security.confirmation import AutoApproveLowRiskConfirmationHandler, ConfirmationHandler
from security.autonomy import AutonomyEngine, AutonomyLevel
from tools.registry import ToolRegistry
from tools.executor import ToolExecutor
from tools.call import ToolCall
from tools.__init__ import register_default_tools
from agents.single_agent import SingleAgent
from agents.supervisor import Supervisor
from memory.working_memory import WorkingMemory
from memory.task_memory import TaskMemory
from memory.episodic_memory import EpisodicMemory
from memory.user_preferences import UserPreferences
from memory.context_engine import ContextCompressor
from memory.manager import MemoryManager
from memory.schema import MemoryType, MemoryScope, SourceType, CreatedBy

from vision.engine import VisionEngine
from voice.engine import VoiceEngine
from voice.wakeword import WakeWordEngine
from recovery.engine import RecoveryEngine
from personality.engine import PersonalityEngine
from core.math_engine import MathEngine
from core.router import CapabilityRouter, CapabilityType
from core.product_search import ProductSearchEngine
from core.file_save_flow import FileSaveManager
from core.research import ResearchEngine
from core.semantic_dispatcher import SemanticDispatcher
from core.intent_taxonomy import IntentPayload, SemanticIntent
from core.math_phonetics import MathPhoneticizer
from core.auto_memory import AutoMemoryExtractor
from computer.controller import ComputerController
from platform_layer import get_platform

KNOWN_WEBSITES = {
    "wikipedia": ("Wikipedia", "https://www.wikipedia.org"),
    "youtube": ("YouTube", "https://www.youtube.com"),
    "google": ("Google", "https://www.google.com"),
    "github": ("GitHub", "https://github.com"),
    "reddit": ("Reddit", "https://www.reddit.com"),
    "twitter": ("Twitter", "https://twitter.com"),
    "x": ("X", "https://x.com"),
    "facebook": ("Facebook", "https://www.facebook.com"),
    "instagram": ("Instagram", "https://www.instagram.com"),
    "chatgpt": ("ChatGPT", "https://chatgpt.com"),
    "openai": ("OpenAI", "https://openai.com"),
    "gmail": ("Gmail", "https://mail.google.com"),
    "amazon": ("Amazon", "https://www.amazon.in"),
    "amazon.in": ("Amazon", "https://www.amazon.in"),
    "amazon.com": ("Amazon", "https://www.amazon.com"),
    "flipkart": ("Flipkart", "https://www.flipkart.com"),
    "netflix": ("Netflix", "https://www.netflix.com"),
    "linkedin": ("LinkedIn", "https://www.linkedin.com"),
    "twitch": ("Twitch", "https://twitch.tv"),
    "discord": ("Discord", "https://discord.com"),
    "spotify": ("Spotify", "https://open.spotify.com"),
    "whatsapp": ("WhatsApp", "https://web.whatsapp.com"),
    "maps": ("Google Maps", "https://maps.google.com"),
    "google maps": ("Google Maps", "https://maps.google.com"),
    "stackoverflow": ("Stack Overflow", "https://stackoverflow.com"),
    "leetcode": ("LeetCode", "https://leetcode.com"),
    "hackerrank": ("HackerRank", "https://www.hackerrank.com"),
    "kaggle": ("Kaggle", "https://www.kaggle.com"),
    "geeksforgeeks": ("GeeksforGeeks", "https://www.geeksforgeeks.org"),
    "arxiv": ("arXiv", "https://arxiv.org"),
    "stack overflow": ("Stack Overflow", "https://stackoverflow.com"),
    "duckduckgo": ("DuckDuckGo", "https://duckduckgo.com"),
    "bing": ("Bing", "https://www.bing.com"),
    "ebay": ("eBay", "https://www.ebay.com"),
    "imdb": ("IMDb", "https://www.imdb.com"),
    "pinterest": ("Pinterest", "https://www.pinterest.com"),
    "pintrest": ("Pinterest", "https://www.pinterest.com"),
}

KNOWN_SEARCH_SITES = {
    "youtube": ("YouTube", "https://www.youtube.com/results?search_query={query}"),
    "yt": ("YouTube", "https://www.youtube.com/results?search_query={query}"),
    "you tube": ("YouTube", "https://www.youtube.com/results?search_query={query}"),
    "google": ("Google", "https://www.google.com/search?q={query}"),
    "google search": ("Google", "https://www.google.com/search?q={query}"),
    "google images": ("Google Images", "https://www.google.com/search?tbm=isch&q={query}"),
    "google photos": ("Google Images", "https://www.google.com/search?tbm=isch&q={query}"),
    "wikipedia": ("Wikipedia", "https://en.wikipedia.org/wiki/Special:Search?search={query}"),
    "wiki": ("Wikipedia", "https://en.wikipedia.org/wiki/Special:Search?search={query}"),
    "amazon": ("Amazon", "https://www.amazon.com/s?k={query}"),
    "amazon.in": ("Amazon", "https://www.amazon.in/s?k={query}"),
    "amazon.com": ("Amazon", "https://www.amazon.com/s?k={query}"),
    "flipkart": ("Flipkart", "https://www.flipkart.com/search?q={query}"),
    "reddit": ("Reddit", "https://www.reddit.com/search/?q={query}"),
    "github": ("GitHub", "https://github.com/search?q={query}"),
    "twitter": ("Twitter", "https://x.com/search?q={query}"),
    "x": ("X", "https://x.com/search?q={query}"),
    "maps": ("Google Maps", "https://www.google.com/maps/search/{query}"),
    "google maps": ("Google Maps", "https://www.google.com/maps/search/{query}"),
    "spotify": ("Spotify", "https://open.spotify.com/search/{query}"),
    "netflix": ("Netflix", "https://www.netflix.com/search?q={query}"),
    "twitch": ("Twitch", "https://twitch.tv/search?term={query}"),
    "stackoverflow": ("Stack Overflow", "https://stackoverflow.com/search?q={query}"),
    "stack overflow": ("Stack Overflow", "https://stackoverflow.com/search?q={query}"),
    "linkedin": ("LinkedIn", "https://www.linkedin.com/search/results/all/?keywords={query}"),
    "duckduckgo": ("DuckDuckGo", "https://duckduckgo.com/?q={query}"),
    "bing": ("Bing", "https://www.bing.com/search?q={query}"),
    "ebay": ("eBay", "https://www.ebay.com/sch/i.html?_nkw={query}"),
    "imdb": ("IMDb", "https://www.imdb.com/find?q={query}"),
    "pinterest": ("Pinterest", "https://www.pinterest.com/search/pins/?q={query}"),
    "pintrest": ("Pinterest", "https://www.pinterest.com/search/pins/?q={query}"),
}

KNOWN_SITE_SECTIONS = {
    # Amazon
    ("amazon", "cart"): ("Amazon Cart", "https://www.amazon.in/gp/cart/view.html"),
    ("amazon", "my cart"): ("Amazon Cart", "https://www.amazon.in/gp/cart/view.html"),
    ("amazon", "shopping cart"): ("Amazon Cart", "https://www.amazon.in/gp/cart/view.html"),
    ("amazon", "the cart"): ("Amazon Cart", "https://www.amazon.in/gp/cart/view.html"),
    ("amazon", "basket"): ("Amazon Cart", "https://www.amazon.in/gp/cart/view.html"),
    ("amazon", "orders"): ("Amazon Orders", "https://www.amazon.in/gp/your-account/order-history"),
    ("amazon", "my orders"): ("Amazon Orders", "https://www.amazon.in/gp/your-account/order-history"),
    ("amazon", "order history"): ("Amazon Orders", "https://www.amazon.in/gp/your-account/order-history"),
    ("amazon", "wishlist"): ("Amazon Wishlist", "https://www.amazon.in/hz/wishlist/ls"),
    ("amazon", "my wishlist"): ("Amazon Wishlist", "https://www.amazon.in/hz/wishlist/ls"),
    ("amazon", "account"): ("Amazon Account", "https://www.amazon.in/your-account"),
    ("amazon", "my account"): ("Amazon Account", "https://www.amazon.in/your-account"),
    ("amazon", "pay"): ("Amazon Pay", "https://www.amazon.in/amazonpay/home"),
    ("amazon", "amazon pay"): ("Amazon Pay", "https://www.amazon.in/amazonpay/home"),

    # Amazon.in
    ("amazon.in", "cart"): ("Amazon Cart", "https://www.amazon.in/gp/cart/view.html"),
    ("amazon.in", "my cart"): ("Amazon Cart", "https://www.amazon.in/gp/cart/view.html"),
    ("amazon.in", "shopping cart"): ("Amazon Cart", "https://www.amazon.in/gp/cart/view.html"),
    ("amazon.in", "orders"): ("Amazon Orders", "https://www.amazon.in/gp/your-account/order-history"),
    ("amazon.in", "my orders"): ("Amazon Orders", "https://www.amazon.in/gp/your-account/order-history"),
    ("amazon.in", "wishlist"): ("Amazon Wishlist", "https://www.amazon.in/hz/wishlist/ls"),
    ("amazon.in", "my wishlist"): ("Amazon Wishlist", "https://www.amazon.in/hz/wishlist/ls"),
    ("amazon.in", "account"): ("Amazon Account", "https://www.amazon.in/your-account"),
    ("amazon.in", "pay"): ("Amazon Pay", "https://www.amazon.in/amazonpay/home"),

    # Amazon.com
    ("amazon.com", "cart"): ("Amazon Cart", "https://www.amazon.com/gp/cart/view.html"),
    ("amazon.com", "my cart"): ("Amazon Cart", "https://www.amazon.com/gp/cart/view.html"),
    ("amazon.com", "shopping cart"): ("Amazon Cart", "https://www.amazon.com/gp/cart/view.html"),
    ("amazon.com", "orders"): ("Amazon Orders", "https://www.amazon.com/gp/your-account/order-history"),
    ("amazon.com", "my orders"): ("Amazon Orders", "https://www.amazon.com/gp/your-account/order-history"),
    ("amazon.com", "wishlist"): ("Amazon Wishlist", "https://www.amazon.com/hz/wishlist/ls"),
    ("amazon.com", "my wishlist"): ("Amazon Wishlist", "https://www.amazon.com/hz/wishlist/ls"),
    ("amazon.com", "account"): ("Amazon Account", "https://www.amazon.com/your-account"),

    # Flipkart
    ("flipkart", "cart"): ("Flipkart Cart", "https://www.flipkart.com/viewcart"),
    ("flipkart", "my cart"): ("Flipkart Cart", "https://www.flipkart.com/viewcart"),
    ("flipkart", "orders"): ("Flipkart Orders", "https://www.flipkart.com/account/orders"),
    ("flipkart", "my orders"): ("Flipkart Orders", "https://www.flipkart.com/account/orders"),
    ("flipkart", "wishlist"): ("Flipkart Wishlist", "https://www.flipkart.com/wishlist"),
    ("flipkart", "my wishlist"): ("Flipkart Wishlist", "https://www.flipkart.com/wishlist"),
    ("flipkart", "account"): ("Flipkart Account", "https://www.flipkart.com/account"),

    # YouTube
    ("youtube", "history"): ("YouTube History", "https://www.youtube.com/feed/history"),
    ("youtube", "my history"): ("YouTube History", "https://www.youtube.com/feed/history"),
    ("youtube", "watch history"): ("YouTube History", "https://www.youtube.com/feed/history"),
    ("youtube", "subscriptions"): ("YouTube Subscriptions", "https://www.youtube.com/feed/subscriptions"),
    ("youtube", "my subscriptions"): ("YouTube Subscriptions", "https://www.youtube.com/feed/subscriptions"),
    ("youtube", "subs"): ("YouTube Subscriptions", "https://www.youtube.com/feed/subscriptions"),
    ("youtube", "library"): ("YouTube Library", "https://www.youtube.com/feed/library"),
    ("youtube", "watch later"): ("YouTube Watch Later", "https://www.youtube.com/playlist?list=WL"),
    ("youtube", "trending"): ("YouTube Trending", "https://www.youtube.com/feed/trending"),

    # GitHub
    ("github", "notifications"): ("GitHub Notifications", "https://github.com/notifications"),
    ("github", "my notifications"): ("GitHub Notifications", "https://github.com/notifications"),
    ("github", "repos"): ("GitHub Repositories", "https://github.com/?tab=repositories"),
    ("github", "repositories"): ("GitHub Repositories", "https://github.com/?tab=repositories"),
    ("github", "my repos"): ("GitHub Repositories", "https://github.com/?tab=repositories"),
    ("github", "my repositories"): ("GitHub Repositories", "https://github.com/?tab=repositories"),
    ("github", "pull requests"): ("GitHub Pull Requests", "https://github.com/pulls"),
    ("github", "prs"): ("GitHub Pull Requests", "https://github.com/pulls"),
    ("github", "issues"): ("GitHub Issues", "https://github.com/issues"),
    ("github", "my issues"): ("GitHub Issues", "https://github.com/issues"),
    ("github", "trending"): ("GitHub Trending", "https://github.com/trending"),
    ("github", "settings"): ("GitHub Settings", "https://github.com/settings"),

    # Twitter / X
    ("x", "bookmarks"): ("X Bookmarks", "https://x.com/i/bookmarks"),
    ("x", "notifications"): ("X Notifications", "https://x.com/notifications"),
    ("x", "messages"): ("X Messages", "https://x.com/messages"),
    ("x", "dms"): ("X Messages", "https://x.com/messages"),
    ("twitter", "bookmarks"): ("Twitter Bookmarks", "https://x.com/i/bookmarks"),
    ("twitter", "notifications"): ("Twitter Notifications", "https://x.com/notifications"),
    ("twitter", "messages"): ("Twitter Messages", "https://x.com/messages"),
    ("twitter", "dms"): ("Twitter Messages", "https://x.com/messages"),

    # Reddit
    ("reddit", "saved"): ("Reddit Saved", "https://www.reddit.com/user/me/saved"),
    ("reddit", "upvoted"): ("Reddit Upvoted", "https://www.reddit.com/user/me/upvoted"),
    ("reddit", "inbox"): ("Reddit Inbox", "https://www.reddit.com/message/inbox"),
    ("reddit", "notifications"): ("Reddit Notifications", "https://www.reddit.com/message/inbox"),

    # Gmail
    ("gmail", "inbox"): ("Gmail Inbox", "https://mail.google.com/mail/u/0/#inbox"),
    ("gmail", "sent"): ("Gmail Sent", "https://mail.google.com/mail/u/0/#sent"),
    ("gmail", "drafts"): ("Gmail Drafts", "https://mail.google.com/mail/u/0/#drafts"),
    ("gmail", "starred"): ("Gmail Starred", "https://mail.google.com/mail/u/0/#starred"),
    ("gmail", "trash"): ("Gmail Trash", "https://mail.google.com/mail/u/0/#trash"),
    ("gmail", "spam"): ("Gmail Spam", "https://mail.google.com/mail/u/0/#spam"),

    # Spotify
    ("spotify", "liked"): ("Spotify Liked Songs", "https://open.spotify.com/collection/tracks"),
    ("spotify", "liked songs"): ("Spotify Liked Songs", "https://open.spotify.com/collection/tracks"),
    ("spotify", "playlists"): ("Spotify Playlists", "https://open.spotify.com/collection/playlists"),
    ("spotify", "library"): ("Spotify Library", "https://open.spotify.com/collection/playlists"),

    # Netflix
    ("netflix", "my list"): ("Netflix My List", "https://www.netflix.com/browse/my-list"),
    ("netflix", "watchlist"): ("Netflix Watchlist", "https://www.netflix.com/browse/my-list"),

    # LinkedIn
    ("linkedin", "notifications"): ("LinkedIn Notifications", "https://www.linkedin.com/notifications/"),
    ("linkedin", "jobs"): ("LinkedIn Jobs", "https://www.linkedin.com/jobs/"),
    ("linkedin", "network"): ("LinkedIn Network", "https://www.linkedin.com/mynetwork/"),
    ("linkedin", "messages"): ("LinkedIn Messages", "https://www.linkedin.com/messaging/"),
}

class MaddyAssistant:
    """
    M.A.D.D.Y. Assistant Main System Engine.
    Unifies reasoning, capability routing, deterministic math, screen awareness,
    tool execution, security policies, context compression, and multi-agent execution.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        policy_mode: PolicyMode = PolicyMode.BALANCED,
        autonomy_level: AutonomyLevel = AutonomyLevel.EXECUTE_TRUSTED,
        confirmation_handler: Optional[ConfirmationHandler] = None,
    ):
        self.settings = settings or get_settings()
        self.memory_dir = get_memory_dir()

        # 1. Intelligence Layer
        cloud_p = CloudLLMProvider()
        if cloud_p.is_configured():
            self.model_provider = cloud_p
        else:
            self.model_provider = OllamaProvider(settings=self.settings)
        self.ollama = self.model_provider
        self.inference_queue = ModelInferenceQueue(self.model_provider)

        # 2. Tool & Security Layer
        self.registry = ToolRegistry()
        register_default_tools(self.registry)

        self.confirmation_handler = confirmation_handler or AutoApproveLowRiskConfirmationHandler()
        self.permission_engine = PermissionEngine(
            policy_mode=policy_mode,
            confirmation_handler=self.confirmation_handler,
        )
        self.autonomy_engine = AutonomyEngine(current_level=autonomy_level)
        self.executor = ToolExecutor(registry=self.registry, permission_engine=self.permission_engine)

        # 3. Memory & Context Layer
        self.user_preferences = UserPreferences()
        self.episodic_memory = EpisodicMemory()
        self.memory_manager = MemoryManager(user_preferences=self.user_preferences, episodic_memory=self.episodic_memory)
        self.context_compressor = ContextCompressor()


        # 4. Perception & Multimodal Layer
        self.vision_engine = VisionEngine()
        self.voice_engine = VoiceEngine()
        self.wakeword_engine = WakeWordEngine(on_command_ready=self.execute)
        self.recovery_engine = RecoveryEngine(max_retries=2)
        self.computer_controller = ComputerController()
        self.platform = get_platform()

        # 5. Personality, Router & Math Engines
        self.personality = PersonalityEngine(user_preferences=self.user_preferences, memory_manager=self.memory_manager)
        self.math_engine = MathEngine()
        self.router = CapabilityRouter(math_engine=self.math_engine)

        # 5b. Verbal Micro-Feedback Engine (Phase 6)
        from voice.micro_feedback import MicroFeedbackEngine
        self.micro_feedback = MicroFeedbackEngine(
            speak_fn=self._speak_micro_feedback,
            address_title=self.personality.get_address_title() or "boss",
        )

        # 6. Orchestration Engine
        self.supervisor = Supervisor(
            model_provider=self.inference_queue,
            executor=self.executor,
            registry=self.registry,
            max_retries=2,
            episodic_memory=self.episodic_memory,
            on_tool_start=self.micro_feedback.on_tool_start,
        )

        # 6b. M.A.D.D.Y. autonomous agent
        from agents.maddy_agent import MaddyAgent
        self.maddy_agent = MaddyAgent(
            model_provider=self.inference_queue,
            executor=self.executor,
            registry=self.registry,
            max_steps=10,
            episodic_memory=self.episodic_memory,
            verification_engine=getattr(self.supervisor, "verification_engine", None),
            address_title=self.personality.get_address_title() or "boss",
            on_tool_start=self.micro_feedback.on_tool_start,
            memory_manager=self.memory_manager,
        )

        # 7. Specialized Shopping, File Saving & Research Engines
        self.product_search_engine = ProductSearchEngine()
        self.file_save_manager = FileSaveManager(workspace_dir=os.getcwd())
        self.research_engine = ResearchEngine()

        # 7b. Spotify Integration Engine
        try:
            from tools.spotify import SpotifyManager
            self.spotify = SpotifyManager(user_preferences=self.user_preferences)
        except Exception:
            self.spotify = None

        # 8. Time, Countdown & Alert Engine
        from tools.timer import TimerManager
        self.timer_manager = TimerManager(voice_callback=self._speak_timer_alert)
        self._last_assistant_response: Optional[str] = None
        self._active_product_search: Optional[Dict[str, Any]] = None
        self._last_executed_command_type: Optional[str] = None

        # 8b. Scheduled Cron & Autonomous Heartbeat Engine (Track 2)
        from tasks.cron_engine import CronEngine
        from tools.cron_tool import set_cron_engine
        self.cron_engine = CronEngine(
            job_executor=lambda goal: self.execute(goal, speak_response=False),
            voice_notifier=lambda text: self._speak_output(text, True),
        )
        set_cron_engine(self.cron_engine)
        self.cron_engine.start()

        self.semantic_dispatcher = SemanticDispatcher(model_provider=self.model_provider)
        self.dispatcher = self.semantic_dispatcher

        # 9. Camera & Webcam Snapshot Engine
        from tools.camera import CameraEngine
        self.camera_engine = CameraEngine()

        # 10. Multi-Turn Conversation History (Rolling dialogue context)
        self.conversation_history: List[Dict[str, str]] = []
        self.auto_memory_extractor = AutoMemoryExtractor()
        self.session_memory: Dict[str, Any] = {
            "math_context": {"expression": None, "variables": []},
            "response_rules": [],
        }
        self._restore_session_math_context()
        self._restore_response_rules()

        # 11. Hardware, GPU & Background Training Watcher
        from hardware.gpu_monitor import GPUMonitor, TrainingWatcher
        self.gpu_monitor = GPUMonitor()
        self.training_watcher = TrainingWatcher(
            gpu_monitor=self.gpu_monitor,
            on_alert_callback=self._handle_hardware_alert,
            check_interval=4.0,
        )

        # 12. Session Capture Engine
        from session_capture import get_session_capture_manager
        self.session_capture_manager = get_session_capture_manager(
            permission_engine=self.permission_engine
        )

    def _handle_hardware_alert(self, alert_type: str, message: str, proc: Optional[Any] = None) -> None:
        """Handle proactive background hardware, process completion, and training alerts."""
        # 1. Spoken audio announcement
        try:
            if hasattr(self, "voice_engine") and self.voice_engine:
                self.voice_engine.speak(message)
        except Exception:
            pass

        # 2. Dispatch mobile Telegram alert for process completion or VRAM warnings
        channel = getattr(proc, "notify_channel", "telegram") if proc else "telegram"
        if alert_type == "PROCESS_EXIT":
            if channel in ("telegram", "all"):
                try:
                    from integrations.messaging import TelegramClient
                    tg = TelegramClient()
                    tg.send_alert(
                        title="Process Execution Complete",
                        body=message,
                        level="INFO",
                    )
                except Exception:
                    pass
        elif alert_type == "VRAM_WARNING":
            try:
                from integrations.messaging import TelegramClient
                tg = TelegramClient()
                tg.send_alert(
                    title="RTX 3050 VRAM Warning",
                    body=message,
                    level="WARNING",
                )
            except Exception:
                pass

    def _store_math_context(self, expression: Optional[str], variables: Optional[List[str]] = None) -> None:
        """Persist a math expression in session memory and the long-term store."""
        if not expression:
            return
        normalized_vars = list(variables or [])
        self.session_memory["math_context"] = {"expression": expression, "variables": normalized_vars}
        try:
            self.memory_manager.remember(
                content=f"Math context: expression={expression}; variables={normalized_vars}",
                memory_type=MemoryType.TASK_STATE,
                scope=MemoryScope.SESSION,
                source_type=SourceType.TASK_RESULT,
                source_reference="math_context",
                source_text=expression,
                created_by=CreatedBy.SYSTEM,
                confidence=1.0,
                importance=0.9,
                tags=["math", "session", "context"],
                metadata={"kind": "math_context", "expression": expression, "variables": normalized_vars},
            )
        except Exception:
            pass

    def _restore_session_math_context(self) -> None:
        """Rehydrate the most recent math expression from the persisted session memory."""
        if not hasattr(self, "memory_manager"):
            return
        try:
            session_records = self.memory_manager.store.list_records(scope=MemoryScope.SESSION, limit=20)
            for rec in sorted(session_records, key=lambda r: r.created_at, reverse=True):
                meta = rec.metadata or {}
                if meta.get("kind") == "math_context" and rec.content.startswith("Math context:"):
                    expr = meta.get("expression") or rec.content.split("expression=", 1)[1].split(";", 1)[0]
                    vars_list = meta.get("variables") or []
                    self.math_engine.last_expression = expr
                    self.math_engine.last_variables = list(vars_list)
                    self.session_memory["math_context"] = {"expression": expr, "variables": list(vars_list)}
                    return
        except Exception:
            pass

    def _restore_response_rules(self) -> None:
        """Load trigger-response rules from memory at startup."""
        if not hasattr(self, "memory_manager"):
            return
        try:
            records = self.memory_manager.store.list_records(limit=100)
            for rec in records:
                meta = rec.metadata or {}
                if meta.get("kind") == "response_rule":
                    self.session_memory["response_rules"].append({
                        "trigger": meta.get("trigger", ""),
                        "response": meta.get("response", ""),
                    })
        except Exception:
            pass

    def _apply_response_rule(self, text: str) -> Optional[str]:
        """Return the stored response if a trigger phrase matches the current input."""
        if not text or not self.session_memory["response_rules"]:
            return None
        clean = text.strip().rstrip("?!.,;: ")
        lower_clean = clean.lower()
        for rule in self.session_memory["response_rules"]:
            trigger = str(rule.get("trigger", "")).strip()
            if not trigger:
                continue
            if lower_clean == trigger.lower():
                return str(rule.get("response", ""))
        return None

    def _remember_response_rule(self, trigger: str, response: str) -> Optional[str]:
        """Persist a custom trigger-response rule in session memory and the long-term store."""
        trigger = str(trigger).strip().rstrip("?!.,;: ")
        response = str(response).strip().rstrip("?!.,;: ")
        if not trigger or not response:
            return None
        rule = {"trigger": trigger, "response": response}
        if rule not in self.session_memory["response_rules"]:
            self.session_memory["response_rules"].append(rule)
        try:
            self.memory_manager.remember(
                content=f"When user says '{trigger}', respond with '{response}'",
                memory_type=MemoryType.USER_PREFERENCE,
                scope=MemoryScope.GLOBAL,
                source_type=SourceType.USER_MESSAGE,
                source_reference="response_rule",
                source_text=f"next time i say {trigger}, you should say {response}",
                created_by=CreatedBy.USER,
                confidence=1.0,
                importance=0.95,
                tags=["response_rule", "instruction"],
                metadata={"kind": "response_rule", "trigger": trigger, "response": response},
            )
        except Exception:
            pass
        return f"Noted — next time you say '{trigger}', I'll respond with '{response}'."

    def execute(self, user_request: str, speak_response: bool = False, _is_subcommand: bool = False) -> Dict[str, Any]:
        """
        Execute a user natural-language request end-to-end.
        Routes intelligently through Fast Path (Math, Conversation, Spotify, Browser, Screen)
        or Supervisor Task DAG. Supports multi-intent compound commands dynamically.
        """
        if not user_request or not user_request.strip():
            return {
                "success": False,
                "error": "Request cannot be empty",
            }

        if not _is_subcommand and hasattr(self, "micro_feedback") and self.micro_feedback:
            self.micro_feedback.reset_turn()

        clean_request = user_request.strip()
        self._restore_session_math_context()
        rule_response = self._apply_response_rule(clean_request)
        if rule_response:
            return self._build_fast_response(rule_response, f"maddy_req_{int(time.time())}", time.time(), TaskMemory(task_id=f"maddy_req_{int(time.time())}"), speak_response)
        request_id = f"maddy_req_{int(time.time())}"
        start_time = time.time()

        # Normalize phonetic / speech recognition slips
        clean_request = re.sub(r"\b(?:on\s+)?spot\s*with\s*(?:it)?\b", "on spotify", clean_request, flags=re.IGNORECASE)
        clean_request = re.sub(r"\bspot\s*if\s*i\b", "spotify", clean_request, flags=re.IGNORECASE)
        clean_request = re.sub(r"\b(?:pin\s*guy|pen\s*guy|pin\s*dive|pin\s*drive)\b", "pendrive", clean_request, flags=re.IGNORECASE)

        # Check retry commands ("try again", "retry", "do it again", "try that again")
        retry_patterns = [r"^(?:try\s+again|retry|do\s+it\s+again|try\s+that\s+again|one\s+more\s+time)$"]
        if any(re.match(p, clean_request.lower()) for p in retry_patterns):
            last_cmd = getattr(self, "_last_user_command", None)
            if last_cmd and last_cmd.lower() != clean_request.lower():
                logger.info(f"Retrying previous user command: '{last_cmd}'")
                return self.execute(last_cmd, speak_response=speak_response, _is_subcommand=_is_subcommand)

        # Store for future retries if this is not a retry query itself
        if not any(re.match(p, clean_request.lower()) for p in retry_patterns):
            self._last_user_command = clean_request

        # 1. Initialize TaskMemory and WorkingMemory
        task_mem = TaskMemory(task_id=request_id)
        task_mem.save_goal(clean_request)
        working_mem = WorkingMemory(goal=clean_request)

        # 1b. Check for multi-intent compound commands (e.g. "set volume to max and play X on spotify, and also find Y")
        if not _is_subcommand:
            sub_commands = self.decompose_compound_request(clean_request)
            if len(sub_commands) > 1:
                return self._execute_compound_request(
                    sub_commands, clean_request, request_id, start_time, task_mem, speak_response
                )

        # 2. Check Name Privacy / Conversational Intercepts ("What is my name?", "Yes/No", "Call me X")
        handled, intercept_resp = self.personality.handle_conversational_intercept(clean_request)
        if handled and intercept_resp:
            return self._build_fast_response(intercept_resp, request_id, start_time, task_mem, speak_response)

        # 2b. Check Conversational File Saving Flow ("save this info to a file", location resolution, dir creation)
        handled_save, save_resp = self.file_save_manager.handle_turn(
            clean_request,
            address=self.personality.get_address_title(),
            recent_context=self._last_assistant_response,
        )
        if handled_save and save_resp:
            return self._build_fast_response(save_resp, request_id, start_time, task_mem, speak_response)

        # 2c. Check Hardware / Screen Brightness Controls
        bright_resp = self._handle_brightness_command(clean_request)
        if bright_resp:
            return self._build_fast_response(bright_resp, request_id, start_time, task_mem, speak_response)

        # 2d. Check Hardware / Master Audio Volume Controls
        vol_resp = self._handle_volume_command(clean_request)
        if vol_resp:
            return self._build_fast_response(vol_resp, request_id, start_time, task_mem, speak_response)

        # 2e. Check Vision & Cursor Automation Controls (Clicks, Scrolling, Typing, Screen Analysis)
        vision_resp = self._handle_vision_cursor_command(clean_request, request_id, task_mem)
        if vision_resp:
            return self._build_fast_response(vision_resp, request_id, start_time, task_mem, speak_response)

        # 2e-2. Check Workstation Lock Controls ("lock pc", "lock the pc", "lock screen", "lock workstation")
        lock_patterns = [
            r"\b(?:lock\s+(?:the\s+)?(?:pc|laptop|computer|screen|workstation|system|windows))\b",
            r"^lock$",
        ]
        if any(re.search(p, clean_request.lower()) for p in lock_patterns):
            locked = self.computer_controller.lock_workstation()
            addr = self.personality.get_address_title() or "boss"
            if locked:
                return self._build_fast_response(f"The PC has been locked, {addr}.", request_id, start_time, task_mem, speak_response)
            else:
                return self._build_fast_response(f"Attempted to lock the PC, but the operating system call failed, {addr}.", request_id, start_time, task_mem, speak_response)

        # 2f. Check Timer and Countdown Controls
        timer_resp = self._handle_timer_command(clean_request)
        if timer_resp:
            return self._build_fast_response(timer_resp, request_id, start_time, task_mem, speak_response)

        # 2f-2. Check Counting Requests ("count from 1 to 100", "count to 20", "count down from 10 to 1")
        count_resp = self._handle_counting_command(clean_request)
        if count_resp:
            return self._build_fast_response(count_resp, request_id, start_time, task_mem, speak_response)

        # 2g. Check Camera and Webcam Controls
        cam_resp = self._handle_camera_command(clean_request)
        if cam_resp:
            return self._build_fast_response(cam_resp, request_id, start_time, task_mem, speak_response)

        # 2h. Check dynamic collection memory commands (add item to / list / remove nth item)
        collection_response = self._handle_collection_memory_command(clean_request)
        if collection_response:
            return self._build_fast_response(collection_response, request_id, start_time, task_mem, speak_response)

        # 2h. Check Explicit Memory Commands (remember, forget, what do you remember, why do you remember)
        mem_resp = self._handle_memory_command(clean_request)
        if mem_resp:
            return self._build_fast_response(mem_resp, request_id, start_time, task_mem, speak_response)

        # 2h-2. Trigger-response memory commands like "next time I say hip hip, you should say hurray"
        rule_pattern = re.search(
            r"next\s+time\s+i\s+say\s+(.+?)(?:,|\s+you\s+should\s+(?:say|respond\s+with))\s+(.+)$",
            clean_request,
            flags=re.IGNORECASE,
        )
        if rule_pattern:
            trigger = rule_pattern.group(1).strip().rstrip("?!.,;: ")
            response = rule_pattern.group(2).strip().rstrip("?!.,;: ")
            saved = self._remember_response_rule(trigger, response)
            if saved:
                return self._build_fast_response(saved, request_id, start_time, task_mem, speak_response)

        # 2h-3. Coding-time shortcut: "it's time to code" / "coding time" -> LeetCode
        if re.search(r"\b(?:coding\s+time|it'?s\s+time\s+to\s+code|time\s+for\s+leetcode|leetcode\s+time|let'?s\s+do\s+some\s+leetcode|practice\s+leetcode)\b", clean_request, flags=re.IGNORECASE):
            try:
                from browser import browser_tab_manager
                reuse_res = browser_tab_manager.open_or_reuse("leetcode", "https://leetcode.com", force_navigate=False)
                if reuse_res.get("success"):
                    msg = "Opened LeetCode and brought it to the front, boss."
                    return self._build_fast_response(msg, request_id, start_time, task_mem, speak_response)
            except Exception:
                pass

        # 2i. Autonomous Background Memory Extraction (ChatGPT / Gemini style)
        if hasattr(self, "auto_memory_extractor") and hasattr(self, "memory_manager"):
            auto_mem_content = self.auto_memory_extractor.process_and_remember(clean_request, self.memory_manager)
            if auto_mem_content:
                m_name = re.match(r"^User name is\s+(.+)$", auto_mem_content, re.IGNORECASE)
                if m_name and hasattr(self, "user_preferences"):
                    new_name = m_name.group(1).strip()
                    self.user_preferences.set_preference("user.real_name", new_name)
                    self.user_preferences.set_preferred_name(new_name)
                if self._is_pure_declarative_statement(clean_request):
                    addr = self.personality.get_address_title() or "boss"
                    ack_msg = f"Got it, {addr}. I've committed that to memory: {auto_mem_content}."
                    return self._build_fast_response(ack_msg, request_id, start_time, task_mem, speak_response)

        # 2j. Check Hardware Status & VRAM Monitoring
        hw_resp = self._handle_hardware_status_command(clean_request)
        if hw_resp:
            return self._build_fast_response(hw_resp, request_id, start_time, task_mem, speak_response)

        # 2k. Check ArXiv Research Paper Digest
        arxiv_resp = self._handle_arxiv_command(clean_request)
        if arxiv_resp:
            return self._build_fast_response(arxiv_resp, request_id, start_time, task_mem, speak_response)

        # 2k2. Check Process / Task Completion Monitoring (e.g. "when it completes notify me on telegram")
        proc_mon_resp = self._handle_process_monitor_command(clean_request)
        if proc_mon_resp:
            return self._build_fast_response(proc_mon_resp, request_id, start_time, task_mem, speak_response)

        # 2l. Check Messaging Controls (Telegram / WhatsApp)
        msg_resp = self._handle_messaging_command(clean_request)
        if msg_resp:
            return self._build_fast_response(msg_resp, request_id, start_time, task_mem, speak_response)

        # 2m. Check Owner Session Capture & Guest Audit Controls
        capture_resp = self._handle_session_capture_command(clean_request, request_id)
        if capture_resp:
            return self._build_fast_response(capture_resp, request_id, start_time, task_mem, speak_response)

        # 2n. Check Fast-Path Application Lifecycle & Window Switching Controls (open, close, switch)
        app_resp = self._handle_app_lifecycle_command(clean_request)
        if app_resp:
            return self._build_fast_response(app_resp, request_id, start_time, task_mem, speak_response)

        # 3. Route Capability
        route_info = self.router.route(clean_request)
        primary_cap = route_info["primary"]

        # 4. PATH A — Deterministic Safe Math & Substitution
        if primary_cap == CapabilityType.MATH or (hasattr(self, "math_engine") and self.math_engine.is_math_query(clean_request)):
            math_det = self.math_engine.evaluate_detailed(clean_request)
            if math_det:
                self._store_math_context(self.math_engine.last_expression, self.math_engine.last_variables)
                display_math = self.personality.format_response(str(math_det["display"]))
                speech_math = math_det.get("speech", display_math)
                return self._build_fast_response(display_math, request_id, start_time, task_mem, speak_response, speech_text=speech_math)

        favorite_song_query = bool(re.search(r"\b(?:what(?:'s|\s+is)|which)\s+(?:my\s+)?(?:favorite|favourite|fav)\s+(?:song|music)\b", clean_request.lower()))
        if favorite_song_query:
            saved_song = self._get_saved_favorite_song()
            if saved_song:
                response = f"Your favorite song is {saved_song}, {self.personality.get_address_title()}."
                return self._build_fast_response(response, request_id, start_time, task_mem, speak_response)

        play_favorite_song_query = bool(re.search(r"\b(?:play|start)\s+(?:my\s+)?(?:favorite|favourite|fav)\s+(?:song|music)\b", clean_request.lower()))
        if play_favorite_song_query:
            saved_song = self._get_saved_favorite_song()
            if saved_song:
                spotify_resp = self._handle_direct_spotify_command(f"play {saved_song}")
                if spotify_resp:
                    return self._build_fast_response(f"Playing your favorite song: {saved_song}.", request_id, start_time, task_mem, speak_response)

        # 4a-2. Product Search Pagination ("next", "move to next page", "next page")
        if self._is_pagination_query(clean_request):
            page_resp = self._handle_product_pagination(clean_request)
            if page_resp:
                self._last_executed_command_type = "product_pagination"
                extra = page_resp.get("data", {})
                return self._build_fast_response(page_resp["response"], request_id, start_time, task_mem, speak_response, extra_data=extra)

        # 4a-3. Error Correction / Negation ("No da kanna , it's not", "no that's not it")
        negation_resp = self._handle_negation_recovery(clean_request)
        if negation_resp:
            self._last_executed_command_type = "negation_correction"
            return self._build_fast_response(negation_resp["response"], request_id, start_time, task_mem, speak_response, extra_data=negation_resp.get("data", {}))

        # 4b. PATH S — Direct Semantic Spotify Controls
        lower_req = clean_request.lower()
        is_conversational_inquiry = bool(
            re.search(
                r"^(?:do\s+(?:you|u)|can\s+(?:you|u)|could\s+(?:you|u)|would\s+(?:you|u)|will\s+(?:you|u)|are\s+(?:you|u)|have\s+(?:you|u)|did\s+(?:you|u)|how\s+do\s+(?:you|u)|how\s+to|why\s+do\s+(?:you|u)|what\s+(?:do|is|are|can)\s+(?:you|u))\b",
                lower_req,
            )
            or re.search(r"\b(?:chess|game|games|football|cricket|tennis|basketball|soccer|hockey|badminton|guitar|piano|violin|drums|flute|cards|poker|monopoly|board\s*game|video\s*game|with\s+me)\b", lower_req)
        )
        has_explicit_music_ref = any(k in lower_req for k in [
            "spotify", "song", "songs", "music", "track", "tracks", "playlist", "album", "artist", "lofi", "melody", "melodies", "soundtrack", "listen to"
        ])

        is_spotify_command = not (is_conversational_inquiry and not has_explicit_music_ref) and (
            (primary_cap == CapabilityType.SPOTIFY and not is_conversational_inquiry)
            or "spotify" in lower_req
            or (CapabilityType.SPOTIFY.value in route_info.get("all_capabilities", []) and not is_conversational_inquiry)
            or (
                re.search(r"\b(?:play|listen\s+to)\b.+\b(?:song|songs|track|tracks|music|album)\b", lower_req)
                and not any(w in lower_req for w in ["youtube", "video", "browser", "chrome", "firefox"])
            )
            or (
                re.search(r"^\s*play\s+[a-zA-Z0-9_\- ']+?\s+(?:from|at)\s+\d+", lower_req)
                and not any(w in lower_req for w in ["youtube", "video"])
            )
        )
        if is_spotify_command:
            spotify_resp = self._handle_direct_spotify_command(clean_request)
            if spotify_resp:
                self._last_executed_command_type = "spotify_playback"
                return self._build_fast_response(spotify_resp, request_id, start_time, task_mem, speak_response)

        # 4c. Active Media Follow-up (e.g. user previously requested Spotify, and now answers with a genre/song like "Some melody in Tamil")
        if getattr(self, "_active_media_platform", None) == "spotify":
            lower_clean = clean_request.lower().strip().rstrip(".?!")
            music_keywords = [
                "song", "songs", "melody", "melodies", "music", "track", "tracks", "playlist", "artist", "singer",
                "tamil", "hindi", "malayalam", "telugu", "kannada", "english", "rock", "pop", "jazz", "lofi", "lo-fi",
                "classical", "folk", "sad", "party", "romantic", "instrumental", "acoustic", "bass", "remix"
            ]
            is_likely_music_query = any(k in lower_clean for k in music_keywords) or (
                len(lower_clean.split()) <= 5
                and not any(cmd in lower_clean for cmd in ["open", "close", "launch", "exit", "kill", "lock", "screenshot", "weather", "battery", "time", "calculate", "solve", "what", "who", "where", "why", "how"])
            )
            if is_likely_music_query and not any(site in lower_clean for site in ["youtube", "amazon", "flipkart", "google"]):
                spotify_resp = self._handle_direct_spotify_command(f"play {clean_request} on spotify")
                if spotify_resp:
                    return self._build_fast_response(spotify_resp, request_id, start_time, task_mem, speak_response)

        # 5. PATH B — Screen-Aware Automatic Perception
        if route_info.get("is_screen_dependent"):
            screen_resp = self._handle_screen_aware_request(clean_request, request_id, task_mem)
            if screen_resp:
                return self._build_fast_response(screen_resp, request_id, start_time, task_mem, speak_response)

        # 5b. PATH R — Multi-Source Dynamic Review Research & Recommendation Synthesis (Reddit, IMDb, Consensus)
        if hasattr(self, "research_engine") and self.research_engine.is_research_or_recommendation_query(clean_request):
            research_res = self.research_engine.research_and_synthesize(
                clean_request,
                model_provider=self.model_provider,
                address=self.personality.get_address_title(),
            )
            if research_res.get("success") and research_res.get("response"):
                formatted_resp = self.personality.format_response(research_res["response"])
                self._last_assistant_response = formatted_resp
                return self._build_fast_response(formatted_resp, request_id, start_time, task_mem, speak_response)

        # 5c. TIER 2 — Granular Semantic Intent Dispatcher (Groq LPU)
        dispatcher = getattr(self, "dispatcher", None) or getattr(self, "semantic_dispatcher", None)
        if dispatcher and hasattr(self, "model_provider"):
            dispatcher.model_provider = self.model_provider
        intent_payload = dispatcher.dispatch(clean_request) if dispatcher else IntentPayload(intent=SemanticIntent.UNKNOWN)

        # S0. Multimodal Screen Awareness & Visual Inspection via Dispatcher or Router
        if intent_payload.intent == SemanticIntent.SCREEN_ANALYZE or route_info.get("is_screen_dependent"):
            screen_resp = self._handle_screen_aware_request(clean_request, request_id, task_mem)
            if screen_resp:
                return self._build_fast_response(screen_resp, request_id, start_time, task_mem, speak_response)

        # S1. Math & Algebra via Dispatcher
        if intent_payload.intent in (SemanticIntent.MATH_ALGEBRA, SemanticIntent.MATH_ARITHMETIC, SemanticIntent.MATH_CONVERSION):
            eq_target = intent_payload.equation or clean_request
            math_det = self.math_engine.evaluate_detailed(eq_target) or self.math_engine.evaluate_detailed(clean_request)
            if math_det:
                self._store_math_context(self.math_engine.last_expression, self.math_engine.last_variables)
                display_math = self.personality.format_response(str(math_det["display"]))
                speech_math = math_det.get("speech", display_math)
                return self._build_fast_response(display_math, request_id, start_time, task_mem, speak_response, speech_text=speech_math)

        # S2. Direct Knowledge Factoid / Explanation (Spoken directly, NO browser window opened)
        is_operational_task = bool(
            re.search(
                r"\b(?:organize|sort|run|build|verify|verification|create|delete|remove|move|copy|execute|install|test|compile|clone|deploy|setup|clean|fix|refactor|update|switch|bring\s+up|open|launch|start|close|exit|terminate)\b",
                clean_request,
                flags=re.IGNORECASE,
            )
        )
        if not is_operational_task and intent_payload.intent in (SemanticIntent.QA_FACTOID, SemanticIntent.QA_EXPLANATION, SemanticIntent.QA_REALTIME_LOOKUP, SemanticIntent.CHITCHAT_BANTER, SemanticIntent.CREATIVE_DRAFT):
            # Check if query references user profile, personal memories, or hardware
            has_user_memory = False
            if hasattr(self, "memory_manager"):
                try:
                    rel_mems = self.memory_manager.retrieve_relevant(clean_request, min_score=0.20)
                    has_user_memory = bool(rel_mems)
                except Exception:
                    pass

            # Conversational dialogue, chitchat, banter, and explanations must always be dynamically synthesized
            # via M.A.D.D.Y. contextual persona prompt in _handle_fast_conversation, never raw classifier direct_speech
            conv_resp = self._handle_fast_conversation(clean_request)
            if conv_resp:
                return self._build_fast_response(conv_resp, request_id, start_time, task_mem, speak_response)

            is_personal_query = bool(re.search(r"\b(?:i|my|me|mine|we|our)\b", clean_request.lower()))
            if intent_payload.direct_speech and not has_user_memory and not is_personal_query:
                formatted_qa = self.personality.format_response(intent_payload.direct_speech)
                return self._build_fast_response(formatted_qa, request_id, start_time, task_mem, speak_response)
            is_question = bool(re.search(r"^(?:who|what|where|when|why|how|which|is|are|was|were|can\s+you\s+explain|explain|define|tell\s+me\s+about)\b", clean_request.lower()))
            if is_question and hasattr(self, "model_provider") and hasattr(self.model_provider, "is_configured") and self.model_provider.is_configured():
                try:
                    from models.base import ModelMessage
                    addr = self.personality.get_address_title() or "boss"
                    qa_prompt = f"User asks: \"{clean_request}\"\nProvide a concise, direct, spoken 1-2 sentence factual answer as M.A.D.D.Y. to {addr}."
                    hist_msgs = [ModelMessage(role=m["role"], content=m["content"]) for m in getattr(self, "conversation_history", [])[-10:]]
                    sys_prompt = self.get_contextual_system_prompt(clean_request)
                    qa_res = self.model_provider.generate(qa_prompt, system_prompt=sys_prompt, messages=hist_msgs)
                    if qa_res and qa_res.content and not qa_res.content.startswith("Neural brain service"):
                        formatted_qa = self.personality.format_response(qa_res.content.strip())
                        return self._build_fast_response(formatted_qa, request_id, start_time, task_mem, speak_response)
                except Exception:
                    pass

        # S3. Media Video Search (YouTube specifically)
        if intent_payload.intent == SemanticIntent.SEARCH_MEDIA_VIDEO:
            # If query is explicitly a song / music request and user did NOT ask for youtube or video, route to Spotify!
            lower_clean = clean_request.lower()
            is_explicit_video = any(k in lower_clean for k in ["youtube", "video", "watch", "mv", "trailer", "clip", "teaser"])
            is_explicit_music = any(k in lower_clean for k in ["song", "songs", "music", "track", "tracks", "album", "audio"]) or primary_cap == CapabilityType.SPOTIFY or "spotify" in lower_clean
            if is_explicit_music and not is_explicit_video:
                spotify_resp = self._handle_direct_spotify_command(clean_request)
                if spotify_resp:
                    return self._build_fast_response(spotify_resp, request_id, start_time, task_mem, speak_response)

            if primary_cap == CapabilityType.SPOTIFY or "spotify" in clean_request.lower():
                pass
            else:
                slots = intent_payload.slots or {}
                video_q = intent_payload.search_query or slots.get("search_query")
                if not video_q:
                    video_q = re.sub(r"\b(?:search\s+youtube\s+for|search\s+for|play|watch|youtube|on\s+youtube|open|launch|go\s+to|start|for\s+me|please)\b", "", clean_request, flags=re.IGNORECASE).strip()
                video_q = (video_q or "").strip().rstrip(".?!")

                is_pure_navigational = not video_q or video_q.lower() in ("youtube", "video", "videos", "app", "open", "launch", "go to")
                if is_pure_navigational:
                    yt_url = "https://www.youtube.com"
                else:
                    encoded_q = urllib.parse.quote_plus(video_q)
                    yt_url = f"https://www.youtube.com/results?search_query={encoded_q}"

                browser_target = "chrome" if "chrome" in clean_request.lower() else ("firefox" if "firefox" in clean_request.lower() else "browser")
                try:
                    args = {"browser_name": browser_target, "url": yt_url}
                    res = self.executor.execute(ToolCall(name="browser.open", arguments=args))
                    data = res.data if isinstance(res.data, dict) else {}
                    reused = data.get("reused_tab", False) or data.get("status") == "reused_existing_tab"

                    addr = self.personality.get_address_title() or "boss"
                    if reused:
                        action_desc = "Switched to your pre-opened YouTube window." if is_pure_navigational else f"Switched to your pre-opened YouTube window searching for '{video_q}'."
                        fallback = f"Switched to your open YouTube window, {addr}."
                    else:
                        action_desc = "Opened YouTube." if is_pure_navigational else f"Opened YouTube searching for '{video_q}'."
                        fallback = f"Opening YouTube, {addr}." if is_pure_navigational else f"Searching for '{video_q}' on YouTube, {addr}."

                    resp_msg = self._generate_dynamic_action_response(clean_request, action_desc, fallback)
                    return self._build_fast_response(resp_msg, request_id, start_time, task_mem, speak_response)
                except Exception as e:
                    logger.warning(f"Error opening YouTube search: {e}")

        # S4. Review Consensus (Reddit / IMDb)
        if intent_payload.intent == SemanticIntent.SEARCH_CONSENSUS and hasattr(self, "research_engine"):
            slots = intent_payload.slots or {}
            consensus_topic = intent_payload.search_query or slots.get("search_query") or clean_request
            consensus_res = self.research_engine.research_and_synthesize(
                consensus_topic,
                model_provider=self.model_provider,
                address=self.personality.get_address_title(),
            )
            if consensus_res.get("success") and consensus_res.get("response"):
                formatted_resp = self.personality.format_response(consensus_res["response"])
                self._last_assistant_response = formatted_resp
                return self._build_fast_response(formatted_resp, request_id, start_time, task_mem, speak_response)

        # S5. Commercial Shopping with Rich Slot Extraction
        if intent_payload.intent == SemanticIntent.SEARCH_COMMERCIAL and hasattr(self, "product_search_engine"):
            slots = intent_payload.slots or {}
            if slots.get("search_query") or slots.get("product_category") or slots.get("brands") or slots.get("max_price") or slots.get("min_price") or slots.get("sort_by") or slots.get("is_branded"):
                parsed_product = self.product_search_engine.build_search_from_slots(slots, clean_request)
            else:
                parsed_product = self.product_search_engine.parse_product_search(clean_request, model_provider=self.model_provider)

            if parsed_product:
                url = parsed_product["url"]
                browser_target = parsed_product["browser"]
                try:
                    args = {"browser_name": browser_target, "url": url}
                    res = self.executor.execute(ToolCall(name="browser.open", arguments=args))
                    if not res.success:
                        if sys.platform.startswith("win"):
                            os.startfile(url)
                        else:
                            import webbrowser
                            webbrowser.open(url)

                    # Ensure browser window is brought directly to front on laptop screen
                    time.sleep(0.5)
                    try:
                        self.platform.focus_window_or_application(browser_target)
                    except Exception:
                        pass

                    # Capture screenshot of search results on laptop screen
                    shot_path = None
                    try:
                        time.sleep(0.8)
                        shot = self.computer_controller.take_screenshot()
                        if shot and shot.get("path") and os.path.exists(shot["path"]):
                            shot_path = shot["path"]
                    except Exception:
                        pass

                    addr = self.personality.get_address_title() or "boss"
                    filter_txt = f" under ₹{parsed_product.get('max_price'):,}" if parsed_product.get('max_price') else ""

                    # Save active search session state for real pagination navigation
                    self._active_product_search = {
                        "site": parsed_product["site"],
                        "product": parsed_product["product"],
                        "base_url": url,
                        "page": 1,
                        "browser": browser_target,
                        "max_price": parsed_product.get("max_price"),
                    }
                    self._last_executed_command_type = "product_search"

                    # Synthesize top recommendations dynamically via LLM
                    recs = self.product_search_engine.synthesize_recommendations(
                        product=parsed_product["product"],
                        max_price=parsed_product.get("max_price"),
                        site=parsed_product["site"],
                        model_provider=self.model_provider,
                    )

                    # Save and launch in temporary Notepad on Windows desktop
                    notepad_path = self.product_search_engine.create_notepad_recommendations(
                        recommendations=recs,
                        query_desc=f"{parsed_product['product']}{filter_txt}",
                        site=parsed_product["site"],
                    )

                    resp_msg = f"🛒 Opened {parsed_product['site']} on your laptop searching for '{parsed_product['product']}'{filter_txt}, {addr}. Top recommended picks and direct links have also been opened in Notepad for you."
                    extra = {
                        "path": shot_path,
                        "recommendations": recs,
                        "notepad_file": notepad_path,
                        "url": url,
                        "site": parsed_product["site"],
                    }
                    return self._build_fast_response(resp_msg, request_id, start_time, task_mem, speak_response, extra_data=extra)
                except Exception as e:
                    logger.warning(f"Error opening commercial search: {e}")

        # S6. Direct Navigational Site & Section Launching
        if intent_payload.intent == SemanticIntent.SEARCH_NAVIGATIONAL:
            if primary_cap == CapabilityType.SPOTIFY or "spotify" in clean_request.lower():
                pass
            elif primary_cap == CapabilityType.APPLICATION or re.search(r"\b(?:switch\s+to|bring\s+up)\b", clean_request.lower()):
                pass
            else:
                slots = intent_payload.slots or {}
                target_site = (intent_payload.target_site or slots.get("target_site") or "").lower().strip()
                section = (slots.get("destination_section") or "").lower().strip()
                direct_url = slots.get("direct_url")

                dest_url = None
                site_name = target_site.capitalize() or "Website"

                # Check direct destination URL from intent resolution first
                if direct_url:
                    dest_url = direct_url
                    site_name = slots.get("entity") or target_site.capitalize() or "Requested Page"
                elif target_site and section and (target_site, section) in KNOWN_SITE_SECTIONS:
                    site_name, dest_url = KNOWN_SITE_SECTIONS[(target_site, section)]
                elif target_site in KNOWN_WEBSITES:
                    site_name, dest_url = KNOWN_WEBSITES[target_site]
                elif "." in target_site:
                    dest_url = f"https://{target_site}" if not target_site.startswith("http") else target_site

                if not dest_url:
                    browser_resp = self._handle_direct_browser_launch(clean_request)
                    if browser_resp:
                        return self._build_fast_response(browser_resp, request_id, start_time, task_mem, speak_response)

                if dest_url:
                    browser_target = "chrome" if "chrome" in clean_request.lower() else "browser"
                    try:
                        args = {"browser_name": browser_target, "url": dest_url}
                        res = self.executor.execute(ToolCall(name="browser.open", arguments=args))
                        if not res.success:
                            if sys.platform.startswith("win"):
                                os.startfile(dest_url)
                            else:
                                import webbrowser
                                webbrowser.open(dest_url)
                        addr = self.personality.get_address_title() or "boss"
                        action_desc = f"Navigated to {site_name}."
                        fallback = f"Opening {site_name}, {addr}."
                        resp_msg = self._generate_dynamic_action_response(clean_request, action_desc, fallback)
                        return self._build_fast_response(resp_msg, request_id, start_time, task_mem, speak_response)
                    except Exception as e:
                        logger.warning(f"Error opening navigational target: {e}")

        # S7. Technical Documentation, GitHub, arXiv Search
        if intent_payload.intent == SemanticIntent.SEARCH_TECHNICAL:
            slots = intent_payload.slots or {}
            tech_q = intent_payload.search_query or slots.get("search_query") or clean_request
            target_site = (intent_payload.target_site or slots.get("target_site") or "").lower()

            if "github" in target_site or "github" in clean_request.lower():
                url = f"https://github.com/search?q={urllib.parse.quote_plus(tech_q)}"
                site_name = "GitHub"
            elif "arxiv" in target_site or "arxiv" in clean_request.lower():
                url = f"https://arxiv.org/search/?query={urllib.parse.quote_plus(tech_q)}&searchtype=all"
                site_name = "arXiv"
            elif "stackoverflow" in target_site or "stack overflow" in clean_request.lower():
                url = f"https://stackoverflow.com/search?q={urllib.parse.quote_plus(tech_q)}"
                site_name = "Stack Overflow"
            else:
                url = f"https://www.google.com/search?q={urllib.parse.quote_plus(tech_q + ' documentation')}"
                site_name = "Documentation"

            browser_target = "chrome" if "chrome" in clean_request.lower() else "browser"
            try:
                args = {"browser_name": browser_target, "url": url}
                res = self.executor.execute(ToolCall(name="browser.open", arguments=args))
                if not res.success:
                    if sys.platform.startswith("win"):
                        os.startfile(url)
                    else:
                        import webbrowser
                        webbrowser.open(url)
                addr = self.personality.get_address_title() or "boss"
                action_desc = f"Opened {site_name} searching for '{tech_q}'."
                fallback = f"Searching for '{tech_q}' on {site_name}, {addr}."
                resp_msg = self._generate_dynamic_action_response(clean_request, action_desc, fallback)
                return self._build_fast_response(resp_msg, request_id, start_time, task_mem, speak_response)
            except Exception as e:
                logger.warning(f"Error opening technical search: {e}")

        # S8. Local Search & Maps
        if intent_payload.intent == SemanticIntent.SEARCH_LOCAL:
            slots = intent_payload.slots or {}
            local_q = intent_payload.search_query or slots.get("search_query") or clean_request
            url = f"https://www.google.com/maps/search/{urllib.parse.quote_plus(local_q)}"
            browser_target = "chrome" if "chrome" in clean_request.lower() else "browser"
            try:
                args = {"browser_name": browser_target, "url": url}
                res = self.executor.execute(ToolCall(name="browser.open", arguments=args))
                if not res.success:
                    if sys.platform.startswith("win"):
                        os.startfile(url)
                    else:
                        import webbrowser
                        webbrowser.open(url)
                addr = self.personality.get_address_title() or "boss"
                action_desc = f"Opened Google Maps searching for '{local_q}'."
                fallback = f"Searching for '{local_q}' on Google Maps, {addr}."
                resp_msg = self._generate_dynamic_action_response(clean_request, action_desc, fallback)
                return self._build_fast_response(resp_msg, request_id, start_time, task_mem, speak_response)
            except Exception as e:
                logger.warning(f"Error opening local search: {e}")

        # S9. OS Colloquial & Multilingual System Commands
        if intent_payload.intent == SemanticIntent.OS_COLLOQUIAL:
            slots = intent_payload.slots or {}
            os_action = intent_payload.os_action or slots.get("os_action")
            os_val = slots.get("os_value")
            os_target = slots.get("os_target")
            addr = self.personality.get_address_title() or "boss"

            # Execute slot-driven action if identified
            if os_action == "set_volume" and os_val is not None:
                try:
                    v = int(os_val)
                    self.platform.set_volume(max(0, min(100, v)))
                    action_desc = f"Adjusted volume to {v}%."
                    fallback = f"Volume adjusted to {v}%, {addr}."
                    resp_msg = self._generate_dynamic_action_response(clean_request, action_desc, fallback)
                    return self._build_fast_response(resp_msg, request_id, start_time, task_mem, speak_response)
                except Exception:
                    pass
            elif os_action == "mute":
                self.platform.set_mute(True)
                action_desc = "Muted system audio."
                fallback = f"Audio muted, {addr}."
                resp_msg = self._generate_dynamic_action_response(clean_request, action_desc, fallback)
                return self._build_fast_response(resp_msg, request_id, start_time, task_mem, speak_response)
            elif os_action == "unmute":
                self.platform.set_mute(False)
                action_desc = "Unmuted system audio."
                fallback = f"Audio unmuted, {addr}."
                resp_msg = self._generate_dynamic_action_response(clean_request, action_desc, fallback)
                return self._build_fast_response(resp_msg, request_id, start_time, task_mem, speak_response)
            elif os_action == "increase_volume":
                curr = self.platform.get_volume() if hasattr(self.platform, "get_volume") else 50
                new_v = min(100, (curr or 50) + 15)
                self.platform.set_volume(new_v)
                action_desc = f"Increased volume to {new_v}%."
                fallback = f"Increased volume to {new_v}%, {addr}."
                resp_msg = self._generate_dynamic_action_response(clean_request, action_desc, fallback)
                return self._build_fast_response(resp_msg, request_id, start_time, task_mem, speak_response)
            elif os_action == "decrease_volume":
                curr = self.platform.get_volume() if hasattr(self.platform, "get_volume") else 50
                new_v = max(0, (curr or 50) - 15)
                self.platform.set_volume(new_v)
                action_desc = f"Reduced volume to {new_v}%."
                fallback = f"Reduced volume to {new_v}%, {addr}."
                resp_msg = self._generate_dynamic_action_response(clean_request, action_desc, fallback)
                return self._build_fast_response(resp_msg, request_id, start_time, task_mem, speak_response)
            elif os_action == "set_brightness" and os_val is not None:
                try:
                    b = int(os_val)
                    self.platform.set_brightness(max(0, min(100, b)))
                    action_desc = f"Set screen brightness to {b}%."
                    fallback = f"Screen brightness set to {b}%, {addr}."
                    resp_msg = self._generate_dynamic_action_response(clean_request, action_desc, fallback)
                    return self._build_fast_response(resp_msg, request_id, start_time, task_mem, speak_response)
                except Exception:
                    pass
            elif os_action in ("dim_brightness", "decrease_brightness"):
                curr = self.platform.get_brightness() if hasattr(self.platform, "get_brightness") else 50
                new_b = max(10, (curr or 50) - 20)
                self.platform.set_brightness(new_b)
                action_desc = f"Reduced screen brightness to {new_b}%."
                fallback = f"Reduced screen brightness to {new_b}%, {addr}."
                resp_msg = self._generate_dynamic_action_response(clean_request, action_desc, fallback)
                return self._build_fast_response(resp_msg, request_id, start_time, task_mem, speak_response)
            elif os_action == "increase_brightness":
                curr = self.platform.get_brightness() if hasattr(self.platform, "get_brightness") else 50
                new_b = min(100, (curr or 50) + 20)
                self.platform.set_brightness(new_b)
                action_desc = f"Increased screen brightness to {new_b}%."
                fallback = f"Increased screen brightness to {new_b}%, {addr}."
                resp_msg = self._generate_dynamic_action_response(clean_request, action_desc, fallback)
                return self._build_fast_response(resp_msg, request_id, start_time, task_mem, speak_response)
            elif os_action == "close_app" and os_target:
                if hasattr(self, "process_controller"):
                    res = self.process_controller.close_application(os_target)
                    if res.get("closed"):
                        action_desc = f"Closed application {os_target}."
                        fallback = f"Closed {os_target}, {addr}."
                        resp_msg = self._generate_dynamic_action_response(clean_request, action_desc, fallback)
                        return self._build_fast_response(resp_msg, request_id, start_time, task_mem, speak_response)
            elif os_action == "launch_app" and os_target:
                if os_target.lower() == "spotify" or primary_cap == CapabilityType.SPOTIFY or "spotify" in clean_request.lower():
                    s_resp = self._handle_direct_spotify_command(clean_request)
                    if s_resp:
                        return self._build_fast_response(s_resp, request_id, start_time, task_mem, speak_response)
                if os_target.lower() in ("chrome", "browser", "firefox", "edge"):
                    b_resp = self._handle_direct_browser_launch(clean_request)
                    if b_resp:
                        return self._build_fast_response(b_resp, request_id, start_time, task_mem, speak_response)
                a_resp = self._handle_direct_app_launch(clean_request)
                if a_resp:
                    return self._build_fast_response(a_resp, request_id, start_time, task_mem, speak_response)

            vol_resp = self._handle_volume_command(clean_request)
            if vol_resp:
                return self._build_fast_response(vol_resp, request_id, start_time, task_mem, speak_response)
            bright_resp = self._handle_brightness_command(clean_request)
            if bright_resp:
                return self._build_fast_response(bright_resp, request_id, start_time, task_mem, speak_response)
            if intent_payload.direct_speech:
                formatted_os = self.personality.format_response(intent_payload.direct_speech)
                return self._build_fast_response(formatted_os, request_id, start_time, task_mem, speak_response)

        # 6. PATH C — Direct Semantic Web Search & Browser Control
        is_launch_cmd = bool(
            re.search(r"\b(?:open|launch|go\s+to|navigate\s+to|switch\s+to)\b", clean_request.lower())
            and not re.search(r"\b(?:search|look\s+up|find)\b", clean_request.lower())
        )
        if primary_cap == CapabilityType.BROWSER and is_launch_cmd:
            browser_resp = self._handle_direct_browser_launch(clean_request)
            if browser_resp:
                return self._build_fast_response(browser_resp, request_id, start_time, task_mem, speak_response)

        is_search = bool(
            re.search(r"\b(?:google|search\s+(?:for\s+)?|look\s+up\s+on\s+google|search\s+on)\b", clean_request.lower())
            or (
                re.search(r"\b(?:picture|pictures|photo|photos|image|images|wallpaper|wallpapers)\b", clean_request.lower())
                and re.search(r"\b(?:search|find|show|look\s+up|download|get|view)\b", clean_request.lower())
                and not re.search(r"\b(?:camera|webcam|capture|take|click|snap)\b", clean_request.lower())
            )
            or re.search(r"\b(?:search|look\s+up|find)\b.+?\b(?:on|in|from)\s+[a-zA-Z0-9_\-\. ]+", clean_request.lower())
            or (hasattr(self, "product_search_engine") and self.product_search_engine.is_product_query(clean_request))
        )
        if is_search or primary_cap == CapabilityType.BROWSER:
            search_resp = self._handle_direct_web_search(clean_request, intent_payload=intent_payload)
            if search_resp:
                if isinstance(search_resp, dict):
                    resp_text = search_resp.get("response", "")
                    extra = {k: v for k, v in search_resp.items() if k != "response"}
                    return self._build_fast_response(resp_text, request_id, start_time, task_mem, speak_response, extra_data=extra)
                return self._build_fast_response(search_resp, request_id, start_time, task_mem, speak_response)

        if primary_cap == CapabilityType.BROWSER and re.search(r"\b(open|launch|go to|navigate to|view|show)\b", clean_request.lower()):
            browser_resp = self._handle_direct_browser_launch(clean_request)
            if browser_resp:
                return self._build_fast_response(browser_resp, request_id, start_time, task_mem, speak_response)

        # 7. PATH D — Direct Semantic Spotify Controls
        if primary_cap == CapabilityType.SPOTIFY:
            spotify_resp = self._handle_direct_spotify_command(clean_request)
            if spotify_resp:
                return self._build_fast_response(spotify_resp, request_id, start_time, task_mem, speak_response)

        # 8. PATH E — Direct Application Control
        if primary_cap == CapabilityType.APPLICATION or re.search(r"\b(?:open|launch|start|switch\s+to|bring\s+up|close|terminate|exit|kill)\s+[a-zA-Z0-9_\-\. ]+", clean_request.lower()):
            app_resp = self._handle_direct_app_launch(clean_request)
            if app_resp:
                return self._build_fast_response(app_resp, request_id, start_time, task_mem, speak_response)

        # 9. PATH F — Fast-Path Conversational & System Queries
        if primary_cap == CapabilityType.CONVERSATION:
            conv_resp = self._handle_fast_conversation(clean_request)
            if conv_resp:
                return self._build_fast_response(conv_resp, request_id, start_time, task_mem, speak_response)

        # 10. PATH G — Supervisor Task Graph Execution
        try:
            supervisor_result = self.supervisor.run(clean_request)
            elapsed = round(time.time() - start_time, 2)
            
            if supervisor_result.status != "SUCCESS":
                fail_reason = supervisor_result.final_response or "required tasks failed or were blocked"
                formatted_response = self.personality.format_task_failure(fail_reason)
            else:
                raw_response = supervisor_result.final_response or "Task execution complete."
                formatted_response = self.personality.format_response(raw_response)

            response_data = {
                "success": supervisor_result.status == "SUCCESS",
                "request_id": request_id,
                "status": supervisor_result.status,
                "user_request": clean_request,
                "response": formatted_response,
                "completed_tasks": len(supervisor_result.completed_tasks),
                "total_tasks": supervisor_result.total_tasks,
                "elapsed_seconds": elapsed,
                "memory_dir": self.memory_dir,
            }

            task_mem.save_result(supervisor_result.status, formatted_response)
            if hasattr(self, "conversation_history") and clean_request:
                self.conversation_history.append({"role": "user", "content": clean_request})
                self.conversation_history.append({"role": "assistant", "content": formatted_response})
                if len(self.conversation_history) > 16:
                    self.conversation_history = self.conversation_history[-16:]
                self._log_session_turn(request_id, clean_request, formatted_response)

            if speak_response and formatted_response:
                self._speak_output(formatted_response, speak_response)

            return response_data

        except Exception as err:
            elapsed = round(time.time() - start_time, 2)
            friendly_msg = self.personality.format_task_failure(str(err))

            error_response = {
                "success": False,
                "request_id": request_id,
                "status": "FAILED",
                "user_request": clean_request,
                "response": friendly_msg,
                "error": str(err),
                "elapsed_seconds": elapsed,
            }
            task_mem.save_result("FAILED", error_response)
            if speak_response and friendly_msg:
                self._speak_output(friendly_msg, speak_response)
            return error_response

    def decompose_compound_request(self, text: str) -> List[str]:
        """
        Dynamically decompose a multi-action compound natural language request into
        discrete, executable sub-commands without hardcoding.
        Preserves single-intent actions (like 'open youtube and search for X' or product queries with multi-specs).
        """
        clean = text.strip()
        # 0. Guard: Strip leading conversational salutation prefix ("Hi maddy,", "hey maddy,", "hello maddy,")
        unprefixed = re.sub(r"^(?:(?:hi|hello|hey)\s+)?(?:maddy\s*[,:]?\s*)", "", clean, flags=re.IGNORECASE).strip()
        if not unprefixed:
            return [clean]
        clean = unprefixed
        lower = clean.lower()

        # 1. Guard: Single-action site search like "open youtube and search for lofi", "can you open wikipedia and search for quantum mechanics"
        if re.search(r"\b(?:open|launch|go\s+to)\s+[a-zA-Z0-9_\-\. ]+?\s+and\s+(?:search|find|look)", lower):
            return [clean]

        # 2. Guard: Product filter with multiple specs like "find phone with 8gb ram and 128gb storage under 20k"
        if any(w in lower for w in ["under", "gb", "ram", "storage", "budget", "price"]) and any(w in lower for w in ["phone", "laptop", "pendrive", "monitor", "tv", "watch"]):
            if not any(k in lower for k in ["play", "volume", "brightness", "spotify", "wallpaper", "calculate", "mute"]):
                return [clean]

        # 3. Protect quoted strings (e.g. songs, movies, file names with 'and' inside)
        quotes: List[str] = []
        def save_quote(m: Any) -> str:
            quotes.append(m.group(0))
            return f"__QUOTE_{len(quotes)-1}__"

        protected = re.sub(r"([\"']).*?\1", save_quote, clean)

        # 4. Action verbs / intent initiators
        action_verbs = r"(?:play|set|open|launch|start|run|close|terminate|exit|search|find|look\s+up|dim|increase|decrease|reduce|raise|lower|turn\s+up|turn\s+down|mute|unmute|save|write|check|show|tell\s+me|calculate|what|how|click|double\s+click|right\s+click|scroll|type)"
        delims = rf"(?:;\s*|,\s*(?:and\s+)?(?:also|then|next)\s+|\b(?:and\s+also|and\s+then)\b|,\s*and\s+|\band\s+(?={action_verbs}\b)|,\s*(?={action_verbs}\b))"

        raw_parts = [p.strip().strip("?.,") for p in re.split(delims, protected, flags=re.IGNORECASE) if p.strip()]

        # 5. Restore quotes
        final_parts = []
        for part in raw_parts:
            for idx, q in enumerate(quotes):
                part = part.replace(f"__QUOTE_{idx}__", q)
            if part and part.lower() not in ("hi", "hello", "hey", "maddy", "hi maddy", "hey maddy", "hello maddy"):
                final_parts.append(part)

        return final_parts if len(final_parts) > 1 else [text.strip()]

    def _execute_compound_request(
        self,
        sub_commands: List[str],
        original_request: str,
        request_id: str,
        start_time: float,
        task_mem: TaskMemory,
        speak_response: bool,
    ) -> Dict[str, Any]:
        """
        Execute decomposed sub-commands sequentially, combining individual action outputs
        into a unified, natural composite response.
        """
        sub_results = []
        individual_responses = []
        all_success = True

        for cmd in sub_commands:
            res = self.execute(cmd, speak_response=False, _is_subcommand=True)
            sub_results.append(res)
            resp = (res.get("response") or "").strip()
            if resp:
                individual_responses.append(resp)
        all_success = all(res.get("success", False) for res in sub_results)
        status = "SUCCESS" if all_success else "FAILED"

        addr = self.personality.get_address_title() or "boss"
        if not all_success:
            failed_sr = next((sr for sr in sub_results if not sr.get("success")), None)
            composite_response = failed_sr.get("response") if failed_sr else f"One or more actions failed, {addr}."
        else:
            composite_response = self._synthesize_compound_response(individual_responses, addr)

        elapsed = round(time.time() - start_time, 2)
        response_data = {
            "success": all_success,
            "request_id": request_id,
            "status": status,
            "user_request": original_request,
            "sub_commands": sub_commands,
            "sub_results": sub_results,
            "response": composite_response,
            "completed_tasks": sum(1 for res in sub_results if res.get("success")),
            "total_tasks": len(sub_commands),
            "elapsed_seconds": elapsed,
            "memory_dir": self.memory_dir,
        }

        task_mem.save_result(status, composite_response)
        self._last_assistant_response = composite_response

        if speak_response and composite_response:
            self._speak_output(composite_response, speak_response)

        return response_data

    def _synthesize_compound_response(self, responses: List[str], address: str) -> str:
        """
        Synthesize multiple sub-command responses into a single natural, non-repetitive response.
        """
        if not responses:
            return f"All commands completed, {address}."
        if len(responses) == 1:
            return responses[0]

        cleaned = []
        for r in responses:
            s = re.sub(rf"(?:,\s*{re.escape(address)})?[.?!]*$", "", r.strip(), flags=re.IGNORECASE).strip()
            if s:
                cleaned.append(s)

        if not cleaned:
            return f"All actions completed, {address}."

        return ". ".join(cleaned) + f", {address}."

    def _handle_brightness_command(self, text: str) -> Optional[str]:
        """Handle natural language display brightness controls."""
        lower = text.lower().strip()
        if not re.search(r"\b(?:brightness|dim\s+(?:the\s+)?(?:screen|display)|display\s+light|screen\s+light)\b", lower):
            return None

        addr = self.personality.get_address_title()

        # Maximum / full / peak brightness
        if re.search(r"\b(?:maximum|max|full|peak|highest|100%)\b", lower):
            success = self.platform.set_brightness(100) if hasattr(self.platform, "set_brightness") else False
            if success:
                return f"Screen brightness set to maximum (100%), {addr}."
            return f"Attempted to maximize screen brightness, but hardware display control was unavailable, {addr}."

        # Minimum / lowest / dimmest brightness
        if re.search(r"\b(?:minimum|min|lowest|dimmest|0%)\b", lower):
            success = self.platform.set_brightness(10) if hasattr(self.platform, "set_brightness") else False
            if success:
                return f"Screen brightness set to minimum (10%), {addr}."
            return f"Attempted to minimize screen brightness, but hardware display control was unavailable, {addr}."

        # Explicit percentage / value: e.g. "to 50%", "50 percent", "to 80"
        m_pct = re.search(r"\b(?:to|at|=)\s*(\d{1,3})\s*(?:%|percent)?\b", lower)
        if not m_pct:
            m_pct = re.search(r"(\d{1,3})\s*(?:%|percent)", lower)

        if m_pct:
            val = int(m_pct.group(1))
            val = max(0, min(100, val))
            success = self.platform.set_brightness(val) if hasattr(self.platform, "set_brightness") else False
            if success:
                return f"Screen brightness adjusted to {val}%, {addr}."
            return f"Attempted to adjust brightness to {val}%, but display hardware control was unavailable, {addr}."

        # Relative adjustments: decrease / lower / reduce / dim
        if any(w in lower for w in ["reduce", "decrease", "lower", "dim", "turn down"]):
            curr = self.platform.get_brightness() if hasattr(self.platform, "get_brightness") else None
            new_val = max(10, (curr - 20) if curr is not None else 50)
            success = self.platform.set_brightness(new_val) if hasattr(self.platform, "set_brightness") else False
            if success:
                return f"Reduced screen brightness to {new_val}%, {addr}."

        # Relative adjustments: increase / raise / boost / brighter
        if any(w in lower for w in ["increase", "raise", "boost", "brighter", "higher", "turn up", "more"]):
            curr = self.platform.get_brightness() if hasattr(self.platform, "get_brightness") else None
            new_val = min(100, (curr + 20) if curr is not None else 80)
            success = self.platform.set_brightness(new_val) if hasattr(self.platform, "set_brightness") else False
            if success:
                return f"Increased screen brightness to {new_val}%, {addr}."

        return None

    def _handle_volume_command(self, text: str) -> Optional[str]:
        """Handle natural language master audio volume and mute controls."""
        lower = text.lower().strip()
        is_vol_match = bool(
            re.search(r"\b(?:volume|sound|audio|mute|unmute)\b", lower)
            or any(w in lower for w in ["சத்தம்", "வால்யூம்", "ஒலி", "மியூட்", "ശബ്ദം", "വോളിയം", "മ്യൂട്ട്"])
        )
        if not is_vol_match:
            return None

        addr = self.personality.get_address_title()

        # Check mute / unmute
        if re.search(r"\bunmute\b", lower) or "அன்மியூட்" in lower or "ശബ്ദം പുനഃസ്ഥാപിക്കുക" in lower:
            success = self.platform.set_mute(False) if hasattr(self.platform, "set_mute") else False
            if success:
                return f"Audio unmuted, {addr}."
            return f"Attempted to unmute audio, but hardware sound control was unavailable, {addr}."

        if (
            (re.search(r"\bmute\b", lower) or any(w in lower for w in ["மியூட்", "மயூട്ട്", "அமைதி", "നിശ്ശബ്ദ"]))
            and not re.search(r"\b(?:unmute|don't|not)\b", lower)
        ):
            success = self.platform.set_mute(True) if hasattr(self.platform, "set_mute") else False
            if success:
                return f"Audio muted, {addr}."
            return f"Attempted to mute audio, but hardware sound control was unavailable, {addr}."

        # Maximum / full / peak volume
        if re.search(r"\b(?:maximum|max|full|peak|highest|100%)\b", lower) or "முழு சத்தம்" in lower:
            success = self.platform.set_volume(100) if hasattr(self.platform, "set_volume") else False
            if success:
                return f"Volume set to maximum (100%), {addr}."
            return f"Attempted to maximize volume, but hardware audio control was unavailable, {addr}."

        # Minimum / lowest / zero volume
        if re.search(r"\b(?:minimum|min|lowest|zero|0%)\b", lower) or "பூஜ்ஜியம்" in lower:
            success = self.platform.set_volume(0) if hasattr(self.platform, "set_volume") else False
            if success:
                return f"Volume set to minimum (0%), {addr}."
            return f"Attempted to minimize volume, but hardware audio control was unavailable, {addr}."

        # Explicit percentage / value: e.g. "volume to 50%", "set volume to 80"
        m_pct = re.search(r"\b(?:to|at|=)\s*(\d{1,3})\s*(?:%|percent)?\b", lower)
        if not m_pct:
            m_pct = re.search(r"(\d{1,3})\s*(?:%|percent)", lower)

        if m_pct:
            val = int(m_pct.group(1))
            val = max(0, min(100, val))
            success = self.platform.set_volume(val) if hasattr(self.platform, "set_volume") else False
            if success:
                return f"Volume adjusted to {val}%, {addr}."
            return f"Attempted to adjust volume to {val}%, but hardware audio control was unavailable, {addr}."

        # Relative adjustments: decrease / lower / reduce / turn down
        if (
            any(w in lower for w in ["reduce", "decrease", "lower", "down", "quieter", "soft", "softer"])
            or any(w in lower for w in ["குறை", "இறக்கு", "குறைக்கவும்", "കുറയ്ക്കൂ", "കുറയ്ക്കുക"])
        ):
            curr = self.platform.get_volume() if hasattr(self.platform, "get_volume") else None
            new_val = max(0, (curr - 15) if curr is not None else 30)
            success = self.platform.set_volume(new_val) if hasattr(self.platform, "set_volume") else False
            if success:
                if any(w in lower for w in ["குறை", "இறக்கு"]):
                    return f"வால்யூம் {new_val}% ஆக குறைக்கப்பட்டது, {addr}."
                elif any(w in lower for w in ["കുറയ്ക്കൂ", "കുറയ്ക്കുക"]):
                    return f"ശബ്ദം {new_val}% ആയി കുറച്ചു, {addr}."
                return f"Reduced volume to {new_val}%, {addr}."

        # Relative adjustments: increase / raise / boost / up / louder
        if (
            any(w in lower for w in ["increase", "raise", "boost", "up", "louder", "more", "higher"])
            or any(w in lower for w in ["கூட்டு", "அதிகரி", "உயர்த்து", "கூட்டவும்", "കൂട്ടൂ", "കൂട്ടുക"])
        ):
            curr = self.platform.get_volume() if hasattr(self.platform, "get_volume") else None
            new_val = min(100, (curr + 15) if curr is not None else 70)
            success = self.platform.set_volume(new_val) if hasattr(self.platform, "set_volume") else False
            if success:
                if any(w in lower for w in ["கூட்டு", "அதிகரி", "உயர்த்து"]):
                    return f"வால்யூம் {new_val}% ஆக அதிகரிக்கப்பட்டது, {addr}."
                elif any(w in lower for w in ["കൂട്ടൂ", "കൂട്ടുക"]):
                    return f"ശബ്ദം {new_val}% ആയി വർദ്ധിപ്പിച്ചു, {addr}."
                return f"Increased volume to {new_val}%, {addr}."

        return None

    def _handle_vision_cursor_command(self, text: str, request_id: str, task_mem: TaskMemory) -> Optional[str]:
        """
        Handle natural language GUI automation, cursor actions, and screen analysis.
        Supports:
        - "click on [element]" / "click [element]"
        - "double click on [element]"
        - "right click on [element]"
        - "scroll down" / "scroll up"
        - "type [text] into [element]"
        - "what's on my screen" / "analyze screen" / "read error"
        """
        lower = text.lower().strip().rstrip(".?!")
        addr = self.personality.get_address_title()

        # 1. Scrolling commands
        if re.search(r"\b(?:scroll\s+down|page\s+down)\b", lower):
            self.computer_controller.scroll(direction="down", amount=4)
            return self.personality.format_response(f"Scrolled down, {addr}.")
        if re.search(r"\b(?:scroll\s+up|page\s+up)\b", lower):
            self.computer_controller.scroll(direction="up", amount=4)
            return self.personality.format_response(f"Scrolled up, {addr}.")

        # 2. Type into element: "type 'hello world' into search bar"
        type_match = re.search(r"\btype\s+['\"]?(.+?)['\"]?\s+(?:into|in|on)\s+(?:the\s+)?(.+?)$", lower)
        if type_match:
            text_to_type = type_match.group(1).strip()
            target_el = type_match.group(2).strip()
            self.computer_controller.click_element(target_el, vision_engine=self.vision_engine)
            time.sleep(0.15)
            self.computer_controller.type_text(text_to_type, press_enter=False)
            return self.personality.format_response(f"Typed \"{text_to_type}\" into {target_el}, {addr}.")

        # 3. Double-click commands: "double click on [element]"
        dbl_match = re.search(r"\bdouble\s+click\s+(?:on\s+)?(?:the\s+)?(.+?)$", lower)
        if dbl_match:
            target_el = dbl_match.group(1).strip()
            res = self.computer_controller.click_element(target_el, vision_engine=self.vision_engine, clicks=2)
            return self.personality.format_response(f"Double-clicked on {target_el}, {addr}.")

        # 4. Right-click commands: "right click on [element]"
        r_match = re.search(r"\bright\s+click\s+(?:on\s+)?(?:the\s+)?(.+?)$", lower)
        if r_match:
            target_el = r_match.group(1).strip()
            res = self.computer_controller.click_element(target_el, vision_engine=self.vision_engine, button="right")
            return self.personality.format_response(f"Right-clicked on {target_el}, {addr}.")

        # 5. Click commands: "click on [element]" / "click the search box"
        click_match = re.search(r"^(?:please\s+)?(?:can\s+you\s+)?click\s+(?:on\s+)?(?:the\s+)?(.+?)$", lower)
        if click_match:
            target_el = click_match.group(1).strip()
            res = self.computer_controller.click_element(target_el, vision_engine=self.vision_engine)
            return self.personality.format_response(f"Clicked on {target_el}, {addr}.")

        # 6. Screen content & visual reasoning: "what's on my screen", "analyze my screen", "read the error on my screen"
        if re.search(r"\b(?:what(?:'s|\s+is)\s+(?:on\s+)?(?:my\s+)?screen|analyze\s+(?:my\s+)?screen|read\s+(?:the\s+)?error|look\s+at\s+(?:the\s+|my\s+)?screen|describe\s+(?:the\s+|my\s+)?screen|what\s+do\s+you\s+see)\b", lower):
            temp_dir = tempfile.gettempdir()
            shot_path = os.path.join(temp_dir, f"maddy_screen_content_{request_id}.png")
            self.computer_controller.take_screenshot(shot_path)
            content_analysis = self.vision_engine.analyze_screen_content(
                shot_path,
                model_provider=self.model_provider,
                user_query=text,
            )
            summary = content_analysis.get("summary", "Screen analyzed.")
            return self.personality.format_response(summary)

        return None

    def _speak_timer_alert(self, message: str):
        """Proactively notify the user when a timer expires."""
        try:
            print(f"\n🔔 [TIMER ALERT]: {message}\n")
            if hasattr(self, "voice_engine") and self.voice_engine:
                self.voice_engine.speak(message)
        except Exception:
            pass

    def _speak_micro_feedback(self, text: str):
        """Verbal micro-feedback cue for long-running operations (Phase 6)."""
        try:
            if hasattr(self, "voice_engine") and self.voice_engine:
                self.voice_engine.speak(text)
        except Exception:
            pass

    def _handle_timer_command(self, text: str) -> Optional[str]:
        """Handle setting, checking, and cancelling countdown timers and alarms."""
        lower = text.lower().strip().rstrip(".?!")
        addr = self.personality.get_address_title()

        # 1. Set Timer / Countdown
        if re.search(r"\b(?:set\s+(?:a\s+|the\s+)?timer|timer\s+for|countdown|set\s+alarm|timer\s+of|start\s+(?:a\s+)?timer)\b", lower):
            from tools.timer import parse_duration_to_seconds, format_duration
            duration_secs = parse_duration_to_seconds(text)
            if duration_secs:
                entry = self.timer_manager.set_timer(duration_secs)
                desc = format_duration(duration_secs)
                return self.personality.format_response(f"Timer set for {desc}, {addr}. I will alert you when it expires.")
            else:
                return self.personality.format_response(f"How long should I set the timer for, {addr}?")

        # 2. Check Timer Status / Time Left
        if re.search(r"\b(?:timer\s+status|how\s+much\s+time\s+left|time\s+left\s+on\s+timer|check\s+(?:the\s+)?timer)\b", lower):
            from tools.timer import format_duration
            active = self.timer_manager.get_active_timers()
            if not active:
                return self.personality.format_response(f"You don't have any active timers running, {addr}.")
            if len(active) == 1:
                t = active[0]
                rem = format_duration(t.remaining_seconds)
                return self.personality.format_response(f"You have {rem} remaining on your {t.label} timer, {addr}.")
            else:
                descriptions = [f"{format_duration(t.remaining_seconds)} on your {t.label}" for t in active]
                return self.personality.format_response(f"Active timers: {', '.join(descriptions)}, {addr}.")

        # 3. Cancel Timer
        if re.search(r"\b(?:cancel\s+(?:all\s+|the\s+)?timers?|stop\s+(?:all\s+|the\s+)?timers?|clear\s+(?:all\s+|the\s+)?timers?)\b", lower):
            cancelled = self.timer_manager.cancel_all_timers()
            if cancelled > 0:
                return self.personality.format_response(f"Cancelled {cancelled} active timer{'s' if cancelled > 1 else ''}, {addr}.")
            return self.personality.format_response(f"No active timers to cancel, {addr}.")

        return None

    def _handle_counting_command(self, text: str) -> Optional[str]:
        """Handle counting requests like 'count from 1 to 100', 'count to 20', 'count down from 10 to 1'."""
        lower = text.lower().strip().rstrip(".?!")

        # 1. "count down from X to Y" or "count down from X"
        cd_match = re.search(r"\bcount\s+down\s+(?:from\s+)?(\d+)(?:\s+(?:to|till)\s+(\d+))?\b", lower)
        if cd_match:
            start = int(cd_match.group(1))
            end = int(cd_match.group(2)) if cd_match.group(2) else 1
            if start < end:
                start, end = end, start
            if (start - end) > 150:
                end = start - 150
            return ", ".join(str(i) for i in range(start, end - 1, -1)) + "."

        # 2. "count from X to Y"
        c_match = re.search(r"\bcount\s+(?:from\s+)?(\d+)\s+(?:to|till|through)\s+(\d+)\b", lower)
        if c_match:
            start = int(c_match.group(1))
            end = int(c_match.group(2))
            if start > end:
                if (start - end) > 150:
                    end = start - 150
                return ", ".join(str(i) for i in range(start, end - 1, -1)) + "."
            else:
                if (end - start) > 150:
                    end = start + 150
                return ", ".join(str(i) for i in range(start, end + 1)) + "."

        # 3. "count to X" or "count up to X"
        to_match = re.search(r"\bcount\s+(?:up\s+)?to\s+(\d+)\b", lower)
        if to_match:
            end = int(to_match.group(1))
            end = min(end, 150)
            return ", ".join(str(i) for i in range(1, end + 1)) + "."

        return None

    def _handle_camera_command(self, text: str) -> Optional[str]:
        """Handle hardware camera and webcam snapshot / application commands."""
        lower = text.lower().strip()
        is_camera = bool(
            re.search(r"\b(?:camera|webcam)\b", lower)
            or any(w in lower for w in ["கேமரா", "ക്യാമറ"])
        )
        is_photo = bool(
            re.search(r"\b(?:photo|picture|snap|snapshot|selfie|capture\s+(?:a\s+)?photo|take\s+(?:a\s+)?photo|take\s+(?:a\s+)?picture|click\s+(?:a\s+)?photo)\b", lower)
            or any(w in lower for w in ["புகைப்படம்", "போட்டோ", "ഫോട്ടോ"])
        )

        if not (is_camera or (is_photo and any(w in lower for w in ["capture", "take", "click", "snap", "open", "shoot", "send", "share", "forward", "dispatch", "எடு", "എടുക്കൂ"]))):
            return None

        addr = self.personality.get_address_title()
        is_referencing_last = any(w in lower for w in ["last photo", "last picture", "previous photo", "that picture", "that photo", "the photo", "the picture"])
        should_capture = bool(
            any(w in lower for w in ["capture", "take", "click", "snap", "shoot", "எடு", "എടുക്കൂ"])
            and not (is_referencing_last and not any(w in lower for w in ["take a", "click a", "capture a", "snap a"]))
        )
        should_open_app = bool(
            any(w in lower for w in ["open", "launch", "start", "show", "திற", "തുറക്കൂ"])
        )

        # 1. User wants to capture a photo
        if should_capture:
            success, file_path, msg = self.camera_engine.capture_photo()
            if success:
                fname = os.path.basename(file_path)

                # Check if user requested sending the captured photo to Telegram
                send_to_telegram = bool(
                    "telegram" in lower
                    or re.search(r"\b(?:send|forward|share|text|message|dispatch)\b.*\btelegram\b", lower)
                )
                if send_to_telegram:
                    from integrations.messaging import TelegramClient
                    tg = TelegramClient()
                    tg_res = tg.send_photo(file_path, caption=f"Snapshot captured by M.A.D.D.Y. ({fname})")
                    if tg_res.get("success"):
                        return self.personality.format_response(
                            f"Photo captured and sent directly to your Telegram, {addr}."
                        )
                    else:
                        err = tg_res.get("error", "Unknown error")
                        return self.personality.format_response(
                            f"Photo captured as {fname}, but failed to send to Telegram: {err}, {addr}."
                        )

                self.camera_engine.open_captured_photo(file_path)
                return self.personality.format_response(f"Photo captured and displayed on your screen as {fname}, {addr}.")
            else:
                if should_open_app or "open" in lower:
                    self.camera_engine.launch_camera_app()
                    return self.personality.format_response(f"Could not capture snapshot directly ({msg}), but I've opened the Camera app for you, {addr}.")
                return self.personality.format_response(f"Unable to capture photo: {msg}, {addr}.")

        # 1b. User wants to send the last captured photo to Telegram
        if ("last" in lower or "that" in lower or "previous" in lower) and any(p in lower for p in ["photo", "picture", "image", "snap"]) and "telegram" in lower:
            last_pic = getattr(self.camera_engine, "last_captured_photo", None)
            if last_pic and os.path.exists(last_pic):
                from integrations.messaging import TelegramClient
                tg = TelegramClient()
                fname = os.path.basename(last_pic)
                tg_res = tg.send_photo(last_pic, caption=f"Photo sent by M.A.D.D.Y. ({fname})")
                if tg_res.get("success"):
                    return self.personality.format_response(f"I've sent your last captured photo to Telegram, {addr}.")
                else:
                    return self.personality.format_response(f"Failed to send photo to Telegram: {tg_res.get('error')}, {addr}.")
            return self.personality.format_response(f"No recent photo found in memory to send, {addr}.")

        # 2. User only wants to open the camera app
        if should_open_app and is_camera:
            opened = self.camera_engine.launch_camera_app()
            if opened:
                return self.personality.format_response(f"Opening camera now, {addr}.")
            return self.personality.format_response(f"Attempted to open camera, but the system camera application could not be launched, {addr}.")

        return None

    def _persist_location_fact(self, fact_text: str, scope: Optional[MemoryScope] = None) -> None:
        """Location is intentionally not stored as a fixed user preference or memory. Use live geolocation instead."""
        if hasattr(self, "user_preferences"):
            for key in ("user.location", "user.city", "user.home_city"):
                self.user_preferences.delete_preference(key)
        return None

    def _resolve_memory_scope(self, text: str) -> Tuple[MemoryScope, str]:
        """Parse natural-language memory commands into a target scope and a clean fact string."""
        lower = text.lower().strip()

        scope_patterns = [
            (MemoryScope.SESSION, [
                r"for\s+this\s+session",
                r"for\s+the\s+session",
                r"this\s+session",
            ]),
            (MemoryScope.PROJECT, [
                r"for\s+this\s+project",
                r"for\s+the\s+project",
                r"for\s+this\s+workspace",
                r"for\s+the\s+workspace",
                r"project\s+memory",
            ]),
            (MemoryScope.GLOBAL, [
                r"forever",
                r"for\s+long\s+term",
                r"permanently",
                r"for\s+later",
            ]),
        ]

        for scope, patterns in scope_patterns:
            for pattern in patterns:
                match = re.search(pattern, lower)
                if match:
                    remainder = re.split(pattern, text, maxsplit=1, flags=re.IGNORECASE)[-1].strip()
                    remainder = remainder.strip(" :;,-")
                    remainder = remainder.rstrip("!?.")
                    if remainder:
                        return scope, remainder
                    break

        remember_match = re.search(
            r"^(?:please\s+)?remember\s+(?:that\s+|to\s+always\s+|i\s+prefer\s+)?(.+)$",
            lower,
            flags=re.IGNORECASE,
        )
        if remember_match:
            return MemoryScope.GLOBAL, remember_match.group(1).strip().rstrip("!?.")

        if re.search(r"\b(?:user|project|workspace)\b", lower):
            return MemoryScope.PROJECT, text.strip().rstrip("!?.")

        return MemoryScope.GLOBAL, text.strip().rstrip("!?.")

    def _extract_favorite_song_from_text(self, text: str) -> Optional[str]:
        """Extract a favorite song from memory text or a direct preference statement."""
        cleaned = text.strip().rstrip(".?!")
        patterns = [
            r"(.+?)\s+(?:is|was)\s+(?:my|your|our)\s+(?:favorite|favourite)\s+(?:song|music)\b",
            r"(?:my|your|our)\s+(?:favorite|favourite)\s+(?:song|music)\s+(?:is|was)\s*(.+?)\s*(?:$|[.;!?])",
            r"(?:remember(?:\s+that)?\s*[,;:]?\s*)(?:.*?,\s*)?(.+?)\s+(?:is|was)\s+(?:my|your|our)\s+(?:favorite|favourite)\s+(?:song|music)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip().strip(" \"'.,;?!") if match.lastindex else ""
                if not candidate:
                    candidate = cleaned
                if "," in candidate:
                    candidate = candidate.split(",")[-1].strip()
                if candidate and len(candidate) > 1:
                    return candidate

        if re.search(r"\b(?:favorite|favourite)\s+(?:song|music)\b", cleaned, flags=re.IGNORECASE):
            segment = re.split(r"\b(?:favorite|favourite)\s+(?:song|music)\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
            candidate = segment.strip().rstrip(" ,;:!?")
            if "," in candidate:
                candidate = candidate.split(",")[-1].strip()
            if candidate:
                return candidate

        return None

    def _normalize_collection_key(self, raw: str) -> str:
        """Convert a natural-language collection name into a stable key, e.g. 'favorite songs' -> 'favorite_song'."""
        text = str(raw or "").lower()
        text = text.replace("favourite", "favorite")
        text = re.sub(r"\b(?:my|the|all|current|list\s+of|list|items?|entries?|things?)\b", " ", text)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        tokens = [tok for tok in text.split() if tok]

        if "favorite" in tokens and "song" in tokens:
            return "favorite_song"
        if "favorite" in tokens and "music" in tokens:
            return "favorite_song"
        if "favorite" in tokens and "songs" in tokens:
            return "favorite_song"

        filtered = []
        for tok in tokens:
            if tok in {"of", "the", "my", "all", "current", "list", "item", "entry", "thing"}:
                continue
            if tok.endswith("ies") and len(tok) > 4:
                tok = tok[:-3] + "y"
            elif tok.endswith("s") and len(tok) > 3 and not tok.endswith("ss"):
                tok = tok[:-1]
            filtered.append(tok)
        if filtered and filtered[0] == "favorite":
            filtered = ["favorite"] + filtered[1:]
        return "_".join(filtered) if filtered else "memory_list"

    def _collection_key_from_text(self, text: str) -> Optional[str]:
        """Infer a collection key from natural-language list commands and queries."""
        lower = text.lower().strip().rstrip("?!.")
        if not lower:
            return None

        patterns = [
            r"(?:add|save|remember|store|record)\s+.+?\s+to\s+(?:my\s+)?(?:list\s+of\s+)?(.+)",
            r"(?:what|show|list|get|give)\s+(?:me\s+)?(?:the\s+)?(?:list\s+of\s+)?(?:my\s+)?(.+)",
            r"(?:remove|delete)\s+(?:the\s+)?(?:\d+(?:st|nd|rd|th)?|first|second|third|last)\s+(?:song|item|entry|thing)\s+from\s+(?:my\s+)?(?:list\s+of\s+)?(.+)",
            r"(?:my\s+)?(?:favorite|favourite)\s+(?:song|songs|music)\s+(?:list|collection)",
        ]
        for pattern in patterns:
            match = re.search(pattern, lower, flags=re.IGNORECASE)
            if match:
                label = match.group(1).strip()
                if label and not re.fullmatch(r"(?:the|a|an|to|from|and|or)\b", label, flags=re.IGNORECASE):
                    return self._normalize_collection_key(label)

        # direct phrase like 'favorite songs' or 'songs i love'
        for phrase in ["favorite songs", "favourite songs", "favorite song list", "favourite song list", "songs i love", "song list"]:
            if phrase in lower:
                return self._normalize_collection_key(phrase)

        return None

    def _list_key_candidates(self, target: str) -> List[str]:
        """Return likely preference keys for a collection name, including normalized variants."""
        key = self._normalize_collection_key(target)
        candidates = {key, key + "s", key.rstrip("s"), key + "_list", key + "_items"}
        lower_target = str(target or "").lower().replace("favourite", "favorite")
        if "favorite" in lower_target and ("song" in lower_target or "music" in lower_target):
            candidates.add("favorite_song")
        if "favorite" in lower_target and "playlist" in lower_target:
            candidates.add("favorite_playlist")
        if hasattr(self, "user_preferences"):
            prefs = self.user_preferences.get_all_preferences()
            for existing in prefs:
                existing_key = self._normalize_collection_key(existing.replace("_", " "))
                if existing_key == key:
                    candidates.add(existing)
        return sorted({c for c in candidates if c})

    def _normalize_favorite_song_name(self, raw: str) -> str:
        """Normalize favorite song text to a stable user-facing title."""
        song = str(raw or "").strip().strip("\"'.,;?! ")
        if not song:
            return ""
        song = re.sub(r"^(?:my|our|your|the)\s+(?:favorite|favourite)\s+(?:song|music)\s+(?:is|was)\s*", "", song, flags=re.IGNORECASE)
        song = re.sub(r"^(?:fav(?:orite)?\s+song|favourite\s+song|favorite\s+music|favourite\s+music)\s*(?:is|was)?\s*", "", song, flags=re.IGNORECASE)
        song = re.sub(r"\s+", " ", song).strip(" \"'.,;?!")
        if "," in song:
            song = song.split(",")[-1].strip()
        return song

    def _get_saved_favorite_song(self) -> Optional[str]:
        """Check stored favorite-song preferences and long-term memories, returning the newest saved title if present."""
        if hasattr(self, "user_preferences"):
            for key in self._list_key_candidates("favorite song"):
                favorites = self.user_preferences.get_preference_list(key, [])
                if favorites:
                    return favorites[-1]

        if not hasattr(self, "memory_manager"):
            return None
        for term in ["favorite song", "favourite song", "favorite music", "favourite music"]:
            try:
                records = self.memory_manager.store.search(query=term, limit=20)
            except Exception:
                records = []
            for rec in records:
                song = self._extract_favorite_song_from_text(rec.content)
                if song:
                    return song
        return None

    def _handle_collection_memory_command(self, text: str) -> Optional[str]:
        """Handle dynamic list-like memories such as favorite songs, playlists, or any user-defined collection."""
        if not hasattr(self, "user_preferences"):
            return None

        lower = text.lower().strip().rstrip("?!.")
        if not lower:
            return None

        # Memory-lookup and personal-fact commands are handled by dedicated handlers and must not be mistaken for collection list commands.
        if re.search(r"\b(?:what\s+do\s+you\s+remember|what\s+do\s+you\s+know\s+about\s+me|show\s+your\s+memory|list\s+(?:your\s+)?memories|why\s+do\s+you\s+(?:remember|know)|what\s+city\s+am\s+i\s+in|where\s+am\s+i|current\s+location|my\s+location|what\s+is\s+my\s+location)\b", lower):
            return None
        if re.match(r"^(?:please\s+)?(?:remember|save|store|record)\s+(?:this|that|forever|for\s+this\s+session|for\s+the\s+session|for\s+this\s+project|for\s+the\s+project)\b", lower):
            return None
        if re.match(r"^(?:please\s+)?(?:remember|save|store|record)\s+(?:my|our|your)\s+(?:favorite|favourite)\s+(?:song|music)\s+is\b", lower):
            return None
        if re.match(r"^(?:please\s+)?(?:remember|save|store|record)\s+(?:i\s+live\s+in|i\s+am\s+from|i\'m\s+from|i\s+am\s+based\s+in|i\'m\s+based\s+in)\b", lower):
            return None

        addr = self.personality.get_address_title()

        add_match = re.search(
            r"^(?:please\s+)?(?:add|save|remember|store|record)\s+(.+?)\s+(?:to|in)\s+(?:my\s+)?(?:list\s+of\s+)?(.+?)\s*$",
            text,
            flags=re.IGNORECASE,
        )
        if add_match:
            item = add_match.group(1).strip().strip("\"'.,;:!? ")
            label = add_match.group(2).strip()
            if item and label:
                key = self._list_key_candidates(label)[0] if self._list_key_candidates(label) else self._normalize_collection_key(label)
                self.user_preferences.add_preference_list(key, item)
                return f"Added '{item}' to your {label} list, {addr}."

        list_match = re.search(
            r"^(?:please\s+)?(?:what|show|list|get|give)\s+(?:me\s+)?(?:the\s+)?(?:list\s+of\s+)?(?:my\s+)?(.+?)\s*$",
            text,
            flags=re.IGNORECASE,
        )
        if list_match:
            label = list_match.group(1).strip()
            if label:
                key = self._list_key_candidates(label)[0] if self._list_key_candidates(label) else self._normalize_collection_key(label)
                values = self.user_preferences.get_preference_list(key, [])
                if not values:
                    # Only claim empty list if user explicitly requested a list or collection
                    if re.search(r"\b(?:list|items|collection|favorites|favourites|todos|tasks|groceries|reminders|watchlist)\b", text, re.IGNORECASE):
                        return f"I don't have any saved entries for {label} yet, {addr}."
                    return None
                return f"Your {label} are: " + ", ".join(f"{i + 1}. {v}" for i, v in enumerate(values)) + f", {addr}."

        remove_match = re.search(
            r"^(?:please\s+)?(?:remove|delete)\s+(?:the\s+)?(?:(\d+)(?:st|nd|rd|th)?|first|second|third|last)\s+(?:song|item|entry|thing|value)\s+(?:from|out\s+of)\s+(?:my\s+)?(?:list\s+of\s+)?(.+?)\s*$",
            lower,
            flags=re.IGNORECASE,
        )
        if remove_match:
            token = remove_match.group(1) or remove_match.group(0)
            label = remove_match.group(2).strip()
            key = self._list_key_candidates(label)[0] if self._list_key_candidates(label) else self._normalize_collection_key(label)
            values = self.user_preferences.get_preference_list(key, [])
            if not values:
                return f"I don't have any entries in your {label} list to remove, {addr}."
            idx = self._parse_list_index(token, len(values))
            if idx is None or idx < 0 or idx >= len(values):
                return f"I couldn't find that position in your {label} list, {addr}."
            removed = values.pop(idx)
            self.user_preferences.set_preference(key, values)
            return f"Removed '{removed}' from your {label} list, {addr}."

        return None

    def _parse_list_index(self, token: str, length: int) -> Optional[int]:
        """Parse '1st', 'second', or 'last' into a zero-based list index."""
        token = str(token or "").strip().lower()
        if not token:
            return None
        rank_map = {"first": 0, "second": 1, "third": 2, "last": max(length - 1, 0)}
        if token in rank_map:
            return rank_map[token]
        m = re.fullmatch(r"(\d+)(?:st|nd|rd|th)?", token)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < length:
                return idx
        return None

    def _handle_memory_command(self, text: str) -> Optional[str]:
        """Handle explicit natural memory commands (remember, what do you remember, why do you remember, forget)."""
        lower = text.lower().strip()
        addr = self.personality.get_address_title()

        # 1. "What do you remember about me?" / "What's in your memory?"
        if re.search(r"\b(?:what\s+do\s+you\s+remember|what\s+do\s+you\s+know\s+about\s+me|show\s+your\s+memory|list\s+(?:your\s+)?memories)\b", lower):
            records = self.memory_manager.what_do_you_remember()
            if not records:
                return self.personality.format_response(f"I don't have any saved persistent memories yet, {addr}.")
            lines = [f"- {r['content']}" for r in records[:8]]
            summary = "\n".join(lines)
            return self.personality.format_response(f"Here is what I remember, {addr}:\n{summary}")

        # 2. "Why do you remember [X]?"
        why_match = re.search(r"\bwhy\s+do\s+you\s+(?:remember|know)\s+(?:that\s+)?(.+)", lower)
        if why_match:
            topic = why_match.group(1).strip().rstrip("?.")
            prov = self.memory_manager.why_do_you_remember(topic)
            if prov:
                src_txt = prov.get("source_text", "")
                src_type = prov.get("source_type", "user interaction")
                return self.personality.format_response(f"I recorded that based on your {src_type}: \"{src_txt}\", {addr}.")
            return self.personality.format_response(f"I don't have a specific provenance record for '{topic}', {addr}.")

        # 3. "Forget [X]" / "Forget that [X]" / "Remove my favorite song from memory"
        forget_match = re.search(r"\b(?:forget|remove|delete|clear)\s+(?:that\s+|my\s+|our\s+|the\s+)?(?:memory\s+of\s+|memory\s+for\s+)?(.+)", lower)
        if forget_match:
            topic = forget_match.group(1).strip().rstrip("?.")
            if topic in ("it", "this", "about it"):
                return self.personality.format_response(f"Understood, ignoring that, {addr}.")
            if re.search(r"\b(?:favorite|favourite)\s+(?:song|music)\b", topic, flags=re.IGNORECASE):
                matches = self.memory_manager.store.search(query="favorite song", limit=20)
                for rec in matches:
                    if re.search(r"\b(?:favorite|favourite)\s+(?:song|music)\b", rec.content, flags=re.IGNORECASE):
                        self.memory_manager.forget(rec.memory_id)
                        return self.personality.format_response(f"I removed your favorite song from memory, {addr}.")
                return self.personality.format_response(f"I didn't find a saved favorite song memory to remove, {addr}.")
            success = self.memory_manager.forget(topic)
            if success:
                return self.personality.format_response(f"I've removed that from memory, {addr}.")
            return self.personality.format_response(f"I couldn't find any memory matching '{topic}' to delete, {addr}.")

        # 4. "Remember that [X]" / "Remember: [X]" / "Actually my favorite song is [X]"
        favorite_song_match = re.search(r"^(?:please\s+)?(?:remember\s+(?:that\s+)?|save\s+)?(?:my|our|your|actually\s+my|actually\s+our|actually\s+your)\s+(?:favorite|favourite)\s+(?:song|music)\s+is\s+(.+)$", lower)
        if favorite_song_match:
            candidate = self._normalize_favorite_song_name(favorite_song_match.group(1))
            if candidate:
                if hasattr(self, "user_preferences"):
                    self.user_preferences.add_preference_list("favorite_song", candidate)
                fact = f"Favorite song - {candidate}"
                scope, fact = self._resolve_memory_scope(text)
                for rec in self.memory_manager.store.search(query="favorite song", limit=20):
                    if re.search(r"\b(?:favorite|favourite)\s+(?:song|music)\b", rec.content, flags=re.IGNORECASE):
                        self.memory_manager.forget(rec.memory_id)
                rec = self.memory_manager.remember(
                    content=fact,
                    memory_type=MemoryType.USER_PREFERENCE,
                    scope=scope,
                    source_type=SourceType.USER_MESSAGE,
                    source_text=text,
                    created_by=CreatedBy.USER,
                    confidence=1.0,
                    importance=0.8,
                    metadata={"kind": "favorite_song", "favorite_song": candidate},
                )
                return self.personality.format_response(f"Noted, {addr}. Favorite song saved as {candidate}.")

        override_match = re.search(
            r"(?:my\s+location\s+is|my\s+current\s+city\s+is|i\s+am\s+in|i\'m\s+in|i\s+am\s+currently\s+in|current\s+city\s+is|i\s+live\s+in|i\s+am\s+from|i\'m\s+from|i\'m\s+based\s+in|i\s+am\s+based\s+in)\s+(.+)$",
            lower,
            flags=re.IGNORECASE,
        )
        if override_match:
            city = override_match.group(1).strip().rstrip(" .?!")
            from tools.weather import set_location_override
            loc = set_location_override(city)
            if loc:
                return self.personality.format_response(
                    f"Understood, {addr}. I’ll use {loc['city']} as your current location."
                )
            return self.personality.format_response(
                f"I heard Coimbatore, {addr}, but I couldn't resolve that city. Please say a clearer city name."
            )

        location_match = re.search(r"(?:i\s+live\s+in|i\s+am\s+from|i\'m\s+from|i\'m\s+based\s+in|i\s+am\s+based\s+in)\s+(.+)$", lower, flags=re.IGNORECASE)
        if location_match:
            city = location_match.group(1).strip().rstrip(" .?!")
            self._persist_location_fact(f"User lives in {city}")
            return self.personality.format_response(
                f"Noted, {addr}. I won't store that as a fixed home location; I will use live geolocation for your current city."
            )

        remember_match = re.search(r"^(?:please\s+)?remember\s+(?:that\s+|to\s+always\s+|i\s+prefer\s+)?(.+)$", lower)
        if remember_match:
            fact = remember_match.group(1).strip().rstrip("!.")
            if fact:
                song_match = re.search(r"(.+?)\s+(?:is|was)\s+(?:my|our|your)\s+(?:favorite|favourite)\s+(?:song|music)\b", fact, flags=re.IGNORECASE)
                if song_match:
                    candidate = song_match.group(1).strip().strip(" \"'.,;?!")
                    if "," in candidate:
                        candidate = candidate.split(",")[-1].strip()
                    if candidate:
                        fact = f"My favorite song is {candidate}"
                scope, fact = self._resolve_memory_scope(text)
                if re.search(r"\b(?:i\s+live\s+in|i\'m\s+based\s+in|i\s+am\s+based\s+in|i\s+am\s+from|i\'m\s+from)\b", fact, flags=re.IGNORECASE):
                    self._persist_location_fact(fact, scope)
                rec = self.memory_manager.remember(
                    content=fact,
                    memory_type=MemoryType.USER_PREFERENCE if "prefer" in lower or "like" in lower or "favorite" in lower or "favourite" in lower else MemoryType.SEMANTIC_FACT,
                    scope=scope,
                    source_type=SourceType.USER_MESSAGE,
                    source_text=text,
                    created_by=CreatedBy.USER,
                    confidence=1.0,
                    importance=0.8,
                )
                return self.personality.format_response(f"Noted, {addr}. I'll remember that {rec.content}.")

        # 5. "next time I say X, you should say Y"
        rule_match = re.search(r"next\s+time\s+i\s+say\s+(.+?)(?:,|\s+you\s+should\s+(?:say|respond\s+with))\s+(.+)$", text, flags=re.IGNORECASE)
        if rule_match:
            trigger = rule_match.group(1).strip().rstrip("?!.,;: ")
            response = rule_match.group(2).strip().rstrip("?!.,;: ")
            saved = self._remember_response_rule(trigger, response)
            if saved:
                return self.personality.format_response(saved)

        return None

    def _handle_hardware_status_command(self, text: str) -> Optional[str]:
        """Handle natural language GPU, VRAM, and system hardware status inquiries."""
        lower = text.lower().strip()
        gpu_patterns = [
            r"\b(?:gpu|vram|graphics\s+card|rtx\s*3050|video\s+memory|cuda|nvidia)\b",
            r"\b(?:gpu\s+temp|gpu\s+temperature|vram\s+usage|gpu\s+usage|gpu\s+load)\b",
            r"how('s|\s+is)\s+my\s+gpu(?:\s+holding\s+up)?",
            r"check\s+(?:the\s+)?gpu",
        ]
        system_patterns = [
            r"\b(?:system\s+status|hardware\s+status|system\s+metrics|hardware\s+metrics|system\s+load|cpu\s+usage|ram\s+usage)\b",
            r"\b(?:how('s|\s+is)\s+my\s+(?:system|hardware|laptop|pc))\b",
            r"resource\s+usage",
        ]

        is_gpu = any(re.search(p, lower) for p in gpu_patterns)
        is_sys = any(re.search(p, lower) for p in system_patterns)

        if not (is_gpu or is_sys):
            return None

        target = "gpu" if (is_gpu and not is_sys) else "all"
        res = self.executor.execute(ToolCall(name="hardware.status", arguments={"target": target}))
        data = res.data if res.data is not None else getattr(res, "result", None)
        if res.success and isinstance(data, dict):
            summary = data.get("spoken_summary") or data.get("message")
            if summary:
                return self.personality.format_response(summary)
        return "Unable to read hardware metrics at this moment."

    @staticmethod
    def _extract_arxiv_topic(text: str) -> str:
        """Extract clean research topic from natural spoken query, stripping fluff and qualifiers."""
        cleaned = text.strip()
        # 1. Strip polite preambles
        cleaned = re.sub(
            r"^(?:please\s+|can\s+you\s+(?:please\s+)?|could\s+you\s+(?:please\s+)?|will\s+you\s+|would\s+you\s+|maddy\s+|hey\s+maddy\s+)*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        # 2. Strip request verbs + optional arXiv mention
        cleaned = re.sub(
            r"^(?:check(?:\s+(?:for|on))?|find|search(?:\s+(?:for|on))?|get|look\s+up|show\s+me|open(?:\s+up)?|fetch|give\s+me|bring\s+up|display)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        # 3. Strip optional arXiv mention if early: e.g. "on arxiv", "in arxiv", "arxiv"
        cleaned = re.sub(r"^(?:on\s+arxiv|in\s+arxiv|arxiv)\s+(?:for\s+)?", "", cleaned, flags=re.IGNORECASE).strip()
        # 4. Strip quantities & paper descriptors: e.g. "top 5 recent research papers on", "some scientific papers about"
        cleaned = re.sub(
            r"^(?:(?:the|a|an|any|some)\s+)?(?:top\s+\d+\s+|\d+\s+|recent\s+|latest\s+|new\s+)*(?:research\s+papers?|academic\s+papers?|scientific\s+papers?|papers?|arxiv(?:\s+papers?)?|literature|digest)\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        # 5. Strip prepositions: on, about, regarding, for, in
        cleaned = re.sub(r"^(?:on|about|regarding|for|in)\s+", "", cleaned, flags=re.IGNORECASE).strip()
        # 6. Strip topic indicators: e.g. "the topic of", "the topic", "the subject of", "the field of"
        cleaned = re.sub(
            r"^(?:the\s+)?(?:topic|subject|area|field|domain)\s+(?:of\s+|about\s+|on\s+)?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        # 7. Strip leading articles: "the", "a", "an"
        cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE).strip()
        # 8. Strip trailing temporal / polite fluff
        cleaned = re.sub(r"\b(?:this\s+week|today|recently|for\s+me|please)\b.*$", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = cleaned.strip("?.,\"' ")
        return cleaned or "concrete defect segmentation"

    def _handle_arxiv_command(self, text: str) -> Optional[str]:
        """Handle natural language queries for arXiv research papers and academic digests."""
        lower = text.lower().strip()
        arxiv_patterns = [
            r"\barxiv\b",
            r"\b(?:research\s+papers?|academic\s+papers?|scientific\s+papers?)\b",
            r"\b(?:new\s+papers?|recent\s+papers?)\s+(?:on|about)\b",
            r"\b(?:papers?\s+on|paper\s+about)\b",
            r"\b(?:digest\s+on|literature\s+on)\b",
            r"\bnewspapers\s+on\b",  # handles Whisper speech recognition slip for "new papers on"
        ]
        if not any(re.search(p, lower) for p in arxiv_patterns):
            return None

        # 1. Parse requested paper count (e.g. "top 5", "3 papers")
        count_match = re.search(r"\b(?:top\s+|first\s+)?(\d+)\s+(?:research\s+|academic\s+|scientific\s+)?papers?\b", lower)
        max_results = int(count_match.group(1)) if count_match else 5
        max_results = min(max(1, max_results), 10)

        # 2. Extract clean research topic from conversational fluff
        clean_topic = self._extract_arxiv_topic(text)

        # 3. Retrieve papers from arXiv API
        res = self.executor.execute(ToolCall(name="arxiv.search", arguments={"query": clean_topic, "max_results": max_results}))
        data = res.data if res.data is not None else getattr(res, "result", None)

        if res.success and isinstance(data, dict):
            summary = data.get("spoken_summary") or data.get("message")

            # 4. Simultaneously open Chrome with the research papers website on screen
            import urllib.parse
            arxiv_url = f"https://arxiv.org/search/?query={urllib.parse.quote_plus(clean_topic)}&searchtype=all&source=header"
            try:
                self.executor.execute(
                    ToolCall(
                        name="browser.open",
                        arguments={"url": arxiv_url, "browser_name": "chrome"},
                    )
                )
            except Exception:
                try:
                    import webbrowser
                    webbrowser.open(arxiv_url)
                except Exception:
                    pass

            if summary:
                return self.personality.format_response(summary)

        return f"I couldn't find any relevant arXiv papers on {clean_topic} right now."

    def _handle_process_monitor_command(self, text: str) -> Optional[str]:
        """
        Handle asynchronous process / task completion monitoring requests.
        Identifies active compute / training / Python workloads and monitors them
        until completion, dispatching Telegram/WhatsApp notifications upon termination.
        """
        lower = text.lower().strip()

        # Check if the query is asking for completion monitoring
        monitor_patterns = [
            r"\b(?:process|task|training|job|script)\s+running\s+now.*(?:when|after|once).*(?:notify|alert|tell|send|give|message)\b",
            r"\b(?:when|after|once)\s+(?:it|the\s+process|the\s+task|training|the\s+training|the\s+job|this\s+process|this|python)\s+(?:completes?|finishes?|is\s+done|terminates?|ends?)\b",
            r"\b(?:give\s+me|send\s+(?:me\s+)?a?)\s+notification\s+(?:on|via)\s+(?:telegram|whatsapp)\s+when\b",
            r"\b(?:notify|alert|tell)\s+(?:my\s+phone|me|telegram|whatsapp)?.*(?:when|after|once)\s+(?:it|the\s+process|the\s+task|training|the\s+job|this|python)\s+(?:completes?|finishes?|is\s+done|terminates?|ends?)\b",
            r"\bmonitor\s+(?:the\s+)?(?:process|task|training|job|pid|it)?.*(?:when|after|once)?.*(?:notify|alert|telegram|whatsapp)\b",
            r"\blet\s+me\s+know\s+(?:on\s+telegram|on\s+whatsapp)?\s*(?:when|after)\s+(?:it|the\s+process|training)\s+(?:completes?|finishes?|ends?|is\s+done)\b",
            r"\bwhen\s+.*(?:completes?|finishes?|is\s+done)\s+(?:give\s+me|send\s+(?:me\s+)?a?|notify|alert).*(?:telegram|whatsapp|phone)\b",
            r"\bwhen\s+it\s+completes\b",
        ]

        if not any(re.search(p, lower) for p in monitor_patterns):
            return None

        # 1. Target Notification Channel
        channel = "whatsapp" if ("whatsapp" in lower and "telegram" not in lower) else "telegram"

        # 2. Extract PID if explicitly specified (e.g. "PID 12345", "process 12345")
        hint = None
        pid_match = re.search(r"\b(?:pid|process)\s*[:=]?\s*(\d+)\b", lower)
        if pid_match:
            hint = pid_match.group(1).strip()
        else:
            # Check for script or process name hint (e.g. "when train_fpn.py completes", "when yolo finishes")
            name_match = re.search(r"\bwhen\s+([a-zA-Z0-9_\-\.]+)\s+(?:completes|finishes|is\s+done)\b", lower)
            if name_match:
                candidate = name_match.group(1).strip()
                if candidate not in ("it", "the", "this", "my"):
                    hint = candidate

        # 3. Locate target process using psutil inspector
        from hardware.gpu_monitor import find_active_user_process
        target_info = find_active_user_process(hint=hint, exclude_pids=[os.getpid()])

        channel_name = "Telegram" if channel == "telegram" else "WhatsApp"
        addr = self.personality.get_address_title() or "boss"

        if target_info:
            pid = target_info["pid"]
            label = target_info.get("label") or target_info.get("name") or f"Process {pid}"
            self.training_watcher.start_monitoring_pid(
                pid=pid,
                label=label,
                notify_channel=channel,
                cmdline=target_info.get("cmdline", ""),
            )
            return self.personality.format_response(
                f"Understood, {addr}. I have identified active process '{label}' (PID {pid}). "
                f"I am monitoring it in the background and will notify your {channel_name} "
                f"the moment it completes."
            )
        else:
            return self.personality.format_response(
                f"I scanned your active system processes, {addr}, but couldn't detect another running "
                f"Python process or task right now. If your process is running under a specific PID, "
                f"say 'monitor PID [number] on {channel_name}' or launch your script and I will track it for you."
            )

    def _handle_messaging_command(self, text: str) -> Optional[str]:
        """
        Handle natural language messaging commands for WhatsApp, Telegram, and Contact Book
        using zero-pattern Semantic LLM Extraction.
        Dynamically extracts intent, contact names (including multi-word names), message bodies,
        and resolves speech-recognition variations effortlessly.
        """
        lower = text.lower().strip()

        # Guard: If text is an event/completion trigger or monitoring request, let process monitor handle it!
        if re.search(r"\b(?:when|after|once)\s+.*(?:completes?|finishes?|is\s+done|terminates?|ends?)\b", lower):
            return None
        if re.search(r"\b(?:process|task|training|job|script)\s+(?:is\s+)?running\s+now\b", lower):
            return None

        import datetime
        now_dt = datetime.datetime.now()
        now_str = now_dt.strftime("%I:%M %p, %A, %B %d, %Y")

        # Check if the query is asking about messaging or contacts
        msg_markers = [
            "whatsapp", "telegram", "contact", "contacts", "text", "message",
            "send a message", "send message", "send a text", "send text",
            "send a heart", "send heart", "dm ", "ping ", "saying "
        ]
        is_messaging = any(m in lower for m in msg_markers) or "next," in lower or "next " in lower or lower.startswith("text") or lower.startswith("send")
        if not is_messaging:
            return None

        from integrations.messaging.contacts_manager import contacts_manager
        saved_contacts_dict = contacts_manager.load_contacts()
        known_contacts_names = list(saved_contacts_dict.keys())

        # 1. Use LLM Semantic Extractor (Groq / Cloud / Local)
        parsed_action = None
        parsed_target = None
        parsed_message = None
        parsed_phone = None
        parsed_delay = 0
        parsed_scheduled_disp = None

        if hasattr(self, "model_provider") and self.model_provider:
            try:
                contacts_json = json.dumps(known_contacts_names)
                sys_prompt = (
                    "You are M.A.D.D.Y.'s natural voice messaging parser.\n"
                    "Analyze the user's voice command and extract messaging intent, entities, and any scheduled time.\n"
                    f"Current system time: {now_str}.\n"
                    "User voice transcription may contain speech recognition slips (e.g. 'Next' for 'Text', 'heart' for '❤️', phonetically misheard names like 'covering rods' for 'Kavin Raj').\n"
                    "If the target name sounds like or phonetically resembles one of the known saved contacts, resolve target to that contact name.\n"
                    "If the user specifies a delay or a future target time (e.g., 'in 10 minutes', 'at 6:00 PM', 'at 12 AM', 'tomorrow at 9 AM'), calculate delay_seconds as integer seconds from current system time.\n"
                    "If no time or delay is specified, delay_seconds MUST be 0.\n\n"
                    f"Known saved contacts: {contacts_json}\n\n"
                    "Output strictly valid JSON matching this schema:\n"
                    "{\n"
                    '  "action": "send_whatsapp" | "send_telegram" | "save_contact" | "list_contacts" | "delete_contact" | "none",\n'
                    '  "target": "<person name or phone number or null>",\n'
                    '  "message": "<message text or null>",\n'
                    '  "phone": "<phone number or null>",\n'
                    '  "delay_seconds": <integer seconds from now to wait before sending, or 0 if immediate>,\n'
                    '  "scheduled_time_display": "<e.g. at 6:00 PM or in 10 minutes or null if immediate>"\n'
                    "}\n"
                    "Do NOT output markdown fences or backticks. Only raw JSON."
                )
                res = self.model_provider.generate(prompt=f"User request: {text}", system_prompt=sys_prompt)
                if res and res.content:
                    c = res.content.strip()
                    c = re.sub(r"^```(?:json)?\s*", "", c)
                    c = re.sub(r"\s*```$", "", c).strip()
                    data = json.loads(c)
                    parsed_action = data.get("action")
                    parsed_target = data.get("target")
                    parsed_message = data.get("message")
                    parsed_phone = data.get("phone")
                    parsed_delay = int(data.get("delay_seconds") or 0)
                    parsed_scheduled_disp = data.get("scheduled_time_display")
            except Exception as e:
                logger.warning(f"Semantic messaging extraction error: {e}")

        # Fallback if model_provider was unavailable or returned none
        if not parsed_action or parsed_action == "none":
            if "list" in lower and "contact" in lower:
                parsed_action = "list_contacts"
            elif "delete" in lower and "contact" in lower:
                parsed_action = "delete_contact"
                m = re.search(r"delete\s+contact\s+(.+)", text, flags=re.IGNORECASE)
                parsed_target = m.group(1).strip() if m else None
            elif ("save" in lower or "add" in lower) and "contact" in lower:
                parsed_action = "save_contact"
            elif "telegram" in lower:
                parsed_action = "send_telegram"
                parsed_message = text
            elif "whatsapp" in lower or "text" in lower:
                parsed_action = "send_whatsapp"
                parsed_target = "myself"
                parsed_message = text
            else:
                return None

        addr = self.personality.get_address_title() or "boss"

        # 2. Execute Action Based on Semantic Extraction
        if parsed_action == "save_contact":
            name = (parsed_target or "").strip()
            phone = (parsed_phone or "").strip()
            if not name or not phone:
                return self.personality.format_response(f"Please tell me the name and phone number to save, {addr}.")
            res = self.executor.execute(ToolCall(name="contact.save", arguments={"name": name, "phone": phone}))
            data = res.data if res.data is not None else getattr(res, "result", None)
            if res.success and isinstance(data, dict):
                return self.personality.format_response(data.get("spoken_response") or data.get("message"))
            return self.personality.format_response(f"I saved contact {name}, {addr}.")

        elif parsed_action == "list_contacts":
            res = self.executor.execute(ToolCall(name="contact.list", arguments={}))
            data = res.data if res.data is not None else getattr(res, "result", None)
            if res.success and isinstance(data, dict):
                return self.personality.format_response(data.get("spoken_response") or data.get("message"))
            return self.personality.format_response("No contacts saved yet.")

        elif parsed_action == "delete_contact":
            name = (parsed_target or "").strip()
            if not name:
                return self.personality.format_response(f"Which contact would you like me to delete, {addr}?")
            res = self.executor.execute(ToolCall(name="contact.delete", arguments={"name": name}))
            data = res.data if res.data is not None else getattr(res, "result", None)
            if res.success and isinstance(data, dict):
                return self.personality.format_response(data.get("spoken_response") or data.get("message"))
            return self.personality.format_response(f"Contact {name} removed, {addr}.")

        elif parsed_action == "send_whatsapp":
            target = (parsed_target or "myself").strip()
            msg = (parsed_message or "Hello!").strip()

            # Resolve contact with fuzzy / difflib matching
            phone_num = contacts_manager.get_phone(target)

            # If recipient is named but has no saved phone number in contacts
            if not phone_num and not contacts_manager.is_phone_number(target) and target.lower() not in ("myself", "me", "self"):
                return self.personality.format_response(
                    f"I found contact '{target}', but you haven't saved a phone number for {target} yet, {addr}. "
                    f"Please say 'save contact {target} as [phone number]' and I'll send it right away."
                )

            # Check if execution is scheduled for a future time
            if parsed_delay > 0:
                def _dispatch_scheduled_whatsapp():
                    try:
                        self.executor.execute(ToolCall(name="whatsapp.send", arguments={"target": target, "phone": target, "message": msg}))
                        done_speech = f"Boss, your scheduled WhatsApp message to {target} has been sent."
                        if hasattr(self, "voice_engine") and self.voice_engine:
                            self.voice_engine.speak(done_speech)
                    except Exception as err:
                        logger.error(f"Error in scheduled WhatsApp dispatch: {err}")

                if hasattr(self, "timer_manager"):
                    self.timer_manager.schedule_task(parsed_delay, _dispatch_scheduled_whatsapp, label=f"WhatsApp to {target}")
                else:
                    import threading
                    t = threading.Timer(parsed_delay, _dispatch_scheduled_whatsapp)
                    t.daemon = True
                    t.start()

                timing_desc = parsed_scheduled_disp or f"in {parsed_delay} seconds"
                return self.personality.format_response(
                    f"Understood, {addr}. I have scheduled your WhatsApp message to {target} {timing_desc}."
                )

            res = self.executor.execute(ToolCall(name="whatsapp.send", arguments={"target": target, "phone": target, "message": msg}))
            data = res.data if res.data is not None else getattr(res, "result", None)
            if res.success and isinstance(data, dict):
                return self.personality.format_response(data.get("spoken_response") or data.get("message", "WhatsApp message prepared."))
            elif not res.success:
                err = res.error.get("message") if isinstance(res.error, dict) else str(res.error)
                return self.personality.format_response(err)

        elif parsed_action == "send_telegram":
            alert_level = "INFO"
            if any(w in lower for w in ["urgent", "critical", "oom", "warning", "emergency", "alert"]):
                alert_level = "CRITICAL" if any(w in lower for w in ["critical", "emergency", "oom"]) else "WARNING"
            msg = parsed_message or "Operational update from M.A.D.D.Y."

            if parsed_delay > 0:
                def _dispatch_scheduled_telegram():
                    try:
                        self.executor.execute(ToolCall(name="telegram.send", arguments={"message": msg, "alert_level": alert_level}))
                        done_speech = "Boss, your scheduled Telegram alert has been sent."
                        if hasattr(self, "voice_engine") and self.voice_engine:
                            self.voice_engine.speak(done_speech)
                    except Exception as err:
                        logger.error(f"Error in scheduled Telegram dispatch: {err}")

                if hasattr(self, "timer_manager"):
                    self.timer_manager.schedule_task(parsed_delay, _dispatch_scheduled_telegram, label="Telegram message")
                else:
                    import threading
                    t = threading.Timer(parsed_delay, _dispatch_scheduled_telegram)
                    t.daemon = True
                    t.start()

                timing_desc = parsed_scheduled_disp or f"in {parsed_delay} seconds"
                return self.personality.format_response(
                    f"Understood, {addr}. I have scheduled your Telegram message {timing_desc}."
                )

            res = self.executor.execute(ToolCall(name="telegram.send", arguments={"message": msg, "alert_level": alert_level}))
            data = res.data if res.data is not None else getattr(res, "result", None)
            if res.success and isinstance(data, dict):
                return self.personality.format_response(data.get("message", "Telegram message dispatched."))
            elif not res.success:
                err = res.error.get("message") if isinstance(res.error, dict) else str(res.error)
                return self.personality.format_response(f"Telegram dispatch failed: {err}")

        return None

    def _handle_session_capture_command(self, text: str, request_id: str) -> Optional[str]:
        """
        Handle owner-activated session capture & guest audit requests via SemanticDispatcher
        and standard ToolExecutor (PermissionEngine -> PathSandbox -> AuditLedger).
        """
        lower = text.lower().strip().rstrip(".?!")
        has_active = hasattr(self, "session_capture_manager") and self.session_capture_manager.has_active_session()

        env_passkey = os.getenv("SESSION_CAPTURE_PASSKEY", "").strip().lower()
        is_env_passkey = bool(env_passkey) and lower == env_passkey

        # If an active session is running and user says dismiss/stop/cancel/standby or passkeys
        if (has_active and lower in (
            "dismiss", "dismiss session", "dismiss capture", "stop", "stop session",
            "cancel", "cancel session", "end session", "terminate session", "close session",
            "stop capturing", "dismiss capturing", "standby", "stop stop"
        )) or lower == "stop stop" or is_env_passkey:
            action = "stop"
        else:
            payload = self.semantic_dispatcher.dispatch(text)
            if payload.intent != SemanticIntent.SESSION_CAPTURE:
                return None
            action = payload.os_action or payload.slots.get("action", "status")

        tool_name = f"session.capture.{action}"

        params: Dict[str, Any] = {"task_id": request_id, "caller_id": "owner"}

        # Only pass capture configuration arguments for start commands
        if action == "start":
            # Extract interval if explicitly specified (e.g. "every 10 seconds", "interval 15s")
            interval_match = re.search(r"\b(\d+)\s*(?:seconds?|s)\b", text, flags=re.IGNORECASE)
            if interval_match:
                try:
                    val = float(interval_match.group(1))
                    params["display_interval"] = val
                    params["camera_interval"] = val
                except Exception:
                    pass

            if any(w in text.lower() for w in ["without telegram", "no telegram", "disable telegram"]):
                params["enable_telegram"] = False
            else:
                from integrations.messaging import TelegramClient
                if "telegram" in text.lower() or TelegramClient().is_configured:
                    params["enable_telegram"] = True
                    params["telegram_chat_id"] = os.getenv("TELEGRAM_CHAT_ID", "").strip() or None

        res = self.executor.execute(ToolCall(name=tool_name, arguments=params))
        data = res.data if res.data is not None else getattr(res, "result", None)
        if res.success and isinstance(data, dict):
            spoken = data.get("spoken_response") or data.get("message")
            if spoken:
                return self.personality.format_response(spoken)
            return self.personality.format_response(f"Session capture action '{action}' completed.")
        elif not res.success:
            err_msg = res.error.get("message") if isinstance(res.error, dict) else str(res.error)
            return self.personality.format_response(f"Session capture request could not be completed: {err_msg}")

        return None

    def _is_pure_declarative_statement(self, text: str) -> bool:
        """Determine if an utterance is a pure declaration of a personal fact/preference without any question or action request."""
        lower = text.lower().strip()
        if "?" in text:
            return False
        # Interrogative words or question clauses anywhere in the utterance
        question_patterns = [
            r"\b(?:who|what|where|when|why|how|which)\b",
            r"\b(?:can|could|would|should|will|do|does|did|is|are|was|were)\s+(?:i|you|we|it|there|anyone)\b",
            r"\b(?:how\s+(?:can|do|to|should|would)|what\s+(?:should|can|do|is|are))\b",
            r"\b(?:any\s+(?:advice|tips|suggestions?|ideas?|ways?|solutions?))\b",
        ]
        if any(re.search(pat, lower) for pat in question_patterns):
            return False

        action_verbs = (
            r"\b(?:find|search|show|open|launch|run|play|calculate|compute|solve|convert|"
            r"evaluate|test|execute|take|click|type|save|send|set|tell\s+me|fix|help|"
            r"suggest|recommend|advise|explain|give\s+me|remedy|cure|treat|diagnose)\b"
        )
        if re.search(action_verbs, lower):
            return False
        return True

    def get_contextual_system_prompt(self, query: Optional[str] = None) -> str:
        """
        Build an enriched system prompt containing M.A.D.D.Y. core personality,
        relevant long-term memories, user profile, and persistent facts
        (matching ChatGPT / Gemini system grounding).
        """
        base_prompt = self.personality.get_llm_system_prompt()

        memories_text = []
        # 1. Query-relevant memories via similarity
        if query and hasattr(self, "memory_manager"):
            try:
                rel_records = self.memory_manager.retrieve_relevant(query, max_records=5, min_score=0.20)
                for r in rel_records:
                    fact = f"- {r.content}"
                    if fact not in memories_text:
                        memories_text.append(fact)
            except Exception:
                pass

        # 2. General active profile & recent memories (deduplicated)
        if hasattr(self, "memory_manager") and hasattr(self.memory_manager, "store"):
            try:
                active_records = self.memory_manager.store.list_records(status=MemoryStatus.ACTIVE, limit=10)
                for r in active_records:
                    fact = f"- {r.content}"
                    if fact not in memories_text:
                        memories_text.append(fact)
            except Exception:
                pass

        # 3. User preferences
        if hasattr(self, "user_preferences"):
            try:
                prefs = self.user_preferences.get_all_preferences()
                for k, v in prefs.items():
                    if k not in ("name", "preferred_name") and isinstance(v, (str, int, float, bool)):
                        pref_str = f"- Preference: {k} is {v}"
                        if pref_str not in memories_text:
                            memories_text.append(pref_str)
            except Exception:
                pass

        if not memories_text:
            return base_prompt

        memory_section = (
            "\n\n### PERSISTENT LONG-TERM MEMORIES & USER CONTEXT:\n"
            "The following facts, hardware specs, projects, and preferences are known about the user:\n"
            + "\n".join(memories_text[:12])
            + "\nUse these details to personalize responses naturally, answer questions about the user's setup or projects, "
            "and maintain conversational continuity across sessions. Do not contradict them."
        )
        return base_prompt + memory_section

    def _handle_screen_aware_request(self, text: str, request_id: str, task_mem: TaskMemory) -> str:
        """Automatically capture screen and extract deep visual context with active window and OCR reasoning."""
        temp_dir = tempfile.gettempdir()
        screenshot_path = os.path.join(temp_dir, f"maddy_screen_{request_id}.png")
        capture_res = self.computer_controller.take_screenshot(screenshot_path)

        if not capture_res.get("path") or not os.path.exists(capture_res["path"]):
            return "Attempted to inspect the screen, but display capture was unavailable."

        content_analysis = self.vision_engine.analyze_screen_content(
            screenshot_path,
            model_provider=self.model_provider,
            user_query=text,
        )
        return self.personality.format_response(content_analysis.get("summary", "Screen inspected."))

    def _generate_dynamic_action_response(self, user_request: str, action_summary: str, default_fallback: str) -> str:
        """Dynamically generate a natural, articulate confirmation via LLM instead of canned strings."""
        if hasattr(self, "model_provider") and hasattr(self.model_provider, "is_configured") and self.model_provider.is_configured():
            try:
                from personality.prompts import MADDY_CORE_SYSTEM_PROMPT
                addr = self.personality.get_address_title() or "Sir"
                switch_hint = " Explicitly include the word 'switched' in your confirmation." if "switched to" in action_summary.lower() else ""
                prompt = (
                    f"User request: \"{user_request}\"\n"
                    f"Action executed: {action_summary}\n"
                    f"Preferred address form: '{addr}'. "
                    f"Respond as M.A.D.D.Y. in 1 concise, natural, articulate sentence (under 12 words). "
                    f"Acknowledge the action directly.{switch_hint} If using an address form, use only '{addr}' at most once. "
                    f"Never use multiple address titles (e.g. do not say both 'boss' and 'sir'). Do not use robotic clichés or asterisks."
                )
                res = self.model_provider.generate(prompt, system_prompt=MADDY_CORE_SYSTEM_PROMPT)
                content = (res.content or "").strip()
                if (
                    content
                    and getattr(res, "finish_reason", None) != "error"
                    and not any(k in content.lower() for k in ("neural brain service", "failed to communicate", "my cloud neural", "rate limit", "catch my breath"))
                ):
                    if switch_hint and "switched" not in content.lower():
                        return self.personality.format_response(default_fallback)
                    return self.personality.format_response(content)
            except Exception:
                pass
        return self.personality.format_response(default_fallback)

    def _handle_app_lifecycle_command(self, clean_request: str) -> Optional[str]:
        """Fast-path application open, close, and window switching with zero LLM overhead."""
        lower = clean_request.lower()
        # If it's a web search, shopping, download, URL, or navigational command, let router / dispatcher handle it!
        web_markers = (
            "download", "page", "website", "site", "cart", "orders", "wishlist",
            "amazon", "flipkart", "youtube", "google", "wikipedia", "github", "reddit",
            "search", "find", "buy", "under", "price", "below", "http://", "https://",
            "camera", "webcam", "photo", "session", "session capture", "capture"
        )
        if any(w in lower for w in web_markers):
            return None

        if re.search(r"\b(?:open|launch|start|switch\s+to|bring\s+up|close|terminate|exit|kill)\s+[a-zA-Z0-9_\-\. ]+", lower):
            return self._handle_direct_app_launch(clean_request)
        return None

    def parse_web_search_command(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse natural language search commands into target site, query, and search URL.
        Supports:
        - "open <site> and search (for) <query>" (e.g. "open youtube and search for most of the best")
        - "search (for) <query> on/in <site>" (e.g. "search for mrbeast on youtube")
        - "search <site> for <query>" (e.g. "search youtube for most of the best")
        - "on/in <site> search (for) <query>" (e.g. "on youtube search for lo-fi music")
        - Standard Google Web & Image search queries
        """
        import urllib.parse

        lower = text.lower().strip().rstrip(".?!")
        lower = re.sub(r"\b(?:pin\s*guy|pen\s*guy|pin\s*dive|pin\s*drive)\b", "pendrive", lower, flags=re.IGNORECASE)
        addr = self.personality.get_address_title()

        # Guard: Local filesystem queries should not be intercepted as web search
        if any(w in lower for w in ["file ", "files", "directory", "folder", "read file", "save file"]):
            if not any(w in lower for w in ["web", "online", "google", "browser", "chrome", "firefox", "edge", "youtube", "amazon", "wikipedia", "reddit", "github"]):
                return None

        # 1. Detect browser preference if specified
        browser_target = "chrome" if "chrome" in lower else ("firefox" if "firefox" in lower else ("edge" if "edge" in lower else "browser"))

        # Strip browser indicators
        cleaned = re.sub(
            r"\b(?:in|on|using|with)\s+(?:the\s+)?(?:google\s+chrome|chrome|firefox|edge|browser)\b",
            "",
            lower,
        ).strip()

        # Strip polite / conversational leading prefixes (e.g. "can you open X", "please search Y")
        cleaned = re.sub(
            r"^(?:maddy\s+|hey\s+maddy\s+)?(?:can\s+you\s+(?:please\s+)?|could\s+you\s+(?:please\s+)?|please\s+|would\s+you\s+(?:please\s+)?|will\s+you\s+(?:please\s+)?|i\s+want\s+you\s+to\s+|help\s+me\s+)",
            "",
            cleaned,
        ).strip()

        # 2. Check whether user requested an image / picture specifically
        is_image = bool(re.search(r"\b(?:picture|pictures|photo|photos|image|images)\b", cleaned))

        # 3. Check E-Commerce Product Search with price and rating filters
        if hasattr(self, "product_search_engine") and self.product_search_engine.is_product_query(cleaned):
            p_res = self.product_search_engine.parse_product_search(text, model_provider=self.model_provider)
            if p_res:
                return {
                    "browser": p_res["browser"],
                    "browser_disp": p_res["browser_disp"],
                    "target_site": p_res["site"],
                    "site_display": p_res["site"],
                    "query": p_res["product"],
                    "url": p_res["url"],
                    "action_desc": p_res["action_desc"],
                    "fallback": p_res["fallback"],
                    "is_product_search": True,
                    "product_info": p_res,
                }

        target_site = None
        query = None

        # Pattern 1: open/launch/go to <site> (and) search/find/look up (for) <query>
        # e.g. "open youtube and search for most of the best", "open amazon search wireless headphones"
        m1 = re.search(
            r"^(?:open|launch|go\s+to)\s+(?P<site>[a-zA-Z0-9_\-\. ]+?)\s+(?:and\s+)?(?:search\s+(?:for\s+)?|look\s+up\s+|find\s+)(?P<query>.+)$",
            cleaned,
        )
        if m1:
            raw_s = m1.group("site").strip()
            raw_q = m1.group("query").strip()
            if raw_s in ("chrome", "google chrome", "browser", "the browser", "firefox", "edge", "the web"):
                target_site = "google"
            else:
                target_site = raw_s
            query = raw_q

        # Pattern 2: search (for) <query> on/in/from <site>
        # e.g. "search for most of the best on youtube", "search mrbeast on youtube", "find wallpaper from pinterest"
        if not query:
            m2 = re.search(
                r"^(?:search\s+(?:for\s+)?|look\s+up\s+|find\s+(?:me\s+)?)(?P<query>.+?)\s+(?:on|in|from)\s+(?P<site>[a-zA-Z0-9_\-\. ]+)$",
                cleaned,
            )
            if m2:
                raw_s = m2.group("site").strip()
                raw_q = m2.group("query").strip()
                if raw_s in ("chrome", "google chrome", "browser", "the browser", "firefox", "edge", "the web"):
                    target_site = "google"
                else:
                    target_site = raw_s
                query = raw_q

        # Pattern 3: search <site> for <query>
        # e.g. "search youtube for most of the best", "search wikipedia for alan turing"
        if not query:
            m3 = re.search(
                r"^(?:search\s+|look\s+up\s+in\s+)(?P<site>[a-zA-Z0-9_\-\.]+)\s+for\s+(?P<query>.+)$",
                cleaned,
            )
            if m3:
                raw_s = m3.group("site").strip().lower()
                raw_q = m3.group("query").strip()
                if raw_s in KNOWN_SEARCH_SITES or raw_s in KNOWN_WEBSITES or raw_s in ("chrome", "browser", "google", "firefox", "edge") or "." in raw_s:
                    if raw_s in ("chrome", "google chrome", "browser", "firefox", "edge"):
                        target_site = "google"
                    else:
                        target_site = raw_s
                    query = raw_q

        # Pattern 4: on/in <site> search (for) <query>
        # e.g. "on youtube search for lo-fi music", "in wikipedia search quantum physics"
        if not query:
            m4 = re.search(
                r"^(?:on|in)\s+(?P<site>[a-zA-Z0-9_\-\. ]+?)\s+(?:search\s+(?:for\s+)?|look\s+up\s+|find\s+)(?P<query>.+)$",
                cleaned,
            )
            if m4:
                raw_s = m4.group("site").strip().lower()
                raw_q = m4.group("query").strip()
                if raw_s in KNOWN_SEARCH_SITES or raw_s in KNOWN_WEBSITES or "." in raw_s:
                    target_site = raw_s
                    query = raw_q

        # Pattern 5: Standard Web Search / Google Search
        if not query:
            # Guard: If this is an explicit site or section navigation request (e.g. "show my cart on amazon", "open my cart in amazon"),
            # avoid intercepting it as a search query so it directly launches the section.
            if self._is_site_or_section_navigation(cleaned):
                return None

            target = re.sub(
                r"^(?:search\s+(?:for\s+)?|find\s+(?:me\s+)?(?:a\s+)?|show\s+(?:me\s+)?(?:a\s+)?|look\s+up\s+|google\s+)",
                "",
                cleaned,
            ).strip()
            if target != cleaned or is_image or "search" in cleaned.split():
                query = target
                if is_image:
                    target_site = "google images"
                else:
                    target_site = "google"

        if not query:
            return None

        # Clean query: strip quotes and leading conversational prefixes
        query = query.strip().strip("'\"").strip().rstrip(".?!")
        if query.lower().startswith("for "):
            query = query[4:].strip()
        query = re.sub(r"^(?:a\s+good|a\s+nice|a\s+great|some|an|a)\s+", "", query, flags=re.IGNORECASE).strip()
        if not query:
            return None

        # Clean site name
        if target_site:
            target_site = target_site.strip().rstrip(".?!")
            target_site = re.sub(r"^(?:the\s+website\s+|website\s+|site\s+|the\s+)", "", target_site).strip()

        # Handle image search override
        if is_image and (not target_site or target_site in ("google", "google search", "google images")):
            target_site = "google images"
            m_of = re.search(r"(?:pictures?|photos?|images?)\s+of\s+(.+)", query)
            if m_of:
                query = m_of.group(1).strip()
            else:
                query = re.sub(r"\b(?:pictures?|photos?|images?)\b", "", query).strip()
            query = re.sub(r"^(?:a|an|the|some)\s+", "", query).strip()

        # Determine target search URL and display name
        encoded_query = urllib.parse.quote_plus(query)
        site_key = (target_site or "google").lower()

        if site_key in KNOWN_SEARCH_SITES:
            site_display, url_tmpl = KNOWN_SEARCH_SITES[site_key]
            url = url_tmpl.format(query=encoded_query)
        elif "." in site_key and not site_key.endswith("."):
            site_display = target_site
            url = f"https://www.google.com/search?q=site:{site_key}+{encoded_query}"
        elif target_site and site_key not in ("google", "google search"):
            site_display = target_site.capitalize()
            url = f"https://www.google.com/search?q={encoded_query}+on+{urllib.parse.quote_plus(target_site)}"
        else:
            site_display = "Google"
            url = f"https://www.google.com/search?q={encoded_query}"

        # Browser display name
        browser_disp = "Chrome" if browser_target == "chrome" else ("Firefox" if browser_target == "firefox" else ("Edge" if browser_target == "edge" else "Browser"))

        # Formulate descriptions
        if site_display == "Google Images":
            action_desc = f"Opened Google Images search for '{query}' in {browser_disp}."
            fallback = f"Showing pictures of {query} in {browser_disp}, {addr}."
        elif site_display == "Google":
            action_desc = f"Searched for '{query}' on Google in {browser_disp}."
            fallback = f"Searching for '{query}' in {browser_disp}, {addr}."
        else:
            action_desc = f"Opened {site_display} and searched for '{query}' in {browser_disp}."
            fallback = f"Searching for '{query}' on {site_display}, {addr}."

        return {
            "browser": browser_target,
            "browser_disp": browser_disp,
            "target_site": target_site,
            "site_display": site_display,
            "query": query,
            "url": url,
            "action_desc": action_desc,
            "fallback": fallback,
        }

    def _handle_direct_web_search(self, text: str, intent_payload: Optional[IntentPayload] = None) -> Optional[str]:
        """Open search query directly in browser on specific websites (YouTube, Wikipedia, etc.) or Google."""
        parsed = None
        if intent_payload and intent_payload.slots:
            slots = intent_payload.slots
            if intent_payload.intent == SemanticIntent.SEARCH_COMMERCIAL or slots.get("brands") or slots.get("max_price") or slots.get("min_price") or slots.get("sort_by") or slots.get("is_branded"):
                p_res = self.product_search_engine.build_search_from_slots(slots, text)
                if p_res:
                    parsed = {
                        "browser": p_res["browser"],
                        "browser_disp": p_res["browser_disp"],
                        "target_site": p_res["site"],
                        "site_display": p_res["site"],
                        "query": p_res["product"],
                        "url": p_res["url"],
                        "action_desc": p_res["action_desc"],
                        "fallback": p_res["fallback"],
                        "is_product_search": True,
                        "product_info": p_res,
                    }

        if not parsed:
            parsed = self.parse_web_search_command(text)
        if not parsed:
            return None

        url = parsed["url"]
        browser_target = parsed["browser"]
        action_desc = parsed["action_desc"]
        fallback = parsed["fallback"]
        query = parsed["query"]
        site_display = parsed["site_display"]

        try:
            args = {"browser_name": browser_target, "url": url}
            res = self.executor.execute(ToolCall(name="browser.open", arguments=args))
            if not res.success:
                if sys.platform.startswith("win"):
                    os.startfile(url)
                else:
                    import webbrowser
                    webbrowser.open(url)

            # Ensure browser window is brought directly to the front on the laptop screen
            time.sleep(0.5)
            try:
                self.platform.focus_window_or_application(browser_target)
            except Exception:
                pass

            # Capture screenshot of the PC screen showing the opened search result
            shot_path = None
            try:
                time.sleep(0.8)
                shot = self.computer_controller.take_screenshot()
                if shot and shot.get("path") and os.path.exists(shot["path"]):
                    shot_path = shot["path"]
            except Exception:
                pass

            # If this was a product query with hardware specs or delivery constraints, provide intelligent model advice
            if parsed.get("is_product_search") and hasattr(self, "product_search_engine"):
                p_info = parsed.get("product_info") or parsed
                adv = self.product_search_engine.advise_and_synthesize(
                    text,
                    p_info,
                    self.model_provider,
                    self.personality.get_address_title(),
                )
                if adv and adv != parsed["fallback"]:
                    formatted = self.personality.format_response(adv)
                    self._last_assistant_response = formatted
                    if shot_path:
                        return {"response": formatted, "path": shot_path}
                    return formatted

            formatted = self._generate_dynamic_action_response(text, action_desc, fallback)
            if shot_path:
                return {"response": formatted, "path": shot_path}
            return formatted
        except Exception as e:
            return f"Attempted to search for '{query}' on {site_display}, but encountered an issue: {str(e)}"

    def _is_site_or_section_navigation(self, text: str) -> bool:
        """Return True if text is requesting navigation to a website section or page rather than a search query."""
        lower = text.lower().strip()
        has_nav_verb = bool(re.search(r"\b(?:open|launch|go\s+to|navigate\s+to|view|show)\b", lower))
        has_section = any(s in lower for s in [
            "cart", "basket", "orders", "wishlist", "history", "subscriptions",
            "notifications", "bookmarks", "inbox", "repos", "repositories"
        ])
        has_site = any(site in lower for site in KNOWN_WEBSITES) or any(site in lower for site, _ in KNOWN_SITE_SECTIONS)

        if has_nav_verb and (has_section or (has_site and not re.search(r"\b(?:search|find|look\s+up)\b", lower))):
            return True
        return False

    def resolve_browser_target(self, raw_target: str) -> Tuple[str, str]:
        """
        Resolve a target string into (display_name, target_url).
        Intelligently resolves:
        1. Direct URLs & domains: 'https://...', 'example.com'
        2. Standalone e-commerce sections: 'my cart', 'orders', 'wishlist' -> Amazon
        3. Site sections with preposition: 'my cart in amazon', 'orders on amazon', 'history on youtube', etc.
        4. Site followed by section: 'amazon cart', 'youtube history', 'github notifications', etc.
        5. Known websites: 'amazon', 'youtube', 'github', etc.
        6. Fallback: Google search for the raw target
        """
        import urllib.parse
        target_clean = raw_target.strip().lower()
        target_clean = re.sub(r"^(?:the\s+website\s+|website\s+|site\s+|the\s+page\s+|page\s+|the\s+)", "", target_clean).strip()

        # 1. Check direct URL
        if target_clean.startswith(("http://", "https://")):
            return (raw_target.strip(), target_clean)

        # 2. Check standalone common e-commerce shortcuts without site specified
        standalone_map = {
            "cart": ("Amazon Cart", "https://www.amazon.in/gp/cart/view.html"),
            "my cart": ("Amazon Cart", "https://www.amazon.in/gp/cart/view.html"),
            "shopping cart": ("Amazon Cart", "https://www.amazon.in/gp/cart/view.html"),
            "the cart": ("Amazon Cart", "https://www.amazon.in/gp/cart/view.html"),
            "basket": ("Amazon Cart", "https://www.amazon.in/gp/cart/view.html"),
            "orders": ("Amazon Orders", "https://www.amazon.in/gp/your-account/order-history"),
            "my orders": ("Amazon Orders", "https://www.amazon.in/gp/your-account/order-history"),
            "order history": ("Amazon Orders", "https://www.amazon.in/gp/your-account/order-history"),
            "wishlist": ("Amazon Wishlist", "https://www.amazon.in/hz/wishlist/ls"),
            "my wishlist": ("Amazon Wishlist", "https://www.amazon.in/hz/wishlist/ls"),
        }
        if target_clean in standalone_map:
            return standalone_map[target_clean]

        # 3. Check section with preposition: "<section> in/on/at/of/from <site>"
        # e.g. "my cart in amazon", "orders on amazon", "history on youtube", "notifications on github"
        m_prep = re.search(
            r"^(?:show\s+|view\s+|check\s+)?(?:my\s+|the\s+)?(?P<section>[a-zA-Z0-9_\- ]+?)\s+(?:in|on|at|of|from)\s+(?:the\s+)?(?P<site>[a-zA-Z0-9_\-\.]+)$",
            target_clean,
            re.IGNORECASE,
        )
        if m_prep:
            site_cand = m_prep.group("site").strip().lower()
            sec_cand = m_prep.group("section").strip().lower()
            sec_clean = re.sub(r"^(?:my\s+|the\s+)", "", sec_cand).strip()

            if (site_cand, sec_clean) in KNOWN_SITE_SECTIONS:
                return KNOWN_SITE_SECTIONS[(site_cand, sec_clean)]
            if (site_cand, sec_cand) in KNOWN_SITE_SECTIONS:
                return KNOWN_SITE_SECTIONS[(site_cand, sec_cand)]

            if site_cand in KNOWN_WEBSITES:
                site_disp, base_url = KNOWN_WEBSITES[site_cand]
                sec_slug = sec_clean.replace(" ", "-")
                return (f"{site_disp} {sec_clean.title()}", f"{base_url}/{urllib.parse.quote(sec_slug)}")

        # 4. Check site followed by section: "<site> [my/the] <section>"
        # e.g. "amazon cart", "amazon my cart", "amazon orders", "youtube history", "github notifications"
        m_sitesec = re.search(
            r"^(?P<site>[a-zA-Z0-9_\-\.]+)\s+(?:my\s+|the\s+)?(?P<section>[a-zA-Z0-9_\- ]+)$",
            target_clean,
            re.IGNORECASE,
        )
        if m_sitesec:
            site_cand = m_sitesec.group("site").strip().lower()
            sec_cand = m_sitesec.group("section").strip().lower()
            sec_clean = re.sub(r"^(?:my\s+|the\s+)", "", sec_cand).strip()

            if (site_cand, sec_clean) in KNOWN_SITE_SECTIONS:
                return KNOWN_SITE_SECTIONS[(site_cand, sec_clean)]
            if (site_cand, sec_cand) in KNOWN_SITE_SECTIONS:
                return KNOWN_SITE_SECTIONS[(site_cand, sec_cand)]

            if site_cand in KNOWN_WEBSITES:
                site_disp, base_url = KNOWN_WEBSITES[site_cand]
                sec_slug = sec_clean.replace(" ", "-")
                return (f"{site_disp} {sec_clean.title()}", f"{base_url}/{urllib.parse.quote(sec_slug)}")

        # 5. Check direct site match in KNOWN_WEBSITES
        if target_clean in KNOWN_WEBSITES:
            return KNOWN_WEBSITES[target_clean]

        # 6. Check domain name with TLD: e.g. "example.com", "sub.domain.org"
        if "." in target_clean and not target_clean.endswith(".") and " " not in target_clean:
            return (target_clean, f"https://{target_clean}")

        # 6b. Live Search Grounding for Dynamic Canonical URL Resolution
        # Uses real-time search engine index to retrieve authentic, verified live URLs rather than hallucinating from frozen weights
        try:
            from browser.engine import BrowserEngine
            b_engine = BrowserEngine()
            
            # Strip conversational filler to prevent diluting search results (e.g. "download page" -> "download")
            clean_search_q = re.sub(r"\b(?:page\s+to\s+download|download\s+page|page|website|site|url|portal)\b", "", raw_target, flags=re.IGNORECASE).strip()
            search_query = clean_search_q if clean_search_q else raw_target

            search_data = b_engine.search(search_query, max_results=5)
            candidates = search_data.get("results", [])
            if not candidates and search_query != raw_target:
                search_data = b_engine.search(raw_target, max_results=5)
                candidates = search_data.get("results", [])

            if candidates:
                # If model provider is available, rank and select the best matching official candidate
                if hasattr(self, "model_provider") and hasattr(self.model_provider, "is_configured") and self.model_provider.is_configured():
                    try:
                        import json
                        cand_summary = "\n".join(
                            f"{idx}. Title: {c['title']}\n   URL: {c['url']}\n   Snippet: {c.get('snippet', '')}"
                            for idx, c in enumerate(candidates, 1)
                        )
                        prompt = (
                            f"User request: \"{raw_target}\"\n"
                            f"Live Search Results:\n{cand_summary}\n\n"
                            f"Select the single most authentic, direct official page or download URL for the user's request from the live results above.\n"
                            f"Respond ONLY with a valid JSON object: {{\"name\": \"Official Name\", \"url\": \"<exact verified url from the list>\"}}.\n"
                            f"Do not write markdown fences or explanation."
                        )
                        res = self.model_provider.generate(prompt)
                        raw = (res.content or "").strip()
                        raw = re.sub(r"^```(?:json)?\s*", "", raw)
                        raw = re.sub(r"\s*```$", "", raw).strip()
                        data = json.loads(raw)
                        if isinstance(data, dict) and data.get("url") and data["url"].startswith("http"):
                            return (data.get("name") or candidates[0]["title"], data["url"])
                    except Exception as e_select:
                        logger.debug(f"LLM candidate selection error, using top search result: {e_select}")

                # Top organic result fallback
                top_cand = candidates[0]
                if top_cand.get("url") and top_cand["url"].startswith("http"):
                    return (top_cand.get("title") or raw_target.strip().title(), top_cand["url"])
        except Exception as e_search:
            logger.debug(f"Live search grounding failed: {e_search}")

        # 7. Fallback to Google Search
        site_name = raw_target.strip().capitalize()
        target_url = f"https://www.google.com/search?q={urllib.parse.quote(raw_target.strip())}"
        return (site_name, target_url)

    def _is_pagination_query(self, text: str) -> bool:
        """Check if user query is a request to advance to the next page of an active product search."""
        if not getattr(self, "_active_product_search", None):
            return False
        lower = text.lower().strip().rstrip(".?!")
        pagination_phrases = [
            "next", "next page", "go to next page", "move to next page", "load next page",
            "next results", "show next page", "next search page", "turn page", "page 2", "page 3"
        ]
        return lower in pagination_phrases or bool(re.match(r"^(?:go\s+to|move\s+to|show\s+me|turn\s+to|load)\s+(?:the\s+)?next\s+page$", lower))

    def _handle_product_pagination(self, text: str) -> Optional[Dict[str, Any]]:
        """Advance active product search to the next page (&page=2) without altering filters."""
        active = getattr(self, "_active_product_search", None)
        if not active:
            return None

        current_page = active.get("page", 1)
        next_page = current_page + 1
        active["page"] = next_page

        base_url = active["base_url"]
        site = active["site"]
        product = active["product"]
        browser_target = active.get("browser", "chrome")

        next_url = self.product_search_engine.build_page_url(base_url, site, next_page)
        active["current_url"] = next_url

        try:
            args = {"browser_name": browser_target, "url": next_url}
            res = self.executor.execute(ToolCall(name="browser.open", arguments=args))
            if not res.success:
                if sys.platform.startswith("win"):
                    os.startfile(next_url)
                else:
                    import webbrowser
                    webbrowser.open(next_url)

            time.sleep(0.5)
            try:
                self.platform.focus_window_or_application(browser_target)
            except Exception:
                pass

            shot_path = None
            try:
                time.sleep(0.8)
                shot = self.computer_controller.take_screenshot()
                if shot and shot.get("path") and os.path.exists(shot["path"]):
                    shot_path = shot["path"]
            except Exception:
                pass

            addr = self.personality.get_address_title() or "boss"
            msg = f"📄 Moved to page {next_page} of your {site} search for '{product}', {addr}."
            return {
                "response": msg,
                "data": {
                    "path": shot_path,
                    "url": next_url,
                    "page": next_page,
                    "site": site,
                    "product": product,
                }
            }
        except Exception as e:
            logger.warning(f"Error paginating product search: {e}")
            return None

    def _handle_negation_recovery(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Handle conversational negation/correction when user catches an assistant mistake,
        especially when previous command mistakenly triggered Spotify instead of page navigation.
        """
        lower = text.lower().strip()
        negation_markers = [
            "no da", "no da kanna", "it's not", "its not", "that's not what i meant", "thats not what i meant",
            "wrong", "no that's wrong", "no stop", "not spotify", "not that", "i meant next page"
        ]
        is_negation = any(m in lower for m in negation_markers)
        if not is_negation:
            return None

        # Case 1: Spotify was mistakenly triggered right after an active product search
        if getattr(self, "_last_executed_command_type", None) == "spotify_playback" and getattr(self, "_active_product_search", None):
            addr = self.personality.get_address_title() or "boss"
            try:
                self.spotify.pause_playback()
            except Exception:
                pass
            page_res = self._handle_product_pagination("next page")
            if page_res:
                page_res["response"] = f"My apologies, {addr}. I mistook your command for Spotify. Moving to page {self._active_product_search.get('page', 2)} of your product search on your screen now."
                return page_res

        # Case 2: General recovery
        addr = self.personality.get_address_title() or "boss"
        return {
            "response": f"Understood, {addr}. My mistake. What would you like me to do instead?",
            "data": {},
        }


    def _handle_direct_browser_launch(self, text: str) -> Optional[str]:
        """Direct semantic browser open and verification, supporting URLs and websites."""
        import urllib.parse
        lower = text.lower().strip()

        # Intercept search commands if accidentally passed here
        if re.search(r"\b(?:and\s+)?(?:search|look\s+up|find)\b", lower):
            search_res = self._handle_direct_web_search(text)
            if search_res:
                return search_res

        target = "chrome" if "chrome" in lower else ("firefox" if "firefox" in lower else ("edge" if "edge" in lower else "browser"))

        site_name = None
        target_url = None

        # Pattern 1: open <site/section> on/in <browser> (e.g. "open wikipedia on chrome", "open my cart in amazon")
        m = re.search(r"\b(?:open|launch|go\s+to|navigate\s+to|view|show)\s+(.+?)(?:\s+(?:on|in|using|with)\s+(?:the\s+)?(?:google\s+chrome|chrome|firefox|edge|browser))?$", lower)
        if m and m.group(1):
            raw_target = m.group(1).strip()
            if raw_target not in ("chrome", "google chrome", "browser", "the browser", "firefox", "edge", "the web"):
                site_name, target_url = self.resolve_browser_target(raw_target)

        # Pattern 2: open <browser> and (open|go to|navigate to|view|show) <site/section>
        if not target_url:
            m2 = re.search(r"\b(?:open|launch)\s+(?:chrome|google\s+chrome|browser|firefox|edge)\s+(?:and\s+)?(?:go\s+to|open|navigate\s+to|view|show)\s+(.+)$", lower)
            if m2 and m2.group(1):
                raw_target = m2.group(1).strip()
                site_name, target_url = self.resolve_browser_target(raw_target)

        # Deduplication check: If target site or browser is already open, focus it instead of duplicating!
        from platform_layer import get_platform
        plat = get_platform()

        if site_name and hasattr(plat, "focus_window_or_application") and plat.focus_window_or_application(site_name):
            action_desc = f"Switched to your open {site_name}."
            fallback = f"Switched to {site_name}."
            return self._generate_dynamic_action_response(text, action_desc, fallback)
        elif not site_name and hasattr(plat, "focus_window_or_application") and plat.focus_window_or_application(target):
            browser_disp = target.capitalize()
            action_desc = f"Switched to your open {browser_disp} window."
            fallback = f"Switched to {browser_disp}."
            return self._generate_dynamic_action_response(text, action_desc, fallback)

        args = {"browser_name": target}
        if target_url:
            args["url"] = target_url

        res = self.executor.execute(ToolCall(name="browser.open", arguments=args))

        if res.success:
            data = res.data or {}
            browser_disp = data.get("browser", target).capitalize()
            reused = data.get("reused_tab", False) or data.get("status") == "reused_existing_tab"
            if reused:
                action_desc = f"Switched to your pre-opened {site_name or browser_disp} tab."
                fallback = f"Switched to {site_name or browser_disp}."
            elif site_name:
                action_desc = f"Opened {site_name} in {browser_disp}."
                fallback = f"Opened {site_name} in {browser_disp}."
            else:
                action_desc = f"{browser_disp} is open and verified running."
                fallback = f"{browser_disp} is open."
            return self._generate_dynamic_action_response(text, action_desc, fallback)
        else:
            err = res.error.get("message") if res.error else "Could not verify browser process"
            return f"Attempted to open browser, but verification failed: {err}"

    def _handle_direct_app_launch(self, text: str) -> Optional[str]:
        """Direct zero-latency execution: open in system if installed, else open in Chrome; handle closing."""
        lower = text.lower().strip()

        # Handle Close / Terminate
        m_close = re.search(r"\b(?:close|terminate|exit|quit|kill)\s+([a-zA-Z0-9_\-\. ]+)", lower)
        if m_close:
            raw_target = m_close.group(1).strip().rstrip(".?!")
            from platform_layer import get_platform
            plat = get_platform()
            try:
                plat.close_application(raw_target)
                action_desc = f"Closed application {raw_target}."
                fallback = f"Closed {raw_target.capitalize()}."
                return self._generate_dynamic_action_response(text, action_desc, fallback)
            except Exception:
                return f"Attempted to close {raw_target.capitalize()}, but no running process was found."

        # Intercept search commands if embedded in launch command (e.g. "open youtube and search for ...")
        if re.search(r"\b(?:and\s+)?(?:search|look\s+up|find)\b", lower):
            search_res = self._handle_direct_web_search(text)
            if search_res:
                return search_res

        # Handle Open / Launch
        # Clean conversational polite preambles ("can you open...", "could you launch...", "please start...")
        cleaned_cmd = re.sub(r"^(?:can\s+you\s+|could\s+you\s+|please\s+|will\s+you\s+|would\s+you\s+|can\s+you\s+please\s+|could\s+you\s+please\s+)", "", lower).strip()
        m = re.search(r"\b(?:open|launch|start|go\s+to|switch\s+to|bring\s+up)\s+(.+)$", cleaned_cmd)
        if m:
            raw_target = m.group(1).strip().rstrip(".?!")
            # Strip trailing conversational politeness / qualifiers ("... for me", "... please", "... for us")
            raw_target = re.sub(r"\b(?:for\s+me|please|for\s+us|right\s+now|now)\b.*$", "", raw_target).strip()
            # Strip leading articles ("the camera" -> "camera", "a terminal" -> "terminal")
            raw_target = re.sub(r"^(?:the|an|a|my)\s+", "", raw_target).strip()

            # If the command specifically requests a browser or is a known website, let browser launch handle it
            target_clean = raw_target.lower()
            if (
                re.search(r"\b(?:on|in|using|with)\s+(?:the\s+)?(?:google\s+chrome|chrome|firefox|edge|browser)\b", target_clean)
                or (target_clean in KNOWN_WEBSITES and target_clean not in ("spotify",))
                or (target_clean in KNOWN_SEARCH_SITES and target_clean not in ("spotify",))
                or ("." in target_clean and not target_clean.endswith(".") and " " not in target_clean)
            ):
                return self._handle_direct_browser_launch(text)

            from platform_layer import get_platform
            plat = get_platform()

            # 0. Deduplication: Check if application or window is ALREADY OPEN -> focus it without opening again!
            if hasattr(plat, "focus_window_or_application") and plat.focus_window_or_application(raw_target):
                action_desc = f"Switched to your open {raw_target}."
                fallback = f"Switched to {raw_target.capitalize()}."
                return self._generate_dynamic_action_response(text, action_desc, fallback)

            # 1. Check if application is available locally on the system
            if hasattr(plat, "is_application_available") and plat.is_application_available(raw_target):
                try:
                    plat.open_application(raw_target)
                    action_desc = f"Opened {raw_target} on the system."
                    fallback = f"Opened {raw_target.capitalize()}."
                    return self._generate_dynamic_action_response(text, action_desc, fallback)
                except Exception:
                    pass

            # 2. If the app is unknown, don't silently fall back to a browser search.
            if not re.search(r"\b(?:youtube|spotify|chrome|firefox|edge|amazon|wikipedia|github|gmail|reddit|linkedin|notion|zoom|slack|discord|teams|calendar|outlook|excel|word|powerpoint|vscode|code)\b", raw_target.lower()) and "." not in raw_target and " " not in raw_target:
                return f"I couldn't find an application named '{raw_target}' on this system."

            # 3. If NOT installed on the system, open it in Chrome / browser only for recognized sites or browsers.
            return self._handle_direct_browser_launch(text)
        return None

    def _handle_direct_spotify_command(self, text: str) -> Optional[str]:
        """Direct semantic Spotify playback / playlist handling with Spotify Web API and Web player."""
        import urllib.parse
        import webbrowser
        import sys

        lower = text.lower().strip()
        # Normalize punctuation commas and semicolons to spaces so Whisper speech pauses never break regex
        normalized = re.sub(r"[,;]+", " ", lower)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        addr = self.personality.get_address_title()
        from platform_layer import get_platform
        plat = get_platform()
        from browser import browser_tab_manager

        # 1. Open / Switch to Spotify Web
        if re.search(r"\b(open|launch|switch\s+to|bring\s+up)\b.*\bspotify\b", normalized):
            reuse_res = browser_tab_manager.open_or_reuse("spotify", "https://open.spotify.com", force_navigate=False)
            if reuse_res.get("reused"):
                return f"Switched to your open Spotify Web, {addr}."
            return f"Opened Spotify Web for you, {addr}."

        # 2. Pause / Stop playback
        if any(w in normalized for w in ["pause", "stop music", "stop song", "pause spotify", "stop spotify"]):
            paused = False
            if self.spotify and self.spotify.is_authenticated():
                try:
                    p_res = self.spotify.pause()
                    if p_res.get("success"):
                        paused = True
                except Exception:
                    pass
            if not paused and sys.platform.startswith("win"):
                try:
                    import ctypes
                    VK_MEDIA_PLAY_PAUSE = 0xB3
                    ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 0, 0)
                    time.sleep(0.04)
                    ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 2, 0)
                except Exception:
                    pass
            return self._generate_dynamic_action_response(text, "Paused Spotify playback.", f"Spotify playback paused, {addr}.")

        # 3. Next Track / Skip
        if "next" in normalized or "skip" in normalized:
            skipped = False
            if self.spotify and self.spotify.is_authenticated():
                try:
                    s_res = self.spotify.next_track()
                    if s_res.get("success"):
                        skipped = True
                except Exception:
                    pass
            if not skipped and sys.platform.startswith("win"):
                try:
                    import ctypes
                    VK_MEDIA_NEXT_TRACK = 0xB0
                    ctypes.windll.user32.keybd_event(VK_MEDIA_NEXT_TRACK, 0, 0, 0)
                    time.sleep(0.04)
                    ctypes.windll.user32.keybd_event(VK_MEDIA_NEXT_TRACK, 0, 2, 0)
                except Exception:
                    pass
            return self._generate_dynamic_action_response(text, "Skipped to next track on Spotify.", f"Skipped to next track, {addr}.")

        # 4. Previous Track
        if "previous" in normalized or "back" in normalized:
            prev_done = False
            if self.spotify and self.spotify.is_authenticated():
                try:
                    pr_res = self.spotify.previous_track()
                    if pr_res.get("success"):
                        prev_done = True
                except Exception:
                    pass
            if not prev_done and sys.platform.startswith("win"):
                try:
                    import ctypes
                    VK_MEDIA_PREV_TRACK = 0xB1
                    ctypes.windll.user32.keybd_event(VK_MEDIA_PREV_TRACK, 0, 0, 0)
                    time.sleep(0.04)
                    ctypes.windll.user32.keybd_event(VK_MEDIA_PREV_TRACK, 0, 2, 0)
                except Exception:
                    pass
            return self._generate_dynamic_action_response(text, "Skipped to previous track on Spotify.", f"Skipped to previous track, {addr}.")

        # 4b. Seek to timestamp or fast-forward/rewind within active playback
        # Absolute seek: "seek to 1:30", "jump to 2 minutes", "go to 1 minute 15 seconds"
        m_seek = re.search(r"\b(?:seek\s+(?:to)?|jump\s+to|go\s+to|set\s+position\s+to)\s+(\d{1,2}):(\d{2})\b", normalized)
        if not m_seek:
            m_seek = re.search(r"\b(?:seek\s+(?:to)?|jump\s+to|go\s+to|set\s+position\s+to)\s+(?:(\d+)\s*(?:minutes?|mins?))?\s*(?:and\s*)?(?:(\d+)\s*(?:seconds?|secs?))\b", normalized)

        if m_seek and not any(k in normalized for k in ["play", "open", "launch", "find", "search"]):
            if ":" in m_seek.group(0):
                mins, secs = int(m_seek.group(1)), int(m_seek.group(2))
            else:
                mins = int(m_seek.group(1)) if m_seek.group(1) else 0
                secs = int(m_seek.group(2)) if m_seek.group(2) else 0
            pos_ms = (mins * 60 + secs) * 1000
            time_str = f"{mins}:{secs:02d}"
            if self.spotify and self.spotify.is_authenticated():
                s_res = self.spotify.seek(pos_ms)
                if s_res.get("success"):
                    browser_tab_manager.find_and_focus_service("spotify")
                    action_desc = f"Jumped to {time_str} on Spotify."
                    fallback = f"Seeking to {time_str} on Spotify, {addr}."
                    return self._generate_dynamic_action_response(text, action_desc, fallback)

        # Relative Fast-Forward: "fast forward 30 seconds", "forward 1 minute", "skip forward 15 seconds"
        m_fwd = re.search(r"\b(?:fast\s+forward|forward|skip\s+forward)\s+(?:by\s+)?(?:(\d+)\s*(?:minutes?|mins?))?\s*(?:and\s*)?(?:(\d+)\s*(?:seconds?|secs?))\b", normalized)
        if m_fwd:
            mins = int(m_fwd.group(1)) if m_fwd.group(1) else 0
            secs = int(m_fwd.group(2)) if m_fwd.group(2) else 0
            delta_ms = (mins * 60 + secs) * 1000
            desc = f"{mins} minute{'s' if mins != 1 else ''} " if mins else ""
            desc += f"{secs} second{'s' if secs != 1 else ''}" if secs else ""
            desc = desc.strip()
            if self.spotify and self.spotify.is_authenticated():
                s_res = self.spotify.seek_relative(delta_ms)
                if s_res.get("success"):
                    browser_tab_manager.find_and_focus_service("spotify")
                    action_desc = f"Fast-forwarded {desc} on Spotify."
                    fallback = f"Fast-forwarded {desc}, {addr}."
                    return self._generate_dynamic_action_response(text, action_desc, fallback)

        # Relative Rewind: "rewind 20 seconds", "go back 15 seconds", "jump back 1 minute"
        m_rew = re.search(r"\b(?:rewind|go\s+back|jump\s+back|backward)\s+(?:by\s+)?(?:(\d+)\s*(?:minutes?|mins?))?\s*(?:and\s*)?(?:(\d+)\s*(?:seconds?|secs?))\b", normalized)
        if m_rew:
            mins = int(m_rew.group(1)) if m_rew.group(1) else 0
            secs = int(m_rew.group(2)) if m_rew.group(2) else 0
            delta_ms = -(mins * 60 + secs) * 1000
            desc = f"{mins} minute{'s' if mins != 1 else ''} " if mins else ""
            desc += f"{secs} second{'s' if secs != 1 else ''}" if secs else ""
            desc = desc.strip()
            if self.spotify and self.spotify.is_authenticated():
                s_res = self.spotify.seek_relative(delta_ms)
                if s_res.get("success"):
                    browser_tab_manager.find_and_focus_service("spotify")
                    action_desc = f"Rewound {desc} on Spotify."
                    fallback = f"Rewound {desc}, {addr}."
                    return self._generate_dynamic_action_response(text, action_desc, fallback)

        # 5. Play Song / Playlist on Spotify
        self._active_media_platform = "spotify"
        cleaned = re.sub(r"\b(?:on|in|from|at|via|through|with|or|using)?\s*spotify\b", "", normalized, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^(?:so\s+)?(?:i'm\s+|i\s+am\s+)?(?:in\s+a\s+mood|on\s+a\s+mood(?:\s+out)?|feeling\s+like|bored)[\s,]+", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\b(?:please|can\s+you\s+(?:please\s+)?|could\s+you\s+(?:please\s+)?|i\s+want\s+to\s+listen\s+to)\b", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip("?.! ")

        # 5a. Check playlist
        if "playlist" in normalized:
            m_pl = re.search(r"play\s+(?:my\s+)?([a-zA-Z0-9_\- ]+?)\s+playlist", cleaned)
            pl_name = m_pl.group(1).strip() if m_pl else "playlist"
            if self.spotify and self.spotify.is_authenticated():
                pl_res = self.spotify.play_playlist(pl_name)
                if pl_res.get("success"):
                    browser_tab_manager.find_and_focus_service("spotify")
                    action_desc = f"Playing '{pl_res.get('playlist', pl_name)}' playlist on Spotify."
                    fallback = f"Playing '{pl_res.get('playlist', pl_name)}' playlist on Spotify, {addr}."
                    return self._generate_dynamic_action_response(text, action_desc, fallback)

            web_url = f"https://open.spotify.com/search/{urllib.parse.quote_plus(pl_name)}"
            reuse_res = browser_tab_manager.open_or_reuse("spotify", web_url, force_navigate=True)
            action_desc = f"Opened Spotify Web playlist search for '{pl_name}'."
            fallback = f"Playing '{pl_name}' playlist on Spotify Web, {addr}."
            return self._generate_dynamic_action_response(text, action_desc, fallback)

        # 5b. Generic play / resume
        elif re.match(r"^play\s*(?:me\s+)?(?:the\s+song|a\s+song|some\s+song|some\s+songs|songs|a\s+track|tracks|some\s+tracks|music|some\s+music|something|anything)?$", cleaned) or cleaned in ("resume", "unpause", "play"):
            if self.spotify and self.spotify.is_authenticated():
                res = self.spotify.play()
                if res.get("success"):
                    browser_tab_manager.find_and_focus_service("spotify")
                    action_desc = "Resumed Spotify playback."
                    fallback = f"Playing Spotify for you, {addr}."
                    return self._generate_dynamic_action_response(text, action_desc, fallback)

            if sys.platform.startswith("win"):
                try:
                    import ctypes
                    VK_MEDIA_PLAY_PAUSE = 0xB3
                    ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 0, 0)
                    time.sleep(0.04)
                    ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 2, 0)
                except Exception:
                    pass
            browser_tab_manager.open_or_reuse("spotify", "https://open.spotify.com", force_navigate=False)
            action_desc = "Resumed Spotify playback."
            fallback = f"Playing Spotify for you, {addr}."
            return self._generate_dynamic_action_response(text, action_desc, fallback)

        # 5c. Specific track / song request (with optional timestamp)
        else:
            is_question = bool(re.search(r"^(?:do\s+(?:you|u)|can\s+(?:you|u)|could\s+(?:you|u)|would\s+(?:you|u)|will\s+(?:you|u)|are\s+(?:you|u)|how\s+do\s+(?:you|u)|how\s+to|why\s+do\s+(?:you|u))\b", normalized))
            is_game = bool(re.search(r"\b(?:chess|game|games|football|cricket|tennis|basketball|soccer|hockey|badminton|guitar|piano|violin|drums|flute|cards|poker|monopoly)\b", normalized))
            has_explicit_music = any(k in normalized for k in ["spotify", "song", "songs", "music", "track", "tracks", "playlist", "album", "artist", "lofi", "melody", "melodies", "tunes", "listen to"])
            if (is_question or is_game) and not has_explicit_music:
                return None

            m_song = re.search(r"\bplay\s+(?:me\s+)?(.+)", cleaned)
            q = ""
            target_pos_ms = None
            time_display_str = None
            if m_song:
                raw_q = m_song.group(1).strip()
                # Pattern 1: mm:ss timestamp (e.g. "at 1:30", "from 2:15", "starting at 1:15")
                m_ts = re.search(r"\b(?:starting\s+(?:at|from)|from|at)\s+(\d{1,2}):(\d{2})\b", raw_q, re.IGNORECASE)
                if m_ts:
                    t_mins, t_secs = int(m_ts.group(1)), int(m_ts.group(2))
                    target_pos_ms = (t_mins * 60 + t_secs) * 1000
                    time_display_str = f"{t_mins}:{t_secs:02d}"
                    raw_q = raw_q[:m_ts.start()] + raw_q[m_ts.end():]
                else:
                    # Pattern 2: X minutes Y seconds / X minutes 25 / X minutes
                    m_ts2 = re.search(
                        r"\b(?:starting\s+(?:at|from)|from|at)\s+(\d+)\s*(?:minutes?|mins?|m)\s*(?:and\s*)?(\d+)?\s*(?:seconds?|secs?|s)?\b",
                        raw_q,
                        re.IGNORECASE,
                    )
                    if m_ts2 and (m_ts2.group(1) or m_ts2.group(2)):
                        t_mins = int(m_ts2.group(1)) if m_ts2.group(1) else 0
                        t_secs = int(m_ts2.group(2)) if m_ts2.group(2) else 0
                        target_pos_ms = (t_mins * 60 + t_secs) * 1000
                        time_display_str = f"{t_mins}:{t_secs:02d}"
                        raw_q = raw_q[:m_ts2.start()] + raw_q[m_ts2.end():]
                    else:
                        # Pattern 3: pure seconds (e.g. from 45 seconds)
                        m_ts3 = re.search(r"\b(?:starting\s+(?:at|from)|from|at)\s+(\d+)\s*(?:seconds?|secs?|s)\b", raw_q, re.IGNORECASE)
                        if m_ts3:
                            t_mins = 0
                            t_secs = int(m_ts3.group(1))
                            target_pos_ms = t_secs * 1000
                            time_display_str = f"{t_mins}:{t_secs:02d}"
                            raw_q = raw_q[:m_ts3.start()] + raw_q[m_ts3.end():]

                # Strip trailing fillers if user was cut off (e.g. 'starting', 'from', 'at')
                raw_q = re.sub(r"\b(?:starting|from|at|start)\s*$", "", raw_q, flags=re.IGNORECASE).strip()

                clean_q = re.sub(r"^(?:the\s+song\s+|a\s+song\s+|some\s+song\s+|some\s+songs\s+|song\s+|songs\s+|music\s+|some\s+music\s+|the\s+track\s+|track\s+|tracks\s+)+", "", raw_q, flags=re.IGNORECASE).strip()
                clean_q = re.sub(r"^(?:called\s+|named\s+)+", "", clean_q, flags=re.IGNORECASE).strip()
                clean_q = re.sub(r"\b(?:song|songs|track|tracks|music)\s+by\b", "by", clean_q, flags=re.IGNORECASE)
                clean_q = re.sub(r"\b(?:song|songs|track|tracks|music)\b", "", clean_q, flags=re.IGNORECASE)
                clean_q = re.sub(r"\s+", " ", clean_q).strip("?.!\"' ")

                clean_q = re.sub(r"\s+", " ", clean_q).strip()

                if clean_q.lower() not in ("", "some", "any", "a", "the", "something", "anything"):
                    q = clean_q

            if q:
                # Primary: Instant direct playback via Spotify Web API / Connect
                if self.spotify and self.spotify.is_authenticated():
                    api_res = self.spotify.search_and_play(q, position_ms=target_pos_ms)
                    if api_res.get("success"):
                        track_id = api_res.get("track_id")
                        if track_id and target_pos_ms is None:
                            # Only force-navigate when there's NO timestamp offset.
                            # Force-navigating reloads the Spotify Web Player, which
                            # resets playback position to 0:00, defeating position_ms.
                            web_track_url = f"https://open.spotify.com/track/{track_id}"
                            browser_tab_manager.open_or_reuse("spotify", web_track_url, force_navigate=True)
                        else:
                            # Timestamp active or no track_id — just focus the existing tab
                            browser_tab_manager.find_and_focus_service("spotify")
                        track_title = api_res.get("track_name", q)
                        artist = api_res.get("artist", "")
                        artist_str = f" by {artist}" if artist else ""
                        ts_str = f" starting at {time_display_str}" if time_display_str else ""
                        action_desc = f"Playing '{track_title}'{artist_str}{ts_str} on Spotify."
                        fallback = f"Playing '{track_title}'{ts_str} on Spotify, {addr}."
                        return self._generate_dynamic_action_response(text, action_desc, fallback)

                # Fallback: Navigate to search page on Spotify Web
                web_url = f"https://open.spotify.com/search/{urllib.parse.quote_plus(q)}"
                reuse_res = browser_tab_manager.open_or_reuse("spotify", web_url, force_navigate=True)
                action_desc = f"Opened Spotify Web searching for '{q}'."
                fallback = f"Playing '{q}' on Spotify Web, {addr}."
                return self._generate_dynamic_action_response(text, action_desc, fallback)
            else:
                # Ambiguous query -> resume current playback
                if self.spotify and self.spotify.is_authenticated():
                    self.spotify.play()
                elif sys.platform.startswith("win"):
                    try:
                        import ctypes
                        VK_MEDIA_PLAY_PAUSE = 0xB3
                        ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 0, 0)
                        time.sleep(0.04)
                        ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 2, 0)
                    except Exception:
                        pass
                browser_tab_manager.open_or_reuse("spotify", "https://open.spotify.com", force_navigate=False)
                action_desc = "Resumed Spotify playback."
                fallback = f"Resumed playback on Spotify, {addr}."
                return self._generate_dynamic_action_response(text, action_desc, fallback)

    def _handle_fast_conversation(self, text: str) -> Optional[str]:
        """Conversational fast path powered by LLM reasoning with Personality Prompt."""
        lower = text.lower().strip()

        # Real hardware and temporal sensor data (LLM cannot physically measure laptop battery or system clock)
        is_timer_keyword = any(k in lower for k in ["timer", "alarm", "countdown", "stopwatch"])
        if not is_timer_keyword and (
            re.search(r"\b(?:what(?:'s|\s+is)\s+(?:the\s+)?time|tell\s+(?:me\s+)?(?:the\s+)?time|current\s+time|time\s+now|\btime\s+please)\b", lower)
            or lower == "time"
            or ("time" in lower and any(q in lower for q in ["what", "tell", "current", "know", "now"]) and "set" not in lower)
        ):
            now = datetime.datetime.now()
            addr = self.personality.get_address_title()
            return f"It is {now.strftime('%I:%M %p').lstrip('0')}, {addr}."

        if any(w in lower for w in ["date", "what day is it", "day of the week", "what's the date", "what is the date", "todays date", "today's date"]):
            now = datetime.datetime.now()
            addr = self.personality.get_address_title()
            return f"Today is {now.strftime('%A, %B %d, %Y')}, {addr}."

        if any(w in lower for w in ["battery status", "battery level", "how much battery"]):
            import psutil
            battery = psutil.sensors_battery()
            addr = self.personality.get_address_title()
            if battery:
                plugged = "plugged in" if battery.power_plugged else "on battery power"
                return f"Battery is at {round(battery.percent)}%, {plugged}, {addr}."
            return f"Battery status is unavailable on this device, {addr}."

        # Real-time live weather data
        if any(w in lower for w in [
            "weather", "temperature", "forecast", "climate", "how hot", "how cold",
            "is it hot", "is it cold", "is it raining", "raining", "humidity", "rain"
        ]):
            from tools.weather import get_weather
            # Extract target city if specified (e.g., 'weather in Paris', 'temperature for London')
            m_city = re.search(r"\b(?:in|for|at)\s+([a-zA-Z\s]+)", text, re.IGNORECASE)
            city_arg = m_city.group(1).strip() if m_city else None
            if city_arg:
                clean_city = re.sub(r"\b(right now|now|today|currently|please|outside|this morning|this evening|at the moment|my location|here)\b.*", "", city_arg, flags=re.IGNORECASE).strip()
                city_arg = clean_city if clean_city else None
            w_res = get_weather(city_arg)
            if w_res.get("success"):
                addr = self.personality.get_address_title()
                return f"{w_res['summary']} {addr.capitalize()}."

        # Real-time live location data
        if any(w in lower for w in [
            "where am i", "my location", "current location", "what city am i in",
            "where are we", "what is my location", "get location", "my current city"
        ]):
            from tools.weather import get_current_location
            loc = get_current_location()
            addr = self.personality.get_address_title()
            if not loc:
                return self.personality.format_response(f"I couldn't determine your current location yet, {addr}. Please tell me your city or set a preferred location.")
            region = loc.get("region") or ""
            country = loc.get("country") or ""
            return self.personality.format_response(f"You are currently in {loc['city']}, {region}, {country}, {addr}.")

        # Neural LLM Thinking Layer — Let the model generate answers dynamically using personality system prompt
        if hasattr(self, "model_provider") and hasattr(self.model_provider, "is_configured") and self.model_provider.is_configured():
            sys_prompt = self.get_contextual_system_prompt(text)
            from models.base import ModelMessage
            hist_msgs = [ModelMessage(role=m["role"], content=m["content"]) for m in getattr(self, "conversation_history", [])[-10:]]
            res = self.model_provider.generate(text, system_prompt=sys_prompt, messages=hist_msgs)
            if res.content:
                return res.content

        return None

    def _build_fast_response(
        self,
        response_text: Any,
        request_id: str,
        start_time: float,
        task_mem: TaskMemory,
        speak_response: bool,
        speech_text: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Standardized packaging for zero-overhead fast-path responses."""
        if isinstance(response_text, dict):
            extra = dict(extra_data or {})
            for k, v in response_text.items():
                if k == "response":
                    continue
                extra[k] = v
            extra_data = extra
            response_text = response_text.get("response", "")

        elapsed = round(time.time() - start_time, 3)
        formatted = self.personality.format_response(str(response_text or ""))
        self._last_assistant_response = formatted

        # Track rolling conversation history for multi-turn dialogue context
        if not hasattr(self, "conversation_history"):
            self.conversation_history = []
        goal_data = task_mem.load_goal() if hasattr(task_mem, "load_goal") else None
        if isinstance(goal_data, dict):
            user_req = goal_data.get("goal") or task_mem.task_id
        elif isinstance(goal_data, str):
            user_req = goal_data
        else:
            user_req = task_mem.task_id

        if user_req:
            self.conversation_history.append({"role": "user", "content": str(user_req)})
            self.conversation_history.append({"role": "assistant", "content": formatted})
            if len(self.conversation_history) > 16:
                self.conversation_history = self.conversation_history[-16:]
            self._log_session_turn(request_id, str(user_req), formatted)

        response_data = {
            "success": True,
            "request_id": request_id,
            "status": "SUCCESS",
            "fast_path": True,
            "user_request": task_mem.task_id,
            "response": formatted,
            "completed_tasks": 1,
            "total_tasks": 1,
            "elapsed_seconds": elapsed,
            "memory_dir": self.memory_dir,
            "data": extra_data or {},
        }

        task_mem.save_result("SUCCESS", formatted)

        if speak_response:
            from core.math_phonetics import MathPhoneticizer
            target_speech = speech_text if speech_text else MathPhoneticizer.sanitize_for_speech(formatted)
            self._speak_output(target_speech, speak_response)

        return response_data

    def _speak_output(self, text: str, speak_option: Any) -> None:
        """Synthesize spoken response using VoiceEngine with full phonetic & symbol sanitization."""
        from core.math_phonetics import MathPhoneticizer
        clean_speech = MathPhoneticizer.sanitize_for_speech(text)
        if len(clean_speech) > 250:
            clean_speech = clean_speech[:250] + "..."
        v_opt = speak_option if isinstance(speak_option, str) and speak_option != "default" else None
        try:
            self.voice_engine.speak(clean_speech, voice=v_opt)
        except Exception:
            pass

    def _log_session_turn(self, session_id: str, user_text: str, assistant_text: str, title: Optional[str] = None):
        """Log conversational turn into SQLite FTS5 for session_search."""
        try:
            from tools.session_search_tool import get_session_search_store
            store = get_session_search_store()
            store.log_message(session_id, "user", str(user_text), title=title or str(user_text)[:60])
            store.log_message(session_id, "assistant", str(assistant_text))
        except Exception as e:
            logger.debug(f"[SessionSearch] Turn logging failed: {e}")

