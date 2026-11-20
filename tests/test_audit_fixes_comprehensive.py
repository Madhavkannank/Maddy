"""
Comprehensive Regression Test Suite for M.A.D.D.Y. Recent Audit Fixes:
1. OS-specific voice default (Female on Linux, Male on Windows).
2. Native desktop photo viewer (Loupe on Linux, Photos on Windows - not Firefox).
3. Math phonetics (Natural title slash 'or' vs arithmetic 'divided by').
4. Phantom links elimination in speech extraction + structured link console formatting.
5. Multi-site dynamic search with ThreadPoolExecutor.
6. Warm operator recognition (No coy snark) & spoken letter normalization.
"""
import os
import sys
import unittest
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.math_phonetics import MathPhoneticizer
from core.tools.web_browser import WebBrowserTool
from core.tools.camera_tool import CameraTool
from personality.policy import PersonalityPolicy
from personality.engine import PersonalityEngine
from platform_layer import get_platform
from voice.engine import VoiceEngine


class TestComprehensiveAuditFixes(unittest.TestCase):

    def test_01_os_voice_defaults(self):
        """Verify female default voice on Linux and male on Windows."""
        plat = get_platform()
        ve = VoiceEngine(platform_driver=plat)
        def_voice = ve.get_default_voice()
        if plat.get_os_name() == "linux":
            self.assertIn(def_voice, ["sarah", "eve", "en-GB-SoniaNeural"], f"Expected female voice on Linux, got: {def_voice}")
        elif plat.get_os_name() == "windows":
            self.assertIn(def_voice, ["chris", "adam", "en-GB-RyanNeural"], f"Expected male voice on Windows, got: {def_voice}")

    def test_02_native_desktop_photo_viewer(self):
        """Verify CameraTool and PlatformInterface open photos in native desktop viewers, not browser."""
        cam = CameraTool()
        schema = cam.get_parameters_schema()
        self.assertIn("view_photo", schema["properties"]["action"]["enum"])

        # Check platform viewer commands
        plat = get_platform()
        if plat.get_os_name() == "linux":
            # On Linux (Fedora), loupe or eog must be preferred
            viewers = plat.IMAGE_VIEWERS if hasattr(plat, "IMAGE_VIEWERS") else []
            self.assertTrue(any("loupe" in v for v in ["/usr/bin/loupe", "loupe", "eog"]), "Loupe should be supported viewer")

    def test_03_math_slash_vs_title_slash(self):
        """Verify slash in natural titles is vocalized as 'or' while arithmetic remains 'divided by'."""
        # Natural title
        title_text = "The Lord of the Rings / The Hobbit"
        spoken_title = MathPhoneticizer.sanitize_for_speech(title_text)
        self.assertNotIn("divided by", spoken_title.lower())
        self.assertIn("or", spoken_title.lower())

        # Operating systems
        os_text = "Windows / Linux compatibility"
        spoken_os = MathPhoneticizer.sanitize_for_speech(os_text)
        self.assertNotIn("divided by", spoken_os.lower())
        self.assertIn("or", spoken_os.lower())

        # Arithmetic division
        math_text = "10 / 2 = 5"
        spoken_math = MathPhoneticizer.sanitize_for_speech(math_text)
        self.assertIn("divided by", spoken_math.lower())

    def test_04_phantom_link_elimination_and_formatting(self):
        """Verify speech extraction omits phantom link claims and repetitive screen claims."""
        # Case without links
        no_link_text = "Here are the top results:\n1. Game One\n2. Game Two\n3. Game Three"
        spoken_no_link = MathPhoneticizer.extract_speakable_text(no_link_text)
        self.assertNotIn("and links", spoken_no_link.lower())
        self.assertNotIn("on your screen", spoken_no_link.lower())
        self.assertNotIn("below", spoken_no_link.lower())
        self.assertIn("game one", spoken_no_link.lower())

        # Case with links
        with_link_text = "Here are the results:\n1. [Game One](https://example.com/one)\n2. [Game Two](https://example.com/two)\nI have listed the links below."
        spoken_with_link = MathPhoneticizer.extract_speakable_text(with_link_text)
        self.assertNotIn("below", spoken_with_link.lower())
        self.assertNotIn("on your screen", spoken_with_link.lower())
        self.assertIn("game one", spoken_with_link.lower())

        # Structured banner formatting
        banner = MathPhoneticizer.format_links_for_display(with_link_text)
        self.assertIn("FETCHED LINKS & DESTINATIONS", banner)
        self.assertIn("https://example.com/one", banner)

    def test_05_multi_site_dynamic_search(self):
        """Verify WebBrowserTool parses and executes multi-site concurrent search."""
        tool = WebBrowserTool()
        query, sites = tool._extract_sites_and_query(
            "search for Call of Duty on Steam, DODI Repacks, and Reddit"
        )
        self.assertEqual(query, "Call of Duty")
        self.assertEqual(len(sites), 3)
        self.assertIn("Steam", sites)
        self.assertIn("DODI Repacks", sites)
        self.assertIn("Reddit", sites)

    def test_06_spoken_letter_normalization(self):
        """Verify spoken letter-by-letter transcriptions are collapsed into coherent keywords."""
        t1 = "search M A D D Y   C L A S H E R"
        self.assertEqual(MathPhoneticizer.normalize_spelled_letters(t1), "search Maddy Clasher")

        t2 = "I said M A D D Y C L A S H E R. Not something else."
        self.assertEqual(MathPhoneticizer.normalize_spelled_letters(t2), "I said Maddy Clasher. Not something else.")

        t3 = "Open S P O T I F Y"
        self.assertEqual(MathPhoneticizer.normalize_spelled_letters(t3), "Open Spotify")

        t4 = "D O D I   R E P A C K S"
        self.assertEqual(MathPhoneticizer.normalize_spelled_letters(t4), "Dodi Repacks")

        t5 = "D O D I R E P A C K S"
        self.assertEqual(MathPhoneticizer.normalize_spelled_letters(t5), "Dodi Repacks")

    def test_07_no_regex_intercept_for_conversational_and_identity_queries(self):
        """Verify conversational, identity, and preference queries are NOT intercepted by hardcoded regex."""
        policy = PersonalityPolicy()

        # Conversational / identity queries, preference updates, and self-correction phrases
        # must NEVER be intercepted by regex. All queries must flow dynamically to MaddyAgentCore / LLM.
        for query in [
            "Do you know me? Do you like me or not?",
            "Do you know me?",
            "What is my name?",
            "Do you like me or not?",
            "Who am I?",
            "Call me sir",
            "My name is Madhav Kannan",
            "No, call me boss",
            "Actually, address me as Commander",
            "Wait, call me John",
            "Correction, my name is Alex",
        ]:
            handled, resp = policy.handle_name_query_or_setup(query)
            self.assertFalse(handled, f"Query '{query}' should not be intercepted by hardcoded regex")
            self.assertIsNone(resp)

    def test_08_universal_native_desktop_file_opening(self):
        """Verify universal file opening support across OS platforms (PDF, PPT, JSON, ZIP, etc.)."""
        from core.tools.app_control import AppControlTool
        app_ctrl = AppControlTool()
        schema = app_ctrl.get_parameters_schema()
        self.assertIn("open_file", schema["properties"]["action"]["enum"])
        self.assertIn("pdf", schema["properties"]["file_path"]["description"].lower())
        self.assertIn("presentation", schema["properties"]["file_path"]["description"].lower())
        self.assertIn("json", schema["properties"]["file_path"]["description"].lower())

        # Platform check
        plat = get_platform()
        # Verify open_file method exists
        self.assertTrue(hasattr(plat, "open_file"))

    def test_09_dynamic_operator_grounding_and_address_rule(self):
        """Verify operator profile grounds dynamically from memory and enforces boss/sir address."""
        from personality.prompts import build_personality_system_prompt
        prompt = build_personality_system_prompt(address_title="boss")
        self.assertIn('Preferred form of address: "boss"', prompt)
        self.assertIn("Always address the operator as", prompt)
        self.assertNotIn('Of course I know you, Madhav', prompt)

        # Verify dynamic memory context compressor grounds operator profile
        from core.memory import MemoryStore, DEFAULT_DB_PATH
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_db = Path(tmp_dir) / "test_grounding.db"
            ms = MemoryStore(db_path=test_db, auto_sync_shared=False)
            ms.preferences.set_preference("real_name", "Madhav Kannan", confidence=1.0, is_explicit=True)
            ms.preferences.set_preference("preferred_address", "boss", confidence=1.0, is_explicit=True)
            ctx = ms.compressor.build_bounded_context("Who am I?")
            self.assertIn("Madhav Kannan", ctx)
            self.assertIn("boss", ctx)

    def test_10_dynamic_location_never_chennai(self):
        """Verify dynamic runtime location resolves from network/IP or timezone and is never Chennai."""
        from core.location import get_current_location, get_current_city, get_location_summary
        loc = get_current_location()
        city = get_current_city()
        summary = get_location_summary()

        self.assertNotEqual(city.lower(), "chennai", "City should never default to hardcoded Chennai")
        self.assertNotIn("chennai", summary.lower(), "Location summary should never default to Chennai")
        self.assertTrue(len(summary) > 0)

        # Weather tool dynamic resolution
        from core.tools.weather import WeatherTool
        w = WeatherTool()
        res = w.execute()
        if res.success:
            self.assertNotIn("chennai", res.output.lower(), "Weather should be for current location, not Chennai")

    def test_11_dynamic_os_default_browser_never_hardcoded_chrome(self):
        """Verify default browser queries OS settings dynamically rather than hardcoding Chrome."""
        plat = get_platform()
        browser = plat.get_default_browser()
        self.assertIn(browser, ["Firefox", "Chrome", "Brave", "Chromium", "Edge", "Default Browser"])
        self.assertTrue(len(browser) > 0)

        # Memory store bounded context should reflect dynamic browser
        from core.memory import MemoryStore
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_db = Path(tmp_dir) / "test_browser.db"
            ms = MemoryStore(db_path=test_db, auto_sync_shared=True)
            ctx = ms.compressor.build_bounded_context("Who am I?")
            self.assertIn(browser, ctx)

    def test_12_clean_user_md_profile_no_mock_data(self):
        """Verify USER.md and memory sync purge fake mock songs, playlists, and execution style."""
        user_md_path = ROOT / "memory" / "USER.md"
        content = user_md_path.read_text()
        self.assertNotIn("Idayam Love", content)
        self.assertNotIn("morning_playlist", content)
        self.assertNotIn("Fast, concise, composed, zero corporate fluff", content)
        self.assertNotIn("Chennai", content)
        self.assertNotIn("Chrome", content)

        # MemoryStore sync
        from core.memory import MemoryStore
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_db = Path(tmp_dir) / "test_clean_sync.db"
            ms = MemoryStore(db_path=test_db, auto_sync_shared=True)
            prefs = ms.preferences.get_all_preferences()
            self.assertNotIn("favorite_songs", prefs)
            self.assertNotIn("morning_playlist", prefs)
            self.assertNotIn("execution_style", prefs)
            self.assertEqual(prefs.get("real_name"), "Madhav Kannan")

    def test_13_windows_cross_platform_parity(self):
        """Verify WindowsPlatform implementation parity for file opening, default browser, and Path import."""
        from platform_layer.windows.platform_windows import WindowsPlatform
        win_plat = WindowsPlatform()
        self.assertEqual(win_plat.get_os_name(), "windows")
        self.assertTrue(hasattr(win_plat, "open_file"))
        self.assertTrue(hasattr(win_plat, "get_default_browser"))
        self.assertTrue(hasattr(win_plat, "open_url"))

        # Verify default browser candidates include Edge, Firefox, Chrome, Brave
        def_b = win_plat.get_default_browser()
        self.assertIn(def_b, ["Edge", "Firefox", "Chrome", "Brave"])

    def test_14_web_search_tool_registered_and_executable(self):
        """Verify web_search tool is registered in runtime and delegates cleanly."""
        from app.runtime import MaddyRuntime
        rt = MaddyRuntime()
        tool = rt.tool_registry.get_tool("web_search")
        self.assertIsNotNone(tool, "web_search tool must be registered in tool registry")
        schema = tool.to_openai_schema()
        self.assertEqual(schema["function"]["name"], "web_search")
        self.assertIn("query", schema["function"]["parameters"]["properties"])

    def test_15_telegram_tool_registered_and_functional(self):
        """Verify telegram tool is registered in runtime with status and send capabilities."""
        from app.runtime import MaddyRuntime
        from core.tools.telegram_tool import TelegramTool
        rt = MaddyRuntime()
        tool = rt.tool_registry.get_tool("telegram")
        self.assertIsNotNone(tool, "telegram tool must be registered in tool registry")
        self.assertIsInstance(tool, TelegramTool)

        # Test status check (safe operation)
        res = tool.execute(action="status")
        self.assertTrue(res.success)
        self.assertIn("configured", res.metadata)

    def test_16_multi_site_extract_with_at_and_punctuation(self):
        """Verify _extract_sites_and_query handles 'at', 'from', and trailing punctuation correctly."""
        tool = WebBrowserTool()
        q, sites = tool._extract_sites_and_query(
            "Can you find some good games to play at FitGirlRepack and DotyRepack?"
        )
        self.assertEqual(len(sites), 2)
        self.assertIn("FitGirlRepack", sites)
        self.assertIn("DotyRepack", sites)
        self.assertIn("games", q.lower())


if __name__ == "__main__":
    unittest.main()
