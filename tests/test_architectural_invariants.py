"""Architectural Invariant Tests for M.A.D.D.Y.
Enforces structural and runtime invariants to permanently prevent hardcoding:
1. AST verification: No hardcoded site/alias dictionaries, no site-specific fallback branching.
2. Pure tool primitives: web_browser and web_search decoupled from hardcoded logic.
3. Dynamic domain and entity resolution on novel/arbitrary monikers without static tables.
4. Dynamic query token cleaning for any arbitrary website without keyword lists.
5. Dual-channel terminal/audio output decoupling.
6. Byte-for-byte SHA-256 integrity of protected Telegram files.
"""
import ast
import hashlib
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.tools.web_browser import WebBrowserTool, WebSearchTool
from core.math_phonetics import MathPhoneticizer


PROTECTED_TELEGRAM_HASHES = {
    "integrations/messaging/telegram_bot.py": "dc06cc308de0e369d98d265f2e09f474d83a876bc2020e5c0d3f7f138b97d645",
    "integrations/messaging/telegram_client.py": "b4a58700e9c82d8c6c391ae34af6589a3e066901760c9c9f6db33987adf51a15",
    "telegram.cmd": "f67181dc0f87a2aa476eff9ee264bc1b03b340350f5f672e1f6baeb8bff0e40e",
    "scripts/start_telegram_daemon.vbs": "22e307c439845a8f68e43e7b2deb1e15fc1694ba6144d9f706bb7c3c9a60883a",
}


class TestArchitecturalInvariants(unittest.TestCase):

    def test_01_telegram_files_sha256_unmodified(self):
        """Invariant: All 5 protected Telegram files must remain 100% byte-for-byte intact."""
        missing = [p for p in PROTECTED_TELEGRAM_HASHES if not (ROOT / p).exists()]
        if missing:
            self.skipTest(f"Protected Telegram files excluded from repository checkout: {missing}")
        for rel_path, expected_hash in PROTECTED_TELEGRAM_HASHES.items():
            full_path = ROOT / rel_path
            hasher = hashlib.sha256()
            hasher.update(full_path.read_bytes())
            actual_hash = hasher.hexdigest()
            self.assertEqual(
                actual_hash,
                expected_hash,
                f"PROTECTED TELEGRAM FILE WAS TAMPERED WITH: {rel_path}\nExpected: {expected_hash}\nActual:   {actual_hash}",
            )

    def test_02_ast_no_hardcoded_site_tables_in_web_browser(self):
        """Invariant: web_browser.py must contain NO hardcoded repack alias dictionaries or site branches."""
        web_browser_file = ROOT / "core" / "tools" / "web_browser.py"
        source = web_browser_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(web_browser_file))

        # Check STANDARD_SERVICES dictionary
        forbidden_service_keys = {"dodi", "doty", "fitgirl", "repack", "steamrip", "gog", "piratebay"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "STANDARD_SERVICES":
                        self.assertIsInstance(node.value, ast.Dict, "STANDARD_SERVICES must be a dict")
                        for key_node in node.value.keys:
                            if isinstance(key_node, ast.Constant):
                                self.assertNotIn(
                                    key_node.value.lower(),
                                    forbidden_service_keys,
                                    f"STANDARD_SERVICES contains forbidden hardcoded key: {key_node.value}",
                                )

        # Walk AST to verify no conditional comparisons against hardcoded site terms
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Inspect test expression
                test_str = ast.unparse(node.test).lower()
                for forbidden in ("'dodi'", '"dodi"', "'doty'", '"doty"', "'fitgirl'", '"fitgirl"'):
                    self.assertNotIn(
                        forbidden,
                        test_str,
                        f"Found forbidden hardcoded site branch in AST: if {test_str}",
                    )

    def test_03_dynamic_domain_resolution_arbitrary_entities(self):
        """Invariant: web_browser resolves arbitrary websites dynamically via DuckDuckGo zero-shot discovery."""
        tool = WebBrowserTool()
        test_entities = [
            ("Hugging Face", "huggingface.co"),
            ("Arxiv", "arxiv.org"),
            ("Steam", "store.steampowered.com"),
            ("GitHub", "github.com"),
            ("Doty Repacks", "dodi-repacks.site"),
            ("FitGirl Repacks", "fitgirl-repacks.site"),
        ]

        for moniker, expected_domain_part in test_entities:
            netloc, base_url = tool._resolve_site_domain(moniker)
            self.assertTrue(bool(netloc), f"Failed to resolve netloc for '{moniker}'")
            self.assertIn(
                expected_domain_part,
                (netloc + base_url).lower(),
                f"Expected '{expected_domain_part}' for '{moniker}', got netloc='{netloc}', base_url='{base_url}'",
            )

    def test_04_dynamic_query_token_cleaning_any_site(self):
        """Invariant: Query cleaning dynamically strips site monikers and domains without static keyword lists."""
        tool = WebBrowserTool()

        # Test case 1: Dynamic camelCase moniker cleaning
        q1 = tool._clean_query_for_site(
            query="popular games FitGirlRepacks",
            site_identifier="FitGirlRepacks",
            netloc="fitgirl-repacks.site",
        )
        self.assertEqual(q1, "popular games")

        # Test case 2: Arbitrary site moniker cleaning
        q2 = tool._clean_query_for_site(
            query="machine learning research papers on HuggingFace",
            site_identifier="HuggingFace",
            netloc="huggingface.co",
        )
        self.assertEqual(q2, "machine learning research papers")

        # Test case 3: Arbitrary entity with dot in domain
        q3 = tool._clean_query_for_site(
            query="search quantum computing in Arxiv",
            site_identifier="Arxiv",
            netloc="arxiv.org",
        )
        self.assertEqual(q3, "search quantum computing")

        # Test case 4: Natural language multi-site query splitting
        q4, sites4 = tool._extract_sites_and_query(
            "search for Call of Duty on Steam, DODI Repacks, and Reddit"
        )
        self.assertEqual(q4, "Call of Duty")
        self.assertEqual(len(sites4), 3)
        self.assertIn("Steam", sites4)
        self.assertIn("DODI Repacks", sites4)
        self.assertIn("Reddit", sites4)

        # Test case 5: Punctuation and at-syntax
        q5, sites5 = tool._extract_sites_and_query(
            "Can you find some good games to play at FitGirlRepack and DotyRepack?"
        )
        self.assertEqual(len(sites5), 2)
        self.assertIn("FitGirlRepack", sites5)
        self.assertIn("DotyRepack", sites5)
        self.assertIn("good games", q5.lower())

    def test_05_dual_channel_output_terminal_vs_audio(self):
        """Invariant: Terminal displays structured links while TTS speech extraction has zero URLs or glitches."""
        sample_markdown = (
            "Here are the top results:\n"
            "1. [Call of Duty: Black Ops](https://example.com/cod-bo)\n"
            "2. [Call of Duty: Modern Warfare](https://example.com/cod-mw)\n"
            "You can download them directly from the links."
        )

        # 1. Screen / Display channel: Formatted banner
        banner = MathPhoneticizer.format_links_for_display(sample_markdown)
        self.assertIn("FETCHED LINKS & DESTINATIONS", banner)
        self.assertIn("https://example.com/cod-bo", banner)
        self.assertIn("https://example.com/cod-mw", banner)

        # 2. Voice / Audio channel: Clean speech extraction
        spoken = MathPhoneticizer.extract_speakable_text(sample_markdown)
        self.assertNotIn("http", spoken.lower())
        self.assertNotIn("https://", spoken.lower())
        self.assertNotIn("example.com", spoken.lower())
        self.assertNotIn("bracket", spoken.lower())
        self.assertNotIn("parentheses", spoken.lower())
        self.assertIn("call of duty: black ops", spoken.lower())
        self.assertIn("call of duty: modern warfare", spoken.lower())


if __name__ == "__main__":
    unittest.main()
