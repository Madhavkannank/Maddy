import unittest
from unittest.mock import MagicMock, patch
import re

class TestSpotifyMediaCommand(unittest.TestCase):
    def test_comma_and_punctuation_handling(self):
        test_inputs = [
            "Play, Pottakaatil Poovasam, Song, On Spotify.",
            "Play perfect song by H. Sheeran on Spotify.",
            "Play, Pottakaatil Poovasam, Song, On Spotify",
            "Hey MADDY play Pottakaatil Poovasam on Spotify",
            "play music on spotify",
            "resume spotify",
            "play some songs on spotify",
        ]
        
        expected = [
            ("track", "pottakaatil poovasam"),
            ("track", "perfect by h. sheeran"),
            ("track", "pottakaatil poovasam"),
            ("track", "pottakaatil poovasam"),
            ("resume", ""),
            ("resume", ""),
            ("resume", ""),
        ]

        for text, (exp_type, exp_query) in zip(test_inputs, expected):
            lower = text.lower().strip()
            normalized = re.sub(r"[,;]+", " ", lower)
            normalized = re.sub(r"\s+", " ", normalized).strip()

            cleaned = re.sub(r"\b(?:on|in|from|at|via|through|with|or|using)?\s*spotify\b", "", normalized, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"^(?:so\s+)?(?:i'm\s+|i\s+am\s+)?(?:in\s+a\s+mood|on\s+a\s+mood(?:\s+out)?|feeling\s+like|bored)[\s,]+", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"\b(?:please|can\s+you\s+(?:please\s+)?|could\s+you\s+(?:please\s+)?|i\s+want\s+to\s+listen\s+to)\b", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"\s+", " ", cleaned).strip("?.! ")

            is_generic = bool(re.match(r"^play\s*(?:me\s+)?(?:the\s+song|a\s+song|some\s+song|some\s+songs|songs|a\s+track|tracks|some\s+tracks|music|some\s+music|something|anything)?$", cleaned) or cleaned in ("resume", "unpause", "play"))

            if is_generic:
                act_type = "resume"
                q = ""
            else:
                m_song = re.search(r"\bplay\s+(?:me\s+)?(.+)", cleaned)
                q = ""
                if m_song:
                    raw_q = m_song.group(1).strip()
                    clean_q = re.sub(r"^(?:the\s+song\s+|a\s+song\s+|some\s+song\s+|some\s+songs\s+|song\s+|songs\s+|music\s+|some\s+music\s+|the\s+track\s+|track\s+|tracks\s+)+", "", raw_q, flags=re.IGNORECASE).strip()
                    clean_q = re.sub(r"^(?:called\s+|named\s+)+", "", clean_q, flags=re.IGNORECASE).strip()
                    clean_q = re.sub(r"\b(?:song|songs|track|tracks|music)\s+by\b", "by", clean_q, flags=re.IGNORECASE)
                    clean_q = re.sub(r"\b(?:song|songs|track|tracks|music)\b", "", clean_q, flags=re.IGNORECASE)
                    clean_q = re.sub(r"\s+", " ", clean_q).strip("?.!\"' ")
                    if clean_q.lower() not in ("", "some", "any", "a", "the", "something", "anything"):
                        q = clean_q
                act_type = "track" if q else "resume"

            self.assertEqual(act_type, exp_type, f"Failed type for {text}")
            self.assertEqual(q, exp_query, f"Failed query for {text}")

    def test_spotify_timestamp_playback(self):
        from app.maddy_assistant import MaddyAssistant
        asst = MaddyAssistant()
        
        # Test extraction of song title + timestamp ms
        test_cases = [
            ("Play Pottakaatil Poovasam starting at 1:30 on Spotify", "pottakaatil poovasam", 90000),
            ("Play Perfect by Ed Sheeran from 2 minutes on Spotify", "perfect by ed sheeran", 120000),
            ("Play Believer at 45 seconds on Spotify", "believer", 45000),
            ("Play Shape of You at 2:05", "shape of you", 125000),
        ]

        with patch.object(asst.spotify, "is_authenticated", return_value=True), \
             patch.object(asst.spotify, "search_and_play", return_value={"success": True, "track_name": "Test Track", "track_id": "123", "artist": "Artist"}) as mock_play:
            with patch("browser.browser_tab_manager.open_or_reuse", return_value={"reused": True}):
                for prompt, exp_q, exp_ms in test_cases:
                    mock_play.reset_mock()
                    resp = asst._handle_direct_spotify_command(prompt)
                    self.assertIsNotNone(resp)
                    mock_play.assert_called_once()
                    call_args = mock_play.call_args
                    self.assertEqual(call_args[0][0], exp_q)
                    self.assertEqual(call_args[1].get("position_ms"), exp_ms)

    def test_spotify_seek_and_relative(self):
        from app.maddy_assistant import MaddyAssistant
        asst = MaddyAssistant()

        # Test absolute seek
        with patch.object(asst.spotify, "is_authenticated", return_value=True), \
             patch.object(asst.spotify, "seek", return_value={"success": True, "action": "seek", "position_ms": 90000, "time_str": "1:30"}) as mock_seek:
            with patch("browser.browser_tab_manager.find_and_focus_service", return_value=True):
                resp = asst._handle_direct_spotify_command("Seek to 1:30 on Spotify")
                self.assertIsNotNone(resp)
                mock_seek.assert_called_once_with(90000)

        # Test relative forward
        with patch.object(asst.spotify, "is_authenticated", return_value=True), \
             patch.object(asst.spotify, "seek_relative", return_value={"success": True, "action": "seek", "position_ms": 60000, "time_str": "1:00"}) as mock_rel:
            with patch("browser.browser_tab_manager.find_and_focus_service", return_value=True):
                resp = asst._handle_direct_spotify_command("Fast forward 30 seconds on Spotify")
                self.assertIsNotNone(resp)
                mock_rel.assert_called_once_with(30000)

        # Test relative rewind
        with patch.object(asst.spotify, "is_authenticated", return_value=True), \
             patch.object(asst.spotify, "seek_relative", return_value={"success": True, "action": "seek", "position_ms": 10000, "time_str": "0:10"}) as mock_rel:
            with patch("browser.browser_tab_manager.find_and_focus_service", return_value=True):
                resp = asst._handle_direct_spotify_command("Rewind 15 seconds on Spotify")
                self.assertIsNotNone(resp)
                mock_rel.assert_called_once_with(-15000)

if __name__ == "__main__":
    unittest.main()
