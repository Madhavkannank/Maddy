import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from voice.engine import VoiceboxProvider, VoiceEngine, EdgeTTSProvider


def test_voicebox_is_available_offline():
    """Verify is_available returns False instantly without hanging when Voicebox is closed."""
    provider = VoiceboxProvider(base_url="http://127.0.0.1:19999")
    assert provider.is_available() is False


def test_voicebox_resolve_profiles():
    """Verify profile resolution for M.A.D.D.Y. and Scarlett Johansson."""
    provider = VoiceboxProvider()
    mock_profiles = [
        {"id": "prof-1", "name": "Paul Bettany (M.A.D.D.Y.)"},
        {"id": "prof-2", "name": "Scarlett Johansson (Samantha)"},
    ]

    with patch.object(provider, "get_profiles", return_value=mock_profiles):
        # Resolve M.A.D.D.Y. and legacy aliases
        assert provider.resolve_profile_id("maddy") == "prof-1"
        assert provider.resolve_profile_id("male") == "prof-1"

        # Resolve Scarlett
        assert provider.resolve_profile_id("scarlett") == "prof-2"
        assert provider.resolve_profile_id("samantha") == "prof-2"
        assert provider.resolve_profile_id("female") == "prof-2"


def test_voicebox_synthesize_mock():
    """Verify synthesize calls Voicebox generate endpoint when active."""
    provider = VoiceboxProvider()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test.wav")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "audio/wav"}
        mock_resp.content = b"RIFFmockwavcontent"

        with patch("httpx.post", return_value=mock_resp):
            with patch.object(provider, "resolve_profile_id", return_value="prof-1"):
                res = provider.synthesize("Hello boss", voice="maddy", output_path=out_path)
                assert res["success"] is True
                assert res["method"] == "voicebox_local_neural"
                assert os.path.exists(out_path)


def test_voice_engine_prioritizes_voicebox_when_available():
    """Verify VoiceEngine automatically selects Voicebox if running and enabled."""
    with patch.dict(os.environ, {"USE_VOICEBOX": "true"}):
        with patch.object(VoiceboxProvider, "is_available", return_value=True):
            engine = VoiceEngine()
            assert isinstance(engine.provider, VoiceboxProvider)
